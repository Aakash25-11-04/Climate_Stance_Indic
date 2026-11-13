"""Explainability (Phase 9): Integrated Gradients (Captum) and LIME for the transformer,
plus a small HTML renderer for token highlights used by app.py and the saved visualisations.

integrated_gradients(model, text, language, target=None, n_steps=32)
    -> {"tokens": [...], "scores": [...], "target": label, "probs": [...]}
lime_explanation(model, text, language, num_samples=300)
    -> {"tokens": [...], "scores": [...], "target": label}
Both attribute only the comment tokens (the claim segment of a pair input is masked out of
the display but kept as context).
"""
from __future__ import annotations

import html

import numpy as np
import torch

from stance.utils import LABEL2ID, LABELS


def _embedding_layer(model):
    m = model.model
    for attr in ("bert", "roberta", "deberta", "electra", "xlm_roberta", "base_model"):
        base = getattr(m, attr, None)
        if base is not None and hasattr(base, "embeddings"):
            return base.embeddings.word_embeddings
    return m.get_input_embeddings()


@torch.no_grad()
def _probs(model, enc):
    model.model.eval()
    out = model.model(**{k: v.to(model.device) for k, v in enc.items()}).logits.float()
    return torch.softmax(out, dim=-1)[0].cpu().numpy()


def integrated_gradients(model, text: str, language: str | None = None, target: str | None = None,
                         n_steps: int = 32) -> dict:
    from captum.attr import LayerIntegratedGradients

    enc = model._encode([text], [language])
    probs = _probs(model, enc)
    tgt = LABEL2ID[target] if target else int(probs.argmax())
    emb = _embedding_layer(model)
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    tt = enc.get("token_type_ids")
    tt = tt.to(model.device) if tt is not None else None
    pad_id = model.tok.pad_token_id or 0
    baseline = torch.full_like(ids, pad_id)
    # keep special tokens in the baseline
    for sid in (model.tok.cls_token_id, model.tok.sep_token_id):
        if sid is not None:
            baseline[ids == sid] = sid

    def fwd(input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        return model.model(**kw).logits[:, tgt]

    model.model.eval()
    model.model.zero_grad()
    lig = LayerIntegratedGradients(fwd, emb)
    was_fp16 = model.fp16
    model.fp16 = False
    with torch.enable_grad():
        attrs = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask, tt) if tt is not None else (mask,),
                              n_steps=n_steps, internal_batch_size=8)
    model.fp16 = was_fp16
    scores = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    scores = scores / (np.abs(scores).max() + 1e-9)
    tokens = model.tok.convert_ids_to_tokens(ids[0].cpu().tolist())
    # drop the claim segment (pair input): everything up to and including the first [SEP]
    keep = np.ones(len(tokens), dtype=bool)
    if model.formulation == "pair" and tt is not None:
        keep = tt[0].cpu().numpy() == 1
        keep[tokens.index(model.tok.sep_token)] = False if model.tok.sep_token in tokens else keep[0]
    tokens_k = [t for t, k in zip(tokens, keep) if k and t not in (model.tok.pad_token,)]
    scores_k = [float(s) for s, t, k in zip(scores, tokens, keep) if k and t not in (model.tok.pad_token,)]
    return {"tokens": tokens_k, "scores": scores_k, "target": LABELS[tgt], "probs": probs.tolist(), "method": "integrated_gradients"}


def lime_explanation(model, text: str, language: str | None = None, num_samples: int = 300, num_features: int = 12) -> dict:
    from lime.lime_text import LimeTextExplainer

    def predict_fn(texts):
        return model.predict_proba(list(texts), [language] * len(texts))

    probs = predict_fn([text])[0]
    tgt = int(probs.argmax())
    explainer = LimeTextExplainer(class_names=LABELS, split_expression=r"\s+", bow=False, random_state=0)
    exp = explainer.explain_instance(text, predict_fn, labels=[tgt], num_features=num_features, num_samples=num_samples)
    pairs = exp.as_list(label=tgt)
    tokens, scores = zip(*pairs) if pairs else ([], [])
    mx = max((abs(s) for s in scores), default=1.0) or 1.0
    return {"tokens": list(tokens), "scores": [float(s) / mx for s in scores], "target": LABELS[tgt], "probs": probs.tolist(), "method": "lime"}


def top_tokens(expl: dict, k: int = 8) -> list[tuple[str, float]]:
    pairs = sorted(zip(expl["tokens"], expl["scores"]), key=lambda p: -abs(p[1]))
    return [(t, round(s, 3)) for t, s in pairs[:k]]


def render_html(expl: dict, title: str | None = None) -> str:
    """Inline-styled HTML: blue = pushes toward the predicted label, orange = against it."""
    spans = []
    for tok, s in zip(expl["tokens"], expl["scores"]):
        tok_disp = tok.replace("##", "").replace("▁", "")
        if not tok_disp.strip():
            continue
        alpha = min(abs(s), 1.0) * 0.75
        colour = f"rgba(42,120,214,{alpha:.2f})" if s > 0 else f"rgba(235,104,52,{alpha:.2f})"
        joiner = "" if tok.startswith("##") else " "
        spans.append(f'{joiner}<span style="background:{colour};border-radius:3px;padding:1px 2px" title="{s:+.2f}">{html.escape(tok_disp)}</span>')
    head = f"<div style='font-size:13px;color:#52514e;margin-bottom:4px'>{html.escape(title or '')} predicted <b>{expl['target']}</b> ({expl.get('method', '')})</div>"
    return head + "<div style='line-height:1.9;font-size:16px'>" + "".join(spans) + "</div>"
