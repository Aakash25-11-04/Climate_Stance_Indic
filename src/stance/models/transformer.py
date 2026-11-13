"""Transformer fine-tuning for stance (Phase 4).

Formulations
  (i)  single : [CLS] text [SEP]
  (ii) pair   : [CLS] claim [SEP] text [SEP]   (NLI-style, target-aware). The claim is given
                in English / Hindi / Bengali according to `claim_lang` (= "auto" picks the
                comment's language, which is the key experiment).

Loss: weighted cross-entropy (inverse-frequency class weights) or focal loss.
Training: AdamW, linear warmup, fp16 autocast on CUDA, gradient accumulation, early
stopping on dev Macro F1, max_length 128, num_workers=0 (low RAM machine).

The class also provides predict_proba(texts, languages) so the model plugs into the same
evaluation / ensembling code as every baseline, and save()/load() for predict.py.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from stance.evaluation.metrics import evaluate
from stance.preprocessing.normalize import Preprocessor
from stance.utils import LABEL2ID, LABELS, abs_path, resolve_device, set_seed

MODEL_ALIASES = {
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "banglabert": "csebuetnlp/banglabert",
    "mdeberta-xnli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
}


class _DS(Dataset):
    def __init__(self, enc, labels=None):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def focal_loss(logits, target, weight=None, gamma: float = 2.0):
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    logp_t = logp.gather(1, target[:, None]).squeeze(1)
    p_t = p.gather(1, target[:, None]).squeeze(1)
    loss = -((1 - p_t) ** gamma) * logp_t
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


class TransformerStanceModel:
    def __init__(self, model_name: str = "muril", formulation: str = "pair", claim_lang: str = "auto",
                 claims: dict | None = None, max_length: int = 128, batch_size: int = 16, grad_accum: int = 1,
                 lr: float = 2e-5, epochs: int = 3, warmup_ratio: float = 0.1, weight_decay: float = 0.01,
                 loss: str = "weighted_ce", focal_gamma: float = 2.0, class_weight_power: float = 0.5,
                 patience: int = 2, fp16: bool = True, seed: int = 42, device: str = "auto",
                 preprocessing: dict | None = None, eval_every_epoch: bool = True, max_train_minutes: float = 170.0,
                 freeze_embeddings: bool = True):
        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.alias = model_name
        self.formulation, self.claim_lang = formulation, claim_lang
        self.claims = claims or {}
        self.max_length, self.batch_size, self.grad_accum = max_length, batch_size, grad_accum
        self.lr, self.epochs, self.warmup_ratio, self.weight_decay = lr, epochs, warmup_ratio, weight_decay
        self.loss_kind, self.focal_gamma, self.class_weight_power = loss, focal_gamma, class_weight_power
        self.patience, self.seed = patience, seed
        self.device = resolve_device(device)
        self.fp16 = fp16 and self.device == "cuda"
        self.pre = Preprocessor.from_config(preprocessing)
        self.eval_every_epoch = eval_every_epoch
        self.max_train_minutes = max_train_minutes
        self.freeze_embeddings = freeze_embeddings
        self.name = f"{model_name}_{formulation}" + (f"_claim-{claim_lang}" if formulation == "pair" else "")
        self.tok = self.model = None
        self.history: list[dict] = []

    # ------------------------------------------------------------------ encoding
    def _claim_for(self, lang: str | None) -> str:
        if self.claim_lang == "auto":
            return self.claims.get(lang or "en", self.claims.get("en", ""))
        return self.claims.get(self.claim_lang, self.claims.get("en", ""))

    def _encode(self, texts, languages=None):
        langs = languages or [None] * len(texts)
        texts = [self.pre(t, l) or "." for t, l in zip(texts, langs)]
        if self.formulation == "pair":
            firsts = [self._claim_for(l) for l in langs]
            enc = self.tok(firsts, texts, truncation="only_second", max_length=self.max_length, padding=True,
                           return_tensors="pt")
        else:
            enc = self.tok(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}

    def _load_backbone(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(LABELS), id2label={i: l for i, l in enumerate(LABELS)},
            label2id=LABEL2ID, ignore_mismatched_sizes=True)
        if self.freeze_embeddings:
            # 4 GB VRAM: the word-embedding matrix (MuRIL 197k x 768, XLM-R 250k x 768) is the largest
            # parameter block; freezing it removes its gradients and AdamW states (~2 GB) so the whole
            # run stays in GPU memory instead of spilling into shared memory.
            emb = self.model.get_input_embeddings()
            emb.weight.requires_grad_(False)
        self.model.to(self.device)

    # ------------------------------------------------------------------ training
    def fit(self, texts, labels, languages=None, dev=None, log=None):
        log = log or (lambda *a, **k: print(*a, **{**k, "flush": True}))
        """dev: optional (texts, labels, languages) used for early stopping on macro F1."""
        set_seed(self.seed)
        if self.model is None:
            self._load_backbone()
        y = np.array([LABEL2ID[l] for l in labels])
        counts = np.bincount(y, minlength=len(LABELS)).astype(float)
        w = (counts.sum() / np.maximum(counts, 1)) ** self.class_weight_power
        w = torch.tensor(w / w.mean(), dtype=torch.float32, device=self.device)
        enc = self._encode(texts, languages)
        dl = DataLoader(_DS(enc, y), batch_size=self.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(self.seed))
        steps_per_epoch = math.ceil(len(dl) / self.grad_accum)
        total = steps_per_epoch * self.epochs
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr, weight_decay=self.weight_decay)
        from transformers import get_linear_schedule_with_warmup

        sched = get_linear_schedule_with_warmup(opt, int(self.warmup_ratio * total), total)
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
        best_f1, best_state, bad, t0 = -1.0, None, 0, time.time()
        self.history = []
        for ep in range(self.epochs):
            self.model.train()
            run_loss, n = 0.0, 0
            for i, batch in enumerate(dl):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                tgt = batch.pop("labels")
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                    logits = self.model(**batch).logits
                logits = logits.float()
                if self.loss_kind == "focal":
                    loss = focal_loss(logits, tgt, w, self.focal_gamma)
                else:
                    loss = F.cross_entropy(logits, tgt, weight=w)
                scaler.scale(loss / self.grad_accum).backward()
                if (i + 1) % self.grad_accum == 0 or (i + 1) == len(dl):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                run_loss += loss.item() * tgt.shape[0]
                n += tgt.shape[0]
                if (i + 1) % 200 == 0:
                    log(f"[train] epoch {ep + 1} step {i + 1}/{len(dl)} loss {run_loss / n:.4f} {(time.time() - t0) / 60:.1f} min")
                if (time.time() - t0) / 60 > self.max_train_minutes:
                    log(f"[train] time budget reached ({self.max_train_minutes} min); stopping")
                    break
            rec = {"epoch": ep + 1, "train_loss": run_loss / max(n, 1), "minutes": (time.time() - t0) / 60}
            if dev is not None:
                dtexts, dlabels, dlangs = dev
                m = evaluate(dlabels, self.predict(dtexts, dlangs), dlangs)
                rec["dev_macro_f1"] = m["macro_f1"]
                rec["dev_accuracy"] = m["accuracy"]
                if m["macro_f1"] > best_f1:
                    best_f1, bad = m["macro_f1"], 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
            self.history.append(rec)
            log(f"[train] {self.name} epoch {ep + 1}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if dev is not None and bad >= self.patience:
                log("[train] early stopping")
                break
            if (time.time() - t0) / 60 > self.max_train_minutes:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_dev_f1 = best_f1
        self.train_seconds = time.time() - t0
        return self

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_proba(self, texts, languages=None, batch_size: int | None = None) -> np.ndarray:
        self.model.eval()
        bs = batch_size or self.batch_size * 2
        out = []
        langs = languages or [None] * len(texts)
        # sort by length for speed, then restore order
        order = np.argsort([len(str(t)) for t in texts])
        for b in range(0, len(texts), bs):
            idx = order[b : b + bs]
            enc = self._encode([texts[i] for i in idx], [langs[i] for i in idx])
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
                logits = self.model(**enc).logits.float()
            out.append((idx, torch.softmax(logits, dim=-1).cpu().numpy()))
        probs = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        for idx, p in out:
            probs[idx] = p
        return probs

    def predict(self, texts, languages=None):
        return [LABELS[i] for i in self.predict_proba(texts, languages).argmax(axis=1)]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path):
        path = abs_path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        meta = {k: getattr(self, k) for k in ("model_name", "alias", "formulation", "claim_lang", "claims", "max_length",
                                               "batch_size", "name", "history", "seed", "loss_kind")}
        meta["preprocessing"] = {k: v for k, v in self.pre.__dict__.items() if not k.startswith("_")}
        meta["best_dev_f1"] = getattr(self, "best_dev_f1", None)
        meta["train_seconds"] = getattr(self, "train_seconds", None)
        with open(path / "stance_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = abs_path(path)
        with open(path / "stance_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(model_name=meta["model_name"], formulation=meta["formulation"], claim_lang=meta["claim_lang"],
                  claims=meta["claims"], max_length=meta["max_length"], batch_size=meta["batch_size"],
                  device=device, preprocessing=meta.get("preprocessing"), seed=meta.get("seed", 42))
        obj.alias, obj.name, obj.history = meta["alias"], meta["name"], meta.get("history", [])
        obj.tok = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path).to(obj.device).eval()
        return obj
