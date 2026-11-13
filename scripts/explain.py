#!/usr/bin/env python
"""Phase 9a: explainability artefacts.

    python scripts/explain.py --model models/muril_pair [--sparse models/tfidf_lr.pkl] [--n 10]

Writes results/explanations/{ig,lime}_<lang>_{correct,wrong}_<k>.html, one combined
results/explanations/index.html, results/explanations/tfidf_top_features.json and a PNG
summary of the TF-IDF coefficients.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stance.explainability import integrated_gradients, lime_explanation, render_html, top_tokens  # noqa: E402
from stance.models.transformer import TransformerStanceModel  # noqa: E402
from stance.utils import LABELS, abs_path, read_data, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", help="transformer dir, or auto (see stance.evaluation.select)")
    ap.add_argument("--sparse", default="auto", help="pickle path, or auto = baseline to beat")
    ap.add_argument("--gold", default="data/gold/test_handcrafted.csv")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    out = abs_path("results/explanations"); out.mkdir(parents=True, exist_ok=True)
    gold = read_data(args.gold)
    from stance.evaluation.select import baseline_to_beat, best_transformer

    if args.model == "auto":
        run, args.model = best_transformer()
        print(f"[explain] transformer: {run} -> {args.model}")
    if args.sparse == "auto":
        args.sparse = f"models/{baseline_to_beat()}.pkl"
    m = TransformerStanceModel.load(args.model, device=args.device)
    probs = m.predict_proba(gold["text"].tolist(), gold["language"].tolist())
    gold["pred"] = [LABELS[i] for i in probs.argmax(1)]
    gold["conf"] = probs.max(1)
    index = ["<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:900px;margin:auto}div.card{border:1px solid #ddd;border-radius:6px;padding:8px;margin:10px 0}</style></head><body>",
             f"<h1>Explanations: {m.name}</h1><p>Blue = pushes toward the predicted label, orange = against it. Integrated Gradients (Captum) and LIME.</p>"]
    summary = []
    for lang in ("hi", "bn"):
        for kind in ("correct", "wrong"):
            sub = gold[(gold.language == lang) & ((gold.pred == gold.label) if kind == "correct" else (gold.pred != gold.label))]
            sub = sub.sample(n=min(args.n, len(sub)), random_state=args.seed)
            index.append(f"<h2>{lang} / {kind} ({len(sub)})</h2>")
            for k, (_, r) in enumerate(sub.iterrows()):
                ig = integrated_gradients(m, r["text"], lang)
                lm = lime_explanation(m, r["text"], lang, num_samples=200)
                card = (f"<div class='card'><b>gold:</b> {r['label']} &nbsp; <b>pred:</b> {r['pred']} ({r['conf']:.2f}) &nbsp; <b>category:</b> {r.get('category', '')}"
                        f"<br><small>{r.get('justification', '')}</small>" + render_html(ig, "IG") + render_html(lm, "LIME") + "</div>")
                (out / f"{lang}_{kind}_{k}.html").write_text("<meta charset='utf-8'>" + card, encoding="utf-8")
                index.append(card)
                summary.append({"id": r["id"], "language": lang, "kind": kind, "label": r["label"], "pred": r["pred"], "category": r.get("category", ""),
                                "ig_top": top_tokens(ig, 6), "lime_top": top_tokens(lm, 6)})
    (out / "index.html").write_text("\n".join(index) + "</body></html>", encoding="utf-8")
    save_json(summary, out / "summary.json")
    sp = abs_path(args.sparse)
    if sp.exists():
        with open(sp, "rb") as f:
            sm = pickle.load(f)
        base = getattr(sm, "base", sm)
        if hasattr(base, "top_features"):
            feats = base.top_features(30)
            save_json(feats, out / "tfidf_top_features.json")
            import matplotlib.pyplot as plt

            from stance.evaluation.plotting import LABEL_COLOURS, style

            style()
            fig, axes = plt.subplots(1, 3, figsize=(14, 6))
            for ax, lab in zip(axes, LABELS):
                items = feats[lab][:20][::-1]
                ax.barh([t for t, _ in items], [c for _, c in items], color=LABEL_COLOURS[lab], edgecolor="#fcfcfb", linewidth=2, height=0.7)
                ax.set_title(f"TF-IDF LR: top features for {lab}"); ax.grid(axis="y", visible=False)
            fig.tight_layout(); fig.savefig(abs_path("results/figures/tfidf_top_features.png")); plt.close(fig)
    print(f"-> {out / 'index.html'}")


if __name__ == "__main__":
    main()
