#!/usr/bin/env python
"""Evaluate a model (final / efficient / any saved transformer dir / predictions CSV) on a gold CSV.

    python scripts/evaluate.py --gold data/gold/test_handcrafted.csv --model final
    python scripts/evaluate.py --gold data/gold/test_real.csv --model final --agreement
    python scripts/evaluate.py --gold data/gold/test_handcrafted.csv --predictions results/predictions/gold_xxx.csv

Reports Macro F1, per-class P/R/F1, accuracy, confusion matrix, per-language and per-category
scores; with --agreement also Cohen's kappa between the human labels and the model (for the
optional real YouTube set).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stance.evaluation import evaluate, evaluate_by, summarize  # noqa: E402
from stance.utils import LABELS, abs_path, read_data, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--model", default="final", help="final | efficient | path to a saved transformer dir")
    ap.add_argument("--predictions", default=None, help="CSV with id,pred (skips model loading)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    gold = read_data(args.gold)
    gold = gold[gold["label"].isin(LABELS)].reset_index(drop=True)
    if "language" not in gold.columns or (gold["language"] == "").all():
        from predict import guess_language

        gold["language"] = gold["text"].map(guess_language)
    if args.predictions:
        pred_df = read_data(args.predictions)
        merged = gold.merge(pred_df[["id", "pred" if "pred" in pred_df.columns else "label"]].rename(columns={"label": "pred"}), on="id")
        preds = merged["pred"].tolist(); gold = merged
        probs = None
    else:
        if args.model in ("final", "efficient"):
            from predict import FinalModel

            fm = FinalModel(args.model, device=args.device)
            probs = fm.predict_proba(gold["text"].tolist(), gold["language"].tolist())
        else:
            from stance.models.transformer import TransformerStanceModel

            m = TransformerStanceModel.load(args.model, device=args.device)
            probs = m.predict_proba(gold["text"].tolist(), gold["language"].tolist())
        preds = [LABELS[i] for i in np.asarray(probs).argmax(1)]
    res = evaluate(gold["label"], preds, gold["language"])
    print(summarize(res))
    print("confusion (rows=true, cols=pred, order Favour/Against/None):")
    print(np.array(res["confusion_matrix"]))
    if "category" in gold.columns:
        res["by_category"] = evaluate_by(gold["label"], preds, gold["category"])
        print(pd.DataFrame(res["by_category"]).T.sort_values("macro_f1").to_string(float_format=lambda x: f"{x:.3f}"))
    if args.agreement:
        res["cohen_kappa"] = float(cohen_kappa_score(gold["label"], preds))
        res["agreement_rate"] = float(np.mean([a == b for a, b in zip(gold["label"], preds)]))
        print(f"Cohen's kappa (human vs model): {res['cohen_kappa']:.3f}; raw agreement {res['agreement_rate']:.3f}")
    out = args.out or ("results/real_agreement.json" if args.agreement else f"results/eval_{Path(args.gold).stem}_{Path(str(args.model)).name}.json")
    save_json(res, out)
    if probs is not None:
        pd.DataFrame({"id": gold["id"], "text": gold["text"], "label": gold["label"], "pred": preds,
                      **{f"p_{l}": np.asarray(probs)[:, i] for i, l in enumerate(LABELS)}}
                     ).to_csv(abs_path(out).with_suffix(".csv"), index=False, encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
