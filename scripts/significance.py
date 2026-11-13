#!/usr/bin/env python
"""Paired bootstrap on the gold set: 95 % CI of each model's macro F1 and of the difference
between every model and a reference model (same resampled items for both).

    python scripts/significance.py --reference tfidf_cnb [--models a b c] [--n 2000]

Models are names X with a file results/predictions/gold_X.csv (columns id,label,pred).
Writes results/significance.csv. A difference whose 95 % CI excludes 0 is marked significant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stance.utils import LABELS, abs_path, read_data  # noqa: E402


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="auto", help="model name, or auto = baseline to beat")
    ap.add_argument("--models", nargs="*", default=None, help="default: every gold_*.csv")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.reference == "auto":
        from stance.evaluation.select import baseline_to_beat

        args.reference = baseline_to_beat()
        print(f"[significance] reference = {args.reference}")
    pred_dir = abs_path("results/predictions")
    names = args.models or sorted(p.stem[len("gold_"):] for p in pred_dir.glob("gold_*.csv")
                                  if "smoke" not in p.stem and "ablation" not in p.stem)
    gold = read_data("data/gold/test_handcrafted.csv")[["id", "label"]]
    preds = {}
    for n in set(names) | {args.reference}:
        f = pred_dir / f"gold_{n}.csv"
        if f.exists():
            d = gold.merge(read_data(f)[["id", "pred"]], on="id", how="left")
            if d["pred"].isna().any() or len(d) != len(gold):
                continue
            preds[n] = d["pred"].to_numpy()
    y = gold["label"].to_numpy()
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(y), size=(args.n, len(y)))
    ref = preds[args.reference]
    ref_boot = np.array([macro_f1(y[i], ref[i]) for i in idx])
    rows = []
    for n, p in preds.items():
        boot = np.array([macro_f1(y[i], p[i]) for i in idx])
        diff = boot - ref_boot
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"model": n, "gold_macro_f1": macro_f1(y, p), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     f"diff_vs_{args.reference}": macro_f1(y, p) - macro_f1(y, ref), "diff_ci_low": lo, "diff_ci_high": hi,
                     "significant": bool(lo > 0 or hi < 0)})
    t = pd.DataFrame(rows).sort_values("gold_macro_f1", ascending=False)
    t.to_csv(abs_path("results/significance.csv"), index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
