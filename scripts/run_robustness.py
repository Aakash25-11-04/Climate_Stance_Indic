#!/usr/bin/env python
"""Phase 7: robustness report. Aggregates every challenge_<model>.csv in results/predictions/
(written by run_baselines.py / train.py / run_llm.py / build_final.py) into
results/robustness.csv (failure rate per category x model) and a heatmap figure.
Optionally re-evaluates saved transformer directories that have no challenge file yet.

    python scripts/run_robustness.py [--models models/muril_pair ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stance.evaluation.challenge import evaluate_challenge, load_challenge  # noqa: E402
from stance.utils import abs_path, read_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="saved transformer dirs to (re)evaluate")
    ap.add_argument("--pred_dir", default="results/predictions")
    ap.add_argument("--out", default="results/robustness.csv")
    args = ap.parse_args()
    pred_dir = abs_path(args.pred_dir)
    challenge = load_challenge()
    for mdir in args.models:
        from stance.models.transformer import TransformerStanceModel

        m = TransformerStanceModel.load(mdir)
        res = evaluate_challenge(m, challenge)
        res["rows"].to_csv(pred_dir / f"challenge_{Path(mdir).name}.csv", index=False, encoding="utf-8")
        print(f"{Path(mdir).name}: fail {res['overall_fail_rate']:.3f}")
    tables = {}
    for p in sorted(pred_dir.glob("challenge_*.csv")):
        name = p.stem[len("challenge_"):]
        df = read_data(p)
        df["fail"] = pd.to_numeric(df["fail"])
        per = df.groupby("category")["fail"].mean()
        per["ALL (mft)"] = df[df["type"] == "mft"]["fail"].mean()
        per["ALL (invariance)"] = df[df["type"] == "invariance"]["fail"].mean() if (df["type"] == "invariance").any() else np.nan
        per["ALL (directional)"] = df[df["type"] == "directional"]["fail"].mean() if (df["type"] == "directional").any() else np.nan
        per["ALL"] = df["fail"].mean()
        for lang in ("hi", "bn"):
            per[f"ALL ({lang})"] = df[df["language"] == lang]["fail"].mean()
        tables[name] = per
    if not tables:
        print("no challenge predictions found"); return
    tab = pd.DataFrame(tables)
    tab = tab[sorted(tab.columns, key=lambda c: tab.loc["ALL", c])]
    tab.to_csv(abs_path(args.out))
    print(tab.to_string(float_format=lambda x: f"{x:.2f}"))
    # heatmap
    import matplotlib.pyplot as plt

    from stance.evaluation.plotting import INK2, style

    style()
    cats = [c for c in tab.index if not c.startswith("ALL")] + ["ALL"]
    sub = tab.loc[cats]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(sub.columns), 0.32 * len(sub) + 1.5))
    im = ax.imshow(sub.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sub.columns))); ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=8); ax.grid(False)
    for i in range(len(sub)):
        for j in range(len(sub.columns)):
            v = sub.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK2)
    ax.set_title("Challenge-suite failure rate per category (lower is better)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(abs_path("results/figures/robustness_heatmap.png")); plt.close(fig)
    print(f"-> {args.out}, results/figures/robustness_heatmap.png")


if __name__ == "__main__":
    main()
