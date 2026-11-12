#!/usr/bin/env python
"""Phase 9b: error taxonomy and model disagreement on the gold set (+ dev).

    python scripts/error_analysis.py --final results/predictions/gold_final_<cand>.csv --baseline results/predictions/gold_tfidf_lr.csv [--others gold_x.csv ...]

Categorises every error with rule-based tags (several tags per error are possible):
  sarcasm, implicit, negation, codemix, romanized, oov/typo, short, long, different_target, conspiracy,
  neutral_topic, offtopic, label_noise (dev only: lexicon and all models agree against the gold label),
  translation_artefact / synthetic_artefact (dev only, by source_type).
Writes results/error_analysis.json, results/error_examples.csv and results/figures/error_taxonomy.png,
plus a "fixed / worsened" comparison between the baseline and the final model per category.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stance.preprocessing.normalize import code_mix_ratio, detect_script  # noqa: E402
from stance.utils import abs_path, read_data, save_json  # noqa: E402

NEG = re.compile(r"नहीं|नही|गलत|मत |না |নেই|নয়|ভুল|nahi|nahin|nhi|not |never|no ")
SARC = re.compile(r"🙄|😂|🤣|🤡|हाँ हाँ|हां हां|हां जी|वाह|बाह|হ্যাঁ হ্যাঁ|haan haan|lol|wah ")
CLIMATE = re.compile(r"जलवायु|वार्मिंग|क्लाइमेट|climate|warming|জলবায়ু|ওয়ার্মিং|ক্লাইমেট|উষ্ণায়ন", re.I)


def tags_for(row: pd.Series) -> list[str]:
    t = str(row["text"]); tags = []
    cat = row.get("category", "")
    cat = "" if cat is None or (isinstance(cat, float) and cat != cat) else str(cat)  # dev rows: NaN category
    if cat and cat not in ("explicit", "nan"):
        tags.append(cat)
    if NEG.search(t) and "negation" not in tags:
        tags.append("negation")
    if SARC.search(t) and "sarcasm" not in tags:
        tags.append("sarcasm")
    if not CLIMATE.search(t) and "implicit" not in tags and row["label"] != "None":
        tags.append("implicit")
    cm = code_mix_ratio(t); scr = detect_script(t)
    if scr == "latin" and "romanized" not in tags:
        tags.append("romanized")
    if scr == "mixed" or (0.2 < cm < 0.8 and "codemixed" not in tags):
        if "codemixed" not in tags:
            tags.append("codemixed")
    if len(t) < 20 and "short" not in tags:
        tags.append("short")
    if len(t) > 250 and "long" not in tags:
        tags.append("long")
    st = str(row.get("source_type", ""))
    if st == "translated":
        tags.append("translation_artefact?")
    if st == "synthetic":
        tags.append("synthetic_artefact?")
    return tags or ["other"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="auto", help="gold predictions CSV, or auto = final model")
    ap.add_argument("--baseline", default="auto", help="gold predictions CSV, or auto = baseline to beat")
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--gold", default="data/gold/test_handcrafted.csv")
    ap.add_argument("--dev_final", default=None, help="dev predictions of the final transformer (dev_<run>.csv)")
    ap.add_argument("--min_errors", type=int, default=100)
    args = ap.parse_args()
    from stance.evaluation.select import baseline_to_beat, best_transformer, final_candidate

    if args.final == "auto":
        cand = final_candidate()
        args.final = f"results/predictions/gold_final_{cand}.csv" if cand else None
        if not args.final or not abs_path(args.final).exists():
            run, _ = best_transformer(); args.final = f"results/predictions/gold_{run}.csv"
    if args.baseline == "auto":
        args.baseline = f"results/predictions/gold_{baseline_to_beat()}.csv"
    if args.dev_final is None:
        run, _ = best_transformer(); args.dev_final = f"results/predictions/dev_{run}.csv"
    print(f"[errors] final={args.final} baseline={args.baseline} dev={args.dev_final}")
    gold = read_data(args.gold)
    fin = read_data(args.final).rename(columns={"pred": "pred_final"})[["id", "pred_final"]]
    bas = read_data(args.baseline).rename(columns={"pred": "pred_baseline"})[["id", "pred_baseline"]]
    df = gold.merge(fin, on="id").merge(bas, on="id")
    for i, o in enumerate(args.others):
        od = read_data(o).rename(columns={"pred": f"pred_other{i}"})[["id", f"pred_other{i}"]]
        df = df.merge(od, on="id")
    df["err_final"] = df.pred_final != df.label
    df["err_baseline"] = df.pred_baseline != df.label
    df["tags"] = df.apply(tags_for, axis=1)
    errors = df[df.err_final].copy()
    frames = [errors.assign(split="gold")]
    if args.dev_final and abs_path(args.dev_final).exists():
        dev = read_data("data/processed/dev.csv")
        dv = read_data(args.dev_final).rename(columns={"pred": "pred_final"})[["id", "pred_final"]]
        dd = dev.merge(dv, on="id")
        dd = dd[dd.pred_final != dd.label].copy()
        dd["tags"] = dd.apply(tags_for, axis=1)
        dd["err_baseline"] = np.nan
        frames.append(dd.assign(split="dev").sample(n=min(len(dd), max(0, args.min_errors - len(errors)) + 50), random_state=42))
    allerr = pd.concat(frames, ignore_index=True)
    counts = {}
    for tags in allerr["tags"]:
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    counts = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    # fixed / worsened per category (gold only)
    per_cat = {}
    for cat, g in df.groupby("category"):
        per_cat[cat] = {"n": int(len(g)), "baseline_errors": int(g.err_baseline.sum()), "final_errors": int(g.err_final.sum()),
                        "fixed_by_final": int((g.err_baseline & ~g.err_final).sum()), "broken_by_final": int((~g.err_baseline & g.err_final).sum()),
                        "disagree": int((g.pred_final != g.pred_baseline).sum())}
    confusion_pairs = errors.groupby(["label", "pred_final"]).size().sort_values(ascending=False)
    report = {"n_errors_total_categorised": int(len(allerr)), "n_gold_errors": int(len(errors)), "tag_counts": counts,
              "per_category_fixed_worsened": per_cat, "top_confusions_final": {f"{a}->{b}": int(n) for (a, b), n in confusion_pairs.items()},
              "disagreement_rate_final_vs_baseline": float((df.pred_final != df.pred_baseline).mean()),
              "both_wrong": int((df.err_final & df.err_baseline).sum()), "only_final_wrong": int((df.err_final & ~df.err_baseline).sum()),
              "only_baseline_wrong": int((~df.err_final & df.err_baseline).sum())}
    save_json(report, "results/error_analysis.json")
    cols = [c for c in ["split", "id", "language", "category", "source_type", "label", "pred_final", "pred_baseline", "tags", "text", "justification"] if c in allerr.columns]
    allerr[cols].to_csv(abs_path("results/error_examples.csv"), index=False, encoding="utf-8")
    import matplotlib.pyplot as plt

    from stance.evaluation.plotting import SERIES, bars, style

    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    items = list(counts.items())[:14][::-1]
    bars(axes[0], [k for k, _ in items], [v for _, v in items], SERIES[0], horizontal=True)
    axes[0].set_title(f"Error taxonomy ({len(allerr)} errors, multi-tag)")
    cats = list(per_cat); x = np.arange(len(cats)); w = 0.38
    axes[1].bar(x - w / 2, [per_cat[c]["baseline_errors"] for c in cats], w, color=SERIES[1], label="baseline errors", edgecolor="#fcfcfb", linewidth=2)
    axes[1].bar(x + w / 2, [per_cat[c]["final_errors"] for c in cats], w, color=SERIES[0], label="final errors", edgecolor="#fcfcfb", linewidth=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats, rotation=45, ha="right", fontsize=8); axes[1].legend(); axes[1].grid(axis="x", visible=False)
    axes[1].set_title("Gold errors per category: baseline vs final")
    fig.tight_layout(); fig.savefig(abs_path("results/figures/error_taxonomy.png")); plt.close(fig)
    print(pd.DataFrame(per_cat).T.to_string())
    print({k: v for k, v in report.items() if k != "per_category_fixed_worsened"})


if __name__ == "__main__":
    main()
