#!/usr/bin/env python
"""Build every results table used in README.md and reports/ from files in results/.

    python scripts/make_tables.py            # -> results/tables.md

Sources: results/baselines.csv, results/predictions/metrics_*_s*.json (transformers),
results/ablations.csv, results/llm.csv, results/robustness.csv, results/final.json,
results/error_analysis.json, data/processed/build_stats.json. Missing sources are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stance.utils import abs_path  # noqa: E402


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def baselines_table() -> str | None:
    p = abs_path("results/baselines.csv")
    if not p.exists():
        return None
    b = pd.read_csv(p)
    cols = ["model", "family", "cv_macro_f1_mean", "cv_macro_f1_std", "dev_macro_f1", "gold_macro_f1",
            "gold_f1_hi", "gold_f1_bn", "challenge_fail_rate", "train_seconds", "cpu_ms_per_example"]
    b = b[[c for c in cols if c in b.columns]].sort_values("gold_macro_f1", ascending=False)
    b = b.rename(columns={"cv_macro_f1_mean": "CV F1", "cv_macro_f1_std": "CV std", "dev_macro_f1": "dev F1",
                          "gold_macro_f1": "gold F1", "gold_f1_hi": "gold hi", "gold_f1_bn": "gold bn",
                          "challenge_fail_rate": "challenge fail", "train_seconds": "train s", "cpu_ms_per_example": "CPU ms/ex"})
    real = b[~b["model"].isin(["majority", "stratified_random", "lexicon"])]
    best = real.iloc[0]["model"] if len(real) else None
    note = (f"\nBaseline to beat (best trained baseline by gold macro F1): **{best}**. The lexicon was written by the "
            "same author as the gold set, so its gold score is optimistic (ASSUMPTIONS #7).\n")
    return "### Baselines (Phase 3)\n\n" + _md(b) + "\n" + note


def transformers_table() -> str | None:
    rows = []
    for p in sorted(abs_path("results/predictions").glob("metrics_*_s*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "history" not in d or "ablation" in d.get("run", "") or "smoke" in d.get("run", ""):
            continue
        rows.append({"run": d["run"], "dev F1": d["dev"]["macro_f1"], "gold F1": d["gold"]["macro_f1"],
                     "gold hi": d["gold"]["per_language"].get("hi", {}).get("macro_f1"),
                     "gold bn": d["gold"]["per_language"].get("bn", {}).get("macro_f1"),
                     "challenge fail": d["challenge"].get("overall_fail_rate"),
                     "epochs": len(d["history"]), "minutes": d["train_seconds"] / 60})
    if not rows:
        return None
    t = pd.DataFrame(rows).sort_values("gold F1", ascending=False)
    # seed aggregation for configs with several seeds
    t["config"] = t["run"].str.replace(r"_s\d+$", "", regex=True)
    multi = t.groupby("config").filter(lambda g: len(g) > 1)
    out = "### Transformers (Phase 4, 1 seed unless stated)\n\n" + _md(t.drop(columns=["config"])) + "\n"
    if t["gold hi"].isna().any():
        out += ("\nRuns with an empty `gold hi` column (BanglaBERT) are trained and scored on Bengali only; "
                "their dev/gold F1 covers the Bengali half and is not comparable with two-language runs.\n")
    if len(multi):
        agg = multi.groupby("config").agg(seeds=("run", "count"), gold_mean=("gold F1", "mean"), gold_std=("gold F1", "std"),
                                          dev_mean=("dev F1", "mean"), fail_mean=("challenge fail", "mean")).reset_index()
        out += "\nSeed variance (all seeds):\n\n" + _md(agg) + "\n"
        # A run whose DEV macro F1 stays below 0.5 failed to start learning (loss flat, near-majority
        # predictions; a known BERT fine-tuning instability). Detected on dev only, never on gold.
        bad = multi[multi["dev F1"] < 0.5]
        if len(bad):
            ok = multi[multi["dev F1"] >= 0.5]
            agg2 = ok.groupby("config").agg(seeds=("run", "count"), gold_mean=("gold F1", "mean"),
                                            gold_std=("gold F1", "std"), dev_mean=("dev F1", "mean"),
                                            fail_mean=("challenge fail", "mean")).reset_index()
            out += ("\nFailed-to-train runs (dev macro F1 < 0.5, detected on dev only): "
                    + ", ".join(f"`{r}`" for r in bad["run"]) + ". Seed variance without them:\n\n" + _md(agg2) + "\n")
    return out


def simple_table(path: str, title: str, sort: str | None = None) -> str | None:
    p = abs_path(path)
    if not p.exists():
        return None
    t = pd.read_csv(p)
    if sort and sort in t.columns:
        t = t.sort_values(sort, ascending=False)
    return f"### {title}\n\n" + _md(t) + "\n"


def robustness_table() -> str | None:
    p = abs_path("results/robustness.csv")
    if not p.exists():
        return None
    t = pd.read_csv(p, index_col=0)
    keep = [c for c in t.columns if not c.startswith(("final_", "ablation"))]
    t = t[keep]
    return "### Challenge-suite failure rate per category (Phase 7, lower is better)\n\n" + t.reset_index().rename(columns={"index": "category"}).to_markdown(index=False, floatfmt=".2f") + "\n"


def final_tables() -> str | None:
    p = abs_path("results/final.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    out = "### Final-model candidates (Phase 8)\n\n"
    c = pd.DataFrame(d["candidates"])
    out += _md(c[[x for x in ["candidate", "dev_macro_f1", "dev_ece", "gold_macro_f1", "gold_f1_hi", "gold_f1_bn", "challenge_fail_rate", "score"] if x in c.columns]]) + "\n"
    out += f"\nSelected: **{d['final'].get('candidate')}**; calibration `{json.dumps(d['final'].get('calibration'))}`.\n"
    if d.get("accuracy_vs_cost"):
        out += "\n### Accuracy vs cost (CPU)\n\n" + _md(pd.DataFrame(d["accuracy_vs_cost"])) + "\n"
    g = d.get("final_gold", {})
    if g.get("per_class"):
        pc = pd.DataFrame(g["per_class"]).T.reset_index().rename(columns={"index": "class"})
        out += "\n### Final model on the gold set: per-class scores\n\n" + _md(pc) + "\n"
        cm = pd.DataFrame(g["confusion_matrix"], index=[f"true {l}" for l in g["confusion_labels"]],
                          columns=[f"pred {l}" for l in g["confusion_labels"]])
        out += "\nConfusion matrix:\n\n" + cm.reset_index().rename(columns={"index": ""}).to_markdown(index=False) + "\n"
    return out


def errors_table() -> str | None:
    p = abs_path("results/error_analysis.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    tags = pd.DataFrame(list(d["tag_counts"].items()), columns=["error tag", "count"])
    pc = pd.DataFrame(d["per_category_fixed_worsened"]).T.reset_index().rename(columns={"index": "category"})
    return ("### Error taxonomy (Phase 9)\n\n" + f"{d['n_errors_total_categorised']} errors categorised "
            f"({d['n_gold_errors']} on gold).\n\n" + _md(tags) + "\n\nFixed / broken by the final model vs the baseline to beat, per gold category:\n\n" + _md(pc) + "\n")


def data_table() -> str | None:
    p = abs_path("data/processed/build_stats.json")
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    from stance.utils import read_data

    files = {"train": "data/processed/train.csv", "dev": "data/processed/dev.csv", "gold": "data/gold/test_handcrafted.csv"}
    rows = [{"split": k, "rows": len(read_data(f))} for k, f in files.items()]  # actual files (after leakage cleaning)
    st = pd.DataFrame(list(s["train_source_type_counts"].items()), columns=["source_type", "train rows"])
    return ("### Data (Phase 1)\n\n" + _md(pd.DataFrame(rows)) + "\n\n" + _md(st) +
            f"\n\nTranslation kept {s['translation']['n_kept']}/{s['translation']['n_translated']} "
            f"(LaBSE back-translation threshold {s['translation']['threshold']}); synthetic share {s['synthetic_share']:.3f}.\n")


def gap_table() -> str | None:
    """Synthetic / translated vs handcrafted gap: out-of-fold macro F1 per train source_type
    (from results/oof/<model>.npy, aligned with train.csv) next to the gold-set macro F1."""
    import numpy as np
    from sklearn.metrics import f1_score

    from stance.utils import LABELS, read_data

    b = abs_path("results/baselines.csv")
    if not b.exists():
        return None
    tab = pd.read_csv(b)
    train = read_data("data/processed/train.csv")
    rows = []
    for _, r in tab.iterrows():
        f = abs_path(f"results/oof/{r['model']}.npy")
        if r["model"] in ("majority", "stratified_random") or not f.exists():
            continue
        oof = np.load(f)
        if len(oof) != len(train):
            continue  # stale OOF from an older build
        pred = np.array(LABELS)[oof.argmax(1)]
        row = {"model": r["model"]}
        for st in ["original", "translated", "transliterated", "codemixed", "synthetic"]:
            m = (train["source_type"] == st).to_numpy()
            if m.any():
                row[f"OOF {st}"] = f1_score(train["label"][m], pred[m], labels=LABELS, average="macro", zero_division=0)
        row["gold (handcrafted)"] = r["gold_macro_f1"]
        rows.append(row)
    if not rows:
        return None
    t = pd.DataFrame(rows).sort_values("gold (handcrafted)", ascending=False)
    head = ("### Synthetic / translated vs handcrafted gap\n\nOut-of-fold macro F1 on each kind of training text "
            "versus macro F1 on the hand-written gold comments.\n\n")
    return head + _md(t) + "\n"


def main() -> None:
    parts = [x for x in [data_table(), baselines_table(), gap_table(), transformers_table(),
                         simple_table("results/ablations.csv", "Augmentation ablations (Phase 6)"),
                         simple_table("results/llm.csv", "Local LLM (Phase 5)"),
                         robustness_table(), final_tables(), errors_table()] if x]
    out = abs_path("results/tables.md")
    out.write_text("# Results tables (generated by scripts/make_tables.py)\n\n" + "\n".join(parts), encoding="utf-8")
    print(f"-> {out} ({len(parts)} sections)")


if __name__ == "__main__":
    main()
