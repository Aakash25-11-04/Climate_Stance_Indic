#!/usr/bin/env python
"""Phase 2: exploratory data analysis -> results/figures/*.png + results/eda_stats.json.
The notebook notebooks/eda.ipynb calls these same functions and adds commentary.

    python scripts/eda.py [--train data/processed/train.csv --dev ... --gold ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import emoji as emoji_lib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from stance.data.dedup import cross_split_leakage  # noqa: E402
from stance.data.transliterate import is_romanized_indic  # noqa: E402
from stance.evaluation.plotting import INK2, LABEL_COLOURS, LANG_COLOURS, SEQ_BLUE, SERIES, bars, style  # noqa: E402
from stance.preprocessing.normalize import Preprocessor, code_mix_ratio  # noqa: E402
from stance.utils import LABELS, abs_path, read_data, save_json  # noqa: E402

FIG = abs_path("results/figures")
_URL = re.compile(r"https?://\S+|www\.\S+")
_HASH = re.compile(r"#\w+")
_MENT = re.compile(r"(?<!\w)@\w+")
STOP = set("है हैं का की के को में और से ये यह वो वह पर भी तो ही नहीं एक कि हम आप मैं है। না এর এই ও কি যে করে হয় আর না এ তে কে আমি আমরা the a an of to is in and it for on that this".split())


def _counts(df, col):
    return {k: int(v) for k, v in df[col].value_counts().items()}


def fig_label_distribution(train, dev, gold):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (name, df) in zip(axes, [("train", train), ("dev", dev), ("gold (test)", gold)]):
        c = df["label"].value_counts().reindex(LABELS).fillna(0)
        bars(ax, LABELS, c.values, [LABEL_COLOURS[l] for l in LABELS])
        ax.set_title(f"{name}: n={len(df)}")
    fig.suptitle("Label distribution per split", fontweight="bold")
    fig.savefig(FIG / "label_distribution.png"); plt.close(fig)


def fig_label_by(train, col, fname, title):
    style()
    ct = pd.crosstab(train[col], train["label"]).reindex(columns=LABELS).fillna(0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ct) + 1.5))
    left = np.zeros(len(ct))
    for lab in LABELS:
        ax.barh(ct.index, ct[lab].values, left=left, color=LABEL_COLOURS[lab], edgecolor="#fcfcfb", linewidth=2, height=0.7, label=lab)
        left += ct[lab].values
    for i, tot in enumerate(left):
        ax.text(tot, i, f" {int(tot)}", va="center", color=INK2, fontsize=8.5)
    ax.grid(axis="y", visible=False); ax.set_title(title); ax.legend(ncol=3, loc="lower right")
    fig.savefig(FIG / fname); plt.close(fig)
    return {str(k): {l: int(v) for l, v in r.items()} for k, r in ct.iterrows()}


def fig_script(train, gold):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    out = {}
    for ax, (name, df) in zip(axes, [("train", train), ("gold", gold)]):
        c = df["script"].value_counts()
        out[name] = {k: int(v) for k, v in c.items()}
        bars(ax, list(c.index), c.values, SERIES[: len(c)])
        ax.set_title(f"script ({name})"); ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIG / "script_distribution.png"); plt.close(fig)
    return out


def fig_length_codemix(train, gold):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, df, col in [("train", train, SERIES[0]), ("gold", gold, SERIES[1])]:
        axes[0].hist(df["text"].str.len().clip(upper=400), bins=40, alpha=0.75, color=col, label=name, edgecolor="#fcfcfb", linewidth=0.5)
        axes[1].hist(df["text"].map(code_mix_ratio), bins=20, alpha=0.75, color=col, label=name, edgecolor="#fcfcfb", linewidth=0.5)
    axes[0].set_title("text length (chars, clipped at 400)"); axes[0].legend()
    axes[1].set_title("code-mixing ratio (share of Latin-script tokens)"); axes[1].legend()
    fig.savefig(FIG / "length_and_codemix.png"); plt.close(fig)
    return {"train_len_median": float(train["text"].str.len().median()), "gold_len_median": float(gold["text"].str.len().median()),
            "train_codemix_mean": float(train["text"].map(code_mix_ratio).mean()), "gold_codemix_mean": float(gold["text"].map(code_mix_ratio).mean())}


def top_ngrams(train, n=15):
    pre = Preprocessor(indic_normalize=False)
    out = {}
    for lang in ("hi", "bn"):
        for lab in LABELS:
            sub = train[(train.language == lang) & (train.label == lab)]
            cnt = Counter()
            for t in sub["text"]:
                toks = [w for w in re.findall(r"\w+", pre(t)) if w not in STOP and len(w) > 1]
                cnt.update(toks)
                cnt.update(" ".join(p) for p in zip(toks, toks[1:]))
            out[f"{lang}/{lab}"] = cnt.most_common(n)
    return out


def fig_top_ngrams(ngrams):
    style()
    keys = list(ngrams)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, k in zip(axes.ravel(), keys):
        items = ngrams[k][:12][::-1]
        if not items:
            continue
        ax.barh([w for w, _ in items], [c for _, c in items], color=LABEL_COLOURS[k.split("/")[1]], edgecolor="#fcfcfb", linewidth=2, height=0.7)
        ax.set_title(k); ax.grid(axis="y", visible=False)
    fig.suptitle("Top uni/bi-grams per class and language (train)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "top_ngrams.png"); plt.close(fig)


def surface_stats(df):
    t = df["text"]
    return {"emoji_share": float(t.map(lambda s: emoji_lib.emoji_count(s) > 0).mean()),
            "url_share": float(t.map(lambda s: bool(_URL.search(s))).mean()),
            "hashtag_share": float(t.map(lambda s: bool(_HASH.search(s))).mean()),
            "mention_share": float(t.map(lambda s: bool(_MENT.search(s))).mean()),
            "romanized_indic_share": float(t.map(is_romanized_indic).mean()),
            "mean_len": float(t.str.len().mean())}


def fig_surface(train_stats, gold_stats):
    style()
    keys = ["emoji_share", "url_share", "hashtag_share", "mention_share", "romanized_indic_share"]
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x - w / 2, [train_stats[k] for k in keys], w, color=SERIES[0], label="train", edgecolor="#fcfcfb", linewidth=2)
    ax.bar(x + w / 2, [gold_stats[k] for k in keys], w, color=SERIES[1], label="gold", edgecolor="#fcfcfb", linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels([k.replace("_share", "") for k in keys]); ax.legend(); ax.grid(axis="x", visible=False)
    ax.set_title("Share of texts with emoji / URL / hashtag / mention / Romanized Indic")
    fig.savefig(FIG / "surface_features.png"); plt.close(fig)


def fig_source_vs_gold_distribution(train, gold):
    """Distribution shift: length and code-mix per source_type vs handcrafted."""
    style()
    groups = [(st, train[train.source_type == st]["text"]) for st in ["original", "translated", "transliterated", "codemixed", "synthetic"] if (train.source_type == st).any()]
    groups.append(("handcrafted (gold)", gold["text"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].boxplot([g.str.len().clip(upper=400) for _, g in groups], tick_labels=[n for n, _ in groups], widths=0.5,
                    medianprops={"color": SERIES[1]}, boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2}, flierprops={"markersize": 2, "markeredgecolor": INK2})
    axes[0].set_title("text length by source_type vs gold"); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar([n for n, _ in groups], [g.map(code_mix_ratio).mean() for _, g in groups], color=SERIES[: len(groups)], edgecolor="#fcfcfb", linewidth=2, width=0.7)
    axes[1].set_title("mean code-mix ratio"); axes[1].tick_params(axis="x", rotation=20); axes[1].grid(axis="x", visible=False)
    fig.savefig(FIG / "source_type_vs_gold.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--dev", default="data/processed/dev.csv")
    ap.add_argument("--gold", default="data/gold/test_handcrafted.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    train, dev, gold = read_data(args.train), read_data(args.dev), read_data(args.gold)
    stats = {"rows": {"train": len(train), "dev": len(dev), "gold": len(gold)}}
    fig_label_distribution(train, dev, gold)
    stats["label_by_language"] = fig_label_by(train, "language", "label_by_language.png", "Train labels per language")
    stats["label_by_source_type"] = fig_label_by(train, "source_type", "label_by_source_type.png", "Train labels per source_type")
    stats["label_by_source"] = fig_label_by(train, "source", "label_by_source.png", "Train labels per original source")
    stats["script"] = fig_script(train, gold)
    stats.update(fig_length_codemix(train, gold))
    ng = top_ngrams(train); fig_top_ngrams(ng)
    stats["top_ngrams"] = {k: v[:10] for k, v in ng.items()}
    stats["surface_train"] = surface_stats(train); stats["surface_gold"] = surface_stats(gold)
    fig_surface(stats["surface_train"], stats["surface_gold"])
    fig_source_vs_gold_distribution(train, gold)
    stats["leakage"] = {"train_vs_dev": cross_split_leakage(train, dev), "train_vs_gold": cross_split_leakage(train, gold)}
    ch = abs_path("data/challenge/mft.csv")
    if ch.exists():
        stats["leakage"]["train_vs_challenge_mft"] = cross_split_leakage(train, read_data(ch))
    # random label-noise audit sample (50 rows, fixed seed) for manual inspection in the notebook
    audit = train.sample(n=min(50, len(train)), random_state=args.seed)[["id", "text", "language", "source_type", "label"]]
    audit.to_csv(abs_path("results/eda_label_audit_sample.csv"), index=False, encoding="utf-8")
    # hard / ambiguous: very short, no climate keyword, or labels that disagree with the lexicon
    from stance.models.sparse import LexiconModel

    lx = LexiconModel()
    lp = lx.predict(train["text"].tolist())
    disagree = train[(np.array(lp) != train["label"].values) & (train.source_type == "translated")].sample(n=min(40, int(((np.array(lp) != train["label"].values) & (train.source_type == "translated")).sum())), random_state=args.seed)
    disagree.assign(lexicon_pred=lx.predict(disagree["text"].tolist()))[["id", "text", "language", "label", "lexicon_pred"]].to_csv(abs_path("results/eda_hard_examples.csv"), index=False, encoding="utf-8")
    stats["lexicon_train_agreement"] = float(np.mean(np.array(lp) == train["label"].values))
    stats["short_texts_lt_15_chars"] = int((train["text"].str.len() < 15).sum())
    save_json(stats, "results/eda_stats.json")
    print({k: v for k, v in stats.items() if k in ("rows", "script", "leakage", "lexicon_train_agreement", "train_len_median", "gold_len_median", "train_codemix_mean", "gold_codemix_mean")})
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
