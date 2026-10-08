"""Apply the data-quality fixes motivated by notebooks/01_data_exploration.ipynb.

Each rule is listed in RULES so the report can cite exactly what was removed.

Usage:
    python data_prep/clean_corpus.py --inp data/processed/corpus_v1.csv \
        --out data/processed/corpus_v1_clean.csv
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Sources whose original target matches the project claim closely.
CORE = {"semeval16", "gwsd", "climate_reddit"}
# Related target or different language/domain: useful, but kept separable.
AUXILIARY = {"case2024", "dhoroni"}

SPAM_TEMPLATE = "you've been fooled"
NON_CONTENT = re.compile(r"#\w+|https?://\S+|@\w+")

RULES = {
    "silver_labels": "drop model-generated (pseudo) labels",
    "spam_template": "drop CASE rows built on one repeated spam template",
    "no_content": "drop rows with < 3 words once hashtags, URLs, mentions are removed",
    "conflicting_duplicates": "drop all copies of a text that carries different labels",
    "duplicates": "keep one copy of a repeated text within a source",
}


def dedup_key(text: str) -> str:
    text = re.sub(r"https?://\S+|@\w+", "", text.lower())
    return re.sub(r"\W+", " ", text).strip()


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned corpus and a per-source table of rows removed by each rule."""
    df = df.copy()
    # SemEval tweets all end with the collection hashtag; it carries no stance.
    df["text"] = df["text"].str.replace("#SemST", "", regex=False).str.strip()

    removed = {}

    def drop(mask: pd.Series, rule: str) -> None:
        nonlocal df
        removed[rule] = df.loc[mask, "source"].value_counts()
        df = df[~mask]

    drop(df["quality"] == "silver", "silver_labels")
    drop(df["text"].str.lower().str.contains(SPAM_TEMPLATE, regex=False), "spam_template")
    content_words = df["text"].map(lambda t: len(NON_CONTENT.sub(" ", t).split()))
    drop((content_words < 3) & (df["lang"] == "en"), "no_content")

    key = df["source"] + "|" + df["text"].map(dedup_key)
    conflicting = key.map(df.groupby(key)["label"].nunique()) > 1
    drop(conflicting, "conflicting_duplicates")
    key = df["source"] + "|" + df["text"].map(dedup_key)
    drop(key.duplicated(), "duplicates")

    df["tier"] = df["source"].map(lambda s: "core" if s in CORE else "auxiliary")
    report = pd.DataFrame(removed).fillna(0).astype(int)
    return df.reset_index(drop=True), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inp", type=Path, default=Path("data/processed/corpus_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/corpus_v1_clean.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cleaned, report = clean(pd.read_csv(args.inp))
    cleaned.to_csv(args.out, index=False)
    log.info("rows removed per rule:\n%s", report.to_string())
    log.info("wrote %d rows to %s", len(cleaned), args.out)


if __name__ == "__main__":
    main()
