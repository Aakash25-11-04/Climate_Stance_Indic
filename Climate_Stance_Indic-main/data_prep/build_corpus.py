"""Download public climate-stance datasets and map them to a shared schema.

Target claim of the project: "Climate change and global warming are serious concerns."
Every source is mapped to favour / against / none with respect to that claim.
How well each source's original target matches the claim is recorded in the
`target` column, because the mismatch matters for later analysis.

Usage:
    python data_prep/build_corpus.py --raw-dir data/raw --out data/processed/corpus_v1.csv
"""

from __future__ import annotations

import argparse
import html
import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

log = logging.getLogger(__name__)

TWEETEVAL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/stance/climate"
GWSD = "https://raw.githubusercontent.com/yiweiluo/GWStance/master/GWSD.tsv"
REDDIT = (
    "https://raw.githubusercontent.com/roopalv54/"
    "finegrained-climate-change-social-media/main/ClimateReddit"
)
CASE = "https://raw.githubusercontent.com/Salman1804102/CASE-EACL-2024/main/Dataset"
DHORONI = "https://raw.githubusercontent.com/ciol-researchlab/Dhoroni/main"

COLUMNS = [
    "uid",
    "text",
    "label",
    "source",
    "lang",
    "domain",
    "target",
    "orig_label",
    "orig_split",
    "quality",
    "agreement",
]


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        urlretrieve(url, dest)
    return dest


def clean(text: str) -> str:
    # Only undo HTML escaping and outer whitespace. Real normalisation belongs
    # to the preprocessing stage so it can be ablated.
    return html.unescape(str(text)).strip()


def load_semeval(raw: Path) -> pd.DataFrame:
    id2label = {0: "none", 1: "against", 2: "favour"}
    frames = []
    for split in ("train", "val", "test"):
        texts = fetch(f"{TWEETEVAL}/{split}_text.txt", raw / f"semeval_{split}_text.txt")
        labels = fetch(f"{TWEETEVAL}/{split}_labels.txt", raw / f"semeval_{split}_labels.txt")
        text_lines = texts.read_text(encoding="utf-8").splitlines()
        label_lines = [int(x) for x in labels.read_text().split()]
        if len(text_lines) != len(label_lines):
            raise ValueError(
                f"SemEval {split}: {len(text_lines)} texts vs {len(label_lines)} labels"
            )
        frames.append(
            pd.DataFrame(
                {
                    "text": text_lines,
                    "orig_label": label_lines,
                    "orig_split": split,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["orig_label"].map(id2label)
    return df.assign(
        source="semeval16",
        lang="en",
        domain="tweet",
        target="Climate Change is a Real Concern (near-identical)",
        quality="gold",
        agreement=float("nan"),
    )


def load_gwsd(raw: Path) -> pd.DataFrame:
    df = pd.read_csv(fetch(GWSD, raw / "GWSD.tsv"), sep="\t")
    workers = [c for c in df.columns if c.startswith("worker_")]
    votes = df[workers].apply(lambda r: r.value_counts(), axis=1).fillna(0)
    majority = votes.idxmax(axis=1)
    vote_share = votes.max(axis=1) / votes.sum(axis=1)

    # The released file carries posterior probabilities from the authors'
    # annotator model for most rows; prefer them over raw majority vote.
    posterior = df[["agree", "disagree", "neutral"]]
    has_post = posterior.notna().all(axis=1)
    post_label = (
        posterior.fillna(-1.0)
        .idxmax(axis=1)
        .map({"agree": "agrees", "disagree": "disagrees", "neutral": "neutral"})
    )
    orig = majority.where(~has_post, post_label)
    conf = vote_share.where(~has_post, posterior.max(axis=1))

    out = pd.DataFrame(
        {
            "text": df["sentence"],
            "orig_label": orig,
            "orig_split": df["in_held_out_test"].map({True: "test", False: "train"}),
            "agreement": conf,
        }
    )
    out["label"] = out["orig_label"].map(
        {"agrees": "favour", "disagrees": "against", "neutral": "none"}
    )
    return out.assign(
        source="gwsd",
        lang="en",
        domain="news",
        target="global warming is a serious concern (identical)",
        quality="gold",
    )


def load_climate_reddit(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gold rows where both annotators agree, silver pseudo-labelled rows)."""
    a1 = pd.read_csv(fetch(f"{REDDIT}/Reddit-Annotator1.tsv", raw / "reddit_a1.tsv"), sep="\t")
    a2 = pd.read_csv(fetch(f"{REDDIT}/Reddit-Annotator2.tsv", raw / "reddit_a2.tsv"), sep="\t")
    if not (a1["permalink"] == a2["permalink"]).all():
        raise ValueError("ClimateReddit annotator files are not row-aligned")
    id2label = {1: "favour", -1: "against", 0: "none"}
    agree = a1["opinion-label"] == a2["opinion-label"]
    gold = pd.DataFrame(
        {
            "text": a1.loc[agree, "body"],
            "orig_label": a1.loc[agree, "opinion-label"],
            "orig_split": "manual",
            "agreement": 1.0,
        }
    )
    gold["label"] = gold["orig_label"].map(id2label)
    gold = gold.assign(
        source="climate_reddit",
        lang="en",
        domain="reddit",
        target="climate change prevention (related)",
        quality="gold",
    )

    pseudo = pd.read_csv(
        fetch(f"{REDDIT}/Pseudo-labeled-Reddit.tsv", raw / "reddit_pseudo.tsv"), sep="\t"
    )
    silver = pd.DataFrame(
        {
            "text": pseudo["body"],
            "orig_label": pseudo["opinion"],
            "orig_split": "pseudo",
            "agreement": float("nan"),
        }
    )
    silver["label"] = silver["orig_label"].map(id2label)
    silver = silver.assign(
        source="climate_reddit_pseudo",
        lang="en",
        domain="reddit",
        target="climate change prevention (related)",
        quality="silver",
    )
    return gold, silver


def load_case2024(raw: Path) -> pd.DataFrame:
    train = pd.read_csv(fetch(f"{CASE}/SubTask-C-train.csv", raw / "case_c_train.csv"))
    val_text = pd.read_csv(
        fetch(f"{CASE}/SubTask-C(indextweet)val.csv", raw / "case_c_val_text.csv")
    )
    val_label = pd.read_csv(
        fetch(f"{CASE}/SubTask-C(indexlabel)val.csv", raw / "case_c_val_label.csv")
    )
    val = val_text.merge(val_label, on="index", how="inner", validate="one_to_one")
    df = pd.concat(
        [train.assign(orig_split="train"), val.assign(orig_split="val")], ignore_index=True
    )
    out = pd.DataFrame(
        {
            "text": df["tweet"],
            "orig_label": df["label"],
            "orig_split": df["orig_split"],
            "agreement": float("nan"),
        }
    )
    out["label"] = out["orig_label"].map({1: "favour", 2: "against", 3: "none"})
    return out.assign(
        source="case2024",
        lang="en",
        domain="tweet",
        target="climate activism (related)",
        quality="gold",
    )


def load_dhoroni(raw: Path) -> pd.DataFrame:
    meta = pd.read_csv(fetch(f"{DHORONI}/dataset.csv", raw / "dhoroni.csv"))
    archive = fetch(f"{DHORONI}/dataset_files.zip", raw / "dhoroni_files.zip")
    with zipfile.ZipFile(archive) as zf:
        bodies = {
            Path(name).name: zf.read(name).decode("utf-8", errors="replace")
            for name in zf.namelist()
            if name.endswith(".txt")
        }
    meta["body"] = meta["File"].map(bodies)
    missing = meta["body"].isna().sum()
    if missing:
        log.warning("Dhoroni: %d articles have no text file", missing)
    meta = meta.dropna(subset=["body"])
    out = pd.DataFrame(
        {
            "text": meta["Title"].fillna("") + "\n" + meta["body"],
            "orig_label": meta["Stance Detection"],
            "orig_split": "all",
            "agreement": float("nan"),
        }
    )
    # "Problematic" is not a stance; it marks articles the annotators flagged.
    out["label"] = out["orig_label"].map(
        {"Positive": "favour", "Negative": "against", "Neutral": "none"}
    )
    return out.assign(
        source="dhoroni",
        lang="bn",
        domain="news",
        target="climate change is real and pressing (close)",
        quality="gold",
    )


def build(raw: Path) -> pd.DataFrame:
    reddit_gold, reddit_silver = load_climate_reddit(raw)
    parts = [
        load_semeval(raw),
        load_gwsd(raw),
        reddit_gold,
        reddit_silver,
        load_case2024(raw),
        load_dhoroni(raw),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].map(clean)

    unmapped = df["label"].isna()
    log.info(
        "dropping %d rows with no target label: %s",
        unmapped.sum(),
        df.loc[unmapped, "orig_label"].value_counts().to_dict(),
    )
    df = df[~unmapped & (df["text"].str.len() > 0)].copy()

    df["uid"] = df["source"] + "-" + df.groupby("source").cumcount().astype(str)
    return df[COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/corpus_v1.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = build(args.raw_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    log.info("wrote %d rows to %s", len(df), args.out)
    log.info("\n%s", pd.crosstab(df["source"], df["label"], margins=True))


if __name__ == "__main__":
    main()
