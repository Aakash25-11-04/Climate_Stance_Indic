"""Count tables for the collected YouTube comments (run after data_prep/collect_youtube.py).

Reads data/processed/youtube_{videos,comments}_v1.csv and writes aggregate tables to
results/youtube/. The tables hold counts only, no comment text and no author information.

Usage:
    python data_prep/youtube_counts.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from stance.preprocess import dedup_key

log = logging.getLogger(__name__)

LENGTH_BINS = [0, 2, 5, 10, 20, 50, float("inf")]
LENGTH_LABELS = ["1-2", "3-5", "6-10", "11-20", "21-50", "51+"]


def by_language_type_script(comments: pd.DataFrame) -> pd.DataFrame:
    unique = comments[~comments["duplicate"]]
    return pd.crosstab(
        [unique["video_lang"], unique["query_type"]],
        unique["script"],
        margins=True,
        margins_name="all",
    )


def per_video(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    counts = comments.groupby("video_id").agg(
        comments=("comment_id", "size"),
        unique=("duplicate", lambda d: int((~d).sum())),
        replies=("is_reply", "sum"),
    )
    table = videos.set_index("video_id")[
        ["channel_title", "video_lang", "query_type", "comment_count", "comments_status"]
    ].join(counts, how="left")
    table[["comments", "unique", "replies"]] = table[["comments", "unique", "replies"]].fillna(0)
    return table.sort_values("unique", ascending=False)


def concentration(table: pd.DataFrame) -> pd.Series:
    collected = table.loc[table["comments"] > 0, "unique"]
    total = collected.sum()
    return pd.Series(
        {
            "videos with comments": len(collected),
            "channels with comments": table.loc[table["comments"] > 0, "channel_title"].nunique(),
            "unique comments per video, median": collected.median(),
            "unique comments per video, max": collected.max(),
            "share of unique comments from top 10 videos": collected.head(10).sum() / total,
        }
    )


def by_length(comments: pd.DataFrame) -> pd.DataFrame:
    unique = comments[~comments["duplicate"]]
    bins = pd.cut(unique["n_words"], LENGTH_BINS, labels=LENGTH_LABELS)
    return pd.crosstab(bins, unique["script"], margins=True, margins_name="all")


def repeated_texts(comments: pd.DataFrame) -> pd.Series:
    sizes = comments.groupby(comments["text"].map(dedup_key)).size()
    repeated = sizes[sizes > 1]
    return pd.Series(
        {
            "comments": len(comments),
            "flagged as repeated copies": int(comments["duplicate"].sum()),
            "texts that occur more than once": len(repeated),
            "largest group of identical texts": int(repeated.max()) if len(repeated) else 0,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/youtube"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    videos = pd.read_csv(args.data_dir / "youtube_videos_v1.csv")
    comments = pd.read_csv(args.data_dir / "youtube_comments_v1.csv", keep_default_na=False)
    comments["duplicate"] = comments["duplicate"].astype(str) == "True"
    comments["is_reply"] = comments["is_reply"].astype(str) == "True"
    comments["n_words"] = pd.to_numeric(comments["n_words"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_table = per_video(videos, comments)
    tables = {
        "by_language_type_script": by_language_type_script(comments),
        "per_video": videos_table,
        "video_concentration": concentration(videos_table).to_frame("value"),
        "by_length": by_length(comments),
        "repeated_texts": repeated_texts(comments).to_frame("value"),
    }
    for name, table in tables.items():
        table.to_csv(args.out_dir / f"{name}.csv", lineterminator="\n")
        shown = table.head(15) if name == "per_video" else table
        log.info("== %s ==\n%s\n", name, shown.to_string())


if __name__ == "__main__":
    main()
