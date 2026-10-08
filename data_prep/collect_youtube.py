"""Collect Hindi and Bengali YouTube comments about climate change.

Uses only the official YouTube Data API v3, within YouTube's Terms of Service; no scraping.
The API key (YT_API_KEY) and the author-hash salt (YT_AUTHOR_SALT) are read from .env and are
never printed, logged or written to disk. Outputs keep no usernames or commenter channel ids.

Stages: search -> video metadata -> comment threads. Every API response is cached under
data/raw/youtube/cache/, so a run stopped by the daily quota exits with status 75 and a rerun
after midnight Pacific time resumes from the cache without paying for the same calls again.

Usage:
    python data_prep/collect_youtube.py --dry-run   # worst-case quota cost, no API calls
    python data_prep/collect_youtube.py
    python data_prep/youtube_counts.py              # count tables for the report
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from stance.preprocess import dedup_key, dominant_script
from stance.youtube import (
    COST,
    ApiError,
    QuotaExhausted,
    QuotaLedger,
    YouTubeClient,
    comment_rows,
)

log = logging.getLogger(__name__)

EX_TEMPFAIL = 75
RAW = Path("data/raw/youtube")
QUERY_TYPES = ("general", "sceptic")


def worst_case_cost(cfg: dict) -> dict[str, int]:
    search_pages = len(cfg["search"]["queries"]) * cfg["search"]["pages_per_query"]
    videos = search_pages * 50
    return {
        "search": search_pages * COST["search"],
        "videos": math.ceil(videos / 50) * COST["videos"],
        "commentThreads": videos * cfg["comments"]["pages_per_video"] * COST["commentThreads"],
    }


def search_videos(client: YouTubeClient, cfg: dict) -> pd.DataFrame:
    rows = []
    for query in cfg["search"]["queries"]:
        params = {
            "part": "snippet",
            "type": "video",
            "maxResults": 50,
            "q": query["q"],
            "relevanceLanguage": query["lang"],
            "regionCode": query["region"],
            "order": cfg["search"]["order"],
            "publishedAfter": cfg["search"]["published_after"],
        }
        rank = 0
        for page in client.pages("search", cfg["search"]["pages_per_query"], **params):
            for item in page.get("items", []):
                rows.append(
                    {
                        "video_id": item["id"]["videoId"],
                        "query": query["q"],
                        "query_lang": query["lang"],
                        "query_region": query["region"],
                        "query_type": query["query_type"],
                        "rank": rank,
                    }
                )
                rank += 1
    return pd.DataFrame(rows)


def video_metadata(client: YouTubeClient, video_ids: list[str]) -> pd.DataFrame:
    rows = []
    for start in range(0, len(video_ids), 50):
        batch = video_ids[start : start + 50]
        page = client.get("videos", part="snippet,statistics", id=",".join(batch), maxResults=50)
        for item in page.get("items", []):
            snippet, stats = item["snippet"], item.get("statistics", {})
            rows.append(
                {
                    "video_id": item["id"],
                    "channel_id": snippet["channelId"],
                    "channel_title": snippet["channelTitle"],
                    "title": snippet["title"],
                    "published_at": snippet["publishedAt"],
                    "default_language": snippet.get("defaultLanguage"),
                    "default_audio_language": snippet.get("defaultAudioLanguage"),
                    "live": snippet.get("liveBroadcastContent", "none") != "none",
                    "view_count": int(stats.get("viewCount", 0)),
                    # Absent when the owner has disabled comments.
                    "comment_count": (
                        int(stats["commentCount"]) if "commentCount" in stats else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def validate(cfg: dict) -> None:
    for query in cfg["search"]["queries"]:
        if query.get("query_type") not in QUERY_TYPES:
            raise ValueError(f"query {query['q']!r}: query_type must be one of {QUERY_TYPES}")


def _single_or(values: pd.Series, label: str) -> str:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else label


def describe_videos(hits: pd.DataFrame, meta: pd.DataFrame, min_comments: int) -> pd.DataFrame:
    found = hits.groupby("video_id").agg(
        queries=("query", lambda s: " | ".join(sorted(set(s)))),
        # A video returned by several queries keeps every fact: "mixed" language,
        # "both" query types. The sampler's pool rules are written against these values.
        video_lang=("query_lang", lambda s: _single_or(s, "mixed")),
        query_type=("query_type", lambda s: _single_or(s, "both")),
        best_rank=("rank", "min"),
    )
    videos = meta.merge(found.reset_index(), on="video_id", how="inner")
    videos["kept"] = ~videos["live"] & (videos["comment_count"].fillna(0) >= min_comments)
    # Fixed processing order, so a resumed run asks for the same pages as the stopped one.
    return videos.sort_values(["best_rank", "video_id"]).reset_index(drop=True)


def collect_comments(
    client: YouTubeClient,
    videos: pd.DataFrame,
    cfg: dict,
    salt: str,
    rows: list[dict],
    status: dict[str, str],
) -> None:
    """Append to `rows` and `status` in place, so a quota stop keeps what was collected."""
    for video_id in videos.loc[videos["kept"], "video_id"]:
        params = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": 100,
            "order": cfg["comments"]["order"],
            "textFormat": "plainText",
        }
        try:
            for page in client.pages(
                "commentThreads", cfg["comments"]["pages_per_video"], **params
            ):
                rows.extend(comment_rows(page, salt))
            status[video_id] = "ok"
        except ApiError as error:
            log.info("video %s skipped: %s", video_id, error.reason)
            status[video_id] = error.reason


def tag_comments(comments: pd.DataFrame, videos: pd.DataFrame) -> pd.DataFrame:
    comments = comments.merge(
        videos[["video_id", "channel_id", "video_lang", "query_type"]], on="video_id", how="left"
    )
    comments["is_reply"] = comments["parent_id"].notna()
    comments["script"] = comments["text"].map(dominant_script)
    comments["n_words"] = comments["text"].str.split().str.len()
    # Copies of the same text (copypasta, repeated slogans) are flagged, not dropped, so the
    # sampler can decide; the first copy in a fixed order is the one kept.
    comments = comments.sort_values(["video_id", "published_at", "comment_id"])
    comments["duplicate"] = comments["text"].map(dedup_key).duplicated()
    return comments.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/youtube.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dry-run", action="store_true", help="print quota cost, call nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate(cfg)

    cost = worst_case_cost(cfg)
    log.info(
        "worst-case cost: %s = %d units (budget %d/day)",
        cost,
        sum(cost.values()),
        cfg["budget_per_day"],
    )
    if args.dry_run:
        return 0

    load_dotenv()
    api_key, salt = os.environ.get("YT_API_KEY"), os.environ.get("YT_AUTHOR_SALT")
    missing = [
        name for name, value in [("YT_API_KEY", api_key), ("YT_AUTHOR_SALT", salt)] if not value
    ]
    if missing:
        log.error("%s not set; add to .env (see .env.example)", ", ".join(missing))
        return 1

    ledger = QuotaLedger(RAW / "quota.json", cfg["budget_per_day"])
    client = YouTubeClient(api_key, RAW / "cache", ledger)
    log.info("quota already used today: %d units", ledger.used)

    videos = None
    comments: list[dict] = []
    status: dict[str, str] = {}
    complete = True
    try:
        hits = search_videos(client, cfg)
        meta = video_metadata(client, sorted(hits["video_id"].unique()))
        videos = describe_videos(hits, meta, cfg["videos"]["min_comments"])
        collect_comments(client, videos, cfg, salt, comments, status)
    except QuotaExhausted as error:
        complete = False
        log.warning("stopping: %s. Rerun after midnight Pacific time to resume from cache.", error)
    finally:
        log.info("API calls: %s; quota used today: %d units", client.calls, ledger.used)

    if videos is None:
        return EX_TEMPFAIL
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos["comments_status"] = videos["video_id"].map(status)
    videos.to_csv(args.out_dir / "youtube_videos_v1.csv", index=False, lineterminator="\n")
    if comments:
        tagged = tag_comments(pd.DataFrame(comments), videos)
        tagged.to_csv(args.out_dir / "youtube_comments_v1.csv", index=False, lineterminator="\n")
        log.info(
            "%d videos found, %d kept, %d collected; %d comments, %d repeated texts flagged",
            len(videos),
            int(videos["kept"].sum()),
            sum(s == "ok" for s in status.values()),
            len(tagged),
            int(tagged["duplicate"].sum()),
        )
    if not complete:
        log.warning("outputs are PARTIAL")
        return EX_TEMPFAIL
    return 0


if __name__ == "__main__":
    sys.exit(main())
