#!/usr/bin/env python
"""Optional: collect public YouTube comments from Hindi/Bengali climate videos (YouTube Data API v3).

    python scripts/collect_youtube.py --config configs/youtube.yaml

Privacy: usernames are replaced by a salted SHA-256 hash, no channel ids / profile data are
stored, and only comment text, language guess, video id and a timestamp bucket are kept.
Needs `YOUTUBE_API_KEY` in `.env` (git-ignored). Uses the REST API via `requests` only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stance.preprocessing.normalize import detect_script  # noqa: E402
from stance.utils import abs_path, load_config  # noqa: E402

API = "https://www.googleapis.com/youtube/v3"


def _get(path: str, params: dict, key: str) -> dict:
    params = dict(params, key=key)
    for attempt in range(3):
        r = requests.get(f"{API}/{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429):
            time.sleep(2 ** attempt)
        else:
            r.raise_for_status()
    r.raise_for_status()
    return {}


def search_videos(query: str, n: int, key: str, lang: str) -> list[str]:
    d = _get("search", {"part": "id", "q": query, "type": "video", "maxResults": n,
                        "relevanceLanguage": lang, "safeSearch": "moderate"}, key)
    return [it["id"]["videoId"] for it in d.get("items", []) if "videoId" in it.get("id", {})]


def comments(video_id: str, n: int, key: str) -> list[dict]:
    out, token = [], None
    while len(out) < n:
        params = {"part": "snippet", "videoId": video_id, "maxResults": min(100, n - len(out)),
                  "textFormat": "plainText", "order": "relevance"}
        if token:
            params["pageToken"] = token
        try:
            d = _get("commentThreads", params, key)
        except requests.HTTPError:
            break  # comments disabled
        for it in d.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            out.append({"text": s.get("textDisplay", ""), "author": s.get("authorDisplayName", ""),
                        "published": s.get("publishedAt", "")[:7], "video_id": video_id})
        token = d.get("nextPageToken")
        if not token:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/youtube.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise SystemExit("Set YOUTUBE_API_KEY in .env (see MANUAL_TASKS.md, section D)")
    y = cfg["youtube"]
    salt = hashlib.sha256(key.encode()).hexdigest()[:16]
    rows, seen = [], set()
    for lang, queries in y["queries"].items():
        for q in queries:
            for vid in search_videos(q, y["max_videos_per_query"], key, lang):
                if vid in seen:
                    continue
                seen.add(vid)
                for c in comments(vid, y["max_comments_per_video"], key):
                    t = " ".join(c["text"].split())
                    if not (y["min_chars"] <= len(t) <= y["max_chars"]):
                        continue
                    scr = detect_script(t)
                    guess = "bn" if scr == "bengali" else "hi" if scr == "devanagari" else lang
                    rows.append({"id": hashlib.sha1(f"{vid}{t}".encode()).hexdigest()[:12], "text": t,
                                 "language_guess": guess, "script": scr, "video_id": vid, "month": c["published"],
                                 "author_hash": hashlib.sha256((salt + c["author"]).encode()).hexdigest()[:16],
                                 "query": q})
                time.sleep(0.2)
    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    out_dir = abs_path(y["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comments_unlabelled.csv", index=False, encoding="utf-8")
    rng = random.Random(cfg["seed"])
    idx = list(range(len(df))); rng.shuffle(idx)
    sample = df.iloc[idx[: y["sample_for_labelling"]]][["id", "text", "language_guess"]].rename(columns={"language_guess": "language"})
    sample["label"] = ""
    sample["annotator_note"] = ""
    tpl = abs_path(y["template"]); tpl.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(tpl, index=False, encoding="utf-8")
    print(f"collected {len(df)} comments from {len(seen)} videos -> {out_dir / 'comments_unlabelled.csv'}")
    print(f"labelling template ({len(sample)} rows) -> {tpl}; follow docs/annotation_guideline.md")


if __name__ == "__main__":
    main()
