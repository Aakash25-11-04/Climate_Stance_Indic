import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))

from collect_youtube import describe_videos, tag_comments, validate  # noqa: E402


def hit(video_id: str, lang: str, query_type: str, rank: int = 0) -> dict:
    return {
        "video_id": video_id,
        "query": f"{lang}-{query_type}",
        "query_lang": lang,
        "query_region": "IN",
        "query_type": query_type,
        "rank": rank,
    }


def meta(video_id: str, comment_count: int | None = 50, live: bool = False) -> dict:
    return {"video_id": video_id, "channel_id": "ch", "live": live, "comment_count": comment_count}


def test_validate_rejects_missing_or_unknown_query_type() -> None:
    validate({"search": {"queries": [{"q": "a", "query_type": "general"}]}})
    for bad in [{"q": "a"}, {"q": "a", "query_type": "skeptic"}]:
        with pytest.raises(ValueError):
            validate({"search": {"queries": [bad]}})


def test_describe_videos_derives_query_type_and_language() -> None:
    hits = pd.DataFrame(
        [
            hit("g", "hi", "general"),
            hit("s", "bn", "sceptic"),
            hit("b", "hi", "general"),
            hit("b", "hi", "sceptic", rank=3),
            hit("m", "hi", "general"),
            hit("m", "bn", "general"),
        ]
    )
    videos = describe_videos(hits, pd.DataFrame([meta(v) for v in "gsbm"]), min_comments=20)
    by_id = videos.set_index("video_id")
    assert by_id["query_type"].to_dict() == {
        "g": "general",
        "s": "sceptic",
        "b": "both",
        "m": "general",
    }
    assert by_id.loc["m", "video_lang"] == "mixed"
    assert by_id.loc["b", "best_rank"] == 0


def test_describe_videos_skips_live_and_comment_disabled_videos() -> None:
    hits = pd.DataFrame([hit(v, "hi", "general") for v in "abcd"])
    meta_rows = [meta("a"), meta("b", comment_count=None), meta("c", live=True), meta("d", 5)]
    videos = describe_videos(hits, pd.DataFrame(meta_rows), min_comments=20)
    assert videos.loc[videos["kept"], "video_id"].tolist() == ["a"]


def test_tag_comments_carries_query_type_into_every_row() -> None:
    videos = pd.DataFrame(
        [
            {"video_id": "v1", "channel_id": "c", "video_lang": "hi", "query_type": "sceptic"},
            {"video_id": "v2", "channel_id": "c", "video_lang": "bn", "query_type": "both"},
        ]
    )
    comments = pd.DataFrame(
        [
            {
                "comment_id": "1",
                "video_id": "v1",
                "parent_id": None,
                "text": "सब झूठ है",
                "published_at": "2025-01-01",
            },
            {
                "comment_id": "2",
                "video_id": "v2",
                "parent_id": "9",
                "text": "সব ঠিক আছে",
                "published_at": "2025-01-02",
            },
            {
                "comment_id": "3",
                "video_id": "v2",
                "parent_id": None,
                "text": "सब  झूठ है!",
                "published_at": "2025-01-03",
            },
        ]
    )
    tagged = tag_comments(comments, videos).set_index("comment_id")
    assert tagged["query_type"].to_dict() == {"1": "sceptic", "2": "both", "3": "both"}
    assert tagged["query_type"].notna().all()
    assert tagged["duplicate"].to_dict() == {"1": False, "2": False, "3": True}
    assert tagged.loc["2", "is_reply"] and tagged.loc["2", "script"] == "bengali"
