import json
from pathlib import Path

import pytest

from stance.youtube import (
    ApiError,
    QuotaExhausted,
    QuotaLedger,
    YouTubeClient,
    author_hash,
    comment_rows,
)

KEY = "AIza-secret-test-key"


def error_body(reason: str, code: int = 403) -> bytes:
    return json.dumps({"error": {"code": code, "errors": [{"reason": reason}]}}).encode()


class FakeTransport:
    def __init__(self, *responses: tuple[int, bytes]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
        self.requests.append((url, headers))
        return self.responses.pop(0)


class Day:
    def __init__(self, day: str = "2026-10-08") -> None:
        self.day = day

    def __call__(self) -> str:
        return self.day


def make_client(tmp_path: Path, transport: FakeTransport, budget: int = 1000, day=None):
    ledger = QuotaLedger(tmp_path / "quota.json", budget, clock=day or Day())
    client = YouTubeClient(KEY, tmp_path / "cache", ledger, transport=transport, retries=2)
    return client, ledger


def ok(payload: dict) -> tuple[int, bytes]:
    return 200, json.dumps(payload).encode()


def test_key_travels_in_header_only(tmp_path: Path) -> None:
    transport = FakeTransport(ok({"items": []}))
    client, _ = make_client(tmp_path, transport)
    client.get("videos", id="abc", part="snippet")
    url, headers = transport.requests[0]
    assert KEY not in url
    assert headers["X-Goog-Api-Key"] == KEY
    cached = "".join(p.read_text() for p in (tmp_path / "cache").rglob("*.json"))
    assert KEY not in cached


def test_cache_hit_costs_nothing(tmp_path: Path) -> None:
    transport = FakeTransport(ok({"items": [1]}))
    client, ledger = make_client(tmp_path, transport)
    first = client.get("search", q="x")
    second = client.get("search", q="x")
    assert first == second
    assert len(transport.requests) == 1
    assert ledger.used == 100


def test_ledger_stops_before_exceeding_budget(tmp_path: Path) -> None:
    transport = FakeTransport(ok({}), ok({}))
    client, ledger = make_client(tmp_path, transport, budget=150)
    client.get("search", q="a")
    with pytest.raises(QuotaExhausted):
        client.get("search", q="b")
    assert len(transport.requests) == 1
    assert ledger.used == 100


def test_api_quota_error_exhausts_the_day_and_resumes_next_day(tmp_path: Path) -> None:
    day = Day("2026-10-08")
    transport = FakeTransport(ok({"n": 1}), (403, error_body("quotaExceeded")), ok({"n": 2}))
    client, ledger = make_client(tmp_path, transport, day=day)
    client.get("videos", id="a")
    with pytest.raises(QuotaExhausted):
        client.get("videos", id="b")

    # A new run on the same day stops without calling the API.
    client, ledger = make_client(tmp_path, transport, day=day)
    assert client.get("videos", id="a") == {"n": 1}
    with pytest.raises(QuotaExhausted):
        client.get("videos", id="b")
    assert len(transport.requests) == 2

    # The next day the ledger starts from zero and only the missing call is made.
    day.day = "2026-10-09"
    assert client.get("videos", id="a") == {"n": 1}
    assert client.get("videos", id="b") == {"n": 2}
    assert len(transport.requests) == 3
    assert ledger.used == 1


def test_comments_disabled_is_cached(tmp_path: Path) -> None:
    transport = FakeTransport((403, error_body("commentsDisabled")))
    client, ledger = make_client(tmp_path, transport)
    for _ in range(2):
        with pytest.raises(ApiError) as error:
            client.get("commentThreads", videoId="v")
        assert error.value.reason == "commentsDisabled"
    assert len(transport.requests) == 1


def test_other_errors_are_not_cached(tmp_path: Path) -> None:
    transport = FakeTransport((400, error_body("badRequest", 400)), ok({"fixed": True}))
    client, _ = make_client(tmp_path, transport)
    with pytest.raises(ApiError):
        client.get("videos", id="v")
    assert client.get("videos", id="v") == {"fixed": True}


def test_transient_errors_are_retried_and_charged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("stance.youtube.time.sleep", lambda s: None)
    transport = FakeTransport((503, b""), (403, error_body("rateLimitExceeded")), ok({"ok": 1}))
    client, ledger = make_client(tmp_path, transport)
    assert client.get("videos", id="v") == {"ok": 1}
    assert ledger.used == 3


def test_pages_follow_tokens(tmp_path: Path) -> None:
    transport = FakeTransport(ok({"nextPageToken": "t1"}), ok({"nextPageToken": "t2"}), ok({}))
    client, _ = make_client(tmp_path, transport)
    assert len(list(client.pages("commentThreads", max_pages=5, videoId="v"))) == 3
    assert "pageToken=t1" in transport.requests[1][0]


def test_pages_stop_at_max_pages(tmp_path: Path) -> None:
    transport = FakeTransport(ok({"nextPageToken": "t1"}), ok({"nextPageToken": "t2"}))
    client, _ = make_client(tmp_path, transport)
    assert len(list(client.pages("commentThreads", max_pages=2, videoId="v"))) == 2
    assert len(transport.requests) == 2


def test_comment_rows_flatten_replies_and_hide_authors() -> None:
    def comment(cid: str, author: str, text: str) -> dict:
        return {
            "id": cid,
            "snippet": {
                "videoId": "v1",
                "textDisplay": text,
                "authorDisplayName": "Real Name",
                "authorChannelId": {"value": author},
                "likeCount": 2,
                "publishedAt": "2025-01-01T00:00:00Z",
                "updatedAt": "2025-01-01T00:00:00Z",
            },
        }

    page = {
        "items": [
            {
                "snippet": {"topLevelComment": comment("c1", "UC1", "जलवायु परिवर्तन सच है")},
                "replies": {"comments": [comment("c1.r1", "UC2", "নিশ্চয়ই")]},
            }
        ]
    }
    rows = list(comment_rows(page, salt="s"))
    assert [r["comment_id"] for r in rows] == ["c1", "c1.r1"]
    assert rows[1]["parent_id"] == "c1"
    assert rows[0]["author_hash"] == author_hash("UC1", "s") != author_hash("UC1", "t")
    assert "Real Name" not in json.dumps(rows) and "UC1" not in json.dumps(rows)
