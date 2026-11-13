"""YouTube Data API v3 access with an on-disk response cache and a daily quota ledger.

The API key is sent in the X-Goog-Api-Key header, so it never appears in a URL, a cache
file, a log line or an exception message. Every response is cached before it is used, so a
run stopped by the daily quota resumes the next day without paying for the same calls twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

API = "https://www.googleapis.com/youtube/v3"
# Units per request (developers.google.com/youtube/v3/determine_quota_cost).
COST = {"search": 100, "videos": 1, "commentThreads": 1}
# The daily quota resets at midnight Pacific time.
QUOTA_TZ = ZoneInfo("America/Los_Angeles")
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
# Stable answers about a video that are worth caching so a rerun does not pay for them again.
CACHEABLE_REASONS = {"commentsDisabled", "videoNotFound", "forbidden"}
RETRY_STATUS = {429, 500, 502, 503, 504}

# (url, headers, timeout) -> (status, body)
Transport = Callable[[str, dict[str, str], float], tuple[int, bytes]]


class QuotaExhausted(RuntimeError):
    """The daily budget is spent, either by our own ledger or by the API's answer."""


class ApiError(RuntimeError):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"HTTP {status}: {reason}")
        self.status = status
        self.reason = reason


def quota_day(now: datetime | None = None) -> str:
    return (now or datetime.now(QUOTA_TZ)).astimezone(QUOTA_TZ).date().isoformat()


class QuotaLedger:
    """Units spent per Pacific-time day, persisted so that separate runs share one budget."""

    def __init__(self, path: Path, budget: int, clock: Callable[[], str] = quota_day) -> None:
        self.path = path
        self.budget = budget
        self.clock = clock
        self.days: dict[str, int] = json.loads(path.read_text()) if path.exists() else {}

    @property
    def used(self) -> int:
        return self.days.get(self.clock(), 0)

    def charge(self, endpoint: str) -> None:
        cost = COST[endpoint]
        if self.used + cost > self.budget:
            raise QuotaExhausted(f"budget of {self.budget} units for {self.clock()} reached")
        self.days[self.clock()] = self.used + cost
        self._save()

    def exhaust(self) -> None:
        """Record that the API refused us, so later runs on the same day stop immediately."""
        self.days[self.clock()] = max(self.used, self.budget)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.days, indent=1, sort_keys=True))


def _urllib_transport(url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _error_reason(body: bytes) -> str:
    try:
        error = json.loads(body)["error"]
        return error.get("errors", [{}])[0].get("reason") or error.get("status", "unknown")
    except (ValueError, KeyError, IndexError, TypeError):
        return "unparseable error body"


class YouTubeClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        ledger: QuotaLedger,
        transport: Transport = _urllib_transport,
        retries: int = 4,
        timeout: float = 30.0,
    ) -> None:
        self._headers = {"X-Goog-Api-Key": api_key, "Accept": "application/json"}
        self.cache_dir = cache_dir
        self.ledger = ledger
        self.transport = transport
        self.retries = retries
        self.timeout = timeout
        self.calls = {"cached": 0, "fetched": 0}

    def cache_path(self, endpoint: str, params: dict[str, object]) -> Path:
        request = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
        return self.cache_dir / endpoint / f"{hashlib.sha1(request.encode()).hexdigest()}.json"

    def get(self, endpoint: str, **params: object) -> dict:
        path = self.cache_path(endpoint, params)
        if path.exists():
            self.calls["cached"] += 1
            cached = json.loads(path.read_text(encoding="utf-8"))
            if "error" in cached:
                raise ApiError(cached["error"]["status"], cached["error"]["reason"])
            return cached["response"]

        status, body = self._fetch(endpoint, params)
        self.calls["fetched"] += 1
        if status == 200:
            response = json.loads(body)
            self._store(path, endpoint, params, {"response": response})
            return response
        reason = _error_reason(body)
        if reason in QUOTA_REASONS:
            self.ledger.exhaust()
            raise QuotaExhausted(f"API refused the request: {reason}")
        if reason in CACHEABLE_REASONS:
            self._store(path, endpoint, params, {"error": {"status": status, "reason": reason}})
        raise ApiError(status, reason)

    def _fetch(self, endpoint: str, params: dict[str, object]) -> tuple[int, bytes]:
        url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
        for attempt in range(self.retries + 1):
            self.ledger.charge(endpoint)
            log.debug("GET %s (%d units used today)", endpoint, self.ledger.used)
            try:
                status, body = self.transport(url, self._headers, self.timeout)
            except OSError as error:
                # Log only the exception type: network errors can carry request details.
                status, body = 0, b""
                log.warning("%s: network error %s", endpoint, type(error).__name__)
            # rateLimitExceeded is a short-term limit; quotaExceeded is final for the day.
            transient = (
                status in RETRY_STATUS
                or status == 0
                or (status == 403 and _error_reason(body) == "rateLimitExceeded")
            )
            if not transient or attempt == self.retries:
                return status, body
            wait = 2**attempt
            log.warning("%s: HTTP %d, retrying in %ds", endpoint, status, wait)
            time.sleep(wait)
        raise AssertionError("unreachable")

    def _store(self, path: Path, endpoint: str, params: dict[str, object], data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"endpoint": endpoint, "params": params, **data}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def pages(self, endpoint: str, max_pages: int, **params: object) -> Iterator[dict]:
        token = None
        for _ in range(max_pages):
            page = self.get(endpoint, **params, **({"pageToken": token} if token else {}))
            yield page
            token = page.get("nextPageToken")
            if not token:
                return


def author_hash(channel_id: str | None, salt: str) -> str | None:
    """Pseudonymous author id: lets us cap comments per author without storing who they are."""
    if not channel_id:
        return None
    return hashlib.sha256(f"{salt}:{channel_id}".encode()).hexdigest()[:16]


def comment_rows(page: dict, salt: str) -> Iterator[dict]:
    """Flatten a commentThreads page into one row per top-level comment or inline reply."""
    for thread in page.get("items", []):
        top = thread["snippet"]["topLevelComment"]
        yield _comment_row(top, None, salt)
        for reply in thread.get("replies", {}).get("comments", []):
            yield _comment_row(reply, top["id"], salt)


def _comment_row(comment: dict, parent_id: str | None, salt: str) -> dict:
    s = comment["snippet"]
    return {
        "comment_id": comment["id"],
        "video_id": s.get("videoId"),
        "parent_id": parent_id,
        "text": s.get("textDisplay", ""),
        "like_count": s.get("likeCount", 0),
        "published_at": s.get("publishedAt"),
        "edited": s.get("updatedAt") != s.get("publishedAt"),
        "author_hash": author_hash(s.get("authorChannelId", {}).get("value"), salt),
    }
