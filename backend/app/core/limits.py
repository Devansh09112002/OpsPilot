"""Rate and budget guards.

Two separate concerns:

* A cheap in-process token bucket caps request rate per client, protecting the
  API from accidental hammering.
* A database-backed per-session daily quota and a global hourly cap bound the
  *paid* LLM spend. That counter must survive a restart, so it lives in
  PostgreSQL rather than memory.

Exceeding a limit returns a clear 429, never a silent extra API charge.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import Lock

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import RateLimitError
from app.db.models import GuestSession, Investigation


class SlidingWindowLimiter:
    """Per-key sliding window, in-process. Adequate for a single backend."""

    def __init__(self, max_events: int, window_seconds: float):
        self.max_events = max_events
        self.window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            q = self._events[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max_events:
                return False, self.window - (now - q[0])
            q.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


_settings = get_settings()
request_limiter = SlidingWindowLimiter(_settings.api_requests_per_minute, 60.0)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def enforce_investigation_budget(db: Session, session: GuestSession) -> None:
    """Bound LLM spend before an investigation is started.

    Raises `RateLimitError` with an actionable message. The caller has not yet
    spent anything when this fires.
    """
    settings = get_settings()
    now = _now()

    # Per-session rolling 24h quota.
    if now - session.quota_window_start >= timedelta(hours=24):
        session.quota_window_start = now
        session.investigations_today = 0

    if session.investigations_today >= settings.investigations_per_session_per_day:
        retry_at = session.quota_window_start + timedelta(hours=24)
        raise RateLimitError(
            f"Demo limit reached: {settings.investigations_per_session_per_day} "
            "AI investigations per visitor per day. The risk queue and model "
            "predictions remain fully available.",
            {"retry_after_utc": retry_at.isoformat(), "limit_type": "per_session_daily"},
        )

    # Global hourly cap, protecting the project's overall API budget.
    since = now - timedelta(hours=1)
    global_count = db.execute(
        select(func.count(Investigation.investigation_id)).where(
            Investigation.created_at >= since
        )
    ).scalar_one()
    if global_count >= settings.investigations_global_per_hour:
        raise RateLimitError(
            "OpsPilot has reached its hourly AI usage cap across all visitors. "
            "Please try again shortly; the risk queue and predictions still work.",
            {"limit_type": "global_hourly"},
        )


def record_investigation_spend(session: GuestSession) -> None:
    """Count an investigation against the session quota once it has started."""
    session.investigations_today += 1
