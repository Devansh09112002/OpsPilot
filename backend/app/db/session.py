"""Database engine and request-scoped session."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_settings = get_settings()


def _is_pooled(url: str) -> bool:
    """True when the DSN points at a connection pooler rather than PostgreSQL.

    Supabase fronts its free databases with PgBouncer on port 6543.
    """
    return "pooler.supabase.com" in url or ":6543" in url


def _engine_kwargs(url: str) -> dict[str, Any]:
    """Engine settings, adjusted for a pooled connection.

    Two things break against a transaction-mode pooler and both fail in a way
    that looks like a random application bug:

    * psycopg3 prepares statements by default. A transaction-mode pooler hands
      the next transaction to a different backend, which has never seen that
      prepared statement, so queries fail intermittently with
      "prepared statement ... does not exist" once traffic picks up.
      `prepare_threshold=None` disables preparation entirely.
    * SQLAlchemy's own pool on top of PgBouncer holds connections a free-tier
      project cannot spare, so it is kept small and connections are recycled
      before the pooler times them out itself.
    """
    kwargs: dict[str, Any] = {
        "pool_pre_ping": True,
        "future": True,
    }
    if _is_pooled(url):
        kwargs |= {
            "connect_args": {"prepare_threshold": None},
            "pool_size": 2,
            "max_overflow": 3,
            "pool_recycle": 280,
        }
    else:
        kwargs |= {
            "pool_size": _settings.db_pool_size,
            "max_overflow": _settings.db_max_overflow,
        }
    return kwargs


engine = create_engine(_settings.database_url, **_engine_kwargs(_settings.database_url))

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
