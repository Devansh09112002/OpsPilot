"""Server-issued guest sessions.

The session id is a 256-bit random token minted by the server and stored in an
HttpOnly cookie. The client never chooses it, so a visitor cannot adopt another
visitor's session by sending a guessed value: an unknown or expired token is
replaced with a fresh one rather than trusted.

Every mutable row (investigation, proposal, ticket, audit event) carries the
session id, and every read filters on it.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import GuestSession
from app.db.session import get_db

TOKEN_BYTES = 32


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _new_session(db: Session) -> GuestSession:
    settings = get_settings()
    now = _now()
    session = GuestSession(
        session_id=secrets.token_urlsafe(TOKEN_BYTES),
        created_at=now,
        last_seen_at=now,
        expires_at=now + timedelta(hours=settings.session_ttl_hours),
        investigations_today=0,
        quota_window_start=now,
    )
    db.add(session)
    db.flush()
    return session


def _set_cookie(response: Response, session: GuestSession) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.session_cookie_name,
        value=session.session_id,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        path="/",
    )


def get_guest_session(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> GuestSession:
    """Resolve, refresh or mint the caller's guest session.

    An unrecognised or expired cookie yields a *new* session rather than an
    error, so the demo always works, while a guessed identifier grants nothing.
    """
    settings = get_settings()
    token = request.cookies.get(settings.session_cookie_name)
    session: GuestSession | None = None

    if token:
        session = db.execute(
            select(GuestSession).where(GuestSession.session_id == token)
        ).scalar_one_or_none()
        if session is not None and session.expires_at <= _now():
            session = None

    if session is None:
        session = _new_session(db)
    else:
        session.last_seen_at = _now()

    db.commit()
    _set_cookie(response, session)
    return session
