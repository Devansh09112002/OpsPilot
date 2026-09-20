"""Uniform API error contract.

Every failure returns the same JSON envelope so the frontend can render an
actionable message instead of a stack trace or a blank screen:

    {"error": {"code": "...", "message": "...", "detail": {...}}}
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

log = get_logger(__name__)


class AppError(HTTPException):
    """Base class carrying a stable machine-readable code."""

    code = "internal_error"
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, detail: dict[str, Any] | None = None):
        super().__init__(status_code=self.status_code, detail=message)
        self.message = message
        self.extra = detail or {}


class NotFoundError(AppError):
    code = "not_found"
    status_code = status.HTTP_404_NOT_FOUND


class ValidationError(AppError):
    code = "invalid_request"
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class ForbiddenError(AppError):
    code = "forbidden"
    status_code = status.HTTP_403_FORBIDDEN


class ConflictError(AppError):
    code = "conflict"
    status_code = status.HTTP_409_CONFLICT


class RateLimitError(AppError):
    code = "rate_limited"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS


class ServiceUnavailableError(AppError):
    """A dependency is down. The rest of the product must keep working."""

    code = "service_unavailable"
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def _envelope(code: str, message: str, detail: dict | None = None) -> dict:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if detail:
        body["error"]["detail"] = detail
    return body


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.message, exc.extra),
    )


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code = {
        400: "invalid_request", 401: "unauthorized", 403: "forbidden",
        404: "not_found", 409: "conflict", 422: "invalid_request",
        429: "rate_limited", 503: "service_unavailable",
    }.get(exc.status_code, "error")
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(code, str(exc.detail)),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak an internal message or stack trace to the client."""
    log.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        exc_info=exc,
    )
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        # The id is the only thing that connects a visitor's report to the
        # log line that explains it. The message stays generic; the id is not
        # sensitive and is worth more than an apology.
        content=_envelope(
            "internal_error",
            "An unexpected error occurred.",
            {"request_id": request_id} if request_id else {},
        ),
        headers={"X-Request-ID": request_id} if request_id else None,
    )
