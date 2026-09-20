"""OpsPilot FastAPI application.

One process serves the API, loads the model artifact and runs the agent
workflow. There is no separate inference service and no agent microservice:
the plan calls for the simplest architecture that meets the requirements.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from app.api.v1.routes import router as v1_router
from app.core.config import get_settings
from app.core.errors import (
    AppError,
    app_error_handler,
    http_error_handler,
    unhandled_error_handler,
)
from app.core.logging import configure_logging, get_logger
from app.ml.predictor import predictor

settings = get_settings()
configure_logging()
log = get_logger("opspilot.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup.

    A failed load is logged and the predictor reports itself unavailable. The
    app still starts, so the order browser keeps working and the health
    endpoint can explain what is wrong.
    """
    predictor.load(settings.artifact_dir)
    if not predictor.available:
        log.warning("starting_without_model", error=predictor.error)
    if not settings.llm_configured:
        log.warning("starting_without_llm_key")
    yield


app = FastAPI(
    title="OpsPilot API",
    version="1.0.0",
    description=(
        "Delivery-risk and AI investigation workbench over the public Olist "
        "e-commerce dataset (CC BY-NC-SA 4.0). Orders are historical; model "
        "serving, investigations and ticketing execute live. All actions are "
        "simulated within the demo environment."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,          # required for the guest session cookie
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    max_age=600,
)


@app.middleware("http")
async def reject_control_characters(request: Request, call_next):
    """Refuse URLs carrying C0 control characters, NUL above all.

    A percent-encoded NUL reaches the application as a perfectly ordinary
    string, is bound into a query, and PostgreSQL then refuses it:
    "text fields cannot contain NUL (0x00) bytes". That surfaced as a 500 on
    seven endpoints, reachable by any anonymous visitor with a URL.

    Sanitising each query would mean remembering to do it in every new one.
    Rejecting at the edge is one rule in one place, and no control character
    is meaningful in an order id, a state code or a search prefix.
    """
    # `request.url.query` is still percent-encoded, so a `%00` there reads as
    # three harmless characters. The decoded values are what reach the
    # database, so those are what get checked.
    candidates = [request.url.path, *request.query_params.keys(), *request.query_params.values()]
    if any(ord(ch) < 0x20 for value in candidates for ch in value):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "bad_request",
                    "message": "The request URL contains control characters.",
                    "detail": {},
                }
            },
        )
    return await call_next(request)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and log one structured line per request."""
    import structlog

    request_id = uuid.uuid4().hex[:16]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    # Also on the request itself: route handlers read it from here, and the
    # error handlers put it in the response so a visitor can quote something
    # that appears in the logs. Without this every route logged request_id
    # as None.
    request.state.request_id = request_id

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.error(
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    if not request.url.path.endswith("/health"):
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
    return response


app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(HTTPException, http_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(v1_router, prefix=settings.api_v1_prefix)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "name": "OpsPilot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
    }
