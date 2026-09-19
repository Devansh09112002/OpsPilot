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
async def request_context(request: Request, call_next):
    """Attach a request id and log one structured line per request."""
    import structlog

    request_id = uuid.uuid4().hex[:16]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

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
