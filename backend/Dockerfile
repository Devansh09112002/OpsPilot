# OpsPilot backend.
#
# Multi-stage so the runtime image carries no build toolchain. The model
# artifact is baked in at build time: it is under 1 MB, and shipping it with
# the image means the container has no start-up dependency on object storage.
#
# Sized for a 512 MB free-tier container: one worker, single-threaded BLAS and
# OpenMP, and no dev dependencies.

# ---------- build ----------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

# libgomp is XGBoost's OpenMP runtime; it is the only native dependency needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 opspilot

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/backend:/app \
    # One thread each: the container has a fraction of a CPU, and extra
    # threads cost memory without buying throughput.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    ARTIFACT_DIR=/app/artifacts

COPY --from=build /opt/venv /opt/venv

WORKDIR /app

# Application code and the pipeline packages the backend imports.
COPY backend/ /app/backend/
COPY data_pipeline/ /app/data_pipeline/
COPY ml_pipeline/ /app/ml_pipeline/
COPY policies/ /app/policies/
COPY artifacts/ /app/artifacts/

RUN chown -R opspilot:opspilot /app
USER opspilot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8000}/api/v1/health || exit 1

# Render supplies $PORT. One worker keeps the memory ceiling predictable.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --app-dir /app/backend"]
