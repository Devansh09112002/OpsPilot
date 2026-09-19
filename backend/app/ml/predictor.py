"""Model serving inside the FastAPI process.

There is no separate inference service. The artifact is loaded once at startup
through the same `ml_pipeline.model.load_artifact` the training pipeline wrote
it with, so training and serving cannot drift apart silently.

If the artifact is missing or corrupt the predictor reports itself unavailable
and the API answers 503 for predictions. It never invents a score.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd

from app.core.logging import get_logger
from data_pipeline.features import MODEL_FEATURES
from ml_pipeline.model import ArtifactError, load_artifact, predict_risk

log = get_logger(__name__)

HIGH_BAND = 0.60
MEDIUM_BAND = 0.30


def risk_band(p: float) -> str:
    if p >= HIGH_BAND:
        return "high"
    if p >= MEDIUM_BAND:
        return "medium"
    return "low"


class Predictor:
    """Thread-safe holder for the loaded artifact."""

    def __init__(self) -> None:
        self._model = None
        self._meta: dict | None = None
        self._error: str | None = None
        self._lock = Lock()

    def load(self, artifact_dir: Path) -> None:
        with self._lock:
            try:
                self._model, self._meta = load_artifact(artifact_dir)
                self._error = None
                log.info(
                    "model_artifact_loaded",
                    model_version=self._meta["model_version"],
                    features=len(self._meta["feature_names"]),
                )
            except (ArtifactError, Exception) as exc:  # noqa: BLE001
                self._model, self._meta = None, None
                self._error = str(exc)
                log.error("model_artifact_load_failed", error=self._error)

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def model_version(self) -> str | None:
        return self._meta["model_version"] if self._meta else None

    @property
    def metadata(self) -> dict | None:
        return self._meta

    def predict_one(self, features: dict) -> float:
        """Score a single order from its stored point-in-time feature document."""
        if self._model is None:
            raise ArtifactError(self._error or "model artifact is not loaded")

        missing = [f for f in MODEL_FEATURES if f not in features]
        if missing:
            raise ArtifactError(f"stored feature document is missing {missing}")

        frame = pd.DataFrame([{f: features[f] for f in MODEL_FEATURES}])
        # Object columns arrive from JSONB; restore numeric dtypes so the fitted
        # imputer and scaler see exactly what they saw during training.
        from data_pipeline.features import NUMERIC_FEATURES

        for col in NUMERIC_FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        return float(predict_risk(self._model, frame)[0])


predictor = Predictor()


def prediction_computed_at() -> datetime:
    return datetime.now(timezone.utc)
