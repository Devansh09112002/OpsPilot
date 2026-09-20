"""Model serving inside the FastAPI process.

There is no separate inference service. The artifact is loaded once at startup
through the same `ml_pipeline.model.load_artifact` the training pipeline wrote
it with, so training and serving cannot drift apart silently.

Two numbers come out of a prediction and they mean different things:

* `ranking_score` is the model's raw output. It has fine resolution and is the
  key the queue sorts by, so every published ranking metric stays valid.
* `probability` is that score passed through the isotonic calibrator fitted on
  the validation split. It is the number a person should read. Before
  calibration a raw 0.65 corresponded to a 2.4% observed late rate, so showing
  the raw value to a human was misleading even with a disclaimer.

If the artifact is missing or corrupt the predictor reports itself unavailable
and the API answers 503. It never invents a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import pandas as pd

from app.core.logging import get_logger
from data_pipeline.features import MODEL_FEATURES, NUMERIC_FEATURES
from ml_pipeline.model import (
    ArtifactError,
    band_for,
    calibrate,
    explain,
    load_artifact,
)

log = get_logger(__name__)

# Fallback bands, used only for an uncalibrated artifact. A calibrated one
# carries thresholds derived from its own validation distribution.
DEFAULT_BANDS = {"high": 0.60, "medium": 0.30}


@dataclass
class Prediction:
    """One scored order, with both numbers and why."""

    probability: float
    ranking_score: float
    band: str
    factors: list[dict]
    calibrated: bool


class Predictor:
    """Thread-safe holder for the loaded artifact."""

    def __init__(self) -> None:
        self._model = None
        self._meta: dict | None = None
        self._calibrator = None
        self._bands: dict[str, float] = dict(DEFAULT_BANDS)
        self._error: str | None = None
        self._lock = Lock()

    def load(self, artifact_dir: Path) -> None:
        with self._lock:
            try:
                self._model, self._meta = load_artifact(artifact_dir)
                self._calibrator = self._meta.get("_calibrator")
                self._bands = dict(
                    self._meta.get("band_thresholds") or DEFAULT_BANDS
                )
                self._error = None
                log.info(
                    "model_artifact_loaded",
                    model_version=self._meta["model_version"],
                    features=len(self._meta["feature_names"]),
                    calibrated=bool(self._calibrator),
                    bands=self._bands,
                )
            except Exception as exc:
                self._model, self._meta, self._calibrator = None, None, None
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
    def calibrated(self) -> bool:
        return self._calibrator is not None

    @property
    def band_thresholds(self) -> dict[str, float]:
        return dict(self._bands)

    @property
    def metadata(self) -> dict | None:
        return self._meta

    def _frame(self, features: dict) -> pd.DataFrame:
        if not features:
            raise ArtifactError(
                "no stored feature document for this order, so it cannot be "
                "scored. Only orders belonging to a demo snapshot are scorable."
            )
        missing = [f for f in MODEL_FEATURES if f not in features]
        if missing:
            raise ArtifactError(f"stored feature document is missing {missing}")

        frame = pd.DataFrame([{f: features[f] for f in MODEL_FEATURES}])
        # Object columns arrive from JSONB; restore numeric dtypes so the fitted
        # imputer and scaler see exactly what they saw during training.
        for col in NUMERIC_FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    def predict_one(self, features: dict, *, with_factors: bool = False) -> Prediction:
        """Score one order from its stored point-in-time feature document."""
        if self._model is None:
            raise ArtifactError(self._error or "model artifact is not loaded")

        frame = self._frame(features)
        raw = float(self._model.predict_proba(frame[MODEL_FEATURES])[:, 1][0])
        probability = float(calibrate(self._calibrator, [raw])[0])

        factors: list[dict] = []
        if with_factors:
            try:
                factors = explain(self._model, frame)
            except Exception as exc:
                log.warning("explanation_failed", error=type(exc).__name__)

        return Prediction(
            probability=probability,
            ranking_score=raw,
            band=band_for(probability, self._bands),
            factors=factors,
            calibrated=self.calibrated,
        )


predictor = Predictor()


def risk_band(probability: float, thresholds: dict[str, float] | None = None) -> str:
    return band_for(probability, thresholds or predictor.band_thresholds)


def prediction_computed_at() -> datetime:
    return datetime.now(UTC)
