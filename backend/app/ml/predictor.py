"""Model serving inside the FastAPI process: the v4 snapshot-day model.

There is no separate inference service. The artifacts in `artifacts/v4/` are
the exact models whose held-out performance is published in
`docs/research_v4.md` (`ml_pipeline.train_v4` refuses to write them unless
they reproduce that figure). An order is scored **as of the start of its
snapshot day**, from what was known then, by three components:

* the **Kaplan-Meier** chance that a parcel on its route, undelivered at its
  age, misses its promise (a stored point-in-time feature);
* a **discrete-time hazard model** that gives the chance of missing the
  promise and the whole remaining arrival distribution;
* a **LambdaMART** ranker trained to put late orders in the top 50.

The ranking score averages each component's percentile rank within the day,
which is what was validated. The hazard model's probability is shown beside
it, labelled as an estimate: it moves with network conditions and tends to run
high in calm periods.

Every score is recomputed live from the stored point-in-time documents. A
parity test asserts the live score equals the stored one. If the artifacts are
missing or altered, the predictor reports itself unavailable and the API
answers 503. It never invents a score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from data_pipeline import spec
from data_pipeline.features import CATEGORICAL_FEATURES, FEATURE_LABELS, NUMERIC_FEATURES
from data_pipeline.snapshot_features import (
    SNAPSHOT_FEATURE_LABELS,
    SNAPSHOT_MODEL_CATEGORICAL,
    SNAPSHOT_MODEL_FEATURES,
    SNAPSHOT_MODEL_NUMERIC,
)

log = get_logger(__name__)

ARTIFACT_SUBDIR = "v4"
FILES = ("hazard.ubj", "lambdamart.joblib", "support.joblib")
# Overdue orders are past their promise and certain to be late; they are not ranked.
UNRANKED_SCORE = -1.0


class ArtifactError(RuntimeError):
    """The served artifacts are missing, altered or cannot score this order."""


@dataclass
class Prediction:
    """One order, scored on its snapshot day."""

    probability: float
    ranking_score: float
    priority_rank: int | None
    cohort_size: int
    band: str
    eta_p10: date | None
    eta_p50: date | None
    eta_p90: date | None
    components: dict[str, float] = field(default_factory=dict)
    factors: list[dict] = field(default_factory=list)
    calibrated: bool = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile_rank(value: float, others: np.ndarray) -> float:
    """pandas' average-method percentile rank of `value` among itself and `others`.

    The order is ranked against every *other* order of the day rather than by
    finding its own stored value: XGBoost's last bits differ between the
    platform that stored the scores and the one serving them, and an exact
    match on a value that differs by 1e-7 would shift the rank.
    """
    less = float(np.sum(others < value))
    equal = float(np.sum(others == value))
    return (less + 1.0 + equal / 2.0) / (len(others) + 1.0)


class Predictor:
    """Thread-safe holder for the served v4 artifacts."""

    def __init__(self) -> None:
        self._model = None      # the ranker; None means unavailable
        self._hazard = None
        self._meta: dict | None = None
        self._categories: dict[str, list] = {}
        self._error: str | None = None
        self._lock = Lock()

    # --- loading ---------------------------------------------------------

    def load(self, artifact_dir: Path) -> None:
        with self._lock:
            try:
                self._load(Path(artifact_dir) / ARTIFACT_SUBDIR)
                self._error = None
                log.info("model_artifact_loaded", model_version=self.model_version)
            except Exception as exc:
                self._model = self._hazard = self._meta = None
                self._error = str(exc)
                log.error("model_artifact_load_failed", error=self._error)

    def _load(self, directory: Path) -> None:
        import joblib
        import xgboost as xgb

        from ml_pipeline.survival import HazardModel

        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise ArtifactError(f"model metadata not found at {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for name in FILES:
            path = directory / name
            if not path.exists():
                raise ArtifactError(f"model artifact not found at {path}")
            expected = meta.get("files", {}).get(name)
            if expected and _sha256(path) != expected:
                raise ArtifactError(f"{name} does not match its recorded checksum")

        support = joblib.load(directory / "support.joblib")
        booster = xgb.Booster()
        booster.load_model(directory / "hazard.ubj")
        hazard = HazardModel(support["context"], support["km"])
        hazard.booster = booster

        self._hazard = hazard
        self._model = joblib.load(directory / "lambdamart.joblib")
        self._categories = support["categories"]
        self._meta = meta

    # --- state -----------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._model is not None and self._hazard is not None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def model_version(self) -> str | None:
        return self._meta["model_version"] if self._meta else None

    @property
    def metadata(self) -> dict | None:
        return self._meta

    @property
    def calibrated(self) -> bool:
        # The probability shown is a model estimate, not calibrated across
        # periods; the ranking is the validated quantity.
        return False

    @property
    def bands(self) -> dict[str, str]:
        return dict((self._meta or {}).get("bands", {}))

    # --- inputs ----------------------------------------------------------

    def _order_frame(self, of) -> pd.DataFrame:
        """The order as the hazard model saw it in training (see survival.prepare)."""
        doc = of.features or {}
        if not doc:
            raise ArtifactError("no stored feature document for this order")
        row = {f: doc.get(f) for f in NUMERIC_FEATURES}
        row.update({
            "customer_state": of.customer_state,
            "seller_state": of.seller_state,
            "product_category": of.product_category,
            "handover_day": pd.Timestamp(of.order_delivered_carrier_date).normalize(),
            "est_day": pd.Timestamp(of.order_estimated_delivery_date).normalize(),
            # Matches `load_base_frame`, which builds the lane with astype(str).
            "lane": f"{of.seller_state if of.seller_state is not None else 'nan'}>"
                    f"{of.customer_state if of.customer_state is not None else 'nan'}",
        })
        frame = pd.DataFrame([row])
        for col in NUMERIC_FEATURES:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in CATEGORICAL_FEATURES:
            frame[col] = pd.Categorical(frame[col], categories=self._categories[col])
        frame["promise_age"] = (frame["est_day"] - frame["handover_day"]).dt.days
        return frame

    @staticmethod
    def _snapshot_frame(doc: dict) -> pd.DataFrame:
        frame = pd.DataFrame([{f: doc.get(f) for f in SNAPSHOT_MODEL_FEATURES}])
        for col in SNAPSHOT_MODEL_NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        for col in SNAPSHOT_MODEL_CATEGORICAL:
            frame[col] = frame[col].astype(object)
        return frame

    # --- scoring ---------------------------------------------------------

    def score(self, db: Session, order_id: str, snapshot_id: str,
              *, with_factors: bool = True) -> Prediction:
        """Score one order live, as of the start of its snapshot day."""
        from app.db.models import OrderFeature, Snapshot, SnapshotOrder

        if not self.available:
            raise ArtifactError(self._error or "model artifacts are not loaded")
        so = db.get(SnapshotOrder, (snapshot_id, order_id))
        of = db.get(OrderFeature, order_id)
        snap = db.get(Snapshot, snapshot_id)
        if so is None or of is None or snap is None:
            raise ArtifactError("this order is not part of that snapshot")

        at = pd.Series([pd.Timestamp(snap.snapshot_at).normalize()])
        frame = self._order_frame(of)
        q = self._hazard.arrival_quantiles(frame, at)[0]
        eta = [pd.Timestamp(v).date() for v in q]

        cohort = db.execute(
            select(SnapshotOrder.km_score, SnapshotOrder.hazard_score,
                   SnapshotOrder.lambdamart_score, SnapshotOrder.ranking_score,
                   SnapshotOrder.order_id)
            .where(SnapshotOrder.snapshot_id == snapshot_id,
                   SnapshotOrder.is_overdue.is_(False))
        ).all()
        n = len(cohort)

        if so.is_overdue:
            # Past its promise and undelivered at the start of the day: late by
            # the calendar-date rule, whatever happens next.
            return Prediction(probability=1.0, ranking_score=UNRANKED_SCORE, priority_rank=None,
                              cohort_size=n, band=spec.priority_band(None, n),
                              eta_p10=eta[0], eta_p50=eta[1], eta_p90=eta[2])

        if not so.snapshot_features:
            raise ArtifactError("no stored snapshot-day features for this order")
        X = self._snapshot_frame(so.snapshot_features)
        km = float(X["km_cond_late"].iloc[0])
        hz = float(self._hazard.predict_late(frame, at)[0])
        lm = float(self._model.predict(X[SNAPSHOT_MODEL_FEATURES])[0])

        others = [r for r in cohort if r[4] != order_id]
        arr = np.array([r[:4] for r in others], dtype=float).reshape(-1, 4)
        ids = np.array([r[4] for r in others])
        ensemble = float(np.mean([
            _percentile_rank(km, arr[:, 0]),
            _percentile_rank(hz, arr[:, 1]),
            _percentile_rank(lm, arr[:, 2]),
        ]))
        # Queue position: score descending, ties broken by order id, exactly
        # as `services.orders.list_orders` sorts.
        rank = int(np.sum(arr[:, 3] > ensemble)
                   + np.sum((arr[:, 3] == ensemble) & (ids < order_id))) + 1
        return Prediction(
            probability=hz, ranking_score=ensemble, priority_rank=rank, cohort_size=n,
            band=spec.priority_band(rank, n), eta_p10=eta[0], eta_p50=eta[1], eta_p90=eta[2],
            components={"kaplan_meier": km, "hazard": hz, "lambdamart": lm},
            factors=self.explain(X) if with_factors else [],
        )

    # --- explanation -----------------------------------------------------

    def explain(self, X: pd.DataFrame, top_n: int = 5) -> list[dict]:
        """Exact TreeSHAP attribution of the LambdaMART score, by source feature.

        One-hot columns are summed back into the feature they came from. These
        say what pushed the order up or down the ranking model's list. They are
        attributions of the model's output, never established causes (EVI-03).
        """
        import xgboost as xgb

        pre = self._model.named_steps["pre"]
        booster = self._model.named_steps["clf"].get_booster()
        transformed = pre.transform(X[SNAPSHOT_MODEL_FEATURES])
        contributions = booster.predict(xgb.DMatrix(transformed), pred_contribs=True)[0][:-1]

        totals: dict[str, float] = {}
        for name, value in zip(transformed.columns, contributions, strict=True):
            source = name if name in SNAPSHOT_MODEL_NUMERIC else max(
                (c for c in SNAPSHOT_MODEL_CATEGORICAL if str(name).startswith(c)),
                key=len, default=str(name))
            totals[source] = totals.get(source, 0.0) + float(value)

        magnitude = sum(abs(v) for v in totals.values()) or 1.0
        ranked = sorted(totals.items(), key=lambda kv: -abs(kv[1]))[:top_n]
        return [
            {
                "feature": name,
                "label": SNAPSHOT_FEATURE_LABELS.get(name) or FEATURE_LABELS.get(name, name),
                "direction": "increases risk" if value > 0 else "decreases risk",
                "share": round(abs(value) / magnitude, 4),
                "contribution": round(value, 4),
            }
            for name, value in ranked
        ]


predictor = Predictor()


def arrival_tag(eta_p50: date | None, promised: date, is_overdue: bool) -> str:
    """Plain reading of the arrival estimate against the promised date."""
    if is_overdue:
        return "overdue"
    if eta_p50 is None:
        return "unknown"
    buffer = (promised - eta_p50).days
    if buffer < 0:
        return "likely_late"
    if buffer <= 1:
        return "tight"
    return "on_track"


def prediction_computed_at() -> datetime:
    return datetime.now(UTC)
