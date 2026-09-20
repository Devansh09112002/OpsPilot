"""Model definitions and the artifact contract shared by training and serving.

`build_preprocessor` is the single definition of the input transformation.
Training fits it; FastAPI loads the fitted object from the artifact. Because
both sides call the same code path, a drift between them is a load error
rather than a silently wrong score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline import spec
from data_pipeline.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

ARTIFACT_FILENAME = "delivery_risk_model.joblib"
METADATA_FILENAME = "delivery_risk_model.meta.json"


@dataclass
class ModelMetadata:
    """Everything a caller needs to state what produced a score."""

    model_version: str
    model_family: str
    trained_at_utc: str
    training_cutoff: str
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    train_rows: int
    train_base_rate: float
    artifact_sha256: str
    selection_rationale: str
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    # Calibration maps the raw ranking score to an estimated probability. The
    # raw score stays the ranking key, so published ranking metrics are
    # unaffected; the calibrated value is what a human should read.
    calibrated: bool = False
    calibration_method: str = ""
    band_thresholds: dict[str, float] = field(default_factory=dict)
    calibration_metrics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics; one-hot the low-cardinality categoricals.

    Missing values stay missing until this point: imputation is fitted on the
    training split only, so no future information reaches a training row.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=30, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES),
         ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])


def build_xgboost(scale_pos_weight: float) -> Pipeline:
    """A compact XGBoost candidate. Fixed hyperparameters, no open-ended search.

    Plan section 4.2 explicitly forbids a large hyperparameter sweep; these
    values are conventional defaults for a small tabular problem.
    """
    from xgboost import XGBClassifier

    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=4,
            random_state=42,
        )),
    ])


class DeadlineProximityRule:
    """The operational rule any ML model has to beat.

    An operations team without a model would rank by how little slack remains
    between carrier handover and the promised date. This reproduces that
    heuristic so the comparison is against practice, not just another model.
    """

    model_family = "rule_deadline_proximity"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> DeadlineProximityRule:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        slack = X["days_handover_to_estimate"].to_numpy(dtype=float)
        slack = np.nan_to_num(slack, nan=float(np.nanmedian(slack)))
        # Monotone decreasing in slack, squashed to (0, 1) for comparability.
        score = 1.0 / (1.0 + np.exp((slack - 7.0) / 4.0))
        return np.column_stack([1.0 - score, score])


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(
    model: Pipeline, meta: ModelMetadata, artifact_dir: Path, calibrator=None
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / ARTIFACT_FILENAME
    joblib.dump(
        {"model": model, "feature_names": MODEL_FEATURES, "calibrator": calibrator},
        path,
    )
    meta.artifact_sha256 = sha256_of_file(path)
    (artifact_dir / METADATA_FILENAME).write_text(meta.to_json(), encoding="utf-8")
    return path


class ArtifactError(RuntimeError):
    """Raised when the model artifact is missing, corrupt or schema-mismatched."""


def load_artifact(artifact_dir: Path) -> tuple[Pipeline, dict]:
    """Load the served model, verifying integrity and feature schema.

    A mismatch raises rather than returning a usable-looking object, so the API
    can answer "prediction unavailable" instead of serving a wrong number.
    """
    path = artifact_dir / ARTIFACT_FILENAME
    meta_path = artifact_dir / METADATA_FILENAME
    if not path.exists():
        raise ArtifactError(f"model artifact not found at {path}")
    if not meta_path.exists():
        raise ArtifactError(f"model metadata not found at {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = sha256_of_file(path)
    if meta.get("artifact_sha256") and meta["artifact_sha256"] != actual:
        raise ArtifactError(
            f"artifact checksum mismatch: metadata says {meta['artifact_sha256'][:16]}, "
            f"file is {actual[:16]}"
        )

    bundle = joblib.load(path)
    model, feature_names = bundle["model"], bundle["feature_names"]
    calibrator = bundle.get("calibrator")
    if meta.get("calibrated") and calibrator is None:
        raise ArtifactError(
            "metadata says the model is calibrated but the artifact contains no "
            "calibrator; refusing to serve an uncalibrated score as a probability"
        )
    meta["_calibrator"] = calibrator
    if list(feature_names) != list(meta["feature_names"]):
        raise ArtifactError("artifact and metadata disagree on the feature schema")
    if list(feature_names) != list(MODEL_FEATURES):
        raise ArtifactError(
            "artifact feature schema does not match the current feature builder; "
            "retrain before serving"
        )
    return model, meta


def predict_risk(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Score a feature frame, enforcing column order and presence."""
    missing = [c for c in MODEL_FEATURES if c not in features.columns]
    if missing:
        raise ArtifactError(f"cannot score: missing feature column(s) {missing}")
    return model.predict_proba(features[MODEL_FEATURES])[:, 1]


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_calibrator(raw_scores, y_true):
    """Fit isotonic regression mapping a raw score to a probability.

    Fitted on the VALIDATION split, never on train (the model already fits
    train, so its scores there are optimistic) and never on test (which must
    stay untouched for the final evaluation).

    Why this is needed here: `scale_pos_weight` rebalances the classes during
    training, which inflates every output. Measured before calibration, a
    predicted 0.65 corresponded to a 2.4% observed late rate - a 27x
    overstatement. Displaying that to a person is misleading even with a
    disclaimer next to it.
    """
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        raw_scores, y_true
    )


def calibrate(calibrator, raw_scores) -> np.ndarray:
    """Map raw ranking scores to estimated probabilities."""
    if calibrator is None:
        return np.asarray(raw_scores, dtype=float)
    return np.asarray(calibrator.predict(np.asarray(raw_scores, dtype=float)))


def derive_band_thresholds(calibrated_scores) -> dict[str, float]:
    """Cut points for the risk bands shown in the product.

    `high` is the policy's escalation threshold, not a quantile. Deriving it
    independently produced a band edge (0.1659) slightly above the policy
    threshold (0.15), so an order could display as "medium risk" and still be
    escalated - a state a reader would rightly find confusing, and one the
    agent benchmark caught. Tying the band to the policy makes "high" mean
    "qualifies for escalation if the deadline is close".

    `medium` is the validation 75th percentile: purely presentational, marking
    the upper quarter of a snapshot.
    """
    scores = np.asarray(calibrated_scores, dtype=float)
    medium = round(float(np.quantile(scores, 0.75)), 4)
    return {
        "high": spec.ESCALATION_THRESHOLD,
        "medium": min(medium, spec.ESCALATION_THRESHOLD),
    }


def band_for(probability: float, thresholds: dict[str, float]) -> str:
    if probability >= thresholds["high"]:
        return "high"
    if probability >= thresholds["medium"]:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Per-order explanation
# ---------------------------------------------------------------------------

def _expanded_to_source(pipeline: Pipeline) -> list[str]:
    """Map each transformed column back to the feature it came from.

    Derived from the fitted encoder's own output names rather than from
    `categories_` lengths: `handle_unknown="infrequent_if_exist"` adds an
    extra "infrequent" column per categorical, so counting categories
    undercounts and the mapping silently misaligns.

    Longest-prefix matching, because `customer_state_BA` also starts with
    nothing else but a shorter feature name could otherwise win.
    """
    pre = pipeline.named_steps["pre"]
    mapping: list[str] = list(NUMERIC_FEATURES)

    for name, transformer, columns in pre.transformers_:
        if name != "cat":
            continue
        onehot = transformer.named_steps["onehot"]
        for output_name in onehot.get_feature_names_out(columns):
            source = max(
                (c for c in columns if str(output_name).startswith(c)),
                key=len,
                default=str(output_name),
            )
            mapping.append(source)
    return mapping


def explain(pipeline: Pipeline, features: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Exact per-order TreeSHAP contributions, aggregated to source features.

    Uses XGBoost's built-in `pred_contribs`, so there is no extra dependency
    and no sampling: contributions sum exactly to the model's margin, which is
    asserted in the tests. One-hot columns are summed back into their source
    feature so a reader sees "destination state", not "customer_state_BA".

    Contributions are in log-odds. Their sign is what matters to a reader -
    which factors pushed this order up or down - so the magnitude is reported
    as a share of the total absolute contribution rather than as a raw number
    that invites over-reading.
    """
    import xgboost as xgb

    from data_pipeline.features import FEATURE_LABELS

    clf = pipeline.named_steps["clf"]
    get_booster = getattr(clf, "get_booster", None)
    if get_booster is None:
        # Only a tree model exposes exact TreeSHAP. A linear or rule model
        # returns no attribution rather than a fabricated one.
        return []
    booster = get_booster()

    transformed = pipeline.named_steps["pre"].transform(features[MODEL_FEATURES])
    matrix = xgb.DMatrix(transformed)
    contributions = booster.predict(matrix, pred_contribs=True)[0][:-1]

    sources = _expanded_to_source(pipeline)
    if len(sources) != len(contributions):
        return []

    totals: dict[str, float] = {}
    for source, value in zip(sources, contributions, strict=True):
        totals[source] = totals.get(source, 0.0) + float(value)

    magnitude = sum(abs(v) for v in totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    return [
        {
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "direction": "increases risk" if value > 0 else "decreases risk",
            "share": round(abs(value) / magnitude, 4),
            "contribution": round(float(value), 4),
        }
        for name, value in ranked
    ]
