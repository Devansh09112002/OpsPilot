"""Model definitions and the artifact contract shared by training and serving.

`build_preprocessor` is the single definition of the input transformation.
Training fits it; FastAPI loads the fitted object from the artifact. Because
both sides call the same code path, a drift between them is a load error
rather than a silently wrong score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DeadlineProximityRule":
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


def save_artifact(model: Pipeline, meta: ModelMetadata, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / ARTIFACT_FILENAME
    joblib.dump({"model": model, "feature_names": MODEL_FEATURES}, path)
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
