"""Models that score an order as of a snapshot day.

Same preprocessing discipline as `ml_pipeline.model` (median imputation and
scaling fitted on training rows only, one-hot categoricals), over the wider
snapshot feature set from `data_pipeline.snapshot_features`.

Two deliberate differences from the handover model:

* **No `scale_pos_weight`.** Reweighting the classes inflated every raw score
  so badly that a raw 0.65 meant a 2.4% late rate. Ranking does not need it,
  and without it the logistic output is already on the probability scale.
* **Optional monotone constraints** on the features whose direction is not in
  doubt - more of the promised window used, a worse survival outlook, a more
  congested lane must never *lower* the risk. Under the period-to-period shift
  this dataset has, a constraint that rules out implausible shapes is cheap
  robustness.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_pipeline.snapshot_features import (
    SNAPSHOT_MODEL_CATEGORICAL,
    SNAPSHOT_MODEL_NUMERIC,
)

# +1: risk may only rise with the feature. -1: may only fall.
MONOTONE: dict[str, int] = {
    "window_used": 1,
    "days_in_transit": 1,
    "km_cond_late": 1,
    "km_cond_late_network": 1,
    "km_survival_now": -1,
    "lane_overdue_share": 1,
    "network_overdue_share": 1,
    "recent_late_rate_lane": 1,
    "recent_late_rate_network": 1,
}

BASE_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 8,
    "random_state": 42,
}


def build_snapshot_preprocessor(*, pandas_out: bool = False) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=30, sparse_output=False)),
    ])
    pre = ColumnTransformer(
        [("num", numeric, SNAPSHOT_MODEL_NUMERIC),
         ("cat", categorical, SNAPSHOT_MODEL_CATEGORICAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if pandas_out:
        # Column names reach XGBoost, so monotone constraints can be stated
        # by feature name rather than by a fragile column position.
        pre.set_output(transform="pandas")
    return pre


def build_snapshot_lr() -> Pipeline:
    return Pipeline([
        ("pre", build_snapshot_preprocessor()),
        ("clf", LogisticRegression(max_iter=3000, C=1.0)),
    ])


def build_snapshot_xgb(*, monotone: bool = False, **overrides: Any) -> Pipeline:
    from xgboost import XGBClassifier

    params = {**BASE_XGB_PARAMS, **overrides}
    if monotone:
        params["monotone_constraints"] = dict(MONOTONE)
    return Pipeline([
        ("pre", build_snapshot_preprocessor(pandas_out=True)),
        ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)),
    ])


def build_snapshot_ranker(**overrides: Any) -> Pipeline:
    """Learning-to-rank within each snapshot day: the decision is a top-K list."""
    from xgboost import XGBRanker

    params = {**BASE_XGB_PARAMS, **overrides}
    return Pipeline([
        ("pre", build_snapshot_preprocessor(pandas_out=True)),
        ("clf", XGBRanker(objective="rank:pairwise", **params)),
    ])
