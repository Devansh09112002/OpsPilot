"""Model report inputs: per-snapshot metrics, error analysis, importances.

Markdown rendering lives in `ml_pipeline.render`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.features import MODEL_FEATURES
from ml_pipeline.metrics import evaluate


def per_snapshot_metrics(model, df: pd.DataFrame, k: int) -> dict:
    """Score each frozen demo snapshot the way the product actually uses it.

    Only pre-deadline orders are ranked, mirroring the risk queue.
    """
    mpath = spec.PROCESSED_DIR / "snapshot_members.parquet"
    if not mpath.exists():
        return {}
    members = pd.read_parquet(mpath)
    out: dict[str, dict] = {}

    for sid in members["snapshot_id"].unique():
        sub = members[(members["snapshot_id"] == sid) & (~members["is_overdue"])]
        merged = sub.merge(df, on="order_id", how="inner", validate="one_to_one")
        if merged.empty or merged[spec.TARGET_NAME].nunique() < 2:
            continue
        scores = model.predict_proba(merged[MODEL_FEATURES])[:, 1]
        out[str(sid)] = evaluate(merged[spec.TARGET_NAME].to_numpy().astype(int), scores, k)
    return out


def calibration_table(y_true, scores, n_bins: int = 10) -> list[dict]:
    """Reliability bins over whatever scores are passed in."""
    from ml_pipeline.metrics import calibration_bins

    return calibration_bins(np.asarray(y_true).astype(int), np.asarray(scores, dtype=float),
                            n_bins=n_bins)


def error_analysis(model, df: pd.DataFrame, k: int) -> dict:
    """Where the served model is wrong on the held-out test split."""
    te = df[df["split"] == "test"].copy()
    te["score"] = model.predict_proba(te[MODEL_FEATURES])[:, 1]
    te = te.sort_values("score", ascending=False)

    top = te.head(k)
    false_positives = top[~top[spec.TARGET_NAME]]
    missed = te[te[spec.TARGET_NAME]].tail(len(te[te[spec.TARGET_NAME]]) // 2)

    def profile(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {}
        return {
            "n": int(len(frame)),
            "median_days_handover_to_estimate": round(
                float(frame["days_handover_to_estimate"].median()), 2),
            "median_hours_purchase_to_handover": round(
                float(frame["hours_purchase_to_handover"].median()), 1),
            "share_cross_state": round(float(frame["is_cross_state"].mean()), 3),
            "top_customer_states": frame["customer_state"].value_counts().head(3).to_dict(),
        }

    return {
        "top_k_false_positives": profile(false_positives),
        "low_ranked_late_orders": profile(missed),
        "overall_test_profile": profile(te),
    }


def feature_importance(model, family: str, top_n: int = 15) -> list[dict]:
    """Best-effort importance readout for the served model family."""
    try:
        pre = model.named_steps["pre"]
        names = list(pre.get_feature_names_out())
        clf = model.named_steps["clf"]
        if family == "logistic_regression":
            vals = np.abs(clf.coef_[0])
            signed = clf.coef_[0]
        else:
            vals = clf.feature_importances_
            signed = vals
        order = np.argsort(-vals)[:top_n]
        return [
            {"feature": str(names[i]), "weight": round(float(signed[i]), 5)}
            for i in order
        ]
    except Exception:  # importances are informational, never load-bearing
        return []


def measure_latency(model, X: pd.DataFrame, n: int = 200) -> dict:
    """Single-row inference latency, as served by the API."""
    sample = X.head(min(n, len(X)))
    times: list[float] = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        t0 = time.perf_counter()
        model.predict_proba(row)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "n_calls": int(len(arr)),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }
