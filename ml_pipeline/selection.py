"""Snapshot-simulated evaluation: score models the way the product is used.

Computing Precision@50 once over a whole 20k-order split measures only 50
orders, so a 0.80-vs-0.90 gap is five orders of noise and far too weak to
select a model on. The deployed product instead ranks one snapshot at a time
(roughly 850-1,550 pre-deadline orders) and reviews the top K of each.

This module reproduces that protocol over a date range: it simulates a snapshot
every `step_days`, ranks the orders genuinely in transit and pre-deadline at
that moment, and averages Precision@K across snapshots. That yields many more
effective top-slot observations and matches the decision the model is for.

Eventual outcomes are read here for offline scoring only. Nothing in this module
is reachable from an as-of API path or an agent tool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.features import MODEL_FEATURES
from ml_pipeline.metrics import precision_at_k, recall_at_k


def simulate_snapshot_dates(
    start: pd.Timestamp, end: pd.Timestamp, step_days: int = 7
) -> list[pd.Timestamp]:
    """Snapshot dates spanning a period, leaving a margin at each edge.

    The margin avoids evaluating on partially-populated cohorts at the split
    boundaries, where orders in transit were handed over in the previous split.
    """
    first = start + pd.Timedelta(days=step_days)
    last = end - pd.Timedelta(days=step_days)
    dates: list[pd.Timestamp] = []
    cur = first
    while cur <= last:
        dates.append(cur)
        cur = cur + pd.Timedelta(days=step_days)
    return dates


def snapshot_cohort(df: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """Orders in transit and still pre-deadline at `at`.

    Membership is derived purely from event timestamps, exactly as the product
    builds a risk queue.
    """
    return df[
        (df["order_delivered_carrier_date"] <= at)
        & (df["order_delivered_customer_date"] > at)
        & (df["order_estimated_delivery_date"].dt.normalize() >= at.normalize())
    ]


def evaluate_over_snapshots(
    model,
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    k: int,
    step_days: int = 7,
    min_cohort: int = 200,
) -> dict:
    """Average Precision@K / Recall@K across simulated snapshots in a period."""
    precisions: list[float] = []
    recalls: list[float] = []
    cohorts: list[int] = []
    base_rates: list[float] = []

    for at in simulate_snapshot_dates(start, end, step_days):
        cohort = snapshot_cohort(df, at)
        if len(cohort) < min_cohort or cohort[spec.TARGET_NAME].nunique() < 2:
            continue
        y = cohort[spec.TARGET_NAME].to_numpy().astype(int)
        scores = model.predict_proba(cohort[MODEL_FEATURES])[:, 1]
        precisions.append(precision_at_k(y, scores, k))
        recalls.append(recall_at_k(y, scores, k))
        cohorts.append(len(cohort))
        base_rates.append(float(y.mean()))

    if not precisions:
        return {"n_snapshots": 0}

    p = np.array(precisions)
    r = np.array(recalls)
    return {
        "n_snapshots": len(p),
        "median_cohort_size": int(np.median(cohorts)),
        "mean_base_rate": round(float(np.mean(base_rates)), 5),
        f"mean_precision_at_{k}": round(float(p.mean()), 4),
        f"std_precision_at_{k}": round(float(p.std(ddof=1)) if len(p) > 1 else 0.0, 4),
        f"mean_recall_at_{k}": round(float(r.mean()), 4),
        f"mean_lift_at_{k}": round(
            float(p.mean() / np.mean(base_rates)) if np.mean(base_rates) > 0 else float("nan"), 3
        ),
        "top_slots_observed": int(len(p) * k),
    }
