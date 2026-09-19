"""Evaluation metrics for a ranking-under-review-capacity decision problem.

The operations team can only review K orders per snapshot, so Precision@K is
the primary metric. Accuracy is deliberately not reported: at a ~4% base rate
a model that predicts "never late" scores 96% and is useless.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of genuinely late orders among the K highest-ranked."""
    k = min(k, len(scores))
    if k == 0:
        return float("nan")
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float(np.mean(y_true[top]))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of all late orders captured in the K highest-ranked."""
    total = float(np.sum(y_true))
    if total == 0:
        return float("nan")
    k = min(k, len(scores))
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float(np.sum(y_true[top]) / total)


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision@K divided by the base rate: how much better than random."""
    base = float(np.mean(y_true))
    if base == 0:
        return float("nan")
    return precision_at_k(y_true, scores, k) / base


def calibration_bins(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Equal-width reliability bins, for the calibration table in the report."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not mask.any():
            continue
        out.append({
            "bin_low": round(float(lo), 3),
            "bin_high": round(float(hi), 3),
            "n": int(mask.sum()),
            "mean_predicted": round(float(scores[mask].mean()), 4),
            "observed_rate": round(float(y_true[mask].mean()), 4),
        })
    return out


def evaluate(
    y_true: np.ndarray, scores: np.ndarray, k: int, with_calibration: bool = False
) -> dict:
    """Full metric bundle for one model on one split."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    result = {
        "n": int(len(y_true)),
        "n_late": int(y_true.sum()),
        "base_rate": round(float(y_true.mean()), 5),
        f"precision_at_{k}": round(precision_at_k(y_true, scores, k), 4),
        f"recall_at_{k}": round(recall_at_k(y_true, scores, k), 4),
        f"lift_at_{k}": round(lift_at_k(y_true, scores, k), 3),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
    }
    # A rule baseline emits an ordinal score, not a probability, so Brier is
    # only meaningful where the scores are genuine probabilities.
    if scores.min() >= 0.0 and scores.max() <= 1.0:
        result["brier"] = round(float(brier_score_loss(y_true, scores)), 5)
    if with_calibration:
        result["calibration"] = calibration_bins(y_true, scores)
    return result
