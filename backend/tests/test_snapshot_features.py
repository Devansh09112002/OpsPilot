"""The snapshot-day features may use only what was known at the start of D.

The decisive test scrambles every outcome that was *not yet settled* at D -
later delivery dates, and the late flag that follows from them - and asserts
that not one feature value moves. A negative control then changes a settled
outcome and asserts that something does move, so the first test is known to
have the power to fail.

These tests read the processed parquet files and the raw CSVs, which CI builds
before the suite runs. They need no database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline import spec
from data_pipeline.snapshot_features import (
    SNAPSHOT_FEATURE_AVAILABILITY,
    SNAPSHOT_MODEL_FEATURES,
    SNAPSHOT_NUMERIC_FEATURES,
    _km_curves,
    attach_labels,
    build_snapshot_rows,
    in_transit_pre_deadline,
    load_base_frame,
)

AT = pd.Timestamp("2018-03-15")


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    if not (spec.PROCESSED_DIR / "order_features.parquet").exists():
        pytest.skip("processed data missing; run python -m data_pipeline.audit")
    return load_base_frame()


def _rows(df: pd.DataFrame) -> pd.DataFrame:
    return (build_snapshot_rows(df, [AT])
            .sort_values("order_id").reset_index(drop=True))


def _scramble_unsettled(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rewrite every outcome that was still open at AT.

    An order is open when it was neither delivered before AT nor past its
    promise. Its delivery is moved to a random later moment (still after AT,
    so whether it is in transit on AT is unchanged) and the late flag is
    recomputed from the new date, exactly as the real label would be.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    delivered = out["order_delivered_customer_date"]
    open_at = (delivered > AT) & (out["est_day"] >= AT)
    later = out.loc[open_at, ["order_delivered_carrier_date"]].max(axis=1).clip(lower=AT)
    jitter = pd.to_timedelta(rng.uniform(0.5, 45.0, open_at.sum()), unit="D")
    out.loc[open_at, "order_delivered_customer_date"] = later + jitter
    out.loc[open_at, spec.TARGET_NAME] = (
        out.loc[open_at, "order_delivered_customer_date"].dt.normalize()
        > out.loc[open_at, "est_day"]
    )
    assert open_at.sum() > 1000, "the scramble should touch thousands of orders"
    return out


def test_no_feature_moves_when_unsettled_outcomes_are_scrambled(base):
    original = _rows(base)
    for seed in (1, 2):
        scrambled = _rows(_scramble_unsettled(base, seed))
        assert list(scrambled["order_id"]) == list(original["order_id"])
        pd.testing.assert_frame_equal(
            original[SNAPSHOT_MODEL_FEATURES], scrambled[SNAPSHOT_MODEL_FEATURES],
            check_exact=False, rtol=1e-12, atol=1e-12,
        )


def test_negative_control_a_settled_outcome_does_move_a_feature(base):
    """Without this, the test above could pass by never reading outcomes."""
    original = _rows(base)
    changed = base.copy()
    settled = (changed["order_delivered_customer_date"] < AT) & (
        changed["order_delivered_customer_date"] >= AT - pd.Timedelta(days=40)
    )
    changed.loc[settled, spec.TARGET_NAME] = ~changed.loc[settled, spec.TARGET_NAME]
    moved = _rows(changed)
    diff = (original[SNAPSHOT_NUMERIC_FEATURES] - moved[SNAPSHOT_NUMERIC_FEATURES]).abs()
    assert diff.to_numpy().max() > 1e-3


def test_cohort_matches_the_published_evaluation_population(base):
    from ml_pipeline.selection import snapshot_cohort

    ours = set(base.loc[in_transit_pre_deadline(base, AT), "order_id"])
    theirs = set(snapshot_cohort(base, AT)["order_id"])
    assert ours == theirs


def test_one_row_per_order_and_day(base):
    rows = build_snapshot_rows(base, pd.date_range("2018-03-01", "2018-03-21", freq="7D"))
    assert not rows.duplicated(["snapshot_at", "order_id"]).any()


def test_every_snapshot_feature_is_justified():
    assert set(SNAPSHOT_NUMERIC_FEATURES) == set(SNAPSHOT_FEATURE_AVAILABILITY)
    for name, (source, moment, why) in SNAPSHOT_FEATURE_AVAILABILITY.items():
        assert source and moment and len(why) > 20, name


def test_labels_record_when_they_became_known(base):
    rows = attach_labels(build_snapshot_rows(base, [AT]), base)
    # A ranked order is undelivered and pre-deadline on AT, so its label
    # cannot have been known on or before AT.
    assert (rows["known_from"] > AT).all()


def test_kaplan_meier_matches_the_empirical_curve_without_censoring():
    durations = np.array([0.5, 1.5, 1.7, 3.2, 5.0])
    surv, n = _km_curves(durations, np.ones(5, dtype=bool), np.zeros(5, dtype=int), 1)
    assert n[0] == 5
    # P(T >= k) for k = 0..6 on the daily grid.
    expected = [1.0, 0.8, 0.4, 0.4, 0.2, 0.2, 0.0]
    assert np.allclose(surv[0, :7], expected)


def test_kaplan_meier_keeps_censored_parcels_at_risk():
    # Two delivered on day 1; two still moving at day 3 (censored).
    durations = np.array([1.2, 1.4, 3.5, 3.9])
    events = np.array([True, True, False, False])
    surv, _ = _km_curves(durations, events, np.zeros(4, dtype=int), 1)
    # Dropping the censored parcels would wrongly say nobody survives day 2.
    assert surv[0, 2] == pytest.approx(0.5)
    assert surv[0, 4] == pytest.approx(0.5)
