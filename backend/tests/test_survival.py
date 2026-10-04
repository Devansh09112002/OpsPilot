"""The survival models: censoring, the as-of rule, and the late-probability arithmetic.

Like `test_snapshot_features.py`, the as-of tests scramble every outcome that
was still open on the day and assert that nothing computed for that day moves.
They read the processed parquet files and raw CSVs; no database is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline import spec
from ml_pipeline.survival import (
    HazardModel,
    WeeklyKM,
    _person_days,
    daily_context,
    prepare,
)

AT = pd.Timestamp("2018-03-15")
FIRST = pd.Timestamp("2017-10-01")


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    if not (spec.PROCESSED_DIR / "order_features.parquet").exists():
        pytest.skip("processed data missing; run python -m data_pipeline.audit")
    from data_pipeline.snapshot_features import load_base_frame

    return load_base_frame()


@pytest.fixture(scope="module")
def df(base) -> pd.DataFrame:
    return prepare(base)


def _scramble_open(base: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Move every delivery still open at AT to a random later moment."""
    rng = np.random.default_rng(seed)
    out = base.copy()
    open_at = (out["order_delivered_customer_date"] > AT) & (out["est_day"] >= AT)
    floor = out.loc[open_at, "order_delivered_carrier_date"].clip(lower=AT)
    out.loc[open_at, "order_delivered_customer_date"] = floor + pd.to_timedelta(
        rng.uniform(0.5, 45.0, open_at.sum()), unit="D")
    out.loc[open_at, spec.TARGET_NAME] = (
        out.loc[open_at, "order_delivered_customer_date"].dt.normalize()
        > out.loc[open_at, "est_day"])
    return out


# --- expansion and censoring ------------------------------------------------

def test_person_days_cover_each_inclusive_range():
    orders = pd.DataFrame({"handover_day": pd.to_datetime(["2018-01-01", "2018-01-05"])})
    rows = _person_days(orders, np.array([0, 2]), np.array([2, 3]))
    assert list(rows["age"]) == [0, 1, 2, 2, 3]
    assert list(rows["day"].dt.day) == [1, 2, 3, 7, 8]


def test_training_rows_never_reach_past_the_cutoff(df):
    cutoff = pd.Timestamp("2017-12-01")
    rows = HazardModel(daily_context(df, pd.date_range(FIRST, cutoff)),
                       WeeklyKM.build(df, FIRST, cutoff)).training_rows(df, cutoff)
    assert (rows["day"] < cutoff).all()
    events = rows[rows["event"] == 1]
    # An event is the delivery day itself, and only a delivery before the cutoff.
    assert (events["day"] == events["delivered_day"]).all()
    assert (events["delivered_day"] < cutoff).all()


def test_a_parcel_still_moving_at_the_cutoff_is_censored_not_an_event(df):
    cutoff = pd.Timestamp("2017-12-01")
    rows = HazardModel(daily_context(df, pd.date_range(FIRST, cutoff)),
                       WeeklyKM.build(df, FIRST, cutoff)).training_rows(df, cutoff)
    moving = rows[rows["delivered_day"] >= cutoff]
    assert len(moving) > 1000
    assert moving["event"].sum() == 0


# --- the as-of rule -----------------------------------------------------------

def test_daily_context_ignores_outcomes_still_open_that_day(base):
    original = daily_context(prepare(base), pd.DatetimeIndex([AT]))
    for seed in (1, 2):
        scrambled = daily_context(prepare(_scramble_open(base, seed)), pd.DatetimeIndex([AT]))
        pd.testing.assert_frame_equal(original, scrambled)


def test_weekly_curves_ignore_outcomes_still_open(base):
    original = WeeklyKM.build(prepare(base), FIRST, AT)
    scrambled = WeeklyKM.build(prepare(_scramble_open(base, 3)), FIRST, AT)
    np.testing.assert_array_equal(original.lane_hazard, scrambled.lane_hazard)
    np.testing.assert_array_equal(original.net_hazard, scrambled.net_hazard)


# --- the late probability ---------------------------------------------------

class _ConstantHazard:
    def __init__(self, h: float):
        self.h = h

    def predict(self, dm):
        return np.full(dm.num_row(), self.h)


def _model_with_constant_hazard(df, h):
    model = HazardModel(daily_context(df, pd.DatetimeIndex([AT])), WeeklyKM.build(df, FIRST, AT))
    model.booster = _ConstantHazard(h)
    return model


def _cohort(df):
    mask = ((df["order_delivered_carrier_date"] <= AT)
            & (df["order_delivered_customer_date"] > AT) & (df["est_day"] >= AT))
    orders = df[mask].head(200).reset_index(drop=True)
    return orders, pd.Series(AT, index=orders.index)


def test_late_means_surviving_every_day_through_the_promised_day(df):
    orders, at = _cohort(df)
    p = _model_with_constant_hazard(df, 0.1).predict_late(orders, at)
    days_at_risk = (orders["est_day"] - AT).dt.days.to_numpy() + 1  # inclusive
    np.testing.assert_allclose(p, 0.9 ** days_at_risk, rtol=1e-6)


def test_the_arrival_estimate_reads_the_survival_curve(df):
    orders, at = _cohort(df)
    q = _model_with_constant_hazard(df, 0.5).arrival_quantiles(orders, at, quantiles=(0.5, 0.75))
    # Half arrive on day one; three quarters by the end of day two.
    assert (pd.to_datetime(q[:, 0]) == AT).all()
    assert (pd.to_datetime(q[:, 1]) == AT + pd.Timedelta(days=1)).all()
