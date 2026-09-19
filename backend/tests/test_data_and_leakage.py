"""Data-integrity, leakage and training-serving parity tests.

These are the tests that must fail the build rather than be relaxed. Each one
encodes a rule from project plan sections 3, 4 and 9.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_pipeline import spec
from data_pipeline.features import (
    FEATURE_AVAILABILITY,
    MODEL_FEATURES,
    OUTCOME_COLUMNS,
    split_features_and_outcomes,
)
from data_pipeline.loading import (
    DataIntegrityError,
    compute_target,
    load_raw,
    select_eligible,
)
from data_pipeline.snapshots import build_snapshot_members
from ml_pipeline.model import load_artifact, predict_risk

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    path = spec.PROCESSED_DIR / "order_features.parquet"
    if not path.exists():
        pytest.skip("run `python -m data_pipeline.audit` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def outcomes() -> pd.DataFrame:
    path = spec.PROCESSED_DIR / "order_outcomes.parquet"
    if not path.exists():
        pytest.skip("run `python -m data_pipeline.audit` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def labelled(features, outcomes) -> pd.DataFrame:
    return features.merge(outcomes, on="order_id", validate="one_to_one")


# --------------------------------------------------------------------------
# Grain
# --------------------------------------------------------------------------

def test_exactly_one_row_per_order(features):
    assert features["order_id"].is_unique


def test_multi_item_orders_do_not_duplicate_a_score(features):
    """A multi-seller, multi-item order must still yield one risk record."""
    multi = features[features["n_items"] > 1]
    assert len(multi) > 0, "fixture should contain multi-item orders"
    assert multi["order_id"].is_unique
    multi_seller = features[features["n_distinct_sellers"] > 1]
    assert len(multi_seller) > 0
    assert multi_seller["order_id"].is_unique


# --------------------------------------------------------------------------
# Target rule
# --------------------------------------------------------------------------

def test_calendar_date_rule_not_naive_timestamp():
    """Same-day afternoon delivery against a midnight promise is NOT late."""
    df = pd.DataFrame({
        "order_delivered_customer_date": pd.to_datetime(
            ["2018-07-10 16:30:00", "2018-07-11 00:05:00", "2018-07-09 23:59:00"]),
        "order_estimated_delivery_date": pd.to_datetime(
            ["2018-07-10 00:00:00", "2018-07-10 00:00:00", "2018-07-10 00:00:00"]),
    })
    assert compute_target(df).tolist() == [False, True, False]


def test_cancelled_orders_are_excluded_not_relabelled():
    raw = load_raw()
    eligible, log = select_eligible(raw["orders"])
    statuses = set(raw["orders"].loc[
        ~raw["orders"]["order_id"].isin(eligible["order_id"]), "order_status"])
    assert "canceled" in statuses
    assert (eligible["order_status"] == "delivered").all()
    # every exclusion is counted, nothing silently vanishes
    assert log.total - sum(n for _, n in log.steps) == log.eligible


def test_orders_certain_to_be_late_are_excluded(labelled):
    """Handover after the promised date is arithmetic, not prediction."""
    assert (labelled["days_handover_to_estimate"] >= 0).all()


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------

def test_feature_table_contains_no_outcome_column(features):
    banned = {"order_delivered_customer_date", "is_late", "order_status"}
    assert banned.isdisjoint(features.columns)


def test_every_model_feature_has_an_availability_justification():
    undocumented = [f for f in MODEL_FEATURES if f not in FEATURE_AVAILABILITY]
    assert undocumented == []


def test_adding_an_unjustified_feature_fails_the_build(monkeypatch):
    """Deliberately inject a feature with no justification; the build must fail."""
    import data_pipeline.features as feat

    monkeypatch.setattr(feat, "MODEL_FEATURES", [*feat.MODEL_FEATURES, "leaked_future_flag"])
    raw = load_raw()
    eligible, _ = select_eligible(raw["orders"])
    with pytest.raises(RuntimeError):
        feat.build_feature_table(raw, eligible)


def test_as_of_history_never_reads_a_later_outcome():
    """The seller-history feature must ignore outcomes at or after handover.

    Two orders from one seller: the first is delivered late but only *after*
    the second order is handed over. The second must therefore see zero prior
    outcomes, not one.
    """
    from data_pipeline.features import _as_of_history

    orders = pd.DataFrame({
        "order_id": ["a", "b"],
        "order_delivered_carrier_date": pd.to_datetime(["2018-01-01", "2018-01-05"]),
        "order_delivered_customer_date": pd.to_datetime(["2018-01-20", "2018-01-25"]),
        spec.TARGET_NAME: [True, True],
    })
    out = _as_of_history(orders, pd.Series(["S1", "S1"]), prior_rate=0.05, name="seller")
    assert out.loc["b", "seller_prior_orders"] == 0.0
    # with no prior outcomes the smoothed rate collapses to the prior
    assert out.loc["b", "seller_prior_late_rate"] == pytest.approx(0.05)


def test_as_of_history_does_read_a_strictly_earlier_outcome():
    from data_pipeline.features import _as_of_history

    orders = pd.DataFrame({
        "order_id": ["a", "b"],
        "order_delivered_carrier_date": pd.to_datetime(["2018-01-01", "2018-03-01"]),
        "order_delivered_customer_date": pd.to_datetime(["2018-01-20", "2018-03-20"]),
        spec.TARGET_NAME: [True, False],
    })
    out = _as_of_history(orders, pd.Series(["S1", "S1"]), prior_rate=0.05, name="seller")
    assert out.loc["b", "seller_prior_orders"] == 1.0
    assert out.loc["b", "seller_prior_late_rate"] > 0.05  # the late outcome moved it up


def test_split_features_and_outcomes_rejects_outcome_leakage():
    df = pd.DataFrame({c: [1] for c in
                       ["order_id", *MODEL_FEATURES, "order_delivered_customer_date", "is_late"]})
    for c in ["order_purchase_timestamp", "order_approved_at",
              "order_delivered_carrier_date", "order_estimated_delivery_date", "split"]:
        df[c] = 1
    feats, outs = split_features_and_outcomes(df)
    assert "is_late" not in feats.columns
    assert set(OUTCOME_COLUMNS) <= set(outs.columns)


# --------------------------------------------------------------------------
# Chronology / splits
# --------------------------------------------------------------------------

def test_splits_are_chronologically_disjoint(features):
    h = features["order_delivered_carrier_date"]
    train_max = h[features["split"] == "train"].max()
    val_min = h[features["split"] == "validation"].min()
    val_max = h[features["split"] == "validation"].max()
    test_min = h[features["split"] == "test"].min()
    assert train_max < spec.TRAIN_END <= val_min
    assert val_max < spec.VALIDATION_END <= test_min


def test_every_split_contains_both_classes(labelled):
    for name in ("train", "validation", "test"):
        sub = labelled[labelled["split"] == name]
        assert sub[spec.TARGET_NAME].sum() > 0
        assert (~sub[spec.TARGET_NAME]).sum() > 0


def test_training_cutoff_violation_is_detected():
    """Move one test order before the cutoff; the assertion must fire."""
    from ml_pipeline.train import assert_no_temporal_leakage

    df = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "order_delivered_carrier_date": pd.to_datetime(
            ["2018-02-01", "2018-04-01", "2018-01-01"]),  # test order before train end
    })
    with pytest.raises(AssertionError):
        assert_no_temporal_leakage(df)


# --------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------

def test_snapshot_membership_uses_timestamps_not_status(features, outcomes):
    members = build_snapshot_members(features, outcomes)
    joined = members.merge(features, on="order_id").merge(outcomes, on="order_id")
    at = pd.to_datetime(joined["snapshot_id"])
    assert (joined["order_delivered_carrier_date"] <= at).all()
    assert (joined["order_delivered_customer_date"] > at).all()


def test_snapshot_orders_are_all_held_out(features, outcomes):
    members = build_snapshot_members(features, outcomes)
    test_ids = set(features.loc[features["split"] == "test", "order_id"])
    assert set(members["order_id"]) <= test_ids


def test_overdue_orders_are_flagged_separately(features, outcomes):
    members = build_snapshot_members(features, outcomes)
    joined = members.merge(features, on="order_id")
    at = pd.to_datetime(joined["snapshot_id"])
    expected = joined["order_estimated_delivery_date"].dt.normalize() < at.dt.normalize()
    assert joined["is_overdue"].to_numpy().tolist() == expected.to_numpy().tolist()


# --------------------------------------------------------------------------
# Ingestion failure modes
# --------------------------------------------------------------------------

def test_missing_file_raises_actionable_error(tmp_path):
    with pytest.raises(DataIntegrityError, match="missing"):
        load_raw(tmp_path)


def test_missing_column_raises_actionable_error(tmp_path):
    for name in spec.SOURCE_FILES.values():
        (tmp_path / name).write_text("wrong_column\n1\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="missing required column"):
        load_raw(tmp_path)


# --------------------------------------------------------------------------
# Training / serving parity
# --------------------------------------------------------------------------

def test_artifact_loads_and_matches_current_feature_schema():
    model, meta = load_artifact(spec.ARTIFACT_DIR)
    assert meta["feature_names"] == list(MODEL_FEATURES)
    assert meta["model_version"]


def test_served_scores_match_the_offline_pipeline(features):
    """The API path and the training path must produce identical numbers."""
    model, _ = load_artifact(spec.ARTIFACT_DIR)
    sample = features[features["split"] == "test"].head(200)

    offline = model.predict_proba(sample[MODEL_FEATURES])[:, 1]
    served = predict_risk(model, sample)
    np.testing.assert_allclose(offline, served, rtol=0, atol=0)


def test_single_row_scoring_matches_batch_scoring(features):
    """Serving scores one order at a time; that must not change the result."""
    model, _ = load_artifact(spec.ARTIFACT_DIR)
    sample = features[features["split"] == "test"].head(25)
    batch = predict_risk(model, sample)
    rows = np.array([predict_risk(model, sample.iloc[[i]])[0] for i in range(len(sample))])
    np.testing.assert_allclose(batch, rows, rtol=1e-12, atol=1e-12)


def test_scoring_rejects_a_missing_feature(features):
    from ml_pipeline.model import ArtifactError

    model, _ = load_artifact(spec.ARTIFACT_DIR)
    broken = features.head(5).drop(columns=["days_handover_to_estimate"])
    with pytest.raises(ArtifactError, match="missing feature"):
        predict_risk(model, broken)


def test_corrupt_artifact_is_rejected(tmp_path):
    """A tampered artifact must fail loudly, never serve a wrong score."""
    import json
    import shutil

    from ml_pipeline.model import ARTIFACT_FILENAME, METADATA_FILENAME, ArtifactError

    shutil.copy(spec.ARTIFACT_DIR / ARTIFACT_FILENAME, tmp_path / ARTIFACT_FILENAME)
    meta = json.loads((spec.ARTIFACT_DIR / METADATA_FILENAME).read_text(encoding="utf-8"))
    meta["artifact_sha256"] = "0" * 64
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ArtifactError, match="checksum mismatch"):
        load_artifact(tmp_path)


def test_missing_artifact_is_rejected(tmp_path):
    from ml_pipeline.model import ArtifactError

    with pytest.raises(ArtifactError, match="not found"):
        load_artifact(tmp_path)
