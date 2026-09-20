"""The risk band shown to a user and the policy that acts on it.

These are two numbers that must be one number. Kept apart, they drift, and the
drift is invisible until an order reads "medium risk" next to an escalation.
"""

from __future__ import annotations

import pytest


def test_high_band_equals_the_policy_escalation_threshold():
    """The displayed band and the policy must never disagree.

    When these were derived independently the band edge landed at 0.1659 and
    the policy at 0.15, so an order could read as "medium risk" and still be
    escalated. The agent benchmark caught it as a failure. Both now read one
    constant, and this test is what keeps them reading it.
    """
    from app.ml.predictor import predictor
    from app.services import policies
    from data_pipeline import spec

    assert policies.ESCALATION_RISK_THRESHOLD == spec.ESCALATION_THRESHOLD
    assert predictor.band_thresholds["high"] == spec.ESCALATION_THRESHOLD
    assert predictor.band_thresholds["medium"] <= spec.ESCALATION_THRESHOLD


def test_no_medium_band_order_clears_the_escalation_threshold(db):
    """The invariant as the product actually stores it, not just in config."""
    from sqlalchemy import func, select

    from app.db.models import SnapshotOrder
    from app.services import policies

    highest_medium = db.execute(
        select(func.max(SnapshotOrder.risk_probability))
        .where(SnapshotOrder.risk_band == "medium")
    ).scalar()
    if highest_medium is None:
        pytest.skip("no medium-band orders in the seeded slice")
    assert highest_medium < policies.ESCALATION_RISK_THRESHOLD
