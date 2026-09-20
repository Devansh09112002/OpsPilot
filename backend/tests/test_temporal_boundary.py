"""The as-of cutoff, tested at the boundary itself.

OpsPilot's central honesty claim is that neither the user nor the agent sees
how an order turned out. Historical context is the one place outcomes are read
at all, and it is allowed only for deliveries strictly earlier than the
snapshot: `order_delivered_customer_date < snapshot_at`.

A mutation audit showed that relaxing that `<` to `<=` broke nothing in the
suite. The rule the architecture calls "the whole rule" was unverified. These
tests exercise the exact instant on both sides of it, so the boundary cannot
move silently again.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from app.db.models import OrderFeature, OrderOutcome, Snapshot
from app.services import analytics

SNAPSHOT = "2018-08-15"
LANE = ("ZZ", "ZY")  # a lane no real order uses, so counts are unambiguous


@pytest.fixture
def boundary_lane(db):
    """Seed one lane with deliveries placed exactly around the cutoff.

    Three orders, all on a synthetic lane:
      - `before`: delivered one second before the snapshot  -> visible
      - `exact` : delivered at the snapshot instant          -> must NOT be visible
      - `after` : delivered one second after                 -> must NOT be visible

    `MIN_SAMPLE` would normally suppress a rate computed from three orders, so
    the tests below assert on the sample size, which is what the cutoff
    actually controls.
    """
    snapshot = db.get(Snapshot, SNAPSHOT)
    assert snapshot is not None
    at = snapshot.snapshot_at

    seeded = {
        "before": at - timedelta(seconds=1),
        "exact": at,
        "after": at + timedelta(seconds=1),
    }
    for name, delivered in seeded.items():
        order_id = f"boundary{name}".ljust(32, "0")[:32]
        db.add(OrderFeature(
            order_id=order_id,
            split="test",
            order_purchase_timestamp=at - timedelta(days=20),
            order_approved_at=at - timedelta(days=19),
            order_delivered_carrier_date=at - timedelta(days=18),
            order_estimated_delivery_date=at + timedelta(days=5),
            features=None,
            customer_state=LANE[1],
            seller_state=LANE[0],
            product_category="boundary_fixture",
            n_items=1,
            total_price=10.0,
            total_freight=1.0,
        ))
        db.add(OrderOutcome(
            order_id=order_id,
            order_delivered_customer_date=delivered,
            is_late=True,
        ))
    db.flush()
    yield at
    for name in seeded:
        order_id = f"boundary{name}".ljust(32, "0")[:32]
        outcome = db.get(OrderOutcome, order_id)
        if outcome is not None:
            db.delete(outcome)
        feature = db.get(OrderFeature, order_id)
        if feature is not None:
            db.delete(feature)
    db.flush()


def test_only_deliveries_strictly_before_the_snapshot_are_counted(db, boundary_lane):
    """One of three synthetic deliveries is observable, not two and not three.

    This is the test the `<=` mutation has to fail. With `<=` the delivery at
    the snapshot instant becomes visible and the count is 2.
    """
    context = analytics.lane_context(db, SNAPSHOT, LANE[0], LANE[1])
    assert context.sample_size == 1, (
        "expected only the delivery one second before the snapshot to be "
        f"observable, got {context.sample_size}"
    )


def test_a_delivery_at_the_exact_snapshot_instant_is_not_observable(db, boundary_lane):
    """Stated separately because it is the specific case `<=` would admit.

    An operator standing at the snapshot moment has not yet seen a delivery
    recorded at that same moment.
    """
    at = boundary_lane
    exact_id = "boundaryexact".ljust(32, "0")[:32]
    outcome = db.get(OrderOutcome, exact_id)
    assert outcome.order_delivered_customer_date == at

    before = analytics.lane_context(db, SNAPSHOT, LANE[0], LANE[1]).sample_size

    # Move that same delivery one second earlier: it must now be counted.
    outcome.order_delivered_customer_date = at - timedelta(seconds=1)
    db.flush()
    after = analytics.lane_context(db, SNAPSHOT, LANE[0], LANE[1]).sample_size

    assert after == before + 1, (
        "moving a delivery from the snapshot instant to one second earlier "
        "must change whether it is observable"
    )


def test_the_route_context_shares_the_same_cutoff(db, boundary_lane):
    """The order-level path must not be looser than the lane-level one."""
    before_id = "boundarybefore".ljust(32, "0")[:32]
    context = analytics.route_context(db, SNAPSHOT, before_id)
    # The subject order is excluded from its own comparison, so the two
    # remaining synthetic orders (at and after the instant) are both invisible.
    assert context.sample_size == 0


def test_the_baseline_excludes_the_boundary_delivery_too(db, boundary_lane):
    """The marketplace baseline is computed by the same helper and must agree."""
    at = boundary_lane
    marketplace_before = analytics.lane_context(
        db, SNAPSHOT, LANE[0], LANE[1]
    ).baseline_sample

    exact_id = "boundaryexact".ljust(32, "0")[:32]
    db.get(OrderOutcome, exact_id).order_delivered_customer_date = at - timedelta(seconds=1)
    db.flush()
    marketplace_after = analytics.lane_context(
        db, SNAPSHOT, LANE[0], LANE[1]
    ).baseline_sample

    assert marketplace_after == marketplace_before + 1


def test_no_snapshot_order_was_delivered_before_its_own_snapshot(db):
    """A prediction is only meaningful for an order still in transit.

    If a snapshot contained an order already delivered at that moment, the
    product would be scoring a settled outcome and calling it a forecast.
    """
    rows = db.execute(
        select(Snapshot.snapshot_id, OrderOutcome.order_id)
        .select_from(Snapshot)
        .join(OrderFeature, OrderFeature.order_delivered_carrier_date < Snapshot.snapshot_at)
        .join(OrderOutcome, OrderOutcome.order_id == OrderFeature.order_id)
        .where(OrderOutcome.order_delivered_customer_date < Snapshot.snapshot_at)
        .limit(5)
    ).all()
    # Orders delivered before a snapshot exist in the dataset; what must not
    # happen is that any of them is a *member* of that snapshot.
    from app.db.models import SnapshotOrder

    leaked = db.execute(
        select(SnapshotOrder.snapshot_id, SnapshotOrder.order_id)
        .join(OrderOutcome, OrderOutcome.order_id == SnapshotOrder.order_id)
        .join(Snapshot, Snapshot.snapshot_id == SnapshotOrder.snapshot_id)
        .where(OrderOutcome.order_delivered_customer_date < Snapshot.snapshot_at)
        .limit(5)
    ).all()
    assert not leaked, (
        "a snapshot contains orders that had already been delivered at that "
        f"moment: {leaked}"
    )
    assert rows is not None  # the join above is only a sanity construction
