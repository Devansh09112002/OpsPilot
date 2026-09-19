"""Historical context, computed as of a snapshot date.

This is the only module that reads `order_outcomes`, and it does so under one
hard rule: an outcome counts only if `order_delivered_customer_date < snapshot`.
Those deliveries had already happened at the snapshot moment, so an operator
standing there would genuinely have known them.

The order under investigation is excluded from its own comparison, and the
service refuses to return a rate computed from too few samples rather than
offering a misleadingly precise number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.db.models import OrderFeature, OrderOutcome, Snapshot

MIN_SAMPLE = 30


@dataclass
class ContextResult:
    """A comparison the agent may cite, or an explicit statement that it cannot."""

    available: bool
    label: str
    sample_size: int
    late_rate: float | None = None
    baseline_rate: float | None = None
    baseline_sample: int = 0
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "label": self.label,
            "sample_size": self.sample_size,
            "late_rate": None if self.late_rate is None else round(self.late_rate, 4),
            "baseline_rate": (
                None if self.baseline_rate is None else round(self.baseline_rate, 4)
            ),
            "baseline_sample": self.baseline_sample,
            "caveats": self.caveats,
        }


def _count_and_late(
    db: Session, snapshot_at: datetime, *conditions
) -> tuple[int, float | None]:
    """Count and late-rate over outcomes observable strictly before `snapshot_at`.

    The `<` is the as-of rule; loosening it to `<=` would admit same-day
    deliveries an operator could not yet have seen.
    """
    stmt = (
        select(
            func.count(OrderOutcome.order_id),
            # PostgreSQL will not cast boolean to float; CASE is the portable form.
            func.avg(case((OrderOutcome.is_late, 1.0), else_=0.0)),
        )
        .select_from(OrderFeature)
        .join(OrderOutcome, OrderOutcome.order_id == OrderFeature.order_id)
        .where(OrderOutcome.order_delivered_customer_date < snapshot_at, *conditions)
    )
    n, avg = db.execute(stmt).one()
    return int(n or 0), (float(avg) if avg is not None else None)


def route_context(
    db: Session, snapshot_id: str, order_id: str
) -> ContextResult:
    """Late rate on this order's seller-state to customer-state route.

    Compared against the marketplace-wide rate over the same as-of window.
    """
    snap = db.get(Snapshot, snapshot_id)
    order = db.get(OrderFeature, order_id)
    if snap is None or order is None:
        return ContextResult(
            available=False,
            label="route comparison",
            sample_size=0,
            caveats=["Order or snapshot not found."],
        )
    if not order.seller_state or not order.customer_state:
        return ContextResult(
            available=False,
            label="route comparison",
            sample_size=0,
            caveats=["This order has no recorded seller or customer state."],
        )

    route_label = f"{order.seller_state} to {order.customer_state}"
    n, rate = _count_and_late(
        db, snap.snapshot_at,
        OrderFeature.seller_state == order.seller_state,
        OrderFeature.customer_state == order.customer_state,
        OrderFeature.order_id != order_id,
    )
    base_n, base_rate = _count_and_late(db, snap.snapshot_at)

    caveats = [
        f"Computed only from orders delivered before {snap.snapshot_at:%Y-%m-%d}; "
        "later outcomes are not visible at this snapshot.",
        "Historical association only. It does not explain why this order may be late.",
    ]
    if n < MIN_SAMPLE:
        return ContextResult(
            available=False,
            label=f"late rate on route {route_label}",
            sample_size=n,
            baseline_rate=base_rate,
            baseline_sample=base_n,
            caveats=[
                f"Only {n} comparable orders had been delivered on this route by "
                f"the snapshot date, below the {MIN_SAMPLE}-order minimum, so no "
                "route rate is reported.",
                *caveats,
            ],
        )

    return ContextResult(
        available=True,
        label=f"late rate on route {route_label}",
        sample_size=n,
        late_rate=rate,
        baseline_rate=base_rate,
        baseline_sample=base_n,
        caveats=caveats,
    )


def category_context(db: Session, snapshot_id: str, order_id: str) -> ContextResult:
    """Late rate for this order's product category, as of the snapshot."""
    snap = db.get(Snapshot, snapshot_id)
    order = db.get(OrderFeature, order_id)
    if snap is None or order is None or not order.product_category:
        return ContextResult(
            available=False,
            label="category comparison",
            sample_size=0,
            caveats=["This order has no recorded product category."],
        )

    n, rate = _count_and_late(
        db, snap.snapshot_at,
        OrderFeature.product_category == order.product_category,
        OrderFeature.order_id != order_id,
    )
    base_n, base_rate = _count_and_late(db, snap.snapshot_at)
    caveats = [
        f"Computed only from orders delivered before {snap.snapshot_at:%Y-%m-%d}.",
        "Historical association only, not a cause.",
    ]
    if n < MIN_SAMPLE:
        return ContextResult(
            available=False,
            label=f"late rate for category {order.product_category}",
            sample_size=n,
            baseline_rate=base_rate,
            baseline_sample=base_n,
            caveats=[f"Only {n} comparable orders, below the {MIN_SAMPLE} minimum.", *caveats],
        )
    return ContextResult(
        available=True,
        label=f"late rate for category {order.product_category}",
        sample_size=n,
        late_rate=rate,
        baseline_rate=base_rate,
        baseline_sample=base_n,
        caveats=caveats,
    )
