"""Lane situations: a snapshot's flagged orders grouped into decisions.

A reviewer looking at the 2018-08-15 snapshot faces 395 flagged orders and, in
v1, one action: open them one at a time. Grouped by lane those 395 orders are
37 lanes, five of which hold 73% of them. The situation is the unit a person
can actually act on, and the unit an escalation is actually about - you raise a
lane with a carrier, not an order.

Two properties this module is careful about:

**It reads no outcome.** Like `services.orders`, every query here touches only
`order_features` and `snapshot_orders`. Outcome-derived lane history is
available, but only through `analytics.lane_context`, which applies the strict
as-of cutoff.

**Its ranking quantity is legitimate only because the scores are calibrated.**
`expected_late` is the sum of member probabilities. Summing raw model output
would produce a number with no units and no meaning, since `scale_pos_weight`
inflates every raw score. Calibration is what makes addition defensible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.db.models import OrderFeature, SnapshotOrder
from app.services.orders import get_snapshot

# Bands that put an order in front of a reviewer at all. Kept in sync with the
# queue's default view: a "low" order is not a pending decision.
FLAGGED_BANDS = ("high", "medium")

# Below this, a lane is a handful of unrelated orders rather than a pattern,
# and grouping them would invent a situation that is not there. Those orders
# remain individually reviewable in the risk queue.
MIN_SITUATION_ORDERS = 3

# Brazilian state codes: exactly two letters. Situation ids are parsed back
# into query parameters, so the shape is validated rather than trusted.
_STATE = re.compile(r"^[A-Z]{2}$")
_SITUATION_ID = re.compile(r"^(?P<snapshot>[0-9]{4}-[0-9]{2}-[0-9]{2})__(?P<seller>[A-Z]{2})-(?P<customer>[A-Z]{2})$")


@dataclass
class SituationMember:
    order_id: str
    risk_probability: float
    risk_band: str
    days_in_transit: float | None
    product_category: str | None


@dataclass
class Situation:
    """One lane's flagged orders in one snapshot."""

    situation_id: str
    snapshot_id: str
    seller_state: str
    customer_state: str
    lane: str
    n_flagged: int
    n_high: int
    n_lane_total: int
    expected_late: float
    mean_risk: float
    max_risk: float
    model_version: str | None
    members: list[SituationMember] = field(default_factory=list)

    @property
    def share_of_lane(self) -> float:
        return self.n_flagged / self.n_lane_total if self.n_lane_total else 0.0

    def as_dict(self, with_members: bool = True) -> dict:
        payload = {
            "situation_id": self.situation_id,
            "snapshot_id": self.snapshot_id,
            "seller_state": self.seller_state,
            "customer_state": self.customer_state,
            "lane": self.lane,
            "n_flagged": self.n_flagged,
            "n_high": self.n_high,
            "n_lane_total": self.n_lane_total,
            "share_of_lane": round(self.share_of_lane, 4),
            "expected_late": round(self.expected_late, 3),
            "mean_risk": round(self.mean_risk, 4),
            "max_risk": round(self.max_risk, 4),
            "model_version": self.model_version,
        }
        if with_members:
            payload["members"] = [
                {
                    "order_id": m.order_id,
                    "risk_probability": round(m.risk_probability, 4),
                    "risk_band": m.risk_band,
                    "days_in_transit": m.days_in_transit,
                    "product_category": m.product_category,
                }
                for m in self.members
            ]
        return payload


def make_situation_id(snapshot_id: str, seller_state: str, customer_state: str) -> str:
    return f"{snapshot_id}__{seller_state}-{customer_state}"


def parse_situation_id(situation_id: str) -> tuple[str, str, str]:
    """Split a situation id, rejecting anything that is not one.

    The parts become query parameters. They are bound, never interpolated, but
    validating the shape here means a malformed id is a clean 404 rather than
    an empty result that reads like a real but empty situation.
    """
    match = _SITUATION_ID.match(situation_id or "")
    if match is None:
        raise NotFoundError(
            f"'{situation_id}' is not a valid situation id.",
            {"expected": "YYYY-MM-DD__XX-YY"},
        )
    return match["snapshot"], match["seller"], match["customer"]


def _flagged_filter():
    return (
        SnapshotOrder.risk_band.in_(FLAGGED_BANDS),
        SnapshotOrder.is_overdue.is_(False),
    )


def list_situations(
    db: Session,
    snapshot_id: str,
    *,
    limit: int = 20,
    min_orders: int = MIN_SITUATION_ORDERS,
) -> list[Situation]:
    """Lanes of this snapshot that carry flagged orders, worst risk first.

    Ranked by expected late orders rather than by count, so a small lane of
    near-certain problems outranks a large lane of marginal ones.
    """
    get_snapshot(db, snapshot_id)

    lane_totals = dict(
        db.execute(
            select(
                func.concat(OrderFeature.seller_state, "-", OrderFeature.customer_state),
                func.count(),
            )
            .select_from(SnapshotOrder)
            .join(OrderFeature, OrderFeature.order_id == SnapshotOrder.order_id)
            .where(SnapshotOrder.snapshot_id == snapshot_id)
            .group_by(OrderFeature.seller_state, OrderFeature.customer_state)
        ).all()
    )

    rows = db.execute(
        select(
            OrderFeature.seller_state,
            OrderFeature.customer_state,
            func.count(),
            func.sum(SnapshotOrder.risk_probability),
            func.avg(SnapshotOrder.risk_probability),
            func.max(SnapshotOrder.risk_probability),
            func.count().filter(SnapshotOrder.risk_band == "high"),
            func.min(SnapshotOrder.model_version),
        )
        .select_from(SnapshotOrder)
        .join(OrderFeature, OrderFeature.order_id == SnapshotOrder.order_id)
        .where(SnapshotOrder.snapshot_id == snapshot_id, *_flagged_filter())
        .group_by(OrderFeature.seller_state, OrderFeature.customer_state)
        .having(func.count() >= min_orders)
        .order_by(func.sum(SnapshotOrder.risk_probability).desc())
        .limit(limit)
    ).all()

    situations: list[Situation] = []
    for seller, customer, n, total, mean, mx, n_high, version in rows:
        if not seller or not customer:
            continue
        situations.append(
            Situation(
                situation_id=make_situation_id(snapshot_id, seller, customer),
                snapshot_id=snapshot_id,
                seller_state=seller,
                customer_state=customer,
                lane=f"{seller} to {customer}",
                n_flagged=int(n),
                n_high=int(n_high or 0),
                n_lane_total=int(lane_totals.get(f"{seller}-{customer}", n)),
                expected_late=float(total or 0.0),
                mean_risk=float(mean or 0.0),
                max_risk=float(mx or 0.0),
                model_version=version,
            )
        )
    return situations


def get_situation(db: Session, situation_id: str) -> Situation:
    """One situation with its member orders, highest risk first."""
    snapshot_id, seller, customer = parse_situation_id(situation_id)
    if not (_STATE.match(seller) and _STATE.match(customer)):
        raise NotFoundError(f"'{situation_id}' is not a valid situation id.")
    get_snapshot(db, snapshot_id)

    rows = db.execute(
        select(
            SnapshotOrder.order_id,
            SnapshotOrder.risk_probability,
            SnapshotOrder.risk_band,
            SnapshotOrder.days_in_transit,
            SnapshotOrder.model_version,
            OrderFeature.product_category,
        )
        .select_from(SnapshotOrder)
        .join(OrderFeature, OrderFeature.order_id == SnapshotOrder.order_id)
        .where(
            SnapshotOrder.snapshot_id == snapshot_id,
            OrderFeature.seller_state == seller,
            OrderFeature.customer_state == customer,
            *_flagged_filter(),
        )
        .order_by(SnapshotOrder.ranking_score.desc())
    ).all()

    if not rows:
        raise NotFoundError(
            f"No flagged orders on lane {seller} to {customer} in snapshot "
            f"{snapshot_id}.",
        )

    lane_total = db.execute(
        select(func.count())
        .select_from(SnapshotOrder)
        .join(OrderFeature, OrderFeature.order_id == SnapshotOrder.order_id)
        .where(
            SnapshotOrder.snapshot_id == snapshot_id,
            OrderFeature.seller_state == seller,
            OrderFeature.customer_state == customer,
        )
    ).scalar_one()

    members = [
        SituationMember(
            order_id=r.order_id,
            risk_probability=float(r.risk_probability),
            risk_band=r.risk_band,
            days_in_transit=r.days_in_transit,
            product_category=r.product_category,
        )
        for r in rows
    ]
    probabilities = [m.risk_probability for m in members]
    return Situation(
        situation_id=make_situation_id(snapshot_id, seller, customer),
        snapshot_id=snapshot_id,
        seller_state=seller,
        customer_state=customer,
        lane=f"{seller} to {customer}",
        n_flagged=len(members),
        n_high=sum(1 for m in members if m.risk_band == "high"),
        n_lane_total=int(lane_total),
        expected_late=sum(probabilities),
        mean_risk=sum(probabilities) / len(probabilities),
        max_risk=max(probabilities),
        model_version=rows[0].model_version,
        members=members,
    )
