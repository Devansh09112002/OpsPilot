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
from app.db.models import OrderFeature, Snapshot, SnapshotOrder
from app.services import policies
from app.services.orders import _days_to_deadline, get_snapshot

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
    product_category: str | None
    days_to_deadline: float
    escalatable: bool


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
    n_escalatable: int = 0
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
            "n_escalatable": self.n_escalatable,
        }
        if with_members:
            payload["members"] = [
                {
                    "order_id": m.order_id,
                    "risk_probability": round(m.risk_probability, 4),
                    "risk_band": m.risk_band,
                    "product_category": m.product_category,
                    "days_to_deadline": m.days_to_deadline,
                    "escalatable": m.escalatable,
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


def _deadline_days():
    """Days from the snapshot to the promised date, as a SQL expression.

    Calendar-date difference, matching `orders._days_to_deadline` exactly: the
    target rule is stated on dates, so the policy window must be too.
    """
    return func.date_part(
        "day",
        func.date_trunc("day", OrderFeature.order_estimated_delivery_date)
        - func.date_trunc("day", Snapshot.snapshot_at),
    )


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
            # ESC-01 per member, expressed in SQL so the list view does not
            # have to load every member row to show a decision-relevant count.
            func.count().filter(
                SnapshotOrder.risk_probability >= policies.ESCALATION_RISK_THRESHOLD,
                _deadline_days() <= policies.ESCALATION_SLACK_DAYS,
            ),
        )
        .select_from(SnapshotOrder)
        .join(OrderFeature, OrderFeature.order_id == SnapshotOrder.order_id)
        .join(Snapshot, Snapshot.snapshot_id == SnapshotOrder.snapshot_id)
        .where(SnapshotOrder.snapshot_id == snapshot_id, *_flagged_filter())
        .group_by(OrderFeature.seller_state, OrderFeature.customer_state)
        .having(func.count() >= min_orders)
        .order_by(func.sum(SnapshotOrder.risk_probability).desc())
        .limit(limit)
    ).all()

    situations: list[Situation] = []
    for seller, customer, n, total, mean, mx, n_high, version, n_esc in rows:
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
                n_escalatable=int(n_esc or 0),
            )
        )
    return situations


def get_situation(db: Session, situation_id: str) -> Situation:
    """One situation with its member orders, highest risk first."""
    snapshot_id, seller, customer = parse_situation_id(situation_id)
    if not (_STATE.match(seller) and _STATE.match(customer)):
        raise NotFoundError(f"'{situation_id}' is not a valid situation id.")
    snap = get_snapshot(db, snapshot_id)

    rows = db.execute(
        select(
            SnapshotOrder.order_id,
            SnapshotOrder.risk_probability,
            SnapshotOrder.risk_band,
            SnapshotOrder.model_version,
            OrderFeature.product_category,
            OrderFeature.order_estimated_delivery_date,
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

    members = []
    for r in rows:
        days_to_deadline = _days_to_deadline(
            snap.snapshot_at, r.order_estimated_delivery_date
        )
        # A member is escalatable only if the backend's own per-order policy
        # reading says so. ESC-05 counts these; it does not re-derive them.
        permitted, _ = policies.escalation_permitted(
            risk_probability=float(r.risk_probability),
            days_to_deadline=days_to_deadline,
            is_overdue=False,
        )
        members.append(SituationMember(
            order_id=r.order_id,
            risk_probability=float(r.risk_probability),
            risk_band=r.risk_band,
            product_category=r.product_category,
            days_to_deadline=days_to_deadline,
            escalatable=permitted,
        ))
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
        n_escalatable=sum(1 for m in members if m.escalatable),
        members=members,
    )
