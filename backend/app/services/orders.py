"""As-of order queries.

Every function here reads `order_features` and `snapshot_orders` only. Neither
table holds a delivery outcome, so no query in this module can return one.
`order_outcomes` is reachable only from `services.analytics`, and only through
a strict as-of cutoff.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.db.models import OrderFeature, Snapshot, SnapshotOrder
from app.schemas.orders import (
    OrderAsOf,
    OrderListItem,
    OrderPage,
    SnapshotStats,
    SnapshotSummary,
)

SortKey = Literal["risk", "deadline", "handover"]


def list_snapshots(db: Session) -> list[SnapshotSummary]:
    rows = db.execute(select(Snapshot).order_by(Snapshot.snapshot_at)).scalars().all()
    return [SnapshotSummary.model_validate(r) for r in rows]


def get_snapshot(db: Session, snapshot_id: str) -> Snapshot:
    snap = db.get(Snapshot, snapshot_id)
    if snap is None:
        raise NotFoundError(
            f"Unknown snapshot '{snapshot_id}'.",
            {"available": [s.snapshot_id for s in db.execute(select(Snapshot)).scalars()]},
        )
    return snap


def snapshot_stats(db: Session, snapshot_id: str) -> SnapshotStats:
    snap = get_snapshot(db, snapshot_id)
    row = db.execute(
        select(
            func.count().filter(SnapshotOrder.risk_band == "high"),
            func.count().filter(SnapshotOrder.risk_band == "medium"),
            func.count().filter(SnapshotOrder.risk_band == "low"),
            func.avg(SnapshotOrder.risk_probability),
            func.min(SnapshotOrder.model_version),
        ).where(
            SnapshotOrder.snapshot_id == snapshot_id,
            SnapshotOrder.is_overdue.is_(False),
        )
    ).one()
    high, medium, low, mean_risk, model_version = row
    return SnapshotStats(
        snapshot_id=snap.snapshot_id,
        label=snap.label,
        snapshot_at=snap.snapshot_at,
        orders_pre_deadline=snap.orders_pre_deadline,
        orders_overdue=snap.orders_overdue,
        high_risk=high or 0,
        medium_risk=medium or 0,
        low_risk=low or 0,
        mean_risk=round(float(mean_risk or 0.0), 4),
        model_version=model_version or "unavailable",
    )


def _days_to_deadline(snapshot_at: datetime, estimated: datetime) -> float:
    return (estimated.date() - snapshot_at.date()).days


def _base_query(snapshot_id: str) -> Select:
    return (
        select(SnapshotOrder, OrderFeature)
        .join(OrderFeature, OrderFeature.order_id == SnapshotOrder.order_id)
        .where(SnapshotOrder.snapshot_id == snapshot_id)
    )


def list_orders(
    db: Session,
    snapshot_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    sort: SortKey = "risk",
    risk_band: str | None = None,
    include_overdue: bool = False,
    customer_state: str | None = None,
    search: str | None = None,
) -> OrderPage:
    """Paginated, filterable risk queue for one snapshot."""
    snap = get_snapshot(db, snapshot_id)
    q = _base_query(snapshot_id)

    if not include_overdue:
        q = q.where(SnapshotOrder.is_overdue.is_(False))
    if risk_band:
        q = q.where(SnapshotOrder.risk_band == risk_band)
    if customer_state:
        q = q.where(OrderFeature.customer_state == customer_state.upper())
    if search:
        # Parameterised LIKE; the value is bound, never interpolated.
        q = q.where(OrderFeature.order_id.like(f"{search.lower()}%"))

    total = db.execute(
        select(func.count()).select_from(q.subquery())
    ).scalar_one()

    # Sort on the raw ranking score, not the calibrated probability. Isotonic
    # calibration creates ties; ordering by the tied values would silently
    # change the ranking every published metric was measured on.
    order_by = {
        "risk": SnapshotOrder.ranking_score.desc(),
        "deadline": OrderFeature.order_estimated_delivery_date.asc(),
        "handover": OrderFeature.order_delivered_carrier_date.desc(),
    }[sort]
    # Tie-break on order_id so pagination is stable.
    rows = db.execute(
        q.order_by(order_by, SnapshotOrder.order_id).limit(limit).offset(offset)
    ).all()

    items = [
        OrderListItem(
            order_id=so.order_id,
            snapshot_id=so.snapshot_id,
            order_estimated_delivery_date=of.order_estimated_delivery_date,
            order_delivered_carrier_date=of.order_delivered_carrier_date,
            days_in_transit=round(so.days_in_transit, 2),
            days_to_deadline=_days_to_deadline(
                snap.snapshot_at, of.order_estimated_delivery_date),
            is_overdue=so.is_overdue,
            risk_probability=round(so.risk_probability, 4),
            ranking_score=round(so.ranking_score, 6),
            risk_band=so.risk_band,
            model_version=so.model_version,
            customer_state=of.customer_state,
            seller_state=of.seller_state,
            product_category=of.product_category,
            n_items=of.n_items,
            total_price=round(of.total_price, 2),
        )
        for so, of in rows
    ]
    return OrderPage(items=items, total=total, limit=limit, offset=offset)


def get_order_as_of(
    db: Session, order_id: str, snapshot_id: str, *, with_factors: bool = False
) -> OrderAsOf:
    """Approved as-of detail for one order. The only order DTO agents receive.

    `with_factors` computes a per-order TreeSHAP attribution. It is off by
    default because the list view would pay for it on every row.
    """
    snap = get_snapshot(db, snapshot_id)
    row = db.execute(
        _base_query(snapshot_id).where(SnapshotOrder.order_id == order_id)
    ).one_or_none()
    if row is None:
        raise NotFoundError(
            f"Order '{order_id}' is not part of snapshot '{snapshot_id}'.",
            {"order_id": order_id, "snapshot_id": snapshot_id},
        )
    so, of = row
    f = of.features or {}

    factors: list = []
    calibrated = True
    from app.ml.predictor import predictor

    if predictor.available:
        calibrated = predictor.calibrated
        if with_factors:
            try:
                factors = predictor.predict_one(f, with_factors=True).factors
            except Exception:
                factors = []

    return OrderAsOf(
        order_id=of.order_id,
        snapshot_id=snapshot_id,
        snapshot_at=snap.snapshot_at,
        order_purchase_timestamp=of.order_purchase_timestamp,
        order_approved_at=of.order_approved_at,
        order_delivered_carrier_date=of.order_delivered_carrier_date,
        order_estimated_delivery_date=of.order_estimated_delivery_date,
        days_in_transit=round(so.days_in_transit, 2),
        days_to_deadline=_days_to_deadline(snap.snapshot_at, of.order_estimated_delivery_date),
        is_overdue=so.is_overdue,
        n_items=of.n_items,
        n_distinct_sellers=int(f.get("n_distinct_sellers") or 1),
        total_price=round(of.total_price, 2),
        total_freight=round(of.total_freight, 2),
        product_category=of.product_category,
        customer_state=of.customer_state,
        seller_state=of.seller_state,
        is_cross_state=bool(f.get("is_cross_state")),
        risk_probability=round(so.risk_probability, 4),
        ranking_score=round(so.ranking_score, 6),
        risk_band=so.risk_band,
        model_version=so.model_version,
        calibrated=calibrated,
        risk_factors=factors,
        prediction_as_of=of.order_delivered_carrier_date,
    )


def get_feature_row(db: Session, order_id: str) -> OrderFeature:
    row = db.get(OrderFeature, order_id)
    if row is None:
        raise NotFoundError(f"Order '{order_id}' not found.", {"order_id": order_id})
    return row
