"""API v1 routes.

The endpoint inventory is deliberately narrow and matches
`docs/api_contracts.md` exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.errors import RateLimitError, ServiceUnavailableError
from app.core.limits import (
    enforce_investigation_budget,
    record_investigation_spend,
    request_limiter,
)
from app.core.logging import get_logger
from app.core.sessions import get_guest_session
from app.db.models import GuestSession
from app.db.session import get_db
from app.ml.predictor import prediction_computed_at, predictor
from app.schemas.investigations import (
    AuditEventOut,
    DecisionResponse,
    InvestigationOut,
    InvestigationRequest,
    TicketListOut,
    TicketOut,
)
from app.schemas.orders import (
    OrderAsOf,
    OrderPage,
    PredictionRequest,
    PredictionResponse,
    SnapshotStats,
    SnapshotSummary,
)
from app.schemas.situations import (
    LaneHistoryOut,
    SituationDetail,
    SituationSummary,
)
from app.services import analytics as analytics_service
from app.services import investigations as investigation_service
from app.services import orders as order_service
from app.services import policies as policy_service
from app.services import proposals as proposal_service
from app.services import situations as situation_service

log = get_logger(__name__)
router = APIRouter()


def rate_limit(request: Request) -> None:
    """Cheap per-client request cap, applied before any real work."""
    client = request.client.host if request.client else "unknown"
    ok, retry_after = request_limiter.check(client)
    if not ok:
        raise RateLimitError(
            "Too many requests. Please slow down.",
            {"retry_after_seconds": round(retry_after, 1)},
        )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["ops"], summary="Liveness")
def health() -> dict:
    return {"status": "ok", "time": datetime.now(UTC).isoformat()}


@router.get("/health/ready", tags=["ops"], summary="Readiness with dependencies")
def readiness(db: Session = Depends(get_db)) -> dict:
    """Reports each dependency separately so a partial outage is visible.

    The service is 'degraded' rather than down when only the LLM is missing:
    the risk queue and predictions still work.
    """
    from app.core.config import get_settings

    checks: dict[str, dict] = {}

    try:
        db.execute(text("select 1"))
        checks["database"] = {"ok": True}
    except Exception as exc:
        checks["database"] = {"ok": False, "error": type(exc).__name__}

    checks["model"] = (
        {"ok": True, "model_version": predictor.model_version}
        if predictor.available
        else {"ok": False, "error": predictor.error or "artifact not loaded"}
    )

    try:
        checks["policy"] = {"ok": True, "version": policy_service.policy_version()}
    except Exception as exc:
        checks["policy"] = {"ok": False, "error": str(exc)}

    settings = get_settings()
    checks["llm"] = {"ok": settings.llm_configured, "configured": settings.llm_configured}

    critical_ok = checks["database"]["ok"] and checks["model"]["ok"]
    status = "ok" if critical_ok and checks["llm"]["ok"] else (
        "degraded" if critical_ok else "unavailable"
    )
    return {"status": status, "checks": checks}


# ---------------------------------------------------------------------------
# Snapshots and orders
# ---------------------------------------------------------------------------

@router.get("/snapshots", response_model=list[SnapshotSummary], tags=["orders"])
def list_snapshots(
    _: None = Depends(rate_limit), db: Session = Depends(get_db)
) -> list[SnapshotSummary]:
    return order_service.list_snapshots(db)


@router.get("/snapshots/{snapshot_id}/stats", response_model=SnapshotStats, tags=["orders"])
def snapshot_stats(
    snapshot_id: str, _: None = Depends(rate_limit), db: Session = Depends(get_db)
) -> SnapshotStats:
    return order_service.snapshot_stats(db, snapshot_id)


@router.get(
    "/snapshots/{snapshot_id}/situations",
    response_model=list[SituationSummary],
    tags=["situations"],
    summary="Lanes carrying flagged orders, worst expected-late first",
)
def list_situations(
    snapshot_id: str,
    limit: int = Query(20, ge=1, le=100),
    min_orders: int = Query(
        situation_service.MIN_SITUATION_ORDERS,
        ge=2,
        le=100,
        description="Below this, a lane is a handful of orders rather than a pattern.",
    ),
    _: None = Depends(rate_limit),
    db: Session = Depends(get_db),
) -> list[SituationSummary]:
    return [
        SituationSummary(**s.as_dict(with_members=False))
        for s in situation_service.list_situations(
            db, snapshot_id, limit=limit, min_orders=min_orders
        )
    ]


@router.get(
    "/situations/{situation_id}",
    response_model=SituationDetail,
    tags=["situations"],
    summary="One lane situation with its member orders",
)
def get_situation(
    situation_id: str,
    _: None = Depends(rate_limit),
    db: Session = Depends(get_db),
) -> SituationDetail:
    situation = situation_service.get_situation(db, situation_id)
    history = analytics_service.lane_context(
        db, situation.snapshot_id, situation.seller_state, situation.customer_state
    )
    return SituationDetail(
        **situation.as_dict(with_members=True),
        lane_history=LaneHistoryOut(**history.as_dict()),
    )


@router.get("/orders", response_model=OrderPage, tags=["orders"])
def list_orders(
    snapshot_id: str = Query(..., min_length=1, max_length=16),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: Literal["risk", "deadline", "handover"] = "risk",
    risk_band_filter: Literal["low", "medium", "high"] | None = Query(None, alias="risk_band"),
    include_overdue: bool = False,
    customer_state: str | None = Query(None, max_length=2),
    search: str | None = Query(None, max_length=64),
    _: None = Depends(rate_limit),
    db: Session = Depends(get_db),
) -> OrderPage:
    return order_service.list_orders(
        db, snapshot_id, limit=limit, offset=offset, sort=sort,
        risk_band=risk_band_filter, include_overdue=include_overdue,
        customer_state=customer_state, search=search,
    )


@router.get("/orders/{order_id}", response_model=OrderAsOf, tags=["orders"])
def get_order(
    order_id: str,
    snapshot_id: str = Query(..., min_length=1, max_length=16),
    _: None = Depends(rate_limit),
    db: Session = Depends(get_db),
) -> OrderAsOf:
    return order_service.get_order_as_of(db, order_id, snapshot_id, with_factors=True)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

@router.post("/predictions", response_model=PredictionResponse, tags=["ml"])
def create_prediction(
    payload: PredictionRequest,
    _: None = Depends(rate_limit),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Serve a live prediction from the loaded artifact.

    Returns 503 when the artifact is unavailable. It never falls back to a
    stored or synthetic value.
    """
    if not predictor.available:
        raise ServiceUnavailableError(
            "The delivery-risk model is not available, so no prediction can be "
            "served. Order data remains accessible.",
            {"reason": predictor.error or "artifact not loaded"},
        )

    order = order_service.get_order_as_of(db, payload.order_id, payload.snapshot_id)
    feature_row = order_service.get_feature_row(db, payload.order_id)
    try:
        prediction = predictor.predict_one(
            feature_row.features or {}, with_factors=True
        )
    except Exception as exc:
        raise ServiceUnavailableError(
            "The model could not score this order.", {"reason": type(exc).__name__}
        ) from exc

    return PredictionResponse(
        order_id=payload.order_id,
        snapshot_id=payload.snapshot_id,
        risk_probability=round(prediction.probability, 4),
        ranking_score=round(prediction.ranking_score, 6),
        risk_band=prediction.band,
        model_version=predictor.model_version or "unknown",
        calibrated=prediction.calibrated,
        risk_factors=prediction.factors,
        prediction_as_of=order.order_delivered_carrier_date,
        computed_at=prediction_computed_at(),
    )


# ---------------------------------------------------------------------------
# Investigations
# ---------------------------------------------------------------------------

@router.post("/investigations", response_model=InvestigationOut, tags=["agent"])
def start_investigation(
    payload: InvestigationRequest,
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> InvestigationOut:
    """Run one bounded investigation synchronously.

    Synchronous is the right call here: the workflow makes a single LLM call
    with a hard timeout, so a job queue would add moving parts without
    changing what the user waits for.
    """
    # Validate the order exists in the snapshot before spending any budget.
    order_service.get_order_as_of(db, payload.order_id, payload.snapshot_id)

    enforce_investigation_budget(db, session)
    record_investigation_spend(session)
    db.commit()

    inv = investigation_service.run_and_persist(
        db,
        session_id=session.session_id,
        order_id=payload.order_id,
        snapshot_id=payload.snapshot_id,
    )
    return investigation_service.to_schema(db, inv)


@router.get("/investigations/{investigation_id}", response_model=InvestigationOut, tags=["agent"])
def get_investigation(
    investigation_id: str,
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> InvestigationOut:
    inv = investigation_service.get_investigation(db, investigation_id, session.session_id)
    return investigation_service.to_schema(db, inv)


@router.get("/investigations", response_model=list[InvestigationOut], tags=["agent"])
def list_investigations(
    limit: int = Query(25, ge=1, le=100),
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> list[InvestigationOut]:
    rows = investigation_service.list_investigations(db, session.session_id, limit)
    return [investigation_service.to_schema(db, r) for r in rows]


# ---------------------------------------------------------------------------
# Proposals: the human decision
# ---------------------------------------------------------------------------

@router.post("/proposals/{proposal_id}/approve", response_model=DecisionResponse, tags=["action"])
def approve(
    proposal_id: str,
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> DecisionResponse:
    return proposal_service.approve_proposal(db, proposal_id, session.session_id)


@router.post("/proposals/{proposal_id}/reject", response_model=DecisionResponse, tags=["action"])
def reject(
    proposal_id: str,
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> DecisionResponse:
    return proposal_service.reject_proposal(db, proposal_id, session.session_id)


# ---------------------------------------------------------------------------
# Tickets and audit
# ---------------------------------------------------------------------------

@router.get("/tickets", response_model=TicketListOut, tags=["action"])
def list_tickets(
    limit: int = Query(50, ge=1, le=200),
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> TicketListOut:
    rows = proposal_service.list_tickets(db, session.session_id, limit)
    return TicketListOut(
        items=[TicketOut.model_validate(t) for t in rows], total=len(rows)
    )


@router.get("/tickets/{ticket_id}", response_model=TicketOut, tags=["action"])
def get_ticket(
    ticket_id: str,
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> TicketOut:
    return TicketOut.model_validate(
        proposal_service.get_ticket(db, ticket_id, session.session_id)
    )


@router.get("/audit", response_model=list[AuditEventOut], tags=["action"])
def list_audit(
    limit: int = Query(100, ge=1, le=200),
    _: None = Depends(rate_limit),
    session: GuestSession = Depends(get_guest_session),
    db: Session = Depends(get_db),
) -> list[AuditEventOut]:
    rows = proposal_service.list_audit_events(db, session.session_id, limit)
    return [AuditEventOut.model_validate(r) for r in rows]


@router.get("/meta", tags=["ops"], summary="Model and policy versions")
def meta() -> dict:
    """What the UI displays in its footer, so claims are always version-labelled."""
    from app.core.config import get_settings

    meta_doc = predictor.metadata or {}
    return {
        "model_version": predictor.model_version,
        "model_family": meta_doc.get("model_family"),
        "model_available": predictor.available,
        "calibrated": predictor.calibrated,
        "calibration_method": meta_doc.get("calibration_method") or None,
        "band_thresholds": predictor.band_thresholds,
        "training_cutoff": meta_doc.get("training_cutoff"),
        "policy_version": policy_service.policy_version(),
        "llm_configured": get_settings().llm_configured,
        "dataset": {
            "name": "Olist Brazilian E-Commerce Public Dataset",
            "url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "license": "CC BY-NC-SA 4.0",
        },
    }
