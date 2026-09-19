"""Investigation orchestration and persistence.

The rule this module exists to enforce: **a proposal is created only by a
completed, verified investigation that recommends escalation and that the
backend's own policy reading permits.** A failed or insufficient-evidence run
persists its honest status and creates nothing.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.graph import InvestigationState, run_investigation
from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.models import (
    Investigation,
    InvestigationStatus,
    Proposal,
    ProposalStatus,
    Ticket,
)
from app.schemas.investigations import (
    EvidenceItem,
    FactItem,
    InvestigationOut,
    ProposalOut,
)
from app.services.proposals import new_id, record_audit

log = get_logger(__name__)

MAX_ACTION_LENGTH = 300


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _persist_evidence(state: InvestigationState) -> list[dict]:
    return [e.model_dump() for e in state.evidence]


def run_and_persist(
    db: Session, *, session_id: str, order_id: str, snapshot_id: str
) -> Investigation:
    """Run one investigation to completion and store the outcome."""
    started = time.perf_counter()
    investigation = Investigation(
        investigation_id=new_id("inv"),
        session_id=session_id,
        order_id=order_id,
        snapshot_id=snapshot_id,
        status=InvestigationStatus.running,
        created_at=_now(),
    )
    db.add(investigation)
    db.flush()

    try:
        state = run_investigation(db, order_id, snapshot_id)
    except Exception as exc:
        log.error("investigation_crashed", order_id=order_id, exc_info=exc)
        investigation.status = InvestigationStatus.failed
        investigation.error_message = (
            "The investigation could not be completed. Order data and the model "
            "prediction remain available."
        )
        investigation.completed_at = _now()
        investigation.duration_ms = int((time.perf_counter() - started) * 1000)
        record_audit(
            db, session_id=session_id, event_type="investigation_failed",
            subject_type="investigation", subject_id=investigation.investigation_id,
            detail={"order_id": order_id, "reason": "exception"},
        )
        db.commit()
        return investigation

    investigation.risk_probability = state.risk_probability
    investigation.model_version = state.model_version
    investigation.prediction_as_of = (
        datetime.fromisoformat(state.prediction_as_of)
        if isinstance(state.prediction_as_of, str) else state.prediction_as_of
    )
    investigation.tool_trace = state.trace
    investigation.llm_input_tokens = state.llm_input_tokens
    investigation.llm_output_tokens = state.llm_output_tokens
    investigation.duration_ms = int((time.perf_counter() - started) * 1000)
    investigation.completed_at = _now()

    if state.status == "completed" and state.report is not None:
        investigation.status = InvestigationStatus.completed
        investigation.report = {
            "summary": state.report.summary,
            "facts": [f.model_dump() for f in state.report.facts],
            "limitations": state.report.limitations,
            "recommendation": state.report.recommendation,
            "recommendation_rationale": state.report.recommendation_rationale,
            "proposed_action": state.report.proposed_action,
            "evidence": _persist_evidence(state),
        }
    elif state.status == "insufficient_evidence":
        investigation.status = InvestigationStatus.insufficient_evidence
        investigation.error_message = state.error_message
        investigation.report = {
            "summary": None,
            "facts": [],
            "limitations": state.failures,
            "recommendation": None,
            "evidence": _persist_evidence(state),
        }
    else:
        investigation.status = InvestigationStatus.failed
        investigation.error_message = state.error_message or "The investigation failed."
        investigation.report = {"evidence": _persist_evidence(state),
                                "limitations": state.failures}

    # --- proposal creation, the only path that exists ---------------------
    if (
        investigation.status is InvestigationStatus.completed
        and state.report is not None
        and state.report.recommendation == "propose_escalation"
        and state.policy_permits_escalation
    ):
        action = (state.report.proposed_action
                  or "Escalate to the carrier for a delivery status check.")
        proposal = Proposal(
            proposal_id=new_id("prp"),
            investigation_id=investigation.investigation_id,
            session_id=session_id,
            order_id=order_id,
            snapshot_id=snapshot_id,
            status=ProposalStatus.pending,
            action_type="carrier_escalation",
            reason=action[:MAX_ACTION_LENGTH],
            evidence=_persist_evidence(state),
            created_at=_now(),
        )
        db.add(proposal)
        record_audit(
            db, session_id=session_id, event_type="proposal_created",
            subject_type="proposal", subject_id=proposal.proposal_id,
            detail={"order_id": order_id, "investigation_id": investigation.investigation_id},
        )

    record_audit(
        db, session_id=session_id,
        event_type=f"investigation_{investigation.status.value}",
        subject_type="investigation", subject_id=investigation.investigation_id,
        detail={
            "order_id": order_id,
            "snapshot_id": snapshot_id,
            "model_version": investigation.model_version,
            "duration_ms": investigation.duration_ms,
        },
    )
    db.commit()

    log.info(
        "investigation_finished",
        investigation_id=investigation.investigation_id,
        order_id=order_id,
        status=investigation.status.value,
        duration_ms=investigation.duration_ms,
        model_version=investigation.model_version,
    )
    return investigation


def get_investigation(db: Session, investigation_id: str, session_id: str) -> Investigation:
    """Fetch an investigation owned by this session, or 404."""
    inv = db.execute(
        select(Investigation).where(
            Investigation.investigation_id == investigation_id,
            Investigation.session_id == session_id,
        )
    ).scalar_one_or_none()
    if inv is None:
        raise NotFoundError(
            "Investigation not found.", {"investigation_id": investigation_id}
        )
    return inv


def to_schema(db: Session, inv: Investigation) -> InvestigationOut:
    """Render the stored investigation as the public response contract."""
    report = inv.report or {}
    proposal = db.execute(
        select(Proposal).where(Proposal.investigation_id == inv.investigation_id)
    ).scalar_one_or_none()

    ticket_id = None
    if proposal is not None and proposal.status is ProposalStatus.approved:
        ticket = db.execute(
            select(Ticket).where(Ticket.proposal_id == proposal.proposal_id)
        ).scalar_one_or_none()
        ticket_id = ticket.ticket_id if ticket else None

    return InvestigationOut(
        investigation_id=inv.investigation_id,
        order_id=inv.order_id,
        snapshot_id=inv.snapshot_id,
        status=inv.status.value,
        prediction_as_of=inv.prediction_as_of,
        model_version=inv.model_version,
        risk_probability=inv.risk_probability,
        summary=report.get("summary"),
        facts=[FactItem(**f) for f in report.get("facts", []) or []],
        evidence=[EvidenceItem(**e) for e in report.get("evidence", []) or []],
        limitations=report.get("limitations", []) or [],
        recommendation=report.get("recommendation"),
        recommendation_rationale=report.get("recommendation_rationale"),
        proposal=ProposalOut.model_validate(proposal) if proposal else None,
        ticket_id=ticket_id,
        error_message=inv.error_message,
        duration_ms=inv.duration_ms,
        created_at=inv.created_at,
        completed_at=inv.completed_at,
    )


def list_investigations(db: Session, session_id: str, limit: int = 25) -> list[Investigation]:
    return list(db.execute(
        select(Investigation)
        .where(Investigation.session_id == session_id)
        .order_by(Investigation.created_at.desc())
        .limit(limit)
    ).scalars().all())
