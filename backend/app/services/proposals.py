"""Proposal approval, rejection and ticket creation.

This module is the trust boundary. The agent can only ever create a *pending*
proposal; turning one into a ticket happens here and nowhere else, and only on
an explicit request from the owning session.

Integrity properties, each covered by a test in `test_approval_security.py`:

* Ownership - a proposal is only visible and decidable by its own session.
  A cross-session identifier returns 404, not 403, so ids cannot be probed.
* Explicitness - approval requires a request to the approve endpoint. No code
  path approves a proposal on the agent's behalf or on restart.
* Idempotency - `tickets.proposal_id` is UNIQUE. A double click or retry
  returns the existing ticket instead of creating a second.
* Atomicity - the status transition, the ticket insert and the audit event
  share one transaction.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError
from app.core.logging import get_logger
from app.db.models import (
    AuditEvent,
    Investigation,
    InvestigationStatus,
    Proposal,
    ProposalStatus,
    Ticket,
)
from app.schemas.investigations import DecisionResponse, ProposalOut, TicketOut

log = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def record_audit(
    db: Session,
    *,
    session_id: str | None,
    event_type: str,
    subject_type: str,
    subject_id: str,
    detail: dict | None = None,
) -> None:
    db.add(AuditEvent(
        session_id=session_id,
        event_type=event_type,
        subject_type=subject_type,
        subject_id=subject_id,
        detail=detail,
        created_at=_now(),
    ))


def _load_owned_proposal(db: Session, proposal_id: str, session_id: str) -> Proposal:
    """Fetch a proposal, or 404 if it does not belong to this session.

    Returning 404 rather than 403 for a cross-session id means an attacker
    cannot use the response to confirm that an identifier exists.
    """
    proposal = db.execute(
        select(Proposal).where(
            Proposal.proposal_id == proposal_id,
            Proposal.session_id == session_id,
        )
    ).scalar_one_or_none()
    if proposal is None:
        raise NotFoundError(
            "Proposal not found.", {"proposal_id": proposal_id}
        )
    return proposal


def _existing_ticket(db: Session, proposal_id: str) -> Ticket | None:
    return db.execute(
        select(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one_or_none()


def approve_proposal(db: Session, proposal_id: str, session_id: str) -> DecisionResponse:
    """Approve a pending proposal and create exactly one ticket."""
    proposal = _load_owned_proposal(db, proposal_id, session_id)

    if proposal.status is ProposalStatus.approved:
        ticket = _existing_ticket(db, proposal_id)
        if ticket is None:  # pragma: no cover - guarded by the UNIQUE constraint
            raise ConflictError("Proposal is approved but its ticket is missing.")
        return DecisionResponse(
            proposal=ProposalOut.model_validate(proposal),
            ticket=TicketOut.model_validate(ticket),
            already_decided=True,
        )

    if proposal.status is ProposalStatus.rejected:
        raise ConflictError(
            "This proposal was rejected and cannot be approved.",
            {"proposal_id": proposal_id, "status": proposal.status.value},
        )

    investigation = db.get(Investigation, proposal.investigation_id)
    if investigation is None or investigation.status is not InvestigationStatus.completed:
        raise ConflictError(
            "The investigation behind this proposal did not complete successfully, "
            "so it cannot be approved.",
            {"proposal_id": proposal_id},
        )
    if not proposal.evidence:
        raise ConflictError(
            "This proposal carries no supporting evidence and cannot be approved.",
            {"proposal_id": proposal_id},
        )

    ticket = Ticket(
        ticket_id=new_id("tkt"),
        proposal_id=proposal.proposal_id,
        session_id=session_id,
        investigation_id=proposal.investigation_id,
        subject_type=proposal.subject_type,
        subject_id=proposal.subject_id,
        order_id=proposal.order_id,
        # The orders this approval actually covers. Recorded on the ticket so
        # the scope of an authorised action is auditable rather than implied
        # by the lane name.
        member_order_ids=proposal.member_order_ids,
        snapshot_id=proposal.snapshot_id,
        status="open",
        action_type=proposal.action_type,
        reason=proposal.reason,
        risk_probability=investigation.risk_probability,
        model_version=investigation.model_version,
        created_at=_now(),
    )
    proposal.status = ProposalStatus.approved
    proposal.decided_at = _now()
    db.add(ticket)
    record_audit(
        db,
        session_id=session_id,
        event_type="proposal_approved",
        subject_type="proposal",
        subject_id=proposal.proposal_id,
        detail={
            "ticket_id": ticket.ticket_id,
            "order_id": proposal.order_id,
            "action_type": proposal.action_type,
        },
    )

    try:
        db.commit()
    except IntegrityError:
        # A concurrent request won the race. The UNIQUE constraint on
        # tickets.proposal_id guarantees only one ticket exists; return it.
        db.rollback()
        existing = _existing_ticket(db, proposal_id)
        if existing is None:  # pragma: no cover - would mean a different failure
            raise
        refreshed = _load_owned_proposal(db, proposal_id, session_id)
        log.info("approval_race_resolved", proposal_id=proposal_id,
                 ticket_id=existing.ticket_id)
        return DecisionResponse(
            proposal=ProposalOut.model_validate(refreshed),
            ticket=TicketOut.model_validate(existing),
            already_decided=True,
        )

    log.info(
        "proposal_approved",
        proposal_id=proposal.proposal_id,
        ticket_id=ticket.ticket_id,
        order_id=proposal.order_id,
    )
    return DecisionResponse(
        proposal=ProposalOut.model_validate(proposal),
        ticket=TicketOut.model_validate(ticket),
        already_decided=False,
    )


def reject_proposal(db: Session, proposal_id: str, session_id: str) -> DecisionResponse:
    """Reject a pending proposal. No ticket is created, ever."""
    proposal = _load_owned_proposal(db, proposal_id, session_id)

    if proposal.status is ProposalStatus.rejected:
        return DecisionResponse(
            proposal=ProposalOut.model_validate(proposal),
            ticket=None,
            already_decided=True,
        )
    if proposal.status is ProposalStatus.approved:
        raise ConflictError(
            "This proposal was already approved and a ticket exists for it.",
            {"proposal_id": proposal_id, "status": proposal.status.value},
        )

    proposal.status = ProposalStatus.rejected
    proposal.decided_at = _now()
    record_audit(
        db,
        session_id=session_id,
        event_type="proposal_rejected",
        subject_type="proposal",
        subject_id=proposal.proposal_id,
        detail={"order_id": proposal.order_id},
    )
    db.commit()
    log.info("proposal_rejected", proposal_id=proposal.proposal_id)
    return DecisionResponse(
        proposal=ProposalOut.model_validate(proposal), ticket=None, already_decided=False
    )


def list_tickets(db: Session, session_id: str, limit: int = 50) -> list[Ticket]:
    return list(db.execute(
        select(Ticket)
        .where(Ticket.session_id == session_id)
        .order_by(Ticket.created_at.desc())
        .limit(limit)
    ).scalars().all())


def get_ticket(db: Session, ticket_id: str, session_id: str) -> Ticket:
    ticket = db.execute(
        select(Ticket).where(
            Ticket.ticket_id == ticket_id, Ticket.session_id == session_id
        )
    ).scalar_one_or_none()
    if ticket is None:
        raise NotFoundError("Ticket not found.", {"ticket_id": ticket_id})
    return ticket


def list_audit_events(db: Session, session_id: str, limit: int = 100) -> list[AuditEvent]:
    return list(db.execute(
        select(AuditEvent)
        .where(AuditEvent.session_id == session_id)
        .order_by(AuditEvent.created_at.desc())
        .limit(limit)
    ).scalars().all())
