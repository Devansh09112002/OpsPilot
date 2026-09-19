"""SQLAlchemy models.

Two boundaries are enforced structurally rather than by convention:

1. `OrderFeature` (as-of) and `OrderOutcome` (eventual truth) are separate
   tables. No as-of query path or agent tool joins the outcome table, so a
   delivery result cannot reach the UI or the LLM by accident.
2. `Investigation`, `Proposal` and `Ticket` all carry `session_id`, and every
   read is filtered by the server-validated guest session.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    from datetime import timezone

    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Historical layer (read-only in the application path)
# ---------------------------------------------------------------------------

class OrderFeature(Base):
    """One row per eligible order: point-in-time facts known at handover.

    Deliberately contains no delivery outcome. See `OrderOutcome`.
    """

    __tablename__ = "order_features"

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    split: Mapped[str] = mapped_column(String(16), index=True)

    order_purchase_timestamp: Mapped[datetime] = mapped_column(DateTime)
    order_approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    order_delivered_carrier_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    order_estimated_delivery_date: Mapped[datetime] = mapped_column(DateTime)

    # Model features, stored as a JSON document so the feature set can evolve
    # with the artifact without a migration per column. The served model
    # validates the schema on load.
    features: Mapped[dict] = mapped_column(JSONB)

    # Denormalised for filtering and display only.
    customer_state: Mapped[str | None] = mapped_column(String(8), index=True)
    seller_state: Mapped[str | None] = mapped_column(String(8), index=True)
    product_category: Mapped[str | None] = mapped_column(String(64), index=True)
    n_items: Mapped[int] = mapped_column(Integer)
    total_price: Mapped[float] = mapped_column(Float)
    total_freight: Mapped[float] = mapped_column(Float)


class OrderOutcome(Base):
    """Eventual delivery truth. Offline scoring and as-of aggregates only.

    Never exposed by an as-of endpoint or an agent tool. The historical-context
    service may aggregate rows whose delivery is strictly earlier than the
    snapshot date, which an operator would genuinely have known.
    """

    __tablename__ = "order_outcomes"

    order_id: Mapped[str] = mapped_column(
        ForeignKey("order_features.order_id", ondelete="CASCADE"), primary_key=True
    )
    order_delivered_customer_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    is_late: Mapped[bool] = mapped_column(Boolean, index=True)


class Snapshot(Base):
    """A frozen historical moment the demo can be viewed at."""

    __tablename__ = "snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    label: Mapped[str] = mapped_column(String(64))
    snapshot_at: Mapped[datetime] = mapped_column(DateTime)
    orders_in_transit: Mapped[int] = mapped_column(Integer)
    orders_pre_deadline: Mapped[int] = mapped_column(Integer)
    orders_overdue: Mapped[int] = mapped_column(Integer)


class SnapshotOrder(Base):
    """Membership of an order in a snapshot, plus its served risk score.

    `risk_probability` is written by the ingestion job using the same artifact
    the API serves, and a parity test asserts the two agree.
    """

    __tablename__ = "snapshot_orders"
    __table_args__ = (
        Index("ix_snapshot_orders_queue", "snapshot_id", "is_overdue", "risk_probability"),
    )

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("snapshots.snapshot_id", ondelete="CASCADE"), primary_key=True
    )
    order_id: Mapped[str] = mapped_column(
        ForeignKey("order_features.order_id", ondelete="CASCADE"), primary_key=True
    )
    is_overdue: Mapped[bool] = mapped_column(Boolean)
    days_in_transit: Mapped[float] = mapped_column(Float)
    risk_probability: Mapped[float] = mapped_column(Float)
    risk_band: Mapped[str] = mapped_column(String(8))
    model_version: Mapped[str] = mapped_column(String(64))


# ---------------------------------------------------------------------------
# Guest session and mutable demo state
# ---------------------------------------------------------------------------

class GuestSession(Base):
    __tablename__ = "guest_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    investigations_today: Mapped[int] = mapped_column(Integer, default=0)
    quota_window_start: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class InvestigationStatus(str, enum.Enum):
    running = "running"
    completed = "completed"
    insufficient_evidence = "insufficient_evidence"
    failed = "failed"


class Investigation(Base):
    __tablename__ = "investigations"
    __table_args__ = (
        Index("ix_investigations_session", "session_id", "created_at"),
    )

    investigation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("guest_sessions.session_id", ondelete="CASCADE"), index=True
    )
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    snapshot_id: Mapped[str] = mapped_column(String(16))

    status: Mapped[InvestigationStatus] = mapped_column(
        Enum(InvestigationStatus, name="investigation_status"),
        default=InvestigationStatus.running,
    )
    prediction_as_of: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)

    report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tool_trace: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    proposal = relationship("Proposal", back_populates="investigation", uselist=False)


class ProposalStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class Proposal(Base):
    """An agent's *suggestion*. Only the backend can turn it into a ticket."""

    __tablename__ = "proposals"
    __table_args__ = (
        # One proposal per investigation: a retry cannot silently create a second.
        UniqueConstraint("investigation_id", name="uq_proposal_investigation"),
    )

    proposal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(
        ForeignKey("investigations.investigation_id", ondelete="CASCADE"), index=True
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("guest_sessions.session_id", ondelete="CASCADE"), index=True
    )
    order_id: Mapped[str] = mapped_column(String(32))
    snapshot_id: Mapped[str] = mapped_column(String(16))

    status: Mapped[ProposalStatus] = mapped_column(
        Enum(ProposalStatus, name="proposal_status"), default=ProposalStatus.pending, index=True
    )
    action_type: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    investigation = relationship("Investigation", back_populates="proposal")


class Ticket(Base):
    """A simulated escalation ticket. No external system is ever contacted."""

    __tablename__ = "tickets"
    __table_args__ = (
        # Idempotency: a double-click or retry can only ever yield one ticket.
        UniqueConstraint("proposal_id", name="uq_ticket_proposal"),
        Index("ix_tickets_session", "session_id", "created_at"),
    )

    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    proposal_id: Mapped[str] = mapped_column(
        ForeignKey("proposals.proposal_id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("guest_sessions.session_id", ondelete="CASCADE"), index=True
    )
    investigation_id: Mapped[str] = mapped_column(String(64))
    order_id: Mapped[str] = mapped_column(String(32))
    snapshot_id: Mapped[str] = mapped_column(String(16))

    status: Mapped[str] = mapped_column(String(16), default="open")
    action_type: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text)
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AuditEvent(Base):
    """Append-only record of every state-changing action."""

    __tablename__ = "audit_events"
    __table_args__ = (
        CheckConstraint("length(event_type) > 0", name="ck_audit_event_type"),
        Index("ix_audit_session", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(48))
    subject_type: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
