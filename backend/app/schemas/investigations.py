"""Investigation, proposal and ticket schemas.

`InvestigationReport` is the validated structured output the agent must
produce. Every quantitative claim carries an `evidence_id` that the backend
resolves against real tool results before the report is persisted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Recommendation = Literal["no_escalation", "monitor", "propose_escalation"]
InvestigationStatusLiteral = Literal[
    "running", "completed", "insufficient_evidence", "failed"
]
ProposalStatusLiteral = Literal["pending", "approved", "rejected"]


class EvidenceItem(BaseModel):
    """One retrieved fact, traceable to the tool call that produced it."""

    evidence_id: str = Field(description="Stable id, e.g. 'prediction.risk_probability'")
    source: str = Field(description="Tool that produced it, e.g. 'get_delivery_prediction'")
    label: str
    value: str
    policy_section: str | None = None


class FactItem(BaseModel):
    """A statement the agent makes, tied to the evidence that supports it."""

    statement: str
    evidence_ids: list[str] = Field(min_length=1)


class InvestigationReport(BaseModel):
    """The LLM's structured output. Validated, then checked against tool data.

    Deliberately NOT `extra="forbid"`: that emits `additionalProperties: false`
    into the JSON schema, which the Gemini API rejects outright
    ("Unknown name additional_properties"). Unknown keys are ignored instead,
    which is safe here because nothing downstream reads an unlisted field and
    `agent.graph.verify` re-checks every claim against real tool output.
    """

    summary: str = Field(min_length=1, max_length=2000)
    facts: list[FactItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendation: Recommendation
    recommendation_rationale: str = Field(min_length=1, max_length=2000)
    proposed_action: str | None = None


class ProposalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    proposal_id: str
    investigation_id: str
    order_id: str
    snapshot_id: str
    status: ProposalStatusLiteral
    action_type: str
    reason: str
    created_at: datetime
    decided_at: datetime | None = None


class InvestigationOut(BaseModel):
    """The full response contract for an investigation."""

    investigation_id: str
    order_id: str
    snapshot_id: str
    status: InvestigationStatusLiteral

    prediction_as_of: datetime | None = None
    model_version: str | None = None
    risk_probability: float | None = None

    summary: str | None = None
    facts: list[FactItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    recommendation_rationale: str | None = None

    proposal: ProposalOut | None = None
    ticket_id: str | None = None

    error_message: str | None = None
    duration_ms: int | None = None
    created_at: datetime
    completed_at: datetime | None = None

    disclaimer: str = (
        "Simulation - OpsPilot demo environment. No courier, seller or customer "
        "is contacted and no real fulfilment action is taken."
    )


class InvestigationRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)
    snapshot_id: str = Field(min_length=1, max_length=16)


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticket_id: str
    proposal_id: str
    investigation_id: str
    order_id: str
    snapshot_id: str
    status: str
    action_type: str
    reason: str
    risk_probability: float | None = None
    model_version: str | None = None
    created_at: datetime


class TicketListOut(BaseModel):
    items: list[TicketOut]
    total: int
    disclaimer: str = (
        "Simulation - no real fulfilment action was taken for any ticket below."
    )


class AuditEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_type: str
    subject_type: str
    subject_id: str
    created_at: datetime
    detail: dict | None = None


class DecisionResponse(BaseModel):
    """Result of approving or rejecting a proposal."""

    proposal: ProposalOut
    ticket: TicketOut | None = None
    already_decided: bool = Field(
        default=False,
        description="True when a repeat request returned the existing outcome.",
    )
