"""The four investigation tools.

This is a closed allowlist. The agent cannot run arbitrary SQL, reach the
public web, or read any table outside these functions. Each tool returns a
`ToolResult` carrying both the payload and the evidence records the backend
will later use to verify the agent's claims.

Every tool is deterministic: the same order and snapshot always produce the
same output. The LLM is used for synthesis, not for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.ml.predictor import predictor
from app.schemas.investigations import EvidenceItem
from app.services import analytics, orders, policies

log = get_logger(__name__)

TOOL_NAMES = (
    "get_order_details",
    "get_delivery_prediction",
    "get_historical_context",
    "get_demo_policy",
)


@dataclass
class ToolResult:
    """Outcome of one tool call, success or failure."""

    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceItem] = field(default_factory=list)
    error: str | None = None

    def trace_entry(self) -> dict:
        """Sanitised record for the persisted tool trace and the logs."""
        return {
            "tool": self.tool,
            "ok": self.ok,
            "error": self.error,
            "evidence_ids": [e.evidence_id for e in self.evidence],
        }


def _safe_error(exc: Exception, what: str) -> str:
    """A message safe to show a user and to place in an LLM prompt.

    Raw exception text from SQLAlchemy or psycopg embeds the full statement,
    column names and bound parameter values. That is schema disclosure, and it
    would also put the string "order_delivered_customer_date" in front of the
    model. Only the exception class is recorded here; the full traceback goes
    to the structured log.
    """
    log.error("tool_failed", tool=what, exc_info=exc)
    return f"{what} could not be retrieved ({type(exc).__name__})."


def get_order_details(db: Session, order_id: str, snapshot_id: str) -> ToolResult:
    """Approved as-of order fields.

    Returns the `OrderAsOf` DTO, which is built from a table with no outcome
    column. There is no parameter that can widen it to the delivery result.
    """
    try:
        order = orders.get_order_as_of(db, order_id, snapshot_id)
    except NotFoundError as exc:
        return ToolResult("get_order_details", ok=False, error=exc.message)

    data = order.model_dump(mode="json")
    evidence = [
        EvidenceItem(
            evidence_id="order.order_id",
            source="get_order_details",
            label="Order identifier",
            value=order.order_id,
        ),
        EvidenceItem(
            evidence_id="order.promised_date",
            source="get_order_details",
            label="Promised delivery date",
            value=order.order_estimated_delivery_date.date().isoformat(),
        ),
        EvidenceItem(
            evidence_id="order.carrier_handover",
            source="get_order_details",
            label="Carrier handover",
            value=order.order_delivered_carrier_date.isoformat(sep=" ", timespec="minutes"),
        ),
        EvidenceItem(
            evidence_id="order.days_to_deadline",
            source="get_order_details",
            label="Days remaining before the promised date",
            value=f"{order.days_to_deadline:.0f}",
        ),
        EvidenceItem(
            evidence_id="order.days_in_transit",
            source="get_order_details",
            label="Days already in transit at this snapshot",
            value=f"{order.days_in_transit:.1f}",
        ),
        EvidenceItem(
            evidence_id="order.route",
            source="get_order_details",
            label="Route",
            value=f"{order.seller_state or 'unknown'} to {order.customer_state or 'unknown'}",
        ),
        EvidenceItem(
            evidence_id="order.value",
            source="get_order_details",
            label="Order value",
            value=f"BRL {order.total_price:.2f} across {order.n_items} item(s)",
        ),
    ]
    return ToolResult("get_order_details", ok=True, data=data, evidence=evidence)


def get_delivery_prediction(db: Session, order_id: str, snapshot_id: str) -> ToolResult:
    """The served model's risk score for this order.

    Recomputed live from the stored point-in-time feature document using the
    loaded artifact. If the artifact is unavailable this fails honestly; it
    never substitutes a placeholder number.
    """
    if not predictor.available:
        return ToolResult(
            "get_delivery_prediction",
            ok=False,
            error=(
                "The delivery-risk model is not currently loaded, so no risk "
                "score can be retrieved for this order."
            ),
        )
    try:
        feature_row = orders.get_feature_row(db, order_id)
        order = orders.get_order_as_of(db, order_id, snapshot_id)
        prediction = predictor.predict_one(feature_row.features or {}, with_factors=True)
    except Exception as exc:
        return ToolResult("get_delivery_prediction", ok=False,
                          error=_safe_error(exc, "The model prediction"))

    data = {
        "order_id": order_id,
        "risk_probability": round(prediction.probability, 4),
        "risk_band": prediction.band,
        "calibrated": prediction.calibrated,
        "model_version": predictor.model_version,
        "prediction_as_of": order.order_delivered_carrier_date.isoformat(),
        "risk_factors": prediction.factors,
        "interpretation": (
            "A calibrated estimate of the chance this order misses its promised "
            "date, fitted on held-out validation data. The listed factors are "
            "attributions of the model's own score for this order, not "
            "established causes of delay."
        ),
    }
    evidence = [
        EvidenceItem(
            evidence_id="prediction.risk_probability",
            source="get_delivery_prediction",
            label="Calibrated risk estimate",
            value=f"{prediction.probability:.4f}",
        ),
        EvidenceItem(
            evidence_id="prediction.model_version",
            source="get_delivery_prediction",
            label="Model version",
            value=str(predictor.model_version),
        ),
        EvidenceItem(
            evidence_id="prediction.as_of",
            source="get_delivery_prediction",
            label="Prediction moment",
            value=order.order_delivered_carrier_date.isoformat(sep=" ", timespec="minutes"),
        ),
    ]
    # Each factor becomes its own citable evidence item, so the agent can
    # reference one without being able to invent it.
    for i, factor in enumerate(prediction.factors, start=1):
        evidence.append(EvidenceItem(
            evidence_id=f"prediction.factor_{i}",
            source="get_delivery_prediction",
            label=f"Risk factor {i}: {factor['label']}",
            value=(f"{factor['direction']} "
                   f"({factor['share']:.0%} of this order's attribution)"),
        ))
    return ToolResult("get_delivery_prediction", ok=True, data=data, evidence=evidence)


def get_historical_context(db: Session, order_id: str, snapshot_id: str) -> ToolResult:
    """Route and category late rates, computed as of the snapshot date.

    Only deliveries completed strictly before the snapshot are counted. A
    comparison drawn from too few orders is reported as unavailable with its
    sample size, rather than as a precise-looking number.
    """
    try:
        route = analytics.route_context(db, snapshot_id, order_id)
        category = analytics.category_context(db, snapshot_id, order_id)
    except Exception as exc:
        return ToolResult("get_historical_context", ok=False,
                          error=_safe_error(exc, "Historical context"))

    data = {"route": route.as_dict(), "category": category.as_dict()}
    evidence: list[EvidenceItem] = []

    if route.available and route.late_rate is not None:
        evidence.append(EvidenceItem(
            evidence_id="history.route_late_rate",
            source="get_historical_context",
            label=route.label,
            value=f"{route.late_rate:.1%} across {route.sample_size:,} prior deliveries",
        ))
    if route.baseline_rate is not None:
        evidence.append(EvidenceItem(
            evidence_id="history.baseline_late_rate",
            source="get_historical_context",
            label="Marketplace-wide late rate at this snapshot",
            value=f"{route.baseline_rate:.1%} across {route.baseline_sample:,} prior deliveries",
        ))
    if category.available and category.late_rate is not None:
        evidence.append(EvidenceItem(
            evidence_id="history.category_late_rate",
            source="get_historical_context",
            label=category.label,
            value=f"{category.late_rate:.1%} across {category.sample_size:,} prior deliveries",
        ))

    return ToolResult("get_historical_context", ok=True, data=data, evidence=evidence)


def get_demo_policy(
    *, risk_probability: float, days_to_deadline: float, is_overdue: bool
) -> ToolResult:
    """The versioned demo policy sections that govern this order.

    Section selection is deterministic code. The exact text is returned so the
    agent quotes the source rather than paraphrasing it, and the backend can
    verify that every cited section id exists.
    """
    try:
        sections = policies.applicable_sections(
            risk_probability=risk_probability,
            days_to_deadline=days_to_deadline,
            is_overdue=is_overdue,
        )
        permitted, rationale = policies.escalation_permitted(
            risk_probability=risk_probability,
            days_to_deadline=days_to_deadline,
            is_overdue=is_overdue,
        )
        version = policies.policy_version()
    except policies.PolicyUnavailableError as exc:
        return ToolResult("get_demo_policy", ok=False, error=str(exc))

    data = {
        "policy_version": version,
        "sections": [s.as_dict() for s in sections],
        "escalation_permitted_by_policy": permitted,
        "policy_determination": rationale,
    }
    evidence = [
        EvidenceItem(
            evidence_id=f"policy.{s.section_id}",
            source="get_demo_policy",
            label=f"{s.section_id} - {s.title}",
            value=s.text,
            policy_section=s.section_id,
        )
        for s in sections
    ]
    return ToolResult("get_demo_policy", ok=True, data=data, evidence=evidence)
