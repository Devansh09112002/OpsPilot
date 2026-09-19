"""The bounded LangGraph investigation workflow.

    gather_order -> gather_prediction -> gather_history -> gather_policy
        -> synthesize -> verify -> finalize

Retrieval is deterministic code, so the graph cannot loop, cannot call a tool
that is not in the allowlist, and cannot run up an unbounded bill: exactly one
LLM call happens, at `synthesize`.

`verify` is the safety gate. It re-checks every evidence id and every
quantitative claim the model made against the real tool results, and it
overrides the recommendation if the model contradicts the backend's own
reading of the policy. A report that fails verification becomes
`insufficient_evidence`, never a ticket.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.agent import tools as agent_tools
from app.agent.llm import LLMUnavailableError, generate_report
from app.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.core.logging import get_logger
from app.schemas.investigations import EvidenceItem, FactItem, InvestigationReport
from app.services import policies

log = get_logger(__name__)

Outcome = Literal["completed", "insufficient_evidence", "failed"]


@dataclass
class InvestigationState:
    """Working state for one investigation. Not shared between requests."""

    order_id: str
    snapshot_id: str
    db: Session

    results: dict[str, agent_tools.ToolResult] = field(default_factory=dict)
    evidence: list[EvidenceItem] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)

    risk_probability: float | None = None
    model_version: str | None = None
    prediction_as_of: Any = None
    days_to_deadline: float | None = None
    is_overdue: bool = False

    policy_permits_escalation: bool = False
    policy_determination: str | None = None

    report: InvestigationReport | None = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    status: Outcome = "failed"
    error_message: str | None = None


def _record(state: InvestigationState, result: agent_tools.ToolResult) -> None:
    state.results[result.tool] = result
    state.trace.append(result.trace_entry())
    if result.ok:
        state.evidence.extend(result.evidence)
    else:
        state.failures.append(f"{result.tool}: {result.error}")


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def gather_order(state: InvestigationState) -> InvestigationState:
    result = agent_tools.get_order_details(state.db, state.order_id, state.snapshot_id)
    _record(state, result)
    if result.ok:
        state.days_to_deadline = float(result.data["days_to_deadline"])
        state.is_overdue = bool(result.data["is_overdue"])
    return state


def gather_prediction(state: InvestigationState) -> InvestigationState:
    if "get_order_details" not in state.results or not state.results["get_order_details"].ok:
        return state  # no order, nothing to predict
    result = agent_tools.get_delivery_prediction(
        state.db, state.order_id, state.snapshot_id
    )
    _record(state, result)
    if result.ok:
        state.risk_probability = float(result.data["risk_probability"])
        state.model_version = result.data["model_version"]
        state.prediction_as_of = result.data["prediction_as_of"]
    return state


def gather_history(state: InvestigationState) -> InvestigationState:
    if not state.results.get("get_order_details", agent_tools.ToolResult("x", False)).ok:
        return state
    _record(state, agent_tools.get_historical_context(
        state.db, state.order_id, state.snapshot_id))
    return state


def gather_policy(state: InvestigationState) -> InvestigationState:
    if state.risk_probability is None or state.days_to_deadline is None:
        state.failures.append(
            "get_demo_policy: policy selection needs a risk score and a deadline, "
            "and at least one was unavailable."
        )
        return state
    result = agent_tools.get_demo_policy(
        risk_probability=state.risk_probability,
        days_to_deadline=state.days_to_deadline,
        is_overdue=state.is_overdue,
    )
    _record(state, result)
    if result.ok:
        state.policy_permits_escalation = bool(result.data["escalation_permitted_by_policy"])
        state.policy_determination = result.data["policy_determination"]
    return state


def _required_evidence_present(state: InvestigationState) -> bool:
    """Policy EVI-01: order id, risk score with version, and remaining days."""
    ids = {e.evidence_id for e in state.evidence}
    return {
        "order.order_id",
        "prediction.risk_probability",
        "prediction.model_version",
        "order.days_to_deadline",
    } <= ids


def synthesize(state: InvestigationState) -> InvestigationState:
    if not _required_evidence_present(state):
        state.status = "insufficient_evidence"
        state.error_message = (
            "Required evidence could not be retrieved, so no assessment was produced. "
            "Policy EVI-01 requires the order identifier, the model risk score with "
            "its version, and the days remaining before the promised date."
        )
        return state

    prompt = build_user_prompt(
        order_id=state.order_id,
        snapshot_id=state.snapshot_id,
        results=state.results,
        evidence_items=state.evidence,
        policy_determination=state.policy_determination,
        failures=state.failures,
    )
    try:
        result = generate_report(SYSTEM_PROMPT, prompt)
    except LLMUnavailableError as exc:
        state.status = "failed"
        state.error_message = str(exc)
        state.trace.append({"tool": "llm_synthesis", "ok": False, "error": str(exc)})
        return state

    state.report = result.report
    state.llm_input_tokens = result.input_tokens
    state.llm_output_tokens = result.output_tokens
    state.trace.append({
        "tool": "llm_synthesis",
        "ok": True,
        "model": result.model,
        "duration_ms": result.duration_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    })
    return state


NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def _numbers_in(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text))


def verify(state: InvestigationState) -> InvestigationState:
    """Check the model's report against the real tool results.

    Three checks, each of which can only *reduce* what the report claims:

    1. Every cited evidence id must exist. Facts citing unknown ids are dropped.
    2. Every policy section id mentioned must exist in the loaded policy.
    3. The recommendation may not exceed what the backend's own policy reading
       permits. A model that recommends escalation against policy is downgraded.
    """
    if state.report is None or state.status in {"failed", "insufficient_evidence"}:
        return state

    known_ids = {e.evidence_id for e in state.evidence}
    report = state.report

    kept: list[FactItem] = []
    dropped = 0
    for fact in report.facts:
        unknown = [i for i in fact.evidence_ids if i not in known_ids]
        if unknown:
            dropped += 1
            log.warning(
                "investigation_fact_rejected",
                order_id=state.order_id,
                unknown_evidence_ids=unknown,
            )
            continue
        kept.append(fact)

    extra_limitations: list[str] = []
    if dropped:
        extra_limitations.append(
            f"{dropped} generated statement(s) cited evidence that does not exist "
            "and were removed by the backend before display."
        )

    # Policy ids mentioned anywhere in the prose must be real.
    valid_sections = {s.section_id for s in policies.all_sections()}
    mentioned = set(re.findall(r"\b(?:ESC|EVI|ACT)-\d{2}\b",
                               f"{report.summary} {report.recommendation_rationale}"))
    invented = mentioned - valid_sections
    if invented:
        extra_limitations.append(
            f"Reference(s) to non-existent policy section(s) {sorted(invented)} were "
            "flagged by the backend and should be disregarded."
        )
        log.warning("investigation_invented_policy", sections=sorted(invented))

    # The recommendation cannot exceed the backend's own policy reading.
    recommendation = report.recommendation
    if recommendation == "propose_escalation" and not state.policy_permits_escalation:
        recommendation = "monitor"
        extra_limitations.append(
            "The generated recommendation to escalate was overridden by the backend "
            f"because the demo policy does not permit it here. {state.policy_determination}"
        )
        log.warning("investigation_recommendation_downgraded", order_id=state.order_id)

    state.report = InvestigationReport(
        summary=report.summary,
        facts=kept,
        limitations=[*report.limitations, *extra_limitations],
        recommendation=recommendation,
        recommendation_rationale=report.recommendation_rationale,
        proposed_action=report.proposed_action if recommendation == "propose_escalation" else None,
    )
    return state


def finalize(state: InvestigationState) -> InvestigationState:
    if state.status in {"failed", "insufficient_evidence"}:
        return state
    state.status = "completed" if state.report is not None else "failed"
    if state.report is None and state.error_message is None:
        state.error_message = "The investigation produced no report."
    return state


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def build_graph():
    """Compile the fixed, acyclic investigation graph."""
    g = StateGraph(InvestigationState)
    g.add_node("gather_order", gather_order)
    g.add_node("gather_prediction", gather_prediction)
    g.add_node("gather_history", gather_history)
    g.add_node("gather_policy", gather_policy)
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)

    g.set_entry_point("gather_order")
    g.add_edge("gather_order", "gather_prediction")
    g.add_edge("gather_prediction", "gather_history")
    g.add_edge("gather_history", "gather_policy")
    g.add_edge("gather_policy", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_graph = None


def run_investigation(db: Session, order_id: str, snapshot_id: str) -> InvestigationState:
    """Execute one bounded investigation and return its final state."""
    global _graph
    if _graph is None:
        _graph = build_graph()

    state = InvestigationState(order_id=order_id, snapshot_id=snapshot_id, db=db)
    final = _graph.invoke(state)
    # LangGraph returns the dataclass state as a mapping; rebuild for typing.
    if isinstance(final, dict):
        for key, value in final.items():
            setattr(state, key, value)
        return state
    return final
