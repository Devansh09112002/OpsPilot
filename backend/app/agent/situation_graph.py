"""The bounded investigation workflow for a lane situation.

    gather_situation -> gather_lane_history -> gather_policy
        -> synthesize -> verify -> finalize

The same shape as the order graph and the same guarantees: a fixed acyclic
graph, deterministic retrieval, at most one LLM call, and the shared safety
gate in `agent.verification` deciding what the report may claim.

Two things are specific to situations.

**Membership is a safety boundary.** A situation report may discuss only the
orders in that situation. The member list goes to `verify_report` as
`allowed_order_ids`, so a statement about an order from another lane is
removed rather than displayed.

**There is a deterministic path.** Every situation can be briefed without a
provider call at all. The free tier allows roughly a hundred investigations a
day and one snapshot alone flags 395 orders, so a product that can only speak
through an LLM stops working exactly when someone is trying it. The
deterministic brief states the same verified facts with no generated prose,
and says so in its own limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.agent import tools as agent_tools
from app.agent.llm import LLMUnavailableError, generate_report
from app.agent.prompts import SITUATION_SYSTEM_PROMPT, build_situation_prompt
from app.agent.verification import verify_report
from app.core.logging import get_logger
from app.schemas.investigations import EvidenceItem, FactItem, InvestigationReport

log = get_logger(__name__)

Outcome = Literal["running", "completed", "insufficient_evidence", "failed"]
Mode = Literal["llm", "deterministic"]


@dataclass
class SituationInvestigationState:
    """Working state for one situation investigation. Never shared."""

    situation_id: str
    db: Session
    mode: Mode = "llm"

    results: dict[str, agent_tools.ToolResult] = field(default_factory=dict)
    evidence: list[EvidenceItem] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)

    snapshot_id: str | None = None
    lane: str | None = None
    member_order_ids: list[str] = field(default_factory=list)
    n_flagged: int = 0
    n_escalatable: int = 0
    expected_late: float | None = None
    mean_risk: float | None = None
    model_version: str | None = None

    policy_permits_escalation: bool = False
    policy_determination: str | None = None

    report: InvestigationReport | None = None
    generated_by: Literal["model", "deterministic"] = "deterministic"
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    status: Outcome = "running"
    error_message: str | None = None


def _record(state: SituationInvestigationState, result: agent_tools.ToolResult) -> None:
    state.results[result.tool] = result
    state.trace.append(result.trace_entry())
    if result.ok:
        state.evidence.extend(result.evidence)
    else:
        state.failures.append(f"{result.tool}: {result.error}")


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def gather_situation(state: SituationInvestigationState) -> SituationInvestigationState:
    result = agent_tools.get_situation_details(state.db, state.situation_id)
    _record(state, result)
    if result.ok:
        state.snapshot_id = result.data["snapshot_id"]
        state.lane = result.data["lane"]
        state.member_order_ids = list(result.data["member_order_ids"])
        state.n_flagged = int(result.data["n_flagged"])
        state.n_escalatable = int(result.data["n_escalatable"])
        state.expected_late = float(result.data["expected_late"])
        state.mean_risk = float(result.data["mean_risk"])
        state.model_version = result.data["model_version"]
    return state


def gather_lane_history(state: SituationInvestigationState) -> SituationInvestigationState:
    if not state.results.get("get_situation_details", agent_tools.ToolResult("x", False)).ok:
        return state
    _record(state, agent_tools.get_lane_history(state.db, state.situation_id))
    return state


def gather_policy(state: SituationInvestigationState) -> SituationInvestigationState:
    if not state.results.get("get_situation_details", agent_tools.ToolResult("x", False)).ok:
        state.failures.append(
            "get_situation_policy: the policy decision needs the situation's member "
            "counts, which could not be retrieved."
        )
        return state
    result = agent_tools.get_situation_policy(
        n_escalatable=state.n_escalatable, n_flagged=state.n_flagged
    )
    _record(state, result)
    if result.ok:
        state.policy_permits_escalation = bool(result.data["escalation_permitted_by_policy"])
        state.policy_determination = result.data["policy_determination"]
    return state


def _required_evidence_present(state: SituationInvestigationState) -> bool:
    """EVI-04: the lane, its size, its ESC-01 count and the model version."""
    ids = {e.evidence_id for e in state.evidence}
    return {
        "situation.lane",
        "situation.n_flagged",
        "situation.n_escalatable",
        "situation.model_version",
    } <= ids


def build_deterministic_brief(
    state: SituationInvestigationState,
) -> InvestigationReport:
    """Assemble a report from verified tool output, with no provider call.

    Every statement here is a restatement of an evidence item, which is why it
    can be produced at all: there is nothing to hallucinate when nothing is
    generated. The recommendation comes straight from the backend's policy
    reading, the same value that would override a model that disagreed.
    """
    lane = state.lane or "this lane"
    facts = [
        FactItem(
            statement=(
                f"{state.n_flagged} orders on lane {lane} are flagged in snapshot "
                f"{state.snapshot_id}, of which {state.n_escalatable} qualify "
                "independently for escalation under ESC-01."
            ),
            evidence_ids=["situation.lane", "situation.n_flagged", "situation.n_escalatable"],
        )
    ]
    known = {e.evidence_id for e in state.evidence}
    if "situation.expected_late" in known and state.expected_late is not None:
        facts.append(FactItem(
            statement=(
                f"The model expects about {state.expected_late:.1f} of these orders "
                "to be delivered late."
            ),
            evidence_ids=["situation.expected_late"],
        ))
    if "situation.share_of_lane" in known:
        share = next(e.value for e in state.evidence if e.evidence_id == "situation.share_of_lane")
        facts.append(FactItem(
            statement=f"Flagged orders are {share} of this lane's volume in the snapshot.",
            evidence_ids=["situation.share_of_lane"],
        ))
    if "lane.late_rate" in known:
        rate = next(e.value for e in state.evidence if e.evidence_id == "lane.late_rate")
        ids = ["lane.late_rate"]
        text = f"Historically this lane ran at {rate}"
        if "lane.baseline_rate" in known:
            base = next(
                e.value for e in state.evidence if e.evidence_id == "lane.baseline_rate"
            )
            text += f", against a marketplace rate of {base}"
            ids.append("lane.baseline_rate")
        facts.append(FactItem(statement=text + ".", evidence_ids=ids))

    recommendation = "propose_escalation" if state.policy_permits_escalation else "monitor"
    limitations = [
        "This brief was assembled directly from tool results without a language "
        "model. It states the retrieved facts and contains no generated analysis.",
        "Lane history is background only; on this dataset a lane's past late rate "
        "is weakly related to its future rate (EVI-05).",
    ]
    limitations.extend(state.failures)

    return InvestigationReport(
        summary=(
            f"Lane {lane}, snapshot {state.snapshot_id}: {state.n_flagged} flagged "
            f"orders, {state.n_escalatable} of them individually escalatable under "
            f"ESC-01, with about "
            f"{state.expected_late:.1f} expected late deliveries."
            if state.expected_late is not None
            else f"Lane {lane}, snapshot {state.snapshot_id}: {state.n_flagged} flagged orders."
        ),
        facts=facts,
        limitations=limitations,
        recommendation=recommendation,
        recommendation_rationale=(
            state.policy_determination
            or "No policy determination was available, so no escalation is proposed."
        ),
        proposed_action=(
            f"Raise one carrier escalation covering the {state.n_flagged} flagged "
            f"orders on lane {lane}."
            if recommendation == "propose_escalation"
            else None
        ),
    )


def synthesize(state: SituationInvestigationState) -> SituationInvestigationState:
    if not _required_evidence_present(state):
        state.status = "insufficient_evidence"
        state.error_message = (
            "Required evidence could not be retrieved, so no assessment was produced. "
            "Policy EVI-04 requires the lane, the number of flagged orders, the number "
            "qualifying under ESC-01, and the model version."
        )
        return state

    if state.mode == "deterministic":
        state.report = build_deterministic_brief(state)
        state.generated_by = "deterministic"
        state.trace.append({"tool": "deterministic_brief", "ok": True, "model": None})
        return state

    prompt = build_situation_prompt(
        situation_id=state.situation_id,
        results=state.results,
        evidence_items=state.evidence,
        policy_determination=state.policy_determination,
        failures=state.failures,
    )
    try:
        result = generate_report(SITUATION_SYSTEM_PROMPT, prompt)
    except LLMUnavailableError as exc:
        # The provider being down or out of quota must not take the product
        # down with it. Fall back to the deterministic brief and say so.
        log.warning("situation_llm_unavailable_falling_back", situation_id=state.situation_id)
        state.report = build_deterministic_brief(state)
        state.generated_by = "deterministic"
        state.report.limitations.insert(
            0,
            "The AI provider was unavailable for this investigation "
            f"({exc}), so this is the deterministic brief.",
        )
        state.trace.append({"tool": "llm_synthesis", "ok": False, "error": str(exc)})
        state.trace.append({"tool": "deterministic_brief", "ok": True, "model": None})
        return state

    state.report = result.report
    state.generated_by = "model"
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


def verify(state: SituationInvestigationState) -> SituationInvestigationState:
    if state.report is None or state.status in {"failed", "insufficient_evidence"}:
        return state
    state.report = verify_report(
        report=state.report,
        known_evidence_ids={e.evidence_id for e in state.evidence},
        policy_permits_escalation=state.policy_permits_escalation,
        policy_determination=state.policy_determination,
        subject=state.situation_id,
        allowed_order_ids=set(state.member_order_ids),
    )
    return state


def finalize(state: SituationInvestigationState) -> SituationInvestigationState:
    if state.status in {"failed", "insufficient_evidence"}:
        return state
    if state.report is None:
        state.status = "failed"
        if state.error_message is None:
            state.error_message = "The investigation produced no report."
    else:
        state.status = "completed"
    return state


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

def build_graph():
    g = StateGraph(SituationInvestigationState)
    g.add_node("gather_situation", gather_situation)
    g.add_node("gather_lane_history", gather_lane_history)
    g.add_node("gather_policy", gather_policy)
    g.add_node("synthesize", synthesize)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)

    g.set_entry_point("gather_situation")
    g.add_edge("gather_situation", "gather_lane_history")
    g.add_edge("gather_lane_history", "gather_policy")
    g.add_edge("gather_policy", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_graph: Any = None


def run_situation_investigation(
    db: Session, situation_id: str, *, mode: Mode = "llm"
) -> SituationInvestigationState:
    """Execute one bounded situation investigation and return its final state."""
    global _graph
    if _graph is None:
        _graph = build_graph()

    state = SituationInvestigationState(situation_id=situation_id, db=db, mode=mode)
    final = _graph.invoke(state)
    if isinstance(final, dict):
        for key, value in final.items():
            setattr(state, key, value)
        return state
    return final
