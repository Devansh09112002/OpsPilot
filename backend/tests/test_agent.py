"""Agent tool, evidence-verification and failure-mode tests.

The paid LLM call is the only thing stubbed. Everything the stub is handed -
order facts, model score, historical rates, policy text - comes from the real
database and the real artifact, so these tests exercise the actual trust
boundary rather than a mock of it.
"""

from __future__ import annotations

import pytest

from app.agent import tools as agent_tools
from app.agent.graph import run_investigation
from app.agent.llm import LLMResult, LLMUnavailableError
from app.schemas.investigations import FactItem, InvestigationReport
from app.services import policies


def _fake_result(report: InvestigationReport) -> LLMResult:
    return LLMResult(
        report=report, input_tokens=100, output_tokens=50,
        duration_ms=10, model="stub",
    )


def _good_report(**overrides) -> InvestigationReport:
    base = {
        "summary": "The order is at elevated risk with little slack remaining.",
        "facts": [FactItem(
            statement="The model scored this order at elevated risk.",
            evidence_ids=["prediction.risk_probability", "prediction.model_version"],
        )],
        "limitations": [],
        "recommendation": "propose_escalation",
        "recommendation_rationale": "Risk is above threshold inside the escalation window.",
        "proposed_action": "Escalate to the carrier for a delivery status check.",
    }
    base.update(overrides)
    return InvestigationReport(**base)


@pytest.fixture
def stub_llm(monkeypatch):
    """Install a stub LLM; the test supplies the report it should return."""
    def install(report: InvestigationReport | Exception):
        def fake(system_prompt: str, user_prompt: str):
            if isinstance(report, Exception):
                raise report
            fake.last_prompt = user_prompt
            fake.last_system = system_prompt
            return _fake_result(report)
        fake.last_prompt = ""
        fake.last_system = ""
        monkeypatch.setattr("app.agent.graph.generate_report", fake)
        return fake
    return install


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def test_order_tool_returns_real_as_of_data(db, seeded_order):
    order_id, snapshot_id = seeded_order
    result = agent_tools.get_order_details(db, order_id, snapshot_id)
    assert result.ok
    assert result.data["order_id"] == order_id
    assert {e.evidence_id for e in result.evidence} >= {
        "order.order_id", "order.days_to_deadline", "order.promised_date"
    }


def test_order_tool_cannot_return_an_outcome(db, seeded_order):
    """There is no tool parameter, and no returned key, for the delivery result."""
    order_id, snapshot_id = seeded_order
    result = agent_tools.get_order_details(db, order_id, snapshot_id)
    forbidden = {"order_delivered_customer_date", "is_late", "order_status"}
    assert forbidden.isdisjoint(result.data)
    blob = str(result.data) + str([e.model_dump() for e in result.evidence])
    for key in forbidden:
        assert key not in blob


def test_order_tool_fails_honestly_for_unknown_order(db):
    result = agent_tools.get_order_details(db, "does-not-exist", "2018-08-15")
    assert not result.ok
    assert result.error
    assert result.evidence == []


def test_prediction_tool_matches_the_served_model(db, seeded_order):
    from app.services import orders as order_service

    order_id, snapshot_id = seeded_order
    result = agent_tools.get_delivery_prediction(db, order_id, snapshot_id)
    assert result.ok
    listed = order_service.get_order_as_of(db, order_id, snapshot_id)
    assert result.data["risk_probability"] == pytest.approx(
        listed.risk_probability, abs=1e-4
    )
    assert result.data["model_version"] == listed.model_version


def test_prediction_tool_fails_when_model_unavailable(db, seeded_order, monkeypatch):
    from app.ml import predictor as predictor_module

    order_id, snapshot_id = seeded_order
    monkeypatch.setattr(predictor_module.predictor, "_model", None)
    result = agent_tools.get_delivery_prediction(db, order_id, snapshot_id)
    assert not result.ok
    assert "not currently loaded" in result.error
    assert result.evidence == []


def test_history_tool_only_counts_deliveries_before_the_snapshot(db, seeded_order):
    """The as-of cutoff is the whole point of this tool."""
    from datetime import datetime

    from sqlalchemy import func, select

    from app.db.models import OrderOutcome

    order_id, snapshot_id = seeded_order
    result = agent_tools.get_historical_context(db, order_id, snapshot_id)
    assert result.ok

    baseline = result.data["route"]["baseline_sample"]
    cutoff = datetime(2018, 8, 15)
    expected = db.execute(
        select(func.count()).select_from(OrderOutcome)
        .where(OrderOutcome.order_delivered_customer_date < cutoff)
    ).scalar_one()
    assert baseline == expected

    total = db.execute(select(func.count()).select_from(OrderOutcome)).scalar_one()
    if total > expected:
        assert baseline < total, "as-of cutoff did not exclude later deliveries"


def test_history_tool_refuses_a_small_sample(db, monkeypatch, seeded_order):
    from app.services import analytics

    order_id, snapshot_id = seeded_order
    monkeypatch.setattr(analytics, "MIN_SAMPLE", 10**9)
    result = agent_tools.get_historical_context(db, order_id, snapshot_id)
    assert result.ok
    assert result.data["route"]["available"] is False
    assert result.data["route"]["late_rate"] is None
    assert any("minimum" in c for c in result.data["route"]["caveats"])
    assert not any(e.evidence_id == "history.route_late_rate" for e in result.evidence)


def test_policy_tool_selects_escalation_section_deterministically():
    result = agent_tools.get_demo_policy(
        risk_probability=0.9, days_to_deadline=1, is_overdue=False
    )
    assert result.ok
    assert result.data["escalation_permitted_by_policy"] is True
    ids = [s["section_id"] for s in result.data["sections"]]
    assert "ESC-01" in ids and "ACT-01" in ids


def test_policy_threshold_is_stated_on_the_calibrated_scale():
    """A threshold on the raw score meant nothing; 0.15 calibrated does."""
    from app.services.policies import ESCALATION_RISK_THRESHOLD

    assert ESCALATION_RISK_THRESHOLD == 0.15
    # Just below the threshold must be refused, just above permitted.
    below = agent_tools.get_demo_policy(
        risk_probability=0.149, days_to_deadline=1, is_overdue=False)
    above = agent_tools.get_demo_policy(
        risk_probability=0.151, days_to_deadline=1, is_overdue=False)
    assert below.data["escalation_permitted_by_policy"] is False
    assert above.data["escalation_permitted_by_policy"] is True


def test_policy_tool_blocks_escalation_when_slack_remains():
    result = agent_tools.get_demo_policy(
        risk_probability=0.9, days_to_deadline=10, is_overdue=False
    )
    assert result.data["escalation_permitted_by_policy"] is False
    assert "ESC-02" in [s["section_id"] for s in result.data["sections"]]


def test_policy_tool_blocks_escalation_for_overdue_orders():
    result = agent_tools.get_demo_policy(
        risk_probability=0.99, days_to_deadline=-5, is_overdue=True
    )
    assert result.data["escalation_permitted_by_policy"] is False
    assert "ESC-04" in [s["section_id"] for s in result.data["sections"]]


def test_missing_policy_file_is_an_honest_failure(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(policies, "POLICY_PATH", Path("/nonexistent/policy.json"))
    policies._load.cache_clear()
    try:
        result = agent_tools.get_demo_policy(
            risk_probability=0.9, days_to_deadline=1, is_overdue=False
        )
        assert not result.ok
        assert "not found" in result.error
    finally:
        policies._load.cache_clear()


# ---------------------------------------------------------------------------
# Graph behaviour
# ---------------------------------------------------------------------------

def test_successful_investigation_calls_every_tool(db, seeded_order, stub_llm):
    stub_llm(_good_report())
    order_id, snapshot_id = seeded_order
    state = run_investigation(db, order_id, snapshot_id)

    assert state.status == "completed"
    called = {t["tool"] for t in state.trace}
    assert called >= set(agent_tools.ORDER_TOOL_NAMES)
    assert state.risk_probability is not None
    assert state.model_version


def test_llm_failure_produces_an_honest_failed_state(db, seeded_order, stub_llm):
    stub_llm(LLMUnavailableError("provider timed out"))
    order_id, snapshot_id = seeded_order
    state = run_investigation(db, order_id, snapshot_id)

    assert state.status == "failed"
    assert "timed out" in state.error_message
    assert state.report is None


def test_missing_order_yields_insufficient_evidence_not_a_report(db, stub_llm):
    called = stub_llm(_good_report())
    state = run_investigation(db, "no-such-order", "2018-08-15")

    assert state.status == "insufficient_evidence"
    assert state.report is None
    assert called.last_prompt == "", "the LLM must not be paid for with no evidence"


def test_model_outage_yields_insufficient_evidence(db, seeded_order, stub_llm, monkeypatch):
    """Policy EVI-01 requires a risk score; without one there is no assessment."""
    from app.ml import predictor as predictor_module

    stub_llm(_good_report())
    monkeypatch.setattr(predictor_module.predictor, "_model", None)
    order_id, snapshot_id = seeded_order
    state = run_investigation(db, order_id, snapshot_id)

    assert state.status == "insufficient_evidence"
    assert "EVI-01" in state.error_message


# ---------------------------------------------------------------------------
# Verification gate
# ---------------------------------------------------------------------------

def test_facts_citing_invented_evidence_are_dropped(db, seeded_order, stub_llm):
    stub_llm(_good_report(facts=[
        FactItem(statement="Grounded claim.",
                 evidence_ids=["prediction.risk_probability"]),
        FactItem(statement="The carrier reported a depot fire.",
                 evidence_ids=["carrier.incident_report"]),
    ]))
    order_id, snapshot_id = seeded_order
    state = run_investigation(db, order_id, snapshot_id)

    statements = [f.statement for f in state.report.facts]
    assert "Grounded claim." in statements
    assert "The carrier reported a depot fire." not in statements
    assert any("cited evidence that does not exist" in limitation
               for limitation in state.report.limitations)


def test_invented_policy_section_is_flagged(db, seeded_order, stub_llm):
    stub_llm(_good_report(
        recommendation_rationale="Escalation is mandated by section ESC-99."
    ))
    order_id, snapshot_id = seeded_order
    state = run_investigation(db, order_id, snapshot_id)

    assert any("ESC-99" in limitation for limitation in state.report.limitations)


def test_escalation_against_policy_is_downgraded(db, stub_llm):
    """A low-risk order must not be escalated, whatever the model recommends."""
    from sqlalchemy import select

    from app.db.models import SnapshotOrder

    low = db.execute(
        select(SnapshotOrder)
        .where(SnapshotOrder.snapshot_id == "2018-08-15",
               SnapshotOrder.is_overdue.is_(False),
               SnapshotOrder.risk_probability < 0.3)
        .order_by(SnapshotOrder.risk_probability)
        .limit(1)
    ).scalar_one_or_none()
    if low is None:
        pytest.skip("fixture has no low-risk order")

    stub_llm(_good_report())  # insists on escalation
    state = run_investigation(db, low.order_id, low.snapshot_id)

    assert state.status == "completed"
    assert state.report.recommendation == "monitor"
    assert state.report.proposed_action is None
    assert any("overridden by the backend" in limitation
               for limitation in state.report.limitations)


def test_prompt_never_contains_an_outcome(db, seeded_order, stub_llm):
    """Whatever the agent is shown, it is never shown the delivery result."""
    called = stub_llm(_good_report())
    order_id, snapshot_id = seeded_order
    run_investigation(db, order_id, snapshot_id)

    prompt = called.last_prompt
    assert prompt
    for banned in ("order_delivered_customer_date", "is_late", "order_status"):
        assert banned not in prompt


def test_retrieved_text_is_framed_as_untrusted(db, seeded_order, stub_llm):
    called = stub_llm(_good_report())
    order_id, snapshot_id = seeded_order
    run_investigation(db, order_id, snapshot_id)

    assert 'trust="untrusted"' in called.last_prompt
    assert "never instructions" in called.last_system
