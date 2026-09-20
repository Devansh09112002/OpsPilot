"""Lane situations: grouping, the agent path, and the approval it can produce.

Written against the acceptance targets recorded in `docs/release_scope_v2.md`
before any of this was evaluated. The interesting cases here are the negative
ones: a report that cites an order from another lane, a provider outage, and a
policy that forbids what the model recommended.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.agent import tools as agent_tools
from app.agent.llm import LLMUnavailableError
from app.agent.situation_graph import run_situation_investigation
from app.core.errors import NotFoundError
from app.db.models import OrderFeature, SnapshotOrder
from app.schemas.investigations import FactItem, InvestigationReport
from app.services import policies, situations

SNAPSHOT = "2018-08-15"


def _any_situation(db):
    found = situations.list_situations(db, SNAPSHOT, limit=1)
    if not found:
        pytest.skip("the seeded slice contains no lane with enough flagged orders")
    return found[0]


def _stub_situation_llm(monkeypatch, report: InvestigationReport | Exception):
    def fake(system_prompt: str, user_prompt: str):
        if isinstance(report, Exception):
            raise report
        from types import SimpleNamespace
        return SimpleNamespace(
            report=report, model="stub", duration_ms=1, input_tokens=10, output_tokens=5
        )
    monkeypatch.setattr("app.agent.situation_graph.generate_report", fake)


# ---------------------------------------------------------------------------
# Grouping correctness
# ---------------------------------------------------------------------------

def test_members_are_real_flagged_orders_of_that_lane(db):
    situation = _any_situation(db)
    detail = situations.get_situation(db, situation.situation_id)
    assert detail.members

    member_ids = [m.order_id for m in detail.members]
    rows = db.execute(
        select(SnapshotOrder.order_id, SnapshotOrder.risk_band, SnapshotOrder.is_overdue,
               OrderFeature.seller_state, OrderFeature.customer_state)
        .join(OrderFeature, OrderFeature.order_id == SnapshotOrder.order_id)
        .where(SnapshotOrder.order_id.in_(member_ids),
               SnapshotOrder.snapshot_id == SNAPSHOT)
    ).all()

    assert len(rows) == len(member_ids), "a member is not in this snapshot"
    for row in rows:
        assert row.risk_band in situations.FLAGGED_BANDS
        assert row.is_overdue is False
        assert row.seller_state == detail.seller_state
        assert row.customer_state == detail.customer_state


def test_expected_late_is_the_sum_of_member_probabilities(db):
    detail = situations.get_situation(db, _any_situation(db).situation_id)
    recomputed = sum(m.risk_probability for m in detail.members)
    assert detail.expected_late == pytest.approx(recomputed, abs=1e-6)
    assert detail.mean_risk == pytest.approx(recomputed / len(detail.members), abs=1e-6)
    assert detail.max_risk == pytest.approx(max(m.risk_probability for m in detail.members))


def test_list_and_detail_agree_on_the_escalatable_count(db):
    """The list computes it in SQL, the detail in Python. They must not differ."""
    for summary in situations.list_situations(db, SNAPSHOT, limit=5):
        detail = situations.get_situation(db, summary.situation_id)
        from_members = sum(1 for m in detail.members if m.escalatable)
        assert summary.n_escalatable == detail.n_escalatable == from_members


def test_a_member_is_escalatable_only_if_the_order_rule_says_so(db):
    """ESC-05 composes ESC-01; it must not invent a second threshold."""
    detail = situations.get_situation(db, _any_situation(db).situation_id)
    for member in detail.members:
        permitted, _ = policies.escalation_permitted(
            risk_probability=member.risk_probability,
            days_to_deadline=member.days_to_deadline,
            is_overdue=False,
        )
        assert member.escalatable is permitted


def test_situations_are_ranked_by_expected_late(db):
    found = situations.list_situations(db, SNAPSHOT, limit=10)
    values = [s.expected_late for s in found]
    assert values == sorted(values, reverse=True)


def test_min_orders_excludes_lanes_that_are_not_a_pattern(db):
    loose = situations.list_situations(db, SNAPSHOT, limit=50, min_orders=2)
    strict = situations.list_situations(db, SNAPSHOT, limit=50, min_orders=10)
    assert all(s.n_flagged >= 2 for s in loose)
    assert all(s.n_flagged >= 10 for s in strict)
    assert len(strict) <= len(loose)


@pytest.mark.parametrize("bad", [
    "nonsense", "", "2018-08-15__SP", "2018-08-15__sp-rj",
    "2018-08-15__SPP-RJ", "../../etc/passwd", "2018-08-15__SP-RJ'; drop table orders--",
])
def test_malformed_situation_ids_are_refused(db, bad):
    with pytest.raises(NotFoundError):
        situations.get_situation(db, bad)


def test_a_valid_but_empty_lane_is_not_found(db):
    with pytest.raises(NotFoundError):
        situations.get_situation(db, f"{SNAPSHOT}__ZZ-ZZ")


# ---------------------------------------------------------------------------
# The leakage boundary
# ---------------------------------------------------------------------------

def test_no_situation_endpoint_exposes_a_delivery_outcome(client, db):
    situation = _any_situation(db)
    for path in (
        f"/api/v1/snapshots/{SNAPSHOT}/situations",
        f"/api/v1/situations/{situation.situation_id}",
    ):
        response = client.get(path)
        assert response.status_code == 200
        blob = json.dumps(response.json())
        for forbidden in ("delivered_customer", "is_late", "order_delivered_customer_date"):
            assert forbidden not in blob, f"{path} leaked {forbidden}"


def test_lane_history_is_bounded_by_the_snapshot(client, db):
    """Lane history may use outcomes, but only ones already observable."""
    situation = _any_situation(db)
    body = client.get(f"/api/v1/situations/{situation.situation_id}").json()
    history = body["lane_history"]
    assert history.get("caveats")
    joined = " ".join(history["caveats"]).lower()
    assert "delivered before" in joined
    if history["available"]:
        assert 0.0 <= history["late_rate"] <= 1.0
        assert history["sample_size"] >= policies_min_sample()


def policies_min_sample() -> int:
    from app.services.analytics import MIN_SAMPLE
    return MIN_SAMPLE


# ---------------------------------------------------------------------------
# The agent path
# ---------------------------------------------------------------------------

def test_deterministic_brief_needs_no_provider_and_cites_only_real_evidence(db):
    situation = _any_situation(db)
    state = run_situation_investigation(db, situation.situation_id, mode="deterministic")

    assert state.status == "completed"
    assert state.generated_by == "deterministic"
    assert state.llm_input_tokens == 0 and state.llm_output_tokens == 0
    assert not any(t["tool"] == "llm_synthesis" for t in state.trace)

    known = {e.evidence_id for e in state.evidence}
    assert state.report is not None and state.report.facts
    for fact in state.report.facts:
        assert set(fact.evidence_ids) <= known
    assert any("without a language model" in limitation
               for limitation in state.report.limitations)


def test_deterministic_recommendation_matches_the_backend_policy_reading(db):
    for summary in situations.list_situations(db, SNAPSHOT, limit=5):
        state = run_situation_investigation(db, summary.situation_id, mode="deterministic")
        permitted, _ = policies.situation_escalation_permitted(
            n_escalatable=summary.n_escalatable, n_flagged=summary.n_flagged
        )
        expected = "propose_escalation" if permitted else "monitor"
        assert state.report.recommendation == expected


def test_a_report_citing_an_order_from_another_lane_has_it_removed(db, monkeypatch):
    """Membership grounding: the new failure mode situations introduce."""
    situation = _any_situation(db)
    foreign = "f" * 32
    _stub_situation_llm(monkeypatch, InvestigationReport(
        summary="Lane review.",
        facts=[
            FactItem(statement=f"Order {foreign} is the worst on this lane.",
                     evidence_ids=["situation.lane"]),
            FactItem(statement="The lane carries several flagged orders.",
                     evidence_ids=["situation.n_flagged"]),
        ],
        limitations=[],
        recommendation="monitor",
        recommendation_rationale="Monitoring.",
    ))
    state = run_situation_investigation(db, situation.situation_id, mode="llm")

    statements = [f.statement for f in state.report.facts]
    assert not any(foreign in s for s in statements), "a foreign order id survived"
    assert len(statements) == 1
    assert any("not part of this situation" in limitation
               for limitation in state.report.limitations)


def test_an_order_that_is_a_member_may_be_cited(db, monkeypatch):
    """The membership rule must not reject legitimate citations."""
    situation = _any_situation(db)
    detail = situations.get_situation(db, situation.situation_id)
    member = detail.members[0].order_id
    _stub_situation_llm(monkeypatch, InvestigationReport(
        summary="Lane review.",
        facts=[FactItem(statement=f"Order {member} is a member of this situation.",
                        evidence_ids=["situation.lane"])],
        limitations=[],
        recommendation="monitor",
        recommendation_rationale="Monitoring.",
    ))
    state = run_situation_investigation(db, situation.situation_id, mode="llm")
    assert len(state.report.facts) == 1
    assert member in state.report.facts[0].statement


def test_a_provider_outage_falls_back_to_the_deterministic_brief(db, monkeypatch):
    situation = _any_situation(db)
    _stub_situation_llm(monkeypatch, LLMUnavailableError("quota exhausted"))
    state = run_situation_investigation(db, situation.situation_id, mode="llm")

    assert state.status == "completed", "an outage must not take the product down"
    assert state.generated_by == "deterministic"
    assert state.report is not None and state.report.facts
    assert any("provider was unavailable" in limitation
               for limitation in state.report.limitations)


def test_an_escalation_against_policy_is_downgraded(db, monkeypatch):
    """The model cannot escalate a lane the backend's policy reading forbids."""
    quiet = [s for s in situations.list_situations(db, SNAPSHOT, limit=20)
             if s.n_escalatable < policies.MIN_ESCALATABLE_MEMBERS]
    if not quiet:
        pytest.skip("no lane below the ESC-05 minimum in the seeded slice")

    _stub_situation_llm(monkeypatch, InvestigationReport(
        summary="This lane is a disaster.",
        facts=[],
        limitations=[],
        recommendation="propose_escalation",
        recommendation_rationale="It looks bad.",
        proposed_action="Escalate the whole lane.",
    ))
    state = run_situation_investigation(db, quiet[0].situation_id, mode="llm")

    assert state.report.recommendation == "monitor"
    assert state.report.proposed_action is None
    assert any("overridden by the backend" in limitation
               for limitation in state.report.limitations)


def test_an_unknown_situation_fails_without_inventing_one(db):
    state = run_situation_investigation(db, f"{SNAPSHOT}__ZZ-ZZ", mode="deterministic")
    assert state.status == "insufficient_evidence"
    assert state.report is None
    assert state.member_order_ids == []


def test_situation_tools_are_in_the_allowlist():
    for name in ("get_situation_details", "get_lane_history", "get_situation_policy"):
        assert name in agent_tools.TOOL_NAMES
        assert name in agent_tools.SITUATION_TOOL_NAMES


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------

def test_approving_a_lane_creates_one_ticket_covering_every_member(client, db):
    situation = None
    for candidate in situations.list_situations(db, SNAPSHOT, limit=20):
        permitted, _ = policies.situation_escalation_permitted(
            n_escalatable=candidate.n_escalatable, n_flagged=candidate.n_flagged
        )
        if permitted:
            situation = candidate
            break
    if situation is None:
        pytest.skip("no escalatable lane in the seeded slice")

    started = client.post(
        f"/api/v1/situations/{situation.situation_id}/investigations?mode=deterministic"
    )
    assert started.status_code == 200
    body = started.json()
    assert body["status"] == "completed"
    assert body["subject_type"] == "situation"
    proposal = body["proposal"]
    assert proposal is not None
    assert len(proposal["member_order_ids"]) == situation.n_flagged

    assert client.get("/api/v1/tickets").json()["total"] == 0

    approved = client.post(f"/api/v1/proposals/{proposal['proposal_id']}/approve")
    assert approved.status_code == 200
    ticket = approved.json()["ticket"]
    assert ticket["subject_type"] == "situation"
    assert ticket["action_type"] == "lane_escalation"
    assert len(ticket["member_order_ids"]) == situation.n_flagged

    # Idempotent: a second approval returns the same ticket, not a new one.
    again = client.post(f"/api/v1/proposals/{proposal['proposal_id']}/approve")
    assert again.status_code == 200
    assert again.json()["ticket"]["ticket_id"] == ticket["ticket_id"]
    assert client.get("/api/v1/tickets").json()["total"] == 1


def test_a_monitored_lane_creates_no_proposal(client, db):
    quiet = [s for s in situations.list_situations(db, SNAPSHOT, limit=20)
             if s.n_escalatable < policies.MIN_ESCALATABLE_MEMBERS]
    if not quiet:
        pytest.skip("no lane below the ESC-05 minimum in the seeded slice")

    body = client.post(
        f"/api/v1/situations/{quiet[0].situation_id}/investigations?mode=deterministic"
    ).json()
    assert body["recommendation"] == "monitor"
    assert body["proposal"] is None
    assert client.get("/api/v1/tickets").json()["total"] == 0


def test_another_session_cannot_see_a_lane_ticket(client, second_client, db):
    situation = None
    for candidate in situations.list_situations(db, SNAPSHOT, limit=20):
        permitted, _ = policies.situation_escalation_permitted(
            n_escalatable=candidate.n_escalatable, n_flagged=candidate.n_flagged
        )
        if permitted:
            situation = candidate
            break
    if situation is None:
        pytest.skip("no escalatable lane in the seeded slice")

    body = client.post(
        f"/api/v1/situations/{situation.situation_id}/investigations?mode=deterministic"
    ).json()
    approved = client.post(f"/api/v1/proposals/{body['proposal']['proposal_id']}/approve")
    ticket_id = approved.json()["ticket"]["ticket_id"]

    assert second_client.get("/api/v1/tickets").json()["total"] == 0
    assert second_client.get(f"/api/v1/tickets/{ticket_id}").status_code == 404
    assert second_client.post(
        f"/api/v1/proposals/{body['proposal']['proposal_id']}/approve"
    ).status_code == 404
