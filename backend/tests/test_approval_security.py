"""Approval, authorization, isolation, idempotency and budget tests.

Each test here corresponds to a P0 row in the plan's test matrix. A failure in
this file is a release blocker, not a cosmetic issue.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import func, select

from app.db.models import Proposal, ProposalStatus, Ticket
from app.schemas.investigations import FactItem, InvestigationReport

API = "/api/v1"


def _report(recommendation: str = "propose_escalation") -> InvestigationReport:
    return InvestigationReport(
        summary="Elevated risk with little slack remaining.",
        facts=[FactItem(statement="Model scored this order high.",
                        evidence_ids=["prediction.risk_probability"])],
        limitations=[],
        recommendation=recommendation,
        recommendation_rationale="Within the escalation window.",
        proposed_action="Escalate to the carrier for a status check.",
    )


@pytest.fixture
def stub_llm_ok(monkeypatch):
    from app.agent.llm import LLMResult

    def fake(system_prompt, user_prompt):
        return LLMResult(report=_report(), input_tokens=10, output_tokens=5,
                         duration_ms=5, model="stub")
    monkeypatch.setattr("app.agent.graph.generate_report", fake)


@pytest.fixture
def investigation_with_proposal(client, seeded_order, stub_llm_ok):
    """Run a real investigation through the API that yields a pending proposal."""
    order_id, snapshot_id = seeded_order
    r = client.post(f"{API}/investigations",
                    json={"order_id": order_id, "snapshot_id": snapshot_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed", body
    assert body["proposal"] is not None, "high-risk order should yield a proposal"
    assert body["proposal"]["status"] == "pending"
    return body


# ---------------------------------------------------------------------------
# The agent proposes; only the human approves
# ---------------------------------------------------------------------------

def test_investigation_creates_a_pending_proposal_not_a_ticket(
    investigation_with_proposal, client, db
):
    body = investigation_with_proposal
    assert body["ticket_id"] is None
    proposal_id = body["proposal"]["proposal_id"]
    assert db.execute(
        select(func.count()).select_from(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one() == 0
    assert client.get(f"{API}/tickets").json()["items"] == []


def test_approval_creates_exactly_one_ticket(investigation_with_proposal, client, db):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    r = client.post(f"{API}/proposals/{proposal_id}/approve")
    assert r.status_code == 200
    body = r.json()
    assert body["proposal"]["status"] == "approved"
    assert body["ticket"]["ticket_id"]
    assert body["already_decided"] is False

    assert db.execute(
        select(func.count()).select_from(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one() == 1

    tickets = client.get(f"{API}/tickets").json()
    assert tickets["total"] == 1
    assert "Simulation" in tickets["disclaimer"]


def test_repeat_approval_is_idempotent(investigation_with_proposal, client, db):
    """A double click must return the same ticket, not create a second."""
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    first = client.post(f"{API}/proposals/{proposal_id}/approve").json()
    second = client.post(f"{API}/proposals/{proposal_id}/approve").json()
    third = client.post(f"{API}/proposals/{proposal_id}/approve").json()

    assert first["ticket"]["ticket_id"] == second["ticket"]["ticket_id"] == third["ticket"]["ticket_id"]
    assert second["already_decided"] is True
    assert db.execute(
        select(func.count()).select_from(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one() == 1


def test_concurrent_approvals_create_one_ticket(
    investigation_with_proposal, app_instance, client, db
):
    """Two simultaneous approvals must resolve to a single ticket."""
    from fastapi.testclient import TestClient

    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    cookies = dict(client.cookies)
    results: list[int] = []
    barrier = threading.Barrier(2)

    def approve():
        with TestClient(app_instance) as c:
            for k, v in cookies.items():
                c.cookies.set(k, v)
            barrier.wait(timeout=10)
            results.append(c.post(f"{API}/proposals/{proposal_id}/approve").status_code)

    threads = [threading.Thread(target=approve) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert results.count(200) >= 1
    db.expire_all()
    assert db.execute(
        select(func.count()).select_from(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one() == 1


def test_rejection_creates_no_ticket(investigation_with_proposal, client, db):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    body = client.post(f"{API}/proposals/{proposal_id}/reject").json()
    assert body["proposal"]["status"] == "rejected"
    assert body["ticket"] is None
    assert db.execute(
        select(func.count()).select_from(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one() == 0
    assert client.get(f"{API}/tickets").json()["total"] == 0


def test_approving_a_rejected_proposal_is_denied(investigation_with_proposal, client, db):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    client.post(f"{API}/proposals/{proposal_id}/reject")
    r = client.post(f"{API}/proposals/{proposal_id}/approve")
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "conflict"
    assert db.execute(
        select(func.count()).select_from(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one() == 0


def test_rejecting_an_approved_proposal_is_denied(investigation_with_proposal, client):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    client.post(f"{API}/proposals/{proposal_id}/approve")
    assert client.post(f"{API}/proposals/{proposal_id}/reject").status_code == 409


def test_repeat_rejection_is_idempotent(investigation_with_proposal, client):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    client.post(f"{API}/proposals/{proposal_id}/reject")
    second = client.post(f"{API}/proposals/{proposal_id}/reject").json()
    assert second["already_decided"] is True
    assert second["ticket"] is None


def test_unknown_proposal_id_is_denied(client):
    r = client.post(f"{API}/proposals/prp_doesnotexist/approve")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

def test_another_guest_cannot_approve_your_proposal(
    investigation_with_proposal, second_client, db
):
    """Guest B guessing Guest A's proposal id gets nothing and creates nothing."""
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    second_client.get(f"{API}/snapshots")  # establish B's own session

    r = second_client.post(f"{API}/proposals/{proposal_id}/approve")
    assert r.status_code == 404, "cross-session approval must be refused"
    assert db.execute(
        select(func.count()).select_from(Ticket).where(Ticket.proposal_id == proposal_id)
    ).scalar_one() == 0


def test_another_guest_cannot_reject_your_proposal(
    investigation_with_proposal, second_client, db
):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    assert second_client.post(f"{API}/proposals/{proposal_id}/reject").status_code == 404
    db.expire_all()
    proposal = db.get(Proposal, proposal_id)
    assert proposal.status is ProposalStatus.pending


def test_tickets_are_isolated_by_guest(investigation_with_proposal, client, second_client):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    ticket_id = client.post(f"{API}/proposals/{proposal_id}/approve").json()["ticket"]["ticket_id"]

    assert client.get(f"{API}/tickets").json()["total"] == 1
    assert second_client.get(f"{API}/tickets").json()["total"] == 0
    assert second_client.get(f"{API}/tickets/{ticket_id}").status_code == 404


def test_investigations_are_isolated_by_guest(investigation_with_proposal, second_client):
    investigation_id = investigation_with_proposal["investigation_id"]
    assert second_client.get(f"{API}/investigations/{investigation_id}").status_code == 404
    assert second_client.get(f"{API}/investigations").json() == []


def test_audit_trail_is_isolated_and_records_the_decision(
    investigation_with_proposal, client, second_client
):
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    client.post(f"{API}/proposals/{proposal_id}/approve")

    events = client.get(f"{API}/audit").json()
    types = {e["event_type"] for e in events}
    assert "proposal_approved" in types
    assert "investigation_completed" in types
    assert second_client.get(f"{API}/audit").json() == []


def test_session_survives_a_refresh(investigation_with_proposal, client):
    """Plan section 5.3: refresh must not lose or duplicate the ticket."""
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    client.post(f"{API}/proposals/{proposal_id}/approve")

    for _ in range(3):
        body = client.get(f"{API}/tickets").json()
        assert body["total"] == 1


def test_a_forged_session_cookie_grants_nothing(
    investigation_with_proposal, second_client, client
):
    """A guessed session token is replaced, never trusted."""
    proposal_id = investigation_with_proposal["proposal"]["proposal_id"]
    victim = dict(client.cookies)["opspilot_session"]

    second_client.cookies.set("opspilot_session", victim[:-4] + "aaaa")
    assert second_client.post(f"{API}/proposals/{proposal_id}/approve").status_code == 404
    assert second_client.get(f"{API}/tickets").json()["total"] == 0


# ---------------------------------------------------------------------------
# Failed investigations never produce an action
# ---------------------------------------------------------------------------

def test_failed_investigation_creates_no_proposal(client, seeded_order, monkeypatch):
    from app.agent.llm import LLMUnavailableError

    def boom(system_prompt, user_prompt):
        raise LLMUnavailableError("provider unavailable")
    monkeypatch.setattr("app.agent.graph.generate_report", boom)

    order_id, snapshot_id = seeded_order
    body = client.post(f"{API}/investigations",
                       json={"order_id": order_id, "snapshot_id": snapshot_id}).json()

    assert body["status"] == "failed"
    assert body["proposal"] is None
    assert body["ticket_id"] is None
    assert "unavailable" in body["error_message"]
    assert client.get(f"{API}/tickets").json()["total"] == 0


def test_investigation_with_no_escalation_creates_no_proposal(
    client, seeded_order, monkeypatch
):
    from app.agent.llm import LLMResult

    def fake(system_prompt, user_prompt):
        return LLMResult(report=_report("no_escalation"), input_tokens=1,
                         output_tokens=1, duration_ms=1, model="stub")
    monkeypatch.setattr("app.agent.graph.generate_report", fake)

    order_id, snapshot_id = seeded_order
    body = client.post(f"{API}/investigations",
                       json={"order_id": order_id, "snapshot_id": snapshot_id}).json()
    assert body["status"] == "completed"
    assert body["recommendation"] == "no_escalation"
    assert body["proposal"] is None


def test_investigation_for_unknown_order_is_rejected_before_spending(client):
    r = client.post(f"{API}/investigations",
                    json={"order_id": "nope", "snapshot_id": "2018-08-15"})
    assert r.status_code == 404


def test_llm_not_configured_gives_an_actionable_message(client, seeded_order, monkeypatch):
    """With no API key the dashboard still works and Investigate explains itself."""
    from app.agent import llm as llm_module
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "llm_api_key", "", raising=False)
    llm_module.reset_client()

    order_id, snapshot_id = seeded_order
    body = client.post(f"{API}/investigations",
                       json={"order_id": order_id, "snapshot_id": snapshot_id}).json()

    assert body["status"] == "failed"
    assert body["proposal"] is None
    assert "No LLM API key" in body["error_message"]
    # The rest of the product is unaffected.
    assert client.get(f"{API}/orders?snapshot_id={snapshot_id}").status_code == 200


# ---------------------------------------------------------------------------
# Budget and rate limits
# ---------------------------------------------------------------------------

def test_investigation_quota_returns_429_not_extra_spend(
    client, seeded_order, stub_llm_ok, monkeypatch
):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "investigations_per_session_per_day", 2, raising=False)

    order_id, snapshot_id = seeded_order
    payload = {"order_id": order_id, "snapshot_id": snapshot_id}
    assert client.post(f"{API}/investigations", json=payload).status_code == 200
    assert client.post(f"{API}/investigations", json=payload).status_code == 200

    r = client.post(f"{API}/investigations", json=payload)
    assert r.status_code == 429
    error = r.json()["error"]
    assert error["code"] == "rate_limited"
    assert error["detail"]["limit_type"] == "per_session_daily"
    assert "risk queue" in error["message"]


def test_request_rate_limit_returns_429(client, monkeypatch):
    from app.core.limits import SlidingWindowLimiter

    limiter = SlidingWindowLimiter(max_events=3, window_seconds=60.0)
    monkeypatch.setattr("app.api.v1.routes.request_limiter", limiter)

    codes = [client.get(f"{API}/snapshots").status_code for _ in range(6)]
    assert 429 in codes
    assert codes[:3] == [200, 200, 200]
