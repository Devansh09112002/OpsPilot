"""Properties the codebase claims that no test was checking.

Each test here exists because a mutation audit broke the corresponding claim
and the whole suite still passed. They are grouped by the claim rather than by
module, and each docstring names the mutation it has to fail.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.agent import tools as agent_tools
from app.core.config import get_settings
from app.db.models import InvestigationStatus
from app.schemas.investigations import FactItem, InvestigationReport
from app.services import investigations as investigation_service

# ---------------------------------------------------------------------------
# Session cookie hardening
#   mutation: httponly=True -> False, secure=settings.cookie_secure -> False
# ---------------------------------------------------------------------------


def _session_cookie_header(client) -> str:
    response = client.get("/api/v1/tickets")
    assert response.status_code == 200
    headers = [
        value for key, value in response.headers.items() if key.lower() == "set-cookie"
    ]
    assert headers, "no session cookie was issued"
    return headers[0]


def test_the_session_cookie_is_httponly(client):
    """A readable session cookie is a session any injected script can steal."""
    assert "httponly" in _session_cookie_header(client).lower()


def test_cookie_security_follows_configuration_and_is_on_in_production(client):
    """`Secure` is environment-driven, so both halves are asserted.

    Locally the API is plain HTTP and a Secure cookie would never be sent; the
    setting exists for that reason. What must not happen is the deployed
    configuration quietly serving an insecure cookie.
    """
    settings = get_settings()
    header = _session_cookie_header(client).lower()
    assert ("secure" in header) is bool(settings.cookie_secure)

    from app.core.config import Settings

    production = Settings(environment="production", _env_file=None)
    assert production.cookie_secure is True
    assert production.cookie_samesite == "none", (
        "a cross-origin deployment needs SameSite=None or the cookie is dropped"
    )


# ---------------------------------------------------------------------------
# Tool error sanitisation
#   mutation: _safe_error returns str(exc) instead of the exception class
# ---------------------------------------------------------------------------


def test_tool_errors_disclose_only_the_exception_class():
    """Raw database errors carry the statement, columns and bound parameters.

    That text reaches both the user and the LLM prompt. It would also put the
    string `order_delivered_customer_date` in front of the model, which is the
    one column the whole design keeps away from it.
    """
    secret = (
        'SELECT order_delivered_customer_date, is_late FROM order_outcomes '
        "WHERE order_id = 'abc123' -- password=hunter2"
    )
    message = agent_tools._safe_error(RuntimeError(secret), "The lane history")

    assert "RuntimeError" in message
    for leak in ("order_delivered_customer_date", "is_late", "SELECT", "hunter2", "abc123"):
        assert leak not in message, f"tool error disclosed {leak!r}: {message}"


def test_a_failing_tool_reports_no_internal_detail(db, monkeypatch):
    """The same property through a real tool rather than the helper alone."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("relation \"order_outcomes\" does not exist, column is_late")

    monkeypatch.setattr("app.services.analytics.lane_context", boom)
    result = agent_tools.get_lane_history(db, "2018-08-15__SP-RJ")

    assert result.ok is False
    assert "is_late" not in (result.error or "")
    assert "order_outcomes" not in (result.error or "")
    assert "RuntimeError" in (result.error or "")


# ---------------------------------------------------------------------------
# Point-in-time history
#   mutation: merge_asof(allow_exact_matches=False) -> True
# ---------------------------------------------------------------------------


def test_prior_history_excludes_an_outcome_at_the_handover_instant():
    """The feature may only use outcomes strictly earlier than the handover.

    `allow_exact_matches=True` would let a delivery recorded at exactly the
    handover moment count as prior knowledge. On this dataset that is the
    difference between a legal feature and a leaking one, and it is silent:
    the pipeline still runs and the model still trains.
    """
    from data_pipeline import spec
    from data_pipeline.features import _as_of_history

    settled = pd.Timestamp("2018-01-10 12:00:00")
    orders = pd.DataFrame({
        "order_id": ["settled", "at_instant", "one_second_later"],
        "order_delivered_carrier_date": [
            pd.Timestamp("2018-01-01 00:00:00"),
            settled,                                   # handover == the delivery
            settled + pd.Timedelta(seconds=1),         # handover just after
        ],
        "order_delivered_customer_date": [
            settled,
            pd.Timestamp("2018-02-01 00:00:00"),
            pd.Timestamp("2018-02-02 00:00:00"),
        ],
        spec.TARGET_NAME: [True, False, False],
    })
    key = pd.Series(["seller-a"] * 3)

    history = _as_of_history(orders, key, prior_rate=0.05, name="seller")

    assert history.loc["at_instant", "seller_prior_orders"] == 0.0, (
        "an outcome recorded at the handover instant was treated as prior "
        "knowledge; merge_asof must not allow exact matches"
    )
    assert history.loc["one_second_later", "seller_prior_orders"] == 1.0, (
        "an outcome one second before handover should be visible"
    )
    assert history.loc["settled", "seller_prior_orders"] == 0.0


# ---------------------------------------------------------------------------
# The proposal gate, independently of verification
#   mutation: drop `and state.policy_permits_escalation` from run_and_persist
# ---------------------------------------------------------------------------


class _FakeState:
    """A report that recommends escalation the policy does not permit.

    Verification normally downgrades such a recommendation before persistence
    ever sees it, which is why removing the second check broke nothing: the
    first layer hid it. Defence in depth is only defence if each layer is
    known to work on its own, so this bypasses the first one deliberately.
    """

    def __init__(self, order_id: str, snapshot_id: str):
        self.order_id = order_id
        self.snapshot_id = snapshot_id
        self.status = "completed"
        self.evidence = []
        self.failures = []
        self.trace = []
        self.risk_probability = 0.9
        self.model_version = "test-model"
        self.prediction_as_of = None
        self.llm_input_tokens = 0
        self.llm_output_tokens = 0
        self.error_message = None
        self.policy_permits_escalation = False   # the backend forbids it
        self.policy_determination = "ESC-03: below the escalation threshold."
        self.report = InvestigationReport(
            summary="Escalate immediately.",
            facts=[FactItem(statement="Grounded enough.", evidence_ids=["order.order_id"])],
            limitations=[],
            recommendation="propose_escalation",   # the model insists
            recommendation_rationale="It looks bad.",
            proposed_action="Escalate to the carrier.",
        )


def test_no_proposal_when_policy_forbids_even_if_the_report_asks(
    db, seeded_order, guest_session, monkeypatch
):
    order_id, snapshot_id = seeded_order
    monkeypatch.setattr(
        investigation_service,
        "run_investigation",
        lambda *_a, **_k: _FakeState(order_id, snapshot_id),
    )

    investigation = investigation_service.run_and_persist(
        db,
        session_id=guest_session.session_id,
        order_id=order_id,
        snapshot_id=snapshot_id,
    )

    assert investigation.status is InvestigationStatus.completed
    assert investigation.proposal is None, (
        "a proposal was created for an escalation the backend's own policy "
        "reading forbids"
    )


def test_no_proposal_for_a_situation_when_policy_forbids(db, guest_session, monkeypatch):
    """The same gate on the lane path."""
    from app.services import situations

    found = situations.list_situations(db, "2018-08-15", limit=1)
    if not found:
        pytest.skip("no situation in the seeded slice")

    class _FakeSituationState(_FakeState):
        def __init__(self, situation_id):
            super().__init__("", "2018-08-15")
            self.situation_id = situation_id
            self.lane = "SP to RJ"
            self.member_order_ids = []
            self.n_flagged = 10
            self.n_escalatable = 0
            self.expected_late = 1.0
            self.mean_risk = 0.2
            self.generated_by = "model"

    monkeypatch.setattr(
        investigation_service,
        "run_situation_investigation",
        lambda *_a, **_k: _FakeSituationState(found[0].situation_id),
    )
    investigation = investigation_service.run_situation_and_persist(
        db,
        session_id=guest_session.session_id,
        situation_id=found[0].situation_id,
    )
    assert investigation.proposal is None


# ---------------------------------------------------------------------------
# The target rule
#   mutation: drop .dt.normalize(), comparing timestamps instead of dates
# ---------------------------------------------------------------------------


def test_lateness_is_decided_on_the_calendar_date_not_the_timestamp():
    """Olist stores the promised date at midnight.

    A raw timestamp comparison marks every same-day afternoon delivery late.
    The plan mandates the calendar-date rule, and the released dataset has
    1,291 orders that sit exactly on this distinction.
    """
    from data_pipeline import spec
    from data_pipeline.loading import compute_target

    orders = pd.DataFrame({
        "order_delivered_customer_date": [
            pd.Timestamp("2018-03-10 17:30:00"),   # promised day, afternoon
            pd.Timestamp("2018-03-10 00:00:00"),   # promised day, midnight
            pd.Timestamp("2018-03-11 00:00:01"),   # one second into the next day
            pd.Timestamp("2018-03-09 23:59:59"),   # a day early
        ],
        "order_estimated_delivery_date": [pd.Timestamp("2018-03-10 00:00:00")] * 4,
    })
    late = compute_target(orders)

    assert late.tolist() == [False, False, True, False], (
        "delivery on the promised calendar date must not count as late, "
        "whatever time of day it arrived"
    )
    assert spec.TARGET_NAME  # the rule and the frozen spec travel together
