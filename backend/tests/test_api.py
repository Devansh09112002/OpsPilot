"""API contract, filtering and failure-mode tests."""

from __future__ import annotations

import pytest

API = "/api/v1"


def test_health_is_live(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_reports_each_dependency(client):
    body = client.get(f"{API}/health/ready").json()
    assert set(body["checks"]) == {"database", "model", "policy", "llm"}
    assert body["checks"]["database"]["ok"] is True
    assert body["checks"]["model"]["ok"] is True


def test_readiness_is_degraded_not_down_without_llm(client):
    """A missing LLM key must not make the product look broken."""
    body = client.get(f"{API}/health/ready").json()
    if not body["checks"]["llm"]["ok"]:
        assert body["status"] == "degraded"
        assert body["checks"]["model"]["ok"] is True


def test_snapshots_are_listed(client):
    items = client.get(f"{API}/snapshots").json()
    assert len(items) >= 1
    assert {"snapshot_id", "orders_pre_deadline", "orders_overdue"} <= set(items[0])


def test_snapshot_stats_come_from_sql(client):
    stats = client.get(f"{API}/snapshots/2018-08-15/stats").json()
    assert stats["high_risk"] + stats["medium_risk"] + stats["low_risk"] > 0
    assert 0.0 <= stats["mean_risk"] <= 1.0
    assert stats["model_version"]


def test_unknown_snapshot_returns_actionable_error(client):
    r = client.get(f"{API}/snapshots/not-a-snapshot/stats")
    assert r.status_code == 404
    body = r.json()["error"]
    assert body["code"] == "not_found"
    assert "available" in body["detail"]


def test_orders_are_ranked_by_risk_descending(client):
    items = client.get(f"{API}/orders?snapshot_id=2018-08-15&limit=25").json()["items"]
    # Calibration is monotonic, so ordering by the raw ranking score also
    # leaves the calibrated probabilities non-increasing.
    risks = [i["risk_probability"] for i in items]
    assert risks == sorted(risks, reverse=True)
    assert all(0.0 <= r <= 1.0 for r in risks)


def test_orders_exclude_overdue_by_default(client):
    items = client.get(f"{API}/orders?snapshot_id=2018-08-15&limit=100").json()["items"]
    assert all(i["is_overdue"] is False for i in items)


def test_risk_band_filter_applies(client):
    items = client.get(
        f"{API}/orders?snapshot_id=2018-08-15&risk_band=high&limit=50"
    ).json()["items"]
    assert items, "fixture should contain high-risk orders"
    assert all(i["risk_band"] == "high" for i in items)


def test_pagination_is_stable_and_non_overlapping(client):
    base = f"{API}/orders?snapshot_id=2018-08-15&limit=10"
    first = client.get(base).json()["items"]
    second = client.get(f"{base}&offset=10").json()["items"]
    assert {i["order_id"] for i in first}.isdisjoint({i["order_id"] for i in second})
    assert client.get(base).json()["items"] == first  # deterministic


def test_invalid_pagination_is_rejected(client):
    assert client.get(f"{API}/orders?snapshot_id=2018-08-15&limit=9999").status_code == 422
    assert client.get(f"{API}/orders?snapshot_id=2018-08-15&offset=-1").status_code == 422


def test_empty_result_is_a_clean_empty_state(client):
    page = client.get(
        f"{API}/orders?snapshot_id=2018-08-15&customer_state=ZZ"
    ).json()
    assert page["items"] == []
    assert page["total"] == 0


# ---------------------------------------------------------------------------
# The leakage boundary, asserted at the HTTP layer
# ---------------------------------------------------------------------------

FORBIDDEN_KEYS = {
    "order_delivered_customer_date", "is_late", "order_status",
    "delivered_at", "actual_delivery_date",
}


def _assert_no_outcome(payload) -> None:
    """Recursively assert no outcome field appears anywhere in a response."""
    if isinstance(payload, dict):
        leaked = FORBIDDEN_KEYS & set(payload)
        assert not leaked, f"as-of payload leaked outcome field(s): {leaked}"
        for v in payload.values():
            _assert_no_outcome(v)
    elif isinstance(payload, list):
        for v in payload:
            _assert_no_outcome(v)


def test_order_list_exposes_no_outcome(client):
    _assert_no_outcome(client.get(f"{API}/orders?snapshot_id=2018-08-15&limit=50").json())


def test_order_detail_exposes_no_outcome(client, seeded_order):
    order_id, snapshot_id = seeded_order
    body = client.get(f"{API}/orders/{order_id}?snapshot_id={snapshot_id}").json()
    _assert_no_outcome(body)
    assert body["prediction_as_of"] == body["order_delivered_carrier_date"]


def test_order_not_in_snapshot_is_404(client):
    r = client.get(f"{API}/orders/{'0' * 32}?snapshot_id=2018-08-15")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def test_prediction_matches_the_stored_queue_score(client, seeded_order):
    """Live inference and the score shown in the queue must agree."""
    order_id, snapshot_id = seeded_order
    listed = client.get(f"{API}/orders/{order_id}?snapshot_id={snapshot_id}").json()
    served = client.post(
        f"{API}/predictions", json={"order_id": order_id, "snapshot_id": snapshot_id}
    ).json()

    assert served["risk_probability"] == pytest.approx(listed["risk_probability"], abs=1e-4)
    assert served["ranking_score"] == pytest.approx(listed["ranking_score"], abs=1e-5)
    assert served["model_version"] == listed["model_version"]
    assert served["risk_band"] == listed["risk_band"]
    assert "carrier handover" in served["disclaimer"].lower()


def test_served_probability_is_calibrated_not_the_raw_score(client, seeded_order):
    """The displayed number must be the calibrated one, not the raw output.

    Before calibration a raw 0.65 corresponded to a ~2% observed late rate, so
    serving the raw value as a probability was misleading.
    """
    order_id, snapshot_id = seeded_order
    served = client.post(
        f"{API}/predictions", json={"order_id": order_id, "snapshot_id": snapshot_id}
    ).json()
    assert served["calibrated"] is True
    assert served["risk_probability"] != pytest.approx(served["ranking_score"], abs=1e-6)
    assert 0.0 <= served["risk_probability"] <= 1.0


def test_prediction_explains_itself(client, seeded_order):
    """A served prediction says which inputs moved its score."""
    order_id, snapshot_id = seeded_order
    served = client.post(
        f"{API}/predictions", json={"order_id": order_id, "snapshot_id": snapshot_id}
    ).json()
    factors = served["risk_factors"]
    assert factors, "a served prediction should carry its attribution"
    assert all(f["direction"] in ("increases risk", "decreases risk") for f in factors)
    assert all(0.0 <= f["share"] <= 1.0 for f in factors)
    # Labels are human-readable, not column names.
    assert any(" " in f["label"] for f in factors)
    # Ordered by importance.
    shares = [f["share"] for f in factors]
    assert shares == sorted(shares, reverse=True)


def test_queue_is_ordered_by_the_raw_score_not_the_calibrated_one(client):
    """Isotonic calibration creates ties; the raw score keeps the ranking exact."""
    items = client.get(f"{API}/orders?snapshot_id=2018-08-15&limit=50").json()["items"]
    ranks = [i["ranking_score"] for i in items]
    assert ranks == sorted(ranks, reverse=True)


def test_prediction_for_unknown_order_is_404(client):
    r = client.post(
        f"{API}/predictions", json={"order_id": "nope", "snapshot_id": "2018-08-15"}
    )
    assert r.status_code == 404


def test_prediction_is_unavailable_not_fabricated_when_artifact_missing(
    client, seeded_order, monkeypatch
):
    """With no artifact the API must refuse, never invent a number."""
    from app.ml import predictor as predictor_module

    order_id, snapshot_id = seeded_order
    monkeypatch.setattr(predictor_module.predictor, "_model", None)
    monkeypatch.setattr(predictor_module.predictor, "_error", "artifact removed for test")
    try:
        r = client.post(
            f"{API}/predictions", json={"order_id": order_id, "snapshot_id": snapshot_id}
        )
        assert r.status_code == 503
        body = r.json()["error"]
        assert body["code"] == "service_unavailable"
        assert "risk_probability" not in str(body)
    finally:
        from app.core.config import get_settings
        predictor_module.predictor.load(get_settings().artifact_dir)


def test_meta_exposes_versions_and_attribution(client):
    meta = client.get(f"{API}/meta").json()
    assert meta["model_version"]
    assert meta["policy_version"].startswith("demo-policy-")
    assert meta["calibrated"] is True
    assert meta["band_thresholds"]["high"] > meta["band_thresholds"]["medium"]
    assert "CC BY-NC-SA" in meta["dataset"]["license"]


def test_no_secret_is_ever_returned_by_the_api(client):
    """Nothing resembling an API key may appear in any public payload."""
    for path in ("/meta", "/health/ready", "/snapshots",
                 "/orders?snapshot_id=2018-08-15&limit=5"):
        text = client.get(f"{API}{path}").text
        assert "sk-ant" not in text
        assert "llm_api_key" not in text.lower()


# ---------------------------------------------------------------------------
# Connection settings for a pooled (Supabase) database
# ---------------------------------------------------------------------------

def test_pooled_dsn_disables_prepared_statements():
    """A transaction-mode pooler breaks psycopg3's prepared statements.

    Without this, queries fail intermittently with "prepared statement does
    not exist" once more than one backend is in play - a bug that only shows
    up under production traffic.
    """
    from app.db.session import _engine_kwargs, _is_pooled

    pooled = "postgresql+psycopg://u:p@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
    assert _is_pooled(pooled)
    kwargs = _engine_kwargs(pooled)
    assert kwargs["connect_args"]["prepare_threshold"] is None
    # A free-tier project cannot spare many connections.
    assert kwargs["pool_size"] <= 3
    assert kwargs["pool_recycle"] < 300


def test_direct_dsn_keeps_normal_pooling():
    from app.db.session import _engine_kwargs, _is_pooled

    direct = "postgresql+psycopg://u:p@127.0.0.1:5432/opspilot"
    assert not _is_pooled(direct)
    kwargs = _engine_kwargs(direct)
    assert "connect_args" not in kwargs


# ---------------------------------------------------------------------------
# Hostile input
#
# A percent-encoded NUL reaches the application as an ordinary string, is
# bound into a query, and PostgreSQL refuses it: "text fields cannot contain
# NUL (0x00) bytes". That produced a 500 on seven endpoints, reachable by any
# anonymous visitor with a URL.
# ---------------------------------------------------------------------------

HOSTILE_VALUES = [
    "", "-1", "0", "99999999", "1e308", "NaN", "' OR 1=1--", "<script>",
    "../../etc/passwd", "%00", "%01", "ab%00cd", "a" * 300, "\U0001F642",
    "2018-13-45", "%", "_", "\\", "2018-08-15' --",
]

HOSTILE_PATHS = [
    "/api/v1/orders?snapshot_id={v}",
    "/api/v1/orders?snapshot_id=2018-08-15&limit={v}",
    "/api/v1/orders?snapshot_id=2018-08-15&offset={v}",
    "/api/v1/orders?snapshot_id=2018-08-15&risk_band={v}",
    "/api/v1/orders?snapshot_id=2018-08-15&sort={v}",
    "/api/v1/orders?snapshot_id=2018-08-15&customer_state={v}",
    "/api/v1/orders?snapshot_id=2018-08-15&search={v}",
    "/api/v1/orders/{v}?snapshot_id=2018-08-15",
    "/api/v1/snapshots/{v}/stats",
    "/api/v1/snapshots/{v}/situations",
    "/api/v1/snapshots/2018-08-15/situations?limit={v}",
    "/api/v1/snapshots/2018-08-15/situations?min_orders={v}",
    "/api/v1/situations/{v}",
    "/api/v1/investigations/{v}",
    "/api/v1/tickets/{v}",
    "/api/v1/audit?limit={v}",
]


def test_no_hostile_value_produces_a_server_error(client):
    """Every rejection must be a deliberate 4xx, never an unhandled 5xx."""
    failures = []
    for template in HOSTILE_PATHS:
        for value in HOSTILE_VALUES:
            url = template.replace("{v}", value)
            response = client.get(url)
            if response.status_code >= 500:
                failures.append(f"{response.status_code} {url[:90]}")
    assert not failures, "unhandled server errors:\n" + "\n".join(failures[:10])


@pytest.mark.parametrize("url", [
    "/api/v1/orders/%00?snapshot_id=2018-08-15",
    "/api/v1/snapshots/%00/stats",
    "/api/v1/tickets/%00",
    "/api/v1/orders?snapshot_id=2018-08-15&customer_state=%00",
    "/api/v1/orders?snapshot_id=2018-08-15&search=ab%00cd",
    "/api/v1/orders?snapshot_id=2018-08-15&search=%01",
])
def test_control_characters_are_refused_at_the_edge(client, url):
    response = client.get(url)
    assert response.status_code == 400, f"{url} returned {response.status_code}"
    assert response.json()["error"]["code"] == "bad_request"


def test_the_guard_does_not_reject_legitimate_requests(client):
    """Unicode, punctuation and ordinary ids must still pass."""
    for url in (
        "/api/v1/snapshots",
        "/api/v1/orders?snapshot_id=2018-08-15&limit=5",
        "/api/v1/orders?snapshot_id=2018-08-15&customer_state=SP",
        "/api/v1/orders?snapshot_id=2018-08-15&search=00",
        "/api/v1/snapshots/2018-08-15/situations?limit=3",
    ):
        assert client.get(url).status_code < 400, url


def test_malformed_post_bodies_are_rejected_without_a_server_error(client):
    bodies = [
        ("/api/v1/predictions", {"order_id": "x", "snapshot_id": "y"}),
        ("/api/v1/predictions", {"features": {"a": 1}}),
        ("/api/v1/predictions", {}),
        ("/api/v1/predictions", {"features": None}),
        ("/api/v1/investigations", {}),
        ("/api/v1/investigations", {"order_id": None, "snapshot_id": None}),
        ("/api/v1/investigations", {"order_id": "'--", "snapshot_id": "2018-08-15"}),
        ("/api/v1/investigations", {"order_id": "a" * 500, "snapshot_id": "2018-08-15"}),
    ]
    for url, body in bodies:
        response = client.post(url, json=body)
        assert response.status_code < 500, f"{url} {body} -> {response.status_code}"


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

def test_a_server_error_is_traceable_without_leaking_anything(app_instance, monkeypatch):
    """A generic message is right; a generic message with no id is useless.

    The request id is what connects a visitor saying "it broke" to the log
    line that says why. The middleware also has to put it on request.state,
    which it did not: every route logged request_id=None.
    """
    from app.services import orders as orders_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("table order_outcomes column is_late, password=hunter2")

    monkeypatch.setattr(orders_module, "list_snapshots", boom)
    # The shared client re-raises server exceptions, which is right for every
    # other test; here the rendered 500 is the subject.
    from fastapi.testclient import TestClient

    with TestClient(app_instance, raise_server_exceptions=False) as raw:
        response = raw.get("/api/v1/snapshots")

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"

    request_id = body["error"]["detail"]["request_id"]
    assert request_id, "a 500 carried no request id"
    assert response.headers.get("X-Request-ID") == request_id

    # And nothing internal escaped with it.
    for leak in ("order_outcomes", "is_late", "hunter2", "RuntimeError", "Traceback"):
        assert leak not in response.text


def test_successful_responses_also_carry_the_request_id(client):
    response = client.get("/api/v1/snapshots")
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID")


def test_routes_receive_the_request_id_rather_than_none(client, monkeypatch):
    """Guards the specific bug: request.state.request_id was never assigned."""
    seen = {}

    from app.api.v1 import routes as routes_module

    original = routes_module.log.info

    def capture(event, **kw):
        if event == "situation_investigation_request":
            seen.update(kw)
        return original(event, **kw)

    monkeypatch.setattr(routes_module.log, "info", capture)
    client.post(
        "/api/v1/situations/2018-08-15__SP-RJ/investigations?mode=deterministic"
    )
    if seen:
        assert seen.get("request_id") is not None, "route logged request_id=None"


# ---------------------------------------------------------------------------
# Documentation that drifts is documentation that misleads
# ---------------------------------------------------------------------------

def test_the_readme_states_the_real_test_count(request):
    """Stop the published test count going stale.

    The README quoted 181 while the suite had 221, and PROGRESS quoted 117.
    A number in a README is a claim; this makes it a checked one.
    """
    import re
    from pathlib import Path

    readme = Path(__file__).resolve().parents[2] / "README.md"
    text = readme.read_text(encoding="utf-8")
    match = re.search(r"pytest backend/tests\s+#\s*(\d+)\s+backend tests", text)
    assert match, "README no longer states a backend test count in the expected form"

    claimed = int(match.group(1))
    actual = request.session.testscollected
    assert claimed == actual, (
        f"README claims {claimed} backend tests; the suite collected {actual}. "
        "Update the README."
    )
