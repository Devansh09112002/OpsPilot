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
    assert served["model_version"] == listed["model_version"]
    assert served["risk_band"] == listed["risk_band"]
    assert "carrier handover" in served["disclaimer"].lower()


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
    assert meta["policy_version"] == "demo-policy-v1"
    assert "CC BY-NC-SA" in meta["dataset"]["license"]


def test_no_secret_is_ever_returned_by_the_api(client):
    """Nothing resembling an API key may appear in any public payload."""
    for path in ("/meta", "/health/ready", "/snapshots",
                 "/orders?snapshot_id=2018-08-15&limit=5"):
        text = client.get(f"{API}{path}").text
        assert "sk-ant" not in text
        assert "llm_api_key" not in text.lower()
