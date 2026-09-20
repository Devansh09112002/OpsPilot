"""Test fixtures.

Tests run against a real PostgreSQL database (`opspilot_test`) seeded from a
small slice of the real ingested data. Nothing is mocked that the product
relies on for correctness - only the paid LLM call is stubbed, and there is an
explicit test that the stub is never silently substituted in production code.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://opspilot:opspilot_dev_pw@127.0.0.1:5433/opspilot_test",
)
# Tests must never spend provider quota: every LLM path is stubbed, and an
# accidental real call would be both slow and billable against the free tier.
os.environ["LLM_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ.setdefault("ENVIRONMENT", "test")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import (
    Base,
    GuestSession,
    OrderFeature,
    OrderOutcome,
    Snapshot,
    SnapshotOrder,
)

# The already-ingested database the test slice is copied from. CI runs its
# PostgreSQL service on 5432, the local portable cluster on 5433, so this is
# read from the environment rather than hardcoded.
SOURCE_URL = os.environ.get(
    "OPSPILOT_TEST_SOURCE_URL",
    "postgresql+psycopg://opspilot:opspilot_dev_pw@127.0.0.1:5433/opspilot",
)

SEED_SNAPSHOT = "2018-08-15"
SEED_ORDERS = 400


@pytest.fixture(scope="session")
def test_engine():
    settings = get_settings()
    engine = create_engine(settings.database_url, future=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    _seed(engine)
    yield engine
    engine.dispose()


def _seed(engine) -> None:
    """Copy a representative slice of real data into the test database.

    Using real rows matters: a hand-written fixture would not exercise the
    JSONB feature documents the served model actually scores.
    """
    src_engine = create_engine(SOURCE_URL, future=True)
    SrcSession = sessionmaker(bind=src_engine, future=True)
    DstSession = sessionmaker(bind=engine, future=True)

    with SrcSession() as src, DstSession() as dst:
        snapshots = src.execute(select(Snapshot)).scalars().all()
        if not snapshots:
            pytest.skip("source database is not ingested; run data_pipeline.ingest")

        members = src.execute(
            select(SnapshotOrder)
            .where(SnapshotOrder.snapshot_id == SEED_SNAPSHOT)
            .order_by(SnapshotOrder.risk_probability.desc())
            .limit(SEED_ORDERS)
        ).scalars().all()
        order_ids = [m.order_id for m in members]

        features = src.execute(
            select(OrderFeature).where(OrderFeature.order_id.in_(order_ids))
        ).scalars().all()
        outcomes = src.execute(
            select(OrderOutcome).where(OrderOutcome.order_id.in_(order_ids))
        ).scalars().all()
        # Extra historical rows so as-of aggregates have a real sample to work on.
        history = src.execute(
            select(OrderFeature, OrderOutcome)
            .join(OrderOutcome, OrderOutcome.order_id == OrderFeature.order_id)
            .where(OrderOutcome.order_delivered_customer_date < datetime(2018, 8, 15))
            .limit(4000)
        ).all()

        for s in snapshots:
            dst.add(Snapshot(
                snapshot_id=s.snapshot_id, label=s.label, snapshot_at=s.snapshot_at,
                orders_in_transit=s.orders_in_transit,
                orders_pre_deadline=s.orders_pre_deadline,
                orders_overdue=s.orders_overdue,
            ))
        seen: set[str] = set()
        for f in features:
            seen.add(f.order_id)
            dst.add(_clone_feature(f))
        for hf, ho in history:
            if hf.order_id in seen:
                continue
            seen.add(hf.order_id)
            dst.add(_clone_feature(hf))
            dst.add(OrderOutcome(
                order_id=ho.order_id,
                order_delivered_customer_date=ho.order_delivered_customer_date,
                is_late=ho.is_late,
            ))
        dst.flush()
        for o in outcomes:
            if o.order_id in seen and not dst.get(OrderOutcome, o.order_id):
                dst.add(OrderOutcome(
                    order_id=o.order_id,
                    order_delivered_customer_date=o.order_delivered_customer_date,
                    is_late=o.is_late,
                ))
        for m in members:
            dst.add(SnapshotOrder(
                snapshot_id=m.snapshot_id, order_id=m.order_id,
                is_overdue=m.is_overdue, days_in_transit=m.days_in_transit,
                risk_probability=m.risk_probability, risk_band=m.risk_band,
                model_version=m.model_version,
            ))
        dst.commit()
    src_engine.dispose()


def _clone_feature(f: OrderFeature) -> OrderFeature:
    return OrderFeature(
        order_id=f.order_id, split=f.split,
        order_purchase_timestamp=f.order_purchase_timestamp,
        order_approved_at=f.order_approved_at,
        order_delivered_carrier_date=f.order_delivered_carrier_date,
        order_estimated_delivery_date=f.order_estimated_delivery_date,
        features=f.features, customer_state=f.customer_state,
        seller_state=f.seller_state, product_category=f.product_category,
        n_items=f.n_items, total_price=f.total_price, total_freight=f.total_freight,
    )


@pytest.fixture(scope="session", autouse=True)
def loaded_model():
    """Load the real artifact once. Tests assert on genuine model output."""
    from app.ml.predictor import predictor

    predictor.load(get_settings().artifact_dir)
    if not predictor.available:
        pytest.skip(f"model artifact unavailable: {predictor.error}")
    return predictor


@pytest.fixture
def db(test_engine) -> Session:
    SessionTest = sessionmaker(bind=test_engine, expire_on_commit=False, future=True)
    with SessionTest() as session:
        yield session


@pytest.fixture(scope="session")
def app_instance(test_engine):
    """The real FastAPI app, wired to the test database."""
    from app.db import session as db_session
    from app.main import app
    from app.ml.predictor import predictor

    SessionTest = sessionmaker(bind=test_engine, expire_on_commit=False, future=True)

    def override_get_db():
        s = SessionTest()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[db_session.get_db] = override_get_db
    predictor.load(get_settings().artifact_dir)
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app_instance) -> TestClient:
    """A fresh guest: each client gets its own cookie jar, hence its own session."""
    from app.core.limits import request_limiter

    request_limiter.reset()
    with TestClient(app_instance) as c:
        yield c


@pytest.fixture
def second_client(app_instance) -> TestClient:
    """A different guest, for session-isolation tests."""
    with TestClient(app_instance) as c:
        yield c


@pytest.fixture
def seeded_order(db) -> tuple[str, str]:
    """A high-risk pre-deadline order that should qualify for escalation."""
    row = db.execute(
        select(SnapshotOrder)
        .where(
            SnapshotOrder.snapshot_id == SEED_SNAPSHOT,
            SnapshotOrder.is_overdue.is_(False),
        )
        .order_by(SnapshotOrder.risk_probability.desc())
        .limit(1)
    ).scalar_one()
    return row.order_id, row.snapshot_id


@pytest.fixture
def guest_session(db) -> GuestSession:
    import secrets

    now = datetime.now(UTC).replace(tzinfo=None)
    s = GuestSession(
        session_id=secrets.token_urlsafe(32),
        created_at=now, last_seen_at=now,
        expires_at=now + timedelta(hours=72),
        investigations_today=0, quota_window_start=now,
    )
    db.add(s)
    db.commit()
    return s
