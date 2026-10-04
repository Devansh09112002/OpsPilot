"""The served v4 model: bands, ranks, live parity and artifact integrity.

The ranking is the validated quantity, so the invariants here are about it:
"high" is exactly the day's top 50, ranks follow the queue's own order, and a
score recomputed live equals the stored one.
"""

from __future__ import annotations

import json
import shutil
from datetime import date

import pytest
from sqlalchemy import select

from app.db.models import SnapshotOrder
from data_pipeline import spec

SNAPSHOT = "2018-08-15"


def _ranked(db):
    return db.execute(
        select(SnapshotOrder).where(SnapshotOrder.snapshot_id == SNAPSHOT,
                                    SnapshotOrder.is_overdue.is_(False))
    ).scalars().all()


def test_high_band_is_exactly_the_days_review_list(db):
    """One cut in the system: the band a user reads and the policy's review list."""
    rows = _ranked(db)
    assert rows, "the seeded snapshot has no ranked orders"
    for r in rows:
        assert (r.risk_band == "high") == (r.priority_rank <= spec.REVIEW_CAPACITY_K)
        assert r.risk_band == spec.priority_band(r.priority_rank, len(rows))
    assert sum(r.risk_band == "high" for r in rows) == spec.REVIEW_CAPACITY_K


def test_ranks_are_the_queue_order(db):
    """Rank 1..n, in exactly the order the API sorts: score desc, order id asc."""
    rows = _ranked(db)
    ordered = sorted(rows, key=lambda r: (-r.ranking_score, r.order_id))
    assert [r.priority_rank for r in ordered] == list(range(1, len(rows) + 1))


def test_overdue_orders_are_not_ranked(db):
    overdue = db.execute(
        select(SnapshotOrder).where(SnapshotOrder.is_overdue.is_(True))
    ).scalars().all()
    if not overdue:
        pytest.skip("no overdue orders in the seeded slice")
    assert all(r.priority_rank is None and r.risk_probability == 1.0 for r in overdue)


def test_a_live_score_equals_the_stored_one(db, loaded_model):
    """Parity across the whole queue's shape: top, middle and bottom."""
    rows = sorted(_ranked(db), key=lambda r: r.priority_rank)
    sample = rows[:15] + rows[len(rows) // 2: len(rows) // 2 + 10] + rows[-10:]
    for so in sample:
        live = loaded_model.score(db, so.order_id, so.snapshot_id, with_factors=False)
        assert live.priority_rank == so.priority_rank
        assert live.band == so.risk_band
        assert live.probability == pytest.approx(so.risk_probability, abs=1e-12)
        assert live.ranking_score == pytest.approx(so.ranking_score, abs=1e-12)
        assert (live.eta_p10, live.eta_p50, live.eta_p90) == (so.eta_p10, so.eta_p50, so.eta_p90)


def test_arrival_quantiles_are_ordered(db):
    for r in _ranked(db):
        assert r.eta_p10 <= r.eta_p50 <= r.eta_p90


def test_the_arrival_tag_reads_the_forecast_against_the_promise():
    from app.ml.predictor import arrival_tag

    promised = date(2018, 8, 20)
    assert arrival_tag(date(2018, 8, 22), promised, False) == "likely_late"
    assert arrival_tag(date(2018, 8, 20), promised, False) == "tight"
    assert arrival_tag(date(2018, 8, 19), promised, False) == "tight"
    assert arrival_tag(date(2018, 8, 15), promised, False) == "on_track"
    assert arrival_tag(date(2018, 8, 15), promised, True) == "overdue"


def test_a_tampered_artifact_is_refused(tmp_path):
    from app.core.config import get_settings
    from app.ml.predictor import Predictor

    shutil.copytree(get_settings().artifact_dir / "v4", tmp_path / "v4")
    (tmp_path / "v4" / "hazard.ubj").write_bytes(b"not a model")
    p = Predictor()
    p.load(tmp_path)
    assert not p.available
    assert "checksum" in (p.error or "")


def test_a_missing_artifact_is_refused(tmp_path):
    from app.ml.predictor import Predictor

    p = Predictor()
    p.load(tmp_path)
    assert not p.available and p.error


def test_the_served_model_records_its_published_test_result():
    from app.core.config import get_settings

    meta = json.loads((get_settings().artifact_dir / "v4" / "meta.json").read_text())
    published = json.loads((spec.DOCS_DIR / "research_v4_test.json").read_text())
    assert meta["test_precision_at_50"] == published["mean_precision_at_50"]["ensemble_rank"]
    assert meta["training_cutoff"] == "2018-06-01"
