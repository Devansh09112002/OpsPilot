"""Loading the shipped model scores into the database.

The API runs this at start-up, so it is how a deploy updates its own database.
It must notice *any* difference from the shipped scores, not just a new
version name: production once kept two swapped ranks through a deploy because
the version string already matched.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import SnapshotOrder
from app.services.model_sync import ScoreSyncError, scores_path, sync_snapshot_scores

SNAPSHOT = "2018-08-15"


@pytest.fixture
def shipped_for_seed(tmp_path):
    """The shipped scores restricted to the seeded snapshot."""
    scores = pd.read_parquet(scores_path(get_settings().artifact_dir))
    out = tmp_path / "v4"
    out.mkdir()
    scores[scores["snapshot_id"] == SNAPSHOT].to_parquet(out / "snapshot_scores.parquet",
                                                         index=False)
    return tmp_path


def _two_ranked(db):
    return db.execute(
        select(SnapshotOrder).where(SnapshotOrder.snapshot_id == SNAPSHOT,
                                    SnapshotOrder.priority_rank.in_([13, 14]))
    ).scalars().all()


def test_nothing_to_do_when_the_database_matches(db, shipped_for_seed):
    assert sync_snapshot_scores(db, shipped_for_seed)["status"] == "current"


def test_a_content_difference_is_repaired_even_with_the_same_version(db, shipped_for_seed):
    a, b = _two_ranked(db)
    a.priority_rank, b.priority_rank = b.priority_rank, a.priority_rank
    db.commit()

    result = sync_snapshot_scores(db, shipped_for_seed)
    assert result["status"] == "updated"

    db.expire_all()
    shipped = pd.read_parquet(shipped_for_seed / "v4" / "snapshot_scores.parquet")
    expected = dict(zip(shipped["order_id"], shipped["priority_rank"], strict=True))
    for row in _two_ranked(db):
        assert row.priority_rank == expected[row.order_id]
    assert sync_snapshot_scores(db, shipped_for_seed)["status"] == "current"


def test_a_membership_mismatch_is_refused(db, shipped_for_seed, tmp_path_factory):
    scores = pd.read_parquet(shipped_for_seed / "v4" / "snapshot_scores.parquet")
    other = tmp_path_factory.mktemp("partial")
    (other / "v4").mkdir()
    scores.iloc[:-5].to_parquet(other / "v4" / "snapshot_scores.parquet", index=False)
    with pytest.raises(ScoreSyncError, match="membership differs"):
        sync_snapshot_scores(db, other, force=True)
