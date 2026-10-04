"""Load the served model's snapshot scores into `snapshot_orders`.

The demo snapshots are fixed historical days, so every order's snapshot-day
score, priority rank and arrival estimate is computed once, offline, by
`python -m ml_pipeline.train_v4` and shipped in
`artifacts/v4/snapshot_scores.parquet`. This module writes them to the
database. The API calls it at start-up, which is how a deployed database is
brought up to date by the deploy itself, with no separate credential.

It refuses to write if the shipped scores and the database disagree on which
orders belong to which snapshot: a partial or mismatched update would show
scores for orders the model never ranked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, MetaData, Table, func, insert, select, text
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import SnapshotOrder

log = get_logger(__name__)

# Overdue orders are not ranked; this keeps them after every ranked order.
UNRANKED_SCORE = -1.0


class ScoreSyncError(RuntimeError):
    """The shipped scores do not match the orders in the database."""


def scores_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / "v4" / "snapshot_scores.parquet"


def _rows(scores: pd.DataFrame) -> list[dict]:
    def date_or_none(v):
        return None if pd.isna(v) else pd.Timestamp(v).date()

    out = []
    for r in scores.itertuples(index=False):
        out.append({
            "b_snapshot_id": r.snapshot_id,
            "b_order_id": r.order_id,
            "risk_probability": float(r.risk_probability),
            "ranking_score": (UNRANKED_SCORE if r.is_overdue else float(r.ensemble_score)),
            "risk_band": r.risk_band,
            "model_version": r.model_version,
            "priority_rank": None if pd.isna(r.priority_rank) else int(r.priority_rank),
            "km_score": None if pd.isna(r.km_score) else float(r.km_score),
            "hazard_score": float(r.hazard_score),
            "lambdamart_score": None if pd.isna(r.lambdamart_score) else float(r.lambdamart_score),
            "eta_p10": date_or_none(r.eta_p10),
            "eta_p50": date_or_none(r.eta_p50),
            "eta_p90": date_or_none(r.eta_p90),
            "snapshot_features": (None if r.snapshot_features is None
                                  else json.loads(r.snapshot_features)),
        })
    return out


def bulk_update(db: Session, rows: list[dict]) -> None:
    """Update many snapshot_orders rows in one statement.

    Rows go into a temporary table with batched multi-row inserts, then one
    UPDATE ... FROM applies them. One row at a time took a minute for 3,955
    rows against a remote pooler; this is a handful of round trips. The temp
    table lives inside the transaction, which a transaction-mode pooler allows.
    """
    if not rows:
        return
    target = SnapshotOrder.__table__
    columns = [c for c in rows[0] if c not in ("b_snapshot_id", "b_order_id")]
    temp = Table(
        "_score_sync", MetaData(),
        Column("b_snapshot_id", target.c.snapshot_id.type),
        Column("b_order_id", target.c.order_id.type),
        *[Column(c, target.c[c].type) for c in columns],
        prefixes=["TEMPORARY"],
        postgresql_on_commit="DROP",
    )
    conn = db.connection()
    temp.create(conn)
    conn.execute(insert(temp), rows)
    assignments = ", ".join(f"{c} = t.{c}" for c in columns)
    conn.execute(text(
        f"UPDATE snapshot_orders s SET {assignments} FROM _score_sync t "
        "WHERE s.snapshot_id = t.b_snapshot_id AND s.order_id = t.b_order_id"
    ))


def _differing_rows(db: Session, scores: pd.DataFrame) -> int:
    """How many stored rows differ from the shipped ones, by content.

    Comparing only `model_version` missed a real difference in production: the
    same version name had been loaded with an earlier tie-break, so two ranks
    stayed swapped after a deploy that "found nothing to do".
    """
    stored = pd.DataFrame(
        db.execute(select(
            SnapshotOrder.snapshot_id, SnapshotOrder.order_id, SnapshotOrder.model_version,
            SnapshotOrder.priority_rank, SnapshotOrder.risk_band,
            SnapshotOrder.ranking_score, SnapshotOrder.eta_p50,
        )).all(),
        columns=["snapshot_id", "order_id", "model_version", "priority_rank",
                 "risk_band", "ranking_score", "eta_p50"],
    )
    want = scores.assign(
        ranking_score=scores["ensemble_score"].where(~scores["is_overdue"], UNRANKED_SCORE),
        eta_p50=pd.to_datetime(scores["eta_p50"]).dt.date,
    )[stored.columns]
    m = want.merge(stored, on=["snapshot_id", "order_id"], how="outer",
                   suffixes=("", "_db"), indicator=True)
    differs = (
        (m["_merge"] != "both")
        | (m["model_version"] != m["model_version_db"])
        | (m["priority_rank"].fillna(-1).astype(float)
           != m["priority_rank_db"].fillna(-1).astype(float))
        | (m["risk_band"] != m["risk_band_db"])
        | ((m["ranking_score"] - m["ranking_score_db"]).abs() > 1e-12)
        | (m["eta_p50"] != m["eta_p50_db"])
    )
    return int(differs.sum())


def sync_snapshot_scores(db: Session, artifact_dir: Path, *, force: bool = False) -> dict:
    """Bring `snapshot_orders` up to the shipped scores. Idempotent."""
    path = scores_path(artifact_dir)
    if not path.exists():
        raise ScoreSyncError(f"shipped scores not found at {path}")
    scores = pd.read_parquet(path)
    version = str(scores["model_version"].iloc[0])

    total = db.execute(select(func.count()).select_from(SnapshotOrder)).scalar_one()
    if total and not force and _differing_rows(db, scores) == 0:
        return {"status": "current", "model_version": version, "updated": 0}

    stored = set(db.execute(select(SnapshotOrder.snapshot_id, SnapshotOrder.order_id)).all())
    shipped = set(zip(scores["snapshot_id"], scores["order_id"], strict=True))
    if stored != shipped:
        raise ScoreSyncError(
            f"snapshot membership differs: {len(stored - shipped)} stored orders have no "
            f"shipped score, {len(shipped - stored)} shipped scores have no stored order"
        )

    rows = _rows(scores)
    bulk_update(db, rows)
    db.commit()
    log.info("snapshot_scores_synced", model_version=version, rows=len(rows))
    return {"status": "updated", "model_version": version, "updated": len(rows)}
