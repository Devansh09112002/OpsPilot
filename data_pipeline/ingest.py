"""Load the validated feature tables and snapshot queues into PostgreSQL.

Risk scores written here come from the same artifact the API serves, loaded
through the same `load_artifact` / `predict_risk` path. A parity test asserts
that a stored score equals a freshly served one; nothing synthetic is inserted.

Rerunning is idempotent: each table is replaced wholesale inside one
transaction, so a partial run cannot leave the database half-populated.
"""

from __future__ import annotations

import sys

import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "backend"))

from app.db.models import (  # noqa: E402
    OrderFeature,
    OrderOutcome,
    Snapshot,
    SnapshotOrder,
)
from app.db.session import SessionLocal, engine  # noqa: E402
from data_pipeline import spec  # noqa: E402
from data_pipeline.features import MODEL_FEATURES  # noqa: E402
from ml_pipeline.model import load_artifact, predict_risk  # noqa: E402

BATCH = 2000

# Risk bands are presentation only. They are cut on the served score, and the
# UI always shows the numeric probability alongside the band.
HIGH_BAND = 0.60
MEDIUM_BAND = 0.30


def risk_band(p: float) -> str:
    if p >= HIGH_BAND:
        return "high"
    if p >= MEDIUM_BAND:
        return "medium"
    return "low"


def _require(path):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m data_pipeline.audit` first.")
    return pd.read_parquet(path)


def ingest() -> dict:
    features = _require(spec.PROCESSED_DIR / "order_features.parquet")
    outcomes = _require(spec.PROCESSED_DIR / "order_outcomes.parquet")
    members = _require(spec.PROCESSED_DIR / "snapshot_members.parquet")

    model, meta = load_artifact(spec.ARTIFACT_DIR)
    model_version = meta["model_version"]
    print(f"scoring with artifact {model_version} (sha {meta['artifact_sha256'][:12]})")

    # An order can be in transit across several snapshots, so membership is
    # many-to-one against the one-row-per-order feature table.
    scored = members.merge(features, on="order_id", how="inner", validate="many_to_one")
    scored["risk_probability"] = predict_risk(model, scored)
    scored["risk_band"] = [risk_band(p) for p in scored["risk_probability"]]

    stats: dict[str, int] = {}
    with SessionLocal() as db:  # type: Session
        # Children first: snapshot_orders and outcomes reference order_features.
        db.execute(delete(SnapshotOrder))
        db.execute(delete(Snapshot))
        db.execute(delete(OrderOutcome))
        db.execute(delete(OrderFeature))
        db.flush()

        feature_rows = [
            {
                "order_id": r.order_id,
                "split": r.split,
                "order_purchase_timestamp": r.order_purchase_timestamp.to_pydatetime(),
                "order_approved_at": (
                    None if pd.isna(r.order_approved_at) else r.order_approved_at.to_pydatetime()
                ),
                "order_delivered_carrier_date": r.order_delivered_carrier_date.to_pydatetime(),
                "order_estimated_delivery_date": r.order_estimated_delivery_date.to_pydatetime(),
                "features": {
                    f: (None if pd.isna(v := getattr(r, f)) else
                        (v.item() if hasattr(v, "item") else v))
                    for f in MODEL_FEATURES
                },
                "customer_state": None if pd.isna(r.customer_state) else str(r.customer_state),
                "seller_state": None if pd.isna(r.seller_state) else str(r.seller_state),
                "product_category": (
                    None if pd.isna(r.product_category) else str(r.product_category)
                ),
                "n_items": int(r.n_items),
                "total_price": float(r.total_price),
                "total_freight": float(r.total_freight),
            }
            for r in features.itertuples(index=False)
        ]
        for i in range(0, len(feature_rows), BATCH):
            db.bulk_insert_mappings(OrderFeature, feature_rows[i:i + BATCH])
        stats["order_features"] = len(feature_rows)

        outcome_rows = [
            {
                "order_id": r.order_id,
                "order_delivered_customer_date": r.order_delivered_customer_date.to_pydatetime(),
                "is_late": bool(r.is_late),
            }
            for r in outcomes.itertuples(index=False)
        ]
        for i in range(0, len(outcome_rows), BATCH):
            db.bulk_insert_mappings(OrderOutcome, outcome_rows[i:i + BATCH])
        stats["order_outcomes"] = len(outcome_rows)

        snapshot_rows = []
        for s in spec.SNAPSHOTS:
            sid = s["snapshot_id"]
            sub = scored[scored["snapshot_id"] == sid]
            if sub.empty:
                continue
            snapshot_rows.append({
                "snapshot_id": sid,
                "label": s["label"],
                "snapshot_at": pd.Timestamp(sid).to_pydatetime(),
                "orders_in_transit": int(len(sub)),
                "orders_pre_deadline": int((~sub["is_overdue"]).sum()),
                "orders_overdue": int(sub["is_overdue"].sum()),
            })
        db.bulk_insert_mappings(Snapshot, snapshot_rows)
        stats["snapshots"] = len(snapshot_rows)

        member_rows = [
            {
                "snapshot_id": r.snapshot_id,
                "order_id": r.order_id,
                "is_overdue": bool(r.is_overdue),
                "days_in_transit": float(r.days_in_transit),
                "risk_probability": float(r.risk_probability),
                "risk_band": r.risk_band,
                "model_version": model_version,
            }
            for r in scored.itertuples(index=False)
        ]
        for i in range(0, len(member_rows), BATCH):
            db.bulk_insert_mappings(SnapshotOrder, member_rows[i:i + BATCH])
        stats["snapshot_orders"] = len(member_rows)

        db.commit()

    with engine.connect() as conn:
        conn.execute(text("ANALYZE"))
        conn.commit()

    return stats


def main() -> int:
    stats = ingest()
    print("INGEST COMPLETE")
    for table, n in stats.items():
        print(f"  {table:<18s} {n:>7,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
