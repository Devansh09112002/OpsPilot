"""Fixed historical snapshots taken from the held-out test period.

A snapshot answers: "which orders were in the carrier network on date D?"
Membership is derived from event timestamps only, never from the terminal
`order_status` label, because that label already encodes the outcome
(plan section 3.4).
"""

from __future__ import annotations

import pandas as pd

from data_pipeline import spec


def build_snapshot_members(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (snapshot_id, order_id) that was in transit.

    `is_overdue` separates orders already past their promised date at the
    snapshot moment. Those are shown in a distinct status and excluded from
    the ranking demo, so the product never presents an already-known event as
    an impressive prediction.
    """
    # Only held-out orders may appear in the demo. An order handed over before
    # the test cutoff can still be in transit on a snapshot date, but it was
    # used for model selection, so showing its score would overstate the demo.
    joined = features[features["split"] == "test"].merge(
        outcomes, on="order_id", how="inner", validate="one_to_one"
    )
    rows: list[pd.DataFrame] = []

    for snap in spec.SNAPSHOTS:
        at = pd.Timestamp(snap["snapshot_id"])
        in_transit = joined[
            (joined["order_delivered_carrier_date"] <= at)
            & (joined["order_delivered_customer_date"] > at)
        ].copy()
        if in_transit.empty:
            continue
        in_transit["snapshot_id"] = snap["snapshot_id"]
        in_transit["snapshot_at"] = at
        in_transit["is_overdue"] = (
            in_transit["order_estimated_delivery_date"].dt.normalize() < at.normalize()
        )
        in_transit["days_in_transit"] = (
            at - in_transit["order_delivered_carrier_date"]
        ).dt.total_seconds() / 86400.0
        rows.append(
            in_transit[[
                "snapshot_id", "snapshot_at", "order_id",
                "is_overdue", "days_in_transit",
            ]]
        )

    if not rows:
        raise RuntimeError("no snapshot produced any in-transit order")
    return pd.concat(rows, ignore_index=True)


def validate_snapshots(members: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Assert every snapshot is usable for the demo, returning its sizes."""
    stats: dict[str, dict] = {}
    test_ids = set(features.loc[features["split"] == "test", "order_id"])

    for snap in spec.SNAPSHOTS:
        sid = snap["snapshot_id"]
        sub = members[members["snapshot_id"] == sid]
        if sub.empty:
            raise RuntimeError(f"snapshot {sid} is empty")
        leaked = set(sub["order_id"]) - test_ids
        if leaked:
            raise RuntimeError(
                f"snapshot {sid} contains {len(leaked)} orders outside the held-out "
                "test split; the demo would be scored by a model that saw them."
            )
        pre_deadline = int((~sub["is_overdue"]).sum())
        if pre_deadline < spec.REVIEW_CAPACITY_K:
            raise RuntimeError(
                f"snapshot {sid} has only {pre_deadline} pre-deadline orders, "
                f"fewer than the review capacity K={spec.REVIEW_CAPACITY_K}"
            )
        stats[sid] = {
            "label": snap["label"],
            "in_transit": int(len(sub)),
            "pre_deadline": pre_deadline,
            "overdue": int(sub["is_overdue"].sum()),
        }
    return stats
