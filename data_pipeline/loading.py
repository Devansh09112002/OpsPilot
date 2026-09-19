"""Raw Olist loading, integrity checks and eligibility rules.

This module owns the single definition of "eligible order" and of the
prediction target. Everything downstream (audit, features, training,
serving) imports it so the rules cannot drift apart.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_pipeline import spec

ORDER_TIMESTAMPS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


class DataIntegrityError(RuntimeError):
    """Raised when a raw input is missing, truncated or structurally wrong."""


@dataclass
class ExclusionLog:
    """Counts every order removed by the eligibility funnel, in order."""

    total: int = 0
    steps: list[tuple[str, int]] = field(default_factory=list)
    eligible: int = 0

    def record(self, label: str, removed: int) -> None:
        self.steps.append((label, int(removed)))

    def as_dict(self) -> dict:
        return {
            "total_orders": self.total,
            "exclusions": [{"reason": r, "orders_removed": n} for r, n in self.steps],
            "eligible_orders": self.eligible,
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_raw(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the required Olist CSVs, failing loudly on any missing input.

    A missing file or column is an actionable validation error, never a
    silent partial success (plan section 9.2, "Ingestion").
    """
    raw_dir = raw_dir or spec.RAW_DIR
    required_columns = {
        "orders": {"order_id", "customer_id", "order_status", *ORDER_TIMESTAMPS},
        "order_items": {
            "order_id", "order_item_id", "product_id", "seller_id",
            "shipping_limit_date", "price", "freight_value",
        },
        "customers": {"customer_id", "customer_unique_id", "customer_state", "customer_city"},
        "sellers": {"seller_id", "seller_state", "seller_city"},
        "products": {"product_id", "product_category_name"},
        "category_translation": {"product_category_name", "product_category_name_english"},
    }

    frames: dict[str, pd.DataFrame] = {}
    for key, filename in spec.SOURCE_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise DataIntegrityError(
                f"Required Olist file missing: {path}. "
                "Run `python -m data_pipeline.fetch` or follow docs/data_audit.md."
            )
        if path.stat().st_size == 0:
            raise DataIntegrityError(f"Required Olist file is empty: {path}")
        df = pd.read_csv(path)
        missing = required_columns[key] - set(df.columns)
        if missing:
            raise DataIntegrityError(
                f"{filename} is missing required column(s): {sorted(missing)}"
            )
        frames[key] = df

    for col in ORDER_TIMESTAMPS:
        frames["orders"][col] = pd.to_datetime(frames["orders"][col], errors="coerce")
    frames["order_items"]["shipping_limit_date"] = pd.to_datetime(
        frames["order_items"]["shipping_limit_date"], errors="coerce"
    )

    if not frames["orders"]["order_id"].is_unique:
        raise DataIntegrityError("order_id is not unique in olist_orders_dataset.csv")

    return frames


def compute_target(orders: pd.DataFrame) -> pd.Series:
    """Late = delivered on a later CALENDAR DATE than promised.

    Olist stores `order_estimated_delivery_date` at midnight, so a raw
    timestamp comparison wrongly marks every same-day afternoon delivery
    as late (1,291 orders in this release). Plan section 3.2 mandates the
    calendar-date rule; `normalize()` implements it.
    """
    return (
        orders["order_delivered_customer_date"].dt.normalize()
        > orders["order_estimated_delivery_date"].dt.normalize()
    )


def select_eligible(orders: pd.DataFrame) -> tuple[pd.DataFrame, ExclusionLog]:
    """Apply the training-eligibility funnel from plan section 3.2.

    Cancelled / unresolved orders are excluded and counted, never silently
    relabelled as on-time.
    """
    log = ExclusionLog(total=len(orders))
    df = orders

    # Each predicate is evaluated against the *current* frame, so the counts
    # read as a funnel: an order is attributed to the first rule it violates.
    rules: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("missing carrier handover timestamp",
         lambda d: d["order_delivered_carrier_date"].notna()),
        ("missing estimated delivery date",
         lambda d: d["order_estimated_delivery_date"].notna()),
        ("missing actual delivery timestamp",
         lambda d: d["order_delivered_customer_date"].notna()),
        ("purchase timestamp after carrier handover",
         lambda d: d["order_purchase_timestamp"] <= d["order_delivered_carrier_date"]),
        ("carrier handover after customer delivery",
         lambda d: d["order_delivered_carrier_date"] <= d["order_delivered_customer_date"]),
        ("order_status is not 'delivered'",
         lambda d: d["order_status"] == "delivered"),
        # Handover after the promised date makes lateness an arithmetic
        # certainty (delivery >= handover > promise), not a prediction. Such
        # orders are always 'overdue' rather than queued, so including them
        # would inflate every ranking metric. See docs/data_audit.md section 2.
        ("carrier handover after the promised date (outcome already certain)",
         lambda d: d["order_delivered_carrier_date"].dt.normalize()
                   <= d["order_estimated_delivery_date"].dt.normalize()),
    ]
    for label, predicate in rules:
        keep = predicate(df)
        log.record(label, int((~keep).sum()))
        df = df[keep]

    df = df.copy()
    df[spec.TARGET_NAME] = compute_target(df)
    log.eligible = len(df)
    return df, log


def assign_split(handover: pd.Series) -> pd.Series:
    """Chronological split on the prediction moment. Frozen in spec.py."""
    return pd.Series(
        pd.cut(
            handover,
            bins=[pd.Timestamp.min, spec.TRAIN_END, spec.VALIDATION_END, pd.Timestamp.max],
            labels=["train", "validation", "test"],
            right=False,
        ),
        index=handover.index,
    ).astype(str)
