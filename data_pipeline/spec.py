"""Frozen data & modelling specification for OpsPilot.

Every constant here was fixed after the initial data inspection
(see docs/data_audit.md) and BEFORE any model comparison was run, as
required by OpsPilot_Project_Plan.md section 3.4. Changing a split date
invalidates the published evaluation in docs/model_report.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
DOCS_DIR = REPO_ROOT / "docs"

# --- Source files (subset of the Olist release that the product justifies using) ---
# olist_order_reviews / payments / geolocation are deliberately NOT ingested:
# reviews are a forbidden post-outcome signal, payments add no as-of value at
# carrier handover, and geolocation is redundant with the coarse state fields.
SOURCE_FILES: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# SHA256 of the upstream Olist CSVs, cross-verified against two independent
# public mirrors on 2026-09-20. Used by the ingestion integrity check.
SOURCE_SHA256: dict[str, str] = {
    "olist_orders_dataset.csv": "8DF58EF3D2D7E9944010F7BE",
    "olist_order_items_dataset.csv": "0BC4D068C4FE38CBB01BD90E",
    "olist_customers_dataset.csv": "983A422239E1712DED753B3B",
    "olist_sellers_dataset.csv": "1F643D2B950373B85735E779",
    "olist_products_dataset.csv": "3E6569628A17FBC75FD206EE",
}

# --- Prediction contract (plan section 3.2) ---
PREDICTION_MOMENT_COLUMN = "order_delivered_carrier_date"
TARGET_NAME = "is_late"

# --- Frozen chronological splits, cut on the prediction moment ---
TRAIN_END = pd.Timestamp("2018-03-01")   # train:      handover <  2018-03-01
VALIDATION_END = pd.Timestamp("2018-06-01")  # validation: 2018-03-01 <= handover < 2018-06-01
# test: handover >= 2018-06-01

# --- Frozen demo snapshots, all inside the held-out test period ---
SNAPSHOTS: list[dict[str, str]] = [
    {"snapshot_id": "2018-06-20", "label": "20 June 2018"},
    {"snapshot_id": "2018-07-18", "label": "18 July 2018"},
    {"snapshot_id": "2018-08-15", "label": "15 August 2018"},
]

# Operational review capacity used for the primary Precision@K metric.
REVIEW_CAPACITY_K = 50

RANDOM_SEED = 42
