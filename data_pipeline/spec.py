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

# SHA256 of the upstream Olist CSVs, cross-verified byte-for-byte against two
# independent public mirrors before first use. `data_pipeline.fetch` refuses
# any file that does not match, so the pipeline cannot train on unknown data.
SOURCE_SHA256_FULL: dict[str, str] = {
    "olist_orders_dataset.csv":
        "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
    "olist_order_items_dataset.csv":
        "0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279",
    "olist_customers_dataset.csv":
        "983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2",
    "olist_sellers_dataset.csv":
        "1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e",
    "olist_products_dataset.csv":
        "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
    "product_category_name_translation.csv":
        "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
}

# Uppercase prefixes used by the audit report's inventory table.
SOURCE_SHA256: dict[str, str] = {
    name: digest[:24].upper() for name, digest in SOURCE_SHA256_FULL.items()
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

# The calibrated probability at or above which the demo policy permits a
# carrier escalation (with <= 3 days of slack). Stated on the CALIBRATED
# estimate, so it means what it says: roughly a one-in-seven chance of missing
# the date, against a marketplace baseline near 3%.
#
# The "high" risk band is cut here too. A band that disagreed with the policy
# would produce the confusing state of a "medium risk" order being escalated,
# which is exactly what the benchmark caught when they were set independently.
ESCALATION_THRESHOLD = 0.15

RANDOM_SEED = 42
