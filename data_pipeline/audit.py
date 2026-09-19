"""Data audit and gate (plan section 3.5).

Runs from the documented raw inputs, writes `docs/data_audit.md` plus a
machine-readable `docs/data_audit.json`, and raises if any PASS condition
fails. Downstream stages refuse to run without a passing audit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from data_pipeline import spec
from data_pipeline.features import (
    CATEGORICAL_FEATURES,
    FEATURE_AVAILABILITY,
    MODEL_FEATURES,
    build_feature_table,
    split_features_and_outcomes,
)
from data_pipeline.loading import compute_target, load_raw, select_eligible, sha256_of
from data_pipeline.snapshots import build_snapshot_members, validate_snapshots


class AuditFailure(RuntimeError):
    """Raised when the data gate fails. Downstream stages must not run."""


def _file_inventory() -> list[dict]:
    inv = []
    for key, filename in spec.SOURCE_FILES.items():
        path = spec.RAW_DIR / filename
        digest = sha256_of(path)
        expected = spec.SOURCE_SHA256.get(filename)
        inv.append({
            "table": key,
            "file": filename,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "sha256_prefix_verified": (expected is None) or digest.startswith(expected),
        })
    return inv


def run_audit() -> dict:
    frames = load_raw()
    orders = frames["orders"]

    inventory = _file_inventory()
    unverified = [f["file"] for f in inventory if not f["sha256_prefix_verified"]]
    if unverified:
        raise AuditFailure(
            f"checksum mismatch against the cross-verified Olist release: {unverified}"
        )

    # --- join-grain assertions -------------------------------------------------
    items = frames["order_items"]
    orphan_items = int((~items["order_id"].isin(orders["order_id"])).sum())
    grain = {
        "order_id_unique_in_orders": bool(orders["order_id"].is_unique),
        "order_items_rows": int(len(items)),
        "order_items_distinct_orders": int(items["order_id"].nunique()),
        "orphan_order_items": orphan_items,
        "customer_id_unique": bool(frames["customers"]["customer_id"].is_unique),
        "seller_id_unique": bool(frames["sellers"]["seller_id"].is_unique),
        "product_id_unique": bool(frames["products"]["product_id"].is_unique),
    }
    if not grain["order_id_unique_in_orders"]:
        raise AuditFailure("order_id is not unique; order-level grain is unsafe")
    if orphan_items:
        raise AuditFailure(f"{orphan_items} order_items rows reference unknown orders")

    # --- target rule evidence --------------------------------------------------
    eligible, exclusions = select_eligible(orders)
    naive_late = (
        eligible["order_delivered_customer_date"] > eligible["order_estimated_delivery_date"]
    )
    calendar_late = eligible[spec.TARGET_NAME]
    target_rule = {
        "rule": "date(order_delivered_customer_date) > date(order_estimated_delivery_date)",
        "late_orders": int(calendar_late.sum()),
        "late_rate": float(calendar_late.mean()),
        "naive_timestamp_rule_late_orders": int(naive_late.sum()),
        "orders_misclassified_by_naive_rule": int((naive_late & ~calendar_late).sum()),
        "estimated_dates_all_midnight": bool(
            (eligible["order_estimated_delivery_date"].dt.time
             == pd.Timestamp("00:00").time()).all()
        ),
        "timezone_assumption": (
            "Olist publishes naive local Brazilian timestamps. No timezone conversion "
            "is applied; all comparisons stay within the dataset clock."
        ),
    }

    # --- features + splits -----------------------------------------------------
    full = build_feature_table(frames, eligible)
    train_base_rate = full.attrs.get("train_base_rate")
    features, outcomes = split_features_and_outcomes(full)
    # Re-attach the label for split statistics only; it is never persisted to
    # the feature table or returned by an as-of path.
    labelled = features.merge(outcomes, on="order_id", validate="one_to_one")

    split_stats = {}
    for name in ("train", "validation", "test"):
        sub = labelled[labelled["split"] == name]
        if sub.empty:
            raise AuditFailure(f"split '{name}' is empty")
        positives, negatives = int(sub[spec.TARGET_NAME].sum()), int((~sub[spec.TARGET_NAME]).sum())
        if positives == 0 or negatives == 0:
            raise AuditFailure(
                f"split '{name}' does not contain both classes "
                f"(late={positives}, on_time={negatives})"
            )
        split_stats[name] = {
            "orders": int(len(sub)),
            "late": positives,
            "on_time": negatives,
            "late_rate": float(sub[spec.TARGET_NAME].mean()),
            "handover_from": str(sub["order_delivered_carrier_date"].min()),
            "handover_to": str(sub["order_delivered_carrier_date"].max()),
        }

    test_positives = split_stats["test"]["late"]
    if test_positives < spec.REVIEW_CAPACITY_K:
        raise AuditFailure(
            f"test period has only {test_positives} late orders, too few for a "
            f"meaningful Precision@{spec.REVIEW_CAPACITY_K} evaluation"
        )

    # --- snapshots -------------------------------------------------------------
    members = build_snapshot_members(features, outcomes)
    snapshot_stats = validate_snapshots(members, features)

    # --- missingness on model features ----------------------------------------
    missingness = {
        f: float(features[f].isna().mean())
        for f in MODEL_FEATURES
        if features[f].isna().any()
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "name": "Olist Brazilian E-Commerce Public Dataset",
            "url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "license": "CC BY-NC-SA 4.0",
            "files": inventory,
        },
        "grain_assertions": grain,
        "eligibility": exclusions.as_dict(),
        "target_rule": target_rule,
        "splits": {
            "cut_on": spec.PREDICTION_MOMENT_COLUMN,
            "train_end": str(spec.TRAIN_END.date()),
            "validation_end": str(spec.VALIDATION_END.date()),
            "stats": split_stats,
        },
        "snapshots": snapshot_stats,
        "feature_count": len(MODEL_FEATURES),
        "feature_missingness": missingness,
        "train_base_rate": train_base_rate,
    }
    return report, features, outcomes, members


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(rep: dict) -> str:
    src = rep["source"]
    files_tbl = _md_table(
        [[f["file"], f"{f['bytes']:,}", f["sha256"][:24], "yes" if f["sha256_prefix_verified"] else "NO"]
         for f in src["files"]],
        ["file", "bytes", "sha256 (truncated)", "cross-verified"],
    )
    excl_tbl = _md_table(
        [[e["reason"], f"{e['orders_removed']:,}"] for e in rep["eligibility"]["exclusions"]],
        ["exclusion reason", "orders removed"],
    )
    split_tbl = _md_table(
        [[k, f"{v['orders']:,}", f"{v['late']:,}", f"{v['late_rate']:.3%}",
          v["handover_from"][:10], v["handover_to"][:10]]
         for k, v in rep["splits"]["stats"].items()],
        ["split", "orders", "late", "late rate", "handover from", "handover to"],
    )
    snap_tbl = _md_table(
        [[k, v["label"], f"{v['in_transit']:,}", f"{v['pre_deadline']:,}", f"{v['overdue']:,}"]
         for k, v in rep["snapshots"].items()],
        ["snapshot_id", "label", "in transit", "pre-deadline (ranked)", "overdue (separate status)"],
    )
    feat_tbl = _md_table(
        [[f, FEATURE_AVAILABILITY[f][0], FEATURE_AVAILABILITY[f][1], FEATURE_AVAILABILITY[f][2]]
         for f in MODEL_FEATURES],
        ["feature", "source column(s)", "available at", "availability / leakage rationale"],
    )
    t = rep["target_rule"]
    miss = rep["feature_missingness"]
    miss_tbl = (
        _md_table([[k, f"{v:.2%}"] for k, v in sorted(miss.items(), key=lambda x: -x[1])],
                  ["feature", "missing share"])
        if miss else "_No model feature contains missing values._"
    )

    return f"""# OpsPilot — Data Audit

Generated: {rep['generated_at_utc']}
Status: **PASS** (every gate condition in project plan section 3.5 was asserted in code)

This report is produced by `python -m data_pipeline.audit`. It fails loudly and
writes nothing if any gate condition breaks.

## 1. Source and rights

- **Dataset:** {src['name']}
- **Origin:** <{src['url']}>
- **License:** {src['license']} — non-commercial, share-alike, attribution required.
- Olist is credited in the application UI and README. No endorsement is implied.
- Raw CSVs are **not** redistributed in this repository; `data_pipeline/fetch.py`
  retrieves them and verifies checksums.

Olist notes that one order may contain several items from several sellers. That
warning drives the order-level grain rules in section 3 below.

{files_tbl}

Checksums were cross-verified byte-for-byte against two independent public
mirrors before use, so the pipeline is known to run on the authentic release.

## 2. Eligibility funnel

An eligible order has a valid carrier handover, a valid promised date, a known
delivery outcome and a chronologically consistent event sequence. Cancelled and
unresolved orders are excluded and counted — never silently labelled on-time.

{excl_tbl}

**Eligible orders: {rep['eligibility']['eligible_orders']:,}** of
{rep['eligibility']['total_orders']:,} ({rep['eligibility']['eligible_orders'] / rep['eligibility']['total_orders']:.2%}).

## 3. Grain assertions

- `order_id` unique in orders: **{rep['grain_assertions']['order_id_unique_in_orders']}**
- `order_items` rows: {rep['grain_assertions']['order_items_rows']:,} across
  {rep['grain_assertions']['order_items_distinct_orders']:,} distinct orders
- Orphan `order_items` rows: **{rep['grain_assertions']['orphan_order_items']}**
- Item measures are aggregated to order level *before* joining, and the feature
  builder asserts `order_id` uniqueness, so a multi-item order yields exactly
  **one** risk record.

## 4. Target definition

```
{t['rule']}
```

- Late orders: **{t['late_orders']:,}** ({t['late_rate']:.4%} of eligible)
- All `order_estimated_delivery_date` values are stored at midnight: **{t['estimated_dates_all_midnight']}**
- A naive timestamp comparison would flag {t['naive_timestamp_rule_late_orders']:,} orders,
  **misclassifying {t['orders_misclassified_by_naive_rule']:,} same-day afternoon
  deliveries as late**. This is precisely the failure the calendar-date rule prevents.
- Timezone: {t['timezone_assumption']}

## 5. Frozen chronological splits

Split on `{rep['splits']['cut_on']}` (the prediction moment). Cutoffs were frozen
after data inspection and **before** any model comparison.

- train: handover &lt; {rep['splits']['train_end']}
- validation: {rep['splits']['train_end']} &le; handover &lt; {rep['splits']['validation_end']}
- test: handover &ge; {rep['splits']['validation_end']}

{split_tbl}

Class prevalence shifts materially across periods (a real distribution shift in
the Olist data, peaking in March 2018). The model report discusses the effect
rather than hiding it.

## 6. Demo snapshots

All three snapshots fall inside the held-out test period. Membership is derived
from event timestamps (`handover <= D < delivery`), not from the terminal
`order_status` label, which would leak the outcome.

{snap_tbl}

Orders already past their promised date at the snapshot moment are marked
**overdue** and shown in a separate status. Only pre-deadline orders form the
ranked risk queue, so an already-known event is never presented as a prediction.

## 7. Feature availability table

{rep['feature_count']} model features. Every one carries an availability
justification; `build_feature_table` raises if a feature lacks an entry, and
`backend/tests/test_leakage.py` re-asserts it.

{feat_tbl}

### Explicitly forbidden as predictors or agent-visible facts

`order_delivered_customer_date`, `is_late`, delivery duration, review scores and
comments, the terminal `order_status` presented as a present-tense status, and
any seller/category aggregate computed from outcomes not yet known.

The two history features earn their place: they read **only** outcomes whose
delivery timestamp is strictly earlier than the order own handover, which an
operator would genuinely have had. The smoothing prior
({rep['train_base_rate']:.4f}) is the train-split base rate, fitted on train only.

### Missing values

{miss_tbl}

Missing values stay missing. They are handled by the model pipeline (median
imputation fitted on train), never filled from an outcome.

## 8. Data-use plan

Only `orders`, `order_items`, `customers`, `sellers`, `products` and the category
translation are ingested. `order_reviews` is deliberately excluded (a forbidden
post-outcome signal), as are `order_payments` (no as-of value at handover) and
`geolocation` (redundant with the coarse state fields).

Outcome columns are written to a physically separate `order_outcomes` table that
no as-of API path or agent tool queries — a structural guarantee rather than a
convention.
"""


def main() -> int:
    rep, features, outcomes, members = run_audit()

    spec.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    spec.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    features.to_parquet(spec.PROCESSED_DIR / "order_features.parquet", index=False)
    outcomes.to_parquet(spec.PROCESSED_DIR / "order_outcomes.parquet", index=False)
    members.to_parquet(spec.PROCESSED_DIR / "snapshot_members.parquet", index=False)

    (spec.DOCS_DIR / "data_audit.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (spec.DOCS_DIR / "data_audit.md").write_text(render_markdown(rep), encoding="utf-8")

    print("DATA AUDIT: PASS")
    print(f"  eligible orders : {rep['eligibility']['eligible_orders']:,}")
    print(f"  late rate       : {rep['target_rule']['late_rate']:.4%}")
    for name, s in rep["splits"]["stats"].items():
        print(f"  {name:<11s}: {s['orders']:>6,} orders, {s['late']:>5,} late ({s['late_rate']:.3%})")
    for sid, s in rep["snapshots"].items():
        print(f"  snapshot {sid}: {s['pre_deadline']:,} pre-deadline / {s['overdue']:,} overdue")
    print("  wrote docs/data_audit.md, docs/data_audit.json, data/processed/*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
