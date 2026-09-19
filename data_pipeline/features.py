"""Point-in-time feature construction: exactly one row per eligible order.

Every column produced here must be knowable at `order_delivered_carrier_date`.
`FEATURE_AVAILABILITY` is the machine-readable justification table that the
audit renders and that `backend/tests/test_leakage.py` asserts against; adding
a feature without an entry there fails the build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.loading import assign_split

# Smoothing weight for the as-of seller/route history features. Fixed, not tuned.
HISTORY_SMOOTHING_ALPHA = 20.0

# feature -> (source column(s), availability moment, leakage rationale)
FEATURE_AVAILABILITY: dict[str, tuple[str, str, str]] = {
    "days_handover_to_estimate": (
        "order_estimated_delivery_date, order_delivered_carrier_date",
        "carrier handover",
        "Promise date is set at purchase; handover is the prediction moment.",
    ),
    "hours_purchase_to_handover": (
        "order_purchase_timestamp, order_delivered_carrier_date",
        "carrier handover",
        "Elapsed fulfilment time; both endpoints at or before the prediction moment.",
    ),
    "hours_approval_to_handover": (
        "order_approved_at, order_delivered_carrier_date",
        "carrier handover",
        "Approval precedes handover; NaN kept as missing, never imputed from an outcome.",
    ),
    "hours_purchase_to_approval": (
        "order_purchase_timestamp, order_approved_at",
        "payment approval",
        "Both events precede handover.",
    ),
    "shipping_limit_slack_hours": (
        "order_items.shipping_limit_date, order_delivered_carrier_date",
        "carrier handover",
        "Seller shipping deadline is assigned at purchase; slack measured vs actual handover.",
    ),
    "purchase_dow": ("order_purchase_timestamp", "purchase", "Calendar attribute of a past event."),
    "purchase_hour": ("order_purchase_timestamp", "purchase", "Calendar attribute of a past event."),
    "handover_dow": ("order_delivered_carrier_date", "carrier handover", "Calendar attribute of the prediction moment."),
    "handover_month": ("order_delivered_carrier_date", "carrier handover", "Calendar attribute of the prediction moment."),
    "handover_week_of_year": ("order_delivered_carrier_date", "carrier handover", "Seasonality at the prediction moment."),
    "n_items": ("order_items", "purchase", "Order composition is fixed at purchase."),
    "n_distinct_products": ("order_items", "purchase", "Order composition is fixed at purchase."),
    "n_distinct_sellers": ("order_items", "purchase", "Multi-seller orders ship separately; known at purchase."),
    "total_price": ("order_items.price", "purchase", "Monetary value fixed at purchase."),
    "total_freight": ("order_items.freight_value", "purchase", "Freight quoted at purchase."),
    "max_item_price": ("order_items.price", "purchase", "Monetary value fixed at purchase."),
    "freight_ratio": ("order_items", "purchase", "freight / (price + freight); both fixed at purchase."),
    "total_weight_g": ("products.product_weight_g", "purchase", "Static product attribute."),
    "max_weight_g": ("products.product_weight_g", "purchase", "Static product attribute."),
    "total_volume_cm3": ("products dimension columns", "purchase", "Static product attribute."),
    "max_product_photos": ("products.product_photos_qty", "purchase", "Static product attribute."),
    "customer_state": ("customers.customer_state", "purchase", "Coarse geography, static."),
    "seller_state": ("sellers.seller_state", "purchase", "Coarse geography of primary seller, static."),
    "product_category": ("products.product_category_name", "purchase", "Static product attribute."),
    "is_cross_state": ("customers/sellers state", "purchase", "Both endpoints static at purchase."),
    "n_seller_states": ("sellers.seller_state", "purchase", "Order composition fixed at purchase."),
    "seller_prior_orders": (
        "derived: prior eligible orders of the same seller",
        "carrier handover",
        "Counts ONLY orders whose delivery outcome occurred strictly before this "
        "order handover, so the statistic was observable at prediction time.",
    ),
    "seller_prior_late_rate": (
        "derived: prior outcomes of the same seller",
        "carrier handover",
        "Smoothed late rate over strictly-earlier known outcomes; the smoothing "
        "prior is the TRAIN-split base rate, fitted on train only.",
    ),
    "route_prior_orders": (
        "derived: prior orders on the same seller_state -> customer_state route",
        "carrier handover",
        "Same strict as-of rule as the seller history feature.",
    ),
    "route_prior_late_rate": (
        "derived: prior outcomes on the same route",
        "carrier handover",
        "Smoothed late rate over strictly-earlier known outcomes.",
    ),
}

NUMERIC_FEATURES = [
    "days_handover_to_estimate", "hours_purchase_to_handover", "hours_approval_to_handover",
    "hours_purchase_to_approval", "shipping_limit_slack_hours", "purchase_dow", "purchase_hour",
    "handover_dow", "handover_month", "handover_week_of_year", "n_items", "n_distinct_products",
    "n_distinct_sellers", "total_price", "total_freight", "max_item_price", "freight_ratio",
    "total_weight_g", "max_weight_g", "total_volume_cm3", "max_product_photos",
    "is_cross_state", "n_seller_states", "seller_prior_orders", "seller_prior_late_rate",
    "route_prior_orders", "route_prior_late_rate",
]
CATEGORICAL_FEATURES = ["customer_state", "seller_state", "product_category"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Order attributes that are NOT model inputs but are safe to expose as-of.
CONTEXT_COLUMNS = [
    "order_id", "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_estimated_delivery_date", "split",
]
# Outcome columns. Stored in a physically separate table; never in an as-of DTO.
OUTCOME_COLUMNS = ["order_id", "order_delivered_customer_date", "is_late"]


def _aggregate_items(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Collapse order_items to exactly one row per order_id.

    Plan section 3.3: a multi-item order must produce ONE risk record.
    """
    prod = products.copy()
    for c in ("product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"):
        prod[c] = pd.to_numeric(prod.get(c), errors="coerce")
    prod["volume_cm3"] = (
        prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    )
    prod["product_photos_qty"] = pd.to_numeric(prod.get("product_photos_qty"), errors="coerce")

    it = items.merge(
        prod[["product_id", "product_weight_g", "volume_cm3",
              "product_photos_qty", "product_category_name"]],
        on="product_id", how="left", validate="many_to_one",
    )

    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_distinct_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        max_item_price=("price", "max"),
        total_weight_g=("product_weight_g", "sum"),
        max_weight_g=("product_weight_g", "max"),
        total_volume_cm3=("volume_cm3", "sum"),
        max_product_photos=("product_photos_qty", "max"),
        earliest_shipping_limit=("shipping_limit_date", "min"),
    )
    denom = agg["total_price"] + agg["total_freight"]
    agg["freight_ratio"] = np.where(denom > 0, agg["total_freight"] / denom, np.nan)

    # Primary seller / category = the highest-priced line of the order.
    primary = (
        it.sort_values(["order_id", "price"], ascending=[True, False])
          .groupby("order_id")
          .agg(primary_seller_id=("seller_id", "first"),
               product_category=("product_category_name", "first"))
    )
    return agg.join(primary)


def _as_of_history(
    orders: pd.DataFrame, key: pd.Series, prior_rate: float, name: str
) -> pd.DataFrame:
    """Expanding late rate over outcomes observable at each order handover.

    For order *i* with handover h_i and group key g, this counts only orders of
    group g whose `order_delivered_customer_date` is strictly earlier than h_i.
    Those outcomes were genuinely known to an operator at h_i, which is what
    makes the feature point-in-time legal (plan section 3.3). Using a later
    outcome here would be exactly the leakage the audit tests for.
    """
    events = pd.DataFrame({
        "key": key.to_numpy(),
        "t": orders["order_delivered_customer_date"].to_numpy(),
        "late": orders[spec.TARGET_NAME].astype(float).to_numpy(),
    }).sort_values(["key", "t"], kind="mergesort")
    events["cum_n"] = events.groupby("key").cumcount() + 1.0
    events["cum_late"] = events.groupby("key")["late"].cumsum()

    queries = pd.DataFrame({
        "order_id": orders["order_id"].to_numpy(),
        "key": key.to_numpy(),
        "t": orders["order_delivered_carrier_date"].to_numpy(),
    })

    # merge_asof with `by=` still requires both frames globally sorted on `on`.
    events = events.sort_values("t", kind="mergesort")
    queries = queries.sort_values("t", kind="mergesort")

    merged = pd.merge_asof(
        queries, events[["key", "t", "cum_n", "cum_late"]],
        on="t", by="key", direction="backward", allow_exact_matches=False,
    )
    n = merged["cum_n"].fillna(0.0)
    late = merged["cum_late"].fillna(0.0)
    out = pd.DataFrame({
        "order_id": merged["order_id"].to_numpy(),
        f"{name}_prior_orders": n.to_numpy(),
        f"{name}_prior_late_rate":
            ((late + HISTORY_SMOOTHING_ALPHA * prior_rate) / (n + HISTORY_SMOOTHING_ALPHA)).to_numpy(),
    })
    return out.set_index("order_id")


def build_feature_table(
    frames: dict[str, pd.DataFrame], eligible: pd.DataFrame
) -> pd.DataFrame:
    """Produce the one-row-per-order point-in-time feature table."""
    items = _aggregate_items(frames["order_items"], frames["products"])

    df = eligible.merge(items, on="order_id", how="inner", validate="one_to_one")
    df = df.merge(
        frames["customers"][["customer_id", "customer_state"]],
        on="customer_id", how="left", validate="many_to_one",
    )
    df = df.merge(
        frames["sellers"][["seller_id", "seller_state"]].rename(
            columns={"seller_id": "primary_seller_id"}),
        on="primary_seller_id", how="left", validate="many_to_one",
    )
    seller_states = (
        frames["order_items"]
        .merge(frames["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left")
        .groupby("order_id")["seller_state"].nunique()
        .rename("n_seller_states")
    )
    df = df.merge(seller_states, on="order_id", how="left")

    hv = df["order_delivered_carrier_date"]
    df["days_handover_to_estimate"] = (
        df["order_estimated_delivery_date"].dt.normalize() - hv.dt.normalize()
    ).dt.total_seconds() / 86400.0
    df["hours_purchase_to_handover"] = (
        hv - df["order_purchase_timestamp"]).dt.total_seconds() / 3600.0
    df["hours_approval_to_handover"] = (
        hv - df["order_approved_at"]).dt.total_seconds() / 3600.0
    df["hours_purchase_to_approval"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600.0
    df["shipping_limit_slack_hours"] = (
        df["earliest_shipping_limit"] - hv).dt.total_seconds() / 3600.0

    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["handover_dow"] = hv.dt.dayofweek
    df["handover_month"] = hv.dt.month
    df["handover_week_of_year"] = hv.dt.isocalendar().week.astype(int)
    df["is_cross_state"] = (df["customer_state"] != df["seller_state"]).astype(int)

    df["split"] = assign_split(hv)

    # Smoothing prior is fitted on the TRAIN split only.
    train_rate = float(df.loc[df["split"] == "train", spec.TARGET_NAME].mean())
    ordered = df.sort_values("order_delivered_carrier_date", kind="mergesort")
    seller_hist = _as_of_history(ordered, ordered["primary_seller_id"], train_rate, "seller")
    route_key = ordered["seller_state"].astype(str) + ">" + ordered["customer_state"].astype(str)
    route_hist = _as_of_history(ordered, route_key, train_rate, "route")

    df = df.set_index("order_id").join(seller_hist).join(route_hist).reset_index()
    df.attrs["train_base_rate"] = train_rate

    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        raise RuntimeError(f"feature builder did not produce: {missing}")
    undocumented = [f for f in MODEL_FEATURES if f not in FEATURE_AVAILABILITY]
    if undocumented:
        raise RuntimeError(
            f"features lack an availability justification: {undocumented}. "
            "Add them to FEATURE_AVAILABILITY before training."
        )
    if not df["order_id"].is_unique:
        raise RuntimeError("feature table has more than one row per order_id")
    return df


def split_features_and_outcomes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically separate as-of features from delivery outcomes.

    After this call the feature frame contains no outcome column at all, so an
    as-of query path cannot leak one even by accident. `order_status` is dropped
    too: its terminal value ("delivered") already encodes the outcome and must
    never be shown as a present-tense status (plan section 3.3).
    """
    outcomes = df[OUTCOME_COLUMNS].copy()

    keep = CONTEXT_COLUMNS + MODEL_FEATURES
    features = df[keep].copy()

    banned = {"order_delivered_customer_date", spec.TARGET_NAME, "order_status"}
    present = banned & set(features.columns)
    if present:
        raise RuntimeError(f"as-of feature table still exposes outcome columns: {present}")
    return features, outcomes
