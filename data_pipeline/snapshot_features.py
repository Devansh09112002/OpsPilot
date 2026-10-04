"""Point-in-time features for an order *as of a snapshot day*.

The v1/v2 model scores an order once, at carrier handover, and never again.
The product, however, ranks the orders that are still in transit on a later
day D - and by then something new is known: the parcel has not arrived. An
order that has used 90% of its promised window and is still undelivered is a
very different risk from one handed over yesterday, and a handover score
cannot tell them apart.

This module builds one row per (order, snapshot day) with every feature
computable from what an operator would have known at the start of D:

* the order's own handover-time features (unchanged, already point-in-time);
* elapsed time and remaining time against the promise;
* lane transit-time survival curves, estimated with Kaplan-Meier over recent
  handovers so that parcels still in transit count as *censored* rather than
  being silently dropped;
* recent late rates, computed only over orders whose lateness is already
  settled at D;
* how congested the lane and the network are today.

**The as-of rule.** An order's lateness is *known at D* when it was delivered
strictly before D, or when its promised date has already passed (an
undelivered order past its promise is late, whatever happens next). Every
outcome-derived statistic here is computed over known-at-D orders only, which
`backend/tests/test_snapshot_features.py` checks by scrambling every outcome
that is not yet known and asserting no feature moves.

**A bias this fixes.** The handover-time history features count only orders
already *delivered*. Late orders are delivered later, so a recent window
under-counts them, and most of all during a disruption - exactly when the
number matters. Counting "past its promise and still undelivered" as a known
late outcome removes that censoring bias.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _aggregate_items,
)
from data_pipeline.loading import load_raw

DAY = pd.Timedelta(days=1)

# Kaplan-Meier transit-time curves: handovers in the last 120 days, on a
# daily grid. 99.3% of parcels arrive within 45 days, so 120 is generous.
KM_LOOKBACK_DAYS = 120
KM_MAX_DAY = 120
# Below this many recent handovers a lane's curve is too noisy; fall back to
# the destination state, then to the whole network.
MIN_KM_ORDERS = 50

# "Recent" for regime-tracking late rates.
RECENT_DAYS = 28
# Shrinkage towards the network rate for small groups. Fixed, not tuned.
SMOOTHING_ALPHA = 20.0

# feature -> (source, available at, why it is legal)
SNAPSHOT_FEATURE_AVAILABILITY: dict[str, tuple[str, str, str]] = {
    "days_in_transit": (
        "order_delivered_carrier_date, snapshot day", "snapshot day",
        "Clock time since handover. Handover is in the past at D.",
    ),
    "days_left": (
        "order_estimated_delivery_date, snapshot day", "snapshot day",
        "Calendar days until the promise, which was set at purchase.",
    ),
    "window_used": (
        "handover, promise, snapshot day", "snapshot day",
        "Share of the promised window already consumed while undelivered.",
    ),
    "km_survival_now": (
        "derived: lane transit curve", "snapshot day",
        "Kaplan-Meier share of recent lane parcels still in transit at this "
        "age. Parcels not yet delivered at D are censored, never dropped.",
    ),
    "km_cond_late": (
        "derived: lane transit curve", "snapshot day",
        "Kaplan-Meier chance of still being undelivered at the promise, "
        "given undelivered now. Same as-of curve as km_survival_now.",
    ),
    "km_cond_late_network": (
        "derived: network transit curve", "snapshot day",
        "The same conditional chance over every lane: the network's state.",
    ),
    "recent_late_rate_network": (
        "derived: orders whose promise fell in the last 28 days", "snapshot day",
        "Every such order's lateness is settled at D (delivered, or past its "
        "promise and undelivered).",
    ),
    "recent_late_rate_lane": (
        "derived: lane orders whose promise fell in the last 28 days", "snapshot day",
        "As recent_late_rate_network, smoothed towards it.",
    ),
    "lane_in_transit": (
        "derived: lane orders handed over and undelivered at D", "snapshot day",
        "A count an operator can see on the day.",
    ),
    "lane_overdue_share": (
        "derived: lane orders in transit and past their promise", "snapshot day",
        "Undelivered and past the promise is observable on the day.",
    ),
    "network_overdue_share": (
        "derived: all orders in transit and past their promise", "snapshot day",
        "As lane_overdue_share, over the whole network.",
    ),
    "seller_known_late_rate": (
        "derived: the primary seller's earlier orders", "snapshot day",
        "Over orders whose lateness is known at D, smoothed towards the "
        "network rate. Replaces delivered-only counting, which under-counts "
        "recent late orders.",
    ),
    "seller_known_orders": (
        "derived: the primary seller's earlier orders", "snapshot day",
        "How many of them are settled at D.",
    ),
    "lane_known_late_rate": (
        "derived: earlier orders on the lane", "snapshot day",
        "As seller_known_late_rate, for the lane.",
    ),
}

SNAPSHOT_NUMERIC_FEATURES = list(SNAPSHOT_FEATURE_AVAILABILITY)

# The handover-time numerics are kept: they are already point-in-time, and the
# model can learn whether they still matter once elapsed time is known.
SNAPSHOT_MODEL_NUMERIC = NUMERIC_FEATURES + SNAPSHOT_NUMERIC_FEATURES
SNAPSHOT_MODEL_CATEGORICAL = list(CATEGORICAL_FEATURES)
SNAPSHOT_MODEL_FEATURES = SNAPSHOT_MODEL_NUMERIC + SNAPSHOT_MODEL_CATEGORICAL

ROW_KEYS = ["snapshot_at", "order_id"]


def load_base_frame() -> pd.DataFrame:
    """Order-level features, outcomes and the primary seller, one row per order.

    Outcomes are present because as-of statistics need them. Nothing returned
    by `build_snapshot_rows` exposes an outcome for its own order; the label
    column is attached separately and only for offline training/evaluation.
    """
    features = pd.read_parquet(spec.PROCESSED_DIR / "order_features.parquet")
    outcomes = pd.read_parquet(spec.PROCESSED_DIR / "order_outcomes.parquet")
    df = features.merge(outcomes, on="order_id", how="inner", validate="one_to_one")

    frames = load_raw()
    items = _aggregate_items(frames["order_items"], frames["products"])
    df = df.merge(
        items[["primary_seller_id"]], left_on="order_id", right_index=True,
        how="left", validate="one_to_one",
    )
    df["est_day"] = df["order_estimated_delivery_date"].dt.normalize()
    df["lane"] = df["seller_state"].astype(str) + ">" + df["customer_state"].astype(str)
    return df.reset_index(drop=True)


def in_transit_pre_deadline(df: pd.DataFrame, at: pd.Timestamp) -> pd.Series:
    """The ranked population on day D: handed over, undelivered, promise not passed.

    Identical to `ml_pipeline.selection.snapshot_cohort`, so results stay
    comparable with every published figure.
    """
    return (
        (df["order_delivered_carrier_date"] <= at)
        & (df["order_delivered_customer_date"] > at)
        & (df["est_day"] >= at.normalize())
    )


def _known_at(df: pd.DataFrame, at: pd.Timestamp) -> pd.Series:
    """Orders whose lateness is settled at the start of D.

    Delivered strictly before D (the existing strict as-of rule), or promised
    for a day before D: undelivered past its promise is late regardless of
    when it eventually arrives.
    """
    handed_over = df["order_delivered_carrier_date"] < at
    return handed_over & (
        (df["order_delivered_customer_date"] < at) | (df["est_day"] < at.normalize())
    )


def _km_curves(
    durations: np.ndarray, events: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-day Kaplan-Meier survival curves, one per group.

    Returns `(S, n)` where `S[g, k]` estimates P(transit >= k days) and `n[g]`
    is the number of orders behind group g's curve.
    """
    k = np.clip(np.floor(durations).astype(int), 0, KM_MAX_DAY)
    width = KM_MAX_DAY + 1
    ev = np.zeros((n_groups, width))
    tot = np.zeros((n_groups, width))
    np.add.at(ev, (groups, k), events.astype(float))
    np.add.at(tot, (groups, k), 1.0)
    # At risk on day j: every order whose (possibly censored) duration reached j.
    at_risk = np.cumsum(tot[:, ::-1], axis=1)[:, ::-1]
    hazard = np.divide(ev, at_risk, out=np.zeros_like(ev), where=at_risk > 0)
    survival = np.ones((n_groups, width + 1))
    survival[:, 1:] = np.cumprod(1.0 - hazard, axis=1)
    return survival, tot.sum(axis=1)


def _smoothed(late: pd.Series, n: pd.Series, prior: float) -> pd.Series:
    return (late + SMOOTHING_ALPHA * prior) / (n + SMOOTHING_ALPHA)


def features_at(df: pd.DataFrame, at: pd.Timestamp, cohort: np.ndarray) -> pd.DataFrame:
    """Snapshot features for the cohort rows of `df`, as of the start of `at`."""
    at = pd.Timestamp(at).normalize()
    rows = df.loc[cohort].copy()
    handover = rows["order_delivered_carrier_date"]

    # --- the order's own clock -------------------------------------------
    elapsed = (at - handover).dt.total_seconds() / 86400.0
    # Late means delivered on a later calendar day than promised, i.e. at or
    # after the midnight that follows the promised day.
    late_after = (rows["est_day"] + DAY - handover).dt.total_seconds() / 86400.0
    rows["days_in_transit"] = elapsed
    rows["days_left"] = (rows["est_day"] - at).dt.days.astype(float)
    rows["window_used"] = elapsed / np.maximum(late_after, 0.5)

    # --- Kaplan-Meier transit curves over recent handovers ------------------
    recent = df[
        (df["order_delivered_carrier_date"] < at)
        & (df["order_delivered_carrier_date"] >= at - pd.Timedelta(days=KM_LOOKBACK_DAYS))
    ]
    delivered = (recent["order_delivered_customer_date"] < at).to_numpy()
    end = recent["order_delivered_customer_date"].where(delivered, at)
    durations = ((end - recent["order_delivered_carrier_date"]).dt.total_seconds()
                 / 86400.0).to_numpy()

    k_now = np.clip(np.floor(elapsed.to_numpy()).astype(int), 0, KM_MAX_DAY)
    k_late = np.clip(np.floor(late_after.to_numpy()).astype(int), 0, KM_MAX_DAY)

    def curve_values(recent_key: pd.Series, cohort_key: pd.Series):
        codes, uniques = pd.factorize(recent_key, sort=True)
        surv, n = _km_curves(durations, delivered, codes, max(len(uniques), 1))
        index = pd.Index(uniques)
        g = index.get_indexer(cohort_key)
        ok = g >= 0
        g_safe = np.where(ok, g, 0)
        now = surv[g_safe, k_now]
        later = surv[g_safe, k_late]
        cond = np.divide(later, now, out=np.zeros_like(later), where=now > 0)
        support = np.where(ok, n[g_safe], 0.0)
        return now, cond, support

    net_now, net_cond, _ = curve_values(
        pd.Series(0, index=recent.index), pd.Series(0, index=rows.index)
    )
    dest_now, dest_cond, dest_n = curve_values(recent["customer_state"], rows["customer_state"])
    lane_now, lane_cond, lane_n = curve_values(recent["lane"], rows["lane"])

    use_lane = lane_n >= MIN_KM_ORDERS
    use_dest = ~use_lane & (dest_n >= MIN_KM_ORDERS)
    rows["km_survival_now"] = np.where(use_lane, lane_now, np.where(use_dest, dest_now, net_now))
    rows["km_cond_late"] = np.where(use_lane, lane_cond, np.where(use_dest, dest_cond, net_cond))
    rows["km_cond_late_network"] = net_cond

    # --- recent late rates over settled orders ------------------------------
    known = _known_at(df, at)
    window = known & (df["est_day"] >= at - pd.Timedelta(days=RECENT_DAYS)) & (df["est_day"] < at)
    w = df.loc[window]
    network_recent = float(w[spec.TARGET_NAME].mean()) if len(w) else float("nan")
    lane_recent = w.groupby("lane")[spec.TARGET_NAME].agg(["sum", "count"])
    lane_late = rows["lane"].map(lane_recent["sum"]).fillna(0.0)
    lane_n_recent = rows["lane"].map(lane_recent["count"]).fillna(0.0)
    rows["recent_late_rate_network"] = network_recent
    rows["recent_late_rate_lane"] = _smoothed(
        lane_late, lane_n_recent, network_recent if np.isfinite(network_recent) else 0.0
    )

    # --- congestion on the day ----------------------------------------------
    moving = (df["order_delivered_carrier_date"] <= at) & (df["order_delivered_customer_date"] > at)
    overdue = moving & (df["est_day"] < at.normalize())
    lane_moving = df.loc[moving].groupby("lane").size()
    lane_overdue = df.loc[overdue].groupby("lane").size()
    n_moving = rows["lane"].map(lane_moving).fillna(0.0)
    n_overdue = rows["lane"].map(lane_overdue).fillna(0.0)
    rows["lane_in_transit"] = n_moving
    network_overdue = float(overdue.sum()) / max(float(moving.sum()), 1.0)
    rows["lane_overdue_share"] = _smoothed(n_overdue, n_moving, network_overdue)
    rows["network_overdue_share"] = network_overdue

    # --- seller and lane history over settled orders ------------------------
    settled = df.loc[known]
    prior = float(settled[spec.TARGET_NAME].mean()) if len(settled) else 0.0
    by_seller = settled.groupby("primary_seller_id")[spec.TARGET_NAME].agg(["sum", "count"])
    s_late = rows["primary_seller_id"].map(by_seller["sum"]).fillna(0.0)
    s_n = rows["primary_seller_id"].map(by_seller["count"]).fillna(0.0)
    rows["seller_known_late_rate"] = _smoothed(s_late, s_n, prior)
    rows["seller_known_orders"] = s_n
    by_lane = settled.groupby("lane")[spec.TARGET_NAME].agg(["sum", "count"])
    l_late = rows["lane"].map(by_lane["sum"]).fillna(0.0)
    l_n = rows["lane"].map(by_lane["count"]).fillna(0.0)
    rows["lane_known_late_rate"] = _smoothed(l_late, l_n, prior)

    rows["snapshot_at"] = at
    keep = [*ROW_KEYS, "split", "lane", *SNAPSHOT_MODEL_FEATURES]
    return rows[keep]


def build_snapshot_rows(
    df: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    *,
    cohort_filter: Callable[[pd.DataFrame, pd.Timestamp], pd.Series] | None = None,
) -> pd.DataFrame:
    """One row per (order, snapshot day) for every order ranked on that day.

    `cohort_filter` narrows the ranked population further, e.g. to orders
    handed over inside an evaluation window.
    """
    undocumented = [f for f in SNAPSHOT_NUMERIC_FEATURES if f not in SNAPSHOT_FEATURE_AVAILABILITY]
    if undocumented:
        raise RuntimeError(f"snapshot features lack a justification: {undocumented}")

    parts = []
    for at in dates:
        at = pd.Timestamp(at).normalize()
        mask = in_transit_pre_deadline(df, at)
        if cohort_filter is not None:
            mask &= cohort_filter(df, at)
        if not mask.any():
            continue
        parts.append(features_at(df, at, mask.to_numpy()))
    if not parts:
        return pd.DataFrame(columns=[*ROW_KEYS, "split", "lane", *SNAPSHOT_MODEL_FEATURES])
    out = pd.concat(parts, ignore_index=True)
    if out.duplicated(ROW_KEYS).any():
        raise RuntimeError("more than one row per (snapshot, order)")
    return out


def attach_labels(rows: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Offline only: the eventual outcome, plus when it became known.

    `known_from` is the first day at which the label was settled, so a training
    set assembled for a cutoff can refuse labels that were not yet known.
    """
    lab = df[["order_id", spec.TARGET_NAME, "order_delivered_customer_date", "est_day"]].copy()
    delivered_day = lab["order_delivered_customer_date"].dt.normalize() + DAY
    lab["known_from"] = np.minimum(delivered_day, lab["est_day"] + DAY)
    return rows.merge(
        lab[["order_id", spec.TARGET_NAME, "known_from"]],
        on="order_id", how="left", validate="many_to_one",
    )


# Plain-language names for the snapshot-day inputs, used when a served score
# explains itself.
SNAPSHOT_FEATURE_LABELS: dict[str, str] = {
    "days_in_transit": "days in transit so far",
    "days_left": "days left before the promised date",
    "window_used": "share of the promised delivery window already used",
    "km_survival_now": "share of similar parcels on this route still undelivered at this age",
    "km_cond_late": "route's chance of missing the promise, given still undelivered",
    "km_cond_late_network": "network-wide chance of missing the promise, given still undelivered",
    "recent_late_rate_network": "network late rate over the last four weeks",
    "recent_late_rate_lane": "this route's late rate over the last four weeks",
    "lane_in_transit": "parcels in transit on this route today",
    "lane_overdue_share": "share of this route's parcels already overdue today",
    "network_overdue_share": "share of all parcels already overdue today",
    "seller_known_late_rate": "seller's late rate on orders already settled",
    "seller_known_orders": "seller's orders already settled",
    "lane_known_late_rate": "route's late rate on orders already settled",
}
