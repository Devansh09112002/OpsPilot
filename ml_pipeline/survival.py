"""Survival models for "will this parcel miss its promised day?".

Two learned models, both of which handle *censoring* - parcels still moving
at the training cutoff contribute what is known about them rather than being
dropped or mislabelled:

**Discrete-time hazard model (`HazardModel`).** One row per order per calendar
day it was at risk, label = delivered that day. XGBoost learns the daily
delivery hazard h(day | order, context). For an order still undelivered at
the start of day D, promised for day P,

    P(late) = product over c = D..P of (1 - h(c))

which is *exactly* the target - "not delivered by the end of the promised
day" - with no approximation. Kaplan-Meier is the special case in which the
hazard depends on the lane and the parcel's age only; this model also sees
the order's own details and the day's congestion. The same hazards give the
whole remaining delivery-time distribution, hence a revised arrival estimate.

**Accelerated failure time (`AFTModel`).** One row per order. XGBoost's
`survival:aft` fits log(transit time) with normal errors, right-censored at
the cutoff. P(late | undelivered now) = S(promise) / S(now).

Every context feature is computed as of the *start* of its day, from orders
whose outcome was settled by then (see `data_pipeline.snapshot_features`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from data_pipeline.snapshot_features import (
    KM_LOOKBACK_DAYS,
    MIN_KM_ORDERS,
    RECENT_DAYS,
    SMOOTHING_ALPHA,
    _known_at,
)

LABEL = spec.TARGET_NAME
MAX_AGE = 120
FIRST_DAY = pd.Timestamp("2017-01-01")

XGB_PARAMS = {
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "tree_method": "hist",
    "nthread": 16,
    "seed": 42,
}
N_ROUNDS = 500

CONTEXT_COLUMNS = [
    "recent_late_rate_network", "recent_late_rate_lane", "lane_overdue_share",
    "network_overdue_share", "lane_in_transit",
]
HAZARD_FEATURES = [
    *NUMERIC_FEATURES, *CATEGORICAL_FEATURES,
    "age", "days_to_promise", "frac_window", "dow",
    *CONTEXT_COLUMNS, "km_hazard", "km_hazard_network",
]
AFT_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *CONTEXT_COLUMNS]


# ---------------------------------------------------------------------------
# As-of context, by calendar day and lane
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Day-granular columns used throughout. Pure arithmetic, no outcomes."""
    out = df.copy()
    out["handover_day"] = out["order_delivered_carrier_date"].dt.normalize()
    out["delivered_day"] = out["order_delivered_customer_date"].dt.normalize()
    out["promise_age"] = (out["est_day"] - out["handover_day"]).dt.days
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")
    out["lane"] = out["lane"].astype("category")
    return out


def daily_context(df: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Congestion and recent late rates per (day, lane), as of each day's start."""
    lanes = pd.Index(sorted(df["lane"].unique()))
    frames = []
    h, dl = df["order_delivered_carrier_date"], df["order_delivered_customer_date"]
    for day in days:
        known = _known_at(df, day)
        window = known & (df["est_day"] >= day - pd.Timedelta(days=RECENT_DAYS)) & (df["est_day"] < day)
        w = df.loc[window]
        net_recent = float(w[LABEL].mean()) if len(w) else np.nan
        lane_recent = w.groupby("lane", observed=True)[LABEL].agg(["sum", "count"]).reindex(lanes)
        moving = (h <= day) & (dl > day)
        overdue = moving & (df["est_day"] < day)
        n_moving = df.loc[moving].groupby("lane", observed=True).size().reindex(lanes).fillna(0.0)
        n_overdue = df.loc[overdue].groupby("lane", observed=True).size().reindex(lanes).fillna(0.0)
        net_overdue = float(overdue.sum()) / max(float(moving.sum()), 1.0)
        prior = net_recent if np.isfinite(net_recent) else 0.0
        frames.append(pd.DataFrame({
            "day": day,
            "lane": lanes,
            "recent_late_rate_network": net_recent,
            "recent_late_rate_lane": (
                (lane_recent["sum"].fillna(0.0) + SMOOTHING_ALPHA * prior)
                / (lane_recent["count"].fillna(0.0) + SMOOTHING_ALPHA)
            ).to_numpy(),
            "lane_overdue_share": (
                (n_overdue + SMOOTHING_ALPHA * net_overdue) / (n_moving + SMOOTHING_ALPHA)
            ).to_numpy(),
            "network_overdue_share": net_overdue,
            "lane_in_transit": n_moving.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


@dataclass
class WeeklyKM:
    """Kaplan-Meier daily delivery hazards by parcel age, refreshed weekly.

    Each week's curves use handovers from the 120 days before the week began,
    with parcels undelivered at that moment censored.
    """

    weeks: pd.DatetimeIndex
    lane_index: pd.Index
    dest_index: pd.Index
    lane_hazard: np.ndarray      # (week, lane, age)
    dest_hazard: np.ndarray      # (week, dest, age)
    net_hazard: np.ndarray       # (week, age)
    lane_n: np.ndarray           # (week, lane)
    dest_n: np.ndarray           # (week, dest)

    @classmethod
    def build(cls, df: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> WeeklyKM:
        weeks = pd.date_range(first - pd.Timedelta(days=first.dayofweek), last, freq="W-MON")
        lanes = pd.Index(sorted(df["lane"].unique()))
        dests = pd.Index(sorted(df["customer_state"].astype(str).unique()))
        width = MAX_AGE + 1
        lh = np.zeros((len(weeks), len(lanes), width), np.float32)
        dh = np.zeros((len(weeks), len(dests), width), np.float32)
        nh = np.zeros((len(weeks), width), np.float32)
        ln = np.zeros((len(weeks), len(lanes)), np.float32)
        dn = np.zeros((len(weeks), len(dests)), np.float32)
        lane_code = lanes.get_indexer(df["lane"])
        dest_code = dests.get_indexer(df["customer_state"].astype(str))
        hd, dd = df["handover_day"], df["delivered_day"]
        for wi, week in enumerate(weeks):
            sel = ((hd < week) & (hd >= week - pd.Timedelta(days=KM_LOOKBACK_DAYS))).to_numpy()
            event = (dd[sel] < week).to_numpy()
            last_age = np.where(event, (dd[sel] - hd[sel]).dt.days, (week - hd[sel]).dt.days - 1)
            last_age = np.clip(last_age, 0, MAX_AGE)
            for codes, n_groups, hazard_out, n_out in (
                (lane_code[sel], len(lanes), lh, ln),
                (dest_code[sel], len(dests), dh, dn),
                (np.zeros(sel.sum(), int), 1, None, None),
            ):
                ev = np.zeros((n_groups, width))
                tot = np.zeros((n_groups, width))
                np.add.at(ev, (codes, last_age), event.astype(float))
                np.add.at(tot, (codes, last_age), 1.0)
                at_risk = np.cumsum(tot[:, ::-1], axis=1)[:, ::-1]
                haz = np.divide(ev, at_risk, out=np.full_like(ev, np.nan), where=at_risk > 0)
                if hazard_out is None:
                    nh[wi] = haz[0]
                else:
                    hazard_out[wi] = haz
                    n_out[wi] = tot.sum(axis=1)
        return cls(weeks, lanes, dests, lh, dh, nh, ln, dn)

    def lookup(self, day: pd.Series, lane: pd.Series, dest: pd.Series, age: np.ndarray):
        """Hazard at each (day, lane, age) with lane -> destination -> network fallback."""
        wi = np.searchsorted(self.weeks.values, day.values, side="right") - 1
        wi = np.clip(wi, 0, len(self.weeks) - 1)
        a = np.clip(age, 0, MAX_AGE).astype(int)
        li = self.lane_index.get_indexer(lane)
        di = self.dest_index.get_indexer(dest.astype(str))
        net = self.net_hazard[wi, a]
        lane_h = np.where(li >= 0, self.lane_hazard[wi, np.maximum(li, 0), a], np.nan)
        dest_h = np.where(di >= 0, self.dest_hazard[wi, np.maximum(di, 0), a], np.nan)
        lane_ok = (li >= 0) & (self.lane_n[wi, np.maximum(li, 0)] >= MIN_KM_ORDERS) & np.isfinite(lane_h)
        dest_ok = (di >= 0) & (self.dest_n[wi, np.maximum(di, 0)] >= MIN_KM_ORDERS) & np.isfinite(dest_h)
        km = np.where(lane_ok, lane_h, np.where(dest_ok, dest_h, net))
        return km.astype(np.float32), net.astype(np.float32)


# ---------------------------------------------------------------------------
# Discrete-time hazard model
# ---------------------------------------------------------------------------

EXPAND_COLUMNS = [
    "handover_day", "delivered_day", "est_day", "promise_age", "lane", "_at",
    *NUMERIC_FEATURES, *CATEGORICAL_FEATURES,
]


def _person_days(orders: pd.DataFrame, start_age: np.ndarray, end_age: np.ndarray) -> pd.DataFrame:
    """Expand each order into one row per calendar-day age in [start, end]."""
    lengths = np.maximum(end_age - start_age + 1, 0)
    idx = np.repeat(np.arange(len(orders)), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    keep = [c for c in EXPAND_COLUMNS if c in orders.columns]
    rows = orders[keep].iloc[idx].reset_index(drop=True)
    rows["age"] = np.repeat(start_age, lengths) + offsets
    rows["day"] = rows["handover_day"] + pd.to_timedelta(rows["age"], unit="D")
    return rows


def _hazard_frame(rows: pd.DataFrame, context: pd.DataFrame, km: WeeklyKM,
                  context_day: pd.Series | None = None) -> pd.DataFrame:
    """Attach the per-day features. `context_day` freezes context (at prediction)."""
    cday = rows["day"] if context_day is None else context_day
    ctx = context.set_index(["day", "lane"])
    key = pd.MultiIndex.from_arrays([cday, rows["lane"]])
    joined = ctx.reindex(key)
    for col in CONTEXT_COLUMNS:
        rows[col] = joined[col].to_numpy()
    rows["days_to_promise"] = (rows["est_day"] - rows["day"]).dt.days
    rows["frac_window"] = rows["age"] / np.maximum(rows["promise_age"], 1)
    rows["dow"] = rows["day"].dt.dayofweek
    rows["km_hazard"], rows["km_hazard_network"] = km.lookup(
        cday, rows["lane"], rows["customer_state"], rows["age"].to_numpy())
    return rows


class HazardModel:
    """XGBoost on person-days. `predict_late` is P(not delivered by end of promise)."""

    def __init__(self, context: pd.DataFrame, km: WeeklyKM):
        self.context, self.km, self.booster = context, km, None

    def training_rows(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        orders = df[(df["handover_day"] >= FIRST_DAY) & (df["handover_day"] < cutoff)]
        observed = orders["delivered_day"] < cutoff
        end_day = orders["delivered_day"].where(observed, cutoff - pd.Timedelta(days=1))
        end_age = (end_day - orders["handover_day"]).dt.days.to_numpy()
        rows = _person_days(orders, np.zeros(len(orders), int), np.minimum(end_age, MAX_AGE))
        event_age = (rows["delivered_day"] - rows["handover_day"]).dt.days
        rows["event"] = ((rows["delivered_day"] < cutoff) & (rows["age"] == event_age)).astype(int)
        return _hazard_frame(rows, self.context, self.km)

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> HazardModel:
        import xgboost as xgb

        rows = self.training_rows(df, cutoff)
        dm = xgb.DMatrix(rows[HAZARD_FEATURES], label=rows["event"], enable_categorical=True)
        self.booster = xgb.train({**XGB_PARAMS, "objective": "binary:logistic"}, dm, N_ROUNDS)
        self.n_rows = len(rows)
        return self

    def _future_rows(self, orders: pd.DataFrame, at: pd.Series, horizon: np.ndarray) -> pd.DataFrame:
        start = (at - orders["handover_day"]).dt.days.to_numpy()
        rows = _person_days(orders.assign(_at=at.to_numpy()), start, start + horizon)
        rows["_row"] = np.repeat(np.arange(len(orders)), np.maximum(horizon + 1, 0))
        # The future is unknown on the snapshot day, so every future day is
        # described with the context as it stood that morning.
        return _hazard_frame(rows, self.context, self.km, context_day=rows["_at"])

    def hazards(self, orders: pd.DataFrame, at: pd.Series, horizon: np.ndarray) -> pd.DataFrame:
        import xgboost as xgb

        rows = self._future_rows(orders, at, horizon)
        dm = xgb.DMatrix(rows[HAZARD_FEATURES], enable_categorical=True)
        rows["hazard"] = self.booster.predict(dm)
        return rows[["_row", "day", "hazard"]]

    def predict_late(self, orders: pd.DataFrame, at: pd.Series) -> np.ndarray:
        """Survive every day from the snapshot day through the promised day."""
        horizon = (orders["est_day"] - at).dt.days.to_numpy()
        h = self.hazards(orders, at, horizon)
        log_s = np.log1p(-np.clip(h["hazard"].to_numpy(), 0, 1 - 1e-9))
        out = np.zeros(len(orders))
        np.add.at(out, h["_row"].to_numpy(), log_s)
        return np.exp(out)

    def arrival_quantiles(self, orders: pd.DataFrame, at: pd.Series,
                          quantiles=(0.1, 0.5, 0.9), horizon_days: int = 60) -> np.ndarray:
        """Revised arrival estimate: the day by which each quantile has arrived."""
        horizon = np.full(len(orders), horizon_days)
        h = self.hazards(orders, at, horizon)
        h["log_s"] = np.log1p(-np.clip(h["hazard"], 0, 1 - 1e-9))
        h["surv"] = np.exp(h.groupby("_row")["log_s"].cumsum())
        out = np.empty((len(orders), len(quantiles)), dtype="datetime64[ns]")
        for qi, q in enumerate(quantiles):
            hit = h[h["surv"] <= 1 - q].groupby("_row")["day"].min()
            last = h.groupby("_row")["day"].max()
            out[:, qi] = hit.reindex(range(len(orders))).fillna(last).to_numpy()
        return out


# ---------------------------------------------------------------------------
# Accelerated failure time
# ---------------------------------------------------------------------------

class AFTModel:
    """XGBoost `survival:aft`, normal errors, right-censored at the cutoff."""

    SCALE = 1.0

    def __init__(self, context: pd.DataFrame):
        self.context, self.booster = context, None

    def _features(self, orders: pd.DataFrame, day: pd.Series) -> pd.DataFrame:
        ctx = self.context.set_index(["day", "lane"]).reindex(
            pd.MultiIndex.from_arrays([day, orders["lane"]]))
        X = orders[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].copy()
        for col in CONTEXT_COLUMNS:
            X[col] = ctx[col].to_numpy()
        return X[AFT_FEATURES]

    def fit(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> AFTModel:
        import xgboost as xgb

        orders = df[(df["handover_day"] >= FIRST_DAY) & (df["handover_day"] < cutoff)]
        h = orders["order_delivered_carrier_date"]
        observed = (orders["order_delivered_customer_date"] < cutoff).to_numpy()
        t_obs = (orders["order_delivered_customer_date"] - h).dt.total_seconds() / 86400
        t_cen = (cutoff - h).dt.total_seconds() / 86400
        lower = np.where(observed, t_obs, t_cen).clip(0.01)
        upper = np.where(observed, t_obs.clip(0.01), np.inf)
        dm = xgb.DMatrix(self._features(orders, orders["handover_day"]), enable_categorical=True)
        dm.set_float_info("label_lower_bound", lower)
        dm.set_float_info("label_upper_bound", upper)
        params = {**XGB_PARAMS, "objective": "survival:aft", "eval_metric": "aft-nloglik",
                  "aft_loss_distribution": "normal",
                  "aft_loss_distribution_scale": self.SCALE}
        self.booster = xgb.train(params, dm, N_ROUNDS)
        return self

    def predict_late(self, orders: pd.DataFrame, at: pd.Series) -> np.ndarray:
        import xgboost as xgb
        from scipy.stats import norm

        dm = xgb.DMatrix(self._features(orders, orders["handover_day"]), enable_categorical=True)
        mu = self.booster.predict(dm, output_margin=True)
        h = orders["order_delivered_carrier_date"]
        now = ((at - h).dt.total_seconds() / 86400).clip(lower=0.01).to_numpy()
        promise = ((orders["est_day"] + pd.Timedelta(days=1) - h).dt.total_seconds()
                   / 86400).clip(lower=0.01).to_numpy()
        s_now = norm.sf((np.log(now) - mu) / self.SCALE)
        s_promise = norm.sf((np.log(promise) - mu) / self.SCALE)
        return np.divide(s_promise, s_now, out=np.zeros_like(s_now), where=s_now > 0)
