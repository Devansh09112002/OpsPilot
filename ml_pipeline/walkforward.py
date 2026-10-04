"""Walk-forward development and the one-time test for the v3 snapshot model.

The protocol is fixed in `docs/preregistration_v3.md`; this module implements
it and nothing else.

    python -m ml_pipeline.walkforward develop   # folds F1-F4, writes the dev report
    python -m ml_pipeline.walkforward test      # once, after selection is frozen

Each fold trains only on labels that were settled before the fold began, so
the evaluation is what a team retraining at that moment would actually have
seen. Results are compared *paired*, snapshot by snapshot, because every
candidate is scored on the same days and the day-to-day swing in late rates
is far larger than the gaps between models.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.features import MODEL_FEATURES
from data_pipeline.snapshot_features import (
    SNAPSHOT_MODEL_FEATURES,
    attach_labels,
    build_snapshot_rows,
    load_base_frame,
)
from ml_pipeline.metrics import precision_at_k, recall_at_k
from ml_pipeline.model import (
    DeadlineProximityRule,
    build_logistic_regression,
    build_xgboost,
    calibrate,
    fit_calibrator,
)
from ml_pipeline.selection import simulate_snapshot_dates
from ml_pipeline.snapshot_model import (
    build_snapshot_lr,
    build_snapshot_ranker,
    build_snapshot_xgb,
)

K = spec.REVIEW_CAPACITY_K
TRAIN_FIRST_DAY = pd.Timestamp("2017-01-01")
TRAIN_STEP_DAYS = 3
EVAL_STEP_DAYS = 7
TEST_START = spec.VALIDATION_END  # 2018-06-01
LABEL = spec.TARGET_NAME
OUT_DIR = spec.DOCS_DIR


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    regime: str


FOLDS = [
    Fold("F1", pd.Timestamp("2017-07-01"), pd.Timestamp("2017-10-01"), "calm"),
    Fold("F2", pd.Timestamp("2017-10-01"), pd.Timestamp("2018-01-01"), "Black Friday peak"),
    Fold("F3", pd.Timestamp("2018-01-01"), pd.Timestamp("2018-04-01"), "March 2018 disruption"),
    Fold("F4", pd.Timestamp("2018-04-01"), TEST_START, "recovery"),
]

BASELINES = ["rule_handover", "lr_handover", "xgb_handover", "rule_window", "rule_km"]
CANDIDATES = [
    "lr_snapshot", "xgb_snapshot", "xgb_snapshot_mono",
    "xgb_snapshot_deep", "xgb_snapshot_shallow", "rank_snapshot",
]
FREE_RULES = ["rule_window", "rule_km"]


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def handover_window(start: pd.Timestamp, end: pd.Timestamp):
    def keep(df: pd.DataFrame, _at: pd.Timestamp) -> pd.Series:
        h = df["order_delivered_carrier_date"]
        return (h >= start) & (h < end)
    return keep


def eval_rows(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = simulate_snapshot_dates(start, end, EVAL_STEP_DAYS)
    rows = build_snapshot_rows(df, dates, cohort_filter=handover_window(start, end))
    return attach_labels(rows, df)


def all_training_rows(df: pd.DataFrame, last_day: pd.Timestamp) -> pd.DataFrame:
    """Every third day from 2017-01-01; per-fold filtering happens later."""
    dates = pd.date_range(TRAIN_FIRST_DAY, last_day, freq=f"{TRAIN_STEP_DAYS}D")
    return attach_labels(build_snapshot_rows(df, dates), df)


def training_rows_for(rows: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Snapshot rows before the cutoff whose label was settled by then."""
    keep = (rows["snapshot_at"] < cutoff) & (rows["known_from"] <= cutoff)
    return rows.loc[keep].reset_index(drop=True)


def handover_training_for(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Order-level rows for the v2 recipe, with the same settled-label rule."""
    delivered_day = df["order_delivered_customer_date"].dt.normalize() + pd.Timedelta(days=1)
    known_from = np.minimum(delivered_day, df["est_day"] + pd.Timedelta(days=1))
    keep = (df["order_delivered_carrier_date"] < cutoff) & (known_from <= cutoff)
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Candidates: fit on training data, return a scorer for evaluation rows
# ---------------------------------------------------------------------------

Scorer = Callable[[pd.DataFrame], np.ndarray]


def _y(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL].astype(int).to_numpy()


def fit_candidate(name: str, snap_train: pd.DataFrame, order_train: pd.DataFrame) -> Scorer:
    if name == "rule_handover":
        rule = DeadlineProximityRule()
        return lambda rows: rule.predict_proba(rows[MODEL_FEATURES])[:, 1]
    if name == "rule_window":
        return lambda rows: rows["window_used"].to_numpy(dtype=float)
    if name == "rule_km":
        return lambda rows: rows["km_cond_late"].to_numpy(dtype=float)

    if name == "lr_handover":
        model = build_logistic_regression().fit(order_train[MODEL_FEATURES], _y(order_train))
        return lambda rows: model.predict_proba(rows[MODEL_FEATURES])[:, 1]
    if name == "xgb_handover":
        y = _y(order_train)
        pos = max(int(y.sum()), 1)
        model = build_xgboost(scale_pos_weight=float((len(y) - pos) / pos))
        model.fit(order_train[MODEL_FEATURES], y)
        return lambda rows: model.predict_proba(rows[MODEL_FEATURES])[:, 1]

    X, y = snap_train[SNAPSHOT_MODEL_FEATURES], _y(snap_train)
    if name == "lr_snapshot":
        model = build_snapshot_lr().fit(X, y)
    elif name == "xgb_snapshot":
        model = build_snapshot_xgb().fit(X, y)
    elif name == "xgb_snapshot_mono":
        model = build_snapshot_xgb(monotone=True).fit(X, y)
    elif name == "xgb_snapshot_deep":
        model = build_snapshot_xgb(max_depth=7, min_child_weight=20, n_estimators=600).fit(X, y)
    elif name == "xgb_snapshot_shallow":
        model = build_snapshot_xgb(max_depth=3, n_estimators=800).fit(X, y)
    elif name == "rank_snapshot":
        qid = pd.factorize(snap_train["snapshot_at"])[0]
        model = build_snapshot_ranker().fit(X, y, clf__qid=qid)
        return lambda rows: model.predict(rows[SNAPSHOT_MODEL_FEATURES])
    else:
        raise KeyError(name)
    return lambda rows: model.predict_proba(rows[SNAPSHOT_MODEL_FEATURES])[:, 1]


PROBABILISTIC = {"lr_handover", "lr_snapshot", "xgb_snapshot", "xgb_snapshot_mono",
                 "xgb_snapshot_deep", "xgb_snapshot_shallow"}
# Everything whose output is itself a probability estimate, trained or not.
# The Kaplan-Meier rule is one: it is P(miss the promise | undelivered now).
PROBABILITY_VALUED = PROBABILISTIC | {"rule_km"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def per_snapshot(rows: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    out = []
    for at, idx in rows.groupby("snapshot_at").indices.items():
        y = rows[LABEL].to_numpy().astype(int)[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        rec = {"snapshot_at": pd.Timestamp(at).date().isoformat(), "n": len(idx),
               "late": int(y.sum()), "base_rate": float(y.mean())}
        for name, s in scores.items():
            rec[name] = precision_at_k(y, np.asarray(s)[idx], K)
            rec[f"{name}__recall"] = recall_at_k(y, np.asarray(s)[idx], K)
        out.append(rec)
    return pd.DataFrame(out)


def paired(table: pd.DataFrame, a: str, b: str) -> dict:
    d = (table[a] - table[b]).to_numpy()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    return {"a": a, "b": b, "mean_diff": float(d.mean()), "se": se,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "n": int(len(d))}


def bootstrap_ci(table: pd.DataFrame, a: str, b: str, n_boot: int = 10000,
                 seed: int = 42) -> tuple[float, float]:
    d = (table[a] - table[b]).to_numpy()
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

def develop() -> dict:
    t0 = time.perf_counter()
    df = load_base_frame()
    print(f"base frame {len(df):,} orders")
    train_all = all_training_rows(df, TEST_START - pd.Timedelta(days=1))
    print(f"training snapshot rows (all dates) {len(train_all):,}")

    names = BASELINES + CANDIDATES
    fold_tables: list[pd.DataFrame] = []
    oof: dict[str, list[pd.DataFrame]] = {n: [] for n in PROBABILITY_VALUED}
    fit_seconds: dict[str, list[float]] = {n: [] for n in names}
    train_sizes = {}

    for fold in FOLDS:
        snap_train = training_rows_for(train_all, fold.start)
        order_train = handover_training_for(df, fold.start)
        rows = eval_rows(df, fold.start, fold.end)
        train_sizes[fold.name] = {"snapshot_rows": len(snap_train),
                                  "orders": len(order_train), "eval_rows": len(rows)}
        print(f"\n[{fold.name}] {fold.regime}: train {len(snap_train):,} snapshot rows / "
              f"{len(order_train):,} orders, eval {len(rows):,} rows, "
              f"base rate {rows[LABEL].mean():.3f}")

        scores: dict[str, np.ndarray] = {}
        for name in names:
            s = time.perf_counter()
            try:
                scorer = fit_candidate(name, snap_train, order_train)
                scores[name] = scorer(rows)
            except Exception as exc:  # reported, never silently replaced
                print(f"  {name:<22} FAILED: {type(exc).__name__}: {exc}")
                continue
            fit_seconds[name].append(time.perf_counter() - s)
            if name in PROBABILITY_VALUED:
                oof[name].append(pd.DataFrame({
                    "fold": fold.name, "p": scores[name], "y": rows[LABEL].astype(int)}))

        table = per_snapshot(rows, scores)
        table.insert(0, "fold", fold.name)
        fold_tables.append(table)
        means = {n: round(float(table[n].mean()), 3) for n in names if n in table}
        print("  P@50:", means)

    table = pd.concat(fold_tables, ignore_index=True)
    names = [n for n in names if n in table]

    fold_means = table.groupby("fold")[names].mean()
    score = fold_means.mean(axis=0)
    summary = {
        n: {
            "score": float(score[n]),
            "fold_means": {f: float(fold_means.loc[f, n]) for f in fold_means.index},
            "mean_recall": float(table[f"{n}__recall"].mean()),
            "fit_seconds_mean": float(np.mean(fit_seconds[n])) if fit_seconds[n] else None,
        }
        for n in names
    }

    # --- the pre-registered selection rule --------------------------------
    learned = [n for n in CANDIDATES if n in names]
    best = max(learned, key=lambda n: score[n])
    best_rule = max([r for r in FREE_RULES if r in names], key=lambda n: score[n])
    vs_rule = paired(table, best, best_rule)
    decision = [f"best v3 candidate by fold-mean score: {best} ({score[best]:.4f})",
                f"best free rule: {best_rule} ({score[best_rule]:.4f})"]
    selected = best
    if vs_rule["mean_diff"] <= vs_rule["se"]:
        selected = best_rule
        decision.append(f"guard 1: advantage over {best_rule} {vs_rule['mean_diff']:+.4f} "
                        f"<= SE {vs_rule['se']:.4f} -> serve the rule")
    else:
        decision.append(f"guard 1 passed: {vs_rule['mean_diff']:+.4f} vs SE {vs_rule['se']:.4f}")
        if best != "lr_snapshot" and "lr_snapshot" in names:
            vs_lr = paired(table, best, "lr_snapshot")
            if vs_lr["mean_diff"] <= vs_lr["se"]:
                selected = "lr_snapshot"
                decision.append(f"guard 2: advantage over lr_snapshot {vs_lr['mean_diff']:+.4f} "
                                f"<= SE {vs_lr['se']:.4f} -> serve lr_snapshot")
            else:
                decision.append(f"guard 2 passed: {vs_lr['mean_diff']:+.4f} vs "
                                f"SE {vs_lr['se']:.4f}")

    # --- calibration choice (step 5) ---------------------------------------
    calibration = None
    if selected in PROBABILITY_VALUED and oof[selected]:
        frames = {f["fold"].iloc[0]: f for f in oof[selected]}
        raw, iso = [], []
        for prev, cur in zip(["F1", "F2", "F3"], ["F2", "F3", "F4"], strict=True):
            if prev not in frames or cur not in frames:
                continue
            cal = fit_calibrator(frames[prev]["p"].to_numpy(), frames[prev]["y"].to_numpy())
            y = frames[cur]["y"].to_numpy()
            raw.append(brier(y, frames[cur]["p"]))
            iso.append(brier(y, calibrate(cal, frames[cur]["p"].to_numpy())))
        calibration = {
            "raw_brier_mean": float(np.mean(raw)), "isotonic_prev_fold_brier_mean": float(np.mean(iso)),
            "raw_by_fold": raw, "isotonic_by_fold": iso,
            "choice": "raw" if np.mean(raw) <= np.mean(iso) else "isotonic_prev_fold",
        }
        decision.append(f"calibration: raw {np.mean(raw):.5f} vs isotonic {np.mean(iso):.5f} "
                        f"-> {calibration['choice']}")

    pairs = [paired(table, a, b) for a, b in [
        (selected, "xgb_handover"), (selected, "lr_handover"), (selected, "rule_window"),
        (selected, "rule_km"), ("rule_window", "xgb_handover"), ("rule_km", "xgb_handover"),
        ("xgb_snapshot", "lr_snapshot"), ("xgb_snapshot_mono", "xgb_snapshot"),
        ("rank_snapshot", "xgb_snapshot"),
    ] if a in names and b in names and a != b]

    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "docs/preregistration_v3.md",
        "k": K,
        "folds": [{"name": f.name, "start": str(f.start.date()), "end": str(f.end.date()),
                   "regime": f.regime, **train_sizes[f.name]} for f in FOLDS],
        "summary": summary,
        "selected": selected,
        "decision": decision,
        "calibration": calibration,
        "paired": pairs,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    OUT_DIR.joinpath("research_v3_development.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    table.to_csv(OUT_DIR / "research_v3_development_snapshots.csv", index=False)
    print("\nSELECTED:", selected)
    for line in decision:
        print("  " + line)
    return result


# ---------------------------------------------------------------------------
# Test: once, after selection is frozen
# ---------------------------------------------------------------------------

TEST_JSON = OUT_DIR / "research_v3_test.json"


def _calibrator_for(selected: str, df: pd.DataFrame, train_all: pd.DataFrame):
    """Isotonic map from the last development fold's out-of-fold predictions.

    Step 5 of the pre-registration: the map is fitted on predictions the
    model made for a period it had not trained on, never on test.
    """
    last = FOLDS[-1]
    scorer = fit_candidate(selected, training_rows_for(train_all, last.start),
                           handover_training_for(df, last.start))
    rows = eval_rows(df, last.start, last.end)
    return fit_calibrator(scorer(rows), _y(rows))


def test_once() -> dict:
    if TEST_JSON.exists():
        raise SystemExit(
            f"{TEST_JSON.name} already exists. The pre-registered test runs once; "
            "a rerun has to be justified in docs/preregistration_v3.md first."
        )
    dev_path = OUT_DIR / "research_v3_development.json"
    if not dev_path.exists():
        raise SystemExit("run the develop stage first")
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    selected = dev["selected"]
    cal_choice = (dev.get("calibration") or {}).get("choice", "raw")

    df = load_base_frame()
    train_all = all_training_rows(df, TEST_START - pd.Timedelta(days=1))
    snap_train = training_rows_for(train_all, TEST_START)
    order_train = handover_training_for(df, TEST_START)
    end = df["order_delivered_carrier_date"].max() + pd.Timedelta(seconds=1)
    rows = eval_rows(df, TEST_START, end)
    print(f"test: {rows['snapshot_at'].nunique()} snapshots, {len(rows):,} rows, "
          f"base rate {rows[LABEL].mean():.4f}; selected = {selected} ({cal_choice})")

    scores: dict[str, np.ndarray] = {}
    for name in BASELINES + CANDIDATES:
        try:
            scores[name] = fit_candidate(name, snap_train, order_train)(rows)
        except Exception as exc:
            print(f"  {name:<22} FAILED: {type(exc).__name__}: {exc}")

    # The published v2 artifact, exactly as deployed.
    from ml_pipeline.model import load_artifact

    artifact, meta = load_artifact(spec.ARTIFACT_DIR)
    scores["v2_published"] = artifact.predict_proba(rows[MODEL_FEATURES])[:, 1]

    table = per_snapshot(rows, scores)
    names = [n for n in scores if n in table]
    means = {n: float(table[n].mean()) for n in names}

    comparisons = {}
    for other in ["xgb_handover", "v2_published", "lr_handover", "rule_handover",
                  "rule_window", "rule_km"]:
        if other in table and other != selected:
            comparisons[other] = {**paired(table, selected, other),
                                  "ci95": bootstrap_ci(table, selected, other)}

    y = _y(rows)
    calibration = None
    if selected in PROBABILITY_VALUED:
        p = scores[selected]
        if cal_choice == "isotonic_prev_fold":
            p = calibrate(_calibrator_for(selected, df, train_all), p)
        reference = float(snap_train[LABEL].mean())
        b_model = brier(y, p)
        b_ref = brier(y, np.full_like(p, reference))
        b_hind = brier(y, np.full_like(p, y.mean()))
        flagged = p >= np.quantile(p, 0.9)
        calibration = {
            "choice": cal_choice,
            "brier_model": b_model,
            "brier_constant_pre_test_rate": b_ref,
            "pre_test_rate": reference,
            "brier_skill_vs_pre_test_rate": 1.0 - b_model / b_ref,
            "brier_constant_test_rate_hindsight": b_hind,
            "brier_skill_vs_hindsight_rate": 1.0 - b_model / b_hind,
            "mean_predicted": float(np.mean(p)),
            "observed_rate": float(y.mean()),
            "top_decile_mean_predicted": float(p[flagged].mean()),
            "top_decile_observed": float(y[flagged].mean()),
        }

    primary_ci = comparisons.get("xgb_handover", {}).get("ci95", (float("nan"),) * 2)
    targets = {
        "precision_at_50_at_least_0_20": means.get(selected, 0.0) >= 0.20,
        "beats_retrained_v2_ci_excludes_zero": primary_ci[0] > 0,
        "beats_published_v2": comparisons.get("v2_published", {}).get("mean_diff", 0) > 0,
        "brier_skill_positive": bool(calibration and calibration["brier_skill_vs_pre_test_rate"] > 0),
    }

    import hashlib

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "docs/preregistration_v3.md",
        # What the test was run against, so an edit made afterwards shows.
        "preregistration_sha256": digest(OUT_DIR / "preregistration_v3.md"),
        "development_result_sha256": digest(dev_path),
        "selected": selected,
        "n_snapshots": int(len(table)),
        "base_rate": float(y.mean()),
        "mean_precision_at_50": means,
        "mean_recall_at_50": {n: float(table[f"{n}__recall"].mean()) for n in names},
        "comparisons": comparisons,
        "calibration": calibration,
        "targets": targets,
    }
    TEST_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    table.to_csv(OUT_DIR / "research_v3_test_snapshots.csv", index=False)
    print(json.dumps({"mean_precision_at_50": {k: round(v, 3) for k, v in means.items()},
                      "targets": targets}, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "test"])
    args = parser.parse_args(argv)
    if args.stage == "develop":
        develop()
    else:
        test_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
