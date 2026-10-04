"""v4 development and test: survival models and learning-to-rank.

Protocol: `docs/preregistration_v4.md`. Same folds, cohorts and metric as v3
(`ml_pipeline.walkforward`), so every number is comparable.

    python -m ml_pipeline.walkforward_v4 develop
    python -m ml_pipeline.walkforward_v4 test      # only if develop selected a model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.snapshot_features import SNAPSHOT_MODEL_FEATURES, load_base_frame
from ml_pipeline.model import calibrate, fit_calibrator
from ml_pipeline.snapshot_model import build_snapshot_preprocessor
from ml_pipeline.survival import (
    FIRST_DAY,
    AFTModel,
    HazardModel,
    WeeklyKM,
    daily_context,
    prepare,
)
from ml_pipeline.walkforward import (
    FOLDS,
    TEST_START,
    all_training_rows,
    bootstrap_ci,
    brier,
    eval_rows,
    paired,
    per_snapshot,
    training_rows_for,
)

LABEL = spec.TARGET_NAME
OUT = spec.DOCS_DIR
LAST_DAY = pd.Timestamp("2018-08-31")
V4 = ["hazard_gbm", "aft_xgb", "lambdamart", "ensemble_rank"]
BASELINE = "rule_km"
PROBABILITY_VALUED = {"rule_km", "hazard_gbm", "aft_xgb"}
DEV_JSON = OUT / "research_v4_development.json"
TEST_JSON = OUT / "research_v4_test.json"


class Workspace:
    """Everything shared across folds, built once."""

    def __init__(self) -> None:
        t = time.perf_counter()
        self.base = load_base_frame()
        self.df = prepare(self.base)
        self.context = daily_context(self.df, pd.date_range(FIRST_DAY, LAST_DAY))
        self.km = WeeklyKM.build(self.df, FIRST_DAY, LAST_DAY)
        self.snap_train = all_training_rows(self.base, TEST_START - pd.Timedelta(days=1))
        self.by_id = self.df.set_index("order_id")
        print(f"workspace ready in {time.perf_counter() - t:.0f}s")

    def orders_for(self, rows: pd.DataFrame) -> pd.DataFrame:
        return self.by_id.loc[rows["order_id"]].reset_index()


def _lambdamart(snap_train: pd.DataFrame):
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRanker

    model = Pipeline([
        ("pre", build_snapshot_preprocessor(pandas_out=True)),
        ("clf", XGBRanker(
            objective="rank:ndcg", lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=50,
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, min_child_weight=10, tree_method="hist",
            n_jobs=16, random_state=42,
        )),
    ])
    qid = pd.factorize(snap_train["snapshot_at"])[0]
    model.fit(snap_train[SNAPSHOT_MODEL_FEATURES], snap_train[LABEL].astype(int), clf__qid=qid)
    return model


def score_fold(ws: Workspace, cutoff: pd.Timestamp, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit every candidate on what was known before `cutoff`; score `rows`."""
    rows = rows.reset_index(drop=True)
    orders = ws.orders_for(rows)
    at = rows["snapshot_at"]
    scores = {BASELINE: rows["km_cond_late"].to_numpy(dtype=float)}

    for name, fit_predict in [
        ("hazard_gbm", lambda: HazardModel(ws.context, ws.km).fit(ws.df, cutoff).predict_late(orders, at)),
        ("aft_xgb", lambda: AFTModel(ws.context).fit(ws.df, cutoff).predict_late(orders, at)),
        ("lambdamart", lambda: _lambdamart(training_rows_for(ws.snap_train, cutoff))
            .predict(rows[SNAPSHOT_MODEL_FEATURES])),
    ]:
        s = time.perf_counter()
        try:
            scores[name] = np.asarray(fit_predict(), dtype=float)
            print(f"  {name:<14} {time.perf_counter() - s:6.1f}s")
        except Exception as exc:  # reported, never replaced
            print(f"  {name:<14} FAILED: {type(exc).__name__}: {exc}")

    parts = [n for n in (BASELINE, "hazard_gbm", "lambdamart") if n in scores]
    if len(parts) == 3:
        frame = pd.DataFrame({n: scores[n] for n in parts}).assign(at=at)
        ranks = frame.groupby("at")[parts].rank(pct=True)
        scores["ensemble_rank"] = ranks.mean(axis=1).to_numpy()
    return scores


def develop() -> dict:
    t0 = time.perf_counter()
    ws = Workspace()
    tables, oof = [], {n: [] for n in PROBABILITY_VALUED}
    for fold in FOLDS:
        rows = eval_rows(ws.base, fold.start, fold.end).reset_index(drop=True)
        print(f"\n[{fold.name}] {fold.regime}: {len(rows):,} eval rows")
        scores = score_fold(ws, fold.start, rows)
        table = per_snapshot(rows, scores)
        table.insert(0, "fold", fold.name)
        tables.append(table)
        for n in PROBABILITY_VALUED & scores.keys():
            oof[n].append(pd.DataFrame({"fold": fold.name, "p": scores[n],
                                        "y": rows[LABEL].astype(int)}))
        print("  P@50:", {n: round(float(table[n].mean()), 3) for n in scores})

    table = pd.concat(tables, ignore_index=True)
    names = [n for n in [BASELINE, *V4] if n in table]
    fold_means = table.groupby("fold")[names].mean()
    score = fold_means.mean(axis=0)

    candidates = [n for n in V4 if n in names]
    best = max(candidates, key=lambda n: score[n])
    vs = paired(table, best, BASELINE)
    passed = vs["mean_diff"] > 2 * vs["se"]
    decision = [
        f"best v4 candidate: {best} ({score[best]:.4f}); {BASELINE}: {score[BASELINE]:.4f}",
        f"paired advantage over {BASELINE}: {vs['mean_diff']:+.4f}, 2 SE = {2 * vs['se']:.4f}",
        ("bar passed -> test" if passed else
         "bar NOT passed -> v3 stands, the test is not run, v4 is a negative result"),
    ]

    calibration = None
    if passed and best in PROBABILITY_VALUED:
        frames = {f["fold"].iloc[0]: f for f in oof[best]}
        raw, iso = [], []
        for prev, cur in (("F1", "F2"), ("F2", "F3"), ("F3", "F4")):
            cal = fit_calibrator(frames[prev]["p"].to_numpy(), frames[prev]["y"].to_numpy())
            y = frames[cur]["y"].to_numpy()
            raw.append(brier(y, frames[cur]["p"]))
            iso.append(brier(y, calibrate(cal, frames[cur]["p"].to_numpy())))
        calibration = {"raw": float(np.mean(raw)), "isotonic_prev_fold": float(np.mean(iso)),
                       "choice": "raw" if np.mean(raw) <= np.mean(iso) else "isotonic_prev_fold"}
        decision.append(f"calibration -> {calibration['choice']}")

    brier_dev = {n: float(np.mean([brier(f["y"].to_numpy(), f["p"].to_numpy()) for f in oof[n]]))
                 for n in PROBABILITY_VALUED if oof[n]}

    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "docs/preregistration_v4.md",
        "summary": {n: {"score": float(score[n]),
                        "fold_means": {f: float(fold_means.loc[f, n]) for f in fold_means.index}}
                    for n in names},
        "paired_vs_baseline": {n: paired(table, n, BASELINE) for n in candidates},
        "brier_by_model": brier_dev,
        "selected": best if passed else None,
        "decision": decision,
        "calibration": calibration,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    DEV_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    table.to_csv(OUT / "research_v4_development_snapshots.csv", index=False)
    print("\n" + "\n".join(decision))
    return result


def test_once() -> dict:
    if TEST_JSON.exists():
        raise SystemExit("the v4 test already ran; it runs once")
    dev = json.loads(DEV_JSON.read_text(encoding="utf-8"))
    selected = dev["selected"]
    if not selected:
        raise SystemExit("development selected no model; per section 5 the test is not run")

    ws = Workspace()
    end = ws.base["order_delivered_carrier_date"].max() + pd.Timedelta(seconds=1)
    rows = eval_rows(ws.base, TEST_START, end).reset_index(drop=True)
    scores = score_fold(ws, TEST_START, rows)
    table = per_snapshot(rows, scores)
    comp = {**paired(table, selected, BASELINE), "ci95": bootstrap_ci(table, selected, BASELINE)}

    y = rows[LABEL].astype(int).to_numpy()

    def calibration_of(p: np.ndarray) -> dict:
        b, b_hind = brier(y, p), brier(y, np.full_like(p, y.mean()))
        return {"brier": b, "brier_hindsight_constant": b_hind,
                "skill_vs_hindsight": 1 - b / b_hind,
                "mean_predicted": float(p.mean()), "observed": float(y.mean())}

    calibration = calibration_of(scores[selected]) if selected in PROBABILITY_VALUED else None
    # Amendment 2: descriptive only, selects nothing.
    descriptive = {f"calibration_{n}": calibration_of(scores[n])
                   for n in ("rule_km", "hazard_gbm") if n in scores}
    eta = None
    if "hazard_gbm" in scores:
        orders = ws.orders_for(rows)
        model = HazardModel(ws.context, ws.km).fit(ws.df, TEST_START)
        q = model.arrival_quantiles(orders, rows["snapshot_at"])
        actual = orders["delivered_day"].to_numpy()
        err = np.abs((q[:, 1] - actual) / np.timedelta64(1, "D"))
        covered = (actual >= q[:, 0]) & (actual <= q[:, 2])
        eta = {"median_abs_error_days": float(np.median(err)),
               "mean_abs_error_days": float(np.mean(err)),
               "coverage_80pct_interval": float(covered.mean())}

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "preregistration_sha256": digest(OUT / "preregistration_v4.md"),
        "development_result_sha256": digest(DEV_JSON),
        "selected": selected,
        "mean_precision_at_50": {n: float(table[n].mean()) for n in scores},
        "vs_baseline": comp,
        "calibration": calibration,
        "descriptive": {**descriptive, "arrival_estimate_hazard_gbm": eta},
        "targets": {
            "primary_ci_excludes_zero": comp["ci95"][0] > 0,
            "brier_skill_vs_hindsight_positive": (
                None if calibration is None else calibration["skill_vs_hindsight"] > 0),
            "eta_coverage_70_to_90": (
                None if selected != "hazard_gbm" or eta is None
                else 0.70 <= eta["coverage_80pct_interval"] <= 0.90),
        },
    }
    TEST_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    table.to_csv(OUT / "research_v4_test_snapshots.csv", index=False)
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "test"])
    args = parser.parse_args(argv)
    develop() if args.stage == "develop" else test_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
