"""Package the served v4 model and score the three demo snapshots.

    python -m ml_pipeline.train_v4

The served model is exactly the one scored once on the held-out test in
`docs/research_v4.md`: the hazard model and LambdaMART trained on everything
known before 2018-06-01, combined with the Kaplan-Meier rule by averaging
within-day ranks. This script refits them (deterministic seeds), asserts the
test Precision@50 it reproduces equals the published figure, and writes:

    artifacts/v4/hazard.ubj           daily delivery hazard (XGBoost booster)
    artifacts/v4/lambdamart.joblib    top-50 ranker (sklearn pipeline)
    artifacts/v4/support.joblib       as-of context and survival curves for
                                      the demo days, so the API can re-score live
    artifacts/v4/snapshot_scores.parquet  scores, bands and arrival estimates
                                      for every demo-snapshot order
    artifacts/v4/meta.json            version, cutoff, checksums, bands, metrics

The demo snapshots lie inside the test period, so their scores come from the
same frozen models whose test performance is published.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.features import CATEGORICAL_FEATURES
from data_pipeline.snapshot_features import (
    SNAPSHOT_MODEL_FEATURES,
    build_snapshot_rows,
)
from ml_pipeline.survival import HazardModel, WeeklyKM
from ml_pipeline.walkforward import TEST_START, eval_rows, per_snapshot, training_rows_for
from ml_pipeline.walkforward_v4 import Workspace, _lambdamart

OUT = spec.ARTIFACT_DIR / "v4"
MODEL_VERSION = "survival-ensemble-v4"
LABEL = spec.TARGET_NAME


def ensemble(frame: pd.DataFrame, by: str = "snapshot_at") -> np.ndarray:
    """Mean of the three components' within-day percentile ranks."""
    parts = ["km_score", "hazard_score", "lambdamart_score"]
    return frame.groupby(by)[parts].rank(pct=True).mean(axis=1).to_numpy()


def score_rows(rows: pd.DataFrame, ws: Workspace, hazard: HazardModel, ranker) -> pd.DataFrame:
    rows = rows.reset_index(drop=True).copy()
    orders = ws.orders_for(rows)
    rows["km_score"] = rows["km_cond_late"].to_numpy(dtype=float)
    rows["hazard_score"] = hazard.predict_late(orders, rows["snapshot_at"])
    rows["lambdamart_score"] = ranker.predict(rows[SNAPSHOT_MODEL_FEATURES])
    rows["ensemble_score"] = ensemble(rows)
    return rows


def demo_snapshot_scores(ws: Workspace, hazard: HazardModel, ranker) -> pd.DataFrame:
    members = pd.read_parquet(spec.PROCESSED_DIR / "snapshot_members.parquet")
    test_ids = set(ws.base.loc[ws.base["split"] == "test", "order_id"])
    frames = []
    for snap in spec.SNAPSHOTS:
        sid = snap["snapshot_id"]
        at = pd.Timestamp(sid)
        mem = members[members["snapshot_id"] == sid]
        rows = build_snapshot_rows(
            ws.base, [at], cohort_filter=lambda d, _at: d["order_id"].isin(test_ids))
        expected = set(mem.loc[~mem["is_overdue"], "order_id"])
        if set(rows["order_id"]) != expected:
            raise RuntimeError(f"{sid}: snapshot features and snapshot membership disagree")
        ranked = score_rows(rows, ws, hazard, ranker)
        ranked["is_overdue"] = False
        # Position in the queue exactly as the API sorts it: score descending,
        # ties broken by order id, so the stored rank is reproducible live.
        order = ranked.sort_values(["ensemble_score", "order_id"], ascending=[False, True]).index
        ranked.loc[order, "priority_rank"] = np.arange(1, len(ranked) + 1)
        ranked["priority_rank"] = ranked["priority_rank"].astype(int)
        ranked["risk_band"] = [spec.priority_band(r, len(ranked)) for r in ranked["priority_rank"]]

        overdue = mem.loc[mem["is_overdue"], ["order_id"]].copy()
        overdue["snapshot_at"] = at
        overdue["is_overdue"] = True
        # Past its promise and still undelivered at the start of the day: late
        # by the calendar-date rule whatever happens next.
        overdue["hazard_score"] = 1.0
        overdue["priority_rank"] = None
        overdue["risk_band"] = spec.priority_band(None, 0)

        both = pd.concat([ranked, overdue], ignore_index=True)
        orders = ws.orders_for(both)
        q = hazard.arrival_quantiles(orders, both["snapshot_at"])
        both["eta_p10"], both["eta_p50"], both["eta_p90"] = q[:, 0], q[:, 1], q[:, 2]
        both["snapshot_id"] = sid
        frames.append(both)

    out = pd.concat(frames, ignore_index=True)
    out["risk_probability"] = out["hazard_score"]
    out["priority_rank"] = out["priority_rank"].astype("Int64")
    out["snapshot_features"] = [
        None if overdue else json.dumps({f: (None if pd.isna(v) else v) for f, v in
                                         zip(SNAPSHOT_MODEL_FEATURES, vals, strict=True)},
                                        default=str)
        for overdue, vals in zip(out["is_overdue"],
                                 out.reindex(columns=SNAPSHOT_MODEL_FEATURES).itertuples(index=False),
                                 strict=True)
    ]
    keep = ["snapshot_id", "order_id", "is_overdue", "km_score", "hazard_score",
            "lambdamart_score", "ensemble_score", "priority_rank", "risk_probability", "risk_band",
            "eta_p10", "eta_p50", "eta_p90", "snapshot_features"]
    return out[keep]


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ws = Workspace()
    hazard = HazardModel(ws.context, ws.km).fit(ws.df, TEST_START)
    ranker = _lambdamart(training_rows_for(ws.snap_train, TEST_START))
    print("models fitted on everything known before", TEST_START.date())

    # The served model must be the tested one: reproduce the published figure.
    end = ws.base["order_delivered_carrier_date"].max() + pd.Timedelta(seconds=1)
    test_rows = score_rows(eval_rows(ws.base, TEST_START, end), ws, hazard, ranker)
    reproduced = float(per_snapshot(test_rows, {"e": test_rows["ensemble_score"].to_numpy()})["e"].mean())
    published = json.loads((spec.DOCS_DIR / "research_v4_test.json").read_text())[
        "mean_precision_at_50"]["ensemble_rank"]
    if abs(reproduced - published) > 1e-9:
        raise RuntimeError(f"served model reproduces {reproduced}, published {published}")
    print(f"reproduced test Precision@50 {reproduced:.4f} == published")

    OUT.mkdir(parents=True, exist_ok=True)
    hazard.booster.save_model(OUT / "hazard.ubj")
    joblib.dump(ranker, OUT / "lambdamart.joblib")

    # Live re-scoring needs the context and survival curves of the demo days
    # only; storing those keeps the API's memory small.
    days = [pd.Timestamp(s["snapshot_id"]) for s in spec.SNAPSHOTS]
    ctx = ws.context[ws.context["day"].isin(days)].reset_index(drop=True)
    wi = sorted({int(np.searchsorted(ws.km.weeks.values, np.datetime64(d), side="right") - 1)
                 for d in days})
    km = WeeklyKM(ws.km.weeks[wi], ws.km.lane_index, ws.km.dest_index,
                  ws.km.lane_hazard[wi], ws.km.dest_hazard[wi], ws.km.net_hazard[wi],
                  ws.km.lane_n[wi], ws.km.dest_n[wi])
    categories = {c: list(ws.df[c].cat.categories) for c in CATEGORICAL_FEATURES}
    joblib.dump({"context": ctx, "km": km, "categories": categories}, OUT / "support.joblib")

    scores = demo_snapshot_scores(ws, hazard, ranker)
    scores["model_version"] = MODEL_VERSION
    scores.to_parquet(OUT / "snapshot_scores.parquet", index=False)

    test = json.loads((spec.DOCS_DIR / "research_v4_test.json").read_text())
    meta = {
        "model_version": MODEL_VERSION,
        "model_family": "survival ensemble (Kaplan-Meier + discrete-time hazard + LambdaMART)",
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_cutoff": str(TEST_START.date()),
        "scored_on": "the snapshot day, from what was known at its start",
        "bands": {"high": f"priority rank <= {spec.REVIEW_CAPACITY_K}",
                  "medium": f"rest of the top {spec.MEDIUM_BAND_SHARE:.0%} of the day's queue",
                  "low": "the remainder"},
        "files": {p.name: _sha(p) for p in sorted(OUT.iterdir()) if p.name != "meta.json"},
        "test_precision_at_50": test["mean_precision_at_50"]["ensemble_rank"],
        "test_vs_kaplan_meier": test["vs_baseline"],
        "arrival_estimate_test": test["descriptive"]["arrival_estimate_hazard_gbm"],
        "report": "docs/research_v4.md",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for sid, g in scores.groupby("snapshot_id"):
        r = g[~g["is_overdue"]]
        print(f"{sid}: {len(r):,} ranked, bands {r['risk_band'].value_counts().to_dict()}, "
              f"mean P(late) {r['risk_probability'].mean():.3f}")
    print("wrote", ", ".join(p.name for p in sorted(OUT.iterdir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
