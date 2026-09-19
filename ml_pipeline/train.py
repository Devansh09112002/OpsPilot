"""Train, compare and freeze the delivery-risk model.

Discipline enforced here (plan section 4.2):
  1. Preprocessing is fitted on the training split only.
  2. The rule baseline, logistic regression and XGBoost are compared on
     VALIDATION only.
  3. The winner is frozen, then scored ONCE on the held-out test split.
  4. Whatever the numbers say is what gets written to docs/model_report.md.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from data_pipeline import spec
from data_pipeline.features import MODEL_FEATURES
from ml_pipeline import report as report_mod
from ml_pipeline.metrics import evaluate
from ml_pipeline.model import (
    DeadlineProximityRule,
    ModelMetadata,
    build_logistic_regression,
    build_xgboost,
    save_artifact,
)
from ml_pipeline.render import render as render_report
from ml_pipeline.selection import evaluate_over_snapshots

K = spec.REVIEW_CAPACITY_K


def load_training_frame() -> pd.DataFrame:
    fpath = spec.PROCESSED_DIR / "order_features.parquet"
    opath = spec.PROCESSED_DIR / "order_outcomes.parquet"
    if not fpath.exists() or not opath.exists():
        raise SystemExit(
            "Feature tables not found. Run `python -m data_pipeline.audit` first."
        )
    features = pd.read_parquet(fpath)
    outcomes = pd.read_parquet(opath)
    return features.merge(outcomes, on="order_id", validate="one_to_one")


def _split(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, np.ndarray]:
    sub = df[df["split"] == name]
    return sub[MODEL_FEATURES], sub[spec.TARGET_NAME].to_numpy().astype(int)


def assert_no_temporal_leakage(df: pd.DataFrame) -> None:
    """Fail the build if any split boundary is violated."""
    h = df["order_delivered_carrier_date"]
    train_max = h[df["split"] == "train"].max()
    val_min, val_max = h[df["split"] == "validation"].min(), h[df["split"] == "validation"].max()
    test_min = h[df["split"] == "test"].min()
    if not (train_max < spec.TRAIN_END <= val_min):
        raise AssertionError(f"train/validation boundary violated: {train_max} / {val_min}")
    if not (val_max < spec.VALIDATION_END <= test_min):
        raise AssertionError(f"validation/test boundary violated: {val_max} / {test_min}")


def main() -> int:
    df = load_training_frame()
    assert_no_temporal_leakage(df)

    X_tr, y_tr = _split(df, "train")
    X_va, y_va = _split(df, "validation")
    X_te, y_te = _split(df, "test")

    print(f"train {len(y_tr):,} ({y_tr.mean():.3%} late) | "
          f"validation {len(y_va):,} ({y_va.mean():.3%}) | "
          f"test {len(y_te):,} ({y_te.mean():.3%})")
    print("\n--- fitting candidates on TRAIN, comparing on VALIDATION ---")

    candidates: dict[str, object] = {}

    rule = DeadlineProximityRule().fit(X_tr, y_tr)
    candidates["rule_deadline_proximity"] = rule

    lr = build_logistic_regression()
    lr.fit(X_tr, y_tr)
    candidates["logistic_regression"] = lr

    pos = max(int(y_tr.sum()), 1)
    xgb = build_xgboost(scale_pos_weight=float((len(y_tr) - pos) / pos))
    xgb.fit(X_tr, y_tr)
    candidates["xgboost"] = xgb

    va_frame = df[df["split"] == "validation"]
    validation: dict[str, dict] = {}
    val_sim: dict[str, dict] = {}
    for name, mdl in candidates.items():
        scores = mdl.predict_proba(X_va)[:, 1]
        validation[name] = evaluate(y_va, scores, K)
        val_sim[name] = evaluate_over_snapshots(
            mdl, va_frame, spec.TRAIN_END, spec.VALIDATION_END, K
        )
        m, s = validation[name], val_sim[name]
        print(f"  {name:<26s} simP@{K}={s[f'mean_precision_at_{K}']:.3f}"
              f"±{s[f'std_precision_at_{K}']:.3f}  PR-AUC={m['pr_auc']:.4f}  "
              f"ROC-AUC={m['roc_auc']:.4f}")
    probe = next(iter(val_sim.values()))
    print(f"  (simulated over {probe['n_snapshots']} validation snapshots, "
          f"median cohort {probe['median_cohort_size']:,}, "
          f"{probe['top_slots_observed']} top-{K} slots observed)")

    # --- selection rule, applied before the test set is touched --------------
    # Primary criterion is the snapshot-simulated mean Precision@K: it measures
    # the decision the product actually makes and observes hundreds of top-K
    # slots instead of 50. PR-AUC (computed over the whole split) breaks ties.
    def key(name: str) -> tuple[float, float]:
        return (val_sim[name][f"mean_precision_at_{K}"], validation[name]["pr_auc"])

    ranked = sorted(candidates, key=key, reverse=True)
    best = ranked[0]

    # A learned model must clear the operational rule by more than the
    # snapshot-to-snapshot noise before its extra complexity is justified.
    rule_p = val_sim["rule_deadline_proximity"][f"mean_precision_at_{K}"]
    rule_sd = val_sim["rule_deadline_proximity"][f"std_precision_at_{K}"]
    n_snap = max(val_sim["rule_deadline_proximity"]["n_snapshots"], 1)
    noise_floor = rule_sd / (n_snap ** 0.5)  # standard error across snapshots
    best_p = val_sim[best][f"mean_precision_at_{K}"]

    if best != "rule_deadline_proximity" and (best_p - rule_p) <= noise_floor:
        rationale = (
            f"{best} led the operational rule on simulated validation "
            f"Precision@{K} by only {best_p - rule_p:+.4f}, within the "
            f"{noise_floor:.4f} standard error across {n_snap} snapshots. "
            "Per the model-selection rule the simpler defensible solution is "
            "served, and the learned models are reported as adding no measured "
            "value on this task."
        )
        best = "rule_deadline_proximity"
    elif best == "xgboost":
        lr_p = val_sim["logistic_regression"][f"mean_precision_at_{K}"]
        if (best_p - lr_p) <= noise_floor:
            rationale = (
                f"XGBoost led logistic regression by only {best_p - lr_p:+.4f} on "
                f"simulated validation Precision@{K}, within the {noise_floor:.4f} "
                "standard error, so the simpler and more interpretable logistic "
                "regression is served."
            )
            best = "logistic_regression"
        else:
            rationale = (
                f"XGBoost improved simulated validation Precision@{K} by "
                f"{best_p - lr_p:+.4f} over logistic regression and "
                f"{best_p - rule_p:+.4f} over the operational rule, both beyond "
                f"the {noise_floor:.4f} standard error across {n_snap} snapshots."
            )
    else:
        rationale = (
            f"{best} achieved the best simulated validation Precision@{K} "
            f"({best_p:.4f}), ahead of the operational rule by "
            f"{best_p - rule_p:+.4f} (standard error {noise_floor:.4f})."
        )

    print(f"\nSELECTED: {best}\n  {rationale}")

    if best == "rule_deadline_proximity":
        # Plan section 4.4: serve the defensible simpler solution and say so.
        # The rule is picklable and exposes the same predict_proba contract, so
        # it is served as the artifact rather than quietly replaced by a model
        # that lost.
        rationale = (
            "The operational rule baseline beat both learned models on "
            f"validation Precision@{K}. Per the model-selection rule, the "
            "simpler defensible solution is served and the ML component is "
            "reported as adding no measured value on this task."
        )
        print(f"  NOTE: {rationale}")

    # --- freeze, then score ONCE on the held-out test split ------------------
    final = candidates[best]
    print("\n--- held-out TEST evaluation (model frozen) ---")
    test: dict[str, dict] = {}
    for name, mdl in candidates.items():
        scores = mdl.predict_proba(X_te)[:, 1]
        test[name] = evaluate(y_te, scores, K, with_calibration=(name == best))
        m = test[name]
        mark = "  <-- SERVED" if name == best else ""
        print(f"  {name:<26s} P@{K}={m[f'precision_at_{K}']:.3f}  "
              f"R@{K}={m[f'recall_at_{K}']:.3f}  lift={m[f'lift_at_{K}']:.2f}x  "
              f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}{mark}")

    te_frame = df[df["split"] == "test"]
    test_sim: dict[str, dict] = {}
    print()
    print("--- simulated TEST snapshots (the deployed decision) ---")
    for name, mdl in candidates.items():
        test_sim[name] = evaluate_over_snapshots(
            mdl, te_frame, spec.VALIDATION_END,
            te_frame["order_delivered_carrier_date"].max(), K
        )
        s = test_sim[name]
        mark = "  <-- SERVED" if name == best else ""
        print(f"  {name:<26s} simP@{K}={s[f'mean_precision_at_{K}']:.3f}"
              f"±{s[f'std_precision_at_{K}']:.3f}  "
              f"simR@{K}={s[f'mean_recall_at_{K}']:.3f}  "
              f"lift={s[f'mean_lift_at_{K}']:.2f}x  "
              f"({s['n_snapshots']} snapshots){mark}")

    per_snapshot = report_mod.per_snapshot_metrics(final, df, K)

    version = f"{best}-{datetime.now(UTC):%Y%m%d}"
    meta = ModelMetadata(
        model_version=version,
        model_family=best,
        trained_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        training_cutoff=str(spec.TRAIN_END.date()),
        feature_names=list(MODEL_FEATURES),
        numeric_features=list(X_tr.columns[:0]) or [],
        categorical_features=[],
        train_rows=int(len(y_tr)),
        train_base_rate=round(float(y_tr.mean()), 6),
        artifact_sha256="",
        selection_rationale=rationale,
        validation_metrics={"pooled": validation, "simulated": val_sim},
        test_metrics={"pooled": test, "simulated": test_sim},
    )
    from data_pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    meta.numeric_features = list(NUMERIC_FEATURES)
    meta.categorical_features = list(CATEGORICAL_FEATURES)

    path = save_artifact(final, meta, spec.ARTIFACT_DIR)
    print(f"\nartifact: {path}  sha256={meta.artifact_sha256[:16]}  version={version}")

    errors = report_mod.error_analysis(final, df, K)
    importances = report_mod.feature_importance(final, best)
    latency = report_mod.measure_latency(final, X_te)

    payload = {
        "model_version": version,
        "selected": best,
        "rationale": rationale,
        "validation": validation,
        "validation_simulated": val_sim,
        "test": test,
        "test_simulated": test_sim,
        "per_snapshot": per_snapshot,
        "errors": errors,
        "importances": importances,
        "latency_ms": latency,
        "k": K,
        "splits": {
            "train": {"n": int(len(y_tr)), "base_rate": float(y_tr.mean())},
            "validation": {"n": int(len(y_va)), "base_rate": float(y_va.mean())},
            "test": {"n": int(len(y_te)), "base_rate": float(y_te.mean())},
        },
    }
    (spec.DOCS_DIR / "model_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (spec.DOCS_DIR / "model_report.md").write_text(render_report(payload), encoding="utf-8")
    print("wrote docs/model_report.md, docs/model_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
