"""Render docs/research_v4.md from the v4 development and test results.

    python -m ml_pipeline.render_v4

Every number is read from `docs/research_v4_*.json`, the per-snapshot CSVs
and, for comparison, `docs/research_v3_test.json`. Nothing is typed in.
"""

from __future__ import annotations

import json

import pandas as pd

from data_pipeline import spec

DOCS = spec.DOCS_DIR
LABELS = {
    "rule_km": "Kaplan-Meier rule (v3)",
    "hazard_gbm": "Discrete-time hazard model (XGBoost)",
    "aft_xgb": "Accelerated failure time (XGBoost AFT)",
    "lambdamart": "LambdaMART (rank:ndcg, top-50)",
    "ensemble_rank": "**Ensemble: KM + hazard + LambdaMART**",
}


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def _na(value):
    return "not applicable" if value is None else ("met" if value else "**missed**")


def render() -> str:
    dev = json.loads((DOCS / "research_v4_development.json").read_text(encoding="utf-8"))
    test = json.loads((DOCS / "research_v4_test.json").read_text(encoding="utf-8"))
    v3 = json.loads((DOCS / "research_v3_test.json").read_text(encoding="utf-8"))
    snaps = pd.read_csv(DOCS / "research_v4_test_snapshots.csv")

    sel = test["selected"]
    p = test["mean_precision_at_50"]
    base = v3["base_rate"]
    v2 = v3["mean_precision_at_50"]["v2_published"]
    recall = {n: float(snaps[f"{n}__recall"].mean()) for n in p}
    vs = test["vs_baseline"]
    desc = test["descriptive"]
    eta = desc.get("arrival_estimate_hazard_gbm") or {}
    cal_km, cal_hz = desc["calibration_rule_km"], desc["calibration_hazard_gbm"]
    folds = list(next(iter(dev["summary"].values()))["fold_means"])

    dev_rows = [[LABELS[n], *[f"{s['fold_means'][f]:.3f}" for f in folds], f"**{s['score']:.3f}**"]
                for n, s in dev["summary"].items()]
    test_rows = [["v2 as deployed (handover XGBoost)", f"{v2:.3f}", f"{v2 / base:.1f}x",
                  f"{v3['mean_recall_at_50']['v2_published']:.3f}"]]
    test_rows += [[LABELS[n], f"{p[n]:.3f}", f"{p[n] / base:.1f}x", f"{recall[n]:.3f}"]
                  for n in ["rule_km", "hazard_gbm", "aft_xgb", "lambdamart", "ensemble_rank"]]
    snap_rows = [[r.snapshot_at, str(r.late), f"{r.rule_km:.2f}", f"{r.hazard_gbm:.2f}",
                  f"{r.lambdamart:.2f}", f"{getattr(r, sel):.2f}"] for r in snaps.itertuples()]
    targets = test["targets"]

    return f"""# OpsPilot v4 - Survival models and learning-to-rank

Every number is produced by `python -m ml_pipeline.walkforward_v4` and
rendered by `python -m ml_pipeline.render_v4`. The candidates, their fixed
settings, the selection bar and the test targets were written in
[`preregistration_v4.md`](preregistration_v4.md) before any v4 model existed.
The test ran once against pre-registration `{test['preregistration_sha256'][:16]}...`.

**This is the second use of the test period**, disclosed in the
pre-registration: v3's test results were known when v4 was designed. The bar
to reach the test was raised to two standard errors to offset that.

## 1. Headline

| | Precision@50 on test | late orders in the top 50 | lift over random | Recall@50 |
|---|---|---|---|---|
| v2, as deployed | {v2:.3f} | ~{v2 * 50:.0f} | {v2 / base:.1f}x | {v3['mean_recall_at_50']['v2_published']:.3f} |
| v3, Kaplan-Meier | {p['rule_km']:.3f} | ~{p['rule_km'] * 50:.0f} | {p['rule_km'] / base:.1f}x | {recall['rule_km']:.3f} |
| **v4, ensemble** | **{p[sel]:.3f}** | **~{p[sel] * 50:.0f}** | **{p[sel] / base:.1f}x** | **{recall[sel]:.3f}** |

Against v3 on the same 11 test days: **{vs['mean_diff']:+.3f}**, 95% bootstrap
interval [{vs['ci95'][0]:+.3f}, {vs['ci95'][1]:+.3f}], better on
{vs['wins']} of {vs['n']} days.

## 2. The models

- **Discrete-time hazard model.** One row per parcel per day it was still
  undelivered; XGBoost learns the chance it is delivered *that day*, from its
  own details, its age, days left to the promise, its lane's Kaplan-Meier
  hazard and the day's congestion. Parcels still moving at the training
  cutoff are censored, not dropped. The chance of being late is the chance of
  surviving every day through the promised day - exactly the target. Kaplan-
  Meier is the special case that sees only the lane and the age.
- **Accelerated failure time.** XGBoost `survival:aft`: one row per parcel,
  log transit time with normal errors, right-censored.
- **LambdaMART.** XGBoost `rank:ndcg` that optimises the top 50 of each
  snapshot day directly.
- **Ensemble.** The average of each order's within-day rank under
  Kaplan-Meier, the hazard model and LambdaMART. Nothing is fitted, so it
  cannot overfit the combination.

## 3. Development (walk-forward, four regimes)

{_table(dev_rows, ['model', *folds, 'mean'])}

{chr(10).join('- ' + d for d in dev['decision'])}

No single learned model beat Kaplan-Meier in development. Their **combination**
did, in every regime it did not tie. The three make different mistakes:
Kaplan-Meier knows only the lane's timing, the hazard model adds the order's
own details, and LambdaMART is trained on the ranking itself.

## 4. Test, run once

{_table(test_rows, ['model', 'Precision@50', 'lift', 'Recall@50'])}

{_table([
    ['beats v3 on test, 95% interval above zero',
     f"[{vs['ci95'][0]:+.3f}, {vs['ci95'][1]:+.3f}]", _na(targets['primary_ci_excludes_zero'])],
    ['Brier skill > 0 vs the hindsight test rate', 'the ensemble is a ranking',
     _na(targets['brier_skill_vs_hindsight_positive'])],
    ['arrival estimate: 80% interval covers 70-90%', 'applies to a selected hazard model',
     _na(targets['eta_coverage_70_to_90'])],
], ['pre-registered target', 'result', 'status'])}

Per snapshot (Precision@50):

{_table(snap_rows, ['snapshot', 'late', 'KM', 'hazard', 'LambdaMART', 'ensemble'])}

## 5. Two capabilities, reported descriptively

Amendment 2 of the pre-registration: these numbers select nothing.

**Calibrated risk.** The hazard model is the first model in OpsPilot to beat
a constant set at the *hindsight* test late rate:

| | Brier | skill vs hindsight constant | mean predicted | observed |
|---|---|---|---|---|
| Kaplan-Meier | {cal_km['brier']:.4f} | {cal_km['skill_vs_hindsight']:+.3f} | {cal_km['mean_predicted']:.3f} | {cal_km['observed']:.3f} |
| Hazard model | {cal_hz['brier']:.4f} | **{cal_hz['skill_vs_hindsight']:+.3f}** | {cal_hz['mean_predicted']:.3f} | {cal_hz['observed']:.3f} |

It still over-predicts in a calm period, by less.

**A revised arrival date.** For every order in transit, the hazard model's
survival curve gives the day by which 10%, 50% and 90% of similar parcels
arrive. On test, the median estimate was off by **{eta.get('median_abs_error_days', float('nan')):.0f}
day(s)** for the typical parcel (mean {eta.get('mean_abs_error_days', float('nan')):.1f}), and the 80% range
contained the real delivery day **{eta.get('coverage_80pct_interval', float('nan')):.0%}** of the time.
An operations team can act on "most likely arrives on the 14th, almost surely by
the 19th" more directly than on a risk score.

## 6. What a careful reader should know

- **Second look at test.** Disclosed above. The two-SE development bar and the
  single tested model limit how much the second look can inflate the result.
- **LambdaMART alone scored {p['lambdamart']:.3f} on test**, close to the
  ensemble. In development it did not beat Kaplan-Meier alone, so it was not
  eligible on its own; the ensemble's advantage is that it was consistently
  ahead in development *and* on test.
- **The hazard model under-predicts delivery speed when conditions improve
  quickly** (seen in development: April-May 2018, after the March disruption).
  Its calibration advantage on test is real but is not a guarantee across regimes.
- **Settings were not tuned.** Every model used fixed, pre-registered
  hyperparameters; a tuned version might do better and would need its own
  evaluation.

## 7. Reproduce

```bash
python -m ml_pipeline.walkforward_v4 develop   # ~3 min
python -m ml_pipeline.walkforward_v4 test      # refuses to run twice
python -m ml_pipeline.render_v4
pytest backend/tests/test_survival.py backend/tests/test_snapshot_features.py
```
"""


def main() -> int:
    (DOCS / "research_v4.md").write_text(render(), encoding="utf-8")
    print("wrote docs/research_v4.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
