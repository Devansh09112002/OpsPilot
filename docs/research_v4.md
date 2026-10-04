# OpsPilot v4 - Survival models and learning-to-rank

Every number is produced by `python -m ml_pipeline.walkforward_v4` and
rendered by `python -m ml_pipeline.render_v4`. The candidates, their fixed
settings, the selection bar and the test targets were written in
[`preregistration_v4.md`](preregistration_v4.md) before any v4 model existed.
The test ran once against pre-registration `9b854387371ccb43...`.

**This is the second use of the test period**, disclosed in the
pre-registration: v3's test results were known when v4 was designed. The bar
to reach the test was raised to two standard errors to offset that.

## 1. Headline

| | Precision@50 on test | late orders in the top 50 | lift over random | Recall@50 |
|---|---|---|---|---|
| v2, as deployed | 0.135 | ~7 | 2.8x | 0.122 |
| v3, Kaplan-Meier | 0.349 | ~17 | 7.3x | 0.312 |
| **v4, ensemble** | **0.405** | **~20** | **8.5x** | **0.363** |

Against v3 on the same 11 test days: **+0.056**, 95% bootstrap
interval [+0.035, +0.075], better on
10 of 11 days.

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

| model | F1 | F2 | F3 | F4 | mean |
|---|---|---|---|---|---|
| Kaplan-Meier rule (v3) | 0.420 | 0.497 | 0.658 | 0.660 | **0.559** |
| Discrete-time hazard model (XGBoost) | 0.378 | 0.513 | 0.678 | 0.620 | **0.547** |
| Accelerated failure time (XGBoost AFT) | 0.295 | 0.470 | 0.609 | 0.603 | **0.494** |
| LambdaMART (rank:ndcg, top-50) | 0.407 | 0.497 | 0.662 | 0.651 | **0.554** |
| **Ensemble: KM + hazard + LambdaMART** | 0.432 | 0.528 | 0.684 | 0.660 | **0.576** |

- best v4 candidate: ensemble_rank (0.5759); rule_km: 0.5587
- paired advantage over rule_km: +0.0190, 2 SE = 0.0130
- bar passed -> test

No single learned model beat Kaplan-Meier in development. Their **combination**
did, in every regime it did not tie. The three make different mistakes:
Kaplan-Meier knows only the lane's timing, the hazard model adds the order's
own details, and LambdaMART is trained on the ranking itself.

## 4. Test, run once

| model | Precision@50 | lift | Recall@50 |
|---|---|---|---|
| v2 as deployed (handover XGBoost) | 0.135 | 2.8x | 0.122 |
| Kaplan-Meier rule (v3) | 0.349 | 7.3x | 0.312 |
| Discrete-time hazard model (XGBoost) | 0.367 | 7.7x | 0.322 |
| Accelerated failure time (XGBoost AFT) | 0.385 | 8.1x | 0.334 |
| LambdaMART (rank:ndcg, top-50) | 0.400 | 8.4x | 0.359 |
| **Ensemble: KM + hazard + LambdaMART** | 0.405 | 8.5x | 0.363 |

| pre-registered target | result | status |
|---|---|---|
| beats v3 on test, 95% interval above zero | [+0.035, +0.075] | met |
| Brier skill > 0 vs the hindsight test rate | the ensemble is a ranking | not applicable |
| arrival estimate: 80% interval covers 70-90% | applies to a selected hazard model | not applicable |

Per snapshot (Precision@50):

| snapshot | late | KM | hazard | LambdaMART | ensemble |
|---|---|---|---|---|---|
| 2018-06-08 | 16 | 0.14 | 0.14 | 0.16 | 0.18 |
| 2018-06-15 | 38 | 0.26 | 0.22 | 0.26 | 0.24 |
| 2018-06-22 | 46 | 0.30 | 0.32 | 0.28 | 0.34 |
| 2018-06-29 | 53 | 0.36 | 0.34 | 0.38 | 0.38 |
| 2018-07-06 | 43 | 0.26 | 0.28 | 0.40 | 0.32 |
| 2018-07-13 | 48 | 0.44 | 0.44 | 0.50 | 0.50 |
| 2018-07-20 | 60 | 0.36 | 0.44 | 0.50 | 0.46 |
| 2018-07-27 | 62 | 0.24 | 0.24 | 0.32 | 0.34 |
| 2018-08-03 | 88 | 0.54 | 0.62 | 0.58 | 0.60 |
| 2018-08-10 | 133 | 0.56 | 0.62 | 0.64 | 0.64 |
| 2018-08-17 | 89 | 0.38 | 0.38 | 0.38 | 0.46 |

## 5. Two capabilities, reported descriptively

Amendment 2 of the pre-registration: these numbers select nothing.

**Calibrated risk.** The hazard model is the first model in OpsPilot to beat
a constant set at the *hindsight* test late rate:

| | Brier | skill vs hindsight constant | mean predicted | observed |
|---|---|---|---|---|
| Kaplan-Meier | 0.0473 | -0.043 | 0.118 | 0.048 |
| Hazard model | 0.0413 | **+0.090** | 0.075 | 0.048 |

It still over-predicts in a calm period, by less.

**A revised arrival date.** For every order in transit, the hazard model's
survival curve gives the day by which 10%, 50% and 90% of similar parcels
arrive. On test, the median estimate was off by **2
day(s)** for the typical parcel (mean 3.4), and the 80% range
contained the real delivery day **87%** of the time.
An operations team can act on "most likely arrives on the 14th, almost surely by
the 19th" more directly than on a risk score.

## 6. What a careful reader should know

- **Second look at test.** Disclosed above. The two-SE development bar and the
  single tested model limit how much the second look can inflate the result.
- **LambdaMART alone scored 0.400 on test**, close to the
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
