# OpsPilot v3 - Scoring on the snapshot day

Every number here is produced by `python -m ml_pipeline.walkforward` and
rendered by `python -m ml_pipeline.render_v3`. The protocol, candidate list,
selection rule and test targets were fixed in
[`preregistration_v3.md`](preregistration_v3.md) before any v3 result existed.
The test ran once; its record carries the SHA-256 of the pre-registration it
was run against (`422ab659c7c67a3b...`).

## 1. Headline

On the held-out test period (11 weekly snapshots, June-August 2018, the same
protocol as every published v2 figure), the selected method puts
**34.9%** genuinely late orders in its top 50, against **13.5%**
for the deployed v2 model and a **4.8%** base rate:

- **2.6x** the deployed model's precision; lift over random
  rises from 2.8x to **7.3x**.
- It catches **31%** of all late orders in the cohort with 50
  reviews, against 12%.
- It won **11 of 11** test days
  against the deployed model.

The selected method is a **Kaplan-Meier survival estimate with no training at
all**: the chance a parcel misses its promised date, given that it is still
undelivered today, read off the recent transit-time curve of its lane.

## 2. What changed, and why it matters more than the algorithm

v2 scored an order once, at carrier handover. The product ranks orders that
are still moving on a later day, and by then one more fact is known: the
parcel has not arrived. "Still undelivered after using 90% of its window" is
the strongest warning sign in this data, and a handover score cannot see it.

v3 builds one row per (order, snapshot day) from what was known at the start
of that day (`data_pipeline/snapshot_features.py`): elapsed and remaining
time, lane Kaplan-Meier curves in which parcels still in transit count as
censored rather than being dropped, recent late rates over orders whose
outcome is already settled, and today's congestion. A test scrambles every
outcome not yet settled on the day and asserts that no feature moves; a
negative control asserts that changing a settled outcome does move one.

The same change fixes a quiet bias in the v2 history features. They count
only orders already *delivered*; late orders arrive later, so recent late
rates were under-counted, most of all during a disruption. Counting
"past its promise and still undelivered" as a known late outcome removes it.

## 3. Development: four regimes, walk-forward

Each fold trains only on labels settled before it begins, then scores weekly
snapshots inside it. Mean Precision@50 per fold:

| model | F1 calm | F2 Black Friday peak | F3 March 2018 disruption | F4 recovery | mean |
|---|---|---|---|---|---|
| Deadline rule at handover | 0.157 | 0.305 | 0.256 | 0.217 | **0.234** |
| Logistic regression at handover | 0.103 | 0.308 | 0.431 | 0.383 | **0.306** |
| XGBoost at handover, retrained | 0.148 | 0.308 | 0.378 | 0.423 | **0.314** |
| Share of window used (no training) | 0.388 | 0.493 | 0.664 | 0.603 | **0.537** |
| **Kaplan-Meier survival (no training)** | 0.420 | 0.497 | 0.658 | 0.660 | **0.559** |
| Logistic regression, snapshot day | 0.343 | 0.532 | 0.647 | 0.646 | **0.542** |
| XGBoost, snapshot day | 0.337 | 0.455 | 0.644 | 0.611 | **0.512** |
| XGBoost + monotone constraints | 0.320 | 0.462 | 0.636 | 0.640 | **0.515** |
| XGBoost, deeper | 0.305 | 0.457 | 0.605 | 0.591 | **0.490** |
| XGBoost, shallower | 0.302 | 0.450 | 0.638 | 0.629 | **0.505** |
| XGBoost ranker (rank:pairwise) | 0.413 | 0.502 | 0.653 | 0.634 | **0.551** |

**Selection, by the pre-registered rule.**
- best v3 candidate by fold-mean score: rank_snapshot (0.5505)
- best free rule: rule_km (0.5587)
- guard 1: advantage over rule_km -0.0062 <= SE 0.0072 -> serve the rule
- calibration: raw 0.11080 vs isotonic 0.11372 -> raw

The no-training survival rule was not beaten by any learned model by more
than one standard error, so it is what the rule serves. Over all
42 development snapshots it beat the v2 recipe by
+0.244 (standard error 0.020),
winning 39 of 42.

## 4. Test, run once

| model | Precision@50 | lift | Recall@50 | late orders in the top 50 |
|---|---|---|---|---|
| v2 as deployed (handover XGBoost) | 0.135 | 2.8x | 0.122 | ~7 of 50 |
| Deadline rule at handover | 0.116 | 2.4x | 0.107 | ~6 of 50 |
| Logistic regression at handover | 0.167 | 3.5x | 0.162 | ~8 of 50 |
| XGBoost at handover, retrained | 0.211 | 4.4x | 0.208 | ~11 of 50 |
| Share of window used (no training) | 0.391 | 8.2x | 0.327 | ~20 of 50 |
| **Kaplan-Meier survival (no training)** | 0.349 | 7.3x | 0.312 | ~17 of 50 |
| Logistic regression, snapshot day | 0.353 | 7.4x | 0.309 | ~18 of 50 |
| XGBoost, snapshot day | 0.373 | 7.8x | 0.328 | ~19 of 50 |
| XGBoost + monotone constraints | 0.380 | 8.0x | 0.338 | ~19 of 50 |
| XGBoost, deeper | 0.360 | 7.6x | 0.317 | ~18 of 50 |
| XGBoost, shallower | 0.365 | 7.7x | 0.320 | ~18 of 50 |
| XGBoost ranker (rank:pairwise) | 0.402 | 8.4x | 0.357 | ~20 of 50 |

| Kaplan-Meier survival (no training) versus | mean difference | 95% bootstrap interval | days won |
|---|---|---|---|
| XGBoost at handover, retrained | +0.138 | [+0.084, +0.189] | 10 of 11 |
| v2 as deployed (handover XGBoost) | +0.215 | [+0.165, +0.255] | 11 of 11 |
| Logistic regression at handover | +0.182 | [+0.122, +0.240] | 10 of 11 |
| Deadline rule at handover | +0.233 | [+0.171, +0.291] | 11 of 11 |
| Share of window used (no training) | -0.042 | [-0.096, +0.013] | 4 of 11 |

| pre-registered target | result | status |
|---|---|---|
| mean test Precision@50 >= 0.20 | 0.349 | met |
| beats v2 retrained on the same data, interval above zero | [+0.084, +0.189] | met |
| beats the deployed v2 model | +0.215 | met |
| Brier skill > 0 against the pre-test late rate | +0.157 | met |

Per snapshot (Precision@50):

| snapshot | ranked orders | late | v2 deployed | v2 retrained | selected |
|---|---|---|---|---|---|
| 2018-06-08 | 1,034 | 16 | 0.12 | 0.18 | 0.14 |
| 2018-06-15 | 1,522 | 38 | 0.06 | 0.24 | 0.26 |
| 2018-06-22 | 1,329 | 46 | 0.10 | 0.22 | 0.30 |
| 2018-06-29 | 1,310 | 53 | 0.08 | 0.20 | 0.36 |
| 2018-07-06 | 1,231 | 43 | 0.06 | 0.12 | 0.26 |
| 2018-07-13 | 666 | 48 | 0.12 | 0.16 | 0.44 |
| 2018-07-20 | 1,366 | 60 | 0.08 | 0.12 | 0.36 |
| 2018-07-27 | 1,420 | 62 | 0.04 | 0.12 | 0.24 |
| 2018-08-03 | 1,260 | 88 | 0.36 | 0.42 | 0.54 |
| 2018-08-10 | 1,564 | 133 | 0.28 | 0.36 | 0.56 |
| 2018-08-17 | 1,495 | 89 | 0.18 | 0.18 | 0.38 |

## 5. What a careful reader should know

- **Other snapshot-day methods did as well or better on test**: XGBoost ranker (rank:pairwise) (0.402), Share of window used (no training) (0.391), XGBoost + monotone constraints (0.380), XGBoost, snapshot day (0.373), XGBoost, shallower (0.365), XGBoost, deeper (0.360), Logistic regression, snapshot day (0.353).
  The gap between the share-of-window rule and the selected method is
  -0.042 with a 95% interval of
  [-0.096, +0.013],
  which includes zero. The selection was made on development and is **not**
  revisited: switching to whichever scored best on test would make the test
  figure meaningless. The robust finding is that *scoring on the snapshot
  day* works: every one of the seven snapshot-day methods scored between
  0.349 and
  0.402, against
  0.211 for the best handover-time model. Which
  snapshot-day method is best is not resolved by 11 test days.
- **Retraining alone helps v2 a lot.** The v2 recipe retrained on data up to
  June 2018 scores 0.211 against 0.135 as deployed.
  The comparison that isolates the new information is therefore against the
  retrained model: +0.138, interval
  [+0.084, +0.189].
- **Calibration is better but still overstates in a calm period.** Brier
  0.0473 against 0.0561
  for a constant at the pre-test rate (skill +0.157),
  but slightly worse than a constant at the *hindsight* test rate (skill
  -0.043). Mean predicted
  0.118 against 0.048 observed;
  in the top decile 0.45 against
  0.24. The 120-day curves behind the
  June snapshots still contain the March disruption. A shorter or
  recency-weighted window is the obvious next step, and would need its own
  pre-registered round.
- **The test period has been looked at before.** v2's models were scored on
  it, and their per-snapshot results inspected, before v3 began. No v3
  candidate or snapshot-day feature had been computed on it.

## 6. Reproduce

```bash
python -m ml_pipeline.walkforward develop   # ~20 min on a laptop CPU
python -m ml_pipeline.walkforward test      # refuses to run if a result exists
python -m ml_pipeline.render_v3
pytest backend/tests/test_snapshot_features.py
```
