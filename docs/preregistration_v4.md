# Pre-registration: v4 survival and learning-to-rank models

Written on 2026-10-04 **before** any v4 candidate was built or scored. It
follows v3 (`preregistration_v3.md`, `research_v3.md`), whose selected method
is the no-training Kaplan–Meier rule, `rule_km`. Sections 1–6 may not change
once development results exist; amendments go in section 7 with a date.

---

## 1. The question

Does a model that learns from each order's own details beat the Kaplan–Meier
rule at ranking the orders still in transit on a snapshot day? The target,
the population, the folds and the metric are all unchanged from v3.

## 2. Disclosure: what is already known

This is the **second use** of the 2018-06/08 test period, and the reader
should weigh the result accordingly:

- v3 scored twelve methods on it. Their test results are known to the author,
  including that the pairwise ranker (0.402) and the share-of-window rule
  (0.391) scored above the selected `rule_km` (0.349), within noise.
- LambdaMART is a candidate here partly *because* the pairwise ranker did
  well, both in development and on test. This is a bias, and it is stated.
- Development probes after v3 (dev folds only) found that shorter or
  recency-weighted Kaplan–Meier windows, finer geography, and a rank blend of
  the two free rules did not beat `rule_km`. None of them is a v4 candidate.

To offset the second look, the bar for reaching test is stricter (section 5),
and only **one** model may be tested.

## 3. Development protocol

Identical to v3: folds F1–F4, settled-label training before each fold,
weekly evaluation snapshots, cohort limited to handovers inside the fold,
Precision@50 per snapshot, paired comparisons.

## 4. Candidates

| id | method |
|---|---|
| `hazard_gbm` | Discrete-time survival model. One row per order per calendar day at risk, label = delivered that day. XGBoost learns the daily delivery hazard from the order's handover features, its age, days to the promise, the lane's Kaplan–Meier hazard and the day's congestion. On a snapshot day the chance of being late is the product of surviving every remaining day up to and including the promised day. |
| `aft_xgb` | XGBoost accelerated-failure-time model (`survival:aft`, normal errors), one row per order with right-censoring at the training cutoff. P(late \| undelivered now) = S(promise) / S(now). |
| `lambdamart` | XGBoost `rank:ndcg` on the v3 snapshot rows, grouped by snapshot day, with top-k pair selection. |
| `ensemble_rank` | Equal-weight average of the within-snapshot ranks of `rule_km`, `hazard_gbm` and `lambdamart`. Nothing is fitted. |

Fixed settings, chosen before any result: XGBoost `max_depth=6`,
`learning_rate=0.05`, `n_estimators=500`, `subsample=0.9`,
`colsample_bytree=0.8`, `min_child_weight=10`; AFT scale 1.0; LambdaMART
truncation at 50. No other configuration is tried. A candidate that errors
is reported as failed, not replaced.

## 5. Selection rule (development only)

1. Score each candidate by the mean of its four fold-mean Precision@50.
2. Take the best v4 candidate.
3. **Bar to reach test:** its paired advantage over `rule_km`, across all
   development snapshots, must exceed **two** standard errors. If it does not,
   v3 stands, **the test is not run**, and v4 is reported as a negative result.
4. If the winner is probability-valued, choose raw versus isotonic
   calibration as in v3 step 5.

## 6. Test (only if section 5 selects a model)

The selected model is retrained on everything known before 2018-06-01 and
scored once on the 11 test snapshots.

| | target |
|---|---|
| **Primary** | paired advantage over `rule_km` on test, 95% bootstrap interval excluding zero |
| Secondary | Brier skill > 0 against a constant at the **hindsight** test late rate (v3: −0.043) |
| Secondary, survival models only | the revised arrival estimate: median absolute error in days, and the coverage of an 80% interval (target: between 70% and 90%) |

A missed target is reported as missed.

## 7. Amendments

**2026-10-04, after development, before any test computation.**

1. Development selected `ensemble_rank` (+0.0190 over `rule_km`, two
   standard errors = 0.0130). It is an average of ranks, not a probability,
   so the two secondary targets in section 6 (Brier skill, arrival estimate)
   **do not apply to the selected model** and are reported as not applicable
   rather than as met or missed.
2. Because the arrival estimate is a capability of `hazard_gbm`, one of the
   ensemble's three components, the test also reports `hazard_gbm`'s and
   `rule_km`'s calibration and `hazard_gbm`'s arrival estimate
   **descriptively**. These numbers select nothing and are not targets.
