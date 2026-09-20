# OpsPilot - Model Report

**Model version:** `xgboost-20260920`
**Served family:** xgboost
**Primary metric:** mean Precision@50 across simulated snapshots - the share of
genuinely late orders among the top 50 an operations team would review.

Every number here is produced by `python -m ml_pipeline.train`. Nothing in this
file is hand-entered.

## 1. Headline result

Ranking one snapshot at a time, exactly as the deployed risk queue does, the
served model puts **13.5%** genuinely late
orders in its top 50, against a **4.8%** background
rate. That is a **2.82x lift**: a reviewer
working the flagged queue meets a late order
2.8 times as often as one reviewing 50 orders
at random, while catching 12.2% of all late
orders in the cohort.

The operational rule the product must beat manages
11.6% (2.44x) on
the same snapshots.

These results are modest in absolute terms. Delivery lateness is only partly
predictable from what is known at carrier handover, and this report does not
inflate that.

## 2. Why accuracy is not reported

The test-period base rate is 3.015%. A model that always
answers "on time" scores 97.0% accuracy and finds
nothing. Ranking quality under a fixed review capacity is the only metric that
reflects the actual decision.

## 3. Evaluation protocol

Two protocols appear below and they answer different questions.

- **Simulated snapshots (primary).** Every 7 days across a period, take the
  orders genuinely in transit and still pre-deadline at that moment, rank them,
  and measure Precision@50. Cohorts run to roughly 1,000-2,000 orders, matching
  the product. Averaging across snapshots observes hundreds of top-50 slots
  instead of 50.
- **Pooled split (secondary).** Rank an entire split at once and take the top
  50. That measures only 50 orders out of ~19,000 and picks the most extreme
  cases across three and a half months, so it reads far higher than anything the
  product can deliver. Reported for completeness, not as a headline.

Model selection used the simulated protocol on **validation only**.

## 4. Data splits

Chronological, cut on `order_delivered_carrier_date` (the prediction moment),
frozen before any model comparison.

| split | orders | late rate |
|---|---|---|
| train | 56,363 | 6.023% |
| validation | 20,781 | 10.779% |
| test | 18,808 | 3.015% |

The late rate moves considerably between periods: validation covers the
March-2018 disruption at 10.8% while test sits at
3.0%. That is genuine distribution shift in the Olist
data, and it is why validation figures run higher than test figures throughout.

## 5. Validation comparison - selection happened here

Simulated snapshots, validation period only:

| model | mean Precision@K | mean Recall@K | Lift@K | snapshots | base rate |
|---|---|---|---|---|---|
| rule_deadline_proximity | 0.225 +/- 0.095 | 0.029 | 1.05x | 12 | 21.381% |
| logistic_regression | 0.542 +/- 0.183 | 0.067 | 2.53x | 12 | 21.381% |
| xgboost | 0.583 +/- 0.205 | 0.072 | 2.73x | 12 | 21.381% |

**Selection:** XGBoost improved simulated validation Precision@50 by +0.0416 over logistic regression and +0.3583 over the operational rule, both beyond the 0.0275 standard error across 12 snapshots.

Pooled over the whole validation split, for reference:

| model | Precision@50 | Recall@50 | Lift@50 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| rule_deadline_proximity | 0.800 | 0.018 | 7.42x | 0.1438 | 0.5502 | 0.11289 |
| logistic_regression | 0.800 | 0.018 | 7.42x | 0.2754 | 0.7113 | 0.26453 |
| xgboost | 0.900 | 0.020 | 8.35x | 0.2742 | 0.6922 | 0.32394 |

## 6. Held-out test results - scored after freezing

Simulated snapshots, the number that describes the product:

| model | mean Precision@K | mean Recall@K | Lift@K | snapshots | base rate |
|---|---|---|---|---|---|
| rule_deadline_proximity | 0.116 +/- 0.102 | 0.107 | 2.44x | 11 | 4.768% |
| logistic_regression | 0.167 +/- 0.092 | 0.160 | 3.51x | 11 | 4.768% |
| xgboost | 0.135 +/- 0.101 | 0.122 | 2.82x | 11 | 4.768% |

**Ordering instability, stated plainly.** On the held-out test period `logistic_regression` scored +0.033 higher than the served `xgboost` (standard error 0.030 across 11 snapshots, so the gap sits around the noise floor). `xgboost` was selected on validation and that selection was **not** revisited after seeing test results; doing so would make the test estimate meaningless. The honest reading is that the two learned models are not clearly distinguishable on this data, and that both beat the operational rule.

Pooled over the whole test split:

| model | Precision@50 | Recall@50 | Lift@50 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| rule_deadline_proximity | 1.000 | 0.088 | 33.17x | 0.3062 | 0.7938 | 0.07690 |
| logistic_regression | 0.620 | 0.055 | 20.57x | 0.1628 | 0.7966 | 0.35925 |
| xgboost | 0.960 | 0.085 | 31.84x | 0.2752 | 0.7633 | 0.28485 |

The pooled row needs care. A pooled Precision@50 of
0.960 looks impressive, but it comes from taking
the 50 most extreme orders out of 18,808 spanning months. The
product never gets to make that choice: it ranks one snapshot at a time, which
is the 0.135 figure above. Wherever the two
are quoted, the simulated one is the honest one.

## 7. Calibration

The raw model output is **not** a probability. `scale_pos_weight` rebalances
the classes during training, which inflates every score: measured on held-out
data, a raw 0.65 corresponded to a **2.4%** observed late rate, a 27x
overstatement. Displaying that number to a person is misleading however
carefully it is captioned.

An isotonic regression fitted on the **validation** split maps the raw score to
an estimated probability. Isotonic is monotonic, so the ranking is unchanged
and every metric above still holds; the queue continues to sort on the raw
score because calibration introduces ties.

|  | raw score | calibrated |
|---|---|---|
| Brier score (test) | 0.28485 | **0.02952** |
| mean predicted (test) | 0.4824 | 0.0886 |
| observed base rate | 0.0301 | 0.0301 |

Calibration improves the Brier score **9.6x**, and the mean calibrated estimate (0.0886) now sits close to the observed base rate (0.0301).

**High** is cut at 0.1500, which is the policy's escalation threshold rather than a quantile: a band that disagreed with the policy would let an order read as "medium risk" and still be escalated. So **high** means exactly "clears the escalation threshold". **Medium** at 0.1250 is the validation 75th percentile and is presentational only.

Reliability of the **calibrated** estimate on the held-out test split:

| predicted band | orders | mean predicted | observed late rate |
|---|---|---|---|
| 0.0-0.1 | 10,098 | 0.046 | 0.012 |
| 0.1-0.2 | 8,410 | 0.126 | 0.037 |
| 0.2-0.3 | 96 | 0.257 | 0.146 |
| 0.3-0.4 | 17 | 0.365 | 0.353 |
| 0.4-0.5 | 105 | 0.470 | 0.429 |
| 0.5-0.6 | 9 | 0.560 | 0.333 |
| 0.6-0.7 | 6 | 0.619 | 0.667 |
| 0.7-0.8 | 19 | 0.728 | 0.684 |
| 0.8-0.9 | 28 | 0.883 | 0.929 |
| 0.9-1.0 | 20 | 0.933 | 1.000 |

The calibrated value is what the product displays and what the demo policy's
escalation threshold is stated on. `docs/api_contracts.md` returns both:
`risk_probability` (calibrated, for reading) and `ranking_score` (raw, the
sort key).

## 8. The three frozen demo snapshots

Exactly what the deployed risk queue shows, restricted to pre-deadline orders.

| snapshot | pre-deadline orders | base rate | Precision@50 | Recall@50 | Lift@50 | ROC-AUC |
|---|---|---|---|---|---|---|
| 2018-06-20 | 1,449 | 3.037% | 0.060 | 0.068 | 1.98x | 0.6218 |
| 2018-07-18 | 851 | 6.345% | 0.100 | 0.093 | 1.58x | 0.5802 |
| 2018-08-15 | 1,531 | 7.185% | 0.220 | 0.100 | 3.06x | 0.6465 |

These cohorts hold roughly 850-1,550 orders with 40-110 late ones, so individual
snapshot figures carry visible sampling noise. The averaged row in section 6 is
the more reliable estimate.

## 9. Feature influence

| feature | weight |
|---|---|
| customer_state_RJ | +0.0339 |
| customer_state_SP | +0.0330 |
| route_prior_late_rate | +0.0266 |
| shipping_limit_slack_hours | +0.0234 |
| customer_state_MG | +0.0224 |
| is_cross_state | +0.0217 |
| seller_state_SP | +0.0204 |
| customer_state_PE | +0.0188 |
| handover_week_of_year | +0.0188 |
| days_handover_to_estimate | +0.0187 |
| route_prior_orders | +0.0176 |
| handover_month | +0.0170 |
| customer_state_AL | +0.0170 |
| hours_purchase_to_handover | +0.0161 |
| customer_state_MA | +0.0148 |

## 10. Error analysis

**False positives in the pooled top 50** (2 of 50): median slack
at handover 1.0 days, median
fulfilment time 272.4 h,
0% cross-state. These are genuinely tight
shipments that happened to arrive on time.

**Late orders ranked low** (283): median slack
10.0 days. Orders with
comfortable slack that were delayed anyway carry no signal observable at
handover. This is the irreducible error class for a prediction made at carrier
handover, and it caps how high Precision@50 can go.

## 11. Inference cost and latency

Single-row `predict_proba` over 200 held-out rows, development
machine, CPU only: median **4.18 ms**, p95 **5.07 ms**.
Model inference is not the latency bottleneck; the network round trip and, for
investigations, the LLM call dominate.

## 12. Limitations

- Trained on 2016-2018 Brazilian marketplace data. It does not transfer to
  another market or period without retraining.
- The prediction is made once, at carrier handover. The dataset carries no
  in-transit scan events, so the score is never refreshed mid-transit; the UI
  labels it "predicted at carrier handover" rather than implying live
  recalculation.
- The output ranks risk. It does not diagnose a cause, and neither this report
  nor the agent presents a correlation as a causal explanation.
- Two features read prior delivery outcomes. They are restricted to outcomes
  strictly earlier than each order own handover, but they still make the model
  sensitive to how much history a seller or route has accumulated.
- Class prevalence shift between periods is real and unmodelled. Absolute
  precision on a future period could differ materially from the test figure.
- Orders handed to the carrier after their promised date are excluded from the
  modelling population: they are late by arithmetic, not by prediction. The
  product shows them as *overdue* instead. See `docs/data_audit.md` section 2.
