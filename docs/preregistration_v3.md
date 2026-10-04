# Pre-registration: v3 snapshot-day model

Written **before** any v3 candidate was trained or scored, and before any
snapshot-day feature was computed on the held-out test period. Nothing in
sections 1–6 may change after development results exist; if it does, the
change and the reason go in section 7 with a date.

---

## 1. The question

The served v2 model scores an order once, at carrier handover. The product
ranks orders that are still in transit on a later day, when one more fact is
known: the parcel has not arrived. v3 asks whether scoring each order **as of
the snapshot day**, with only what was known at the start of that day, ranks
late orders better.

Prediction target, eligible population and ranked cohort are unchanged from
v2, so every number below is comparable with `docs/model_report.md`:

- target: `date(delivered) > date(promised)`;
- cohort on day D: handed over by D, undelivered at D, promise not yet passed.

## 2. Data used and what is off-limits

- **Development** uses only orders handed over before **2018-06-01**.
- **Test** is the same held-out period as v2: handover on or after
  2018-06-01, 11 weekly snapshots from 2018-06-08, cohort restricted to
  test-period handovers - exactly the protocol behind the published 0.135.
- The test period has been looked at before: v2's three models were scored
  on it, and their per-snapshot results were inspected on 2026-10-04. No
  snapshot-day feature and no v3 candidate has been computed on it. This is
  stated rather than hidden.

## 3. Development protocol: walk-forward

Four folds. Each trains on everything whose label was **known before the fold
began**, then scores weekly snapshots inside the fold:

| fold | evaluated handovers | regime |
|---|---|---|
| F1 | 2017-07-01 to 2017-10-01 | calm (~3% late) |
| F2 | 2017-10-01 to 2018-01-01 | Black Friday peak |
| F3 | 2018-01-01 to 2018-04-01 | the March 2018 disruption |
| F4 | 2018-04-01 to 2018-06-01 | recovery |

- Training rows: one per ranked order every 3rd day before the fold, kept
  only if the order's lateness was settled before the fold start.
- Evaluation snapshots: weekly from fold start + 7 days, ending 7 days before
  fold end; cohort limited to orders handed over inside the fold (mirrors the
  test protocol).
- Metric: Precision@50 per snapshot. Also reported: Recall@50, lift, Brier.

## 4. Candidates

Baselines (the bar to clear):

| id | what |
|---|---|
| `rule_handover` | v2 deadline-proximity rule |
| `lr_handover` | v2 logistic regression, retrained per fold |
| `xgb_handover` | v2 XGBoost recipe, retrained per fold |
| `rule_window` | share of the promised window already used - **no training** |
| `rule_km` | Kaplan–Meier chance of missing the promise given undelivered now - **no training** |

v3 candidates, all on snapshot-day rows and features:

| id | what |
|---|---|
| `lr_snapshot` | logistic regression |
| `xgb_snapshot` | XGBoost, base parameters, no class reweighting |
| `xgb_snapshot_mono` | as above with monotone constraints |
| `xgb_snapshot_deep` | `max_depth=7, min_child_weight=20, n_estimators=600` |
| `xgb_snapshot_shallow` | `max_depth=3, n_estimators=800` |
| `rank_snapshot` | XGBoost `rank:pairwise`, grouped by snapshot day |

No other configuration is tried. If a candidate errors it is reported as
failed, not replaced.

## 5. Selection rule (applied to development results only)

1. Score each candidate by the mean of its four fold-mean Precision@50
   (equal weight per regime).
2. Take the best-scoring v3 candidate.
3. **Simplicity guard against the free rules.** Compare it with the better of
   `rule_window` and `rule_km`, paired over every development snapshot. If its
   mean advantage is not more than one standard error, the rule is served.
4. **Simplicity guard against logistic regression.** If the winner is an
   XGBoost variant and its paired advantage over `lr_snapshot` is not more
   than one standard error, `lr_snapshot` is served.
5. **Calibration.** For the selected candidate, compare the raw output with
   an isotonic map fitted on the previous fold's out-of-fold predictions,
   using mean Brier over F2–F4. The lower one is used.

## 6. Test, run once, with targets fixed now

The selected configuration is retrained on every row whose label was known
before 2018-06-01 and scored once on the 11 test snapshots. Every baseline
and every candidate is reported on test as well, not only the winner.

| | target |
|---|---|
| **Primary** | mean test Precision@50 **≥ 0.20** (v2 served: 0.135) |
| **Primary** | paired advantage over `xgb_handover` retrained on the same data, 95% bootstrap interval excluding zero |
| Secondary | paired advantage over the published v2 artifact |
| Secondary | Brier skill > 0 against a constant equal to the pre-test late rate |

A missed target is reported as missed.

## 7. Amendments

**2026-10-04, after development, before any test computation.**

1. *Implementation correction, not a protocol change.* Development selected
   `rule_km` under guard 1. Step 5 applies to "the selected candidate", and
   `rule_km` outputs a probability (the Kaplan–Meier chance of missing the
   promise given that the parcel is undelivered now), so step 5 applies to it as
   written. The first implementation ran step 5 only for trained models. The
   code was corrected to cover every probability-valued candidate, and the
   development stage was re-run so the recorded result is produced by the
   code. No candidate, selection rule, fold or target changed.
2. *Clarification.* "The pre-test late rate" in section 6 is the late rate
   over the pre-test **snapshot rows**, the same population the test rows
   come from. The order-level rate is a different population.
