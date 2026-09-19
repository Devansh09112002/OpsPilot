# OpsPilot — Data Audit

Generated: 2026-09-19T22:50:40+00:00
Status: **PASS** (every gate condition in project plan section 3.5 was asserted in code)

This report is produced by `python -m data_pipeline.audit`. It fails loudly and
writes nothing if any gate condition breaks.

## 1. Source and rights

- **Dataset:** Olist Brazilian E-Commerce Public Dataset
- **Origin:** <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>
- **License:** CC BY-NC-SA 4.0 — non-commercial, share-alike, attribution required.
- Olist is credited in the application UI and README. No endorsement is implied.
- Raw CSVs are **not** redistributed in this repository; `data_pipeline/fetch.py`
  retrieves them and verifies checksums.

Olist notes that one order may contain several items from several sellers. That
warning drives the order-level grain rules in section 3 below.

| file | bytes | sha256 (truncated) | cross-verified |
|---|---|---|---|
| olist_orders_dataset.csv | 17,654,914 | 8DF58EF3D2D7E9944010F7BE | yes |
| olist_order_items_dataset.csv | 15,438,671 | 0BC4D068C4FE38CBB01BD90E | yes |
| olist_customers_dataset.csv | 9,033,957 | 983A422239E1712DED753B3B | yes |
| olist_sellers_dataset.csv | 174,703 | 1F643D2B950373B85735E779 | yes |
| olist_products_dataset.csv | 2,379,446 | 3E6569628A17FBC75FD206EE | yes |
| product_category_name_translation.csv | 2,613 | A81F0D1F27B27E7293F761BC | yes |

Checksums were cross-verified byte-for-byte against two independent public
mirrors before use, so the pipeline is known to run on the authentic release.

## 2. Eligibility funnel

An eligible order has a valid carrier handover, a valid promised date, a known
delivery outcome and a chronologically consistent event sequence. Cancelled and
unresolved orders are excluded and counted — never silently labelled on-time.

| exclusion reason | orders removed |
|---|---|
| missing carrier handover timestamp | 1,783 |
| missing estimated delivery date | 0 |
| missing actual delivery timestamp | 1,183 |
| purchase timestamp after carrier handover | 165 |
| carrier handover after customer delivery | 23 |
| order_status is not 'delivered' | 6 |
| carrier handover after the promised date (outcome already certain) | 329 |

The final rule deserves explanation. In 329 orders the carrier handover happens
*after* the promised delivery date. Because delivery can never precede handover,
every one of those orders is late by arithmetic — 100% observed, as expected.
They are not predictions. Leaving them in was measurably distorting: the
operational rule baseline reached Precision@50 = 1.000 on validation purely by
surfacing them. The product never queues such an order either (all 3,831
pre-deadline snapshot orders have non-negative handover slack), so the modelling
population is scoped to orders whose outcome was still genuinely open at the
prediction moment. This restriction was applied before any model was selected.

**Eligible orders: 95,952** of
99,441 (96.49%).

## 3. Grain assertions

- `order_id` unique in orders: **True**
- `order_items` rows: 112,650 across
  98,666 distinct orders
- Orphan `order_items` rows: **0**
- Item measures are aggregated to order level *before* joining, and the feature
  builder asserts `order_id` uniqueness, so a multi-item order yields exactly
  **one** risk record.

## 4. Target definition

```
date(order_delivered_customer_date) > date(order_estimated_delivery_date)
```

- Late orders: **6,202** (6.4636% of eligible)
- All `order_estimated_delivery_date` values are stored at midnight: **True**
- A naive timestamp comparison would flag 7,493 orders,
  **misclassifying 1,291 same-day afternoon
  deliveries as late**. This is precisely the failure the calendar-date rule prevents.
- Timezone: Olist publishes naive local Brazilian timestamps. No timezone conversion is applied; all comparisons stay within the dataset clock.

## 5. Frozen chronological splits

Split on `order_delivered_carrier_date` (the prediction moment). Cutoffs were frozen
after data inspection and **before** any model comparison.

- train: handover &lt; 2018-03-01
- validation: 2018-03-01 &le; handover &lt; 2018-06-01
- test: handover &ge; 2018-06-01

| split | orders | late | late rate | handover from | handover to |
|---|---|---|---|---|---|
| train | 56,363 | 3,395 | 6.023% | 2016-10-08 | 2018-02-28 |
| validation | 20,781 | 2,240 | 10.779% | 2018-03-01 | 2018-05-30 |
| test | 18,808 | 567 | 3.015% | 2018-06-01 | 2018-08-29 |

Class prevalence shifts materially across periods (a real distribution shift in
the Olist data, peaking in March 2018). The model report discusses the effect
rather than hiding it.

## 6. Demo snapshots

All three snapshots fall inside the held-out test period. Membership is derived
from event timestamps (`handover <= D < delivery`), not from the terminal
`order_status` label, which would leak the outcome.

| snapshot_id | label | in transit | pre-deadline (ranked) | overdue (separate status) |
|---|---|---|---|---|
| 2018-06-20 | 20 June 2018 | 1,452 | 1,449 | 3 |
| 2018-07-18 | 18 July 2018 | 872 | 851 | 21 |
| 2018-08-15 | 15 August 2018 | 1,631 | 1,531 | 100 |

Orders already past their promised date at the snapshot moment are marked
**overdue** and shown in a separate status. Only pre-deadline orders form the
ranked risk queue, so an already-known event is never presented as a prediction.

## 7. Feature availability table

30 model features. Every one carries an availability
justification; `build_feature_table` raises if a feature lacks an entry, and
`backend/tests/test_leakage.py` re-asserts it.

| feature | source column(s) | available at | availability / leakage rationale |
|---|---|---|---|
| days_handover_to_estimate | order_estimated_delivery_date, order_delivered_carrier_date | carrier handover | Promise date is set at purchase; handover is the prediction moment. |
| hours_purchase_to_handover | order_purchase_timestamp, order_delivered_carrier_date | carrier handover | Elapsed fulfilment time; both endpoints at or before the prediction moment. |
| hours_approval_to_handover | order_approved_at, order_delivered_carrier_date | carrier handover | Approval precedes handover; NaN kept as missing, never imputed from an outcome. |
| hours_purchase_to_approval | order_purchase_timestamp, order_approved_at | payment approval | Both events precede handover. |
| shipping_limit_slack_hours | order_items.shipping_limit_date, order_delivered_carrier_date | carrier handover | Seller shipping deadline is assigned at purchase; slack measured vs actual handover. |
| purchase_dow | order_purchase_timestamp | purchase | Calendar attribute of a past event. |
| purchase_hour | order_purchase_timestamp | purchase | Calendar attribute of a past event. |
| handover_dow | order_delivered_carrier_date | carrier handover | Calendar attribute of the prediction moment. |
| handover_month | order_delivered_carrier_date | carrier handover | Calendar attribute of the prediction moment. |
| handover_week_of_year | order_delivered_carrier_date | carrier handover | Seasonality at the prediction moment. |
| n_items | order_items | purchase | Order composition is fixed at purchase. |
| n_distinct_products | order_items | purchase | Order composition is fixed at purchase. |
| n_distinct_sellers | order_items | purchase | Multi-seller orders ship separately; known at purchase. |
| total_price | order_items.price | purchase | Monetary value fixed at purchase. |
| total_freight | order_items.freight_value | purchase | Freight quoted at purchase. |
| max_item_price | order_items.price | purchase | Monetary value fixed at purchase. |
| freight_ratio | order_items | purchase | freight / (price + freight); both fixed at purchase. |
| total_weight_g | products.product_weight_g | purchase | Static product attribute. |
| max_weight_g | products.product_weight_g | purchase | Static product attribute. |
| total_volume_cm3 | products dimension columns | purchase | Static product attribute. |
| max_product_photos | products.product_photos_qty | purchase | Static product attribute. |
| is_cross_state | customers/sellers state | purchase | Both endpoints static at purchase. |
| n_seller_states | sellers.seller_state | purchase | Order composition fixed at purchase. |
| seller_prior_orders | derived: prior eligible orders of the same seller | carrier handover | Counts ONLY orders whose delivery outcome occurred strictly before this order handover, so the statistic was observable at prediction time. |
| seller_prior_late_rate | derived: prior outcomes of the same seller | carrier handover | Smoothed late rate over strictly-earlier known outcomes; the smoothing prior is the TRAIN-split base rate, fitted on train only. |
| route_prior_orders | derived: prior orders on the same seller_state -> customer_state route | carrier handover | Same strict as-of rule as the seller history feature. |
| route_prior_late_rate | derived: prior outcomes on the same route | carrier handover | Smoothed late rate over strictly-earlier known outcomes. |
| customer_state | customers.customer_state | purchase | Coarse geography, static. |
| seller_state | sellers.seller_state | purchase | Coarse geography of primary seller, static. |
| product_category | products.product_category_name | purchase | Static product attribute. |

### Explicitly forbidden as predictors or agent-visible facts

`order_delivered_customer_date`, `is_late`, delivery duration, review scores and
comments, the terminal `order_status` presented as a present-tense status, and
any seller/category aggregate computed from outcomes not yet known.

The two history features earn their place: they read **only** outcomes whose
delivery timestamp is strictly earlier than the order own handover, which an
operator would genuinely have had. The smoothing prior
(0.0602) is the train-split base rate, fitted on train only.

### Missing values

| feature | missing share |
|---|---|
| max_product_photos | 1.38% |
| product_category | 1.38% |
| max_weight_g | 0.02% |
| hours_approval_to_handover | 0.01% |
| hours_purchase_to_approval | 0.01% |

Missing values stay missing. They are handled by the model pipeline (median
imputation fitted on train), never filled from an outcome.

## 8. Data-use plan

Only `orders`, `order_items`, `customers`, `sellers`, `products` and the category
translation are ingested. `order_reviews` is deliberately excluded (a forbidden
post-outcome signal), as are `order_payments` (no as-of value at handover) and
`geolocation` (redundant with the coarse state fields).

Outcome columns are written to a physically separate `order_outcomes` table that
no as-of API path or agent tool queries — a structural guarantee rather than a
convention.
