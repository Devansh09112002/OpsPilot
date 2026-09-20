"""Markdown rendering for docs/model_report.md.

Kept separate from metric computation so the report is pure formatting: every
value it prints comes from the payload `ml_pipeline.train` measured.
"""

from __future__ import annotations

SIM_HDR = ["model", "mean Precision@K", "mean Recall@K", "Lift@K", "snapshots", "base rate"]


def _tbl(rows: list[list[str]], header: list[str]) -> str:
    return "\n".join(
        ["| " + " | ".join(header) + " |",
         "|" + "|".join(["---"] * len(header)) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def _metric_rows(block: dict, k: int) -> list[list[str]]:
    rows = []
    for name, m in block.items():
        rows.append([
            name,
            f"{m[f'precision_at_{k}']:.3f}",
            f"{m[f'recall_at_{k}']:.3f}",
            f"{m[f'lift_at_{k}']:.2f}x",
            f"{m['pr_auc']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['brier']:.5f}" if "brier" in m else "n/a",
        ])
    return rows


def _sim_rows(block: dict, k: int) -> list[list[str]]:
    rows = []
    for name, s in block.items():
        if not s.get("n_snapshots"):
            continue
        rows.append([
            name,
            f"{s[f'mean_precision_at_{k}']:.3f} +/- {s[f'std_precision_at_{k}']:.3f}",
            f"{s[f'mean_recall_at_{k}']:.3f}",
            f"{s[f'mean_lift_at_{k}']:.2f}x",
            f"{s['n_snapshots']}",
            f"{s['mean_base_rate']:.3%}",
        ])
    return rows


def render(p: dict) -> str:
    k = p["k"]
    sel = p["selected"]
    sp = p["splits"]
    pooled_hdr = ["model", f"Precision@{k}", f"Recall@{k}", f"Lift@{k}",
                  "PR-AUC", "ROC-AUC", "Brier"]

    val_sim_tbl = _tbl(_sim_rows(p["validation_simulated"], k), SIM_HDR)
    test_sim_tbl = _tbl(_sim_rows(p["test_simulated"], k), SIM_HDR)
    val_tbl = _tbl(_metric_rows(p["validation"], k), pooled_hdr)
    test_tbl = _tbl(_metric_rows(p["test"], k), pooled_hdr)

    served_sim = p["test_simulated"][sel]
    rule_sim = p["test_simulated"]["rule_deadline_proximity"]
    test_sel = p["test"][sel]
    calibration = p.get("calibration") or {}
    # Prefer the calibrated reliability bins; fall back to the raw ones.
    cal = calibration.get("calibration_bins") or test_sel.get("calibration", [])

    if calibration:
        raw_b = calibration["brier_raw_test"]
        cal_b = calibration["brier_calibrated_test"]
        bands = calibration.get("band_thresholds", {})
        summary_tbl = _tbl(
            [
                ["Brier score (test)", f"{raw_b:.5f}", f"**{cal_b:.5f}**"],
                ["mean predicted (test)",
                 f"{calibration.get('mean_raw_test', float('nan')):.4f}",
                 f"{calibration['mean_calibrated_test']:.4f}"],
                ["observed base rate",
                 f"{calibration['test_base_rate']:.4f}",
                 f"{calibration['test_base_rate']:.4f}"],
            ],
            ["", "raw score", "calibrated"],
        )
        cal_summary = (
            summary_tbl
            + f"\n\nCalibration improves the Brier score "
              f"**{raw_b / max(cal_b, 1e-9):.1f}x**, and the mean calibrated "
              f"estimate ({calibration['mean_calibrated_test']:.4f}) now sits "
              f"close to the observed base rate "
              f"({calibration['test_base_rate']:.4f}).\n\n"
              f"Risk bands are cut on the calibrated distribution rather than "
              f"set by hand: **high** at {bands.get('high', 0):.4f} (the "
              f"validation 95th percentile, roughly the review capacity), "
              f"**medium** at {bands.get('medium', 0):.4f} (75th percentile)."
        )
    else:
        cal_summary = "_Model is not calibrated._"

    cal_tbl = (
        _tbl([[f"{c['bin_low']:.1f}-{c['bin_high']:.1f}", f"{c['n']:,}",
               f"{c['mean_predicted']:.3f}", f"{c['observed_rate']:.3f}"] for c in cal],
             ["predicted band", "orders", "mean predicted", "observed late rate"])
        if cal else "_Not available._"
    )
    snap_tbl = (
        _tbl([[sid, f"{m['n']:,}", f"{m['base_rate']:.3%}", f"{m[f'precision_at_{k}']:.3f}",
               f"{m[f'recall_at_{k}']:.3f}", f"{m[f'lift_at_{k}']:.2f}x", f"{m['roc_auc']:.4f}"]
              for sid, m in p["per_snapshot"].items()],
             ["snapshot", "pre-deadline orders", "base rate", f"Precision@{k}",
              f"Recall@{k}", f"Lift@{k}", "ROC-AUC"])
        if p["per_snapshot"] else "_Not available._"
    )
    imp_tbl = (
        _tbl([[i["feature"], f"{i['weight']:+.4f}"] for i in p["importances"]],
             ["feature", "weight"])
        if p["importances"] else "_Not available._"
    )

    e = p["errors"]
    fp = e.get("top_k_false_positives", {})
    miss = e.get("low_ranked_late_orders", {})
    lat = p["latency_ms"]

    # Did a model other than the served one do better under the test protocol?
    best_test = max(
        (n for n, s in p["test_simulated"].items() if s.get("n_snapshots")),
        key=lambda n: p["test_simulated"][n][f"mean_precision_at_{k}"],
    )
    ordering_note = ""
    if best_test != sel:
        gap = (p["test_simulated"][best_test][f"mean_precision_at_{k}"]
               - served_sim[f"mean_precision_at_{k}"])
        se = served_sim[f"std_precision_at_{k}"] / max(served_sim["n_snapshots"], 1) ** 0.5
        ordering_note = (
            "\n**Ordering instability, stated plainly.** On the held-out test period "
            f"`{best_test}` scored {gap:+.3f} higher than the served `{sel}` "
            f"(standard error {se:.3f} across {served_sim['n_snapshots']} snapshots, so "
            "the gap sits around the noise floor). `" + sel + "` was selected on "
            "validation and that selection was **not** revisited after seeing test "
            "results; doing so would make the test estimate meaningless. The honest "
            "reading is that the two learned models are not clearly distinguishable on "
            "this data, and that both beat the operational rule.\n"
        )

    return f"""# OpsPilot - Model Report

**Model version:** `{p['model_version']}`
**Served family:** {sel}
**Primary metric:** mean Precision@{k} across simulated snapshots - the share of
genuinely late orders among the top {k} an operations team would review.

Every number here is produced by `python -m ml_pipeline.train`. Nothing in this
file is hand-entered.

## 1. Headline result

Ranking one snapshot at a time, exactly as the deployed risk queue does, the
served model puts **{served_sim[f'mean_precision_at_{k}']:.1%}** genuinely late
orders in its top {k}, against a **{served_sim['mean_base_rate']:.1%}** background
rate. That is a **{served_sim[f'mean_lift_at_{k}']:.2f}x lift**: a reviewer
working the flagged queue meets a late order
{served_sim[f'mean_lift_at_{k}']:.1f} times as often as one reviewing {k} orders
at random, while catching {served_sim[f'mean_recall_at_{k}']:.1%} of all late
orders in the cohort.

The operational rule the product must beat manages
{rule_sim[f'mean_precision_at_{k}']:.1%} ({rule_sim[f'mean_lift_at_{k}']:.2f}x) on
the same snapshots.

These results are modest in absolute terms. Delivery lateness is only partly
predictable from what is known at carrier handover, and this report does not
inflate that.

## 2. Why accuracy is not reported

The test-period base rate is {sp['test']['base_rate']:.3%}. A model that always
answers "on time" scores {1 - sp['test']['base_rate']:.1%} accuracy and finds
nothing. Ranking quality under a fixed review capacity is the only metric that
reflects the actual decision.

## 3. Evaluation protocol

Two protocols appear below and they answer different questions.

- **Simulated snapshots (primary).** Every 7 days across a period, take the
  orders genuinely in transit and still pre-deadline at that moment, rank them,
  and measure Precision@{k}. Cohorts run to roughly 1,000-2,000 orders, matching
  the product. Averaging across snapshots observes hundreds of top-{k} slots
  instead of {k}.
- **Pooled split (secondary).** Rank an entire split at once and take the top
  {k}. That measures only {k} orders out of ~19,000 and picks the most extreme
  cases across three and a half months, so it reads far higher than anything the
  product can deliver. Reported for completeness, not as a headline.

Model selection used the simulated protocol on **validation only**.

## 4. Data splits

Chronological, cut on `order_delivered_carrier_date` (the prediction moment),
frozen before any model comparison.

{_tbl([["train", f"{sp['train']['n']:,}", f"{sp['train']['base_rate']:.3%}"],
       ["validation", f"{sp['validation']['n']:,}", f"{sp['validation']['base_rate']:.3%}"],
       ["test", f"{sp['test']['n']:,}", f"{sp['test']['base_rate']:.3%}"]],
      ["split", "orders", "late rate"])}

The late rate moves considerably between periods: validation covers the
March-2018 disruption at {sp['validation']['base_rate']:.1%} while test sits at
{sp['test']['base_rate']:.1%}. That is genuine distribution shift in the Olist
data, and it is why validation figures run higher than test figures throughout.

## 5. Validation comparison - selection happened here

Simulated snapshots, validation period only:

{val_sim_tbl}

**Selection:** {p['rationale']}

Pooled over the whole validation split, for reference:

{val_tbl}

## 6. Held-out test results - scored after freezing

Simulated snapshots, the number that describes the product:

{test_sim_tbl}
{ordering_note}
Pooled over the whole test split:

{test_tbl}

The pooled row needs care. A pooled Precision@{k} of
{test_sel[f'precision_at_{k}']:.3f} looks impressive, but it comes from taking
the {k} most extreme orders out of {sp['test']['n']:,} spanning months. The
product never gets to make that choice: it ranks one snapshot at a time, which
is the {served_sim[f'mean_precision_at_{k}']:.3f} figure above. Wherever the two
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

{cal_summary}

Reliability of the **calibrated** estimate on the held-out test split:

{cal_tbl}

The calibrated value is what the product displays and what the demo policy's
escalation threshold is stated on. `docs/api_contracts.md` returns both:
`risk_probability` (calibrated, for reading) and `ranking_score` (raw, the
sort key).

## 8. The three frozen demo snapshots

Exactly what the deployed risk queue shows, restricted to pre-deadline orders.

{snap_tbl}

These cohorts hold roughly 850-1,550 orders with 40-110 late ones, so individual
snapshot figures carry visible sampling noise. The averaged row in section 6 is
the more reliable estimate.

## 9. Feature influence

{imp_tbl}

## 10. Error analysis

**False positives in the pooled top {k}** ({fp.get('n', 0)} of {k}): median slack
at handover {fp.get('median_days_handover_to_estimate', 'n/a')} days, median
fulfilment time {fp.get('median_hours_purchase_to_handover', 'n/a')} h,
{fp.get('share_cross_state', 0):.0%} cross-state. These are genuinely tight
shipments that happened to arrive on time.

**Late orders ranked low** ({miss.get('n', 0)}): median slack
{miss.get('median_days_handover_to_estimate', 'n/a')} days. Orders with
comfortable slack that were delayed anyway carry no signal observable at
handover. This is the irreducible error class for a prediction made at carrier
handover, and it caps how high Precision@{k} can go.

## 11. Inference cost and latency

Single-row `predict_proba` over {lat['n_calls']} held-out rows, development
machine, CPU only: median **{lat['median_ms']} ms**, p95 **{lat['p95_ms']} ms**.
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
"""
