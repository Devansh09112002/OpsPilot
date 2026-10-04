"""Render docs/research_v3.md from the walk-forward and test results.

    python -m ml_pipeline.render_v3

Every number comes from `docs/research_v3_development.json`,
`docs/research_v3_test.json` and their per-snapshot CSVs. Nothing is typed in.
"""

from __future__ import annotations

import json

import pandas as pd

from data_pipeline import spec

DOCS = spec.DOCS_DIR

LABELS = {
    "v2_published": "v2 as deployed (handover XGBoost)",
    "rule_handover": "Deadline rule at handover",
    "lr_handover": "Logistic regression at handover",
    "xgb_handover": "XGBoost at handover, retrained",
    "rule_window": "Share of window used (no training)",
    "rule_km": "**Kaplan-Meier survival (no training)**",
    "lr_snapshot": "Logistic regression, snapshot day",
    "xgb_snapshot": "XGBoost, snapshot day",
    "xgb_snapshot_mono": "XGBoost + monotone constraints",
    "xgb_snapshot_deep": "XGBoost, deeper",
    "xgb_snapshot_shallow": "XGBoost, shallower",
    "rank_snapshot": "XGBoost ranker (rank:pairwise)",
}


HANDOVER = {"rule_handover", "lr_handover", "xgb_handover", "v2_published"}


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render() -> str:
    dev = json.loads((DOCS / "research_v3_development.json").read_text(encoding="utf-8"))
    test = json.loads((DOCS / "research_v3_test.json").read_text(encoding="utf-8"))
    test_snaps = pd.read_csv(DOCS / "research_v3_test_snapshots.csv")

    sel = test["selected"]
    p_test = test["mean_precision_at_50"]
    r_test = test["mean_recall_at_50"]
    base = test["base_rate"]
    v2 = p_test["v2_published"]
    comp = test["comparisons"]
    cal = test["calibration"] or {}
    folds = dev["folds"]

    # --- development table -------------------------------------------------
    order = ["rule_handover", "lr_handover", "xgb_handover", "rule_window", "rule_km",
             "lr_snapshot", "xgb_snapshot", "xgb_snapshot_mono", "xgb_snapshot_deep",
             "xgb_snapshot_shallow", "rank_snapshot"]
    dev_rows = []
    for n in order:
        s = dev["summary"].get(n)
        if not s:
            continue
        dev_rows.append([LABELS[n], *[f"{s['fold_means'][f['name']]:.3f}" for f in folds],
                         f"**{s['score']:.3f}**"])
    dev_tbl = _table(dev_rows, ["model", *[f"{f['name']} {f['regime']}" for f in folds],
                                "mean"])

    # --- test table ---------------------------------------------------------
    test_rows = []
    for n in ["v2_published", *order]:
        if n not in p_test:
            continue
        test_rows.append([LABELS[n], f"{p_test[n]:.3f}", f"{p_test[n] / base:.1f}x",
                          f"{r_test[n]:.3f}", f"~{p_test[n] * 50:.0f} of 50"])
    test_tbl = _table(test_rows, ["model", "Precision@50", "lift", "Recall@50",
                                  "late orders in the top 50"])

    comp_rows = [[LABELS.get(k, k), f"{c['mean_diff']:+.3f}",
                  f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]",
                  f"{c['wins']} of {c['n']}"] for k, c in comp.items()]
    comp_tbl = _table(comp_rows, [f"{LABELS[sel].strip('*')} versus", "mean difference",
                                  "95% bootstrap interval", "days won"])

    snap_tbl = _table(
        [[r.snapshot_at, f"{r.n:,}", str(r.late), f"{r.v2_published:.2f}",
          f"{r.xgb_handover:.2f}", f"{getattr(r, sel):.2f}"]
         for r in test_snaps.itertuples()],
        ["snapshot", "ranked orders", "late", "v2 deployed", "v2 retrained", "selected"],
    )

    better_on_test = sorted(
        (n for n in p_test if p_test[n] > p_test[sel] and n != sel),
        key=lambda n: -p_test[n],
    )
    better_txt = ", ".join(f"{LABELS[n].strip('*')} ({p_test[n]:.3f})" for n in better_on_test)
    window_c = comp.get("rule_window", {})

    targets = test["targets"]
    t_rows = [
        ["mean test Precision@50 >= 0.20", f"{p_test[sel]:.3f}",
         "met" if targets["precision_at_50_at_least_0_20"] else "**missed**"],
        ["beats v2 retrained on the same data, interval above zero",
         f"[{comp['xgb_handover']['ci95'][0]:+.3f}, {comp['xgb_handover']['ci95'][1]:+.3f}]",
         "met" if targets["beats_retrained_v2_ci_excludes_zero"] else "**missed**"],
        ["beats the deployed v2 model", f"{comp['v2_published']['mean_diff']:+.3f}",
         "met" if targets["beats_published_v2"] else "**missed**"],
        ["Brier skill > 0 against the pre-test late rate",
         f"{cal.get('brier_skill_vs_pre_test_rate', float('nan')):+.3f}",
         "met" if targets["brier_skill_positive"] else "**missed**"],
    ]
    targets_tbl = _table(t_rows, ["pre-registered target", "result", "status"])

    return f"""# OpsPilot v3 - Scoring on the snapshot day

Every number here is produced by `python -m ml_pipeline.walkforward` and
rendered by `python -m ml_pipeline.render_v3`. The protocol, candidate list,
selection rule and test targets were fixed in
[`preregistration_v3.md`](preregistration_v3.md) before any v3 result existed.
The test ran once; its record carries the SHA-256 of the pre-registration it
was run against (`{test['preregistration_sha256'][:16]}...`).

## 1. Headline

On the held-out test period (11 weekly snapshots, June-August 2018, the same
protocol as every published v2 figure), the selected method puts
**{p_test[sel]:.1%}** genuinely late orders in its top 50, against **{v2:.1%}**
for the deployed v2 model and a **{base:.1%}** base rate:

- **{p_test[sel] / v2:.1f}x** the deployed model's precision; lift over random
  rises from {v2 / base:.1f}x to **{p_test[sel] / base:.1f}x**.
- It catches **{r_test[sel]:.0%}** of all late orders in the cohort with 50
  reviews, against {r_test['v2_published']:.0%}.
- It won **{comp['v2_published']['wins']} of {comp['v2_published']['n']}** test days
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

{dev_tbl}

**Selection, by the pre-registered rule.**
{chr(10).join('- ' + d for d in dev['decision'])}

The no-training survival rule was not beaten by any learned model by more
than one standard error, so it is what the rule serves. Over all
{dev['paired'][0]['n']} development snapshots it beat the v2 recipe by
{dev['paired'][0]['mean_diff']:+.3f} (standard error {dev['paired'][0]['se']:.3f}),
winning {dev['paired'][0]['wins']} of {dev['paired'][0]['n']}.

## 4. Test, run once

{test_tbl}

{comp_tbl}

{targets_tbl}

Per snapshot (Precision@50):

{snap_tbl}

## 5. What a careful reader should know

- **Other snapshot-day methods did as well or better on test**: {better_txt or 'none'}.
  The gap between the share-of-window rule and the selected method is
  {window_c.get('mean_diff', float('nan')):+.3f} with a 95% interval of
  [{window_c.get('ci95', [float('nan')] * 2)[0]:+.3f}, {window_c.get('ci95', [float('nan')] * 2)[1]:+.3f}],
  which includes zero. The selection was made on development and is **not**
  revisited: switching to whichever scored best on test would make the test
  figure meaningless. The robust finding is that *scoring on the snapshot
  day* works: every one of the seven snapshot-day methods scored between
  {min(p_test[n] for n in order if n not in HANDOVER):.3f} and
  {max(p_test[n] for n in order if n not in HANDOVER):.3f}, against
  {p_test['xgb_handover']:.3f} for the best handover-time model. Which
  snapshot-day method is best is not resolved by 11 test days.
- **Retraining alone helps v2 a lot.** The v2 recipe retrained on data up to
  June 2018 scores {p_test['xgb_handover']:.3f} against {v2:.3f} as deployed.
  The comparison that isolates the new information is therefore against the
  retrained model: {comp['xgb_handover']['mean_diff']:+.3f}, interval
  [{comp['xgb_handover']['ci95'][0]:+.3f}, {comp['xgb_handover']['ci95'][1]:+.3f}].
- **Calibration is better but still overstates in a calm period.** Brier
  {cal.get('brier_model', float('nan')):.4f} against {cal.get('brier_constant_pre_test_rate', float('nan')):.4f}
  for a constant at the pre-test rate (skill {cal.get('brier_skill_vs_pre_test_rate', float('nan')):+.3f}),
  but slightly worse than a constant at the *hindsight* test rate (skill
  {cal.get('brier_skill_vs_hindsight_rate', float('nan')):+.3f}). Mean predicted
  {cal.get('mean_predicted', float('nan')):.3f} against {cal.get('observed_rate', float('nan')):.3f} observed;
  in the top decile {cal.get('top_decile_mean_predicted', float('nan')):.2f} against
  {cal.get('top_decile_observed', float('nan')):.2f}. The 120-day curves behind the
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
"""


def main() -> int:
    (DOCS / "research_v3.md").write_text(render(), encoding="utf-8")
    print("wrote docs/research_v3.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
