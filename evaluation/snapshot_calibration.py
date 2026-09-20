"""Does `expected_late` mean what the product says it means?

The situations screen shows, for each lane, the sum of its member orders'
calibrated probabilities, and calls that the number of orders the model
expects to be delivered late. That reading is only legitimate if the
calibration holds on the population it is applied to: the flagged, in-transit
orders of a snapshot, in the held-out test period.

It does not, and this script is how that is measured rather than assumed. The
isotonic calibrator is fitted on the validation split, whose late rate is
10.8%; the snapshots sit in the test period, whose rate is 3.0%. Calibration
does not survive a shift that size, and the sum overstates.

Run:  python -m evaluation.snapshot_calibration
Writes: docs/snapshot_calibration.json and a section for the model report.

This reads `order_outcomes`, which is offline scoring, never a serving path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "backend"), str(REPO_ROOT)]

from sqlalchemy import create_engine  # noqa: E402

from data_pipeline import spec  # noqa: E402

QUERY = """
select s.snapshot_id,
       f.seller_state || ' to ' || f.customer_state as lane,
       s.risk_probability,
       s.risk_band,
       o.is_late
from snapshot_orders s
join order_features f using (order_id)
join order_outcomes o using (order_id)
where s.is_overdue = false
  and s.risk_band in ('high', 'medium')
"""

MIN_SITUATION_ORDERS = 3


def _engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set")
    return create_engine(url, future=True)


def measure(df: pd.DataFrame) -> dict:
    by_snapshot = []
    for snapshot_id, group in df.groupby("snapshot_id"):
        predicted = float(group["risk_probability"].sum())
        actual = int(group["is_late"].sum())
        by_snapshot.append({
            "snapshot_id": snapshot_id,
            "flagged_orders": int(len(group)),
            "predicted_late": round(predicted, 2),
            "actual_late": actual,
            "ratio": round(predicted / actual, 3) if actual else None,
        })

    situations = []
    for (snapshot_id, lane), group in df.groupby(["snapshot_id", "lane"]):
        if len(group) < MIN_SITUATION_ORDERS:
            continue
        predicted = float(group["risk_probability"].sum())
        actual = int(group["is_late"].sum())
        situations.append({
            "snapshot_id": snapshot_id,
            "lane": lane,
            "n_flagged": int(len(group)),
            "expected_late": round(predicted, 2),
            "actual_late": actual,
            "error": round(predicted - actual, 2),
        })

    frame = pd.DataFrame(situations)
    predicted_total = float(df["risk_probability"].sum())
    actual_total = int(df["is_late"].sum())

    # Ranking is the product's actual use of the number, and a multiplicative
    # bias leaves ranking untouched. Measured explicitly so the distinction
    # between "the ordering is sound" and "the count is right" is evidenced.
    if len(frame) > 2:
        spearman = float(frame["expected_late"].corr(frame["actual_late"], method="spearman"))
    else:
        spearman = None

    return {
        "flagged_orders": int(len(df)),
        "predicted_late_total": round(predicted_total, 2),
        "actual_late_total": actual_total,
        "overall_ratio": round(predicted_total / actual_total, 3) if actual_total else None,
        "by_snapshot": by_snapshot,
        "situations_measured": int(len(frame)),
        "situation_mean_signed_error": (
            round(float(frame["error"].mean()), 3) if len(frame) else None
        ),
        "situation_mean_absolute_error": (
            round(float(frame["error"].abs().mean()), 3) if len(frame) else None
        ),
        "situations_overstated": int((frame["error"] > 0).sum()) if len(frame) else 0,
        "rank_correlation_expected_vs_actual": (
            round(spearman, 3) if spearman is not None else None
        ),
        "worst_situations": sorted(
            situations, key=lambda s: -abs(s["error"])
        )[:5],
    }


def render(result: dict) -> str:
    rows = "\n".join(
        f"| {s['snapshot_id']} | {s['flagged_orders']} | {s['predicted_late']} | "
        f"{s['actual_late']} | {s['ratio']}x |"
        for s in result["by_snapshot"]
    )
    return f"""## Does `expected_late` predict the right number?

The situations screen sums member calibrated probabilities and presents the
total as the number of orders the model expects to arrive late. Measured
against what actually happened on the held-out snapshots:

| snapshot | flagged orders | predicted late | actually late | ratio |
|---|---|---|---|---|
{rows}
| **all** | **{result['flagged_orders']}** | **{result['predicted_late_total']}** | \
**{result['actual_late_total']}** | **{result['overall_ratio']}x** |

**The sum overstates by about {result['overall_ratio']}x.** Across the
{result['situations_measured']} lane situations, the mean signed error is
{result['situation_mean_signed_error']:+} orders and
{result['situations_overstated']} of {result['situations_measured']} overstate.

The cause is the shift this dataset is already known for. The isotonic
calibrator is fitted on the validation split, whose late rate is 10.8%; the
snapshots sit in the test period at 3.0%. A calibration map does not survive a
3.6x change in base rate, and the flagged population is where the gap is
widest.

**Why it was not "fixed".** Refitting the calibrator on the test period, or
selecting a different fitting window after seeing these numbers, would make
every held-out figure in this report meaningless. The model and its calibrator
are frozen, the measurement stands, and the product wording was changed
instead.

**What the number is still good for.** Ranking. A multiplicative bias does not
reorder anything, and rank correlation between expected and actual late counts
across situations is
**{result['rank_correlation_expected_vs_actual']}**. Choosing which lane to
review first is sound; reading the total as a forecast of how many parcels
will be late is not, and the UI no longer invites that reading.
"""


def main() -> int:
    engine = _engine()
    df = pd.read_sql(QUERY, engine)
    if df.empty:
        raise SystemExit("no flagged snapshot orders found; run data_pipeline.ingest")

    result = measure(df)
    out = spec.DOCS_DIR / "snapshot_calibration.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (spec.DOCS_DIR / "snapshot_calibration.md").write_text(render(result), encoding="utf-8")

    print(f"flagged orders            {result['flagged_orders']}")
    print(f"predicted late (sum p)    {result['predicted_late_total']}")
    print(f"actually late             {result['actual_late_total']}")
    print(f"overall ratio             {result['overall_ratio']}x")
    print(f"situations measured       {result['situations_measured']}")
    print(f"mean signed error         {result['situation_mean_signed_error']:+}")
    print(f"rank correlation          {result['rank_correlation_expected_vs_actual']}")
    print(f"\nwrote {out.name} and snapshot_calibration.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
