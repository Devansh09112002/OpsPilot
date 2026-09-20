"""Response contracts for lane situations.

No field here is outcome-derived except `lane_history`, which comes from
`analytics.lane_context` and is therefore already bounded by the as-of cutoff.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SituationMemberOut(BaseModel):
    order_id: str
    risk_probability: float = Field(description="Calibrated probability of a late delivery.")
    risk_band: str
    product_category: str | None = None
    days_to_deadline: float
    escalatable: bool = Field(
        description="Whether this order independently qualifies under ESC-01."
    )


class SituationSummary(BaseModel):
    situation_id: str
    snapshot_id: str
    seller_state: str
    customer_state: str
    lane: str
    n_flagged: int = Field(description="Flagged, not-overdue orders on this lane.")
    n_high: int
    n_lane_total: int = Field(description="All orders on this lane in the snapshot.")
    share_of_lane: float
    expected_late: float = Field(
        description=(
            "Sum of the member orders' calibrated risk estimates. Summable "
            "only because the scores are calibrated, but measured against "
            "held-out snapshots it overstates the number actually late by "
            "roughly half again (docs/snapshot_calibration.md). Intended for "
            "comparing lanes, not as a forecast of a count."
        )
    )
    mean_risk: float
    max_risk: float
    n_escalatable: int = Field(
        0, description='Members that independently qualify under ESC-01.'
    )
    model_version: str | None = None


class LaneHistoryOut(BaseModel):
    available: bool
    label: str
    sample_size: int
    late_rate: float | None = None
    baseline_rate: float | None = None
    baseline_sample: int = 0
    caveats: list[str] = Field(default_factory=list)


class SituationDetail(SituationSummary):
    members: list[SituationMemberOut] = Field(default_factory=list)
    lane_history: LaneHistoryOut
