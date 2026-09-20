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
    days_in_transit: float | None = None
    product_category: str | None = None


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
            "Sum of member calibrated probabilities: how many of these orders "
            "the model expects to be delivered late. Meaningful only because "
            "the scores are calibrated."
        )
    )
    mean_risk: float
    max_risk: float
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
