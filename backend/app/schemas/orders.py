"""As-of DTOs.

`OrderAsOf` is the *only* shape in which order data leaves the backend, for the
UI and for agent tools alike. It is built from `order_features` and physically
cannot carry a delivery outcome, because the source table has no such column.

Scores are made **on the snapshot day**, from what was known at its start
(the served v4 model; see `docs/research_v4.md`). The priority rank is the
validated quantity; the probability is an estimate that moves with network
conditions; the arrival dates are a forecast and can be wrong.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskBand = Literal["low", "medium", "high"]
ArrivalTag = Literal["likely_late", "tight", "on_track", "overdue", "unknown"]


class RiskFactor(BaseModel):
    """One driver of a single order's score, from exact TreeSHAP."""

    feature: str
    label: str
    direction: Literal["increases risk", "decreases risk"]
    share: float = Field(description="Share of this order's total attribution.")
    contribution: float = Field(description="Signed log-odds contribution.")


class SnapshotSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    snapshot_id: str
    label: str
    snapshot_at: datetime
    orders_in_transit: int
    orders_pre_deadline: int
    orders_overdue: int


class SnapshotStats(BaseModel):
    """Risk distribution for the queue header. Every number comes from SQL."""

    snapshot_id: str
    label: str
    snapshot_at: datetime
    orders_pre_deadline: int
    orders_overdue: int
    high_risk: int
    medium_risk: int
    low_risk: int
    mean_risk: float
    model_version: str


class OrderListItem(BaseModel):
    """One row of the risk queue."""

    model_config = ConfigDict(from_attributes=True)

    order_id: str
    snapshot_id: str
    order_estimated_delivery_date: datetime
    order_delivered_carrier_date: datetime
    days_in_transit: float
    days_to_deadline: float
    is_overdue: bool
    risk_probability: float = Field(
        description="Model estimate of the chance this order misses its promised date. "
                    "Moves with network conditions and runs high in calm periods."
    )
    ranking_score: float = Field(
        description="The queue's sort key: the average within-day percentile rank of "
                    "the three model components. Not a probability."
    )
    risk_band: RiskBand
    model_version: str
    priority_rank: int | None = Field(
        default=None,
        description="Position in the day's queue, 1 = review first. Null when overdue.",
    )
    expected_arrival: date | None = Field(
        default=None, description="Forecast arrival day: half of similar parcels arrive by it.")
    arrival_earliest: date | None = Field(
        default=None, description="Day by which 10% of similar parcels arrive.")
    arrival_latest: date | None = Field(
        default=None, description="Day by which 90% of similar parcels arrive.")
    buffer_days: int | None = Field(
        default=None,
        description="Promised day minus expected arrival. Negative means likely late.",
    )
    arrival_tag: ArrivalTag = "unknown"
    customer_state: str | None = None
    seller_state: str | None = None
    product_category: str | None = None
    n_items: int
    total_price: float


class OrderPage(BaseModel):
    items: list[OrderListItem]
    total: int
    limit: int
    offset: int


class OrderAsOf(BaseModel):
    """Full as-of detail for one order at one snapshot.

    Contains no `order_delivered_customer_date`, no `is_late` and no terminal
    `order_status`. This is enforced by the table schema, not by remembering to
    exclude columns.
    """

    order_id: str
    snapshot_id: str
    snapshot_at: datetime

    order_purchase_timestamp: datetime
    order_approved_at: datetime | None
    order_delivered_carrier_date: datetime
    order_estimated_delivery_date: datetime

    days_in_transit: float
    days_to_deadline: float
    is_overdue: bool

    n_items: int
    n_distinct_sellers: int
    total_price: float
    total_freight: float
    product_category: str | None
    customer_state: str | None
    seller_state: str | None
    is_cross_state: bool

    risk_probability: float = Field(
        description="Model estimate of the chance this order misses its promised date."
    )
    ranking_score: float
    risk_band: RiskBand
    model_version: str
    priority_rank: int | None = Field(
        default=None,
        description="Position in the day's queue, 1 = review first. Null when overdue.",
    )
    expected_arrival: date | None = Field(
        default=None, description="Forecast arrival day: half of similar parcels arrive by it.")
    arrival_earliest: date | None = Field(
        default=None, description="Day by which 10% of similar parcels arrive.")
    arrival_latest: date | None = Field(
        default=None, description="Day by which 90% of similar parcels arrive.")
    buffer_days: int | None = Field(
        default=None,
        description="Promised day minus expected arrival. Negative means likely late.",
    )
    arrival_tag: ArrivalTag = "unknown"
    cohort_size: int | None = Field(
        default=None, description="Orders ranked in this snapshot's queue.")
    calibrated: bool = Field(
        default=False,
        description="False: the probability is a model estimate, not calibrated "
                    "across periods. The priority rank is the validated quantity.",
    )
    risk_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="What moved this order's ranking-model score, from exact TreeSHAP.",
    )
    prediction_as_of: datetime = Field(
        description="The moment the score refers to: the start of the snapshot day."
    )


class PredictionRequest(BaseModel):
    order_id: str = Field(min_length=1, max_length=64)
    snapshot_id: str = Field(min_length=1, max_length=16)


class PredictionResponse(BaseModel):
    order_id: str
    snapshot_id: str
    risk_probability: float
    ranking_score: float
    risk_band: RiskBand
    model_version: str
    priority_rank: int | None = Field(
        default=None,
        description="Position in the day's queue, 1 = review first. Null when overdue.",
    )
    expected_arrival: date | None = Field(
        default=None, description="Forecast arrival day: half of similar parcels arrive by it.")
    arrival_earliest: date | None = Field(
        default=None, description="Day by which 10% of similar parcels arrive.")
    arrival_latest: date | None = Field(
        default=None, description="Day by which 90% of similar parcels arrive.")
    buffer_days: int | None = Field(
        default=None,
        description="Promised day minus expected arrival. Negative means likely late.",
    )
    arrival_tag: ArrivalTag = "unknown"
    cohort_size: int | None = None
    calibrated: bool = False
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    prediction_as_of: datetime
    computed_at: datetime
    disclaimer: str = (
        "Scored on the snapshot day from what was known at its start, recomputed "
        "live from the stored point-in-time features. The arrival dates are a "
        "forecast and can be wrong. Not a causal diagnosis: the factors shown are "
        "what moved the ranking model's score, not established causes of delay."
    )
