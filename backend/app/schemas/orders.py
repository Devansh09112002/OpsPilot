"""As-of DTOs.

`OrderAsOf` is the *only* shape in which order data leaves the backend, for the
UI and for agent tools alike. It is built from `order_features` and physically
cannot carry a delivery outcome, because the source table has no such column.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskBand = Literal["low", "medium", "high"]


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
        description="Calibrated estimate that this order misses its promised date."
    )
    ranking_score: float = Field(
        description="Raw model output. The queue's sort key; finer resolution "
                    "than the calibrated probability, not a probability itself."
    )
    risk_band: RiskBand
    model_version: str
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
        description="Calibrated estimate that this order misses its promised date."
    )
    ranking_score: float
    risk_band: RiskBand
    model_version: str
    calibrated: bool = Field(
        default=True,
        description="Whether risk_probability has been calibrated to observed "
                    "frequencies. False means it is a ranking score only.",
    )
    risk_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="What drove this order's score, from exact TreeSHAP.",
    )
    prediction_as_of: datetime = Field(
        description="The moment the prediction refers to: carrier handover."
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
    calibrated: bool = True
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    prediction_as_of: datetime
    computed_at: datetime
    disclaimer: str = (
        "Predicted at carrier handover from information available at that moment. "
        "Not a live re-forecast and not a causal diagnosis: the factors shown are "
        "what moved the model's score, not established causes of delay."
    )
