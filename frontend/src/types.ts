/** Mirrors the Pydantic schemas in backend/app/schemas. Kept in one file so a
 *  contract change is a single obvious diff. See docs/api_contracts.md. */

export type RiskBand = "low" | "medium" | "high";

/** One driver of a single order's score, from exact TreeSHAP. */
export interface RiskFactor {
  feature: string;
  label: string;
  direction: "increases risk" | "decreases risk";
  share: number;
  contribution: number;
}

export type Recommendation = "no_escalation" | "monitor" | "propose_escalation";

export type InvestigationStatus =
  | "running"
  | "completed"
  | "insufficient_evidence"
  | "failed";

export type ProposalStatus = "pending" | "approved" | "rejected";

export interface Snapshot {
  snapshot_id: string;
  label: string;
  snapshot_at: string;
  orders_in_transit: number;
  orders_pre_deadline: number;
  orders_overdue: number;
}

export interface SnapshotStats {
  snapshot_id: string;
  label: string;
  snapshot_at: string;
  orders_pre_deadline: number;
  orders_overdue: number;
  high_risk: number;
  medium_risk: number;
  low_risk: number;
  mean_risk: number;
  model_version: string;
}

export interface OrderListItem {
  order_id: string;
  snapshot_id: string;
  order_estimated_delivery_date: string;
  order_delivered_carrier_date: string;
  days_in_transit: number;
  days_to_deadline: number;
  is_overdue: boolean;
  /** Calibrated estimate that this order misses its promised date. */
  risk_probability: number;
  /** Raw model output; the queue's sort key, not a probability. */
  ranking_score: number;
  risk_band: RiskBand;
  model_version: string;
  customer_state: string | null;
  seller_state: string | null;
  product_category: string | null;
  n_items: number;
  total_price: number;
}

export interface OrderPage {
  items: OrderListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface OrderAsOf {
  order_id: string;
  snapshot_id: string;
  snapshot_at: string;
  order_purchase_timestamp: string;
  order_approved_at: string | null;
  order_delivered_carrier_date: string;
  order_estimated_delivery_date: string;
  days_in_transit: number;
  days_to_deadline: number;
  is_overdue: boolean;
  n_items: number;
  n_distinct_sellers: number;
  total_price: number;
  total_freight: number;
  product_category: string | null;
  customer_state: string | null;
  seller_state: string | null;
  is_cross_state: boolean;
  risk_probability: number;
  ranking_score: number;
  risk_band: RiskBand;
  model_version: string;
  calibrated: boolean;
  risk_factors: RiskFactor[];
  prediction_as_of: string;
}

export interface EvidenceItem {
  evidence_id: string;
  source: string;
  label: string;
  value: string;
  policy_section: string | null;
}

export interface FactItem {
  statement: string;
  evidence_ids: string[];
}

export interface Proposal {
  proposal_id: string;
  investigation_id: string;
  subject_type: "order" | "situation";
  subject_id: string;
  order_id: string | null;
  member_order_ids: string[] | null;
  snapshot_id: string;
  status: ProposalStatus;
  action_type: string;
  reason: string;
  created_at: string;
  decided_at: string | null;
}

export interface Investigation {
  investigation_id: string;
  subject_type: "order" | "situation";
  subject_id: string;
  /** "model" when an LLM wrote the prose, "deterministic" when the backend
   *  assembled it from tool results with no provider call. */
  generated_by: "model" | "deterministic";
  situation: InvestigationSituation | null;
  order_id: string | null;
  snapshot_id: string;
  status: InvestigationStatus;
  prediction_as_of: string | null;
  model_version: string | null;
  risk_probability: number | null;
  summary: string | null;
  facts: FactItem[];
  evidence: EvidenceItem[];
  limitations: string[];
  recommendation: Recommendation | null;
  recommendation_rationale: string | null;
  proposal: Proposal | null;
  ticket_id: string | null;
  error_message: string | null;
  duration_ms: number | null;
  created_at: string;
  completed_at: string | null;
  disclaimer: string;
}

export interface Ticket {
  ticket_id: string;
  proposal_id: string;
  subject_type: "order" | "situation";
  subject_id: string;
  member_order_ids: string[] | null;
  investigation_id: string;
  order_id: string;
  snapshot_id: string;
  status: string;
  action_type: string;
  reason: string;
  risk_probability: number | null;
  model_version: string | null;
  created_at: string;
}

export interface TicketList {
  items: Ticket[];
  total: number;
  disclaimer: string;
}

export interface DecisionResponse {
  proposal: Proposal;
  ticket: Ticket | null;
  already_decided: boolean;
}

export interface AuditEvent {
  event_type: string;
  subject_type: string;
  subject_id: string;
  created_at: string;
  detail: Record<string, unknown> | null;
}

export interface Meta {
  model_version: string | null;
  model_family: string | null;
  model_available: boolean;
  calibrated: boolean;
  calibration_method: string | null;
  band_thresholds: { high: number; medium: number };
  training_cutoff: string | null;
  policy_version: string;
  llm_configured: boolean;
  dataset: { name: string; url: string; license: string };
}


export interface InvestigationSituation {
  situation_id: string;
  lane: string;
  n_flagged: number;
  n_escalatable: number;
  expected_late: number;
}

export interface SituationSummary {
  situation_id: string;
  snapshot_id: string;
  seller_state: string;
  customer_state: string;
  lane: string;
  n_flagged: number;
  n_high: number;
  n_lane_total: number;
  share_of_lane: number;
  /** Sum of member calibrated probabilities: how many of these orders the
   *  model expects to arrive late. Only meaningful because the scores are
   *  calibrated. */
  expected_late: number;
  mean_risk: number;
  max_risk: number;
  n_escalatable: number;
  model_version: string | null;
}

export interface SituationMember {
  order_id: string;
  risk_probability: number;
  risk_band: RiskBand;
  product_category: string | null;
  days_to_deadline: number;
  escalatable: boolean;
}

export interface LaneHistory {
  available: boolean;
  label: string;
  sample_size: number;
  late_rate: number | null;
  baseline_rate: number | null;
  baseline_sample: number;
  caveats: string[];
}

export interface SituationDetail extends SituationSummary {
  members: SituationMember[];
  lane_history: LaneHistory;
}
