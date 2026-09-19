/** Mirrors the Pydantic schemas in backend/app/schemas. Kept in one file so a
 *  contract change is a single obvious diff. See docs/api_contracts.md. */

export type RiskBand = "low" | "medium" | "high";

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
  risk_probability: number;
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
  risk_band: RiskBand;
  model_version: string;
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
  order_id: string;
  snapshot_id: string;
  status: ProposalStatus;
  action_type: string;
  reason: string;
  created_at: string;
  decided_at: string | null;
}

export interface Investigation {
  investigation_id: string;
  order_id: string;
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
  training_cutoff: string | null;
  policy_version: string;
  llm_configured: boolean;
  dataset: { name: string; url: string; license: string };
}
