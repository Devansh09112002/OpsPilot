/** Typed API client.
 *
 *  Every request sends credentials so the server-issued guest session cookie
 *  travels with it. The backend error envelope is unwrapped into `ApiError`,
 *  which carries a message worth showing a user rather than a status code.
 */

import type {
  AuditEvent,
  DecisionResponse,
  Investigation,
  Meta,
  OrderAsOf,
  OrderPage,
  RiskBand,
  Snapshot,
  SnapshotStats,
  TicketList,
} from "../types";

const BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

export class ApiError extends Error {
  // Declared as fields rather than constructor parameter properties, which
  // Vite's `erasableSyntaxOnly` rejects.
  status: number;
  code: string;
  detail?: Record<string, unknown>;

  constructor(
    message: string,
    status: number,
    code: string,
    detail?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }

  /** True when retrying later could plausibly succeed. */
  get isTransient(): boolean {
    return this.status === 429 || this.status === 503 || this.status >= 500;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}/api/v1${path}`, {
      ...init,
      credentials: "include",
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      "Could not reach the OpsPilot API. Check your connection and try again.",
      0,
      "network_error",
    );
  }

  if (!response.ok) {
    let message = `Request failed (HTTP ${response.status}).`;
    let code = "error";
    let detail: Record<string, unknown> | undefined;
    try {
      const body = await response.json();
      if (body?.error) {
        message = body.error.message ?? message;
        code = body.error.code ?? code;
        detail = body.error.detail;
      } else if (Array.isArray(body?.detail)) {
        message = "The request was rejected as invalid.";
        code = "invalid_request";
      }
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(message, response.status, code, detail);
  }

  return (await response.json()) as T;
}

export interface OrderQuery {
  snapshotId: string;
  limit?: number;
  offset?: number;
  sort?: "risk" | "deadline" | "handover";
  riskBand?: RiskBand | "";
  includeOverdue?: boolean;
  customerState?: string;
}

export const api = {
  meta: () => request<Meta>("/meta"),
  readiness: () =>
    request<{ status: string; checks: Record<string, { ok: boolean }> }>("/health/ready"),

  snapshots: () => request<Snapshot[]>("/snapshots"),
  snapshotStats: (id: string) => request<SnapshotStats>(`/snapshots/${id}/stats`),

  orders: (q: OrderQuery) => {
    const params = new URLSearchParams({ snapshot_id: q.snapshotId });
    if (q.limit != null) params.set("limit", String(q.limit));
    if (q.offset != null) params.set("offset", String(q.offset));
    if (q.sort) params.set("sort", q.sort);
    if (q.riskBand) params.set("risk_band", q.riskBand);
    if (q.includeOverdue) params.set("include_overdue", "true");
    if (q.customerState) params.set("customer_state", q.customerState);
    return request<OrderPage>(`/orders?${params.toString()}`);
  },

  order: (orderId: string, snapshotId: string) =>
    request<OrderAsOf>(
      `/orders/${encodeURIComponent(orderId)}?snapshot_id=${encodeURIComponent(snapshotId)}`,
    ),

  investigate: (orderId: string, snapshotId: string) =>
    request<Investigation>("/investigations", {
      method: "POST",
      body: JSON.stringify({ order_id: orderId, snapshot_id: snapshotId }),
    }),

  investigation: (id: string) =>
    request<Investigation>(`/investigations/${encodeURIComponent(id)}`),

  approve: (proposalId: string) =>
    request<DecisionResponse>(
      `/proposals/${encodeURIComponent(proposalId)}/approve`,
      { method: "POST" },
    ),

  reject: (proposalId: string) =>
    request<DecisionResponse>(
      `/proposals/${encodeURIComponent(proposalId)}/reject`,
      { method: "POST" },
    ),

  tickets: () => request<TicketList>("/tickets"),
  audit: () => request<AuditEvent[]>("/audit"),
};
