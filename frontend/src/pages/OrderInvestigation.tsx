/** Screen B - Order detail and AI investigation.
 *
 *  Three things this screen is careful about:
 *  - The risk score is always labelled with its model version and the moment
 *    it refers to, never presented as a live re-forecast.
 *  - Every generated statement is shown with the evidence ids that support it,
 *    and the evidence panel lists the underlying tool results.
 *  - A proposal is inert until the visitor approves it. The button posts to
 *    the backend, which is what actually authorises the ticket.
 */

import { useCallback, useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import {
  ErrorState,
  RiskBadge,
  Spinner,
  formatDate,
  formatDateTime,
  formatBRL,
  formatRisk,
} from "../components/common";
import { ApiError, api } from "../lib/api";
import type { Investigation, OrderAsOf } from "../types";

const STEPS = [
  "Retrieving as-of order record",
  "Serving the delivery-risk model",
  "Computing historical comparisons",
  "Retrieving applicable demo policy",
  "Synthesising the evidence-backed report",
];

const RECOMMENDATION_LABEL: Record<string, string> = {
  no_escalation: "No escalation",
  monitor: "Monitor",
  propose_escalation: "Escalation proposed",
};

export default function OrderInvestigation() {
  const { orderId = "" } = useParams();
  const [params] = useSearchParams();
  const snapshotId = params.get("snapshot") ?? "";

  const [order, setOrder] = useState<OrderAsOf | null>(null);
  const [orderError, setOrderError] = useState<string | null>(null);
  const [laneIds, setLaneIds] = useState<Set<string>>(new Set());

  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [investigationError, setInvestigationError] = useState<string | null>(null);

  const [deciding, setDeciding] = useState(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setOrderError(null);
    api
      .order(orderId, snapshotId)
      .then((o) => !cancelled && setOrder(o))
      .catch((e: ApiError) => !cancelled && setOrderError(e.message));
    return () => {
      cancelled = true;
    };
  }, [orderId, snapshotId]);

  // Most lanes carry no cluster of flagged orders, so the link to a lane
  // situation is offered only where one exists. A link that usually dead-ends
  // is worse than no link.
  useEffect(() => {
    if (!snapshotId) return;
    let cancelled = false;
    api
      .situations(snapshotId, 100)
      .then((list) => !cancelled && setLaneIds(new Set(list.map((s) => s.situation_id))))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [snapshotId]);

  // The backend runs the investigation synchronously; this only paces the
  // progress list so the wait is legible. It never fabricates a result.
  useEffect(() => {
    if (!running) return;
    setStep(0);
    const timer = setInterval(
      () => setStep((s) => Math.min(s + 1, STEPS.length - 1)),
      2200,
    );
    return () => clearInterval(timer);
  }, [running]);

  const investigate = useCallback(async () => {
    setRunning(true);
    setInvestigationError(null);
    setDecisionError(null);
    try {
      setInvestigation(await api.investigate(orderId, snapshotId));
    } catch (e) {
      const err = e as ApiError;
      setInvestigationError(
        err.status === 429
          ? err.message
          : `${err.message} The order data and model prediction above remain available.`,
      );
    } finally {
      setRunning(false);
    }
  }, [orderId, snapshotId]);

  const decide = useCallback(
    async (action: "approve" | "reject") => {
      if (!investigation?.proposal) return;
      setDeciding(true);
      setDecisionError(null);
      try {
        const fn = action === "approve" ? api.approve : api.reject;
        await fn(investigation.proposal.proposal_id);
        // Re-read from the server so what is displayed is what was persisted.
        setInvestigation(await api.investigation(investigation.investigation_id));
      } catch (e) {
        setDecisionError((e as ApiError).message);
      } finally {
        setDeciding(false);
      }
    },
    [investigation],
  );

  if (orderError) {
    return (
      <div className="stack">
        <Link to="/">&larr; Back to risk queue</Link>
        <ErrorState title="Order unavailable" message={orderError} />
      </div>
    );
  }

  if (!order) {
    return (
      <div className="state">
        <Spinner /> <span className="muted">Loading order…</span>
      </div>
    );
  }

  const proposal = investigation?.proposal ?? null;

  return (
    <div className="stack">
      <div className="row">
        <Link to={`/?snapshot=${snapshotId}`}>&larr; Back to risk queue</Link>
        <span className="spacer" />
        <span className="faint small mono">{order.order_id}</span>
      </div>

      <div className="split">
        {/* ---------------- order facts ---------------- */}
        <div className="stack">
          <div className="card">
            <div className="card__title">
              <h2>Model assessment</h2>
              {order.is_overdue ? (
                <span className="badge badge--overdue">overdue</span>
              ) : (
                <RiskBadge band={order.risk_band} />
              )}
            </div>
            <div style={{ fontSize: "2.4rem", fontWeight: 650, lineHeight: 1.1 }}>
              {formatRisk(order.risk_probability)}
            </div>
            <p className="faint small" style={{ marginTop: "0.4rem" }}>
              {order.calibrated ? "Calibrated estimate" : "Uncalibrated score"} that
              this order misses its promised date, predicted at carrier handover
              on {formatDateTime(order.prediction_as_of)} by model{" "}
              <code>{order.model_version}</code>
              {order.calibrated && (
                <> against a marketplace baseline near 3%</>
              )}
              . Not a diagnosis of a cause.
            </p>

            {order.risk_factors.length > 0 && (
              <div style={{ marginTop: "1rem" }}>
                <h3 className="faint" style={{ marginBottom: "0.5rem" }}>
                  What moved this score
                </h3>
                {order.risk_factors.map((f) => (
                  <div className="factor" key={f.feature}>
                    <div className="factor__row">
                      <span className="factor__label">{f.label}</span>
                      <span
                        className={
                          f.direction === "increases risk"
                            ? "factor__dir factor__dir--up"
                            : "factor__dir factor__dir--down"
                        }
                      >
                        {f.direction === "increases risk" ? "▲" : "▼"}{" "}
                        {(f.share * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="factor__bar" aria-hidden="true">
                      <span
                        style={{
                          width: `${Math.round(f.share * 100)}%`,
                          background:
                            f.direction === "increases risk"
                              ? "var(--risk-high)"
                              : "var(--risk-low)",
                        }}
                      />
                    </div>
                  </div>
                ))}
                <p className="faint small" style={{ marginTop: "0.6rem", marginBottom: 0 }}>
                  Exact attribution of this order's score to its inputs
                  (TreeSHAP). These are what moved the model, not established
                  causes of delay.
                </p>
              </div>
            )}
          </div>

          <div className="card">
            <h1 style={{ marginBottom: "0.7rem" }}>Order as of {formatDate(order.snapshot_at)}</h1>
            <dl className="kv">
              <dt>Purchased</dt>
              <dd>{formatDateTime(order.order_purchase_timestamp)}</dd>
              <dt>Payment approved</dt>
              <dd>
                {order.order_approved_at ? formatDateTime(order.order_approved_at) : "--"}
              </dd>
              <dt>Carrier handover</dt>
              <dd>{formatDateTime(order.order_delivered_carrier_date)}</dd>
              <dt>Promised delivery</dt>
              <dd>{formatDate(order.order_estimated_delivery_date)}</dd>
              <dt>Days in transit</dt>
              <dd>{order.days_in_transit.toFixed(1)}</dd>
              <dt>Days to deadline</dt>
              <dd>{order.days_to_deadline.toFixed(0)}</dd>
              <dt>Items</dt>
              <dd>
                {order.n_items} from {order.n_distinct_sellers} seller
                {order.n_distinct_sellers === 1 ? "" : "s"}
              </dd>
              <dt>Order value</dt>
              <dd>{formatBRL(order.total_price)}</dd>
              <dt>Freight</dt>
              <dd>{formatBRL(order.total_freight)}</dd>
              <dt>Category</dt>
              <dd>{order.product_category ?? "--"}</dd>
              <dt>Route</dt>
              <dd className="mono">
                {order.seller_state ?? "--"} &rarr; {order.customer_state ?? "--"}
                {order.is_cross_state ? " (interstate)" : " (same state)"}
                {order.seller_state &&
                  order.customer_state &&
                  laneIds.has(
                    `${snapshotId}__${order.seller_state}-${order.customer_state}`,
                  ) && (
                  <div style={{ marginTop: "0.25rem" }}>
                    <Link
                      to={`/situations/${encodeURIComponent(
                        `${snapshotId}__${order.seller_state}-${order.customer_state}`,
                      )}`}
                      data-testid="lane-link"
                    >
                      See this lane&rsquo;s situation
                    </Link>
                  </div>
                )}
              </dd>
            </dl>
            <p className="faint small" style={{ marginTop: "0.8rem", marginBottom: 0 }}>
              These are the only order facts available at this snapshot. The
              eventual delivery outcome is deliberately withheld from this view
              and from the AI investigation.
            </p>
          </div>
        </div>

        {/* ---------------- investigation ---------------- */}
        <div className="stack">
          <div className="card">
            <div className="card__title">
              <h2>AI investigation</h2>
              {investigation && (
                <span className="badge badge--neutral">
                  {investigation.status.replace(/_/g, " ")}
                </span>
              )}
            </div>

            {!investigation && !running && !investigationError && (
              <>
                <p className="muted">
                  Runs a bounded workflow that retrieves this order's as-of
                  record, re-serves the delivery-risk model, computes historical
                  comparisons from deliveries completed before this snapshot,
                  and applies the versioned demo policy. Every figure in the
                  report is checked against the tool that produced it.
                </p>
                <button className="btn-primary" onClick={investigate} data-testid="investigate">
                  Investigate with AI
                </button>
              </>
            )}

            {running && (
              <div data-testid="investigation-running">
                <p className="muted">
                  <Spinner /> Investigation in progress…
                </p>
                <ol className="progress-steps">
                  {STEPS.map((label, i) => (
                    <li key={label} className={i < step ? "done" : i === step ? "active" : ""}>
                      {i < step ? "✓" : i === step ? "→" : "·"} {label}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {investigationError && (
              <div className="banner banner--error" data-testid="investigation-error">
                <strong>Investigation unavailable.</strong> {investigationError}
                <div style={{ marginTop: "0.6rem" }}>
                  <button onClick={investigate}>Retry</button>
                </div>
              </div>
            )}

            {investigation && investigation.status === "failed" && (
              <div className="banner banner--error">
                <strong>Investigation failed.</strong>{" "}
                {investigation.error_message ??
                  "The investigation could not be completed."}{" "}
                No ticket was created. The order data and model prediction
                remain available.
                <div style={{ marginTop: "0.6rem" }}>
                  <button onClick={investigate}>Retry</button>
                </div>
              </div>
            )}

            {investigation && investigation.status === "insufficient_evidence" && (
              <div className="banner banner--warn">
                <strong>Insufficient evidence.</strong>{" "}
                {investigation.error_message}
              </div>
            )}

            {investigation && investigation.status === "completed" && (
              <div className="stack" data-testid="investigation-report">
                <p>{investigation.summary}</p>

                {investigation.facts.length > 0 && (
                  <div>
                    <h3 className="faint" style={{ marginBottom: "0.5rem" }}>
                      Findings
                    </h3>
                    {investigation.facts.map((f, i) => (
                      <div className="fact" key={i}>
                        {f.statement}
                        <div className="fact__cites">
                          {f.evidence_ids.map((id) => (
                            <span className="cite" key={id}>
                              {id}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {investigation.limitations.length > 0 && (
                  <div>
                    <h3 className="faint" style={{ marginBottom: "0.5rem" }}>
                      Limitations
                    </h3>
                    {investigation.limitations.map((l, i) => (
                      <div className="limitation" key={i}>
                        {l}
                      </div>
                    ))}
                  </div>
                )}

                <div>
                  <h3 className="faint" style={{ marginBottom: "0.3rem" }}>
                    Recommendation
                  </h3>
                  <p style={{ marginBottom: "0.25rem" }}>
                    <strong>
                      {RECOMMENDATION_LABEL[investigation.recommendation ?? ""] ??
                        investigation.recommendation}
                    </strong>
                  </p>
                  <p className="muted small">{investigation.recommendation_rationale}</p>
                </div>

                <details>
                  <summary className="faint small" style={{ cursor: "pointer" }}>
                    Evidence used ({investigation.evidence.length} items)
                  </summary>
                  <div style={{ marginTop: "0.6rem" }}>
                    {investigation.evidence.map((e) => (
                      <div className="evidence-item" key={e.evidence_id}>
                        <div className="evidence-item__id">{e.evidence_id}</div>
                        <div className="evidence-item__label">{e.label}</div>
                        <div
                          className={
                            e.policy_section
                              ? "evidence-item__value--policy"
                              : "evidence-item__value"
                          }
                        >
                          {e.value}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>

                {investigation.duration_ms != null && (
                  <p className="faint small" style={{ marginBottom: 0 }}>
                    Completed in {(investigation.duration_ms / 1000).toFixed(1)}s using
                    model <code>{investigation.model_version}</code>.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* ---------------- approval gate ---------------- */}
          {proposal && (
            <div
              className={`proposal-box${proposal.status !== "pending" ? " proposal-box--decided" : ""}`}
              data-testid="proposal"
            >
              <div className="card__title">
                <h2>Proposed action</h2>
                <span className="badge badge--neutral">{proposal.status}</span>
              </div>
              <p style={{ marginBottom: "0.3rem" }}>{proposal.reason}</p>
              <p className="faint small">
                The agent can only propose. Nothing is recorded until you decide.
              </p>

              {decisionError && (
                <div className="banner banner--error" style={{ margin: "0.6rem 0 0" }}>
                  {decisionError}
                </div>
              )}

              {proposal.status === "pending" && (
                <div className="proposal-actions">
                  <button
                    className="btn-approve"
                    disabled={deciding}
                    onClick={() => decide("approve")}
                    data-testid="approve"
                  >
                    {deciding ? "Working…" : "Approve escalation"}
                  </button>
                  <button
                    className="btn-reject"
                    disabled={deciding}
                    onClick={() => decide("reject")}
                    data-testid="reject"
                  >
                    Reject
                  </button>
                </div>
              )}

              {proposal.status === "approved" && investigation?.ticket_id && (
                <p style={{ marginTop: "0.7rem", marginBottom: 0 }} data-testid="ticket-link">
                  Ticket <code>{investigation.ticket_id}</code> created.{" "}
                  <Link to="/tickets">View your tickets</Link>
                </p>
              )}

              {proposal.status === "rejected" && (
                <p className="muted" style={{ marginTop: "0.7rem", marginBottom: 0 }}>
                  Rejected. No ticket was created.
                </p>
              )}
            </div>
          )}

          {investigation && (
            <p className="faint small">{investigation.disclaimer}</p>
          )}
        </div>
      </div>
    </div>
  );
}
