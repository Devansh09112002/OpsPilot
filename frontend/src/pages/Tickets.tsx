/** Screen C - Tickets and action history for this guest session only. */

import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  EmptyState,
  ErrorState,
  Spinner,
  formatDateTime,
  shortId,
} from "../components/common";
import { ApiError, api } from "../lib/api";
import type { AuditEvent, TicketList } from "../types";

const EVENT_LABEL: Record<string, string> = {
  investigation_completed: "Investigation completed",
  investigation_failed: "Investigation failed",
  investigation_insufficient_evidence: "Investigation found insufficient evidence",
  proposal_created: "Escalation proposed",
  proposal_approved: "Escalation approved",
  proposal_rejected: "Escalation rejected",
};

export default function Tickets() {
  const [tickets, setTickets] = useState<TicketList | null>(null);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([api.tickets(), api.audit()])
      .then(([t, a]) => {
        setTickets(t);
        setEvents(a);
      })
      .catch((e: ApiError) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  if (loading) {
    return (
      <div className="state">
        <Spinner /> <span className="muted">Loading your tickets…</span>
      </div>
    );
  }

  if (error) {
    return <ErrorState message={error} onRetry={load} />;
  }

  return (
    <div className="stack">
      <div>
        <h1>Your tickets</h1>
        <p className="muted">
          Escalations you approved in this browser session. Tickets are scoped
          to your guest session, persist across refreshes, and are not visible
          to other visitors.
        </p>
      </div>

      <div className="banner banner--warn">
        <strong>Simulation.</strong> No real fulfilment action was taken for any
        ticket below. No courier, seller or customer is contacted.
      </div>

      {tickets && tickets.total === 0 ? (
        <div className="card">
          <EmptyState
            title="No tickets yet"
            message="Open an order from the risk queue, run an AI investigation, and approve a proposed escalation to create one."
          />
          <div style={{ textAlign: "center" }}>
            <Link to="/">Go to the risk queue</Link>
          </div>
        </div>
      ) : (
        <div className="table-wrap">
          <table data-testid="tickets-table">
            <thead>
              <tr>
                <th>Ticket</th>
                <th>Subject</th>
                <th>Status</th>
                <th>Action</th>
                <th>Reason</th>
                <th className="num">Risk at handover</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {tickets?.items.map((t) => (
                <tr key={t.ticket_id} data-testid="ticket-row">
                  <td className="mono">{shortId(t.ticket_id, 14)}</td>
                  <td>
                    {t.subject_type === "situation" ? (
                      <>
                        <Link to={`/situations/${encodeURIComponent(t.subject_id)}`}>
                          Lane {t.subject_id.split("__")[1]?.replace("-", " to ")}
                        </Link>
                        <div className="muted small">
                          covers {t.member_order_ids?.length ?? 0} orders
                        </div>
                      </>
                    ) : (
                      <Link
                        to={`/orders/${t.order_id}?snapshot=${t.snapshot_id}`}
                        className="mono"
                      >
                        {shortId(t.order_id ?? "", 12)}
                      </Link>
                    )}
                  </td>
                  <td>
                    <span className="badge badge--low">{t.status}</span>
                  </td>
                  <td className="muted">{t.action_type.replace(/_/g, " ")}</td>
                  <td style={{ whiteSpace: "normal", maxWidth: "26rem" }}>{t.reason}</td>
                  <td className="num mono">
                    {t.risk_probability != null ? t.risk_probability.toFixed(3) : "--"}
                  </td>
                  <td>{formatDateTime(t.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <div className="card__title">
          <h2>Action history</h2>
          <button onClick={load}>Refresh</button>
        </div>
        {events.length === 0 ? (
          <p className="muted small" style={{ marginBottom: 0 }}>
            Nothing recorded yet in this session.
          </p>
        ) : (
          <div>
            {events.map((e, i) => (
              <div className="evidence-item" key={`${e.subject_id}-${i}`}>
                <div className="row">
                  <span>{EVENT_LABEL[e.event_type] ?? e.event_type}</span>
                  <span className="spacer" />
                  <span className="faint small">{formatDateTime(e.created_at)}</span>
                </div>
                <div className="evidence-item__id">
                  {e.subject_type} {shortId(e.subject_id, 18)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
