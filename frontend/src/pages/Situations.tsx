/** Screen D - Lane situations.
 *
 *  The queue answers "which orders look risky". This screen answers the
 *  question a reviewer actually has: "where is the risk concentrated, and what
 *  is the one thing to do about it". The 2018-08-15 snapshot flags 395 orders
 *  across 37 lanes, five of which hold roughly three quarters of them.
 *
 *  Every figure is a backend aggregate. `expected_late` in particular is the
 *  sum of member calibrated probabilities, computed in SQL; nothing here adds
 *  anything up in the browser.
 */

import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { EmptyState, ErrorState, TableSkeleton, formatRisk } from "../components/common";
import { ApiError, api } from "../lib/api";
import type { SituationSummary, Snapshot } from "../types";

export default function Situations() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const [snapshots, setSnapshots] = useState<Snapshot[] | null>(null);
  const [rows, setRows] = useState<SituationSummary[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const snapshotId = params.get("snapshot") ?? "";

  // Functional update, for the same reason as the risk queue: the default
  // snapshot arriving and a visitor changing the filter can race.
  const update = useCallback(
    (key: string, value: string) => {
      setParams((prev) => {
        const next = new URLSearchParams(prev);
        if (value) next.set(key, value);
        else next.delete(key);
        return next;
      });
    },
    [setParams],
  );

  useEffect(() => {
    api
      .snapshots()
      .then((list) => {
        setSnapshots(list);
        if (!snapshotId && list.length) {
          update("snapshot", list[list.length - 1].snapshot_id);
        }
      })
      .catch((e: unknown) =>
        setError(e instanceof ApiError ? e.message : "Could not load snapshots."),
      );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!snapshotId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .situations(snapshotId, 20)
      .then((list) => !cancelled && setRows(list))
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof ApiError ? e.message : "Could not load situations.");
        setRows(null);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [snapshotId]);

  const totalFlagged = rows?.reduce((sum, r) => sum + r.n_flagged, 0) ?? 0;
  const totalExpected = rows?.reduce((sum, r) => sum + r.expected_late, 0) ?? 0;

  return (
    <section>
      <div className="row">
        <div>
          <h1>Lane situations</h1>
          <p className="muted" style={{ maxWidth: "62ch" }}>
            Flagged orders grouped by the lane they travel, ranked by how many
            risk they carry. A lane is one decision: you
            escalate a route with a carrier, not an order at a time.
          </p>
        </div>
        <label className="field">
          <span>Snapshot</span>
          <select
            value={snapshotId}
            disabled={!snapshots}
            onChange={(e) => update("snapshot", e.target.value)}
            data-testid="situation-snapshot"
          >
            {(snapshots ?? []).map((s) => (
              <option key={s.snapshot_id} value={s.snapshot_id}>
                {s.snapshot_id}
              </option>
            ))}
          </select>
        </label>
      </div>

      {rows && rows.length > 0 && (
        <div className="stat-row" data-testid="situation-stats">
          <div className="stat">
            <span className="stat__value">{rows.length}</span>
            <span className="stat__label">lanes with a cluster of flagged orders</span>
          </div>
          <div className="stat">
            <span className="stat__value">{totalFlagged}</span>
            <span className="stat__label">flagged orders they account for</span>
          </div>
          <div className="stat">
            <span className="stat__value">{totalExpected.toFixed(1)}</span>
            <span className="stat__label">total risk load</span>
          </div>
        </div>
      )}

      {error && <ErrorState message={error} onRetry={() => update("snapshot", snapshotId)} />}

      {loading && !error && <TableSkeleton rows={6} cols={6} />}

      {!loading && !error && rows && rows.length === 0 && (
        <EmptyState
          title="No lane situations in this snapshot"
          message="No lane carries enough flagged orders to be treated as a pattern. Individual orders remain reviewable in the risk queue."
        />
      )}

      {!loading && !error && rows && rows.length > 0 && (
        <div className="table-wrap">
          <table data-testid="situation-table">
            <thead>
              <tr>
                <th>Lane</th>
                <th className="num">Flagged</th>
                <th className="num">Risk load</th>
                <th className="num">Mean risk</th>
                <th className="num">Share of lane</th>
                <th>Escalatable</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.situation_id}
                  className="clickable"
                  data-testid="situation-row"
                  onClick={() => navigate(`/situations/${encodeURIComponent(r.situation_id)}`)}
                >
                  <td>
                    {/* Keyboard-reachable on its own; the row click is a
                        mouse convenience layered on top. */}
                    <Link
                      to={`/situations/${encodeURIComponent(r.situation_id)}`}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <strong>{r.lane}</strong>
                    </Link>
                    <div className="muted small">{r.situation_id}</div>
                  </td>
                  <td className="num">
                    {r.n_flagged}
                    {r.n_high > 0 && <span className="muted small"> ({r.n_high} high)</span>}
                  </td>
                  <td className="num">
                    <strong>{r.expected_late.toFixed(1)}</strong>
                  </td>
                  <td className="num">{formatRisk(r.mean_risk)}</td>
                  <td className="num">
                    {(r.share_of_lane * 100).toFixed(0)}%
                    <div className="muted small">
                      of {r.n_lane_total}
                    </div>
                  </td>
                  <td>
                    {r.n_escalatable >= 3 ? (
                      <span className="badge badge--high">{r.n_escalatable} qualify</span>
                    ) : (
                      <span className="muted small">
                        {r.n_escalatable} of 3 needed
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="muted small" style={{ marginTop: "1rem", maxWidth: "72ch" }}>
        <p>
          <strong>Risk load</strong> is the sum of the member orders&rsquo;
          calibrated risk estimates. Measured against what actually happened on
          these snapshots it <strong>overstates</strong> the number of orders
          that were really late &mdash; by about half again &mdash; because the
          model was calibrated on a period with a much higher late rate. Use it
          to compare one lane with another, which is what it is reliable for,
          rather than as a forecast of how many parcels will miss their date.
        </p>
        <p>
          <strong>High</strong> counts orders whose risk clears the escalation
          threshold. <strong>Qualify</strong> is smaller because escalation also
          requires three days or less before the promised date (ESC-01); a
          high-risk order with a week of slack is monitored, not escalated. A
          lane needs three qualifying orders before it can be escalated as one.
        </p>
      </div>
    </section>
  );
}
