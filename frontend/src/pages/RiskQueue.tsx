/** Screen A - Risk Queue.
 *
 *  Every number shown here comes from the backend: snapshot counts and risk
 *  distribution are SQL aggregates, and each row's score was produced by the
 *  versioned model. Nothing is computed or invented in the browser.
 */

import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import {
  EmptyState,
  ErrorState,
  RiskBadge,
  RiskScore,
  TableSkeleton,
  formatDate,
  formatBRL,
  formatRisk,
  shortId,
} from "../components/common";
import { ApiError, api } from "../lib/api";
import type { OrderPage, RiskBand, Snapshot, SnapshotStats } from "../types";

const PAGE_SIZE = 25;

export default function RiskQueue() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const [snapshots, setSnapshots] = useState<Snapshot[] | null>(null);
  const [stats, setStats] = useState<SnapshotStats | null>(null);
  const [page, setPage] = useState<OrderPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const snapshotId = params.get("snapshot") ?? "";
  const riskBand = (params.get("band") ?? "") as RiskBand | "";
  const sort = (params.get("sort") ?? "risk") as "risk" | "deadline" | "handover";
  const includeOverdue = params.get("overdue") === "1";
  const offset = Number(params.get("offset") ?? 0);

  // The functional form matters. Two updates race on first load: the default
  // snapshot, set when the snapshot list arrives, and any filter the visitor
  // changes while it is still loading. Building the next params from a
  // captured `params` would let the later call clobber the earlier one,
  // leaving the queue with no snapshot at all and permanently empty.
  const update = useCallback(
    (patch: Record<string, string | null>) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          for (const [k, v] of Object.entries(patch)) {
            if (v === null || v === "") next.delete(k);
            else next.set(k, v);
          }
          // Any filter change resets pagination; otherwise offset can exceed total.
          if (!("offset" in patch)) next.delete("offset");
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  useEffect(() => {
    let cancelled = false;
    api
      .snapshots()
      .then((list) => {
        if (cancelled || !list.length) return;
        setSnapshots(list);
        // Default to the most recent snapshot, but never overwrite a snapshot
        // the visitor (or a shared link) already chose.
        setParams(
          (prev) => {
            if (prev.get("snapshot")) return prev;
            const next = new URLSearchParams(prev);
            next.set("snapshot", list[list.length - 1].snapshot_id);
            return next;
          },
          { replace: true },
        );
      })
      .catch((e: ApiError) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!snapshotId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      api.snapshotStats(snapshotId),
      api.orders({
        snapshotId,
        limit: PAGE_SIZE,
        offset,
        sort,
        riskBand,
        includeOverdue,
      }),
    ])
      .then(([s, p]) => {
        if (cancelled) return;
        setStats(s);
        setPage(p);
      })
      .catch((e: ApiError) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [snapshotId, riskBand, sort, includeOverdue, offset]);

  const snapshot = snapshots?.find((s) => s.snapshot_id === snapshotId);

  return (
    <div className="stack">
      <div>
        <h1>Delivery risk queue</h1>
        <p className="muted" style={{ maxWidth: "70ch" }}>
          Real Olist marketplace orders that were in transit on the selected
          historical date. Each figure is a <strong>calibrated estimate</strong>{" "}
          of the chance that order missed its promised date, produced by the
          trained model from information available{" "}
          <strong>at carrier handover</strong> and fitted to observed
          frequencies on held-out data. The data is historical; the scoring, AI
          investigation and ticketing you trigger here run live.
        </p>
      </div>

      <div className="toolbar">
        <div className="field">
          <label htmlFor="snapshot">Historical snapshot</label>
          <select
            id="snapshot"
            value={snapshotId}
            onChange={(e) => update({ snapshot: e.target.value })}
            disabled={!snapshots}
          >
            {(snapshots ?? []).map((s) => (
              <option key={s.snapshot_id} value={s.snapshot_id}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="band">Risk band</label>
          <select
            id="band"
            value={riskBand}
            disabled={!snapshots}
            onChange={(e) => update({ band: e.target.value })}
          >
            <option value="">All bands</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="sort">Sort by</label>
          <select
            id="sort"
            value={sort}
            disabled={!snapshots}
            onChange={(e) => update({ sort: e.target.value })}
          >
            <option value="risk">Highest risk</option>
            <option value="deadline">Nearest deadline</option>
            <option value="handover">Most recent handover</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="overdue">Past-deadline orders</label>
          <select
            id="overdue"
            value={includeOverdue ? "1" : "0"}
            disabled={!snapshots}
            onChange={(e) => update({ overdue: e.target.value === "1" ? "1" : null })}
          >
            <option value="0">Hidden</option>
            <option value="1">Included</option>
          </select>
        </div>
      </div>

      {snapshot && (
        <div className="banner">
          Viewing <strong>{snapshot.label}</strong>. The queue ranks the{" "}
          <strong>{snapshot.orders_pre_deadline.toLocaleString()}</strong> orders
          still before their promised date. A further{" "}
          {snapshot.orders_overdue.toLocaleString()} were already past it on this
          date and are shown separately as <em>overdue</em> rather than as
          predictions, since their outcome was already determined.
        </div>
      )}

      {stats && (
        <div className="stat-row">
          <div className="stat">
            <div className="stat__label">Eligible for review</div>
            <div className="stat__value">{stats.orders_pre_deadline.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="stat__label">High risk</div>
            <div className="stat__value stat__value--high">{stats.high_risk.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="stat__label">Medium risk</div>
            <div className="stat__value stat__value--medium">
              {stats.medium_risk.toLocaleString()}
            </div>
          </div>
          <div className="stat">
            <div className="stat__label">Low risk</div>
            <div className="stat__value stat__value--low">{stats.low_risk.toLocaleString()}</div>
          </div>
          <div className="stat">
            <div className="stat__label">Mean estimated risk</div>
            <div className="stat__value">{formatRisk(stats.mean_risk)}</div>
          </div>
        </div>
      )}

      {stats && stats.high_risk + stats.medium_risk > 0 && (
        <p className="muted small" data-testid="situations-hint">
          Reviewing{" "}
          {(stats.high_risk + stats.medium_risk).toLocaleString()} flagged orders
          one at a time is slow.{" "}
          <Link to={`/situations?snapshot=${encodeURIComponent(snapshotId)}`}>
            Group them by lane
          </Link>{" "}
          to see where the risk is concentrated.
        </p>
      )}

      {error ? (
        <ErrorState message={error} onRetry={() => update({})} />
      ) : (
        <>
          <div className="table-wrap">
            <table data-testid="risk-queue">
              <thead>
                <tr>
                  <th>Order</th>
                  <th className="num">Est. late</th>
                  <th>Band</th>
                  <th className="num">Days to deadline</th>
                  <th>Promised</th>
                  <th>Handover</th>
                  <th>Route</th>
                  <th>Category</th>
                  <th className="num">Value</th>
                </tr>
              </thead>
              {loading ? (
                <TableSkeleton rows={8} cols={9} />
              ) : (
                <tbody>
                  {(page?.items ?? []).map((o) => (
                    <tr
                      key={o.order_id}
                      className="clickable"
                      data-testid="order-row"
                      data-order-id={o.order_id}
                      onClick={() =>
                        navigate(`/orders/${o.order_id}?snapshot=${o.snapshot_id}`)
                      }
                    >
                      <td className="mono">
                        {/* A real link, not just a row click: the row handler
                            is a mouse convenience, and on its own it left the
                            queue unreachable by keyboard. */}
                        <Link
                          to={`/orders/${o.order_id}?snapshot=${o.snapshot_id}`}
                          onClick={(e) => e.stopPropagation()}
                        >
                          {shortId(o.order_id, 12)}
                        </Link>
                      </td>
                      <td className="num">
                        <RiskScore value={o.risk_probability} band={o.risk_band} />
                      </td>
                      <td>
                        {o.is_overdue ? (
                          <span className="badge badge--overdue">overdue</span>
                        ) : (
                          <RiskBadge band={o.risk_band} />
                        )}
                      </td>
                      <td className="num">{o.days_to_deadline.toFixed(0)}</td>
                      <td>{formatDate(o.order_estimated_delivery_date)}</td>
                      <td>{formatDate(o.order_delivered_carrier_date)}</td>
                      <td className="mono">
                        {o.seller_state ?? "--"} &rarr; {o.customer_state ?? "--"}
                      </td>
                      <td className="muted">{o.product_category ?? "--"}</td>
                      <td className="num">{formatBRL(o.total_price)}</td>
                    </tr>
                  ))}
                </tbody>
              )}
            </table>

            {!loading && page && page.items.length === 0 && (
              <EmptyState
                title="No orders match these filters"
                message="Try a different risk band or snapshot."
              />
            )}
          </div>

          {page && page.total > 0 && (
            <div className="row">
              <span className="muted small">
                Showing {offset + 1}&ndash;{Math.min(offset + PAGE_SIZE, page.total)} of{" "}
                {page.total.toLocaleString()} orders
                {stats && <> &middot; scored by <code>{stats.model_version}</code></>}
              </span>
              <span className="spacer" />
              <button
                disabled={offset === 0}
                onClick={() => update({ offset: String(Math.max(0, offset - PAGE_SIZE)) })}
              >
                Previous
              </button>
              <button
                disabled={offset + PAGE_SIZE >= page.total}
                onClick={() => update({ offset: String(offset + PAGE_SIZE) })}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
