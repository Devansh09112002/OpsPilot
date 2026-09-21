/** Screen E - one lane situation, its investigation and the decision.
 *
 *  Same trust rules as the order screen, plus two of its own:
 *
 *  - The escalation here covers every member order, so the count is stated on
 *    the approval control rather than left to be inferred from the lane name.
 *  - A brief can be produced with or without the LLM. Which one produced the
 *    text on screen is labelled, because a reader is entitled to know whether
 *    they are reading generated prose or assembled facts.
 */

import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ErrorState, RiskBadge, Spinner, formatRisk, shortId } from "../components/common";
import { ApiError, api } from "../lib/api";
import type { Investigation, SituationDetail as Detail } from "../types";

const RECOMMENDATION_LABEL: Record<string, string> = {
  propose_escalation: "Propose a lane escalation",
  monitor: "Monitor",
  no_escalation: "No escalation",
};

export default function SituationDetail() {
  const { situationId = "" } = useParams();

  const [situation, setSituation] = useState<Detail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [running, setRunning] = useState(false);
  const [investigationError, setInvestigationError] = useState<string | null>(null);
  const [deciding, setDeciding] = useState(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .situation(situationId)
      .then((s) => !cancelled && setSituation(s))
      .catch((e: unknown) => {
        if (cancelled) return;
        setLoadError(
          e instanceof ApiError ? e.message : "Could not load this situation.",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [situationId]);

  const investigate = useCallback(
    async (mode: "llm" | "deterministic") => {
      setRunning(true);
      setInvestigationError(null);
      setDecisionError(null);
      try {
        setInvestigation(await api.investigateSituation(situationId, mode));
      } catch (e) {
        const err = e as ApiError;
        setInvestigationError(
          err.status === 429
            ? err.message
            : `${err.message} The lane figures above remain available.`,
        );
      } finally {
        setRunning(false);
      }
    },
    [situationId],
  );

  const decide = useCallback(
    async (action: "approve" | "reject") => {
      if (!investigation?.proposal) return;
      setDeciding(true);
      setDecisionError(null);
      try {
        const fn = action === "approve" ? api.approve : api.reject;
        await fn(investigation.proposal.proposal_id);
        // Re-read from the server so the screen shows what was persisted.
        setInvestigation(await api.investigation(investigation.investigation_id));
      } catch (e) {
        setDecisionError((e as ApiError).message);
      } finally {
        setDeciding(false);
      }
    },
    [investigation],
  );

  if (loadError) {
    return (
      <div className="stack">
        <Link to="/situations">&larr; All situations</Link>
        <ErrorState message={loadError} />
      </div>
    );
  }

  if (!situation) {
    return (
      <div className="state">
        <Spinner />
        <p className="muted">Loading the situation…</p>
      </div>
    );
  }

  const history = situation.lane_history;
  const proposal = investigation?.proposal ?? null;

  return (
    <div className="stack">
      <Link to={`/situations?snapshot=${encodeURIComponent(situation.snapshot_id)}`}>
        &larr; All situations
      </Link>

      <section className="card">
        <div className="row">
          <div>
            <h1 style={{ marginBottom: "0.2rem" }}>Lane {situation.lane}</h1>
            <p className="muted" style={{ margin: 0 }}>
              Snapshot {situation.snapshot_id} &middot; model{" "}
              <code>{situation.model_version ?? "unavailable"}</code>
            </p>
          </div>
        </div>

        <div className="stat-row" data-testid="situation-detail-stats">
          <div className="stat">
            <span className="stat__value">{situation.n_flagged}</span>
            <span className="stat__label">flagged orders</span>
          </div>
          <div className="stat">
            <span className="stat__value">{situation.expected_late.toFixed(1)}</span>
            <span className="stat__label">risk load</span>
          </div>
          <div className="stat">
            <span className="stat__value">{situation.n_escalatable}</span>
            <span className="stat__label">qualify under ESC-01</span>
          </div>
          <div className="stat">
            <span className="stat__value">
              {(situation.share_of_lane * 100).toFixed(0)}%
            </span>
            <span className="stat__label">of this lane&rsquo;s volume</span>
          </div>
          <div className="stat">
            <span className="stat__value">{formatRisk(situation.max_risk)}</span>
            <span className="stat__label">worst single order</span>
          </div>
        </div>

        <h2 className="faint">Lane history</h2>
        {history.available ? (
          <p style={{ marginTop: 0 }}>
            <strong>{formatRisk(history.late_rate ?? 0)}</strong> of{" "}
            {history.sample_size.toLocaleString()} orders delivered on this lane
            before the snapshot were late
            {history.baseline_rate != null && (
              <>
                , against <strong>{formatRisk(history.baseline_rate)}</strong>{" "}
                across the marketplace
              </>
            )}
            .
          </p>
        ) : (
          <p className="muted" style={{ marginTop: 0 }}>
            No lane rate is reported.
          </p>
        )}
        {history.caveats.map((c, i) => (
          <div className="limitation" key={i}>
            {c}
          </div>
        ))}
      </section>

      <section className="card">
        <h2 style={{ marginTop: 0 }}>AI investigation</h2>

        {!investigation && !running && (
          <>
            <p className="muted" style={{ maxWidth: "62ch" }}>
              Retrieves this lane&rsquo;s orders, the model&rsquo;s aggregate,
              the as-of lane history and the applicable policy, then produces a
              cited assessment. The deterministic brief uses the same verified
              facts without calling a language model, and works when the free
              provider tier is exhausted.
            </p>
            <div className="proposal-actions">
              <button
                className="btn-primary"
                onClick={() => investigate("llm")}
                data-testid="investigate-situation"
              >
                Investigate this lane
              </button>
              <button
                onClick={() => investigate("deterministic")}
                data-testid="investigate-deterministic"
              >
                Deterministic brief (no AI)
              </button>
            </div>
          </>
        )}

        {running && (
          <div data-testid="situation-investigation-running">
            <Spinner />
            <p className="muted">Retrieving evidence and assessing the lane…</p>
          </div>
        )}

        {investigationError && (
          <div className="banner banner--error" data-testid="situation-investigation-error">
            <p style={{ marginBottom: "0.5rem" }}>{investigationError}</p>
            <button onClick={() => investigate("deterministic")}>
              Try the deterministic brief
            </button>
          </div>
        )}

        {investigation && investigation.status === "completed" && (
          <div className="stack" data-testid="situation-report">
            <div
              className={
                investigation.generated_by === "deterministic"
                  ? "banner banner--warn"
                  : "banner"
              }
              data-testid="generated-by"
            >
              {investigation.generated_by === "deterministic" ? (
                <>
                  <strong>Deterministic brief.</strong> Assembled from tool
                  results with no language model. Every sentence restates a
                  retrieved fact.
                </>
              ) : (
                <>
                  <strong>AI-generated assessment.</strong> Every claim was
                  checked against real tool output before display.
                </>
              )}
            </div>

            <p>{investigation.summary}</p>

            {investigation.facts.length > 0 && (
              <div>
                <h2 className="faint" style={{ marginBottom: "0.5rem" }}>
                  Findings
                </h2>
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
                <h2 className="faint" style={{ marginBottom: "0.5rem" }}>
                  Limitations
                </h2>
                {investigation.limitations.map((l, i) => (
                  <div className="limitation" key={i}>
                    {l}
                  </div>
                ))}
              </div>
            )}

            <div>
              <h2 className="faint" style={{ marginBottom: "0.3rem" }}>
                Recommendation
              </h2>
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
          </div>
        )}

        {investigation && investigation.status !== "completed" && (
          <div className="banner banner--warn" data-testid="situation-investigation-incomplete">
            <strong>No assessment was produced.</strong>{" "}
            {investigation.error_message}
          </div>
        )}
      </section>

      {proposal && (
        <section className="card" data-testid="situation-proposal">
          <h2 style={{ marginTop: 0 }}>Proposed action</h2>
          <p>{proposal.reason}</p>
          <p className="muted small">
            This escalation would cover{" "}
            <strong>{proposal.member_order_ids?.length ?? 0} orders</strong> on
            lane {situation.lane}. It is a proposal only: nothing happens until
            you approve it, and no courier, seller or customer is contacted
            either way.
          </p>

          {decisionError && <div className="banner banner--error">{decisionError}</div>}

          {proposal.status === "pending" && (
            <div className="proposal-actions">
              <button
                className="btn-approve"
                disabled={deciding}
                onClick={() => decide("approve")}
                data-testid="approve-situation"
              >
                Approve escalation for {proposal.member_order_ids?.length ?? 0} orders
              </button>
              <button
                className="btn-reject"
                disabled={deciding}
                onClick={() => decide("reject")}
                data-testid="reject-situation"
              >
                Reject
              </button>
            </div>
          )}

          {proposal.status === "approved" && (
            <p style={{ marginTop: "0.7rem", marginBottom: 0 }} data-testid="situation-ticket-link">
              Approved. <Link to="/tickets">View the ticket</Link>.
            </p>
          )}

          {proposal.status === "rejected" && (
            <p className="muted" style={{ marginTop: "0.7rem", marginBottom: 0 }}>
              Rejected. No ticket was created.
            </p>
          )}
        </section>
      )}

      <section className="card">
        <h2 style={{ marginTop: 0 }}>
          Member orders ({situation.members.length})
        </h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Order</th>
                <th className="num">Risk</th>
                <th>Band</th>
                <th className="num">Days to deadline</th>
                <th>Category</th>
                <th>ESC-01</th>
              </tr>
            </thead>
            <tbody>
              {situation.members.map((m) => (
                <tr key={m.order_id} data-testid="situation-member">
                  <td>
                    <Link
                      to={`/orders/${encodeURIComponent(m.order_id)}?snapshot=${encodeURIComponent(situation.snapshot_id)}`}
                    >
                      {shortId(m.order_id)}
                    </Link>
                  </td>
                  <td className="num">{formatRisk(m.risk_probability)}</td>
                  <td>
                    <RiskBadge band={m.risk_band} />
                  </td>
                  <td className="num">{m.days_to_deadline.toFixed(0)}</td>
                  <td className="muted small">{m.product_category ?? "--"}</td>
                  <td>
                    {m.escalatable ? (
                      <span className="badge badge--high">qualifies</span>
                    ) : (
                      <span className="muted small">no</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
