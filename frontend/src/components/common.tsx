/** Small shared presentational pieces. */

import type { RiskBand } from "../types";

export function RiskBadge({ band }: { band: RiskBand }) {
  return <span className={`badge badge--${band}`}>{band}</span>;
}

/** Calibrated risk as a percentage, with a bar scaled so the useful range is
 *  readable. Most calibrated estimates sit under 0.25, so scaling the bar to
 *  the full 0-100% would render every row as a nearly-empty sliver. */
export function RiskScore({ value, band }: { value: number; band: RiskBand }) {
  const colour = `var(--risk-${band})`;
  const width = Math.min(100, Math.round((value / 0.4) * 100));
  return (
    <span className="risk-cell">
      <span className="mono">{formatRisk(value)}</span>
      <span className="risk-bar" aria-hidden="true">
        <span style={{ width: `${width}%`, background: colour }} />
      </span>
    </span>
  );
}

/** A calibrated probability, shown as a percentage a person can act on. */
export function formatRisk(value: number): string {
  if (value >= 0.095) return `${(value * 100).toFixed(0)}%`;
  return `${(value * 100).toFixed(1)}%`;
}

export function Spinner() {
  return <span className="spinner" role="status" aria-label="Loading" />;
}

export function ErrorState({
  title = "Something went wrong",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="state" role="alert">
      <h3>{title}</h3>
      <p className="muted">{message}</p>
      {onRetry && (
        <button onClick={onRetry} style={{ marginTop: "0.6rem" }}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <div className="state">
      <h3>{title}</h3>
      <p className="muted">{message}</p>
    </div>
  );
}

export function TableSkeleton({ rows = 8, cols = 7 }: { rows?: number; cols?: number }) {
  return (
    <tbody>
      {Array.from({ length: rows }).map((_, r) => (
        <tr key={r}>
          {Array.from({ length: cols }).map((__, c) => (
            <td key={c}>
              <div className="skeleton" style={{ width: c === 0 ? "80%" : "55%" }} />
            </td>
          ))}
        </tr>
      ))}
    </tbody>
  );
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function shortId(id: string, n = 10): string {
  return id.length > n ? `${id.slice(0, n)}…` : id;
}

export function formatBRL(v: number): string {
  return `R$ ${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}
