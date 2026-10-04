/** Small shared presentational pieces. */

import type { ArrivalTag, RiskBand } from "../types";

export function RiskBadge({ band }: { band: RiskBand }) {
  return <span className={`badge badge--${band}`}>{band}</span>;
}

/** The model's estimated chance of missing the promise, as a percentage with a
 *  bar. An estimate, not a calibrated frequency: it runs high in calm periods,
 *  which is why the queue is ordered by priority rank rather than by this. */
export function RiskScore({ value, band }: { value: number; band: RiskBand }) {
  const colour = `var(--risk-${band})`;
  const width = Math.min(100, Math.round(value * 100));
  return (
    <span className="risk-cell">
      <span className="mono">{formatRisk(value)}</span>
      <span className="risk-bar" aria-hidden="true">
        <span style={{ width: `${width}%`, background: colour }} />
      </span>
    </span>
  );
}

/** A probability estimate, shown as a percentage. */
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

const TAG_LABEL: Record<ArrivalTag, string> = {
  likely_late: "Likely late",
  tight: "Tight",
  on_track: "On track",
  overdue: "Overdue",
  unknown: "--",
};
const TAG_CLASS: Record<ArrivalTag, string> = {
  likely_late: "badge--high",
  tight: "badge--medium",
  on_track: "badge--low",
  overdue: "badge--overdue",
  unknown: "badge--neutral",
};

/** How the forecast arrival compares with the promised date. A forecast. */
export function ArrivalBadge({ tag }: { tag: ArrivalTag }) {
  return <span className={`badge ${TAG_CLASS[tag]}`}>{TAG_LABEL[tag]}</span>;
}

/** A calendar day ("2018-08-18"), formatted without a timezone shift.
 *  `new Date("2018-08-18")` is UTC midnight, which renders as the previous
 *  day anywhere west of Greenwich. */
export function formatDay(day: string | null): string {
  if (!day) return "--";
  const [y, m, d] = day.slice(0, 10).split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

/** Days of buffer against the promise, in words. */
export function formatBuffer(days: number | null): string {
  if (days === null) return "--";
  if (days < 0) return `${-days} day${days === -1 ? "" : "s"} after the promise`;
  if (days === 0) return "on the promised day";
  return `${days} day${days === 1 ? "" : "s"} before the promise`;
}
