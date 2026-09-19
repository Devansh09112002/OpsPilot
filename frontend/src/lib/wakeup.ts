/** Free-tier cold-start handling.
 *
 *  Render's free web service spins down after ~15 minutes of inactivity and
 *  takes up to a minute to come back. Without this, a first-time visitor sees
 *  an apparently broken page for that whole minute.
 *
 *  `wakeBackend` polls the health endpoint until it answers, reporting
 *  progress so the UI can explain the wait rather than just spinning.
 */

const BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

export type WakeState = "checking" | "cold" | "awake" | "unreachable";

/** Considered a cold start if health does not answer within this long. */
const WARM_TIMEOUT_MS = 3_500;
/** Give up after this; a free instance should be back well inside it. */
const MAX_WAIT_MS = 90_000;
const POLL_INTERVAL_MS = 2_500;

async function pingHealth(timeoutMs: number): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE}/api/v1/health`, {
      signal: controller.signal,
      credentials: "omit",
    });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

export async function wakeBackend(onState: (s: WakeState) => void): Promise<WakeState> {
  onState("checking");

  if (await pingHealth(WARM_TIMEOUT_MS)) {
    onState("awake");
    return "awake";
  }

  // Not warm. Keep trying while telling the user why it is slow.
  onState("cold");
  const deadline = Date.now() + MAX_WAIT_MS;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    if (await pingHealth(WARM_TIMEOUT_MS * 2)) {
      onState("awake");
      return "awake";
    }
  }

  onState("unreachable");
  return "unreachable";
}
