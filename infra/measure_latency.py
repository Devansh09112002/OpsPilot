"""Measure deployed API latency against the plan's release targets.

Plan section 9.3 states two targets that can only be checked against the
running deployment:

  * non-LLM API response: p95 <= 2 s, excluding cold starts
  * investigation: warm-system p95 <= 30 s

This warms the instance first, then reports median and p95 per endpoint so the
numbers describe a warm system rather than a cold start.

    python -m infra.measure_latency
    python -m infra.measure_latency --investigations 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from infra.provision import read_env  # noqa: E402

SNAPSHOT = "2018-08-15"
NON_LLM_TARGET_P95 = 2.0
INVESTIGATION_TARGET_P95 = 30.0


def _pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    idx = min(int(len(ordered) * p), len(ordered) - 1)
    return ordered[idx]


def time_get(client: httpx.Client, url: str, n: int) -> list[float]:
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.get(url, timeout=60)
        r.raise_for_status()
        samples.append(time.perf_counter() - t0)
    return samples


def main(argv: list[str] | None = None) -> int:
    env = read_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=env.get("OPSPILOT_API_URL"))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--investigations", type=int, default=0,
                        help="LLM-backed samples; each consumes free-tier quota")
    args = parser.parse_args(argv)

    if not args.api:
        print("ERROR: no API URL", file=sys.stderr)
        return 2
    api = args.api.rstrip("/")

    with httpx.Client(follow_redirects=True) as client:
        print("Warming the instance (a free-tier service sleeps when idle)...")
        t0 = time.perf_counter()
        client.get(f"{api}/api/v1/health", timeout=120)
        print(f"  first request: {time.perf_counter() - t0:.2f}s")
        for _ in range(3):
            client.get(f"{api}/api/v1/snapshots", timeout=60)

        endpoints = {
            "GET /health": f"{api}/api/v1/health",
            "GET /snapshots": f"{api}/api/v1/snapshots",
            "GET /snapshots/{id}/stats": f"{api}/api/v1/snapshots/{SNAPSHOT}/stats",
            "GET /orders (25 rows)": f"{api}/api/v1/orders?snapshot_id={SNAPSHOT}&limit=25",
            "GET /meta": f"{api}/api/v1/meta",
        }

        print(f"\nNon-LLM endpoints, {args.samples} warm samples each")
        print(f"  {'endpoint':<30s} {'median':>9s} {'p95':>9s} {'max':>9s}")
        worst_p95 = 0.0
        for label, url in endpoints.items():
            s = time_get(client, url, args.samples)
            p95 = _pct(s, 0.95)
            worst_p95 = max(worst_p95, p95)
            print(f"  {label:<30s} {statistics.median(s):>8.3f}s "
                  f"{p95:>8.3f}s {max(s):>8.3f}s")

        ok = worst_p95 <= NON_LLM_TARGET_P95
        print(f"\n  worst p95 {worst_p95:.3f}s against a {NON_LLM_TARGET_P95:.0f}s target: "
              f"{'PASS' if ok else 'FAIL'}")

        if args.investigations:
            order = client.get(
                f"{api}/api/v1/orders?snapshot_id={SNAPSHOT}&risk_band=high&limit=1"
            ).json()["items"][0]["order_id"]
            print(f"\nInvestigations, {args.investigations} samples "
                  "(each consumes free-tier quota)")
            samples: list[float] = []
            for i in range(args.investigations):
                t0 = time.perf_counter()
                r = client.post(f"{api}/api/v1/investigations",
                                json={"order_id": order, "snapshot_id": SNAPSHOT},
                                timeout=180)
                elapsed = time.perf_counter() - t0
                status = r.json().get("status") if r.status_code == 200 else f"HTTP {r.status_code}"
                print(f"  sample {i + 1}: {elapsed:.2f}s  ({status})")
                if r.status_code == 200:
                    samples.append(elapsed)
            if samples:
                p95 = _pct(samples, 0.95)
                inv_ok = p95 <= INVESTIGATION_TARGET_P95
                print(f"\n  median {statistics.median(samples):.2f}s, p95 {p95:.2f}s "
                      f"against a {INVESTIGATION_TARGET_P95:.0f}s target: "
                      f"{'PASS' if inv_ok else 'FAIL'}")
                ok = ok and inv_ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
