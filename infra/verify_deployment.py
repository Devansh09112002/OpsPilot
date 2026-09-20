"""End-to-end verification of the deployed OpsPilot, over HTTP.

Exercises the real public URLs the way a visitor does, and reports what
actually happened rather than that requests returned 200. Complements the
Playwright suite: this is scriptable, has no browser dependency, and is what
CI or a release check can run.

    python -m infra.verify_deployment
    python -m infra.verify_deployment --api https://... --web https://...
    python -m infra.verify_deployment --skip-investigation   # save LLM quota

Exit 0 only when every critical check passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from infra.provision import read_env  # noqa: E402

# A free-tier instance can be asleep; allow for a cold start.
COLD_START_TIMEOUT = 120
SNAPSHOT = "2018-08-15"

OUTCOME_MARKERS = ("order_delivered_customer_date", "is_late", "actual_delivery")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    critical: bool = True


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", critical: bool = True) -> bool:
        self.checks.append(Check(name, ok, detail, critical))
        mark = "PASS" if ok else ("FAIL" if critical else "warn")
        print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
        return ok

    @property
    def failed_critical(self) -> list[Check]:
        return [c for c in self.checks if c.critical and not c.ok]


def wake(client: httpx.Client, api: str, report: Report) -> bool:
    """Wait out a free-tier cold start before judging anything else."""
    started = time.time()
    while time.time() - started < COLD_START_TIMEOUT:
        try:
            r = client.get(f"{api}/api/v1/health", timeout=30)
            if r.status_code == 200:
                elapsed = time.time() - started
                report.add("API reachable", True, f"{elapsed:.1f}s (cold start allowed)")
                return True
        except Exception:
            pass
        time.sleep(5)
    report.add("API reachable", False,
               f"no response within {COLD_START_TIMEOUT}s")
    return False


def verify(api: str, web: str, *, run_investigation: bool) -> Report:
    report = Report()
    api = api.rstrip("/")
    web = web.rstrip("/")

    print(f"\nAPI: {api}\nWeb: {web}\n")
    # One client keeps the guest-session cookie across the whole journey, which
    # is what makes the ticket checks meaningful.
    with httpx.Client(follow_redirects=True, timeout=90) as client:
        print("Availability")
        if not wake(client, api, report):
            return report

        r = client.get(f"{api}/api/v1/health/ready")
        body = r.json() if r.status_code == 200 else {}
        checks = body.get("checks", {})
        report.add("database reachable from the API", checks.get("database", {}).get("ok") is True)
        report.add("model artifact loaded",
                   checks.get("model", {}).get("ok") is True,
                   checks.get("model", {}).get("model_version", ""))
        report.add("policy loaded", checks.get("policy", {}).get("ok") is True)
        llm_ok = checks.get("llm", {}).get("ok") is True
        report.add("LLM configured", llm_ok,
                   "" if llm_ok else "investigations will be unavailable",
                   critical=False)

        print("\nFrontend")
        r = client.get(web)
        report.add("frontend served over HTTPS", r.status_code == 200 and web.startswith("https://"))
        report.add("deep link /tickets resolves (SPA rewrite)",
                   client.get(f"{web}/tickets").status_code == 200)
        html = r.text
        bundle = None
        for token in html.split('src="')[1:]:
            path = token.split('"')[0]
            if path.endswith(".js"):
                bundle = path
                break
        if bundle:
            js = client.get(f"{web}{bundle}").text
            leaked = any(m in js for m in ("sk-ant-", "sbp_", "rnd_")) or "AIza" in js
            report.add("no credential in the JS bundle", not leaked, f"{len(js):,} bytes")
            report.add("bundle points at the deployed API", api in js)

        print("\nData and model")
        snaps = client.get(f"{api}/api/v1/snapshots").json()
        report.add("snapshots present", len(snaps) == 3, f"{len(snaps)} snapshots")

        stats = client.get(f"{api}/api/v1/snapshots/{SNAPSHOT}/stats").json()
        total_banded = stats.get("high_risk", 0) + stats.get("medium_risk", 0) + stats.get("low_risk", 0)
        report.add("risk distribution computed from SQL", total_banded > 0,
                   f"{total_banded:,} orders banded, mean risk {stats.get('mean_risk')}")

        page = client.get(f"{api}/api/v1/orders?snapshot_id={SNAPSHOT}&limit=25").json()
        items = page.get("items", [])
        report.add("risk queue returns orders", len(items) > 0, f"{page.get('total', 0):,} total")
        if items:
            risks = [i["risk_probability"] for i in items]
            report.add("scores are valid probabilities", all(0 <= r <= 1 for r in risks))
            report.add("queue is ranked by risk", risks == sorted(risks, reverse=True))
            report.add("every score carries a model version",
                       all(i.get("model_version") for i in items),
                       items[0].get("model_version", ""))

        print("\nLeakage boundary")
        blob = str(page)
        report.add("order list exposes no delivery outcome",
                   not any(m in blob for m in OUTCOME_MARKERS))
        order_id = items[0]["order_id"] if items else None
        if order_id:
            detail = client.get(
                f"{api}/api/v1/orders/{order_id}?snapshot_id={SNAPSHOT}"
            ).json()
            report.add("order detail exposes no delivery outcome",
                       not any(m in str(detail) for m in OUTCOME_MARKERS))
            report.add("prediction is labelled as of carrier handover",
                       detail.get("prediction_as_of") == detail.get("order_delivered_carrier_date"))

            served = client.post(f"{api}/api/v1/predictions",
                                 json={"order_id": order_id, "snapshot_id": SNAPSHOT}).json()
            report.add("live inference matches the stored queue score",
                       abs(served.get("risk_probability", -1)
                           - detail.get("risk_probability", -2)) < 1e-4,
                       f"{served.get('risk_probability')} vs {detail.get('risk_probability')}")

        print("\nSession and authorization")
        # The session is minted by the first endpoint that needs one, so the
        # cookie has to be checked *after* such a request, not before.
        r = client.get(f"{api}/api/v1/tickets")
        set_cookie = r.headers.get("set-cookie", "")
        tickets = r.json()
        report.add(
            "guest session cookie issued",
            bool(client.cookies.get("opspilot_session"))
            or "opspilot_session" in set_cookie,
        )
        if set_cookie:
            lowered = set_cookie.lower()
            report.add("session cookie is HttpOnly", "httponly" in lowered)
            report.add(
                "session cookie is Secure and SameSite=None (cross-origin)",
                "secure" in lowered and "samesite=none" in lowered,
            )
        report.add("tickets start empty for a new visitor",
                   tickets.get("total") == 0)
        report.add("ticket list is labelled as simulated",
                   "simulation" in tickets.get("disclaimer", "").lower())

        with httpx.Client(timeout=60) as stranger:
            stranger.get(f"{api}/api/v1/snapshots")
            r = stranger.post(f"{api}/api/v1/proposals/prp_doesnotexist/approve")
            report.add("unknown proposal id is refused", r.status_code == 404,
                       f"HTTP {r.status_code}")

        if not (run_investigation and llm_ok and order_id):
            print("\nInvestigation: skipped")
            return report

        print("\nInvestigation, approval and ticket")
        # Pick a high-risk order: those are the ones policy can escalate.
        high = client.get(
            f"{api}/api/v1/orders?snapshot_id={SNAPSHOT}&risk_band=high&limit=1"
        ).json().get("items", [])
        target = high[0]["order_id"] if high else order_id

        started = time.time()
        r = client.post(f"{api}/api/v1/investigations",
                        json={"order_id": target, "snapshot_id": SNAPSHOT}, timeout=180)
        elapsed = time.time() - started
        if r.status_code != 200:
            report.add("investigation completes", False,
                       f"HTTP {r.status_code}: {r.text[:160]}")
            return report

        inv = r.json()
        report.add("investigation completes", inv.get("status") == "completed",
                   f"{inv.get('status')} in {elapsed:.1f}s")
        report.add("report is grounded in cited evidence",
                   len(inv.get("evidence", [])) > 0 and len(inv.get("facts", [])) > 0,
                   f"{len(inv.get('facts', []))} facts, "
                   f"{len(inv.get('evidence', []))} evidence items")

        known = {e["evidence_id"] for e in inv.get("evidence", [])}
        cited = {i for f in inv.get("facts", []) for i in f.get("evidence_ids", [])}
        report.add("every cited evidence id is real", cited <= known,
                   f"unknown: {sorted(cited - known)}" if cited - known else "")
        report.add("investigation leaks no delivery outcome",
                   not any(m in str(inv.get("facts")) + str(inv.get("summary"))
                           for m in OUTCOME_MARKERS))

        proposal = inv.get("proposal")
        if not proposal:
            report.add("a proposal was offered", False,
                       f"recommendation was {inv.get('recommendation')}", critical=False)
            return report

        report.add("proposal starts pending, not approved",
                   proposal.get("status") == "pending")
        report.add("no ticket exists before approval",
                   client.get(f"{api}/api/v1/tickets").json().get("total") == 0)

        pid = proposal["proposal_id"]
        first = client.post(f"{api}/api/v1/proposals/{pid}/approve").json()
        report.add("approval creates a ticket", bool(first.get("ticket")),
                   (first.get("ticket") or {}).get("ticket_id", ""))

        second = client.post(f"{api}/api/v1/proposals/{pid}/approve").json()
        same = (first.get("ticket") or {}).get("ticket_id") == \
               (second.get("ticket") or {}).get("ticket_id")
        report.add("repeat approval is idempotent", same and second.get("already_decided") is True)

        listed = client.get(f"{api}/api/v1/tickets").json()
        report.add("exactly one ticket after a double approval",
                   listed.get("total") == 1, f"{listed.get('total')} tickets")

        with httpx.Client(timeout=60) as stranger:
            stranger.get(f"{api}/api/v1/snapshots")
            r = stranger.get(f"{api}/api/v1/tickets").json()
            report.add("another visitor cannot see these tickets", r.get("total") == 0)
            tid = (first.get("ticket") or {}).get("ticket_id")
            if tid:
                report.add("another visitor cannot fetch the ticket by id",
                           stranger.get(f"{api}/api/v1/tickets/{tid}").status_code == 404)

        audit = client.get(f"{api}/api/v1/audit").json()
        events = {e["event_type"] for e in audit}
        report.add("audit records the approval", "proposal_approved" in events,
                   ", ".join(sorted(events)))

    return report


def main(argv: list[str] | None = None) -> int:
    env = read_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=env.get("OPSPILOT_API_URL"))
    parser.add_argument("--web", default=env.get("OPSPILOT_WEB_URL"))
    parser.add_argument("--skip-investigation", action="store_true",
                        help="skip the LLM-backed journey to preserve free-tier quota")
    args = parser.parse_args(argv)

    if not args.api or not args.web:
        print("ERROR: set --api and --web, or run provisioning first.", file=sys.stderr)
        return 2

    report = verify(args.api, args.web, run_investigation=not args.skip_investigation)

    passed = sum(1 for c in report.checks if c.ok)
    print(f"\n{passed}/{len(report.checks)} checks passed")
    if report.failed_critical:
        print("\nCRITICAL FAILURES:", file=sys.stderr)
        for c in report.failed_critical:
            print(f"  - {c.name}: {c.detail}", file=sys.stderr)
        return 1
    print("DEPLOYMENT VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
