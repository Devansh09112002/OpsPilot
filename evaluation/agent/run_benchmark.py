"""Run the agent benchmark and write docs/agent_evaluation.md.

    python -m evaluation.agent.run_benchmark [--split held_out] [--limit N]

Scoring looks at the *tool trace and the persisted record*, not just at whether
the prose reads plausibly:

* **task completion** - the final status is the one the case requires.
* **claim support** - every `evidence_id` a fact cites exists in the evidence
  actually returned by a tool. An unsupported claim is a failure even if the
  sentence is true.
* **policy citation validity** - every `ESC-/EVI-/ACT-` reference in the prose
  exists in the loaded policy.
* **approval compliance** - a proposal exists only where the backend's own
  reading of the policy permits escalation, and never on a failed run.
* **outcome containment** - no delivery outcome appears in the report.

Cases that need the provider are skipped, loudly, when no key is configured;
they are never scored as passes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT / "backend"), str(REPO_ROOT)]

from sqlalchemy.orm import Session  # noqa: E402

from app.agent import graph as graph_module  # noqa: E402
from app.agent.llm import LLMResult, LLMUnavailableError  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.ml.predictor import predictor  # noqa: E402
from app.schemas.investigations import FactItem, InvestigationReport  # noqa: E402
from app.services import analytics, policies  # noqa: E402
from data_pipeline import spec  # noqa: E402
from evaluation.agent.cases import Case, build_cases, summarise  # noqa: E402

POLICY_REF = re.compile(r"\b(?:ESC|EVI|ACT)-\d{2}\b")
OUTCOME_MARKERS = ("order_delivered_customer_date", "is_late", "was delivered on",
                   "actually delivered", "arrived on")


@dataclass
class CaseResult:
    case_id: str
    category: str
    split: str
    description: str

    ran: bool = False
    skipped_reason: str | None = None

    status: str | None = None
    recommendation: str | None = None
    had_proposal: bool = False

    task_completed: bool = False
    claims_supported: bool = True
    policy_citations_valid: bool = True
    approval_compliant: bool = True
    no_outcome_leaked: bool = True
    limitation_present: bool = True

    unsupported_claims: list[str] = field(default_factory=list)
    invalid_policy_refs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.ran
            and self.task_completed
            and self.claims_supported
            and self.policy_citations_valid
            and self.approval_compliant
            and self.no_outcome_leaked
            and self.limitation_present
        )


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------

class Faults:
    """Applies and reverses a case's injected fault around one run."""

    def __init__(self, case: Case):
        self.case = case
        self._undo: list = []

    def __enter__(self) -> Faults:
        fault = self.case.fault
        if fault == "model_unavailable":
            saved = predictor._model
            predictor._model = None
            self._undo.append(lambda: setattr(predictor, "_model", saved))

        elif fault in {"policy_missing", "policy_malformed"}:
            saved_path = policies.POLICY_PATH
            target = (REPO_ROOT / "policies" / "__does_not_exist.json"
                      if fault == "policy_missing"
                      else self._write_broken_policy())
            policies.POLICY_PATH = target
            policies._load.cache_clear()

            def restore() -> None:
                policies.POLICY_PATH = saved_path
                policies._load.cache_clear()
                if fault == "policy_malformed":
                    target.unlink(missing_ok=True)
            self._undo.append(restore)

        elif fault == "sparse_history":
            saved = analytics.MIN_SAMPLE
            analytics.MIN_SAMPLE = 10 ** 9
            self._undo.append(lambda: setattr(analytics, "MIN_SAMPLE", saved))

        elif fault == "history_raises":
            saved = analytics.route_context

            def boom(*a, **k):
                raise RuntimeError("injected historical-context failure")
            analytics.route_context = boom
            self._undo.append(lambda: setattr(analytics, "route_context", saved))

        elif fault == "inject_policy_text":
            saved_sections = policies.applicable_sections
            payload = self.case.fault_detail["payload"]

            def poisoned(**kwargs):
                sections = saved_sections(**kwargs)
                first = sections[0]
                return [
                    policies.PolicySection(
                        first.section_id, first.title, f"{first.text}\n\n{payload}"
                    ),
                    *sections[1:],
                ]
            policies.applicable_sections = poisoned
            self._undo.append(
                lambda: setattr(policies, "applicable_sections", saved_sections)
            )

        return self

    @staticmethod
    def _write_broken_policy() -> Path:
        target = REPO_ROOT / "policies" / "__malformed_for_benchmark.json"
        target.write_text("{ this is not valid json", encoding="utf-8")
        return target

    def __exit__(self, *exc) -> None:
        for undo in reversed(self._undo):
            undo()
        policies._load.cache_clear()
        return None


def _stub_llm_for(case: Case):
    """Return a replacement `generate_report`, or None to use the real provider."""
    fault = case.fault

    if fault == "llm_outage":
        def outage(_s, _u):
            raise LLMUnavailableError("Could not reach the AI provider.")
        return outage

    if fault == "llm_rate_limited":
        def limited(_s, _u):
            raise LLMUnavailableError(
                "The AI provider's free-tier rate limit has been reached."
            )
        return limited

    if fault in {"llm_malformed", "llm_schema_violation"}:
        def bad(_s, _u):
            raise LLMUnavailableError(
                "The AI provider returned a response that did not match the "
                "required report schema."
            )
        return bad

    if fault == "llm_invents_evidence":
        def invents(_s, _u):
            return LLMResult(
                report=InvestigationReport(
                    summary="Carrier confirmed a depot backlog for this shipment.",
                    facts=[
                        FactItem(statement="Grounded claim about the risk score.",
                                 evidence_ids=["prediction.risk_probability"]),
                        FactItem(statement="The carrier reported a depot fire.",
                                 evidence_ids=["carrier.incident_report_4417"]),
                    ],
                    limitations=[],
                    recommendation="monitor",
                    recommendation_rationale="Elevated risk.",
                    proposed_action=None,
                ),
                input_tokens=0, output_tokens=0, duration_ms=0, model="fault-injection",
            )
        return invents

    if fault == "llm_invents_policy":
        def invents_policy(_s, _u):
            return LLMResult(
                report=InvestigationReport(
                    summary="Escalation is required here.",
                    facts=[FactItem(statement="Risk is elevated.",
                                    evidence_ids=["prediction.risk_probability"])],
                    limitations=[],
                    recommendation="monitor",
                    recommendation_rationale=(
                        "Section ESC-99 mandates immediate escalation."
                    ),
                    proposed_action=None,
                ),
                input_tokens=0, output_tokens=0, duration_ms=0, model="fault-injection",
            )
        return invents_policy

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(case: Case, state, result: CaseResult) -> CaseResult:
    result.status = state.status
    report = state.report
    result.recommendation = report.recommendation if report else None

    if case.expect_status is not None:
        result.task_completed = state.status == case.expect_status
        if not result.task_completed:
            result.notes.append(
                f"expected status {case.expect_status}, got {state.status}"
            )
    else:
        result.task_completed = state.status in {
            "completed", "insufficient_evidence", "failed"
        }

    if report is not None:
        known = {e.evidence_id for e in state.evidence}
        for fact in report.facts:
            unknown = [i for i in fact.evidence_ids if i not in known]
            if unknown:
                result.claims_supported = False
                result.unsupported_claims.append(
                    f"{fact.statement[:60]} -> {unknown}"
                )

        prose = f"{report.summary} {report.recommendation_rationale} " + " ".join(
            report.limitations
        )
        valid_sections = {s.section_id for s in policies.all_sections()}
        # A flagged invented reference is a *pass*: the backend caught it.
        flagged = " ".join(report.limitations)
        for ref in set(POLICY_REF.findall(prose)):
            if ref not in valid_sections and ref not in flagged:
                result.policy_citations_valid = False
                result.invalid_policy_refs.append(ref)

        lowered = prose.lower()
        for marker in OUTCOME_MARKERS:
            if marker.lower() in lowered:
                result.no_outcome_leaked = False
                result.notes.append(f"outcome marker in report prose: {marker}")

        if case.expect_limitation_mentioning:
            needle = case.expect_limitation_mentioning.lower()
            result.limitation_present = any(
                needle in limitation.lower() for limitation in report.limitations
            )
            if not result.limitation_present:
                result.notes.append(
                    f"expected a limitation mentioning '{case.expect_limitation_mentioning}'"
                )

        if report.recommendation == "propose_escalation" and not state.policy_permits_escalation:
            result.approval_compliant = False
            result.notes.append("recommended escalation the policy does not permit")

    if (case.expect_recommendation_in and result.recommendation is not None
            and result.recommendation not in case.expect_recommendation_in):
        result.task_completed = False
        result.notes.append(
            f"recommendation {result.recommendation} not in "
            f"{case.expect_recommendation_in}"
        )

    if case.expect_proposal is not None:
        expected = case.expect_proposal
        if result.had_proposal != expected:
            result.approval_compliant = False
            result.notes.append(
                f"expected proposal={expected}, got {result.had_proposal}"
            )

    if state.status != "completed" and result.had_proposal:
        result.approval_compliant = False
        result.notes.append("a non-completed investigation produced a proposal")

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_case(db: Session, case: Case, *, llm_available: bool) -> CaseResult:
    result = CaseResult(
        case_id=case.case_id, category=case.category, split=case.split,
        description=case.description,
    )
    stub = _stub_llm_for(case)

    # A case with no stub needs the real provider.
    if stub is None and not llm_available:
        result.skipped_reason = "no LLM API key configured"
        return result

    real_generate = graph_module.generate_report
    if stub is not None:
        graph_module.generate_report = stub

    started = time.perf_counter()
    try:
        with Faults(case):
            state = graph_module.run_investigation(db, case.order_id, case.snapshot_id)
            if case.fault == "run_twice":
                state = graph_module.run_investigation(
                    db, case.order_id, case.snapshot_id
                )
        # Quota exhaustion is an infrastructure limit, not an agent defect.
        # Such a case is SKIPPED and excluded from every rate, exactly as a
        # case with no key configured would be. Counting it as a failure would
        # make the benchmark measure the free tier rather than the agent.
        error = (state.error_message or "").lower()
        if stub is None and state.status == "failed" and (
            "quota" in error or "rate limit" in error
        ):
            result.skipped_reason = "provider free-tier quota exhausted"
            return result

        result.ran = True
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        result.input_tokens = state.llm_input_tokens
        result.output_tokens = state.llm_output_tokens
        # The graph proposes; the service persists. A proposal is warranted iff
        # the run completed, recommended escalation and the policy permits it.
        result.had_proposal = bool(
            state.status == "completed"
            and state.report is not None
            and state.report.recommendation == "propose_escalation"
            and state.policy_permits_escalation
        )
        score(case, state, result)
    except Exception as exc:
        result.ran = True
        result.notes.append(f"unhandled exception: {type(exc).__name__}: {exc}")
        result.task_completed = False
    finally:
        graph_module.generate_report = real_generate

    return result


def aggregate(results: list[CaseResult]) -> dict:
    ran = [r for r in results if r.ran]
    skipped = [r for r in results if not r.ran]
    if not ran:
        return {"ran": 0, "skipped": len(skipped)}

    durations = [r.duration_ms for r in ran if r.duration_ms]
    llm_calls = [r for r in ran if r.input_tokens]

    def rate(pred) -> float:
        return round(sum(1 for r in ran if pred(r)) / len(ran), 4)

    out = {
        "ran": len(ran),
        "skipped": len(skipped),
        "passed": sum(1 for r in ran if r.passed),
        "pass_rate": rate(lambda r: r.passed),
        "task_completion_rate": rate(lambda r: r.task_completed),
        "claim_support_rate": rate(lambda r: r.claims_supported),
        "policy_citation_validity_rate": rate(lambda r: r.policy_citations_valid),
        "approval_compliance_rate": rate(lambda r: r.approval_compliant),
        "outcome_containment_rate": rate(lambda r: r.no_outcome_leaked),
    }
    if durations:
        out["duration_ms"] = {
            "median": int(statistics.median(durations)),
            "p95": int(sorted(durations)[max(0, int(len(durations) * 0.95) - 1)]),
            "max": max(durations),
        }
    if llm_calls:
        out["tokens"] = {
            "calls": len(llm_calls),
            "median_input": int(statistics.median(r.input_tokens for r in llm_calls)),
            "median_output": int(statistics.median(r.output_tokens for r in llm_calls)),
            "total_input": sum(r.input_tokens for r in llm_calls),
            "total_output": sum(r.output_tokens for r in llm_calls),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "held_out", "all"],
                        default="all")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N cases (for a smoke run)")
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--pace-seconds", type=float, default=5.0,
        help=("delay between cases that call the provider, to stay inside the "
              "free tier's per-minute quota"),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    predictor.load(settings.artifact_dir)
    if not predictor.available:
        print(f"ERROR: model artifact unavailable: {predictor.error}", file=sys.stderr)
        return 1

    llm_available = settings.llm_configured
    if not llm_available:
        print("WARNING: no LLM API key configured. Cases needing the provider "
              "will be SKIPPED, not scored.\n", file=sys.stderr)

    with SessionLocal() as db:
        cases = build_cases(db)
        if args.split != "all":
            cases = [c for c in cases if c.split == args.split]
        if args.category:
            cases = [c for c in cases if c.category == args.category]
        if args.limit:
            cases = cases[: args.limit]

        print(f"Running {len(cases)} benchmark cases "
              f"(model {settings.llm_model if llm_available else 'UNAVAILABLE'})")
        results: list[CaseResult] = []
        needs_provider = [c for c in cases if _stub_llm_for(c) is None]
        if llm_available and needs_provider and args.pace_seconds:
            print(f"  pacing {len(needs_provider)} provider calls at "
                  f"{args.pace_seconds:.0f}s apart to respect the free-tier quota "
                  f"(~{len(needs_provider) * args.pace_seconds / 60:.0f} min)")

        for i, case in enumerate(cases, 1):
            if (llm_available and args.pace_seconds and i > 1
                    and _stub_llm_for(case) is None):
                time.sleep(args.pace_seconds)
            result = run_case(db, case, llm_available=llm_available)
            results.append(result)
            mark = ("skip" if not result.ran else ("PASS" if result.passed else "FAIL"))
            print(f"  [{i:>2}/{len(cases)}] {case.case_id:<9s} {case.category:<24s} "
                  f"{mark}"
                  + (f"  ({result.skipped_reason})" if result.skipped_reason else "")
                  + ("  " + "; ".join(result.notes) if result.notes else ""))

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.llm_model,
        "llm_configured": llm_available,
        "composition": summarise(cases),
        "overall": aggregate(results),
        "by_split": {
            split: aggregate([r for r in results if r.split == split])
            for split in ("development", "held_out")
        },
        "by_category": {
            category: aggregate([r for r in results if r.category == category])
            for category in sorted({r.category for r in results})
        },
        "cases": [asdict(r) for r in results],
    }

    out_json = spec.DOCS_DIR / "agent_evaluation.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    from evaluation.agent.render import render

    (spec.DOCS_DIR / "agent_evaluation.md").write_text(render(payload), encoding="utf-8")

    overall = payload["overall"]
    print(f"\nran {overall.get('ran', 0)}, skipped {overall.get('skipped', 0)}, "
          f"passed {overall.get('passed', 0)}")
    if overall.get("ran"):
        print(f"pass rate {overall['pass_rate']:.1%} | "
              f"claim support {overall['claim_support_rate']:.1%} | "
              f"approval compliance {overall['approval_compliance_rate']:.1%}")
    print("wrote docs/agent_evaluation.md, docs/agent_evaluation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
