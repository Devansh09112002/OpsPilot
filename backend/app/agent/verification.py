"""The safety gate, shared by every investigation graph.

This is the one place that decides what a generated report is allowed to
claim, and it lives apart from the graphs on purpose: an order investigation
and a situation investigation must be verified by *the same* code. Two copies
of a safety check are two checks that drift.

Every rule here can only ever *reduce* what a report asserts. Nothing in this
module can add a claim, raise a recommendation, or authorise an action.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.schemas.investigations import FactItem, InvestigationReport
from app.services import policies

log = get_logger(__name__)

_POLICY_REF = re.compile(r"\b(?:ESC|EVI|ACT)-\d{2}\b")
# Olist order ids are 32 lowercase hex characters. Used to catch a report that
# cites an order outside the subject it was asked about.
_ORDER_ID = re.compile(r"\b[0-9a-f]{32}\b")


def _prose(report: InvestigationReport) -> str:
    return " ".join(
        [report.summary, report.recommendation_rationale, *(f.statement for f in report.facts)]
    )


def verify_report(
    *,
    report: InvestigationReport,
    known_evidence_ids: set[str],
    policy_permits_escalation: bool,
    policy_determination: str | None,
    subject: str,
    allowed_order_ids: set[str] | None = None,
) -> InvestigationReport:
    """Return the report a user is allowed to see.

    `allowed_order_ids`, when given, is the membership boundary: a situation
    report may discuss only the orders in that situation. A report that cites
    an order from somewhere else is not merely wrong, it is evidence the model
    invented an identifier, so those statements are removed.
    """
    extra_limitations: list[str] = []

    # 1. Every cited evidence id must have come from a tool.
    kept: list[FactItem] = []
    dropped_unknown = 0
    dropped_foreign = 0
    for fact in report.facts:
        unknown = [i for i in fact.evidence_ids if i not in known_evidence_ids]
        if unknown:
            dropped_unknown += 1
            log.warning(
                "investigation_fact_rejected", subject=subject, unknown_evidence_ids=unknown
            )
            continue
        # 2. A fact may not name an order outside the subject.
        if allowed_order_ids is not None:
            foreign = sorted(set(_ORDER_ID.findall(fact.statement)) - allowed_order_ids)
            if foreign:
                dropped_foreign += 1
                log.warning(
                    "investigation_foreign_order_cited",
                    subject=subject,
                    foreign_order_ids=foreign[:5],
                )
                continue
        kept.append(fact)

    if dropped_unknown:
        extra_limitations.append(
            f"{dropped_unknown} generated statement(s) cited evidence that does not "
            "exist and were removed by the backend before display."
        )
    if dropped_foreign:
        extra_limitations.append(
            f"{dropped_foreign} generated statement(s) referred to orders that are not "
            "part of this situation and were removed by the backend."
        )

    # The summary and rationale are prose, not claims tied to evidence, so they
    # are flagged rather than dropped - removing them would leave an empty
    # report where a caveat is more useful.
    if allowed_order_ids is not None:
        stray = sorted(
            set(_ORDER_ID.findall(f"{report.summary} {report.recommendation_rationale}"))
            - allowed_order_ids
        )
        if stray:
            extra_limitations.append(
                "The assessment text mentions order identifier(s) that are not part "
                f"of this situation ({', '.join(stray[:3])}); disregard them."
            )
            log.warning(
                "investigation_foreign_order_in_prose", subject=subject, foreign_order_ids=stray[:5]
            )

    # 3. Every policy reference must exist. An unreadable policy is a
    #    limitation, not an exception: the investigation still returns.
    try:
        valid_sections = {s.section_id for s in policies.all_sections()}
        policy_readable = True
    except policies.PolicyUnavailableError:
        valid_sections = set()
        policy_readable = False

    mentioned = set(_POLICY_REF.findall(_prose(report)))
    if not policy_readable:
        extra_limitations.append(
            "The demonstration policy document could not be read, so no policy "
            "citation in this report could be verified and no escalation is proposed."
        )
        log.warning("investigation_policy_unreadable", subject=subject)
    elif invented := (mentioned - valid_sections):
        extra_limitations.append(
            f"Reference(s) to non-existent policy section(s) {sorted(invented)} were "
            "flagged by the backend and should be disregarded."
        )
        log.warning("investigation_invented_policy", sections=sorted(invented))

    # 4. The recommendation may not exceed the backend's own policy reading.
    recommendation = report.recommendation
    if recommendation == "propose_escalation" and not (
        policy_permits_escalation and policy_readable
    ):
        recommendation = "monitor"
        extra_limitations.append(
            "The generated recommendation to escalate was overridden by the backend "
            f"because the demo policy does not permit it here. {policy_determination or ''}".strip()
        )
        log.warning("investigation_recommendation_downgraded", subject=subject)

    return InvestigationReport(
        summary=report.summary,
        facts=kept,
        limitations=[*report.limitations, *extra_limitations],
        recommendation=recommendation,
        recommendation_rationale=report.recommendation_rationale,
        proposed_action=(
            report.proposed_action if recommendation == "propose_escalation" else None
        ),
    )
