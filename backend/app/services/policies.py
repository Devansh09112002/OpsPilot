"""Versioned demonstration policy lookup.

Deterministic retrieval over a handful of short sections. This is not a vector
store and is not described as semantic RAG: the sections are selected by the
order's own situation, and the exact source text is returned verbatim so the
agent can quote rather than paraphrase.

Policy text is treated as untrusted data. It is data the agent reasons over,
never instructions that can change what the agent is allowed to do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from data_pipeline import spec

POLICY_PATH = Path(__file__).resolve().parents[3] / "policies" / "demo_policy_v3.json"

# The one cut in the policy: the day's priority review list, which is also the
# "high" band. It is a rank because the ranking is the served model's
# validated output (docs/research_v4.md); its probability moves with network
# conditions, so a fixed probability threshold would escalate far more orders
# on a congested day than on a calm one.
REVIEW_LIST_SIZE = spec.REVIEW_CAPACITY_K
ESCALATION_SLACK_DAYS = 3


@dataclass(frozen=True)
class PolicySection:
    section_id: str
    title: str
    text: str

    def as_dict(self) -> dict:
        return {"section_id": self.section_id, "title": self.title, "text": self.text}


class PolicyUnavailableError(RuntimeError):
    """Raised when the policy document is missing or malformed."""


@lru_cache(maxsize=1)
def _load() -> tuple[str, dict[str, PolicySection]]:
    if not POLICY_PATH.exists():
        raise PolicyUnavailableError(f"policy document not found at {POLICY_PATH}")
    try:
        doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        sections = {
            s["section_id"]: PolicySection(s["section_id"], s["title"], s["text"])
            for s in doc["sections"]
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PolicyUnavailableError(f"policy document is malformed: {exc}") from exc
    if not sections:
        raise PolicyUnavailableError("policy document contains no sections")
    return doc["policy_version"], sections


def policy_version() -> str:
    return _load()[0]


def get_section(section_id: str) -> PolicySection:
    _, sections = _load()
    section = sections.get(section_id)
    if section is None:
        raise PolicyUnavailableError(
            f"policy section '{section_id}' does not exist in {policy_version()}"
        )
    return section


def all_sections() -> list[PolicySection]:
    return list(_load()[1].values())


def applicable_sections(
    *, in_review_list: bool, days_to_deadline: float, is_overdue: bool
) -> list[PolicySection]:
    """Deterministically select the sections that govern this order.

    The selection is code, not a model decision, so the agent cannot wander
    into a policy that does not apply.
    """
    ids: list[str] = []
    if is_overdue:
        ids.append("ESC-04")
    elif in_review_list:
        ids.append("ESC-01" if days_to_deadline <= ESCALATION_SLACK_DAYS else "ESC-02")
    else:
        ids.append("ESC-03")

    ids += ["EVI-01", "EVI-02", "EVI-03", "EVI-06", "ACT-01", "ACT-02"]
    return [get_section(i) for i in ids]


def escalation_permitted(
    *, in_review_list: bool, days_to_deadline: float, is_overdue: bool,
    priority_rank: int | None = None,
) -> tuple[bool, str]:
    """The backend's own reading of the policy.

    Used to validate the agent's recommendation. If the agent proposes an
    escalation the policy does not support, the backend downgrades it rather
    than trusting the model.
    """
    rank = f"priority rank {priority_rank}" if priority_rank else "the order"
    if is_overdue:
        return False, (
            "ESC-04: the promised date has already passed, so this order belongs to "
            "the overdue-recovery process, not predictive escalation."
        )
    if not in_review_list:
        return False, (
            f"ESC-03: {rank} is outside today's top-{REVIEW_LIST_SIZE} priority review "
            "list, so it is not escalated on the model's assessment alone."
        )
    if days_to_deadline > ESCALATION_SLACK_DAYS:
        return False, (
            f"ESC-02: {rank} is in today's top-{REVIEW_LIST_SIZE} review list but "
            f"{days_to_deadline:.0f} days of slack remain, above the "
            f"{ESCALATION_SLACK_DAYS}-day limit, so the order is monitored."
        )
    return True, (
        f"ESC-01: {rank} is in today's top-{REVIEW_LIST_SIZE} priority review list "
        f"with {days_to_deadline:.0f} days remaining, within the "
        f"{ESCALATION_SLACK_DAYS}-day escalation window."
    )


# A lane needs a cluster of individually-escalatable orders, not one bad order
# that happens to share a route with others. ESC-05.
MIN_ESCALATABLE_MEMBERS = 3


def situation_applicable_sections(*, n_escalatable: int) -> list[PolicySection]:
    """Sections governing a lane situation. Selected in code, not by the model."""
    ids = ["ESC-05" if n_escalatable >= MIN_ESCALATABLE_MEMBERS else "ESC-03"]
    ids += ["ESC-01", "EVI-01", "EVI-04", "EVI-05", "EVI-06", "ACT-01", "ACT-02"]
    return [get_section(i) for i in ids]


def situation_escalation_permitted(
    *, n_escalatable: int, n_flagged: int
) -> tuple[bool, str]:
    """The backend's own reading of ESC-05.

    Deliberately composed from the per-order rule: a member counts here only if
    `escalation_permitted` already returned True for it. That keeps exactly one
    cut in the system - a second, lane-specific one would be another number to
    drift apart from the first, which the v1 band/policy split taught us.
    """
    if n_escalatable >= MIN_ESCALATABLE_MEMBERS:
        return True, (
            f"ESC-05: {n_escalatable} of {n_flagged} flagged orders on this lane "
            f"each qualify independently under ESC-01, at or above the "
            f"{MIN_ESCALATABLE_MEMBERS}-order minimum for a lane escalation."
        )
    return False, (
        f"ESC-03/ESC-05: only {n_escalatable} of {n_flagged} flagged orders on "
        f"this lane qualify under ESC-01, below the {MIN_ESCALATABLE_MEMBERS}-order "
        "minimum, so the lane is monitored rather than escalated."
    )
