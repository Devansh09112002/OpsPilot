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

POLICY_PATH = Path(__file__).resolve().parents[3] / "policies" / "demo_policy_v2.json"

# Shared with the model's "high" band cut, so a band and the policy can never
# disagree. Stated on the CALIBRATED probability, so the number means what it
# says: v1 used 0.60 on the raw output, where an observed late rate of ~2% sat
# behind a 0.60 "score" - a threshold that read as strict and was not.
ESCALATION_RISK_THRESHOLD = spec.ESCALATION_THRESHOLD
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
    *, risk_probability: float, days_to_deadline: float, is_overdue: bool
) -> list[PolicySection]:
    """Deterministically select the sections that govern this order.

    The selection is code, not a model decision, so the agent cannot wander
    into a policy that does not apply.
    """
    ids: list[str] = []
    if is_overdue:
        ids.append("ESC-04")
    elif risk_probability >= ESCALATION_RISK_THRESHOLD:
        ids.append("ESC-01" if days_to_deadline <= ESCALATION_SLACK_DAYS else "ESC-02")
    else:
        ids.append("ESC-03")

    ids += ["EVI-01", "EVI-02", "EVI-03", "ACT-01", "ACT-02"]
    return [get_section(i) for i in ids]


def escalation_permitted(
    *, risk_probability: float, days_to_deadline: float, is_overdue: bool
) -> tuple[bool, str]:
    """The backend's own reading of the policy.

    Used to validate the agent's recommendation. If the agent proposes an
    escalation the policy does not support, the backend downgrades it rather
    than trusting the model.
    """
    if is_overdue:
        return False, (
            "ESC-04: the promised date has already passed, so this order belongs to "
            "the overdue-recovery process, not predictive escalation."
        )
    if risk_probability < ESCALATION_RISK_THRESHOLD:
        return False, (
            f"ESC-03: calibrated risk estimate {risk_probability:.3f} is below "
            f"the {ESCALATION_RISK_THRESHOLD:.2f} escalation threshold."
        )
    if days_to_deadline > ESCALATION_SLACK_DAYS:
        return False, (
            f"ESC-02: calibrated risk estimate {risk_probability:.3f} meets the "
            f"threshold but {days_to_deadline:.0f} days of slack remain, above "
            f"the {ESCALATION_SLACK_DAYS}-day limit, so the order is monitored."
        )
    return True, (
        f"ESC-01: calibrated risk estimate {risk_probability:.3f} is at or above "
        f"{ESCALATION_RISK_THRESHOLD:.2f} with {days_to_deadline:.0f} days "
        f"remaining, within the {ESCALATION_SLACK_DAYS}-day escalation window."
    )


# A lane needs a cluster of individually-escalatable orders, not one bad order
# that happens to share a route with others. ESC-05.
MIN_ESCALATABLE_MEMBERS = 3


def situation_applicable_sections(*, n_escalatable: int) -> list[PolicySection]:
    """Sections governing a lane situation. Selected in code, not by the model."""
    ids = ["ESC-05" if n_escalatable >= MIN_ESCALATABLE_MEMBERS else "ESC-03"]
    ids += ["ESC-01", "EVI-01", "EVI-04", "EVI-05", "ACT-01", "ACT-02"]
    return [get_section(i) for i in ids]


def situation_escalation_permitted(
    *, n_escalatable: int, n_flagged: int
) -> tuple[bool, str]:
    """The backend's own reading of ESC-05.

    Deliberately composed from the per-order rule: a member counts here only if
    `escalation_permitted` already returned True for it. That keeps exactly one
    risk threshold in the system - a second, lane-specific threshold would be
    another pair of numbers to drift apart, which is the bug the v1 band/policy
    split already taught us to avoid.
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
