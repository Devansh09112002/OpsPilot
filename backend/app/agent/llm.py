"""Anthropic client wrapper for the investigation synthesis step.

The LLM is asked for exactly one thing: turn verified tool output into a
structured report. It has no tool-calling authority here, because retrieval is
already done deterministically by `agent.tools`. That keeps the failure surface
small and the cost to one bounded call.

Structured output is enforced with `messages.parse()` against a Pydantic model,
so a malformed response is a caught exception rather than free-text the backend
would have to guess at.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.investigations import InvestigationReport

log = get_logger(__name__)


class LLMUnavailableError(RuntimeError):
    """The provider could not produce a usable report. Never fabricate one."""


@dataclass
class LLMResult:
    report: InvestigationReport
    input_tokens: int
    output_tokens: int
    duration_ms: int
    model: str


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    settings = get_settings()
    if not settings.llm_configured:
        raise LLMUnavailableError(
            "No LLM API key is configured, so AI investigations are disabled. "
            "Order data and model predictions remain available."
        )
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return _client


def reset_client() -> None:
    """Drop the cached client. Used by tests that swap settings."""
    global _client
    _client = None


def generate_report(system_prompt: str, user_prompt: str) -> LLMResult:
    """One bounded call returning a validated `InvestigationReport`.

    Raises `LLMUnavailableError` on any provider failure. The caller records an
    honest failed/insufficient-evidence investigation; it never invents output.
    """
    settings = get_settings()
    client = get_client()
    started = time.perf_counter()

    try:
        response = client.messages.parse(
            model=settings.llm_model,
            max_tokens=settings.llm_max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=InvestigationReport,
        )
    except APITimeoutError as exc:
        raise LLMUnavailableError(
            f"The AI provider did not respond within "
            f"{settings.llm_timeout_seconds:.0f}s."
        ) from exc
    except RateLimitError as exc:
        raise LLMUnavailableError(
            "The AI provider is rate limiting OpsPilot. Please retry shortly."
        ) from exc
    except APIConnectionError as exc:
        raise LLMUnavailableError("Could not reach the AI provider.") from exc
    except APIStatusError as exc:
        # Do not surface provider internals (which can echo the key) to the client.
        log.error("llm_status_error", status=exc.status_code)
        raise LLMUnavailableError(
            f"The AI provider returned an error (HTTP {exc.status_code})."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - includes schema validation failures
        raise LLMUnavailableError(
            f"The AI provider returned an unusable response: {type(exc).__name__}"
        ) from exc

    if getattr(response, "stop_reason", None) == "refusal":
        raise LLMUnavailableError(
            "The AI provider declined to complete this investigation."
        )

    report = getattr(response, "parsed_output", None)
    if report is None:
        raise LLMUnavailableError(
            "The AI provider returned no structured report for this investigation."
        )

    usage = getattr(response, "usage", None)
    return LLMResult(
        report=report,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        duration_ms=int((time.perf_counter() - started) * 1000),
        model=settings.llm_model,
    )
