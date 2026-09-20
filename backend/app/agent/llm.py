"""Google Gemini client for the investigation synthesis step.

Provider choice: Gemini's free tier is the only one available to this project
at zero budget, which is why this is not the Anthropic SDK. The rest of the
agent is provider-agnostic: retrieval, evidence verification and the approval
gate are all deterministic backend code, so swapping providers means changing
this file only.

The LLM is asked for exactly one thing: turn already-verified tool output into
a structured report. It has no tool-calling authority, because retrieval has
already happened in `agent.tools`. That keeps the failure surface small, the
latency bounded and the free-tier quota consumption to one call per
investigation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.investigations import InvestigationReport

log = get_logger(__name__)

# Gemini returns a finish reason; only STOP means we have a usable answer.
_BAD_FINISH = {
    "SAFETY": "the provider's safety filter blocked the response",
    "RECITATION": "the provider blocked the response as recitation",
    "MAX_TOKENS": "the response hit the output token limit before completing",
    "PROHIBITED_CONTENT": "the provider blocked the response as prohibited content",
    "BLOCKLIST": "the provider blocked the response",
    "MALFORMED_FUNCTION_CALL": "the provider returned a malformed call",
}


class LLMUnavailableError(RuntimeError):
    """The provider could not produce a usable report. Never fabricate one."""


@dataclass
class LLMResult:
    report: InvestigationReport
    input_tokens: int
    output_tokens: int
    duration_ms: int
    model: str


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    settings = get_settings()
    if not settings.llm_configured:
        raise LLMUnavailableError(
            "No LLM API key is configured, so AI investigations are disabled. "
            "Order data and model predictions remain available."
        )
    if _client is None:
        _client = genai.Client(
            api_key=settings.llm_api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),  # milliseconds
            ),
        )
    return _client


# Models that answered 400 to a thinking budget. Discovering this costs a
# wasted request, and on a 20-request daily quota paying that on every call
# would halve the usable budget - so it is remembered for the process
# lifetime and the budget is simply not sent to those models again.
_NO_THINKING: set[str] = set()


def reset_client() -> None:
    """Drop the cached client. Used by tests that swap settings."""
    global _client
    _client = None
    _NO_THINKING.clear()


def _build_config(settings) -> genai_types.GenerateContentConfig:
    thinking = None
    if settings.llm_thinking_budget >= 0:
        # Budget 0 disables thinking, which keeps latency and free-tier quota
        # predictable. This task is bounded synthesis over verified evidence,
        # not open-ended reasoning.
        thinking = genai_types.ThinkingConfig(
            thinking_budget=settings.llm_thinking_budget
        )
    return genai_types.GenerateContentConfig(
        system_instruction=None,  # set per call
        response_mime_type="application/json",
        response_schema=InvestigationReport,
        max_output_tokens=settings.llm_max_output_tokens,
        temperature=settings.llm_temperature,
        thinking_config=thinking,
    )


def _extract_report(response) -> InvestigationReport:
    """Get a validated report out of the response, or raise.

    `response.parsed` is the SDK's own validation. It can be None when the
    model returns JSON the schema converter could not round-trip, so the raw
    text is validated as a fallback. Both paths end in a real
    `InvestigationReport` or an exception - never a partially-filled object.
    """
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, InvestigationReport):
        return parsed
    if isinstance(parsed, dict):
        return InvestigationReport.model_validate(parsed)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise LLMUnavailableError(
            "The AI provider returned an empty response for this investigation."
        )
    # Some models wrap JSON in a fenced block despite the mime type.
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    try:
        return InvestigationReport.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMUnavailableError(
            f"The AI provider returned a response that did not match the required "
            f"report schema ({type(exc).__name__})."
        ) from exc


def _check_blocked(response) -> None:
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None) if feedback else None
    if block_reason:
        raise LLMUnavailableError(
            f"The AI provider declined this request ({block_reason})."
        )

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise LLMUnavailableError("The AI provider returned no candidate response.")

    finish = getattr(candidates[0], "finish_reason", None)
    name = getattr(finish, "name", str(finish) if finish else "")
    if name and name.upper() in _BAD_FINISH:
        raise LLMUnavailableError(
            f"The investigation could not be completed: "
            f"{_BAD_FINISH[name.upper()]}."
        )


class _TryNext(Exception):
    """This model cannot serve the request; another in the chain may.

    `reason` distinguishes a daily-quota exhaustion from a transient provider
    fault, so the message shown when the whole chain fails is accurate rather
    than always blaming quota.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _call_one(client, model: str, config, user_prompt: str, *, allow_retry=True):
    """One attempt against one model. Raises `_TryNext` on 429/404/5xx."""
    if model in _NO_THINKING and config.thinking_config is not None:
        config = config.model_copy(update={"thinking_config": None})
        allow_retry = False
    try:
        return client.models.generate_content(
            model=model, contents=user_prompt, config=config
        )
    except genai_errors.ClientError as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        # Models differ in which optional knobs they accept - gemini-3.5-flash-lite
        # rejects a thinking budget outright. Retry once without it before
        # giving up on an otherwise working model.
        if status == 400 and allow_retry and config.thinking_config is not None:
            log.info("llm_retry_without_thinking", model=model)
            _NO_THINKING.add(model)
            retry = config.model_copy(update={"thinking_config": None})
            return _call_one(client, model, retry, user_prompt, allow_retry=False)
        if status == 429:
            # Free-tier quota is per model per day, so the next model in the
            # chain is very likely to have budget left.
            log.warning("llm_model_quota_exhausted", model=model)
            raise _TryNext("quota") from exc
        if status in (401, 403):
            raise LLMUnavailableError(
                "The AI provider rejected OpsPilot's credentials."
            ) from exc
        if status == 404:
            # A retired or mistyped model name; try the next one.
            log.warning("llm_model_unavailable", model=model)
            raise _TryNext("retired") from exc
        # Never surface provider text: it can echo the request or the key.
        log.error("llm_client_error", status=status, model=model)
        raise LLMUnavailableError(
            f"The AI provider rejected the request (HTTP {status})."
        ) from exc
    except genai_errors.ServerError as exc:
        log.warning("llm_server_error", error=type(exc).__name__, model=model)
        raise _TryNext("server_error") from exc
    except genai_errors.APIError as exc:
        log.error("llm_api_error", error=type(exc).__name__, model=model)
        raise LLMUnavailableError("The AI provider returned an error.") from exc


def generate_report(system_prompt: str, user_prompt: str) -> LLMResult:
    """One bounded investigation, across a chain of interchangeable models.

    Gemini's free tier caps requests *per model per day* (measured: 20/day for
    gemini-3.5-flash). Rather than let one exhausted model take the whole demo
    down, this walks `settings.llm_model_chain` and uses the first model with
    budget left. Every model in the chain is the same flash tier producing the
    same structured report, so which one answered is recorded rather than
    hidden - `LLMResult.model` carries it, and the UI shows it.

    Raises `LLMUnavailableError` when the whole chain is exhausted. The caller
    records an honest failed investigation; it never invents output.
    """
    settings = get_settings()
    client = get_client()
    config = _build_config(settings)
    config.system_instruction = system_prompt

    started = time.perf_counter()
    chain = settings.llm_model_chain
    response = None
    used_model = ""
    reasons: list[str] = []

    # A total deadline across the chain, not just per call. A timeout already
    # aborts immediately rather than walking the chain, but several models
    # failing slowly would still hold the single free-tier worker for minutes
    # and block every other visitor. Two call-timeouts is the whole budget.
    total_budget = settings.llm_timeout_seconds * 2

    for model in chain:
        if time.perf_counter() - started > total_budget:
            reasons.append("deadline")
            log.warning(
                "llm_chain_deadline_reached",
                elapsed_s=round(time.perf_counter() - started, 1),
                tried=len(reasons),
            )
            break
        try:
            response = _call_one(client, model, config, user_prompt)
            used_model = model
            break
        except _TryNext as skip:
            reasons.append(skip.reason)
            continue
        except LLMUnavailableError:
            raise
        except Exception as exc:  # transport failures, timeouts
            log.error("llm_transport_error", error=type(exc).__name__, model=model)
            raise LLMUnavailableError(
                f"Could not reach the AI provider within "
                f"{settings.llm_timeout_seconds:.0f}s."
            ) from exc

    if response is None:
        # Report what actually happened. Calling a transient 503 a quota
        # problem would send a visitor away for a day over a blip.
        if reasons and all(r == "quota" for r in reasons):
            raise LLMUnavailableError(
                "Every configured AI model has exhausted its free-tier quota "
                "for today. The risk queue and model predictions are "
                "unaffected; investigations resume when the quota resets."
            )
        raise LLMUnavailableError(
            "The AI provider is temporarily unavailable across every "
            "configured model. Please retry shortly; the risk queue and model "
            "predictions are unaffected."
        )

    _check_blocked(response)
    report = _extract_report(response)

    if used_model != chain[0]:
        log.info("llm_served_by_fallback", model=used_model, primary=chain[0])

    usage = getattr(response, "usage_metadata", None)
    return LLMResult(
        report=report,
        input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        duration_ms=int((time.perf_counter() - started) * 1000),
        model=used_model,
    )
