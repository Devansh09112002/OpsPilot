"""Gemini provider-integration tests.

These exercise the response-handling paths without calling the API, so the
provider layer is covered before a key exists and remains covered in CI, where
no key is configured. They assert the property that matters most: every
failure mode produces an honest `LLMUnavailableError`, never a partial or
invented report, and never leaks the API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.agent import llm as llm_module
from app.agent.llm import (
    LLMUnavailableError,
    _check_blocked,
    _extract_report,
    generate_report,
)
from app.schemas.investigations import InvestigationReport

VALID = {
    "summary": "Elevated risk with little slack remaining.",
    "facts": [{"statement": "Model scored it high.",
               "evidence_ids": ["prediction.risk_probability"]}],
    "limitations": [],
    "recommendation": "monitor",
    "recommendation_rationale": "Slack remains.",
    "proposed_action": None,
}


def _response(**kwargs):
    base = {
        "parsed": None,
        "text": None,
        "prompt_feedback": None,
        "candidates": [SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        "usage_metadata": SimpleNamespace(prompt_token_count=100,
                                          candidates_token_count=50),
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Structured-output extraction
# ---------------------------------------------------------------------------

def test_parsed_model_is_used_directly():
    report = InvestigationReport.model_validate(VALID)
    assert _extract_report(_response(parsed=report)) is report


def test_parsed_dict_is_validated():
    out = _extract_report(_response(parsed=dict(VALID)))
    assert isinstance(out, InvestigationReport)
    assert out.recommendation == "monitor"


def test_raw_json_text_is_validated_when_parsed_is_none():
    """The SDK can return parsed=None; raw text must still be usable."""
    out = _extract_report(_response(text=json.dumps(VALID)))
    assert out.summary == VALID["summary"]


def test_fenced_json_is_recovered():
    fenced = f"```json\n{json.dumps(VALID)}\n```"
    out = _extract_report(_response(text=fenced))
    assert out.recommendation == "monitor"


def test_empty_response_raises_rather_than_returning_a_blank_report():
    with pytest.raises(LLMUnavailableError, match="empty response"):
        _extract_report(_response(text="   "))


def test_malformed_json_raises():
    with pytest.raises(LLMUnavailableError, match="did not match the required"):
        _extract_report(_response(text="{not json at all"))


def test_schema_violation_raises_instead_of_being_coerced():
    """A response missing a required field must not become a partial report."""
    bad = dict(VALID)
    del bad["recommendation"]
    with pytest.raises(LLMUnavailableError, match="did not match the required"):
        _extract_report(_response(text=json.dumps(bad)))


def test_invalid_recommendation_value_is_rejected():
    bad = dict(VALID, recommendation="delete_the_order")
    with pytest.raises(LLMUnavailableError):
        _extract_report(_response(text=json.dumps(bad)))


# ---------------------------------------------------------------------------
# Blocked / truncated responses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reason,fragment",
    [("SAFETY", "safety filter"),
     ("MAX_TOKENS", "output token limit"),
     ("RECITATION", "recitation"),
     ("PROHIBITED_CONTENT", "prohibited")],
)
def test_bad_finish_reasons_are_honest_failures(reason, fragment):
    response = _response(
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=reason))]
    )
    with pytest.raises(LLMUnavailableError, match=fragment):
        _check_blocked(response)


def test_prompt_level_block_is_reported():
    response = _response(prompt_feedback=SimpleNamespace(block_reason="SAFETY"))
    with pytest.raises(LLMUnavailableError, match="declined"):
        _check_blocked(response)


def test_no_candidates_is_reported():
    with pytest.raises(LLMUnavailableError, match="no candidate"):
        _check_blocked(_response(candidates=[]))


def test_stop_finish_reason_passes():
    _check_blocked(_response())  # must not raise


# ---------------------------------------------------------------------------
# Transport failures
# ---------------------------------------------------------------------------

@pytest.fixture
def with_key(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "llm_api_key", "test-key-not-real", raising=False)
    llm_module.reset_client()
    yield settings
    llm_module.reset_client()


def _client_error(status: int) -> genai_errors.ClientError:
    exc = genai_errors.ClientError.__new__(genai_errors.ClientError)
    Exception.__init__(exc, f"HTTP {status}")
    exc.code = status
    return exc


def test_rate_limit_produces_an_actionable_message(with_key, monkeypatch):
    def boom(*a, **k):
        raise _client_error(429)
    monkeypatch.setattr(llm_module.genai.Client, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(llm_module, "get_client",
                        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=boom)))

    with pytest.raises(LLMUnavailableError, match="free-tier rate limit"):
        generate_report("sys", "user")


def test_bad_credentials_do_not_echo_the_key(with_key, monkeypatch):
    def boom(*a, **k):
        raise _client_error(403)
    monkeypatch.setattr(llm_module, "get_client",
                        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=boom)))

    with pytest.raises(LLMUnavailableError) as excinfo:
        generate_report("sys", "user")
    assert "test-key-not-real" not in str(excinfo.value)
    assert "credentials" in str(excinfo.value)


def test_server_error_is_transient_and_actionable(with_key, monkeypatch):
    exc = genai_errors.ServerError.__new__(genai_errors.ServerError)
    Exception.__init__(exc, "503")

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(llm_module, "get_client",
                        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=boom)))

    with pytest.raises(LLMUnavailableError, match="temporarily unavailable"):
        generate_report("sys", "user")


def test_timeout_is_reported_with_the_configured_budget(with_key, monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("read timeout")
    monkeypatch.setattr(llm_module, "get_client",
                        lambda: SimpleNamespace(models=SimpleNamespace(generate_content=boom)))

    with pytest.raises(LLMUnavailableError, match="Could not reach the AI provider"):
        generate_report("sys", "user")


def test_missing_key_disables_investigations_cleanly(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "llm_api_key", "", raising=False)
    llm_module.reset_client()
    with pytest.raises(LLMUnavailableError, match="No LLM API key"):
        generate_report("sys", "user")


def test_successful_call_reports_real_token_usage(with_key, monkeypatch):
    response = _response(parsed=InvestigationReport.model_validate(VALID))
    monkeypatch.setattr(
        llm_module, "get_client",
        lambda: SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **kw: response)),
    )
    result = generate_report("sys", "user")
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.model == with_key.llm_model
    assert result.duration_ms >= 0


def test_report_schema_is_accepted_by_the_gemini_schema_converter():
    """Guards against a schema Gemini cannot express.

    The SDK converts the Pydantic model to Gemini's schema dialect. If that
    conversion ever fails, structured output silently stops working, so it is
    asserted here rather than discovered in production.
    """
    from google.genai import _transformers

    schema = _transformers.t_schema(None, InvestigationReport)
    assert schema is not None
    names = set(schema.properties or {})
    assert {"summary", "facts", "recommendation"} <= names
