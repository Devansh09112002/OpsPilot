"""Environment-driven application settings.

No secret has a usable default. `llm_api_key` defaults to empty, and the agent
reports itself unavailable rather than falling back to a canned response.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- application ---
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_v1_prefix: str = "/api/v1"

    # --- database ---
    database_url: str = Field(
        default="postgresql+psycopg://opspilot:opspilot_dev_pw@127.0.0.1:5433/opspilot"
    )
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # --- model artifact ---
    artifact_dir: Path = Field(default=REPO_ROOT / "artifacts")

    # --- guest sessions ---
    session_cookie_name: str = "opspilot_session"
    session_ttl_hours: int = 72
    cookie_secure: bool = Field(default=False)
    cookie_samesite: str = Field(default="lax")

    # --- CORS ---
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")

    # --- LLM (Google Gemini free tier) ---
    llm_provider: str = Field(default="gemini")
    # Accepts GEMINI_API_KEY as well as LLM_API_KEY, since that is the name
    # Google's console and most hosting platforms use.
    llm_api_key: str = Field(default="", validation_alias=AliasChoices(
        "LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"))
    llm_model: str = Field(default="gemini-3.5-flash-lite")
    # Gemini's free tier caps requests PER MODEL PER DAY (measured: 20/day for
    # gemini-3.5-flash). When one model's daily quota is exhausted the client
    # falls through to the next, which multiplies the usable daily budget
    # without spending anything. Ordered cheapest-capable first.
    llm_model_fallbacks: str = Field(
        default="gemini-3.5-flash,gemini-3.6-flash,gemini-3-flash-preview,gemini-3.1-flash-lite"
    )
    llm_max_output_tokens: int = 2000
    llm_temperature: float = 0.2
    # 0 disables Gemini's thinking mode. This task is bounded synthesis over
    # already-verified evidence, so thinking buys little and costs latency and
    # free-tier quota. -1 leaves the model's default in place.
    llm_thinking_budget: int = 0
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 1

    # --- cost and rate limits (plan section 8.2) ---
    # Sized to the MEASURED Gemini free-tier quota: 20 requests per model per
    # day. With the fallback chain that is roughly 100/day, so the app stops
    # itself below that and returns a clear 429 rather than letting the
    # provider fail the request.
    investigations_per_session_per_day: int = 5
    investigations_global_per_hour: int = 25
    investigations_global_per_day: int = 80
    api_requests_per_minute: int = 120

    @field_validator("cookie_samesite")
    @classmethod
    def _valid_samesite(cls, v: str) -> str:
        allowed = {"lax", "strict", "none"}
        if v.lower() not in allowed:
            raise ValueError(f"cookie_samesite must be one of {allowed}")
        return v.lower()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_model_chain(self) -> list[str]:
        """Primary model first, then each distinct fallback."""
        chain = [self.llm_model.strip()]
        for name in self.llm_model_fallbacks.split(","):
            name = name.strip()
            if name and name not in chain:
                chain.append(name)
        return chain

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
