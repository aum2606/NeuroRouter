"""Typed configuration loaded from YAML with environment overrides."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppMetadata(ConfigModel):
    name: str = "NeuroRouter"
    environment: str = "development"
    log_level: str = "INFO"


class CapabilitySettings(ConfigModel):
    web_search_available: bool = False
    rag_available: bool = True
    code_available: bool = False
    finance_agent_available: bool = False


class TelemetrySettings(ConfigModel):
    database_path: Path = Path("data/neurorouter.db")
    log_raw_jev_responses: bool = True


class JevSettings(ConfigModel):
    model: str = "jev-latest"
    timeout_seconds: float = Field(default=30.0, gt=0)


class LLMProviderSettings(ConfigModel):
    """Configuration for one interchangeable LLM backend."""

    enabled: bool = False
    free_tier_only: bool = True
    api_key_env: str | None = None
    base_url: str | None = None
    model_tiers: dict[str, str]


class LLMSettings(ConfigModel):
    """LLM selection with an explicit guard against paid-model fallback."""

    provider: Literal["mock", "groq", "gemini", "openrouter"] = "mock"
    allow_paid_models: bool = False
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    providers: dict[str, LLMProviderSettings]

    @model_validator(mode="after")
    def validate_selected_provider(self) -> "LLMSettings":
        selected = self.providers.get(self.provider)
        if selected is None:
            raise ValueError(f"missing configuration for LLM provider: {self.provider}")
        if not selected.enabled:
            raise ValueError(f"selected LLM provider is disabled: {self.provider}")
        if not self.allow_paid_models and not selected.free_tier_only:
            raise ValueError("paid-model fallback is disabled")
        required_tiers = {"fast", "standard", "reasoning"}
        if required_tiers - selected.model_tiers.keys():
            raise ValueError("selected LLM provider must configure all model tiers")
        return self


class UISettings(ConfigModel):
    page_title: str = "NeuroRouter Control Plane"
    page_icon: str = "🧠"


class RuntimeSettings(BaseSettings):
    """Secret-free runtime settings safe to expose in routing state."""

    model_config = SettingsConfigDict(
        env_prefix="NEUROROUTER_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    app: AppMetadata
    capabilities: CapabilitySettings
    telemetry: TelemetrySettings
    jev: JevSettings
    llm: LLMSettings
    ui: UISettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let environment values override the YAML supplied as init data."""
        return env_settings, dotenv_settings, init_settings, file_secret_settings


class RoutingThresholds(ConfigModel):
    web_threshold: float = Field(ge=0, le=1)
    rag_threshold: float = Field(ge=0, le=1)
    code_threshold: float = Field(ge=0, le=1)
    data_threshold: float = Field(ge=0, le=1)
    citations_threshold: float = Field(ge=0, le=1)
    multi_source_threshold: float = Field(ge=0, le=1)


class QualityThresholds(ConfigModel):
    answers_request_minimum: float = Field(ge=0, le=1)
    evidence_support_minimum: float = Field(ge=0, le=1)
    unsupported_claims_maximum: float = Field(ge=0, le=1)
    missing_information_maximum: float = Field(ge=0, le=1)
    contradiction_maximum: float = Field(ge=0, le=1)
    additional_retrieval_threshold: float = Field(ge=0, le=1)
    max_retries: int = Field(default=2, ge=0, le=10)


class PolicyThresholds(ConfigModel):
    routing: RoutingThresholds
    quality: QualityThresholds


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return loaded


@lru_cache(maxsize=1)
def load_settings(settings_path: Path | None = None) -> RuntimeSettings:
    path = settings_path or PROJECT_ROOT / "config" / "settings.yaml"
    return RuntimeSettings(**_read_yaml(path))


@lru_cache(maxsize=1)
def load_thresholds(thresholds_path: Path | None = None) -> PolicyThresholds:
    path = thresholds_path or PROJECT_ROOT / "config" / "thresholds.yaml"
    return PolicyThresholds.model_validate(_read_yaml(path))


class SecretSettings(BaseSettings):
    """Credentials kept separate so they cannot leak into RouterState."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    typesafe_api_key: str | None = Field(default=None, alias="TYPESAFE_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
