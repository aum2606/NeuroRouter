"""Typed configuration loaded from YAML with environment overrides."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
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


class LLMSettings(ConfigModel):
    provider: str = "mock"
    model_tiers: dict[str, str]


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
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
