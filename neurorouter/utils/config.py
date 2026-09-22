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


class WebSearchSettings(ConfigModel):
    provider: Literal["wikipedia", "static"] = "wikipedia"
    max_queries: int = Field(default=3, ge=1, le=10)
    results_per_query: int = Field(default=5, ge=1, le=20)
    timeout_seconds: float = Field(default=10.0, gt=0, le=60)


class RAGSettings(ConfigModel):
    """Local, key-free retrieval configuration."""

    collection_name: str = Field(default="neurorouter_documents", min_length=3)
    persist_directory: Path = Path("data/chroma")
    embedding_provider: Literal["hashing"] = "hashing"
    embedding_dimensions: int = Field(default=384, ge=64, le=4096)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=5000)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    minimum_relevance: float = Field(default=0.15, ge=0, le=1)
    max_file_size_mb: int = Field(default=25, ge=1, le=200)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "RAGSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class CodeExecutionSettings(ConfigModel):
    """Guardrails for optional, restricted local code execution."""

    enabled: bool = False
    language: Literal["python"] = "python"
    timeout_seconds: float = Field(default=2.0, gt=0, le=10)
    max_source_characters: int = Field(default=10000, ge=100, le=50000)
    max_output_characters: int = Field(default=20000, ge=1000, le=100000)


class FinanceSettings(ConfigModel):
    """Deterministic finance-analysis policy settings."""

    require_web_for_current_data: bool = True
    ratio_precision: int = Field(default=4, ge=0, le=8)


class AggregationSettings(ConfigModel):
    """Evidence-packet limits applied before synthesis."""

    context_budget_characters: int = Field(default=24000, ge=1000, le=500000)
    max_agent_output_characters: int = Field(default=2000, ge=100, le=20000)


class EvaluationSettings(ConfigModel):
    """Paths and deterministic scoring settings for router evaluations."""

    dataset_path: Path = Path("evals/routing_dataset.jsonl")
    database_path: Path = Path("data/evaluations.db")
    probability_threshold: float = Field(default=0.5, ge=0, le=1)
    calibration_bins: int = Field(default=10, ge=2, le=20)


class JevFallbackSettings(ConfigModel):
    """Conservative normalized values used only when Jev is unavailable."""

    enabled: bool = True
    intent: Literal[
        "general_qa", "research", "document_qa", "coding", "data_analysis", "finance", "mixed"
    ] = "general_qa"
    intent_confidence: float = Field(default=0.0, ge=0, le=1)
    needs_web: float = Field(default=0.0, ge=0, le=1)
    needs_rag: float = Field(default=0.0, ge=0, le=1)
    needs_code: float = Field(default=0.0, ge=0, le=1)
    needs_data_analysis: float = Field(default=0.0, ge=0, le=1)
    needs_current_information: float = Field(default=0.0, ge=0, le=1)
    needs_citations: float = Field(default=0.0, ge=0, le=1)
    needs_multi_source_research: float = Field(default=0.0, ge=0, le=1)
    complexity_score: float = Field(default=1.0, ge=0, le=3)
    complexity_confidence: float = Field(default=0.0, ge=0, le=1)
    risk_score: float = Field(default=1.0, ge=0, le=3)
    risk_confidence: float = Field(default=0.0, ge=0, le=1)


class JevSettings(ConfigModel):
    model: str = "jev-latest"
    timeout_seconds: float = Field(default=30.0, gt=0)
    fallback: JevFallbackSettings = Field(default_factory=JevFallbackSettings)


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
        if (
            not self.allow_paid_models
            and self.provider == "openrouter"
            and any(
                model != "openrouter/free" and not model.endswith(":free")
                for model in selected.model_tiers.values()
            )
        ):
            raise ValueError("OpenRouter paid models are forbidden when billing is disabled")
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
    web_search: WebSearchSettings
    rag: RAGSettings
    code_execution: CodeExecutionSettings
    finance: FinanceSettings
    aggregation: AggregationSettings
    evaluation: EvaluationSettings
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


class PlanningThresholds(ConfigModel):
    intent_specialist_minimum_confidence: float = Field(ge=0, le=1)
    standard_model_complexity: float = Field(ge=0, le=3)
    reasoning_model_complexity: float = Field(ge=0, le=3)
    quality_gate_complexity_threshold: float = Field(ge=0, le=3)
    quality_gate_risk_threshold: float = Field(ge=0, le=3)
    review_risk_threshold: float = Field(ge=0, le=3)
    review_confidence_threshold: float = Field(ge=0, le=1)
    require_review_on_missing_capability: bool = True

    @model_validator(mode="after")
    def validate_model_tier_order(self) -> "PlanningThresholds":
        if self.standard_model_complexity > self.reasoning_model_complexity:
            raise ValueError("standard model threshold cannot exceed reasoning model threshold")
        return self


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
    planning: PlanningThresholds
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
