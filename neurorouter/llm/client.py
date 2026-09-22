"""Stable language-model contracts independent of provider SDKs."""

import json
import re
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(LLMModel):
    role: MessageRole
    content: str = Field(min_length=1)


class LLMRequest(LLMModel):
    model: str = Field(min_length=1)
    messages: list[LLMMessage] = Field(min_length=1)
    max_output_tokens: int = Field(default=4096, gt=0)


class LLMUsage(LLMModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMResponse(LLMModel):
    content: str = Field(min_length=1)
    provider: str
    model: str
    latency_ms: float = Field(ge=0)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class LLMProviderError(RuntimeError):
    """Raised when a configured provider cannot produce a valid response."""


class LLMConfigurationError(ValueError):
    """Raised before a request when provider configuration is incomplete or unsafe."""


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class LLMProvider(ABC):
    """Provider interface used by synthesis and later structured quality stages."""

    name: str

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a text response."""
        raise NotImplementedError

    async def generate_structured(
        self,
        request: LLMRequest,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        """Generate and validate JSON against a Pydantic response model."""
        response = await self.generate(request)
        payload = self._json_payload(response.content)
        return response_model.model_validate(payload)

    @staticmethod
    def _json_payload(content: str) -> Any:
        stripped = content.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
        if fenced:
            stripped = fenced.group(1)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as error:
            raise LLMProviderError("Provider returned invalid structured JSON") from error
