import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from neurorouter.llm.client import (
    LLMConfigurationError,
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    MessageRole,
)
from neurorouter.llm.providers import (
    GeminiProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    build_llm_provider,
)
from neurorouter.utils.config import SecretSettings, load_settings


class CapturingTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


def _request(model: str = "test-model") -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=[
            LLMMessage(role=MessageRole.SYSTEM, content="Use evidence."),
            LLMMessage(role=MessageRole.USER, content="Answer this."),
        ],
        max_output_tokens=321,
    )


def test_openai_compatible_provider_builds_and_parses_chat_completion() -> None:
    transport = CapturingTransport(
        {
            "model": "resolved-model",
            "choices": [{"message": {"content": "Grounded answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )
    provider = OpenAICompatibleProvider(
        name="groq",
        api_key="secret-value",
        base_url="https://api.groq.test/openai/v1",
        timeout_seconds=5,
        transport=transport,
    )

    response = asyncio.run(provider.generate(_request("openai/gpt-oss-20b")))

    assert response.content == "Grounded answer"
    assert response.usage.total_tokens == 14
    assert transport.calls[0]["url"].endswith("/chat/completions")
    assert transport.calls[0]["payload"]["max_completion_tokens"] == 321
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer secret-value"


def test_gemini_provider_uses_header_auth_and_generate_content_shape() -> None:
    transport = CapturingTransport(
        {
            "modelVersion": "gemini-3.8-flash-001",
            "candidates": [
                {
                    "content": {"parts": [{"text": "Gemini answer"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 3,
                "totalTokenCount": 11,
            },
        }
    )
    provider = GeminiProvider(
        api_key="gemini-secret",
        base_url="https://generativelanguage.googleapis.com",
        timeout_seconds=5,
        transport=transport,
    )

    response = asyncio.run(provider.generate(_request("gemini-3.8-flash")))
    call = transport.calls[0]

    assert response.content == "Gemini answer"
    assert response.usage.input_tokens == 8
    assert call["headers"]["x-goog-api-key"] == "gemini-secret"
    assert "gemini-secret" not in call["url"]
    assert call["payload"]["systemInstruction"]["parts"][0]["text"] == "Use evidence."
    assert call["payload"]["generationConfig"]["maxOutputTokens"] == 321


class StructuredAnswer(BaseModel):
    accepted: bool


def test_mock_provider_supports_structured_generation() -> None:
    provider = MockLLMProvider(lambda request: '```json\n{"accepted": true}\n```')

    result = asyncio.run(provider.generate_structured(_request(), StructuredAnswer))

    assert result.accepted is True


def test_malformed_provider_response_fails_visibly() -> None:
    provider = OpenAICompatibleProvider(
        name="openrouter",
        api_key="key",
        base_url="https://openrouter.test/api/v1",
        timeout_seconds=5,
        transport=CapturingTransport({"choices": []}),
    )

    with pytest.raises(LLMProviderError, match="assistant text"):
        asyncio.run(provider.generate(_request()))


def test_provider_factory_requires_key_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    groq_settings = load_settings().llm.model_copy(update={"provider": "groq"})

    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        build_llm_provider(groq_settings, SecretSettings(_env_file=None))

    assert build_llm_provider(load_settings().llm, SecretSettings(_env_file=None)).name == "mock"
