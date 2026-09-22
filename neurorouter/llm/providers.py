"""Minimal HTTP adapters for the configured free-tier LLM providers."""

import asyncio
import json
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from neurorouter.llm.client import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
)
from neurorouter.utils.config import LLMSettings, SecretSettings


class JSONTransport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]: ...


class UrllibJSONTransport:
    """Small standard-library JSON transport with bounded error content."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                parsed = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read(1000).decode("utf-8", errors="replace")
            raise LLMProviderError(f"Provider HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            reason = getattr(error, "reason", str(error))
            raise LLMProviderError(f"Provider connection failed: {reason}") from error
        except json.JSONDecodeError as error:
            raise LLMProviderError("Provider returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise LLMProviderError("Provider returned a non-object JSON response")
        return parsed


class MockLLMProvider(LLMProvider):
    """No-key deterministic provider for local development and tests."""

    name = "mock"

    def __init__(self, responder: Callable[[LLMRequest], str] | None = None) -> None:
        self.responder = responder or self._default_response

    async def generate(self, request: LLMRequest) -> LLMResponse:
        started = perf_counter()
        content = self.responder(request)
        return LLMResponse(
            content=content,
            provider=self.name,
            model=request.model,
            latency_ms=(perf_counter() - started) * 1000,
            finish_reason="stop",
            raw_response={"development_mock": True},
        )

    @staticmethod
    def _default_response(request: LLMRequest) -> str:
        del request
        return (
            "Mock synthesis completed without an external API call. Configure a free-tier "
            "Groq, Gemini, or OpenRouter key to generate the final natural-language answer."
        )


class OpenAICompatibleProvider(LLMProvider):
    """Chat Completions adapter shared by Groq and OpenRouter."""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        transport: JSONTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport or UrllibJSONTransport()
        self.extra_headers = dict(extra_headers or {})

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": request.model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "max_completion_tokens": request.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        started = perf_counter()
        raw = await asyncio.to_thread(
            self.transport.post_json,
            f"{self.base_url}/chat/completions",
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency = (perf_counter() - started) * 1000
        try:
            choice = raw["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise TypeError
        except (KeyError, IndexError, TypeError) as error:
            raise LLMProviderError("Provider response did not contain assistant text") from error
        usage = raw.get("usage") or {}
        return LLMResponse(
            content=content,
            provider=self.name,
            model=str(raw.get("model") or request.model),
            latency_ms=latency,
            usage=LLMUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            finish_reason=choice.get("finish_reason"),
            raw_response=raw,
        )


class GeminiProvider(LLMProvider):
    """Gemini generateContent REST adapter using header-based API-key authentication."""

    name = "gemini"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        transport: JSONTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport or UrllibJSONTransport()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        system_text = "\n\n".join(
            message.content for message in request.messages if message.role == MessageRole.SYSTEM
        )
        contents = [
            {
                "role": "model" if message.role == MessageRole.ASSISTANT else "user",
                "parts": [{"text": message.content}],
            }
            for message in request.messages
            if message.role != MessageRole.SYSTEM
        ]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": request.max_output_tokens},
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
        model = quote(request.model, safe="-._")
        started = perf_counter()
        raw = await asyncio.to_thread(
            self.transport.post_json,
            f"{self.base_url}/v1beta/models/{model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency = (perf_counter() - started) * 1000
        try:
            candidate = raw["candidates"][0]
            parts = candidate["content"]["parts"]
            content = "".join(part.get("text", "") for part in parts).strip()
            if not content:
                raise TypeError
        except (KeyError, IndexError, TypeError) as error:
            raise LLMProviderError("Gemini response did not contain candidate text") from error
        usage = raw.get("usageMetadata") or {}
        return LLMResponse(
            content=content,
            provider=self.name,
            model=str(raw.get("modelVersion") or request.model),
            latency_ms=latency,
            usage=LLMUsage(
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount"),
            ),
            finish_reason=candidate.get("finishReason"),
            raw_response=raw,
        )


def build_llm_provider(
    settings: LLMSettings,
    secrets: SecretSettings,
    *,
    transport: JSONTransport | None = None,
) -> LLMProvider:
    """Build only the explicitly selected provider; never perform billing fallback."""
    provider_name = settings.provider
    if provider_name == "mock":
        return MockLLMProvider()
    provider_settings = settings.providers[provider_name]
    api_key = getattr(secrets, f"{provider_name}_api_key", None)
    if not api_key:
        raise LLMConfigurationError(
            f"{provider_settings.api_key_env or provider_name.upper() + '_API_KEY'} is required"
        )
    if not provider_settings.base_url:
        raise LLMConfigurationError(f"base_url is required for provider {provider_name}")
    if provider_name == "gemini":
        return GeminiProvider(
            api_key=api_key,
            base_url=provider_settings.base_url,
            timeout_seconds=settings.request_timeout_seconds,
            transport=transport,
        )
    headers = (
        {
            "HTTP-Referer": "https://github.com/aum2606/NeuroRouter",
            "X-Title": "NeuroRouter",
        }
        if provider_name == "openrouter"
        else {}
    )
    return OpenAICompatibleProvider(
        name=provider_name,
        api_key=api_key,
        base_url=provider_settings.base_url,
        timeout_seconds=settings.request_timeout_seconds,
        transport=transport,
        extra_headers=headers,
    )
