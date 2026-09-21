import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from neurorouter.core.router import JevRouter
from neurorouter.core.state_builder import StateBuilder
from neurorouter.jev.client import JevClient
from neurorouter.schemas.routing import Intent
from neurorouter.utils.config import load_settings
from tests.test_routing_parser import _raw_response


class RecordingClient:
    def __init__(self, response: Mapping[str, Any] | None = None) -> None:
        self.response = response
        self.calls: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    async def evaluate(
        self, *, state: Mapping[str, Any], questions: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append((state, questions))
        if self.response is None:
            raise ConnectionError("Jev unavailable")
        return self.response


def test_router_batches_questions_and_preserves_raw_response() -> None:
    settings = load_settings()
    client = RecordingClient(_raw_response())
    router = JevRouter(client, settings.jev.fallback)
    state = StateBuilder(settings).build("Research current battery technology")

    result = asyncio.run(router.route(state))

    assert len(client.calls) == 1
    assert len(client.calls[0][1]) == 10
    assert result.decision.intent is Intent.RESEARCH
    assert result.raw_response["model"] == "jev-1.13.0"
    assert result.decision.jev_latency_ms >= 0


def test_router_applies_capability_aware_fallback() -> None:
    settings = load_settings()
    fallback = settings.jev.fallback.model_copy(
        update={"needs_web": 0.9, "needs_rag": 0.8, "complexity_score": 1.5}
    )
    client: JevClient = RecordingClient()
    state = StateBuilder(settings).build("Fallback safely")

    result = asyncio.run(JevRouter(client, fallback).route(state))

    assert result.raw_response == {
        "fallback_applied": True,
        "error_type": "ConnectionError",
    }
    assert result.decision.needs_web == 0.0  # capability disabled in settings
    assert result.decision.needs_rag == pytest.approx(0.8)
    assert result.decision.complexity_probabilities == {1: 0.5, 2: 0.5}


def test_router_reraises_when_fallback_is_disabled() -> None:
    settings = load_settings()
    disabled = settings.jev.fallback.model_copy(update={"enabled": False})
    state = StateBuilder(settings).build("Do not fall back")

    with pytest.raises(ConnectionError, match="unavailable"):
        asyncio.run(JevRouter(RecordingClient(), disabled).route(state))
