import asyncio

import pytest

from neurorouter.core.synthesizer import Synthesizer
from neurorouter.llm.model_router import ModelRouter
from neurorouter.llm.prompts import SynthesisPromptRenderer, load_prompts
from neurorouter.llm.providers import MockLLMProvider
from neurorouter.schemas.context import EvidencePacket
from neurorouter.schemas.execution import AgentName, ExecutionPlan, ModelTier
from neurorouter.utils.config import load_settings


def test_synthesizer_uses_plan_tier_and_returns_usage_metadata() -> None:
    settings = load_settings().llm
    captured = []

    def respond(request):
        captured.append(request)
        return "Candidate response"

    synthesizer = Synthesizer(
        provider=MockLLMProvider(respond),
        model_router=ModelRouter(settings),
        prompt_renderer=SynthesisPromptRenderer(load_prompts().synthesis),
        settings=settings,
    )
    plan = ExecutionPlan(agents=[AgentName.GENERAL], model_tier=ModelTier.STANDARD)
    packet = EvidencePacket(total_characters=0, dropped_items=0, duplicate_items=0)

    result = asyncio.run(
        synthesizer.synthesize(request_text="Explain routing", plan=plan, evidence=packet)
    )

    assert result.response == "Candidate response"
    assert result.model == "mock-standard"
    assert result.model_tier is ModelTier.STANDARD
    assert captured[0].model == "mock-standard"
    assert len(captured[0].messages) == 2


def test_synthesizer_rejects_provider_configuration_mismatch() -> None:
    settings = load_settings().llm.model_copy(update={"provider": "groq"})
    synthesizer = Synthesizer(
        provider=MockLLMProvider(),
        model_router=ModelRouter(settings),
        prompt_renderer=SynthesisPromptRenderer(load_prompts().synthesis),
        settings=settings,
    )

    with pytest.raises(ValueError, match="provider mismatch"):
        asyncio.run(
            synthesizer.synthesize(
                request_text="test",
                plan=ExecutionPlan(agents=[AgentName.GENERAL]),
                evidence=EvidencePacket(total_characters=0, dropped_items=0, duplicate_items=0),
            )
        )
