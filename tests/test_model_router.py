import pytest

from neurorouter.llm.model_router import ModelRouter
from neurorouter.schemas.execution import ModelTier
from neurorouter.utils.config import LLMSettings, load_settings


def test_model_router_maps_deterministic_tier_to_configuration() -> None:
    settings = load_settings().llm

    selection = ModelRouter(settings).select(ModelTier.REASONING)

    assert selection.provider == "mock"
    assert selection.model == "mock-reasoning"
    assert selection.free_tier_only is True


def test_openrouter_paid_model_is_rejected_when_billing_disabled() -> None:
    payload = load_settings().llm.model_dump()
    payload["provider"] = "openrouter"
    payload["providers"]["openrouter"]["model_tiers"]["fast"] = "openai/paid-model"

    with pytest.raises(ValueError, match="paid models"):
        LLMSettings.model_validate(payload)
