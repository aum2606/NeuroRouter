"""Map deterministic complexity tiers to configured provider model IDs."""

from pydantic import BaseModel, ConfigDict

from neurorouter.schemas.execution import ModelTier
from neurorouter.utils.config import LLMSettings


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    tier: ModelTier
    model: str
    free_tier_only: bool


class ModelRouter:
    """Resolve model configuration without asking an LLM to select an LLM."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def select(self, tier: ModelTier) -> ModelSelection:
        provider = self.settings.providers[self.settings.provider]
        model = provider.model_tiers[tier.value]
        if not self.settings.allow_paid_models and not provider.free_tier_only:
            raise ValueError("paid-model selection is disabled")
        if (
            not self.settings.allow_paid_models
            and self.settings.provider == "openrouter"
            and model != "openrouter/free"
            and not model.endswith(":free")
        ):
            raise ValueError("configured OpenRouter model is not explicitly free")
        return ModelSelection(
            provider=self.settings.provider,
            tier=tier,
            model=model,
            free_tier_only=provider.free_tier_only,
        )
