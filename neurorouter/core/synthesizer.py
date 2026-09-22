"""Evidence-aware user-response synthesis."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.llm.client import LLMProvider, LLMRequest, LLMUsage
from neurorouter.llm.model_router import ModelRouter
from neurorouter.llm.prompts import SynthesisPromptRenderer
from neurorouter.schemas.context import EvidencePacket
from neurorouter.schemas.execution import ExecutionPlan, ModelTier
from neurorouter.schemas.trace import StageStatus
from neurorouter.telemetry.tracer import RequestTracer
from neurorouter.utils.config import LLMSettings


class SynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1)
    provider: str
    model: str
    model_tier: ModelTier
    latency_ms: float = Field(ge=0)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    evidence_items_used: int = Field(ge=0)
    citations_required: bool


class Synthesizer:
    """Generate a candidate response from the deliberately bounded evidence packet."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model_router: ModelRouter,
        prompt_renderer: SynthesisPromptRenderer,
        settings: LLMSettings,
        tracer: RequestTracer | None = None,
    ) -> None:
        self.provider = provider
        self.model_router = model_router
        self.prompt_renderer = prompt_renderer
        self.settings = settings
        self.tracer = tracer

    async def synthesize(
        self,
        *,
        request_text: str,
        plan: ExecutionPlan,
        evidence: EvidencePacket,
        retry_instruction: str | None = None,
    ) -> SynthesisResult:
        selection = self.model_router.select(plan.model_tier)
        if self.provider.name != selection.provider:
            raise ValueError(
                f"provider mismatch: configured {selection.provider}, got {self.provider.name}"
            )
        messages = self.prompt_renderer.render(
            request_text=request_text,
            plan=plan,
            evidence=evidence,
            citations_required=plan.citations_required,
            retry_instruction=retry_instruction,
        )
        started_at = datetime.now(UTC)
        try:
            response = await self.provider.generate(
                LLMRequest(
                    model=selection.model,
                    messages=messages,
                    max_output_tokens=self.settings.max_output_tokens,
                )
            )
        except Exception as error:
            if self.tracer:
                self.tracer.record_stage(
                    stage="synthesis",
                    component=self.provider.name,
                    status=StageStatus.FAILED,
                    started_at=started_at,
                    dependencies=[agent.value for agent in plan.agents],
                    metadata={"model": selection.model, "model_tier": selection.tier.value},
                    error=str(error),
                )
            raise
        result = SynthesisResult(
            response=response.content,
            provider=response.provider,
            model=response.model,
            model_tier=selection.tier,
            latency_ms=response.latency_ms,
            usage=response.usage,
            evidence_items_used=len(evidence.items),
            citations_required=plan.citations_required,
        )
        if self.tracer:
            self.tracer.record_stage(
                stage="synthesis",
                component=response.provider,
                status=StageStatus.SUCCEEDED,
                started_at=started_at,
                dependencies=[agent.value for agent in plan.agents],
                metadata={
                    "model": response.model,
                    "model_tier": selection.tier.value,
                    "evidence_items": len(evidence.items),
                    "citations_required": plan.citations_required,
                },
                token_usage={
                    key: value
                    for key, value in response.usage.model_dump(exclude_none=True).items()
                },
            )
        return result
