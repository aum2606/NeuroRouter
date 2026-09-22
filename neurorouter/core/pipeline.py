"""Phase 7 execution, aggregation, and synthesis composition."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from neurorouter.agents.base import AgentInput
from neurorouter.core.aggregator import ContextAggregator
from neurorouter.core.orchestrator import OrchestrationResult, Orchestrator
from neurorouter.core.synthesizer import SynthesisResult, Synthesizer
from neurorouter.schemas.context import EvidencePacket
from neurorouter.schemas.trace import StageStatus
from neurorouter.telemetry.tracer import RequestTracer


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orchestration: OrchestrationResult
    evidence: EvidencePacket
    synthesis: SynthesisResult
    degraded: bool


class ExecutionPipeline:
    """Run bounded specialists, normalize their context, and synthesize one candidate."""

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        aggregator: ContextAggregator,
        synthesizer: Synthesizer,
        tracer: RequestTracer | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.aggregator = aggregator
        self.synthesizer = synthesizer
        self.tracer = tracer

    async def execute(self, agent_input: AgentInput) -> PipelineResult:
        orchestration = await self.orchestrator.execute(agent_input)
        started_at = datetime.now(UTC)
        try:
            evidence = self.aggregator.aggregate(orchestration.results)
        except Exception as error:
            if self.tracer:
                self.tracer.record_stage(
                    stage="context_aggregation",
                    component=type(self.aggregator).__name__,
                    status=StageStatus.FAILED,
                    started_at=started_at,
                    dependencies=[result.agent_name.value for result in orchestration.results],
                    error=str(error),
                )
            raise
        if self.tracer:
            self.tracer.record_stage(
                stage="context_aggregation",
                component=type(self.aggregator).__name__,
                status=StageStatus.SUCCEEDED,
                started_at=started_at,
                dependencies=[result.agent_name.value for result in orchestration.results],
                metadata={
                    "items": len(evidence.items),
                    "sources": len(evidence.sources),
                    "dropped_items": evidence.dropped_items,
                    "duplicate_items": evidence.duplicate_items,
                    "total_characters": evidence.total_characters,
                },
            )
        synthesis = await self.synthesizer.synthesize(
            request_text=agent_input.request_text,
            plan=agent_input.plan,
            evidence=evidence,
        )
        return PipelineResult(
            orchestration=orchestration,
            evidence=evidence,
            synthesis=synthesis,
            degraded=not orchestration.success,
        )
