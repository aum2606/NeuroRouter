"""Phase 7 execution, aggregation, and synthesis composition."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from neurorouter.agents.base import AgentInput
from neurorouter.core.aggregator import ContextAggregator
from neurorouter.core.orchestrator import OrchestrationResult, Orchestrator
from neurorouter.core.retry_controller import RetryController
from neurorouter.core.synthesizer import SynthesisResult, Synthesizer
from neurorouter.schemas.context import EvidencePacket
from neurorouter.schemas.quality import QualityOutcome
from neurorouter.schemas.trace import StageStatus
from neurorouter.telemetry.tracer import RequestTracer


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orchestration: OrchestrationResult
    evidence: EvidencePacket
    synthesis: SynthesisResult
    quality: QualityOutcome | None = None
    final_response: str
    degraded: bool


class ExecutionPipeline:
    """Run bounded specialists, normalize their context, and synthesize one candidate."""

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        aggregator: ContextAggregator,
        synthesizer: Synthesizer,
        retry_controller: RetryController | None = None,
        tracer: RequestTracer | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.aggregator = aggregator
        self.synthesizer = synthesizer
        self.retry_controller = retry_controller
        self.tracer = tracer

    async def execute(self, agent_input: AgentInput) -> PipelineResult:
        pipeline_started = datetime.now(UTC)
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
        quality = None
        final_response = synthesis.response
        if agent_input.plan.quality_gate_required:
            if self.retry_controller is None:
                raise RuntimeError("quality gate is required but no RetryController is configured")
            quality = await self.retry_controller.run(
                request_text=agent_input.request_text,
                plan=agent_input.plan,
                initial_synthesis=synthesis,
                agent_results=orchestration.results,
            )
            final_response = quality.final_response
        result = PipelineResult(
            orchestration=orchestration,
            evidence=evidence,
            synthesis=synthesis,
            quality=quality,
            final_response=final_response,
            degraded=not orchestration.success,
        )
        if self.tracer:
            self.tracer.record_stage(
                stage="execution_pipeline",
                component=type(self).__name__,
                status=(StageStatus.SUCCEEDED if not result.degraded else StageStatus.FAILED),
                started_at=pipeline_started,
                dependencies=["policy_planning"],
                metadata={
                    "agents": [agent.value for agent in agent_input.plan.agents],
                    "evidence_items": len(evidence.items),
                    "quality_status": quality.status.value if quality else "not_required",
                },
            )
        return result
