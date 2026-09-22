"""End-to-end request composition with durable observability snapshots."""

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from neurorouter.agents.base import AgentInput, AgentResult
from neurorouter.core.pipeline import PipelineResult
from neurorouter.core.runtime import RuntimeComponents, build_runtime
from neurorouter.schemas.execution import ExecutionPlan
from neurorouter.schemas.quality import QualityStatus
from neurorouter.schemas.routing import JevRoutingResult
from neurorouter.schemas.state import ConversationMessage, RouterState
from neurorouter.schemas.trace import StageStatus, TraceRecord, TraceStatus
from neurorouter.telemetry.database import TraceRepository
from neurorouter.telemetry.models import TraceUpdate
from neurorouter.telemetry.tracer import RequestTracer

RuntimeFactory = Callable[[RequestTracer], RuntimeComponents]


class ControlPlaneResult(BaseModel):
    """Complete, UI-ready result for one request."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trace: TraceRecord
    state: RouterState
    routing: JevRoutingResult
    plan: ExecutionPlan
    pipeline: PipelineResult


class NeuroRouterControlPlane:
    """Compose state, probabilistic routing, policy, execution, and quality control."""

    def __init__(
        self,
        repository: TraceRepository,
        *,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.repository = repository
        self.runtime_factory = runtime_factory or build_runtime

    async def execute(
        self,
        request_text: str,
        *,
        recent_messages: Sequence[ConversationMessage | Mapping[str, Any]] = (),
        attachment_types: Sequence[str] = (),
        context: Mapping[str, Any] | None = None,
    ) -> ControlPlaneResult:
        """Run one bounded request and persist each major state transition."""
        tracer = RequestTracer(self.repository, request_text)
        try:
            runtime = self.runtime_factory(tracer)
            state_started = datetime.now(UTC)
            state = runtime.state_builder.build(
                request_text,
                recent_messages=recent_messages,
                attachment_types=attachment_types,
                indexed_document_count=runtime.indexed_document_count,
                collection_metadata=runtime.collection_metadata,
            )
            tracer.record_stage(
                stage="state_building",
                component="StateBuilder",
                status=StageStatus.SUCCEEDED,
                started_at=state_started,
                dependencies=["request"],
                metadata={
                    "recent_messages": len(state.conversation.recent_messages),
                    "attachments": state.attachments.count,
                    "indexed_documents": runtime.indexed_document_count,
                },
            )
            tracer.snapshot(TraceUpdate(state=state.model_dump(mode="json")))

            routing = await runtime.router.route(state)
            decision = routing.decision
            tracer.snapshot(
                TraceUpdate(
                    route=decision.intent.value,
                    routing_decision=decision.model_dump(mode="json"),
                    raw_jev_response=(
                        routing.raw_response
                        if runtime.settings.telemetry.log_raw_jev_responses
                        else {"logging_disabled": True}
                    ),
                )
            )

            policy_started = datetime.now(UTC)
            plan = runtime.policy_engine.create_plan(decision, state)
            tracer.record_stage(
                stage="policy_planning",
                component="PolicyEngine",
                status=StageStatus.SUCCEEDED,
                started_at=policy_started,
                dependencies=["jev_routing"],
                metadata={
                    "agents": [agent.value for agent in plan.agents],
                    "model_tier": plan.model_tier.value,
                    "quality_gate_required": plan.quality_gate_required,
                    "requires_review": plan.requires_review,
                },
            )
            tracer.snapshot(TraceUpdate(execution_plan=plan.model_dump(mode="json")))

            pipeline = await runtime.pipeline.execute(
                AgentInput(
                    trace_id=tracer.trace_id,
                    request_text=request_text,
                    state=state,
                    plan=plan,
                    context=dict(context or {}),
                )
            )
            status = self._status(plan, pipeline)
            quality_payload = self._quality_payload(
                pipeline,
                include_raw=runtime.settings.telemetry.log_raw_jev_responses,
            )
            tracer.snapshot(
                TraceUpdate(
                    retry_count=pipeline.quality.retry_count if pipeline.quality else 0,
                    synthesis_result=pipeline.synthesis.response,
                    quality_gate_result=quality_payload,
                    llm_provider=pipeline.synthesis.provider,
                    llm_model=pipeline.synthesis.model,
                    model_tier=pipeline.synthesis.model_tier.value,
                    token_usage=self._token_usage(pipeline),
                    agent_executions=[
                        result.model_dump(mode="json") for result in pipeline.orchestration.results
                    ],
                    tool_calls=self._tool_calls(pipeline.orchestration.results),
                )
            )
            final_trace = tracer.finalize(
                status=status,
                final_response=pipeline.final_response,
            )
            return ControlPlaneResult(
                trace=final_trace,
                state=state,
                routing=routing,
                plan=plan,
                pipeline=pipeline,
            )
        except Exception as error:
            tracer.record_stage(
                stage="control_plane",
                component=type(self).__name__,
                status=StageStatus.FAILED,
                started_at=datetime.now(UTC),
                error=str(error),
            )
            tracer.finalize(status=TraceStatus.FAILED, error=str(error))
            raise

    @staticmethod
    def _status(plan: ExecutionPlan, pipeline: PipelineResult) -> TraceStatus:
        if pipeline.degraded or plan.requires_review:
            return TraceStatus.REVIEW
        if pipeline.quality and pipeline.quality.status is QualityStatus.REVIEW:
            return TraceStatus.REVIEW
        return TraceStatus.ACCEPTED

    @staticmethod
    def _quality_payload(
        pipeline: PipelineResult,
        *,
        include_raw: bool,
    ) -> dict[str, Any] | None:
        if pipeline.quality is None:
            return None
        payload = pipeline.quality.model_dump(mode="json")
        if not include_raw:
            for attempt in payload.get("attempts", []):
                attempt["quality"]["raw_response"] = {"logging_disabled": True}
        return payload

    @staticmethod
    def _token_usage(pipeline: PipelineResult) -> dict[str, int]:
        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        usages = [pipeline.synthesis.usage]
        usages.extend(
            result.token_usage
            for result in pipeline.orchestration.results
            if result.token_usage is not None
        )
        seen = False
        for usage in usages:
            for key in totals:
                value = getattr(usage, key, None)
                if value is not None:
                    totals[key] += value
                    seen = True
        return totals if seen else {}

    @staticmethod
    def _tool_calls(results: list[AgentResult]) -> list[dict[str, Any]]:
        return [
            {
                "tool": result.agent_name.value,
                "status": "succeeded" if result.success else "failed",
                "latency_ms": result.latency_ms,
                "evidence_count": len(result.evidence),
                "source_count": len(result.sources),
                "metadata": result.metadata,
                "error": result.error.model_dump(mode="json") if result.error else None,
            }
            for result in results
        ]
