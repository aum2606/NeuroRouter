"""Dependency-aware execution of bounded specialist agents."""

import asyncio
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.agents.base import AgentError, AgentInput, AgentResult, BaseAgent
from neurorouter.schemas.execution import AgentName
from neurorouter.schemas.trace import StageStatus
from neurorouter.telemetry.tracer import RequestTracer


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: UUID
    results: list[AgentResult] = Field(default_factory=list)
    total_latency_ms: float = Field(ge=0)
    success: bool


class Orchestrator:
    """Execute explicit plan groups while isolating individual failures."""

    def __init__(
        self,
        agents: dict[AgentName, BaseAgent],
        *,
        tracer: RequestTracer | None = None,
    ) -> None:
        self.agents = agents
        self.tracer = tracer

    async def execute(self, agent_input: AgentInput) -> OrchestrationResult:
        started = perf_counter()
        plan = agent_input.plan
        pending = list(plan.agents)
        results: dict[AgentName, AgentResult] = {}
        group_for = {agent: group for group in plan.parallel_agents for agent in group}

        while pending:
            progressed = False
            for agent_name in list(pending):
                dependencies = plan.agent_dependencies.get(agent_name, [])
                if not all(dependency in results for dependency in dependencies):
                    continue
                failed_dependencies = [
                    dependency for dependency in dependencies if not results[dependency].success
                ]
                if failed_dependencies:
                    result = self._dependency_failure(agent_name, failed_dependencies)
                    results[agent_name] = result
                    pending.remove(agent_name)
                    self._record(result, dependencies)
                    progressed = True
                    continue

                group = group_for.get(agent_name)
                candidates = group if group is not None else [agent_name]
                batch = [
                    candidate
                    for candidate in candidates
                    if candidate in pending
                    and all(
                        dependency in results and results[dependency].success
                        for dependency in plan.agent_dependencies.get(candidate, [])
                    )
                ]
                if not batch:
                    continue
                batch_results = await asyncio.gather(
                    *(
                        self._run_agent(
                            candidate,
                            agent_input,
                            plan.agent_dependencies.get(candidate, []),
                            results,
                        )
                        for candidate in batch
                    )
                )
                for candidate, result in zip(batch, batch_results, strict=True):
                    results[candidate] = result
                    pending.remove(candidate)
                    self._record(result, plan.agent_dependencies.get(candidate, []))
                progressed = True
                break
            if not progressed:
                raise RuntimeError("execution plan could not make progress")

        ordered = [results[agent] for agent in plan.agents]
        return OrchestrationResult(
            trace_id=agent_input.trace_id,
            results=ordered,
            total_latency_ms=(perf_counter() - started) * 1000,
            success=all(result.success for result in ordered),
        )

    async def _run_agent(
        self,
        agent_name: AgentName,
        base_input: AgentInput,
        dependencies: list[AgentName],
        results: dict[AgentName, AgentResult],
    ) -> AgentResult:
        agent = self.agents.get(agent_name)
        if agent is None:
            return self._failure(
                agent_name,
                code="agent_not_registered",
                message=f"No implementation registered for {agent_name.value}",
            )
        dependency_results = {
            dependency.value: results[dependency].model_dump(mode="json")
            for dependency in dependencies
        }
        context: dict[str, Any] = dict(base_input.context)
        context["dependency_results"] = dependency_results
        return await agent.run(base_input.model_copy(update={"context": context}))

    def _record(self, result: AgentResult, dependencies: list[AgentName]) -> None:
        if self.tracer is None:
            return
        token_usage = result.token_usage.model_dump(exclude_none=True) if result.token_usage else {}
        self.tracer.record_stage(
            stage="agent_execution",
            component=result.agent_name.value,
            status=StageStatus.SUCCEEDED if result.success else StageStatus.FAILED,
            started_at=result.started_at,
            dependencies=[dependency.value for dependency in dependencies],
            metadata={
                **result.metadata,
                "evidence_count": len(result.evidence),
                "source_count": len(result.sources),
            },
            token_usage=token_usage,
            error=result.error.message if result.error else None,
        )

    def _dependency_failure(
        self, agent_name: AgentName, failed_dependencies: list[AgentName]
    ) -> AgentResult:
        names = ", ".join(dependency.value for dependency in failed_dependencies)
        return self._failure(
            agent_name,
            code="dependency_failed",
            message=f"Skipped because dependencies failed: {names}",
            metadata={"skipped": True, "failed_dependencies": names},
        )

    @staticmethod
    def _failure(
        agent_name: AgentName,
        *,
        code: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        now = datetime.now(UTC)
        return AgentResult(
            agent_name=agent_name,
            metadata=metadata or {},
            started_at=now,
            ended_at=now,
            latency_ms=0.0,
            success=False,
            error=AgentError(
                code=code,
                message=message,
                error_type="OrchestrationError",
                retriable=False,
            ),
        )
