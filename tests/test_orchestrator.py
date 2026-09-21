import asyncio
from dataclasses import dataclass
from uuid import uuid4

from neurorouter.agents.base import AgentInput, AgentPayload, BaseAgent
from neurorouter.core.orchestrator import Orchestrator
from neurorouter.core.state_builder import StateBuilder
from neurorouter.schemas.execution import AgentName, ExecutionPlan
from neurorouter.telemetry.tracer import RequestTracer
from neurorouter.utils.config import load_settings


@dataclass
class Gate:
    count: int
    event: asyncio.Event


class GateAgent(BaseAgent):
    def __init__(self, name: AgentName, gate: Gate) -> None:
        self.name = name
        self.gate = gate

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        del agent_input
        self.gate.count += 1
        if self.gate.count == 2:
            self.gate.event.set()
        await asyncio.wait_for(self.gate.event.wait(), timeout=0.5)
        return AgentPayload(output=f"{self.name.value} complete")


class RecordingAgent(BaseAgent):
    def __init__(self, name: AgentName, events: list[str]) -> None:
        self.name = name
        self.events = events

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        self.events.append(self.name.value)
        return AgentPayload(
            output="complete",
            metadata={
                "dependency_result_count": len(agent_input.context.get("dependency_results", {}))
            },
        )


def _input(plan: ExecutionPlan, trace_id=None) -> AgentInput:
    state = StateBuilder(load_settings()).build("orchestrate this")
    return AgentInput(
        trace_id=trace_id or uuid4(),
        request_text="orchestrate this",
        state=state,
        plan=plan,
    )


def test_orchestrator_runs_explicit_parallel_group_and_traces(repository) -> None:
    async def scenario():
        gate = Gate(count=0, event=asyncio.Event())
        plan = ExecutionPlan(
            agents=[AgentName.WEB_RESEARCH, AgentName.RAG],
            parallel_agents=[[AgentName.WEB_RESEARCH, AgentName.RAG]],
            web_allowed=True,
            rag_allowed=True,
            use_web=True,
            use_rag=True,
        )
        tracer = RequestTracer(repository, "parallel request")
        orchestrator = Orchestrator(
            {
                AgentName.WEB_RESEARCH: GateAgent(AgentName.WEB_RESEARCH, gate),
                AgentName.RAG: GateAgent(AgentName.RAG, gate),
            },
            tracer=tracer,
        )
        result = await orchestrator.execute(_input(plan, tracer.trace_id))
        return gate, tracer, result

    gate, tracer, result = asyncio.run(scenario())

    assert gate.count == 2
    assert result.success is True
    assert [item.agent_name for item in result.results] == [
        AgentName.WEB_RESEARCH,
        AgentName.RAG,
    ]
    assert len(repository.list_stage_events(tracer.trace_id)) == 2


def test_orchestrator_honors_dependencies_and_passes_results() -> None:
    events: list[str] = []
    plan = ExecutionPlan(
        agents=[AgentName.WEB_RESEARCH, AgentName.FINANCE],
        agent_dependencies={AgentName.FINANCE: [AgentName.WEB_RESEARCH]},
        web_allowed=True,
        use_web=True,
    )
    orchestrator = Orchestrator(
        {
            AgentName.WEB_RESEARCH: RecordingAgent(AgentName.WEB_RESEARCH, events),
            AgentName.FINANCE: RecordingAgent(AgentName.FINANCE, events),
        }
    )

    result = asyncio.run(orchestrator.execute(_input(plan)))

    assert events == [AgentName.WEB_RESEARCH.value, AgentName.FINANCE.value]
    assert result.results[1].metadata["dependency_result_count"] == 1


def test_missing_agent_is_isolated_and_dependent_agent_is_skipped() -> None:
    plan = ExecutionPlan(
        agents=[AgentName.WEB_RESEARCH, AgentName.FINANCE],
        agent_dependencies={AgentName.FINANCE: [AgentName.WEB_RESEARCH]},
        web_allowed=True,
        use_web=True,
    )

    result = asyncio.run(Orchestrator({}).execute(_input(plan)))

    assert result.success is False
    assert result.results[0].error is not None
    assert result.results[0].error.code == "agent_not_registered"
    assert result.results[1].error is not None
    assert result.results[1].error.code == "dependency_failed"
    assert result.results[1].metadata["skipped"] is True
