import asyncio

from neurorouter.agents.base import AgentInput
from neurorouter.agents.general import GeneralAgent
from neurorouter.core.aggregator import ContextAggregator
from neurorouter.core.orchestrator import Orchestrator
from neurorouter.core.pipeline import ExecutionPipeline
from neurorouter.core.state_builder import StateBuilder
from neurorouter.core.synthesizer import Synthesizer
from neurorouter.llm.model_router import ModelRouter
from neurorouter.llm.prompts import SynthesisPromptRenderer, load_prompts
from neurorouter.llm.providers import MockLLMProvider
from neurorouter.schemas.execution import AgentName, ExecutionPlan
from neurorouter.schemas.quality import QualityOutcome, QualityStatus
from neurorouter.schemas.trace import TraceRecord
from neurorouter.telemetry.tracer import RequestTracer
from neurorouter.utils.config import load_settings


def test_pipeline_composes_execution_aggregation_and_synthesis(repository) -> None:
    settings = load_settings()
    tracer = RequestTracer(repository, "Explain NeuroRouter")
    plan = ExecutionPlan(agents=[AgentName.GENERAL], quality_gate_required=False)
    agent_input = AgentInput(
        trace_id=tracer.trace_id,
        request_text="Explain NeuroRouter",
        state=StateBuilder(settings).build("Explain NeuroRouter"),
        plan=plan,
    )
    synthesizer = Synthesizer(
        provider=MockLLMProvider(lambda request: "Synthesized candidate"),
        model_router=ModelRouter(settings.llm),
        prompt_renderer=SynthesisPromptRenderer(load_prompts().synthesis),
        settings=settings.llm,
        tracer=tracer,
    )
    pipeline = ExecutionPipeline(
        orchestrator=Orchestrator({AgentName.GENERAL: GeneralAgent()}, tracer=tracer),
        aggregator=ContextAggregator(context_budget_characters=1000),
        synthesizer=synthesizer,
        tracer=tracer,
    )

    result = asyncio.run(pipeline.execute(agent_input))
    events = repository.list_stage_events(tracer.trace_id)

    assert result.synthesis.response == "Synthesized candidate"
    assert result.degraded is False
    assert {event.stage for event in events} == {
        "agent_execution",
        "context_aggregation",
        "synthesis",
        "execution_pipeline",
    }


class AcceptingRetryController:
    async def run(self, **kwargs):
        return QualityOutcome(
            final_response="Quality-accepted response",
            status=QualityStatus.ACCEPTED,
            retry_count=0,
        )


def test_pipeline_runs_required_quality_controller(repository) -> None:
    settings = load_settings()
    plan = ExecutionPlan(agents=[AgentName.GENERAL], quality_gate_required=True)
    agent_input = AgentInput(
        trace_id=repository.create_trace(TraceRecord(request_text="quality request")).trace_id,
        request_text="quality request",
        state=StateBuilder(settings).build("quality request"),
        plan=plan,
    )
    synthesizer = Synthesizer(
        provider=MockLLMProvider(lambda request: "Initial response"),
        model_router=ModelRouter(settings.llm),
        prompt_renderer=SynthesisPromptRenderer(load_prompts().synthesis),
        settings=settings.llm,
    )
    pipeline = ExecutionPipeline(
        orchestrator=Orchestrator({AgentName.GENERAL: GeneralAgent()}),
        aggregator=ContextAggregator(context_budget_characters=1000),
        synthesizer=synthesizer,
        retry_controller=AcceptingRetryController(),
    )

    result = asyncio.run(pipeline.execute(agent_input))

    assert result.quality is not None
    assert result.final_response == "Quality-accepted response"
