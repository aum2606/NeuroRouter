import asyncio

import pytest

from neurorouter.core.control_plane import NeuroRouterControlPlane
from neurorouter.core.runtime import build_runtime
from neurorouter.jev.client import StaticJevClient
from neurorouter.schemas.trace import TraceStatus
from neurorouter.utils.config import SecretSettings, load_settings, load_thresholds


def _routing_response() -> dict:
    nouls = {
        name: {"type": "noul", "noul": 0.0}
        for name in (
            "needs_web",
            "needs_rag",
            "needs_code",
            "needs_data_analysis",
            "needs_current_information",
            "needs_citations",
            "needs_multi_source_research",
        )
    }
    return {
        "model": "jev-test",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "general_qa",
                "confidence": 0.95,
                "probabilities": {"general_qa": 0.95, "research": 0.05},
            },
            **nouls,
            "complexity": {
                "type": "score",
                "score": 0.0,
                "confidence": 0.95,
                "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0},
            },
            "risk": {
                "type": "score",
                "score": 0.0,
                "confidence": 0.95,
                "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0},
            },
        },
    }


def test_control_plane_persists_complete_request_lifecycle(repository) -> None:
    settings = load_settings()
    settings = settings.model_copy(
        update={
            "capabilities": settings.capabilities.model_copy(
                update={
                    "web_search_available": False,
                    "rag_available": False,
                    "code_available": False,
                    "finance_agent_available": False,
                }
            )
        }
    )

    def factory(tracer):
        return build_runtime(
            tracer,
            settings=settings,
            thresholds=load_thresholds(),
            secrets=SecretSettings(_env_file=None),
            jev_client=StaticJevClient(_routing_response()),
        )

    result = asyncio.run(
        NeuroRouterControlPlane(repository, runtime_factory=factory).execute("Explain NeuroRouter")
    )
    stored = repository.get_trace(result.trace.trace_id)
    events = repository.list_stage_events(result.trace.trace_id)

    assert stored.status is TraceStatus.ACCEPTED
    assert stored.routing_decision["intent"] == "general_qa"
    assert stored.execution_plan["agents"] == ["GeneralAgent"]
    assert stored.llm_provider == "mock"
    assert stored.agent_executions[0]["success"] is True
    assert stored.tool_calls[0]["tool"] == "GeneralAgent"
    assert stored.final_response == result.pipeline.final_response
    assert {event.stage for event in events} >= {
        "state_building",
        "jev_routing",
        "policy_planning",
        "agent_execution",
        "context_aggregation",
        "synthesis",
        "execution_pipeline",
    }


def test_control_plane_records_terminal_failure(repository) -> None:
    def broken_factory(tracer):
        del tracer
        raise RuntimeError("runtime unavailable")

    control_plane = NeuroRouterControlPlane(repository, runtime_factory=broken_factory)

    with pytest.raises(RuntimeError, match="runtime unavailable"):
        asyncio.run(control_plane.execute("Fail observably"))

    stored = repository.list_traces()[0]
    assert stored.status is TraceStatus.FAILED
    assert stored.error == "runtime unavailable"
