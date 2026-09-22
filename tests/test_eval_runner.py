import asyncio
from types import SimpleNamespace

from neurorouter.core.state_builder import StateBuilder
from neurorouter.evals.runner import EvaluationRunner, JevRoutingPredictor
from neurorouter.evals.schemas import RoutingEvalCase, RoutingPrediction
from neurorouter.schemas.routing import Intent, RoutingDecision
from neurorouter.utils.config import load_settings


def _case(identifier: str) -> RoutingEvalCase:
    return RoutingEvalCase(
        id=identifier,
        query=identifier,
        expected_intent=Intent.GENERAL_QA,
        expected_needs_web=False,
        expected_needs_rag=False,
        expected_needs_code=False,
        expected_needs_data=False,
    )


def _decision() -> RoutingDecision:
    return RoutingDecision(
        intent=Intent.GENERAL_QA,
        intent_confidence=1,
        intent_probabilities={Intent.GENERAL_QA: 1},
        needs_web=0,
        needs_rag=0,
        needs_code=0,
        needs_data_analysis=0,
        needs_current_information=0,
        needs_citations=0,
        needs_multi_source_research=0,
        complexity_score=0,
        complexity_confidence=1,
        complexity_probabilities={0: 1},
        risk_score=0,
        risk_confidence=1,
        risk_probabilities={0: 1},
        jev_latency_ms=1,
    )


class PartialPredictor:
    async def predict(self, case: RoutingEvalCase) -> RoutingPrediction:
        if case.id == "broken":
            raise TimeoutError("provider timeout")
        return RoutingPrediction(
            example_id=case.id,
            decision=_decision(),
            observed_latency_ms=2,
        )


def test_runner_isolates_prediction_failures() -> None:
    report = asyncio.run(
        EvaluationRunner(max_concurrency=2).run(
            [_case("good"), _case("broken")],
            PartialPredictor(),
            router_name="test-router",
            dataset_name="test.jsonl",
        )
    )

    assert report.metrics.evaluated_examples == 1
    assert report.metrics.failed_examples == 1
    assert report.predictions[1].error == "TimeoutError: provider timeout"
    assert report.router_name == "test-router"


class RecordingRouter:
    def __init__(self) -> None:
        self.state = None

    async def route(self, state):
        self.state = state
        return SimpleNamespace(decision=_decision(), raw_response={})


def test_jev_predictor_builds_declared_document_state() -> None:
    case = _case("document").model_copy(
        update={"attachment_types": ["pdf"], "indexed_document_count": 2}
    )
    router = RecordingRouter()

    prediction = asyncio.run(
        JevRoutingPredictor(StateBuilder(load_settings()), router).predict(case)
    )

    assert prediction.decision is not None
    assert router.state.attachments.available_document_types == ["pdf"]
    assert router.state.knowledge_base.has_indexed_documents is True
    assert router.state.knowledge_base.collection_metadata["document_count"] == 2
