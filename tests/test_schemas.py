from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from neurorouter.schemas.routing import Intent, RoutingDecision
from neurorouter.schemas.state import AttachmentState, RequestState, RouterState


def test_router_state_serializes_without_secrets() -> None:
    state = RouterState(request=RequestState(text="Summarize the attached report"))

    payload = state.model_dump(mode="json")

    assert payload["request"]["text"] == "Summarize the attached report"
    assert "api_key" not in str(payload).lower()


def test_attachment_types_require_an_attachment() -> None:
    with pytest.raises(ValidationError):
        AttachmentState(count=0, available_document_types=["pdf"])


def test_routing_decision_preserves_probability_distributions() -> None:
    decision = RoutingDecision(
        intent=Intent.RESEARCH,
        intent_confidence=0.8,
        intent_probabilities={Intent.RESEARCH: 0.8, Intent.GENERAL_QA: 0.2},
        needs_web=0.91,
        needs_rag=0.1,
        needs_code=0.05,
        needs_data_analysis=0.2,
        needs_current_information=0.88,
        needs_citations=0.9,
        needs_multi_source_research=0.75,
        complexity_score=2,
        complexity_confidence=0.7,
        complexity_probabilities={0: 0.05, 1: 0.2, 2: 0.7, 3: 0.05},
        risk_score=0,
        risk_confidence=0.9,
        risk_probabilities={0: 0.9, 1: 0.08, 2: 0.01, 3: 0.01},
        jev_latency_ms=42.0,
        jev_model="jev-latest",
    )

    assert decision.needs_web == pytest.approx(0.91)
    assert decision.intent_probabilities[Intent.RESEARCH] == pytest.approx(0.8)


def test_request_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        RequestState(text="", timestamp=datetime.now(UTC))
