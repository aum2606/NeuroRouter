import asyncio

from neurorouter.core.quality_gate import JevQualityGate
from neurorouter.jev.client import StaticJevClient
from neurorouter.schemas.context import EvidencePacket
from neurorouter.schemas.quality import QualityState


def _state() -> QualityState:
    return QualityState(
        original_request="Explain routing",
        candidate_response="Routing separates judgment from policy.",
        supporting_evidence=EvidencePacket(total_characters=0, dropped_items=0, duplicate_items=0),
        execution_metadata={"retry_index": 0},
    )


def _response() -> dict:
    values = {
        "answers_request": 0.9,
        "supported_by_evidence": 0.8,
        "contains_unsupported_claims": 0.1,
        "missing_important_information": 0.2,
        "contradicts_evidence": 0.05,
        "needs_additional_retrieval": 0.1,
    }
    return {
        "model": "jev-test",
        "answers": {name: {"type": "noul", "noul": value} for name, value in values.items()},
    }


def test_quality_gate_batches_and_normalizes_response() -> None:
    result = asyncio.run(JevQualityGate(StaticJevClient(_response())).evaluate(_state()))

    assert result.evaluation_succeeded is True
    assert result.decision.answers_request == 0.9
    assert result.raw_response["model"] == "jev-test"


def test_quality_gate_exposes_provider_failure_as_conservative_result() -> None:
    result = asyncio.run(JevQualityGate(StaticJevClient({"invalid": True})).evaluate(_state()))

    assert result.evaluation_succeeded is False
    assert result.error_type == "QualityParseError"
    assert result.decision.contains_unsupported_claims == 1.0
    assert result.raw_response["quality_gate_failed"] is True
    assert result.raw_response["provider_response"] == {"invalid": True}
