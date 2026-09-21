import pytest

from neurorouter.jev.parsers import RoutingParseError, parse_routing_response
from neurorouter.schemas.routing import Intent


def _raw_response() -> dict[str, object]:
    nouls = {
        name: {"type": "noul", "noul": probability}
        for name, probability in {
            "needs_web": 0.91,
            "needs_rag": 0.2,
            "needs_code": 0.1,
            "needs_data_analysis": 0.3,
            "needs_current_information": 0.88,
            "needs_citations": 0.9,
            "needs_multi_source_research": 0.72,
        }.items()
    }
    return {
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "answers": {
            "intent": {
                "type": "choice",
                "choice": "research",
                "confidence": 0.81,
                "probabilities": {"research": 0.81, "general_qa": 0.19},
            },
            **nouls,
            "complexity": {
                "type": "score",
                "score": 2.35,
                "confidence": 0.74,
                "probabilities": {"0": 0.01, "1": 0.09, "2": 0.44, "3": 0.46},
            },
            "risk": {
                "type": "score",
                "score": 0.25,
                "confidence": 0.9,
                "probabilities": {"0": 0.76, "1": 0.23, "2": 0.01, "3": 0.0},
            },
        },
    }


def test_parser_preserves_all_probabilities_and_expected_scores() -> None:
    decision = parse_routing_response(_raw_response(), latency_ms=12.5)

    assert decision.intent is Intent.RESEARCH
    assert decision.needs_web == pytest.approx(0.91)
    assert decision.complexity_score == pytest.approx(2.35)
    assert decision.complexity_probabilities[3] == pytest.approx(0.46)
    assert decision.risk_score == pytest.approx(0.25)
    assert decision.jev_model == "jev-1.13.0"


def test_parser_rejects_missing_atomic_answer() -> None:
    raw = _raw_response()
    del raw["answers"]["needs_web"]  # type: ignore[index]

    with pytest.raises(RoutingParseError, match="needs_web"):
        parse_routing_response(raw, latency_ms=1.0)
