import pytest

from neurorouter.jev.quality_parsers import QualityParseError, parse_quality_response


def _raw(**overrides: float) -> dict:
    probabilities = {
        "answers_request": 0.91,
        "supported_by_evidence": 0.84,
        "contains_unsupported_claims": 0.12,
        "missing_important_information": 0.2,
        "contradicts_evidence": 0.04,
        "needs_additional_retrieval": 0.15,
        **overrides,
    }
    return {
        "model": "jev-1.13.0",
        "answers": {
            name: {"type": "noul", "noul": probability}
            for name, probability in probabilities.items()
        },
    }


def test_quality_parser_preserves_noul_yes_probabilities() -> None:
    decision = parse_quality_response(_raw(), latency_ms=12.5)

    assert decision.answers_request == pytest.approx(0.91)
    assert decision.contains_unsupported_claims == pytest.approx(0.12)
    assert decision.jev_latency_ms == pytest.approx(12.5)
    assert decision.jev_model == "jev-1.13.0"


def test_quality_parser_rejects_missing_or_wrong_answer_types() -> None:
    missing = _raw()
    del missing["answers"]["contradicts_evidence"]
    with pytest.raises(QualityParseError, match="contradicts_evidence"):
        parse_quality_response(missing, latency_ms=1)

    wrong = _raw()
    wrong["answers"]["answers_request"]["type"] = "choice"
    with pytest.raises(QualityParseError, match="type 'noul'"):
        parse_quality_response(wrong, latency_ms=1)
