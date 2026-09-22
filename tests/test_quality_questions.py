from typesafe_sdk import Noul

from neurorouter.jev.quality_questions import build_quality_questions


def test_quality_questions_are_six_atomic_nouls_in_one_batch() -> None:
    questions = build_quality_questions()

    assert set(questions) == {
        "answers_request",
        "supported_by_evidence",
        "contains_unsupported_claims",
        "missing_important_information",
        "contradicts_evidence",
        "needs_additional_retrieval",
    }
    assert all(isinstance(question, Noul) for question in questions.values())
    assert all(question.criteria is not None for question in questions.values())
    assert "candidate_response" in str(questions["answers_request"].instructions)
