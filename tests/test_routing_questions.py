from typesafe_sdk import Choice, Noul, Score

from neurorouter.jev.routing_questions import INTENT_CRITERIA, build_routing_questions


def test_routing_questions_are_atomic_and_batched() -> None:
    questions = build_routing_questions()

    assert len(questions) == 10
    assert isinstance(questions["intent"], Choice)
    assert set(questions["intent"].criteria) == set(INTENT_CRITERIA)
    assert all(
        isinstance(questions[name], Noul)
        for name in (
            "needs_web",
            "needs_rag",
            "needs_code",
            "needs_data_analysis",
            "needs_current_information",
            "needs_citations",
            "needs_multi_source_research",
        )
    )
    assert isinstance(questions["complexity"], Score)
    assert isinstance(questions["risk"], Score)
    assert len(questions["complexity"].criteria) == 4
    assert len(questions["risk"].criteria) == 4
