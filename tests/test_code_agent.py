import asyncio

from neurorouter.agents.code import CodeAgent
from neurorouter.tools.code_runner import RestrictedPythonRunner
from tests.test_agents import _agent_input


def test_code_agent_classifies_and_reports_syntax_errors() -> None:
    request = _agent_input(context={}).model_copy(
        update={"request_text": "Debug this:\n```python\nif True print('broken')\n```"}
    )

    result = asyncio.run(CodeAgent().run(request))

    assert result.success is True
    assert result.metadata["task_type"] == "debugging"
    assert result.metadata["syntax_diagnostics"][0]["severity"] == "error"
    assert result.metadata["execution_enabled"] is False


def test_code_agent_only_executes_when_explicitly_requested() -> None:
    context = {"code_to_execute": "print(round(sum([1, 2, 3]) / 3, 2))"}
    agent = CodeAgent(RestrictedPythonRunner(timeout_seconds=1))

    inspected = asyncio.run(agent.run(_agent_input(context=context)))
    executed = asyncio.run(agent.run(_agent_input(context={**context, "execute_code": True})))

    assert inspected.metadata["execution"] is None
    assert executed.metadata["execution"]["stdout"].strip() == "2.0"
    assert executed.evidence[0].metadata["origin"] == "restricted_local_execution"


def test_code_agent_marks_data_analysis_from_plan() -> None:
    agent_input = _agent_input().model_copy(
        update={"plan": _agent_input().plan.model_copy(update={"use_data_analysis": True})}
    )

    result = asyncio.run(CodeAgent().run(agent_input))

    assert result.metadata["task_type"] == "data_analysis"
    assert result.metadata["snippet_present"] is False
