from neurorouter.tools.code_runner import (
    CodeExecutionRequest,
    DisabledCodeExecutionTool,
    RestrictedPythonRunner,
    build_code_execution_tool,
)
from neurorouter.utils.config import load_settings


def test_disabled_runner_never_executes() -> None:
    result = DisabledCodeExecutionTool().execute(
        CodeExecutionRequest(source="print('should not run')")
    )

    assert result.rejected is True
    assert result.stdout == ""
    assert "disabled" in (result.rejection_reason or "").lower()


def test_factory_honors_disabled_configuration() -> None:
    tool = build_code_execution_tool(load_settings().code_execution)

    assert isinstance(tool, DisabledCodeExecutionTool)


def test_restricted_runner_captures_stdout_and_stderr() -> None:
    runner = RestrictedPythonRunner(timeout_seconds=1)

    successful = runner.execute(CodeExecutionRequest(source="print(sum([2, 3, 5]))"))
    failing = runner.execute(CodeExecutionRequest(source="print(1 / 0)"))

    assert successful.exit_code == 0
    assert successful.stdout.strip() == "10"
    assert failing.exit_code == 1
    assert "ZeroDivisionError" in failing.stderr


def test_restricted_runner_rejects_dangerous_language_features() -> None:
    runner = RestrictedPythonRunner()

    imported = runner.execute(CodeExecutionRequest(source="import os\nprint(os.getcwd())"))
    opened = runner.execute(CodeExecutionRequest(source="print(open('secret.txt').read())"))
    unsupported = runner.execute(CodeExecutionRequest(source="print(1)", language="javascript"))

    assert imported.rejected is True
    assert "Import" in (imported.rejection_reason or "")
    assert opened.rejected is True
    assert unsupported.rejected is True


def test_restricted_runner_enforces_timeout_and_source_limit() -> None:
    runner = RestrictedPythonRunner(timeout_seconds=0.01, max_source_characters=100)

    timeout = runner.execute(
        CodeExecutionRequest(source="total = 0\nfor index in range(1000000):\n total += index")
    )
    oversized = runner.execute(CodeExecutionRequest(source="value = 1\n" * 20))

    assert timeout.timed_out is True
    assert oversized.rejected is True
    assert "character limit" in (oversized.rejection_reason or "")


def test_restricted_runner_caps_captured_output() -> None:
    result = RestrictedPythonRunner(max_output_characters=1000).execute(
        CodeExecutionRequest(source="print('x' * 10000)")
    )

    assert result.exit_code == 0
    assert result.output_truncated is True
    assert len(result.stdout) < 1100
