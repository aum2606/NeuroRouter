"""Restricted code-execution boundary for optional local calculations."""

import ast
import os
import subprocess
import sys
import tempfile
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from neurorouter.utils.config import CodeExecutionSettings


class CodeRunnerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CodeExecutionRequest(CodeRunnerModel):
    source: str = Field(min_length=1)
    language: str = "python"


class CodeExecutionResult(CodeRunnerModel):
    language: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    rejected: bool = False
    output_truncated: bool = False
    rejection_reason: str | None = None


class CodeExecutionTool(Protocol):
    @property
    def enabled(self) -> bool: ...

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult: ...


class DisabledCodeExecutionTool:
    """Explicit no-execution implementation used by default."""

    enabled = False

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        return CodeExecutionResult(
            language=request.language,
            rejected=True,
            rejection_reason="Code execution is disabled by configuration.",
        )


class RestrictedPythonRunner:
    """Execute a small calculation-oriented Python subset in an isolated process.

    This boundary is intentionally not a general-purpose Python sandbox. It rejects
    imports, attributes, definitions, filesystem-capable builtins, and other dynamic
    language features before starting a short-lived isolated interpreter.
    """

    enabled = True
    _ALLOWED_CALLS = {
        "abs",
        "all",
        "any",
        "bool",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "round",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    _REJECTED_NODES = (
        ast.AsyncFunctionDef,
        ast.Attribute,
        ast.Await,
        ast.ClassDef,
        ast.Delete,
        ast.FunctionDef,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.While,
        ast.With,
        ast.Yield,
        ast.YieldFrom,
    )
    _OUTPUT_MARKER = "\n[output truncated]\n"
    _BOOTSTRAP = """
import builtins
import sys

remaining = [int(sys.argv[2])]
marker = "\\n[output truncated]\\n"

def safe_print(*values, sep=" ", end="\\n"):
    rendered = sep.join(str(value) for value in values) + end
    if remaining[0] <= 0:
        return
    if len(rendered) > remaining[0]:
        sys.stdout.write(rendered[:remaining[0]] + marker)
        remaining[0] = 0
        return
    sys.stdout.write(rendered)
    remaining[0] -= len(rendered)

allowed_names = {names}
safe_builtins = {name: getattr(builtins, name) for name in allowed_names}
safe_builtins["print"] = safe_print
exec(compile(sys.argv[1], "<restricted-code>", "exec"), {"__builtins__": safe_builtins})
""".strip().replace("{names}", repr(_ALLOWED_CALLS))

    def __init__(
        self,
        *,
        timeout_seconds: float = 2.0,
        max_source_characters: int = 10000,
        max_output_characters: int = 20000,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_source_characters < 1:
            raise ValueError("max_source_characters must be positive")
        if max_output_characters < 1:
            raise ValueError("max_output_characters must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_source_characters = max_source_characters
        self.max_output_characters = max_output_characters

    def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        if request.language.casefold() != "python":
            return self._rejected(request.language, "Only Python is supported.")
        rejection = self._validate(request.source)
        if rejection:
            return self._rejected(request.language, rejection)

        environment = {"PYTHONIOENCODING": "utf-8"}
        if system_root := os.environ.get("SYSTEMROOT"):
            environment["SYSTEMROOT"] = system_root
        try:
            with tempfile.TemporaryDirectory(prefix="neurorouter-code-") as directory:
                completed = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-c",
                        self._BOOTSTRAP,
                        request.source,
                        str(self.max_output_characters),
                    ],
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
            return CodeExecutionResult(
                language="python",
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
                output_truncated=self._OUTPUT_MARKER in completed.stdout,
            )
        except subprocess.TimeoutExpired as error:
            return CodeExecutionResult(
                language="python",
                stdout=self._decode_timeout_stream(error.stdout),
                stderr=self._decode_timeout_stream(error.stderr),
                timed_out=True,
            )

    def _validate(self, source: str) -> str | None:
        if len(source) > self.max_source_characters:
            return f"Source exceeds the {self.max_source_characters}-character limit."
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return f"Python syntax error at line {error.lineno}: {error.msg}"
        for node in ast.walk(tree):
            if isinstance(node, self._REJECTED_NODES):
                return f"{type(node).__name__} is not allowed."
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                return "Private and dunder names are not allowed."
            if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name) or node.func.id not in self._ALLOWED_CALLS
            ):
                return "Only approved calculation builtins may be called."
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and abs(node.value) > 10**6
            ):
                return "Integer literals above 1,000,000 are not allowed."
        return None

    @staticmethod
    def _rejected(language: str, reason: str) -> CodeExecutionResult:
        return CodeExecutionResult(language=language, rejected=True, rejection_reason=reason)

    @staticmethod
    def _decode_timeout_stream(value: bytes | str | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""


def build_code_execution_tool(settings: CodeExecutionSettings) -> CodeExecutionTool:
    """Create the configured tool without allowing implicit execution enablement."""
    if not settings.enabled:
        return DisabledCodeExecutionTool()
    return RestrictedPythonRunner(
        timeout_seconds=settings.timeout_seconds,
        max_source_characters=settings.max_source_characters,
        max_output_characters=settings.max_output_characters,
    )
