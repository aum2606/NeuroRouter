"""Bounded coding specialist with optional restricted execution."""

import asyncio
import re

from neurorouter.agents.base import AgentInput, AgentPayload, BaseAgent, Evidence
from neurorouter.schemas.execution import AgentName
from neurorouter.tools.code_runner import (
    CodeExecutionRequest,
    CodeExecutionTool,
    DisabledCodeExecutionTool,
)


class CodeAgent(BaseAgent):
    """Inspect code requests and optionally execute an explicitly supplied snippet."""

    name = AgentName.CODE

    def __init__(self, execution_tool: CodeExecutionTool | None = None) -> None:
        self.execution_tool = execution_tool or DisabledCodeExecutionTool()

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        task_type = self._task_type(agent_input.request_text, agent_input.plan.use_data_analysis)
        source = self._source(agent_input)
        diagnostics = self._syntax_diagnostics(source) if source else []
        execution_requested = bool(agent_input.context.get("execute_code", False))
        execution = None
        if execution_requested and source:
            execution = await asyncio.to_thread(
                self.execution_tool.execute,
                CodeExecutionRequest(source=source, language="python"),
            )

        evidence: list[Evidence] = []
        if execution and (execution.stdout or execution.stderr):
            captured = "\n".join(
                part
                for part in (
                    f"stdout:\n{execution.stdout.strip()}" if execution.stdout else "",
                    f"stderr:\n{execution.stderr.strip()}" if execution.stderr else "",
                )
                if part
            )
            evidence.append(
                Evidence(
                    evidence_id="local-code-execution",
                    content=captured,
                    relevance_score=1.0,
                    metadata={"origin": "restricted_local_execution"},
                )
            )

        if execution is not None:
            status = (
                "rejected"
                if execution.rejected
                else "timed out"
                if execution.timed_out
                else f"completed with exit code {execution.exit_code}"
            )
            output = f"Code task classified as {task_type}; requested execution {status}."
        elif source:
            output = (
                f"Code task classified as {task_type}; inspected one Python snippet and prepared "
                "it for synthesis. Execution was not requested."
            )
        else:
            output = (
                f"Code task classified as {task_type}. No executable snippet was supplied; "
                "the synthesis stage should produce or explain code for the request."
            )

        return AgentPayload(
            output=output,
            evidence=evidence,
            metadata={
                "task_type": task_type,
                "language": "python" if source else None,
                "snippet_present": source is not None,
                "syntax_diagnostics": diagnostics,
                "execution_requested": execution_requested,
                "execution_enabled": self.execution_tool.enabled,
                "execution": execution.model_dump(mode="json") if execution else None,
            },
        )

    @staticmethod
    def _source(agent_input: AgentInput) -> str | None:
        supplied = agent_input.context.get("code_to_execute")
        if isinstance(supplied, str) and supplied.strip():
            return supplied.strip()
        fenced = re.search(
            r"```(?:python|py)?\s*\n(?P<source>.*?)```",
            agent_input.request_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return fenced.group("source").strip() if fenced else None

    @staticmethod
    def _syntax_diagnostics(source: str) -> list[dict[str, str | int | None]]:
        try:
            compile(source, "<submitted-code>", "exec", flags=0, dont_inherit=True)
        except SyntaxError as error:
            return [
                {
                    "severity": "error",
                    "message": error.msg,
                    "line": error.lineno,
                    "offset": error.offset,
                }
            ]
        return []

    @staticmethod
    def _task_type(request: str, data_analysis: bool) -> str:
        if data_analysis:
            return "data_analysis"
        lowered = request.casefold()
        if any(term in lowered for term in ("bug", "debug", "traceback", "exception", "fix")):
            return "debugging"
        if any(term in lowered for term in ("explain", "review", "what does")):
            return "code_explanation"
        return "code_generation"
