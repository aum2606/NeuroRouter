"""Bounded specialist agent interfaces and initial implementations."""

from neurorouter.agents.base import AgentInput, AgentResult, BaseAgent
from neurorouter.agents.code import CodeAgent
from neurorouter.agents.finance import FinanceAgent
from neurorouter.agents.general import GeneralAgent
from neurorouter.agents.rag import RAGAgent
from neurorouter.agents.research import WebResearchAgent

__all__ = [
    "AgentInput",
    "AgentResult",
    "BaseAgent",
    "CodeAgent",
    "FinanceAgent",
    "GeneralAgent",
    "RAGAgent",
    "WebResearchAgent",
]
