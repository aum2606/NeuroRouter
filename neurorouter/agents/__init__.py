"""Bounded specialist agent interfaces and initial implementations."""

from neurorouter.agents.base import AgentInput, AgentResult, BaseAgent
from neurorouter.agents.general import GeneralAgent
from neurorouter.agents.research import WebResearchAgent

__all__ = ["AgentInput", "AgentResult", "BaseAgent", "GeneralAgent", "WebResearchAgent"]
