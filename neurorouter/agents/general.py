"""General-purpose bounded executor used when no specialist is activated."""

from neurorouter.agents.base import AgentInput, AgentPayload, BaseAgent
from neurorouter.schemas.execution import AgentName


class GeneralAgent(BaseAgent):
    """Prepare a request for later synthesis without fabricating external evidence."""

    name = AgentName.GENERAL

    async def execute(self, agent_input: AgentInput) -> AgentPayload:
        return AgentPayload(
            output=(
                "No specialist retrieval was requested. The synthesis stage should answer from "
                "the original request and permitted conversation context."
            ),
            metadata={
                "mode": "synthesis_handoff",
                "request_length": len(agent_input.request_text),
                "external_evidence": False,
            },
        )
