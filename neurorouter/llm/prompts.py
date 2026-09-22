"""Versioned synthesis-prompt loading and evidence-safe rendering."""

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from neurorouter.llm.client import LLMMessage, MessageRole
from neurorouter.schemas.context import EvidencePacket
from neurorouter.schemas.execution import ExecutionPlan
from neurorouter.utils.config import PROJECT_ROOT


class PromptModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SynthesisPrompt(PromptModel):
    system: str = Field(min_length=1)
    user_template: str = Field(min_length=1)


class PromptConfiguration(PromptModel):
    version: int = Field(ge=1)
    synthesis: SynthesisPrompt
    agents: dict[str, Any] = Field(default_factory=dict)


def load_prompts(path: Path | None = None) -> PromptConfiguration:
    prompt_path = path or PROJECT_ROOT / "config" / "prompts.yaml"
    with prompt_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return PromptConfiguration.model_validate(payload)


class SynthesisPromptRenderer:
    """Render only request, deterministic plan, agent outputs, and normalized evidence."""

    def __init__(self, prompt: SynthesisPrompt) -> None:
        self.prompt = prompt

    def render(
        self,
        *,
        request_text: str,
        plan: ExecutionPlan,
        evidence: EvidencePacket,
        citations_required: bool,
    ) -> list[LLMMessage]:
        citation_policy = (
            "Citations are required. Cite factual evidence using only its assigned [S#] label, "
            "and include a Sources section mapping labels to the supplied source URLs."
            if citations_required
            else "Citations are optional, but do not invent or alter source identifiers."
        )
        replacements = {
            "{{request}}": request_text,
            "{{plan}}": json.dumps(
                {
                    "agents": [agent.value for agent in plan.agents],
                    "model_tier": plan.model_tier.value,
                    "citations_required": plan.citations_required,
                },
                indent=2,
            ),
            "{{agent_outputs}}": json.dumps(
                [output.model_dump(mode="json") for output in evidence.agent_outputs], indent=2
            ),
            "{{evidence}}": json.dumps(
                {
                    "items": [item.model_dump(mode="json") for item in evidence.items],
                    "sources": [source.model_dump(mode="json") for source in evidence.sources],
                },
                indent=2,
            ),
            "{{citation_policy}}": citation_policy,
        }
        user_prompt = self.prompt.user_template
        for token, value in replacements.items():
            user_prompt = user_prompt.replace(token, value)
        return [
            LLMMessage(role=MessageRole.SYSTEM, content=self.prompt.system.strip()),
            LLMMessage(role=MessageRole.USER, content=user_prompt.strip()),
        ]
