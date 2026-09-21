"""Build the validated and deliberately secret-free Jev request state."""

from collections.abc import Mapping, Sequence
from typing import Any

from neurorouter import __version__
from neurorouter.schemas.state import (
    AttachmentState,
    CapabilityState,
    ConversationMessage,
    ConversationState,
    KnowledgeBaseState,
    RequestState,
    RouterState,
    SystemState,
)
from neurorouter.utils.config import RuntimeSettings


class StateBuilder:
    """Translate runtime inputs into the stable state sent to Jev."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings

    def build(
        self,
        request_text: str,
        *,
        recent_messages: Sequence[ConversationMessage | Mapping[str, Any]] = (),
        attachment_count: int | None = None,
        attachment_types: Sequence[str] = (),
        indexed_document_count: int = 0,
        collection_metadata: Mapping[str, Any] | None = None,
        relevant_configuration: Mapping[str, Any] | None = None,
    ) -> RouterState:
        normalized_types = sorted(
            {item.strip().lower() for item in attachment_types if item.strip()}
        )
        count = len(attachment_types) if attachment_count is None else attachment_count
        messages = [
            item
            if isinstance(item, ConversationMessage)
            else ConversationMessage.model_validate(item)
            for item in recent_messages
        ]
        capabilities = self.settings.capabilities
        safe_configuration = {
            "jev_model": self.settings.jev.model,
            "llm_provider": self.settings.llm.provider,
            "llm_paid_models_allowed": self.settings.llm.allow_paid_models,
            **dict(relevant_configuration or {}),
        }
        self._reject_secret_keys(safe_configuration)
        self._reject_secret_keys(collection_metadata or {})

        return RouterState(
            request=RequestState(text=request_text),
            conversation=ConversationState(recent_messages=messages),
            attachments=AttachmentState(
                count=count,
                available_document_types=normalized_types,
            ),
            capabilities=CapabilityState(
                web_search_available=capabilities.web_search_available,
                rag_available=capabilities.rag_available,
                code_available=capabilities.code_available,
                finance_agent_available=capabilities.finance_agent_available,
            ),
            knowledge_base=KnowledgeBaseState(
                has_indexed_documents=indexed_document_count > 0,
                collection_metadata=dict(collection_metadata or {}),
            ),
            system=SystemState(
                environment=self.settings.app.environment,
                app_version=__version__,
                relevant_configuration=safe_configuration,
            ),
        )

    @classmethod
    def _reject_secret_keys(cls, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).lower()
                if any(
                    marker in normalized for marker in ("api_key", "secret", "password", "token")
                ):
                    raise ValueError(
                        f"secret-like configuration key is not allowed in RouterState: {key}"
                    )
                cls._reject_secret_keys(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                cls._reject_secret_keys(item)
