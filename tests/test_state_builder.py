import pytest

from neurorouter.core.state_builder import StateBuilder
from neurorouter.schemas.state import MessageRole
from neurorouter.utils.config import load_settings


def test_state_builder_builds_valid_secret_free_state() -> None:
    state = StateBuilder(load_settings()).build(
        "Summarize the uploaded reports",
        recent_messages=[{"role": "user", "content": "Focus on revenue"}],
        attachment_count=2,
        attachment_types=["PDF", "pdf"],
        indexed_document_count=4,
        collection_metadata={"collection": "reports", "documents": 4},
    )

    assert state.conversation.recent_messages[0].role is MessageRole.USER
    assert state.attachments.count == 2
    assert state.attachments.available_document_types == ["pdf"]
    assert state.knowledge_base.has_indexed_documents is True
    assert "api_key" not in state.model_dump_json().lower()


def test_state_builder_rejects_secret_like_runtime_configuration() -> None:
    with pytest.raises(ValueError, match="secret-like"):
        StateBuilder(load_settings()).build(
            "hello",
            collection_metadata={"providers": [{"provider_api_key": "must-not-leak"}]},
        )
