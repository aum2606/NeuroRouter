"""Minimal provider boundary needed by the Phase 2 router."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class JevClient(Protocol):
    """Submit one state with a batch of independent typed questions."""

    async def evaluate(
        self,
        *,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return the raw provider response without policy interpretation."""
        ...
