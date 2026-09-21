"""Provider-neutral Jev boundary and the TypeSafe SDK adapter."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol, runtime_checkable

from typesafe_sdk import AsyncTypeSafeClient


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


class TypeSafeJevClient:
    """Thin async adapter around ``typesafe-sdk`` System One."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "jev-latest",
        timeout_seconds: float = 30.0,
        sdk_client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._sdk_client = sdk_client

    async def evaluate(
        self,
        *,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self._sdk_client is not None:
            response = await self._sdk_client.system_one(
                state=dict(state), questions=questions, model=self.model
            )
            return response.model_dump(mode="json")

        async with AsyncTypeSafeClient(
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout_seconds,
        ) as client:
            response = await client.system_one(state=dict(state), questions=questions)
        return response.model_dump(mode="json")


class StaticJevClient:
    """No-network client for local demos and deterministic tests."""

    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = dict(response)

    async def evaluate(
        self,
        *,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del state, questions
        return deepcopy(self.response)
