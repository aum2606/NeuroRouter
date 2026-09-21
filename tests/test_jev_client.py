import asyncio

from neurorouter.jev.client import StaticJevClient, TypeSafeJevClient


class FakeResponse:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"model": "jev-test", "answers": {}, "usage": {}}


class FakeSDKClient:
    def __init__(self) -> None:
        self.call: dict[str, object] | None = None

    async def system_one(self, **kwargs: object) -> FakeResponse:
        self.call = kwargs
        return FakeResponse()


def test_typesafe_adapter_forwards_one_batched_call() -> None:
    sdk = FakeSDKClient()
    client = TypeSafeJevClient(model="jev-test", sdk_client=sdk)

    result = asyncio.run(client.evaluate(state={"request": "x"}, questions={"q": "question"}))

    assert result["model"] == "jev-test"
    assert sdk.call == {
        "state": {"request": "x"},
        "questions": {"q": "question"},
        "model": "jev-test",
    }


def test_static_client_returns_an_isolated_copy() -> None:
    client = StaticJevClient({"answers": {"x": 1}})

    first = asyncio.run(client.evaluate(state={}, questions={}))
    first["answers"]["x"] = 2  # type: ignore[index]
    second = asyncio.run(client.evaluate(state={}, questions={}))

    assert second == {"answers": {"x": 1}}
