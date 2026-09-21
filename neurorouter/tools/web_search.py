"""Pluggable web-search providers with a no-key Wikipedia implementation."""

import asyncio
import html
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WebSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snippet: str = ""
    provider: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("search result URL must be HTTP(S)")
        return value


@runtime_checkable
class WebSearchProvider(Protocol):
    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        """Return normalized results ordered by provider relevance."""
        ...


class WikipediaSearchProvider:
    """Search English Wikipedia through the keyless MediaWiki REST API."""

    endpoint = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        if not query.strip():
            raise ValueError("search query cannot be empty")
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        payload = await asyncio.to_thread(self._request, query.strip(), max_results)
        return self.parse_payload(payload, max_results=max_results)

    def _request(self, query: str, max_results: int) -> Mapping[str, Any]:
        url = f"{self.endpoint}?{urlencode({'q': query, 'limit': max_results})}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "NeuroRouter/0.1 (https://github.com/aum2606/NeuroRouter)",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Wikipedia returned a non-object response")
        return payload

    @classmethod
    def parse_payload(
        cls, payload: Mapping[str, Any], *, max_results: int
    ) -> list[WebSearchResult]:
        pages = payload.get("pages", [])
        if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
            raise ValueError("Wikipedia response is missing the pages array")
        results: list[WebSearchResult] = []
        for rank, page in enumerate(pages[:max_results]):
            if not isinstance(page, Mapping):
                continue
            title = str(page.get("title") or "").strip()
            key = str(page.get("key") or title.replace(" ", "_")).strip()
            if not title or not key:
                continue
            excerpt = cls._plain_text(str(page.get("excerpt") or ""))
            description = cls._plain_text(str(page.get("description") or ""))
            snippet = excerpt or description
            encoded_key = quote(key, safe="()_'-")
            results.append(
                WebSearchResult(
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{encoded_key}",
                    snippet=snippet,
                    provider="wikipedia",
                    score=1.0 / (rank + 1),
                    metadata={
                        "rank": rank + 1,
                        "description": description,
                        "page_id": page.get("id"),
                        "coverage": "english_wikipedia",
                    },
                )
            )
        return results

    @staticmethod
    def _plain_text(value: str) -> str:
        without_tags = re.sub(r"<[^>]+>", " ", value)
        return " ".join(html.unescape(without_tags).split())


class StaticWebSearchProvider:
    """Deterministic no-network provider for tests and recorded demos."""

    def __init__(self, results: Mapping[str, Sequence[WebSearchResult]]) -> None:
        self.results = {query: list(items) for query, items in results.items()}

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        return [item.model_copy(deep=True) for item in self.results.get(query, [])[:max_results]]
