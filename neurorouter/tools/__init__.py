"""Provider-neutral external tool interfaces."""

from neurorouter.tools.web_search import (
    StaticWebSearchProvider,
    WebSearchProvider,
    WebSearchResult,
    WikipediaSearchProvider,
)

__all__ = [
    "StaticWebSearchProvider",
    "WebSearchProvider",
    "WebSearchResult",
    "WikipediaSearchProvider",
]
