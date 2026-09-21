import pytest

from neurorouter.tools.web_search import WikipediaSearchProvider


def test_wikipedia_payload_is_normalized_without_network() -> None:
    results = WikipediaSearchProvider.parse_payload(
        {
            "pages": [
                {
                    "id": 42,
                    "key": "Artificial_intelligence",
                    "title": "Artificial intelligence",
                    "excerpt": '<span class="searchmatch">Artificial</span> intelligence',
                    "description": "Machine intelligence",
                }
            ]
        },
        max_results=5,
    )

    assert len(results) == 1
    assert results[0].url == "https://en.wikipedia.org/wiki/Artificial_intelligence"
    assert results[0].snippet == "Artificial intelligence"
    assert results[0].metadata["coverage"] == "english_wikipedia"


def test_wikipedia_payload_requires_pages_array() -> None:
    with pytest.raises(ValueError, match="pages array"):
        WikipediaSearchProvider.parse_payload({"pages": "invalid"}, max_results=5)
