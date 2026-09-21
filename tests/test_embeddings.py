import math

import pytest

from neurorouter.rag.embeddings import HashingEmbeddingProvider


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashingEmbeddingProvider(dimensions=128)

    first = provider.embed_query("probabilistic routing policy")
    second = provider.embed_documents(["probabilistic routing policy"])[0]

    assert first == second
    assert len(first) == 128
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_empty_embedding_is_zero_vector() -> None:
    assert HashingEmbeddingProvider(dimensions=64).embed_query("") == [0.0] * 64
