from pathlib import Path

from neurorouter.rag.embeddings import HashingEmbeddingProvider
from neurorouter.rag.index import ChromaVectorStore
from neurorouter.rag.models import TextChunk


def test_chroma_persists_and_queries_supplied_embeddings(tmp_path: Path) -> None:
    provider = HashingEmbeddingProvider(64)
    chunks = [
        TextChunk(
            chunk_id="chunk-routing",
            document_id="doc-1",
            document="architecture.md",
            text="Jev produces probabilistic routing judgments.",
            chunk_index=0,
            page=2,
        ),
        TextChunk(
            chunk_id="chunk-cooking",
            document_id="doc-2",
            document="recipes.txt",
            text="Bake bread in a hot oven.",
            chunk_index=0,
        ),
    ]
    store = ChromaVectorStore(tmp_path / "chroma", "test_collection")
    store.upsert(chunks, provider.embed_documents([chunk.text for chunk in chunks]))

    matches = store.query(provider.embed_query("probabilistic Jev routing"), top_k=2)
    reopened = ChromaVectorStore(tmp_path / "chroma", "test_collection")

    assert matches[0].chunk_id == "chunk-routing"
    assert matches[0].page == 2
    assert reopened.collection_info().chunk_count == 2
    assert reopened.collection_info().document_count == 2
    catalog = reopened.list_documents()
    assert [record.name for record in catalog] == ["architecture.md", "recipes.txt"]
    assert catalog[0].page_count == 1


def test_chroma_rejects_mismatched_embeddings(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "mismatch_collection")

    try:
        store.upsert(
            [
                TextChunk(
                    chunk_id="one",
                    document_id="doc",
                    document="doc.txt",
                    text="text",
                    chunk_index=0,
                )
            ],
            [],
        )
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected mismatched embeddings to fail")
