from neurorouter.rag.chunker import Chunker
from neurorouter.rag.models import DocumentSection, DocumentType, LoadedDocument


def _document(text: str) -> LoadedDocument:
    return LoadedDocument(
        document_id="doc-1",
        name="notes.md",
        document_type=DocumentType.MARKDOWN,
        sections=[DocumentSection(text=text, page=3)],
        metadata={"topic": "routing"},
    )


def test_chunker_preserves_source_and_overlap() -> None:
    chunks = Chunker(chunk_size=40, overlap=10).chunk(
        _document("Probabilistic routing separates judgments. Deterministic policy creates plans.")
    )

    assert len(chunks) >= 2
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(chunk.document == "notes.md" for chunk in chunks)
    assert chunks[0].text[-10:].strip() in chunks[1].text


def test_chunk_ids_are_stable_and_empty_sections_are_skipped() -> None:
    chunker = Chunker(chunk_size=30, overlap=5)
    document = _document("stable chunks produce stable identifiers")

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert chunker.chunk(_document("   \n\n ")) == []
