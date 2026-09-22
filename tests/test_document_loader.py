from pathlib import Path

import pytest
from pypdf import PdfWriter

from neurorouter.rag.models import DocumentType
from neurorouter.tools.document_loader import (
    DocumentLoader,
    DocumentLoadError,
    UnsupportedDocumentError,
)


@pytest.mark.parametrize(
    ("suffix", "document_type"),
    [(".txt", DocumentType.TEXT), (".md", DocumentType.MARKDOWN)],
)
def test_loads_utf8_text_documents(
    tmp_path: Path, suffix: str, document_type: DocumentType
) -> None:
    path = tmp_path / f"guide{suffix}"
    path.write_text("NeuroRouter preserves local evidence.", encoding="utf-8")

    loaded = DocumentLoader().load(path)

    assert loaded.document_type is document_type
    assert loaded.sections[0].text == "NeuroRouter preserves local evidence."
    assert loaded.metadata["size_bytes"] > 0
    assert len(loaded.document_id) == 64


def test_loads_pdf_pages_with_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)

    loaded = DocumentLoader().load(path)

    assert loaded.document_type is DocumentType.PDF
    assert len(loaded.sections) == 1
    assert loaded.sections[0].page == 1


def test_rejects_unsupported_and_oversized_documents(tmp_path: Path) -> None:
    unsupported = tmp_path / "data.csv"
    unsupported.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentError):
        DocumentLoader().load(unsupported)

    oversized = tmp_path / "large.txt"
    oversized.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(DocumentLoadError, match="exceeds"):
        DocumentLoader(max_file_size_mb=1).load(oversized)


def test_load_bytes_sanitizes_upload_name_and_preserves_type() -> None:
    loaded = DocumentLoader().load_bytes(
        "..\\../architecture.md",
        b"# NeuroRouter\n\nProbabilities remain separate from policy.",
    )

    assert loaded.name == "architecture.md"
    assert loaded.document_type is DocumentType.MARKDOWN
    assert loaded.metadata["document_type"] == "markdown"
    assert "source_path" not in loaded.metadata
