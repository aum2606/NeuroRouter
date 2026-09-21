"""Safe local extraction for the document formats supported by NeuroRouter."""

import hashlib
from pathlib import Path

from neurorouter.rag.models import DocumentSection, DocumentType, LoadedDocument


class UnsupportedDocumentError(ValueError):
    """Raised when an attachment is not a supported document type."""


class DocumentLoadError(RuntimeError):
    """Raised when a supported file cannot be read or extracted."""


class DocumentLoader:
    """Extract PDF pages or UTF-8 text while preserving source metadata."""

    _TYPE_BY_SUFFIX = {
        ".pdf": DocumentType.PDF,
        ".txt": DocumentType.TEXT,
        ".md": DocumentType.MARKDOWN,
        ".markdown": DocumentType.MARKDOWN,
    }

    def __init__(self, *, max_file_size_mb: int = 25) -> None:
        if max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be positive")
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def load(self, path: Path | str) -> LoadedDocument:
        source = Path(path)
        document_type = self._TYPE_BY_SUFFIX.get(source.suffix.casefold())
        if document_type is None:
            raise UnsupportedDocumentError(
                f"Unsupported document type '{source.suffix}'. Use PDF, TXT, or Markdown."
            )
        if not source.is_file():
            raise DocumentLoadError(f"Document does not exist: {source}")
        if source.stat().st_size > self.max_file_size_bytes:
            raise DocumentLoadError(
                f"Document exceeds the {self.max_file_size_bytes // (1024 * 1024)} MB limit"
            )
        try:
            content = source.read_bytes()
            sections = (
                self._load_pdf(source)
                if document_type is DocumentType.PDF
                else [DocumentSection(text=content.decode("utf-8-sig"))]
            )
        except DocumentLoadError:
            raise
        except (OSError, UnicodeDecodeError) as error:
            raise DocumentLoadError(f"Could not load {source.name}: {error}") from error
        return LoadedDocument(
            document_id=hashlib.sha256(content).hexdigest(),
            name=source.name,
            document_type=document_type,
            sections=sections,
            metadata={"source_path": str(source.resolve()), "size_bytes": len(content)},
        )

    @staticmethod
    def _load_pdf(path: Path) -> list[DocumentSection]:
        try:
            from pypdf import PdfReader
        except ImportError as error:  # pragma: no cover - installation guidance
            raise DocumentLoadError(
                "Install NeuroRouter with the 'rag' extra to load PDF documents"
            ) from error
        try:
            reader = PdfReader(path)
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise DocumentLoadError("Password-protected PDFs are not supported")
            return [
                DocumentSection(
                    text=page.extract_text() or "",
                    page=number,
                    metadata={"source_format": "pdf"},
                )
                for number, page in enumerate(reader.pages, start=1)
            ]
        except DocumentLoadError:
            raise
        except Exception as error:
            raise DocumentLoadError(f"Could not extract PDF {path.name}: {error}") from error
