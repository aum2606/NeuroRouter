"""Safe local extraction for the document formats supported by NeuroRouter."""

import hashlib
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

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
        if not source.is_file():
            raise DocumentLoadError(f"Document does not exist: {source}")
        try:
            content = source.read_bytes()
        except OSError as error:
            raise DocumentLoadError(f"Could not load {source.name}: {error}") from error
        return self.load_bytes(source.name, content, source_path=source.resolve())

    def load_bytes(
        self,
        filename: str,
        content: bytes,
        *,
        source_path: Path | None = None,
    ) -> LoadedDocument:
        """Extract an uploaded document without writing it to a temporary file."""
        safe_name = Path(filename.replace("\\", "/")).name
        document_type = self._TYPE_BY_SUFFIX.get(Path(safe_name).suffix.casefold())
        if document_type is None:
            raise UnsupportedDocumentError(
                f"Unsupported document type '{Path(safe_name).suffix}'. Use PDF, TXT, or Markdown."
            )
        if len(content) > self.max_file_size_bytes:
            raise DocumentLoadError(
                f"Document exceeds the {self.max_file_size_bytes // (1024 * 1024)} MB limit"
            )
        try:
            sections = (
                self._load_pdf(BytesIO(content), safe_name)
                if document_type is DocumentType.PDF
                else [DocumentSection(text=content.decode("utf-8-sig"))]
            )
        except DocumentLoadError:
            raise
        except (OSError, UnicodeDecodeError) as error:
            raise DocumentLoadError(f"Could not load {safe_name}: {error}") from error
        metadata: dict[str, str | int] = {
            "size_bytes": len(content),
            "document_type": document_type.value,
        }
        if source_path is not None:
            metadata["source_path"] = str(source_path)
        return LoadedDocument(
            document_id=hashlib.sha256(content).hexdigest(),
            name=safe_name,
            document_type=document_type,
            sections=sections,
            metadata=metadata,
        )

    @staticmethod
    def _load_pdf(source: BinaryIO, filename: str) -> list[DocumentSection]:
        try:
            from pypdf import PdfReader
        except ImportError as error:  # pragma: no cover - installation guidance
            raise DocumentLoadError(
                "Install NeuroRouter with the 'rag' extra to load PDF documents"
            ) from error
        try:
            reader = PdfReader(source)
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
            raise DocumentLoadError(f"Could not extract PDF {filename}: {error}") from error
