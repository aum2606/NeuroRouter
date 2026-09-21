"""Deterministic document chunking with stable identifiers."""

import hashlib
import re

from neurorouter.rag.models import LoadedDocument, TextChunk


class Chunker:
    """Split sections near natural boundaries while retaining bounded overlap."""

    def __init__(self, *, chunk_size: int = 1000, overlap: int = 150) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: LoadedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0
        for section in document.sections:
            for text in self._split(section.text):
                chunk_id = hashlib.sha256(
                    f"{document.document_id}:{section.page}:{chunk_index}:{text}".encode()
                ).hexdigest()[:24]
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document=document.name,
                        text=text,
                        chunk_index=chunk_index,
                        page=section.page,
                        metadata={**document.metadata, **section.metadata},
                    )
                )
                chunk_index += 1
        return chunks

    def _split(self, value: str) -> list[str]:
        text = re.sub(r"[ \t]+", " ", value).strip()
        if not text:
            return []
        pieces: list[str] = []
        start = 0
        while start < len(text):
            upper = min(start + self.chunk_size, len(text))
            end = self._boundary(text, start, upper) if upper < len(text) else upper
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            next_start = max(0, end - self.overlap)
            if next_start <= start:
                next_start = end
            start = next_start
        return pieces

    @staticmethod
    def _boundary(text: str, start: int, upper: int) -> int:
        minimum = start + max(1, (upper - start) // 2)
        for separator in ("\n\n", ". ", "\n", " "):
            position = text.rfind(separator, minimum, upper)
            if position >= minimum:
                return position + len(separator)
        return upper
