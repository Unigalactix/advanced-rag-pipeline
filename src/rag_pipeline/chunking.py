"""Recursive text splitting with configurable chunk size and overlap."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .ingestion import Document

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A retrievable unit of text derived from a :class:`Document`."""

    text: str
    source: str
    doc_id: str = ""
    chunk_index: int = 0
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raw = f"{self.doc_id}:{self.chunk_index}:{self.text}"
            self.chunk_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        self.metadata.setdefault("source", self.source)


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Split *text* into pieces no longer than *chunk_size* using separators in order."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    parts = text.split(sep)
    pieces: list[str] = []
    current = ""
    joiner = sep
    for part in parts:
        candidate = f"{current}{joiner}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.extend(_recursive_split(current, rest, chunk_size) if len(current) > chunk_size else [current])
            if len(part) > chunk_size:
                pieces.extend(_recursive_split(part, rest, chunk_size))
                current = ""
            else:
                current = part
    if current.strip():
        pieces.extend(_recursive_split(current, rest, chunk_size) if len(current) > chunk_size else [current])
    return [p for p in pieces if p.strip()]


def _apply_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Prefix each piece (after the first) with the tail of the previous piece."""
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        # Keep the overlapped chunk within the size budget when possible.
        merged = f"{tail} {cur}" if len(tail) + 1 + len(cur) <= chunk_size else cur
        out.append(merged)
    return out


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[str]:
    """Split a string into chunks of at most *chunk_size* characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    pieces = _recursive_split(text, separators or DEFAULT_SEPARATORS, chunk_size)
    return _apply_overlap(pieces, chunk_size, chunk_overlap)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Chunk every document, preserving source metadata on each chunk."""
    chunks: list[Chunk] = []
    for doc in documents:
        for idx, piece in enumerate(split_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    doc_id=doc.doc_id,
                    chunk_index=idx,
                    metadata={**doc.metadata, "source": doc.source},
                )
            )
    return chunks
