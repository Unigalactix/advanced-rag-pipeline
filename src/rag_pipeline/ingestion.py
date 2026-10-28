"""Load documents from disk into :class:`Document` dataclasses."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass
class Document:
    """A single source document."""

    text: str
    source: str  # file name, e.g. "rag_explained.md"
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            digest = hashlib.sha256(f"{self.source}:{self.text}".encode()).hexdigest()
            self.doc_id = digest[:16]
        self.metadata.setdefault("source", self.source)


def load_document(path: str | Path) -> Document:
    """Load a single ``.txt`` / ``.md`` file."""
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {p.suffix!r} for {p}; expected .txt or .md")
    text = p.read_text(encoding="utf-8", errors="ignore").strip()
    return Document(text=text, source=p.name, metadata={"path": str(p), "suffix": p.suffix.lower()})


def load_documents(directory: str | Path) -> list[Document]:
    """Load every supported file under *directory* (recursively), sorted by name."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Document directory not found: {root}")
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            doc = load_document(path)
            if doc.text:
                documents.append(doc)
    return documents
