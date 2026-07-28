"""Document chunking.

Two shapes of source text need different treatment:

* **Knowledge markdown** is long-form prose and gets split, preferring
  paragraph and heading boundaries so a chunk rarely cuts mid-sentence.
* **Menu passages** are already short, self-contained descriptions of one
  dish. Splitting them would strand a price away from its dish name, so they
  pass through whole.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    """A unit of text with its provenance, ready to embed."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Ordered coarse to fine: keep sections together where possible.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_document(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split long-form text, carrying metadata onto every resulting chunk."""
    if not text.strip():
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    pieces = splitter.split_text(text)

    return [
        Chunk(
            text=piece,
            metadata={**metadata, "chunk_index": i, "chunk_total": len(pieces)},
        )
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]
