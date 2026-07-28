"""FAISS-backed vector store with a provider guard.

Vectors are L2-normalised and stored in an inner-product index, which makes
the similarity score exact cosine similarity in [-1, 1].

Persisted as three files so the index is inspectable rather than an opaque
pickle:

    index.faiss   the FAISS index
    chunks.json   chunk text + metadata, positionally aligned with the index
    meta.json     the EmbeddingSpec the index was built with, plus counts
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.rag.chunking import Chunk
from app.rag.embeddings import EmbeddingProvider, EmbeddingSpec

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


class IndexMismatchError(RuntimeError):
    """Raised when an index was built in a different vector space."""


class IndexMissingError(FileNotFoundError):
    """Raised when no index exists at the configured path."""


@dataclass
class RetrievedChunk:
    """A search hit."""

    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    def cite(self) -> str:
        """Render the hit with its provenance, for the LLM to quote."""
        return f"[source: {self.source}]\n{self.text}"


def _to_matrix(vectors: list[list[float]]) -> np.ndarray:
    """Normalise vectors into a float32 matrix FAISS can consume."""
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    faiss.normalize_L2(matrix)
    return matrix


class VectorStore:
    """An in-memory FAISS index over a fixed set of chunks."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        index: faiss.Index,
        chunks: list[Chunk],
        spec: EmbeddingSpec,
    ) -> None:
        self._provider = provider
        self._index = index
        self._chunks = chunks
        self._spec = spec

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    # ---- construction ----

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        provider: EmbeddingProvider,
        batch_size: int = 64,
    ) -> VectorStore:
        """Embed every chunk and construct the index."""
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks")

        spec = provider.spec
        logger.info("Embedding %d chunks with %s", len(chunks), spec.describe())

        vectors: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors.extend(provider.embed_documents([c.text for c in batch]))

        matrix = _to_matrix(vectors)
        if matrix.shape[1] != spec.dimension:
            raise IndexMismatchError(
                f"Provider {spec.describe()} returned {matrix.shape[1]}-dim vectors"
            )

        index = faiss.IndexFlatIP(spec.dimension)
        index.add(matrix)

        return cls(provider=provider, index=index, chunks=chunks, spec=spec)

    # ---- persistence ----

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / INDEX_FILE))

        (path / CHUNKS_FILE).write_text(
            json.dumps(
                [{"text": c.text, "metadata": c.metadata} for c in self._chunks],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        (path / META_FILE).write_text(
            json.dumps(
                {
                    **self._spec.as_dict(),
                    "chunk_count": len(self._chunks),
                    "built_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info("Saved index of %d chunks to %s", len(self._chunks), path)

    @classmethod
    def load(cls, path: Path, provider: EmbeddingProvider) -> VectorStore:
        """Load an index, refusing one built in a different vector space."""
        index_path = path / INDEX_FILE
        if not index_path.exists():
            raise IndexMissingError(
                f"No vector index at {path}. Build one with:\n"
                f"    python -m scripts.build_index"
            )

        stored = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        stored_spec = EmbeddingSpec(
            provider=stored["provider"],
            model=stored["model"],
            dimension=int(stored["dimension"]),
        )

        # Compare identities, not full specs: identity is provider + model and
        # costs nothing, whereas reading `provider.dimension` can trigger a
        # probe request — an API round trip on every cold start.
        if (stored_spec.provider, stored_spec.model) != provider.identity:
            current = "/".join(provider.identity)
            raise IndexMismatchError(
                f"Index was built with {stored_spec.describe()} but the current "
                f"configuration uses {current}. Rebuild it with:\n"
                f"    python -m scripts.build_index"
            )

        # Same vector space, so the recorded width is authoritative.
        provider.set_dimension(stored_spec.dimension)

        index = faiss.read_index(str(index_path))
        raw = json.loads((path / CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(text=r["text"], metadata=r.get("metadata", {})) for r in raw]

        if index.ntotal != len(chunks):
            raise IndexMismatchError(
                f"Index holds {index.ntotal} vectors but {len(chunks)} chunks "
                f"were loaded; the index directory is corrupt. Rebuild it."
            )

        if index.d != stored_spec.dimension:
            raise IndexMismatchError(
                f"Index is {index.d}-dimensional but its metadata claims "
                f"{stored_spec.dimension}; the index directory is corrupt. "
                f"Rebuild it."
            )

        logger.info("Loaded index of %d chunks (%s)", len(chunks), stored_spec.describe())
        return cls(provider=provider, index=index, chunks=chunks, spec=stored_spec)

    # ---- query ----

    def search(
        self,
        query: str,
        k: int = 4,
        doc_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the k nearest chunks to `query`.

        `doc_type` restricts results by metadata; because FAISS has no
        metadata filter, the search over-fetches and filters afterwards.
        """
        if not query.strip() or len(self._chunks) == 0:
            return []

        fetch = min(k * 4, len(self._chunks)) if doc_type else min(k, len(self._chunks))

        matrix = _to_matrix([self._provider.embed_query(query)])
        scores, indices = self._index.search(matrix, fetch)

        hits: list[RetrievedChunk] = []
        # FAISS returns both arrays with shape (1, fetch), so strict=True is
        # asserting an invariant rather than guarding against ragged input.
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:  # FAISS pads with -1 when fewer results exist
                continue
            chunk = self._chunks[int(idx)]
            if doc_type and chunk.metadata.get("type") != doc_type:
                continue
            if float(score) < min_score:
                continue
            hits.append(
                RetrievedChunk(
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )
            if len(hits) >= k:
                break

        return hits
