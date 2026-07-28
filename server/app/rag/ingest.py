"""Index build pipeline and the process-wide retriever handle."""

from __future__ import annotations

import logging

from app.config import get_settings
from app.db.mongo import close_mongo, get_database, init_mongo
from app.db.repository import MenuRepository
from app.rag.documents import build_corpus
from app.rag.embeddings import get_embedding_provider
from app.rag.vectorstore import IndexMismatchError, IndexMissingError, VectorStore

logger = logging.getLogger(__name__)

_store: VectorStore | None = None


async def build_index() -> VectorStore:
    """Rebuild the vector index from MongoDB and the knowledge corpus.

    Manages its own Mongo lifecycle so it can run as a standalone script.
    """
    settings = get_settings()

    await init_mongo()
    try:
        repo = MenuRepository(get_database())
        chunks = await build_corpus(
            repo=repo,
            knowledge_path=settings.knowledge_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    finally:
        await close_mongo()

    store = VectorStore.build(chunks, get_embedding_provider())
    store.save(settings.faiss_index_path)
    return store


def load_index() -> VectorStore:
    """Load the persisted index into memory, caching it for the process."""
    global _store

    if _store is None:
        settings = get_settings()
        _store = VectorStore.load(settings.faiss_index_path, get_embedding_provider())

    return _store


def get_vector_store() -> VectorStore | None:
    """Return the loaded index, or None if it has not been built yet.

    The API stays up without an index — retrieval-backed answers degrade, but
    the structured menu tools keep working, which is better than refusing to
    boot.
    """
    try:
        return load_index()
    except IndexMissingError:
        logger.warning(
            "No vector index found; retrieval is disabled. "
            "Build one with `python -m scripts.build_index`."
        )
    except IndexMismatchError as exc:
        logger.error("Vector index unusable: %s", exc)
    except ImportError as exc:
        # The local provider's dependencies are optional by design.
        logger.error("Embedding provider unavailable: %s", exc)
    return None


def set_vector_store(store: VectorStore | None) -> None:
    """Override the cached index. Used by tests."""
    global _store
    _store = store
