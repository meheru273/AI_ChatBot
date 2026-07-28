"""Liveness and readiness endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.api.schemas import HealthResponse
from app.config import get_settings
from app.db.mongo import get_database
from app.rag.ingest import get_vector_store
from app.tools import get_tool_names

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/")
async def root() -> dict[str, str]:
    """Liveness — cheap, no dependency checks."""
    return {"status": "healthy", "service": "Banchan RAG Assistant"}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Readiness — reports which subsystems actually came up.

    Degraded dependencies are reported rather than raised: the service stays
    usable with an unbuilt index or an unreachable database, just with fewer
    tools working, and this endpoint says which.
    """
    settings = get_settings()

    mongo_connected = False
    try:
        await get_database().command("ping")
        mongo_connected = True
    except Exception as exc:  # noqa: BLE001 - health must never raise
        logger.warning("Mongo health check failed: %s", exc)

    store = get_vector_store()

    return HealthResponse(
        status="healthy" if mongo_connected else "degraded",
        mongo_connected=mongo_connected,
        index_loaded=store is not None,
        index_chunks=len(store) if store else None,
        embedding_provider=store.spec.describe() if store else None,
        model=settings.groq_model,
        tools=get_tool_names(),
    )
