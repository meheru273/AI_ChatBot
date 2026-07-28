"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, health
from app.config import get_settings
from app.db.mongo import close_mongo, init_mongo
from app.rag.ingest import get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open shared resources once per process.

    Both the Mongo pool and the FAISS index are expensive to create and safe to
    share, so they are established at startup rather than per request. Failures
    are logged rather than fatal: a degraded service that can still answer some
    questions beats one that will not boot, and /health reports which parts
    came up.
    """
    settings = get_settings()

    try:
        await init_mongo()
    except Exception as exc:  # noqa: BLE001
        logger.error("MongoDB unavailable at startup: %s", exc)

    try:
        store = get_vector_store()
        if store is not None:
            logger.info("Vector index ready: %d chunks", len(store))
    except Exception as exc:  # noqa: BLE001
        logger.error("Vector index unavailable at startup: %s", exc)

    logger.info("Listening on port %d", settings.port)
    yield

    await close_mongo()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Banchan RAG Assistant",
        description=(
            "Hybrid retrieval chatbot: LangGraph agent routing between FAISS "
            "semantic search and structured MongoDB queries."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)

    return app


app = create_app()
