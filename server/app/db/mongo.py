"""Async MongoDB connection management.

The client is created once during FastAPI startup and reused for the process
lifetime — Motor pools connections internally, so creating a client per
request would be both slow and a connection leak.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None


async def init_mongo() -> AsyncIOMotorDatabase:
    """Open the connection pool and verify the cluster is reachable."""
    global _client

    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=5_000,
            maxPoolSize=10,
            minPoolSize=1,
        )
        # Fail loudly at startup rather than on the first user question.
        await _client.admin.command("ping")
        logger.info("Connected to MongoDB database %r", settings.mongo_db)

    return get_database()


async def close_mongo() -> None:
    """Close the connection pool."""
    global _client

    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """Return the active database handle.

    Raises if called before `init_mongo()`, which means the FastAPI lifespan
    did not run — usually a sign of importing the app the wrong way.
    """
    if _client is None:
        raise RuntimeError("MongoDB is not initialised; call init_mongo() first")
    return _client[get_settings().mongo_db]
