"""Application configuration.

Every environment variable the backend reads is declared here and nowhere else.
Import `get_settings()` rather than touching `os.environ` directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

SERVER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVER_ROOT.parent


class Settings(BaseSettings):
    """Runtime settings, loaded from the environment or a .env file."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", SERVER_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM ----
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.3

    # ---- Embeddings ----
    # "jina"   -> hosted, free tier with a usable rate limit. Default.
    # "voyage" -> hosted, free tier, but 3 req/min without a card on file.
    # "openai" -> hosted, paid.
    # "local"  -> sentence-transformers, offline, pulls in torch (~2GB).
    #             Not viable on serverless or on small instances.
    embedding_provider: Literal["jina", "voyage", "openai", "local"] = "jina"
    jina_api_key: str | None = None
    jina_embedding_model: str = "jina-embeddings-v3"
    voyage_api_key: str | None = None
    voyage_embedding_model: str = "voyage-3-lite"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # ---- MongoDB (read-only user) ----
    # Both spellings are accepted: MONGODB_URI is the Atlas convention, while
    # the Banchan services use MONGO_URI.
    mongo_uri: str = Field(
        validation_alias=AliasChoices("MONGO_URI", "MONGODB_URI", "mongo_uri")
    )
    # The Banchan menu-service database, holding menuitems/categories/restaurants.
    mongo_db: str = "menu"

    # ---- Retrieval ----
    faiss_index_path: Path = SERVER_ROOT / "data" / "index"
    knowledge_path: Path = SERVER_ROOT / "data" / "knowledge"
    retrieval_top_k: int = 4
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ---- Agent ----
    # Caps the chatbot <-> tools loop so a confused model cannot spin forever.
    max_tool_iterations: int = 8

    # ---- Server ----
    port: int = 8000
    cors_origins: str = "http://localhost:3000"

    @property
    def allowed_origins(self) -> list[str]:
        """CORS origins as a list, normalised.

        The browser's `Origin` header is always scheme + host + port and never
        has a trailing slash, so `banchan-rag-app.vercel.app` or
        `https://example.com/` match nothing and every request fails CORS while
        the backend looks perfectly healthy. Both slips are common enough to be
        worth repairing here, loudly.
        """
        origins: list[str] = []

        for raw in self.cors_origins.split(","):
            origin = raw.strip().rstrip("/")
            if not origin:
                continue

            if origin != "*" and "://" not in origin:
                logger.warning(
                    "CORS origin %r has no scheme; assuming https://%s. "
                    "Set the full origin to silence this.",
                    origin,
                    origin,
                )
                origin = f"https://{origin}"

            origins.append(origin)

        return origins

    @model_validator(mode="after")
    def _require_key_for_hosted_provider(self) -> Settings:
        """Fail at startup rather than on the first user question."""
        required = {
            "jina": ("JINA_API_KEY", self.jina_api_key),
            "voyage": ("VOYAGE_API_KEY", self.voyage_api_key),
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
        }
        if self.embedding_provider in required:
            env_name, value = required[self.embedding_provider]
            if not value:
                raise ValueError(
                    f"EMBEDDING_PROVIDER={self.embedding_provider} requires "
                    f"{env_name}. Set the key, or switch to "
                    f"EMBEDDING_PROVIDER=local for offline embeddings."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
