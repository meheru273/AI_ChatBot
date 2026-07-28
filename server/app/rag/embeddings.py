"""Pluggable embedding providers.

Three backends behind one interface, selected by `EMBEDDING_PROVIDER`:

    voyage  voyage-3-lite            hosted, free tier, MongoDB's own provider
    openai  text-embedding-3-small   hosted, paid
    local   all-MiniLM-L6-v2         offline, installs torch (~2GB)

`local` cannot run on serverless, so deployments use a hosted provider.

Identity vs dimension
---------------------
`identity` (provider + model) is cheap and never calls the API. `dimension`
may require one probe request, so the vector store compares identities on load
and then hands the stored dimension back via `set_dimension()` — that keeps
cold starts free of an extra round trip.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Published dimensions, so the common paths never spend a probe request.
# Unlisted models are measured once on first use.
_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}


@dataclass(frozen=True)
class EmbeddingSpec:
    """Identifies the vector space an index was built in."""

    provider: str
    model: str
    dimension: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
        }

    def describe(self) -> str:
        return f"{self.provider}/{self.model} ({self.dimension}d)"


class EmbeddingProvider(ABC):
    """Common interface over hosted and local embedding models."""

    provider_name: str = "unknown"

    def __init__(self, model: str) -> None:
        self._model_name = model
        self._dimension: int | None = _KNOWN_DIMENSIONS.get(model)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def identity(self) -> tuple[str, str]:
        """(provider, model). Cheap — never calls the API."""
        return (self.provider_name, self._model_name)

    @property
    def dimension(self) -> int:
        """Vector width, probed once if the model is not in the known table."""
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
            logger.info("Probed %s dimension: %d", self._model_name, self._dimension)
        return self._dimension

    def set_dimension(self, dimension: int) -> None:
        """Adopt a dimension recorded at index-build time, skipping the probe."""
        self._dimension = dimension

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(self.provider_name, self._model_name, self.dimension)

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passages for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI over plain REST.

    Called with httpx rather than the `voyageai` SDK to keep the serverless
    bundle small — httpx is already a dependency.

    Voyage distinguishes `document` from `query` inputs and encodes them
    differently; using the right one on each side measurably improves
    retrieval over treating both symmetrically.
    """

    provider_name = "voyage"
    API_URL = "https://api.voyageai.com/v1/embeddings"
    # Voyage caps documents per request; batch well under the limit.
    BATCH_SIZE = 96

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(model)
        self._api_key = api_key

    def _post(self, texts: list[str], input_type: str) -> list[list[float]]:
        import httpx

        response = httpx.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"input": texts, "model": self._model_name, "input_type": input_type},
            timeout=30.0,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Voyage embeddings failed ({response.status_code}): {response.text}"
            )

        payload = response.json()
        # The API does not guarantee ordering, so sort by the echoed index.
        rows = sorted(payload["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            vectors.extend(
                self._post(texts[start : start + self.BATCH_SIZE], "document")
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._post([text], "query")[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Hosted embeddings via the OpenAI API."""

    provider_name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(model)
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Offline embeddings via sentence-transformers.

    Optional dependency — it pulls in torch and is far too large for a
    serverless deployment — so it is imported here rather than at module scope.
    """

    provider_name = "local"

    def __init__(self, model: str) -> None:
        super().__init__(model)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "EMBEDDING_PROVIDER=local requires sentence-transformers. "
                "Install it with `pip install sentence-transformers`, or use a "
                "hosted provider (voyage, openai)."
            ) from exc

        logger.info("Loading local embedding model %r (first run downloads it)", model)
        self._client = SentenceTransformer(model)
        if self._dimension is None:
            self._dimension = self._client.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._client.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._client.encode(text, show_progress_bar=False).tolist()


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Construct the provider named by `EMBEDDING_PROVIDER`.

    Key presence is enforced by Settings validation, so the keys are non-None
    by the time this runs.
    """
    settings = settings or get_settings()

    if settings.embedding_provider == "voyage":
        return VoyageEmbeddingProvider(
            api_key=settings.voyage_api_key or "",
            model=settings.voyage_embedding_model,
        )
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key or "",
            model=settings.openai_embedding_model,
        )
    return LocalEmbeddingProvider(model=settings.local_embedding_model)


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide provider, constructing it on first use."""
    global _provider
    if _provider is None:
        _provider = build_embedding_provider()
        logger.info("Embedding provider ready: %s", "/".join(_provider.identity))
    return _provider


def reset_embedding_provider() -> None:
    """Drop the cached provider. Used by tests and by the index builder."""
    global _provider
    _provider = None
