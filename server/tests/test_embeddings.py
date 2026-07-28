"""Tests for the embedding providers.

The Voyage provider talks REST directly, so httpx is stubbed and the request
it builds is asserted — that is the part worth testing, and it can be checked
without an API key or a network call.
"""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.rag.embeddings import (
    VoyageEmbeddingProvider,
    build_embedding_provider,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def captured_posts(monkeypatch):
    """Stub httpx.post, recording calls and returning canned embeddings."""
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "headers": headers, "body": json})
        return FakeResponse(
            {
                "data": [
                    {"embedding": [float(i)] * 4, "index": i}
                    for i in range(len(json["input"]))
                ]
            }
        )

    monkeypatch.setattr("httpx.post", fake_post)
    return calls


def test_voyage_marks_documents_and_queries_differently(captured_posts):
    """Voyage encodes passages and queries asymmetrically; using the right
    input_type on each side is what makes that pay off."""
    provider = VoyageEmbeddingProvider(api_key="test-key", model="voyage-3-lite")

    provider.embed_documents(["a passage"])
    provider.embed_query("a question")

    assert captured_posts[0]["body"]["input_type"] == "document"
    assert captured_posts[1]["body"]["input_type"] == "query"


def test_voyage_sends_the_api_key_as_a_bearer_token(captured_posts):
    provider = VoyageEmbeddingProvider(api_key="secret-key", model="voyage-3-lite")
    provider.embed_query("hello")

    assert captured_posts[0]["headers"]["Authorization"] == "Bearer secret-key"


def test_voyage_batches_large_document_sets(captured_posts):
    provider = VoyageEmbeddingProvider(api_key="k", model="voyage-3-lite")
    vectors = provider.embed_documents([f"doc {i}" for i in range(250)])

    assert len(captured_posts) == 3  # 96 + 96 + 58
    assert len(vectors) == 250


def test_voyage_reorders_results_by_echoed_index(monkeypatch):
    """The API does not guarantee ordering, so results must be re-sorted or
    embeddings silently attach to the wrong chunks."""

    def out_of_order_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return FakeResponse(
            {
                "data": [
                    {"embedding": [2.0], "index": 2},
                    {"embedding": [0.0], "index": 0},
                    {"embedding": [1.0], "index": 1},
                ]
            }
        )

    monkeypatch.setattr("httpx.post", out_of_order_post)
    provider = VoyageEmbeddingProvider(api_key="k", model="voyage-3-lite")

    assert provider.embed_documents(["a", "b", "c"]) == [[0.0], [1.0], [2.0]]


def test_voyage_raises_a_readable_error_on_failure(monkeypatch):
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **kw: FakeResponse({"detail": "invalid key"}, status_code=401),
    )
    provider = VoyageEmbeddingProvider(api_key="bad", model="voyage-3-lite")

    with pytest.raises(RuntimeError, match="401"):
        provider.embed_query("hello")


def test_voyage_probes_its_dimension_once(captured_posts):
    provider = VoyageEmbeddingProvider(api_key="k", model="unlisted-model")

    assert provider.dimension == 4  # the stub returns 4-wide vectors
    assert provider.dimension == 4
    assert len(captured_posts) == 1, "dimension should be cached after one probe"


def test_known_models_need_no_probe(captured_posts):
    from app.rag.embeddings import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key="k", model="text-embedding-3-small")

    assert provider.dimension == 1536
    assert captured_posts == []


def test_identity_is_provider_and_model():
    provider = VoyageEmbeddingProvider(api_key="k", model="voyage-3-lite")
    assert provider.identity == ("voyage", "voyage-3-lite")


def test_set_dimension_overrides_the_probe(captured_posts):
    provider = VoyageEmbeddingProvider(api_key="k", model="unlisted-model")
    provider.set_dimension(512)

    assert provider.dimension == 512
    assert captured_posts == []


# --------------------------------------------------------------------------
# provider selection
# --------------------------------------------------------------------------


def _settings(**overrides) -> Settings:
    base = {
        "groq_api_key": "g",
        "mongo_uri": "mongodb://localhost:27017",
        "embedding_provider": "voyage",
        "voyage_api_key": "v",
    }
    return Settings(**{**base, **overrides})


def test_factory_selects_voyage():
    provider = build_embedding_provider(_settings())
    assert provider.identity == ("voyage", "voyage-3-lite")


def test_factory_selects_openai():
    provider = build_embedding_provider(
        _settings(embedding_provider="openai", openai_api_key="sk-test")
    )
    assert provider.identity == ("openai", "text-embedding-3-small")


@pytest.mark.parametrize(
    ("provider", "missing_key"),
    [("voyage", "VOYAGE_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_hosted_providers_require_their_key(provider, missing_key):
    """Fail at startup, not on the first user question."""
    with pytest.raises(ValueError, match=missing_key):
        Settings(
            groq_api_key="g",
            mongo_uri="mongodb://localhost:27017",
            embedding_provider=provider,
            voyage_api_key=None,
            openai_api_key=None,
        )


def test_local_provider_needs_no_key():
    settings = _settings(embedding_provider="local", voyage_api_key=None)
    assert settings.embedding_provider == "local"
