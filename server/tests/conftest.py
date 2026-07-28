"""Shared fixtures.

The suite never calls a real LLM, a real embedding API, or a real database.
Retrieval is exercised against a deterministic fake provider so assertions
about ranking are stable, and the agent is replaced by a scripted graph so the
SSE protocol can be tested independently of model behaviour.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_ROOT))

# Must be set before app.config is imported anywhere.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017/test")
os.environ.setdefault("MONGO_DB", "test")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")  # never actually constructed
os.environ.setdefault("RETRIEVAL_TOP_K", "3")
# Registers the web tools; no request is ever made with it.
os.environ.setdefault("JINA_API_KEY", "test-jina-key")

from app.rag.chunking import Chunk  # noqa: E402
from app.rag.embeddings import EmbeddingProvider  # noqa: E402
from app.rag.vectorstore import VectorStore  # noqa: E402

FAKE_DIMENSION = 32


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-words embeddings.

    Real enough for ranking assertions — texts sharing vocabulary land near
    each other — while costing nothing and never varying between runs.
    """

    provider_name = "fake"

    def __init__(
        self,
        dimension: int = FAKE_DIMENSION,
        model: str = "fake-v1",
        provider: str = "fake",
    ) -> None:
        super().__init__(model)
        self.set_dimension(dimension)
        self.provider_name = provider
        self.embed_calls = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            token = "".join(ch for ch in word if ch.isalnum())
            if not token:
                continue
            digest = hashlib.md5(token.encode()).digest()
            vector[digest[0] % self.dimension] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._vector(text)


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            text=(
                "Bibimbap — Rice, seasonal veg, beef, fried egg and gochujang. "
                "Price: £9.50. Contains allergens: eggs, sesame, soybeans."
            ),
            metadata={"type": "menu_item", "source": "menu:Bibimbap", "price": 9.50},
        ),
        Chunk(
            text=(
                "Japchae — Stir-fried glass noodles with vegetables. Price: £5.50. "
                "Suitable for: vegetarian. Contains allergens: sesame, soybeans."
            ),
            metadata={"type": "menu_item", "source": "menu:Japchae", "price": 5.50},
        ),
        Chunk(
            text=(
                "Gochujang is a thick fermented chilli paste made from chilli "
                "powder, glutinous rice and fermented soybeans."
            ),
            metadata={"type": "knowledge", "source": "Korean Food Glossary"},
        ),
        Chunk(
            text=(
                "Delivery fee is £2.99 flat and the minimum order for delivery "
                "is £10.00 within a 5 km radius of West Drayton."
            ),
            metadata={"type": "knowledge", "source": "Ordering and Delivery"},
        ),
    ]


@pytest.fixture
def vector_store(
    fake_provider: FakeEmbeddingProvider, sample_chunks: list[Chunk]
) -> VectorStore:
    return VectorStore.build(sample_chunks, fake_provider)


# --------------------------------------------------------------------------
# Scripted agent graph
# --------------------------------------------------------------------------


class FakeGraph:
    """Stands in for the compiled LangGraph agent.

    Emits the same event shapes `astream_events(version="v2")` produces, so the
    SSE route is tested against the contract it actually consumes.
    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        tool_names: list[str] | None = None,
        raise_at_token: int | None = None,
    ) -> None:
        self.tokens = tokens if tokens is not None else ["Hello", " there", "!"]
        self.tool_names = tool_names or []
        self.raise_at_token = raise_at_token
        self.received_messages: list = []

    async def ainvoke(self, state: dict, config: dict | None = None) -> dict:
        from langchain_core.messages import AIMessage

        self.received_messages = state["messages"]
        return {"messages": [AIMessage(content="".join(self.tokens))]}

    async def astream_events(
        self, state: dict, config: dict | None = None, version: str = "v2"
    ):
        from langchain_core.messages import AIMessageChunk

        self.received_messages = state["messages"]

        for name in self.tool_names:
            yield {"event": "on_tool_start", "name": name, "data": {"input": {}}}
            yield {"event": "on_tool_end", "name": name, "data": {"output": "ok"}}

        for i, token in enumerate(self.tokens):
            if self.raise_at_token is not None and i == self.raise_at_token:
                raise RuntimeError("upstream model exploded")
            yield {
                "event": "on_chat_model_stream",
                "name": "ChatGroq",
                "data": {"chunk": AIMessageChunk(content=token)},
            }


@pytest.fixture
def fake_graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def client(monkeypatch, fake_graph: FakeGraph):
    """An httpx client bound to the ASGI app with the agent stubbed out.

    Returns `(client, graph_holder)`; assign to `graph_holder["graph"]` to swap
    in a differently scripted graph for a single test.
    """
    import httpx

    from app.api.routes import chat as chat_route

    graph_holder = {"graph": fake_graph}
    monkeypatch.setattr(chat_route, "build_agent_graph", lambda: graph_holder["graph"])

    from app.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test"), graph_holder
