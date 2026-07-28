"""Tests for the health endpoints."""

from __future__ import annotations


async def test_root_is_cheap_and_always_healthy(client):
    http, _ = client
    async with http:
        response = await http.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


async def test_health_reports_subsystems_without_raising(client):
    """Mongo is unreachable in tests; health must degrade, not 500."""
    http, _ = client
    async with http:
        response = await http.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded"}
    assert body["mongo_connected"] is False
    assert isinstance(body["index_loaded"], bool)
    assert "search_menu" in body["tools"]


async def test_health_reports_the_loaded_index(client, monkeypatch, vector_store):
    from app.api.routes import health as health_route

    monkeypatch.setattr(health_route, "get_vector_store", lambda: vector_store)

    http, _ = client
    async with http:
        response = await http.get("/health")

    body = response.json()
    assert body["index_loaded"] is True
    assert body["index_chunks"] == len(vector_store)
    assert "fake" in body["embedding_provider"]
