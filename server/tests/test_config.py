"""Tests for settings normalisation.

CORS misconfiguration is the classic deploy failure here: the backend reports
healthy, the frontend shows "Disconnected", and nothing in the logs obviously
explains it. These cover the two ways it usually goes wrong.
"""

from __future__ import annotations

import pytest

from app.config import Settings


def _settings(**overrides) -> Settings:
    base = {
        "groq_api_key": "g",
        "mongo_uri": "mongodb://localhost:27017",
        "embedding_provider": "local",
    }
    return Settings(**{**base, **overrides})


def test_bare_hostname_gets_an_https_scheme():
    """An Origin header is always scheme+host, so a bare hostname matches nothing."""
    settings = _settings(cors_origins="banchan-rag-app.vercel.app")
    assert settings.allowed_origins == ["https://banchan-rag-app.vercel.app"]


def test_trailing_slash_is_stripped():
    settings = _settings(cors_origins="https://example.com/")
    assert settings.allowed_origins == ["https://example.com"]


def test_multiple_origins_are_split_and_trimmed():
    settings = _settings(
        cors_origins=" https://a.vercel.app , http://localhost:3000 ,, b.com "
    )
    assert settings.allowed_origins == [
        "https://a.vercel.app",
        "http://localhost:3000",
        "https://b.com",
    ]


def test_http_localhost_is_left_alone():
    """Local development is http, and must not be rewritten to https."""
    settings = _settings(cors_origins="http://localhost:3000")
    assert settings.allowed_origins == ["http://localhost:3000"]


def test_wildcard_is_preserved():
    settings = _settings(cors_origins="*")
    assert settings.allowed_origins == ["*"]


def test_empty_origins_yield_an_empty_list():
    assert _settings(cors_origins="  , ,").allowed_origins == []


# --------------------------------------------------------------------------
# Mongo URI aliasing
# --------------------------------------------------------------------------


def test_mongodb_uri_alias_is_accepted(monkeypatch):
    """Atlas hands out MONGODB_URI; the Banchan services use MONGO_URI."""
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.setenv("MONGODB_URI", "mongodb://aliased:27017")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")

    assert Settings(_env_file=None).mongo_uri == "mongodb://aliased:27017"


def test_mongo_database_defaults_to_the_menu_service_database(monkeypatch):
    # conftest exports MONGO_DB=test for the suite, so clear it to see the default.
    monkeypatch.delenv("MONGO_DB", raising=False)
    assert _settings().mongo_db == "menu"


@pytest.mark.parametrize("provider", ["jina", "voyage", "openai"])
def test_hosted_provider_without_a_key_fails_fast(provider):
    with pytest.raises(ValueError, match="API_KEY"):
        _settings(
            embedding_provider=provider,
            jina_api_key=None,
            voyage_api_key=None,
            openai_api_key=None,
        )
