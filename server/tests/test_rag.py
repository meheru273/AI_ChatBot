"""Tests for chunking, the vector store, and the provider guard."""

from __future__ import annotations

import pytest

from app.rag.chunking import Chunk, split_document
from app.rag.vectorstore import (
    IndexMismatchError,
    IndexMissingError,
    VectorStore,
)
from tests.conftest import FAKE_DIMENSION, FakeEmbeddingProvider

# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def test_split_document_carries_metadata_onto_every_chunk():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(6))
    chunks = split_document(text, {"source": "faq", "type": "knowledge"}, 300, 40)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "faq" for c in chunks)
    assert all(c.metadata["type"] == "knowledge" for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["chunk_total"] == len(chunks) for c in chunks)


def test_split_document_respects_chunk_size():
    text = "sentence. " * 400
    chunks = split_document(text, {}, 250, 25)
    # The splitter breaks on separators, so allow modest overshoot.
    assert all(len(c.text) <= 300 for c in chunks)


def test_split_document_on_empty_text_returns_nothing():
    assert split_document("   \n  ", {"source": "x"}, 300, 30) == []


# --------------------------------------------------------------------------
# vector store
# --------------------------------------------------------------------------


def test_build_indexes_every_chunk(vector_store, sample_chunks):
    assert len(vector_store) == len(sample_chunks)
    assert vector_store.spec.provider == "fake"


def test_build_rejects_an_empty_corpus(fake_provider):
    with pytest.raises(ValueError, match="zero chunks"):
        VectorStore.build([], fake_provider)


def test_search_ranks_lexically_related_text_first(vector_store):
    hits = vector_store.search("what is gochujang chilli paste", k=2)

    assert hits, "expected at least one hit"
    assert "Gochujang" in hits[0].text
    assert hits[0].source == "Korean Food Glossary"


def test_search_returns_at_most_k(vector_store):
    assert len(vector_store.search("rice", k=2)) <= 2


def test_search_scores_are_cosine_similarities(vector_store):
    hits = vector_store.search("delivery fee minimum order", k=3)
    assert all(-1.0 <= h.score <= 1.0 for h in hits)
    # Results arrive best-first.
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_can_scope_to_a_document_type(vector_store):
    hits = vector_store.search("noodles vegetables", k=4, doc_type="menu_item")
    assert hits
    assert all(h.metadata["type"] == "menu_item" for h in hits)


def test_search_on_empty_query_returns_nothing(vector_store):
    assert vector_store.search("   ", k=3) == []


def test_min_score_filters_weak_hits(vector_store):
    assert vector_store.search("gochujang", k=4, min_score=0.99) == []


# --------------------------------------------------------------------------
# persistence and the provider guard
# --------------------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path, vector_store, fake_provider):
    vector_store.save(tmp_path)

    for filename in ("index.faiss", "chunks.json", "meta.json"):
        assert (tmp_path / filename).exists()

    reloaded = VectorStore.load(tmp_path, fake_provider)
    assert len(reloaded) == len(vector_store)
    assert reloaded.search("gochujang", k=1)[0].text == (
        vector_store.search("gochujang", k=1)[0].text
    )


def test_load_from_an_empty_directory_reports_how_to_build(tmp_path, fake_provider):
    with pytest.raises(IndexMissingError, match="build_index"):
        VectorStore.load(tmp_path, fake_provider)


def test_load_rejects_an_index_from_a_different_model(tmp_path, vector_store):
    """Same dimensions, different model — the vector spaces are unrelated.

    This is the dangerous case: without the guard it loads cleanly and returns
    confident nonsense.
    """
    vector_store.save(tmp_path)
    other = FakeEmbeddingProvider(model="fake-v2")

    with pytest.raises(IndexMismatchError, match="Rebuild|rebuild"):
        VectorStore.load(tmp_path, other)


def test_load_rejects_an_index_from_a_different_provider(tmp_path, vector_store):
    """The realistic switch: same corpus, Voyage index, OpenAI configured."""
    vector_store.save(tmp_path)
    other = FakeEmbeddingProvider(provider="other-vendor")

    with pytest.raises(IndexMismatchError, match="Rebuild|rebuild"):
        VectorStore.load(tmp_path, other)


def test_load_detects_metadata_disagreeing_with_the_index(
    tmp_path, vector_store, fake_provider
):
    import json

    vector_store.save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    meta["dimension"] = 999
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(IndexMismatchError, match="corrupt"):
        VectorStore.load(tmp_path, fake_provider)


def test_load_adopts_the_stored_dimension_without_probing(tmp_path, vector_store):
    """Cold starts must not spend an API round trip just to learn the width."""
    vector_store.save(tmp_path)

    provider = FakeEmbeddingProvider()
    provider._dimension = None  # force a probe if the code asks for one
    calls_before = provider.embed_calls

    VectorStore.load(tmp_path, provider)

    assert provider.embed_calls == calls_before
    assert provider.dimension == FAKE_DIMENSION


def test_load_detects_a_corrupt_index_directory(tmp_path, vector_store, fake_provider):
    import json

    vector_store.save(tmp_path)
    # Drop a chunk so the sidecar and the index disagree.
    chunks = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    (tmp_path / "chunks.json").write_text(
        json.dumps(chunks[:-1]), encoding="utf-8"
    )

    with pytest.raises(IndexMismatchError, match="corrupt"):
        VectorStore.load(tmp_path, fake_provider)


def test_metadata_survives_persistence(tmp_path, vector_store, fake_provider):
    vector_store.save(tmp_path)
    reloaded = VectorStore.load(tmp_path, fake_provider)

    hit = reloaded.search("glass noodles vegetables", k=1, doc_type="menu_item")[0]
    assert hit.metadata["source"].startswith("menu:")
    assert "price" in hit.metadata


def test_retrieved_chunk_cite_includes_the_source(vector_store):
    hit = vector_store.search("delivery fee", k=1)[0]
    assert hit.cite().startswith("[source: ")
    assert hit.text in hit.cite()


# --------------------------------------------------------------------------
# corpus assembly
# --------------------------------------------------------------------------


def test_knowledge_corpus_builds_from_the_shipped_markdown():
    from app.config import get_settings
    from app.rag.documents import build_knowledge_chunks

    settings = get_settings()
    chunks = build_knowledge_chunks(settings.knowledge_path, 500, 50)

    assert chunks, "knowledge markdown should produce chunks"
    assert all(c.metadata["type"] == "knowledge" for c in chunks)
    # Sources come from each file's first heading, not the filename.
    sources = {c.metadata["source"] for c in chunks}
    assert "Korean Food Glossary" in sources
    assert "Frequently Asked Questions" in sources


def test_menu_item_passage_includes_price_and_allergens():
    from app.db.models import Dietary, MenuItem

    item = MenuItem(
        _id="507f1f77bcf86cd799439011",
        name="Japchae",
        description="Stir-fried glass noodles with vegetables",
        price=5.5,
        category_name="Sides",
        dietary=Dietary(isVegetarian=True),
        allergens=["soybeans", "sesame"],
    )
    passage = item.to_passage()

    assert "Japchae" in passage
    assert "£5.50" in passage
    assert "vegetarian" in passage
    assert "sesame" in passage and "soybeans" in passage


def test_menu_item_passage_states_when_no_allergens_are_declared():
    from app.db.models import MenuItem

    item = MenuItem(
        _id="507f1f77bcf86cd799439012", name="Kimchi", description="House", price=2.5
    )
    assert "No declared allergens" in item.to_passage()


def test_chunk_source_defaults_when_metadata_is_missing():
    assert Chunk(text="x").source == "unknown"
