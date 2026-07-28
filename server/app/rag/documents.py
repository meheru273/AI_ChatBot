"""Assemble the corpus that gets embedded.

Two sources feed the index:

1. **MongoDB** — each menu item is rendered into a prose passage
   (`MenuItem.to_passage`) so semantic queries can reach dish content, plus a
   passage per restaurant covering location and service details.
2. **Markdown** in `server/data/knowledge/` — policy, glossary and FAQ
   content that has no representation in the database.

Every chunk carries a `type` and `source` in its metadata so the retriever can
scope a search and the LLM can cite where an answer came from.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.db.repository import MenuRepository
from app.rag.chunking import Chunk, split_document

logger = logging.getLogger(__name__)

TYPE_MENU_ITEM = "menu_item"
TYPE_RESTAURANT = "restaurant"
TYPE_KNOWLEDGE = "knowledge"


async def build_menu_chunks(repo: MenuRepository) -> list[Chunk]:
    """One chunk per menu item. Passages are short, so they are not split."""
    items = await repo.list_all(limit=1000)

    chunks = [
        Chunk(
            text=item.to_passage(),
            metadata={
                "type": TYPE_MENU_ITEM,
                "source": f"menu:{item.name}",
                "item_id": item.id,
                "item_name": item.name,
                "category": item.category_name or "",
                "price": item.price,
                "allergens": item.allergens,
                "is_available": item.is_available,
            },
        )
        for item in items
    ]

    logger.info("Built %d menu chunks", len(chunks))
    return chunks


async def build_restaurant_chunks(repo: MenuRepository) -> list[Chunk]:
    """One chunk per restaurant location."""
    restaurants = await repo.list_restaurants()

    chunks = [
        Chunk(
            text=r.to_passage(),
            metadata={
                "type": TYPE_RESTAURANT,
                "source": f"restaurant:{r.name}",
                "restaurant_id": r.id,
                "postcode": r.address.postcode,
            },
        )
        for r in restaurants
    ]

    logger.info("Built %d restaurant chunks", len(chunks))
    return chunks


def build_knowledge_chunks(
    knowledge_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split every markdown file in the knowledge directory."""
    if not knowledge_path.exists():
        logger.warning("Knowledge directory %s does not exist", knowledge_path)
        return []

    chunks: list[Chunk] = []
    for md_file in sorted(knowledge_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # First heading makes a friendlier citation than the bare filename.
        title = next(
            (
                line.lstrip("# ").strip()
                for line in text.splitlines()
                if line.startswith("# ")
            ),
            md_file.stem.replace("-", " ").title(),
        )
        chunks.extend(
            split_document(
                text,
                metadata={
                    "type": TYPE_KNOWLEDGE,
                    "source": title,
                    "file": md_file.name,
                },
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    logger.info(
        "Built %d knowledge chunks from %d files",
        len(chunks),
        len(list(knowledge_path.glob("*.md"))),
    )
    return chunks


async def build_corpus(
    repo: MenuRepository,
    knowledge_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Full corpus: menu + restaurants + knowledge base."""
    chunks = [
        *await build_menu_chunks(repo),
        *await build_restaurant_chunks(repo),
        *build_knowledge_chunks(knowledge_path, chunk_size, chunk_overlap),
    ]
    logger.info("Corpus: %d chunks total", len(chunks))
    return chunks
