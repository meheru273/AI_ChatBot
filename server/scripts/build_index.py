"""Rebuild the FAISS index from MongoDB and the knowledge corpus.

Run from the `server/` directory:

    python -m scripts.build_index
    python -m scripts.build_index --provider local
    python -m scripts.build_index --dry-run

The index must be rebuilt whenever the menu changes, the knowledge markdown
changes, or the embedding provider changes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow `python scripts/build_index.py` as well as `python -m scripts.build_index`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("build_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["openai", "local"],
        help="Override EMBEDDING_PROVIDER for this build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble and report the corpus without embedding it.",
    )
    return parser.parse_args()


async def dry_run() -> int:
    from collections import Counter

    from app.config import get_settings
    from app.db.mongo import close_mongo, get_database, init_mongo
    from app.db.repository import MenuRepository
    from app.rag.documents import build_corpus

    settings = get_settings()
    await init_mongo()
    try:
        chunks = await build_corpus(
            repo=MenuRepository(get_database()),
            knowledge_path=settings.knowledge_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    finally:
        await close_mongo()

    counts = Counter(c.metadata.get("type", "unknown") for c in chunks)
    print(f"\nCorpus: {len(chunks)} chunks")
    for doc_type, count in sorted(counts.items()):
        print(f"  {doc_type:<12} {count:>4}")

    if chunks:
        lengths = [len(c.text) for c in chunks]
        print(
            f"\nChunk length: min {min(lengths)}, "
            f"mean {sum(lengths) // len(lengths)}, max {max(lengths)}"
        )
        print("\nSample chunk:")
        print(f"  source: {chunks[0].source}")
        print(f"  {chunks[0].text[:220]}...")

    return 0


async def full_build() -> int:
    from app.rag.ingest import build_index

    store = await build_index()
    print(
        f"\nIndex built: {len(store)} chunks, {store.spec.describe()}\n"
        f"Restart the server to pick it up."
    )
    return 0


def main() -> int:
    args = parse_args()

    if args.provider:
        # Set before any app import so Settings reads the override.
        os.environ["EMBEDDING_PROVIDER"] = args.provider

    try:
        return asyncio.run(dry_run() if args.dry_run else full_build())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
