"""Semantic retrieval tool — the RAG half of the hybrid.

Answers open-ended, explanatory and comparative questions by embedding the
query and searching the FAISS index over dish descriptions, restaurant details
and the knowledge base.

Explicitly *not* the path for allergen or price questions: those go through
`filter_menu` and `search_menu`, which read exact structured fields. A near
miss in vector space is an acceptable outcome for "what is gochujang" and an
unacceptable one for "does this contain peanuts".
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag.ingest import get_vector_store

logger = logging.getLogger(__name__)


class SearchKnowledgeInput(BaseModel):
    query: str = Field(
        description=(
            "A natural-language question, phrased as the customer asked it. "
            "Examples: 'what is the difference between japchae and gimbap', "
            "'something warm and comforting', 'how does your kimchi differ from "
            "the usual kind'."
        )
    )


class SearchKnowledgeTool(BaseTool):
    name: str = "search_knowledge"
    description: str = (
        "Semantic search over dish descriptions, restaurant details and our "
        "knowledge base (Korean food glossary, ordering and delivery policy, "
        "allergen policy, FAQ). "
        "Use for open-ended, explanatory or comparative questions, and for vague "
        "requests like 'something spicy and warming'. "
        "Do NOT use for allergen questions (use filter_menu) or for exact prices "
        "(use search_menu)."
    )
    args_schema: type[BaseModel] = SearchKnowledgeInput

    def _run(self, query: str) -> str:
        return self._search(query)

    async def _arun(self, query: str) -> str:
        # Embedding is a blocking call in both providers, but at top_k=4 over a
        # few dozen chunks it is short enough not to warrant a thread hop.
        return self._search(query)

    def _search(self, query: str) -> str:
        store = get_vector_store()
        if store is None:
            return (
                "The knowledge index is not built, so semantic search is "
                "unavailable. Use search_menu, filter_menu, "
                "list_menu_categories or get_restaurant_info instead."
            )

        hits = store.search(query, k=get_settings().retrieval_top_k)
        if not hits:
            return f"Nothing in the knowledge base matches '{query}'."

        blocks = [
            f"[{i}] source: {hit.source} (relevance {hit.score:.2f})\n{hit.text}"
            for i, hit in enumerate(hits, start=1)
        ]
        return (
            f"Retrieved {len(hits)} passage(s) for '{query}':\n\n"
            + "\n\n".join(blocks)
            + "\n\nGround your answer in these passages and mention the source."
        )
