"""Tool registry.

`get_tools()` is the single place the agent learns what it can do. Order is
meaningful only as documentation; the model chooses by description.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.tools.knowledge import SearchKnowledgeTool
from app.tools.menu import (
    FilterMenuTool,
    ListCategoriesTool,
    RestaurantInfoTool,
    SearchMenuTool,
)
from app.tools.web import ReadWebPageTool, WebSearchTool

logger = logging.getLogger(__name__)


@lru_cache
def get_tools() -> tuple[BaseTool, ...]:
    """Every tool available to the agent.

    Cached as a tuple so the bound LLM and the ToolNode share one set of
    instances rather than constructing parallel copies.
    """
    tools: list[BaseTool] = [
        # Structured — exact answers from MongoDB.
        SearchMenuTool(),
        FilterMenuTool(),
        ListCategoriesTool(),
        RestaurantInfoTool(),
        # Semantic — FAISS retrieval over descriptions and knowledge base.
        SearchKnowledgeTool(),
    ]

    # Web access rides on the Jina key that already powers embeddings. Without
    # one the tools are left unregistered rather than registered-and-broken, so
    # the model never offers a capability that cannot work.
    settings = get_settings()
    if settings.web_search_enabled and settings.jina_api_key:
        tools.extend([WebSearchTool(), ReadWebPageTool()])
    elif settings.web_search_enabled:
        logger.warning("Web tools disabled: JINA_API_KEY is not set")

    return tuple(tools)


def get_tool_names() -> list[str]:
    return [tool.name for tool in get_tools()]


__all__ = ["get_tools", "get_tool_names"]
