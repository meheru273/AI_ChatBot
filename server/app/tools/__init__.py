"""Tool registry.

`get_tools()` is the single place the agent learns what it can do. Order is
meaningful only as documentation; the model chooses by description.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import BaseTool

from app.tools.external import build_arxiv_tool, build_wikipedia_tool
from app.tools.knowledge import SearchKnowledgeTool
from app.tools.menu import (
    FilterMenuTool,
    ListCategoriesTool,
    RestaurantInfoTool,
    SearchMenuTool,
)


@lru_cache
def get_tools() -> tuple[BaseTool, ...]:
    """Every tool available to the agent.

    Cached as a tuple so the bound LLM and the ToolNode share one set of
    instances rather than constructing parallel copies.
    """
    return (
        # Structured — exact answers from MongoDB.
        SearchMenuTool(),
        FilterMenuTool(),
        ListCategoriesTool(),
        RestaurantInfoTool(),
        # Semantic — FAISS retrieval over descriptions and knowledge base.
        SearchKnowledgeTool(),
        # General knowledge.
        build_wikipedia_tool(),
        build_arxiv_tool(),
    )


def get_tool_names() -> list[str]:
    return [tool.name for tool in get_tools()]


__all__ = ["get_tools", "get_tool_names"]
