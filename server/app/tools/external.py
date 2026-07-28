"""General-knowledge tools: Wikipedia and arXiv.

Kept deliberately small in output — the agent's job is restaurant assistance,
and long encyclopedia dumps crowd out the retrieved restaurant context in the
model's window.
"""

from __future__ import annotations

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper


def build_wikipedia_tool() -> WikipediaQueryRun:
    """General knowledge — history and origins of dishes, ingredients, places."""
    tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=600)
    )
    tool.description = (
        "Search Wikipedia for general knowledge that is not about this restaurant "
        "— the history or cultural background of a dish, an ingredient, or a place. "
        "Do not use it for our menu, prices, allergens or opening hours."
    )
    return tool


def build_arxiv_tool() -> ArxivQueryRun:
    """Academic search. Rarely relevant here, but restored per the project spec."""
    tool = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=600)
    )
    tool.description = (
        "Search arXiv for academic papers. Only useful for research questions, "
        "for example on food science, fermentation or nutrition. "
        "Never use it for menu or restaurant questions."
    )
    return tool
