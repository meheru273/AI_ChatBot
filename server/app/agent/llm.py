"""Groq chat model construction."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> ChatGroq:
    """The chat model, cached for the process lifetime.

    `streaming=True` is what makes token-level SSE possible; without it Groq
    returns the completion in one piece and the stream degrades to a single
    large event.
    """
    settings = get_settings()
    logger.info("Initialising Groq model %r", settings.groq_model)

    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        streaming=True,
    )
