"""Agent state and conversation-history conversion."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Graph state.

    `add_messages` appends rather than replaces, so tool results and model
    responses accumulate across the agent/tools loop within a single turn.
    """

    messages: Annotated[list[BaseMessage], add_messages]


def to_langchain_messages(history: Iterable[Any]) -> list[BaseMessage]:
    """Convert the API's role/content pairs into LangChain messages.

    Both prior assistant turns and user turns are kept — the previous
    implementation forwarded only the last user message, which left the agent
    unable to resolve anything referential ("is *that one* vegan?").

    The system prompt is not included here; it is prepended per model call so
    it never accumulates in the stored history.
    """
    messages: list[BaseMessage] = []

    for entry in history:
        role = getattr(entry, "role", None) or (
            entry.get("role") if isinstance(entry, dict) else None
        )
        content = getattr(entry, "content", None) or (
            entry.get("content") if isinstance(entry, dict) else None
        )

        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            messages.append(AIMessage(content=str(content)))
        # Any other role (system, tool) is dropped: the system prompt is
        # server-owned, and replaying stale tool output would mislead the model.

    return messages
