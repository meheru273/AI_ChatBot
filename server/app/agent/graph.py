"""LangGraph agent assembly.

    START ──▶ agent ──(tool calls?)──▶ tools ──┐
                │                              │
                │ no                           └── back to agent
                ▼
               END

`tools_condition` routes to END when the model returns no tool calls, so the
graph needs no explicit agent→END edge — the previous implementation had both,
which made the intent ambiguous.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.llm import get_llm
from app.agent.prompts import get_system_prompt
from app.agent.state import AgentState
from app.config import get_settings
from app.tools import get_tools

logger = logging.getLogger(__name__)

# Groq rejects a tool call when the model emits Llama's native
# `<function=name{...}</function>` syntax instead of the JSON the parser
# expects. The arguments are usually correct — only the wrapper is wrong — and
# it happens intermittently, so resampling almost always fixes it.
TOOL_FORMAT_ERROR_MARKERS = ("tool_use_failed", "failed to call a function")
TOOL_CALL_RETRIES = 2


def is_tool_format_error(exc: Exception) -> bool:
    """True for Groq's malformed-tool-call error, which is worth retrying."""
    message = str(exc).lower()
    return any(marker in message for marker in TOOL_FORMAT_ERROR_MARKERS)


@lru_cache
def build_agent_graph() -> CompiledStateGraph:
    """Compile the agent graph. Cached — compilation is not free."""
    tools = list(get_tools())
    llm_with_tools = get_llm().bind_tools(tools)
    system_message = SystemMessage(content=get_system_prompt())

    async def agent_node(state: AgentState) -> dict:
        # The system prompt is prepended per call rather than stored in state,
        # so it cannot be duplicated by history or truncated away.
        messages = [system_message, *state["messages"]]

        for attempt in range(TOOL_CALL_RETRIES + 1):
            try:
                response = await llm_with_tools.ainvoke(messages)
                return {"messages": [response]}
            except Exception as exc:
                if not is_tool_format_error(exc) or attempt == TOOL_CALL_RETRIES:
                    raise
                # Temperature is non-zero, so the retry resamples rather than
                # reproducing the same malformed output.
                logger.warning(
                    "Malformed tool call from the model (attempt %d/%d); retrying",
                    attempt + 1,
                    TOOL_CALL_RETRIES + 1,
                )

        raise RuntimeError("unreachable")  # pragma: no cover

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    logger.info("Agent graph compiled with %d tools", len(tools))
    return builder.compile()


def get_run_config() -> dict:
    """Per-invocation config.

    `recursion_limit` caps the agent↔tools loop. Each pass costs two steps, so
    the limit is doubled and padded to leave room for the final answer.
    """
    return {"recursion_limit": get_settings().max_tool_iterations * 2 + 2}
