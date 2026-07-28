"""Tests for the agent node's resilience to malformed tool calls.

Groq rejects a tool call when Llama emits its native
`<function=name{...}</function>` syntax instead of JSON. The arguments inside
are usually correct and the fault is intermittent, so the node resamples
rather than surfacing a 500 to the user.
"""

from __future__ import annotations

import pytest

from app.agent.graph import (
    TOOL_CALL_RETRIES,
    is_tool_format_error,
)

GROQ_TOOL_ERROR = (
    "Failed to call a function. Please adjust your prompt. See "
    "'failed_generation' for more details. code='tool_use_failed' "
    "failed_generation='<function=search_menu{\"query\": \"bibimbap\"}</function>'"
)


@pytest.mark.parametrize(
    "message",
    [
        GROQ_TOOL_ERROR,
        "tool_use_failed",
        "Failed to call a function.",
        "FAILED TO CALL A FUNCTION",
    ],
)
def test_recognises_groq_malformed_tool_call(message):
    assert is_tool_format_error(Exception(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "Connection reset by peer",
        "401 Invalid API Key",
        "model_decommissioned",
        "rate limit exceeded",
    ],
)
def test_other_failures_are_not_treated_as_retryable(message):
    """A bad key or a dead model must fail immediately, not retry three times."""
    assert is_tool_format_error(Exception(message)) is False


# --------------------------------------------------------------------------
# retry behaviour
# --------------------------------------------------------------------------


class FlakyLLM:
    """Raises the Groq tool error for the first `failures` calls."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls <= self.failures:
            raise Exception(GROQ_TOOL_ERROR)
        return AIMessage(content="recovered")


@pytest.fixture
def agent_node(monkeypatch):
    """Build the real agent node around a supplied fake LLM."""

    def _build(llm):
        from app.agent import graph as graph_module

        class Bindable:
            def bind_tools(self, tools):
                return llm

        monkeypatch.setattr(graph_module, "get_llm", lambda: Bindable())
        monkeypatch.setattr(graph_module, "get_tools", lambda: ())
        graph_module.build_agent_graph.cache_clear()

        compiled = graph_module.build_agent_graph()
        graph_module.build_agent_graph.cache_clear()
        return compiled

    return _build


async def test_recovers_from_a_single_malformed_tool_call(agent_node):
    llm = FlakyLLM(failures=1)
    graph = agent_node(llm)

    result = await graph.ainvoke({"messages": []})

    assert llm.calls == 2
    assert result["messages"][-1].content == "recovered"


async def test_recovers_within_the_retry_budget(agent_node):
    llm = FlakyLLM(failures=TOOL_CALL_RETRIES)
    graph = agent_node(llm)

    result = await graph.ainvoke({"messages": []})

    assert llm.calls == TOOL_CALL_RETRIES + 1
    assert result["messages"][-1].content == "recovered"


async def test_gives_up_once_the_budget_is_exhausted(agent_node):
    llm = FlakyLLM(failures=TOOL_CALL_RETRIES + 1)
    graph = agent_node(llm)

    with pytest.raises(Exception, match="Failed to call a function"):
        await graph.ainvoke({"messages": []})

    assert llm.calls == TOOL_CALL_RETRIES + 1


async def test_non_retryable_errors_fail_on_the_first_attempt(agent_node):
    class DeadModel:
        calls = 0

        async def ainvoke(self, messages):
            DeadModel.calls += 1
            raise Exception("model_decommissioned")

    graph = agent_node(DeadModel())

    with pytest.raises(Exception, match="decommissioned"):
        await graph.ainvoke({"messages": []})

    assert DeadModel.calls == 1, "must not burn retries on a permanent failure"
