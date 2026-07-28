"""Chat endpoints.

Two shapes over the same agent:

* `POST /chat`        — buffered, returns one JSON body. Simple clients, tests.
* `POST /chat/stream` — Server-Sent Events, token by token, plus tool activity.

## SSE event protocol

    event: token       data: {"content": "..."}      incremental model output
    event: tool_start  data: {"tool": "search_menu"} a tool began
    event: tool_end    data: {"tool": "search_menu"} a tool returned
    event: done        data: {"content": "..."}      full answer, terminal
    event: error       data: {"message": "..."}      failure, terminal

`error` exists because the HTTP status is committed the moment the stream
opens. A failure after that point cannot become a 500, so it has to be
reported inside the stream or the client hangs waiting for `done`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import build_agent_graph, get_run_config
from app.agent.state import to_langchain_messages
from app.api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

EVENT_TOKEN = "token"
EVENT_TOOL_START = "tool_start"
EVENT_TOOL_END = "tool_end"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def _prepare(request: ChatRequest) -> list:
    if request.last_user_message() is None:
        raise HTTPException(status_code=422, detail="No user message in request")

    messages = to_langchain_messages(request.messages)
    if not messages:
        raise HTTPException(status_code=422, detail="No usable messages in request")
    return messages


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Buffered chat — runs the agent to completion and returns the answer."""
    messages = _prepare(request)

    try:
        result = await build_agent_graph().ainvoke(
            {"messages": messages}, config=get_run_config()
        )
    except Exception as exc:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    final = result["messages"][-1]
    content = getattr(final, "content", None)
    if not content:
        raise HTTPException(status_code=500, detail="Agent produced no response")

    return ChatResponse(response=str(content))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request) -> EventSourceResponse:
    """Streaming chat over SSE."""
    messages = _prepare(request)

    async def event_generator() -> AsyncIterator[dict]:
        collected: list[str] = []

        try:
            stream = build_agent_graph().astream_events(
                {"messages": messages},
                config=get_run_config(),
                version="v2",
            )

            async for event in stream:
                # Stop doing work the moment the browser goes away — otherwise
                # an abandoned tab keeps burning LLM and database calls.
                if await http_request.is_disconnected():
                    logger.info("Client disconnected; abandoning stream")
                    return

                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    text = getattr(chunk, "content", "")
                    if text:
                        collected.append(text)
                        yield {
                            "event": EVENT_TOKEN,
                            "data": json.dumps({"content": text}),
                        }

                elif kind == "on_tool_start":
                    yield {
                        "event": EVENT_TOOL_START,
                        "data": json.dumps({"tool": event.get("name", "")}),
                    }

                elif kind == "on_tool_end":
                    yield {
                        "event": EVENT_TOOL_END,
                        "data": json.dumps({"tool": event.get("name", "")}),
                    }

            yield {
                "event": EVENT_DONE,
                "data": json.dumps({"content": "".join(collected)}),
            }

        except asyncio.CancelledError:
            # Normal on disconnect; re-raise so the server can unwind cleanly.
            logger.info("Stream cancelled")
            raise

        except Exception as exc:  # noqa: BLE001 - must surface inside the stream
            logger.exception("Streaming agent failed")
            yield {
                "event": EVENT_ERROR,
                "data": json.dumps({"message": str(exc)}),
            }

    return EventSourceResponse(
        event_generator(),
        headers={
            # Nginx and several PaaS proxies buffer responses by default, which
            # holds tokens back until the stream ends and defeats the point.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
