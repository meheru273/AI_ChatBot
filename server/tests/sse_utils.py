"""Minimal SSE parser for tests.

Parses the wire format rather than trusting a client library, so the tests
actually assert that the endpoint emits well-formed Server-Sent Events.
"""

from __future__ import annotations

import json
from typing import Any


async def collect_sse(response) -> list[tuple[str, dict[str, Any]]]:
    """Read an httpx streaming response into (event_name, payload) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    async for raw in response.aiter_lines():
        line = raw.rstrip("\r")

        if not line:  # blank line terminates an event
            if event_name is not None:
                payload = json.loads("\n".join(data_lines)) if data_lines else {}
                events.append((event_name, payload))
            event_name, data_lines = None, []
            continue

        if line.startswith(":"):  # comment / keep-alive ping
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())

    # Flush a trailing event that arrived without a final blank line.
    if event_name is not None:
        payload = json.loads("\n".join(data_lines)) if data_lines else {}
        events.append((event_name, payload))

    return events


def events_of(events: list[tuple[str, dict]], name: str) -> list[dict]:
    return [payload for event_name, payload in events if event_name == name]


def event_names(events: list[tuple[str, dict]]) -> list[str]:
    return [event_name for event_name, _ in events]
