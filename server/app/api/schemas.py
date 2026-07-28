"""Request and response models for the chat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """A full conversation. The last user message is the current turn.

    The whole history is sent so the agent can resolve references across turns;
    the server owns the system prompt and ignores any client-supplied one.
    """

    messages: list[Message] = Field(min_length=1)

    def last_user_message(self) -> str | None:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return None


class ChatResponse(BaseModel):
    response: str


class HealthResponse(BaseModel):
    status: str
    mongo_connected: bool
    index_loaded: bool
    index_chunks: int | None = None
    embedding_provider: str | None = None
    model: str
    tools: list[str]
