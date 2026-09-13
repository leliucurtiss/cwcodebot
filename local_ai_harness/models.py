from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[Any] | None = ""

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    context_files: list[str] = Field(default_factory=list)

    class Config:
        extra = "allow"


class ProjectCreateRequest(BaseModel):
    project_id: str | None = None
    root_path: str | None = None
    memory_summary: str = ""


class ProjectMemoryUpdate(BaseModel):
    memory_summary: str


class ConversationMemoryUpdate(BaseModel):
    memory_summary: str


class ConversationCreateRequest(BaseModel):
    project_id: str
    conversation_id: str | None = None


class DebugContextRequest(BaseModel):
    project_id: str | None = None
    conversation_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    context_files: list[str] = Field(default_factory=list)
