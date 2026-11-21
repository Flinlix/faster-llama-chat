"""The types of what ``begin`` takes and ``conversation`` gives: Chat Completions messages and a whole conversation."""

from typing import Any, Literal, Required, TypedDict

from .tools import ToolSpec

GeneratedTurns = dict[str, Any]
"""The turns kept as generated, JSON-ready, only meaningful to the model that wrote them."""


class ChatFunction(TypedDict):
    """The function a call names, with its arguments as JSON text."""

    name: str
    arguments: str


class ChatToolCall(TypedDict):
    """A call as a Chat Completions message holds it."""

    id: str
    type: Literal["function"]
    function: ChatFunction


class ChatMessage(TypedDict, total=False):
    """A Chat Completions message: a system prompt, a question, a reply with its calls, or a tool's result."""

    role: Required[str]
    content: str | None
    reasoning_content: str
    tool_calls: list[ChatToolCall]
    tool_call_id: str


class Conversation(TypedDict):
    """Everything ``begin`` takes to begin a conversation again, by its parameter names."""

    messages: list[ChatMessage]
    tools: list[ToolSpec] | None
    reasoning: bool
    reasoning_level: str | None
    generated_turns: GeneratedTurns | None
