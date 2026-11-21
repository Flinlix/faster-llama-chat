"""The types of what ``begin`` takes and ``conversation`` gives: Chat Completions messages and a whole conversation."""

from typing import Any, Literal, Required, TypedDict

from .tools import ToolSpec

GeneratedTurns = dict[str, Any]
"""The turns kept as generated, JSON-ready, only meaningful to the model that wrote them."""


class ChatFunction(TypedDict):
    """The function a call names, with its arguments as JSON text; both keys are required."""

    name: str
    arguments: str


class ChatToolCall(TypedDict):
    """A call as a Chat Completions message holds it; every key is required."""

    id: str
    type: Literal["function"]
    function: ChatFunction


class ChatMessage(TypedDict, total=False):
    """A Chat Completions message: a system prompt, a question, a reply with its calls, or a tool's result.

    ``role`` and ``content`` are required; ``reasoning_content`` and
    ``tool_calls`` belong to a reply and may be left out, and a tool's result
    names the call it answers in ``tool_call_id``.
    """

    role: Required[Literal["system", "user", "assistant", "tool"]]
    content: Required[str | None]
    reasoning_content: str
    tool_calls: list[ChatToolCall]
    tool_call_id: str


class Conversation(TypedDict):
    """Everything ``begin`` takes to begin a conversation again, by its parameter names; every key is required."""

    messages: list[ChatMessage]
    tools: list[ToolSpec] | None
    reasoning: bool
    reasoning_level: str | None
    generated_turns: GeneratedTurns | None
