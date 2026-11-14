"""Tool turns written as their steps in order, for tests, expanded into the chat messages ``begin`` takes."""

from __future__ import annotations

from dataclasses import dataclass

from llama_chat.tools import ToolCall


@dataclass(frozen=True)
class Thought:
    """What the model thought before the text and the calls after it."""

    text: str


@dataclass(frozen=True)
class Result:
    """What one call returned: text as it is, anything else as JSON."""

    name: str
    result: object


def call_entry(call: ToolCall) -> dict:
    """``call`` as one entry of a reply's ``tool_calls``, without an id where it has none."""
    if call.status == "unreadable":
        return {"id": call.id, "type": "function", "function": {"name": "unknown", "arguments": {}},
                "unreadable": True}
    entry = {"type": "function", "function": {"name": call.name, "arguments": call.arguments}}
    return {"id": call.id, **entry} if call.id else entry


def tool_turn(*steps) -> list[dict]:
    """The chat messages of one assistant turn given as its steps: text, :class:`Thought`, calls and :class:`Result` in order.

    Calls in a row form one reply, holding the text and the thought before
    them; each result follows the reply of its call. Text after the last
    result closes the turn, and so does an empty text.
    """
    messages, text, thought, calls = [], "", "", []

    def reply() -> dict:
        message = {"role": "assistant", "content": text}
        if thought:
            message["reasoning_content"] = thought
        if calls:
            message["tool_calls"] = [call_entry(call) for call in calls]
        return message

    for step in steps:
        if isinstance(step, str):
            text += step
        elif isinstance(step, Thought):
            thought += step.text
        elif isinstance(step, ToolCall):
            calls.append(step)
        else:
            if calls:
                messages.append(reply())
                text, thought, calls = "", "", []
            messages.append({"role": "tool", "content": step.result})
    if calls or text or thought or not isinstance(steps[-1], Result):
        messages.append(reply())
    return messages
