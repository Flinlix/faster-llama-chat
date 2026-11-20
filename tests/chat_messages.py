"""Tool turns written as their steps in order, for tests, expanded into the chat messages ``begin`` takes."""

from __future__ import annotations

import json
from dataclasses import dataclass

from llama_cpp_conversation.tools import ToolCall


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


def opened(system_prompt: str, messages=()) -> list[dict]:
    """The messages ``begin`` takes for ``system_prompt``, none where it is empty, and ``messages`` after it, each in the Chat Completions form.

    Written for tests whose subject is not that form: a call without an id
    is given the next one this library writes, its arguments are written
    as JSON, and a result without the id of its call answers the oldest
    call still owed one, its value written as JSON.
    """
    from llama_cpp_conversation.messages import call_id
    from llama_cpp_conversation.tools import result_text
    messages = list(messages)
    taken = {c.get("id") for m in messages if isinstance(m, dict) for c in m.get("tool_calls") or ()}
    numbers = (n for n in range(10 ** 6) if call_id(n) not in taken)
    converted, owed = [], []
    for message in messages:
        if not isinstance(message, dict):
            converted.append(message)
            continue
        message = dict(message)
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                call = {**call, "function": dict(call.get("function") or {})}
                call["id"] = call.get("id") or call_id(next(numbers))
                call.setdefault("type", "function")
                if call.pop("unreadable", False):
                    call["function"] = {"name": "unknown", "arguments": "not a call"}
                elif not isinstance(call["function"].get("arguments", ""), str):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
                calls.append(call)
                owed.append(call["id"])
            message["tool_calls"] = calls
        elif message.get("role") == "tool":
            message.pop("name", None)
            if not message.get("tool_call_id") and owed:
                message["tool_call_id"] = owed[0]
            if message.get("tool_call_id") in owed:
                owed.remove(message["tool_call_id"])
            if not isinstance(message.get("content"), str):
                message["content"] = result_text(message.get("content"))
        elif message.get("role") == "user":
            owed = []
        converted.append(message)
    return ([{"role": "system", "content": system_prompt}] if system_prompt else []) + converted


def after_system(messages: list[dict]) -> list[dict]:
    """``messages`` without the system message they open with, where they open with one."""
    return messages[1:] if messages and messages[0].get("role") == "system" else messages
