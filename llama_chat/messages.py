"""A conversation's history as :meth:`~llama_chat.wrapper.ChatWrapper.begin` takes it, and as the messages a chat template is given."""

from __future__ import annotations

import copy

from .tools import ToolCall, ToolResult, ToolTurn, result_text


def as_messages(history) -> list[dict]:
    """``begin``'s history as the messages a template is given.

    A tool turn becomes its assistant messages, each holding the calls made
    before a result, and the results after them, each answering its own call.
    """
    messages, n = [], 0
    for item in history:
        if not isinstance(item, ToolTurn):
            role, text = item
            messages.append({"role": role, "content": text})
            continue
        text, calls, owed = "", [], []          # owed: the ids of the calls still to be answered
        for step in item.steps:
            if isinstance(step, str):
                text += step
            elif isinstance(step, ToolCall):
                calls.append(step)
            elif isinstance(step, ToolResult):
                if calls:
                    messages.append(calling_message(text, calls, n))
                    owed += [call_id(n + k) for k in range(len(calls))]
                    n, text, calls = n + len(calls), "", []
                answered = owed.pop(0) if owed else call_id(n - 1)
                messages.append(result_message(step.name, answered, result_text(step.result)))
        if calls:                       # called, and not answered yet
            messages.append(calling_message(text, calls, n))
            n += len(calls)
        elif text or not (item.steps and isinstance(item.steps[-1], ToolResult)):
            messages.append({"role": "assistant", "content": text})     # a result ends it open
    return messages


def call_id(n: int) -> str:
    """The id of the ``n``-th call this library writes, nine letters and digits."""
    return f"call{n:05d}"


def calling_message(text: str, calls: list[ToolCall], first_id: int) -> dict:
    """An assistant message holding ``text`` and ``calls``, numbered from ``first_id``."""
    return {"role": "assistant", "content": text, "tool_calls": [
        {"id": call_id(first_id + k), "type": "function",
         "function": {"name": call.name, "arguments": copy.deepcopy(call.arguments)}}
        for k, call in enumerate(calls)]}


def result_message(name: str, tool_call_id: str, text: str) -> dict:
    """A tool message answering the call ``tool_call_id`` to ``name`` with ``text``."""
    return {"role": "tool", "name": name, "tool_call_id": tool_call_id, "content": text}


def as_history(messages: list[dict]) -> list:
    """The messages back as ``begin`` takes them."""
    history, steps = [], []
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role == "assistant" and message.get("tool_calls"):
            if message["content"]:
                steps.append(message["content"])
            steps += [ToolCall(c["function"]["name"], copy.deepcopy(c["function"]["arguments"]))
                      for c in message["tool_calls"]]
        elif role == "tool":
            steps.append(ToolResult(message["name"], message["content"]))
        elif role == "assistant" and steps:
            history.append(ToolTurn(tuple(steps + ([message["content"]] if message["content"] else []))))
            steps = []
        else:
            history.append((role, message["content"]))
    if steps:                           # a turn still open
        history.append(ToolTurn(tuple(steps)))
    return history


def exchange_starts(messages: list[dict]) -> list[int]:
    """Where each exchange begins: every user message, where kept history may open.

    A conversation that dropped its oldest messages opens on a user turn, as
    every template writes one. See design_history.md, "Eviction removes whole
    exchanges" under "Structure review: what changed and why".
    """
    return [i for i, m in enumerate(messages) if m["role"] == "user"]
