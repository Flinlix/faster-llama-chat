"""A conversation's history as :meth:`~llama_chat.wrapper.ChatWrapper.begin` takes it, and as the messages a chat template is given."""

from __future__ import annotations

import collections
import copy
import itertools
import re
from dataclasses import replace

from .tools import Thought, ToolCall, ToolResult, ToolTurn, result_text


# The roles a message of the caller's takes: a question and a reply.
MESSAGE_ROLES = ("user", "assistant")
# Where a message holds a reply's thought, which the template is handed in the form it reads.
THOUGHT_KEY = "reasoning_content"


def as_messages(history, min_call_number: int = 0) -> list[dict]:
    """``begin``'s history as the messages a template is given.

    A tool turn becomes its assistant messages, each holding the calls made
    before a result and the thought before them, and the results after them,
    each answering its own call. A call keeps the id it carries; one without
    takes the next id this library writes after every id of the history, and
    numbered from ``min_call_number`` at least, so no two calls share one.

    Raises:
        TypeError: If an item is neither a ``(role, text)`` pair, a message
            dictionary nor a :class:`~llama_chat.tools.ToolTurn`, or a text
            or a thought is not a ``str``.
        ValueError: If a dictionary holds other keys than ``role``,
            ``content`` and, for a reply, :data:`THOUGHT_KEY`, if a role is
            not one of :data:`MESSAGE_ROLES`, or if two calls of the history
            carry the same id.
    """
    given = [step.id for item in history if isinstance(item, ToolTurn)
             for step in item.steps if isinstance(step, ToolCall) and step.id]
    repeated = next((given_id for given_id, n in collections.Counter(given).items() if n > 1), None)
    if repeated is not None:
        raise ValueError(f"two calls of the history carry the id {repeated!r}; "
                         "give each call an id of its own")
    numbered = itertools.count(max(min_call_number, next_call_number(given)))
    messages = []
    for item in history:
        if not isinstance(item, ToolTurn):
            messages.append(_history_message(item))
            continue
        text, thought, calls, owed = "", "", [], []    # owed: the ids of the calls still to be answered
        for step in item.steps:
            if isinstance(step, str):
                text += step
            elif isinstance(step, Thought):
                ensure_text(step.text, "a thought")
                thought += step.text
            elif isinstance(step, ToolCall):
                calls.append(step if step.id else replace(step, id=call_id(next(numbered))))
            elif isinstance(step, ToolResult):
                if calls:
                    messages.append(_with_thought(calling_message(text, calls), thought))
                    owed += [call.id for call in calls]
                    text, thought, calls = "", "", []
                # A ToolTurn holds no result before the call it answers.
                messages.append(result_message(step.name, owed.pop(0), result_text(step.result),
                                               step.result))
        if calls:                       # called, and not answered yet
            messages.append(_with_thought(calling_message(text, calls), thought))
        elif text or thought or not (item.steps and isinstance(item.steps[-1], ToolResult)):
            messages.append(_with_thought({"role": "assistant", "content": text},  # a result ends it open
                                          thought))
    return messages


def kept_items(history, n_messages: int) -> int:
    """How many of the newest items of ``history`` its newest ``n_messages`` messages, as :func:`as_messages` writes them, come from."""
    items = 0
    for item in reversed(history):
        if n_messages <= 0:
            break
        n_messages -= len(as_messages([item]))
        items += 1
    assert n_messages == 0, "the kept messages begin inside an item of the history"
    return items


def _history_message(item) -> dict:
    """A history message from a ``(role, text)`` pair or a ``{"role", "content"}`` dictionary, a reply's thought included."""
    if isinstance(item, dict):
        allowed = {"role", "content"} | ({THOUGHT_KEY} if item.get("role") == "assistant" else set())
        if set(item) - allowed or "content" not in item:
            raise ValueError(f"a history message holds 'role' and 'content', and a reply also "
                             f"{THOUGHT_KEY!r}, but nothing else, not {sorted(item)}; a turn "
                             "that called tools is a ToolTurn")
        role, text, thought = item.get("role"), item["content"], item.get(THOUGHT_KEY, "")
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        (role, text), thought = item, ""
    else:
        raise TypeError(f"a history message is a (role, text) pair, a dict with 'role' and "
                        f"'content', or a ToolTurn, not the {type(item).__name__} {item!r}")
    ensure_role(role)
    ensure_text(text, "a history message's text")
    ensure_text(thought, "a thought")
    return _with_thought({"role": role, "content": text}, thought)


def _with_thought(message: dict, thought: str) -> dict:
    """``message``, holding ``thought`` under :data:`THOUGHT_KEY` where there is one."""
    return {**message, THOUGHT_KEY: thought} if thought else message


def ensure_text(text, what: str) -> None:
    """Refuse a message's text that is not a ``str``, naming the message as ``what``.

    Raises:
        TypeError: If ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"{what} must be a str, not {type(text).__name__}")


def ensure_role(role) -> None:
    """Refuse a message whose role is not one of :data:`MESSAGE_ROLES`, naming where such a message goes instead.

    Raises:
        ValueError: For every other role.
    """
    if role == "system":
        raise ValueError("a system message is the system prompt, which begin() takes as its "
                         "first argument")
    if role == "tool":
        raise ValueError("a tool result is written by inject_tool_result(), and given to "
                         "begin() as a ToolResult inside a ToolTurn")
    if role not in MESSAGE_ROLES:
        raise ValueError(f"a message's role is {' or '.join(map(repr, MESSAGE_ROLES))}, not {role!r}")


def call_id(n: int) -> str:
    """The id of the ``n``-th call this library writes: ``call`` and ``n`` in five base-36 digits, nine letters and digits in all for the first 36**5 calls."""
    digits = ""
    while n or len(digits) < _ID_DIGITS:
        n, digit = divmod(n, 36)
        digits = _BASE36[digit] + digits
    return "call" + digits


def next_call_number(ids) -> int:
    """A number past every one among ``ids`` that :func:`call_id` writes, so that the ids numbered from it are new and nine characters long.

    One past the highest, where that leaves room below 36**5; else the start
    of the lowest run of :data:`_FREE_RUN` numbers none of ``ids`` holds.
    """
    taken = {int(found.group(1), 36) for found in map(_CALL_ID.fullmatch, ids) if found}
    after = max(taken, default=-1) + 1
    if after + _FREE_RUN <= _ID_LIMIT:
        return after
    start = 0
    while clash := [n for n in taken if start <= n < start + _FREE_RUN]:
        start = max(clash) + 1
    return start


# How many base-36 digits a call's number is written with, at the least.
_ID_DIGITS = 5
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_CALL_ID = re.compile(r"call([0-9a-z]{5})")
# The numbers whose ids keep five digits, and how many free numbers a conversation
# is given at once where it starts again from the lowest.
_ID_LIMIT = 36 ** _ID_DIGITS
_FREE_RUN = 1000


# The name a call that could not be read is written under, and its result answered with.
# Which name is used does not change what a template renders around it.
UNKNOWN_FUNCTION = "unknown"
# Where a call that could not be read keeps what the model wrote for it, which is written in its place.
CALL_AS_WRITTEN = "call_as_written"


def calling_message(text: str, calls: list[ToolCall], first_id: int = 0) -> dict:
    """An assistant message holding ``text`` and ``calls``, each under the id it carries, or, where it carries none, numbered from ``first_id``.

    A call that could not be read is written under :data:`UNKNOWN_FUNCTION`,
    its text as the model wrote it beside it.
    """
    numbered = itertools.count(first_id)
    return {"role": "assistant", "content": text, "tool_calls": [
        {"id": call.id or call_id(next(numbered)), "type": "function",
         "function": {"name": UNKNOWN_FUNCTION if call.status == "unreadable" else call.name,
                      "arguments": copy.deepcopy(call.arguments)},
         **({CALL_AS_WRITTEN: call.text} if call.status == "unreadable" else {})}
        for call in calls]}


# Where a result message keeps the value its caller gave, where that was no text, to give it back.
RESULT_AS_GIVEN = "result_as_given"
_NOT_GIVEN = object()


def result_message(name: str, tool_call_id: str, text: str, given=_NOT_GIVEN) -> dict:
    """A tool message answering the call ``tool_call_id`` to ``name`` with ``text``, the value ``given`` beside it where that is no text."""
    message = {"role": "tool", "name": name, "tool_call_id": tool_call_id, "content": text}
    if given is not _NOT_GIVEN and not isinstance(given, str):
        message[RESULT_AS_GIVEN] = copy.deepcopy(given)
    return message


def with_result_text(message: dict, text: str) -> dict:
    """The result ``message`` holding ``text`` instead, which it gives back as it is."""
    return {**{key: value for key, value in message.items() if key != RESULT_AS_GIVEN},
            "content": text}


# The forms chat templates read a reply's thought in: a field of its own under
# either of two names, or a content block of its own ahead of the text.
THOUGHT_FORMS = (THOUGHT_KEY, "thinking", "content_block")


def with_thought_as(message: dict, form: str) -> dict:
    """``message``, whose thought is held under :data:`THOUGHT_KEY`, with that thought in ``form``, one of :data:`THOUGHT_FORMS`."""
    thought = message.get(THOUGHT_KEY)
    if not thought or form == THOUGHT_KEY:
        return message
    rest = {key: value for key, value in message.items() if key != THOUGHT_KEY}
    if form == THOUGHT_FORMS[1]:
        return {**rest, THOUGHT_FORMS[1]: thought}
    return {**rest, "content": [{"type": "thinking", "thinking": thought},
                                {"type": "text", "text": message["content"]}]}


def as_history(messages: list[dict]) -> list:
    """The messages back as ``begin`` takes them: each as a dictionary, a reply's thought under ``reasoning_content``, a tool turn's thoughts as its steps.

    A tool turn that was closed ends on the text of the reply that closed it,
    empty where that reply wrote none; one still open, or left open before a
    later question, ends on a call or a result.
    """
    history, steps = [], []
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role == "user" and steps:
            history.append(ToolTurn(tuple(steps)))
            steps = []
        thought = [Thought(message[THOUGHT_KEY])] if message.get(THOUGHT_KEY) else []
        said = thought + ([message["content"]] if message["content"] else [])
        if role == "assistant" and message.get("tool_calls"):
            steps += said + [ToolCall("", {}, c["id"], "unreadable", c[CALL_AS_WRITTEN])
                             if CALL_AS_WRITTEN in c else
                             ToolCall(c["function"]["name"],
                                      copy.deepcopy(c["function"]["arguments"]), c["id"])
                             for c in message["tool_calls"]]
        elif role == "tool":
            steps.append(ToolResult(message["name"], message.get(RESULT_AS_GIVEN, message["content"])))
        elif role == "assistant" and steps:
            history.append(ToolTurn(tuple(steps + thought + [message["content"]])))
            steps = []
        else:
            history.append({"role": role, "content": message["content"],
                            **({THOUGHT_KEY: message[THOUGHT_KEY]} if thought else {})})
    if steps:                           # a turn still open
        history.append(ToolTurn(tuple(steps)))
    return history


def history_count(messages: list[dict]) -> int:
    """How many items :func:`as_history` gives back for ``messages``, without building them."""
    count, in_turn = 0, False
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role == "user" and in_turn:
            count, in_turn = count + 1, False
        if role == "assistant" and message.get("tool_calls") or role == "tool":
            in_turn = True
        elif role == "assistant" and in_turn:
            count, in_turn = count + 1, False
        else:
            count += 1
    return count + in_turn


def exchange_starts(messages: list[dict]) -> list[int]:
    """Where each exchange begins: every user message, where kept history may open.

    A conversation that dropped its oldest messages opens on a user turn, as
    every template writes one. See design_history.md, "Eviction removes whole
    exchanges" under "Structure review: what changed and why".
    """
    return [i for i, m in enumerate(messages) if m["role"] == "user"]


def exchange_count(messages: list[dict]) -> int:
    """How many exchanges ``messages`` hold: one per user message, and one more for what comes before the first after the system message."""
    history = messages[len(system_part(messages)):]
    return len(exchange_starts(history)) + (1 if history and history[0]["role"] != "user" else 0)


def system_part(messages: list[dict]) -> list[dict]:
    """The system message ``messages`` open with, alone in a list, or an empty list where they open otherwise."""
    return [m for m in messages[:1] if m["role"] == "system"]
