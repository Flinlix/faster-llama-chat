"""A conversation's history as :meth:`~llama_chat.wrapper.ChatWrapper.begin` takes it and :meth:`~llama_chat.wrapper.ChatWrapper.conversation` gives it, and as the messages a chat template is given."""

from __future__ import annotations

import collections
import copy
import hashlib
import itertools
import json
import re

from .errors import InvalidArgumentError
from .tools import ToolCall, result_text


# The roles a message of the caller's takes: a question and a reply.
MESSAGE_ROLES = ("user", "assistant")
# Where a message holds a reply's thought, which the template is handed in the form it reads.
THOUGHT_KEY = "reasoning_content"
# The name a call that could not be read is written under, and its result answered with.
# Which name is used does not change what a template renders around it.
UNKNOWN_FUNCTION = "unknown"
# The mark of a call that could not be read: written only as a kept turn's tokens hold it,
# else left out at a begin.
UNREADABLE_CALL = "unreadable"


def as_messages(history, min_call_number: int = 0) -> list[dict]:
    """``begin``'s history as the messages a template is given.

    Each item is a ``(role, text)`` pair or a chat message: a question, a
    reply with its thought under :data:`THOUGHT_KEY` and its calls under
    ``tool_calls``, or a call's result. A call keeps the id it carries; one
    without takes the next id this library writes after every id of the
    history, and numbered from ``min_call_number`` at least, so no two calls
    share one. A result answers the oldest call still owed one since the
    newest question, in the order the calls were made.

    Raises:
        TypeError: If an item is neither a pair nor a dictionary, or a text,
            a thought or a call's arguments are of another type.
        ValueError: If a message holds keys its role does not take, if a role
            is not ``user``, ``assistant`` or ``tool``, if two calls carry one
            id, if a result answers no call owed one or another than the
            oldest, or if a reply follows a call before its result.
    """
    items = [_history_message(item) for item in history]
    given = [call["id"] for m in items for call in m.get("tool_calls", ()) if call["id"]]
    repeated = next((given_id for given_id, n in collections.Counter(given).items() if n > 1), None)
    if repeated is not None:
        raise InvalidArgumentError("messages", f"two calls of the history carry the id "
                                               f"{repeated!r}; give each call an id of its own")
    numbered = itertools.count(max(min_call_number, next_call_number(given)))
    owed: list[dict] = []           # the calls still to be answered, oldest first
    for message in items:
        if message["role"] == "tool":
            _answer_oldest(message, owed)
            continue
        _ensure_unreadable_answered(owed)
        if message["role"] == "user":
            owed = []
        elif owed:
            raise InvalidArgumentError(
                "messages", f"a reply follows the call {owed[0]['id']!r} before its result; "
                "give each call's result before the next reply")
        for call in message.get("tool_calls", ()):
            call["id"] = call["id"] or call_id(next(numbered))
            owed.append(call)
    _ensure_unreadable_answered(owed)
    return items


def _answer_oldest(result: dict, owed: list[dict]) -> None:
    """Make ``result`` answer the oldest of the ``owed`` calls, taking it off them.

    Raises:
        ValueError: If no call is owed a result, or the result names another
            call or another function.
    """
    if not owed:
        raise InvalidArgumentError("messages", "a tool result answers no call made before it "
                                               "since the newest question")
    call = owed.pop(0)
    named = result.get("tool_call_id") or call["id"]
    if named != call["id"]:
        raise InvalidArgumentError(
            "messages", f"the result for {named!r} is given before the result of the call "
                        f"{call['id']!r}; give results in the order of their calls")
    name = call["function"]["name"]
    if result.get("name", name) != name:
        raise InvalidArgumentError(
            "messages", f"the result of the call {call['id']!r} names the function "
                        f"{result['name']!r}, but the call named {name!r}")
    result.update(name=name, tool_call_id=call["id"])


def _ensure_unreadable_answered(owed: list[dict]) -> None:
    """Refuse a call that could not be read and is left without its result.

    Raises:
        ValueError: If one of ``owed`` is such a call.
    """
    if any(UNREADABLE_CALL in call for call in owed):
        raise InvalidArgumentError(
            "messages", "a call that could not be read was answered when it was written; "
                        "a history cannot leave one owed a result")


# The keys each role's message may hold beside ``role``.
_KEYS = {"user": {"content"}, "assistant": {"content", THOUGHT_KEY, "tool_calls"},
         "tool": {"content", "tool_call_id", "name"}}
# The keys a call of a reply may hold.
_CALL_KEYS = {"id", "type", "function", UNREADABLE_CALL}


def _history_message(item) -> dict:
    """A history message from a ``(role, text)`` pair or a chat message, validated and copied.

    Raises:
        TypeError: If ``item`` is neither, or a text is not a ``str``.
        ValueError: If its role is not one of ``user``, ``assistant`` and
            ``tool``, or it holds keys its role does not take.
    """
    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
        role, text = item
        ensure_role(role)
        ensure_text(text, "a history message's text")
        return {"role": role, "content": text}
    if not isinstance(item, dict):
        raise TypeError(f"a history message is a (role, text) pair or a chat message "
                        f"dictionary, not the {type(item).__name__} {item!r}")
    role = item.get("role")
    if role != "tool":
        ensure_role(role)
    if set(item) - _KEYS[role] - {"role"} or "content" not in item:
        raise InvalidArgumentError(
            "messages", f"a {role} message holds 'role' and 'content', and may hold "
                        f"{sorted(_KEYS[role] - {'content'})}, not {sorted(item)}")
    if role == "tool":
        return _result_of(item)
    ensure_text(item["content"], "a history message's text")
    message = {"role": role, "content": item["content"]}
    if role == "assistant":
        thought = item.get(THOUGHT_KEY, "")
        ensure_text(thought, "a thought")
        message = _with_thought(message, thought)
        if item.get("tool_calls"):
            message["tool_calls"] = [_call_of(call) for call in item["tool_calls"]]
    return message


def _result_of(item: dict) -> dict:
    """A tool message from the result ``item``, its content written as the model reads it."""
    for key in ("name", "tool_call_id"):
        if key in item:
            ensure_text(item[key], f"a tool result's {key}")
    message = result_message(item.get("name", ""), item.get("tool_call_id", ""),
                             result_text(item["content"]), item["content"])
    for key in ("name", "tool_call_id"):
        if key not in item:
            del message[key]
    return message


def _call_of(call) -> dict:
    """One call of a reply's ``tool_calls``, validated and copied.

    Raises:
        TypeError: If its id or name is not a ``str``, or its arguments are
            not a ``dict``.
        ValueError: If it is no dictionary of the keys a call holds, or an
            unreadable call holds a name or arguments.
    """
    if not isinstance(call, dict) or set(call) - _CALL_KEYS or "function" not in call \
            or not isinstance(call["function"], dict) \
            or set(call["function"]) - {"name", "arguments"}:
        raise InvalidArgumentError(
            "messages", f"a call holds 'id', 'type' and 'function' with its 'name' and "
                        f"'arguments', not {call!r}")
    function, given_id = call["function"], call.get("id", "")
    ensure_text(given_id, "a call's id")
    if call.get(UNREADABLE_CALL):
        if function.get("name", UNKNOWN_FUNCTION) != UNKNOWN_FUNCTION or function.get("arguments"):
            raise InvalidArgumentError(
                "messages", f"a call that could not be read has no name or arguments, as a "
                            f"conversation keeps one, not {call!r}")
        return {"id": given_id, "type": "function",
                "function": {"name": UNKNOWN_FUNCTION, "arguments": {}}, UNREADABLE_CALL: True}
    name, arguments = function.get("name"), function.get("arguments", {})
    ensure_text(name, "a call's name")
    if not isinstance(arguments, dict):
        raise TypeError(f"the call to {name!r} has arguments that are not a dict: {arguments!r}")
    return {"id": given_id, "type": "function",
            "function": {"name": name, "arguments": copy.deepcopy(arguments)}}


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
        raise InvalidArgumentError("role", "a system message is the system prompt, which begin() "
                                           "takes as its first argument")
    if role == "tool":
        raise InvalidArgumentError("role", "a tool result is written by inject_tool_result(), and "
                                           "given to begin() as a tool message after its call")
    if role not in MESSAGE_ROLES:
        raise InvalidArgumentError(
            "role", f"a message's role is {' or '.join(map(repr, MESSAGE_ROLES))}, not {role!r}")


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


def calling_message(text: str, calls: list[ToolCall], first_id: int = 0) -> dict:
    """An assistant message holding ``text`` and ``calls``, each under the id it carries, or, where it carries none, numbered from ``first_id``.

    A call that could not be read is written under :data:`UNKNOWN_FUNCTION`,
    marked as one.
    """
    numbered = itertools.count(first_id)
    return {"role": "assistant", "content": text, "tool_calls": [
        {"id": call.id or call_id(next(numbered)), "type": "function",
         "function": {"name": UNKNOWN_FUNCTION if call.status == "unreadable" else call.name,
                      "arguments": copy.deepcopy(call.arguments)},
         **({UNREADABLE_CALL: True} if call.status == "unreadable" else {})}
        for call in calls]}


def portable_digest(messages: list[dict]) -> str:
    """A digest of ``messages`` as a caller keeps them, the same after any number of round trips through :func:`as_history` and :func:`as_messages`."""
    def portable(message: dict) -> dict:
        kept = {"role": message["role"], "content": message.get("content"),
                THOUGHT_KEY: message.get(THOUGHT_KEY)}
        if message.get("tool_calls"):
            kept["calls"] = [{"id": c["id"], "unreadable": True} if UNREADABLE_CALL in c else
                             {"id": c["id"], "name": c["function"]["name"],
                              "arguments": c["function"]["arguments"]}
                             for c in message["tool_calls"]]
        if message["role"] == "tool":
            kept["answers"] = message["tool_call_id"]
        return kept
    return hashlib.sha256(json.dumps([portable(m) for m in messages], sort_keys=True,
                                     ensure_ascii=False, default=repr).encode()).hexdigest()


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


def as_history(messages: list[dict]) -> list[dict]:
    """The messages after the system message as ``begin`` takes them back: chat messages holding only what a caller gave, a result as its caller gave it.

    A tool turn that was closed ends on the reply that closed it, its text
    empty where that reply wrote none; one still open ends on a call or a
    result.
    """
    history = []
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        kept = {key: copy.deepcopy(message[key]) for key in _KEYS[role] | {"role"} if key in message}
        if role == "tool":
            kept["content"] = copy.deepcopy(message.get(RESULT_AS_GIVEN, message["content"]))
        history.append(kept)
    return history


def history_count(messages: list[dict]) -> int:
    """How many messages :func:`as_history` gives back for ``messages``, without building them."""
    return sum(message["role"] != "system" for message in messages)


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
