"""The messages a chat template is given, as a conversation holds them: their keys, the ids of calls, and how messages of each kind are made."""

from __future__ import annotations

import copy
import itertools
import re

from .errors import ArgumentTypeError, InvalidArgumentError
from .tools import ToolCall, ensure_readable
from .loop_variant import LoopVariant


# The roles a message of the caller's takes: a question and a reply.
MESSAGE_ROLES = ("user", "assistant")
# Where a message holds a reply's thought, which the template is handed in the form it reads.
THOUGHT_KEY = "reasoning_content"
# The name a call that could not be read is written under, and its result answered with.
# Which name is used does not change what a template renders around it.
UNKNOWN_FUNCTION = "unknown"
# Where the message that ends a turn kept as the tokens it was generated as holds that turn.
GENERATED_TURN = "generated_turn"
# The mark of a call that could not be read: written only as a generated turn's tokens hold it,
# else left out at a begin.
UNREADABLE_CALL = "unreadable"


# What a question cut to fit the room begins with, in place of what was cut.
QUESTION_CUT_MARK = "[…] "

def ensure_text(text, what: str, field: str = "text") -> None:
    """Refuse a message's text that is not a ``str``, or that no model can read, naming the message as ``what``.

    Raises:
        ArgumentTypeError: If ``text`` is not a ``str``.
        InvalidArgumentError: Naming ``field``, if it holds a lone surrogate.
    """
    if not isinstance(text, str):
        raise ArgumentTypeError(f"{what} must be a str, not {type(text).__name__}")
    ensure_readable(text, field, what)


def ensure_role(role) -> None:
    """Refuse a message whose role is not one of :data:`MESSAGE_ROLES`, naming where such a message goes instead.

    Raises:
        ValueError: For every other role.
    """
    if role == "system":
        raise InvalidArgumentError("role", "a system message is the system prompt, which begin() "
                                           "takes as the first of its messages")
    if role == "tool":
        raise InvalidArgumentError("role", "a tool result is written by inject_tool_result(), and "
                                           "given to begin() as a tool message after its call")
    if role not in MESSAGE_ROLES:
        raise InvalidArgumentError(
            "role", f"a message's role is {' or '.join(map(repr, MESSAGE_ROLES))}, not {role!r}")


def call_id(n: int) -> str:
    """The id of the ``n``-th call this library writes: ``call`` and ``n`` in five base-36 digits, nine letters and digits in all for the first 36**5 calls."""
    digits = ""
    digits_variant = LoopVariant("writing a call id")
    while n or len(digits) < _ID_DIGITS:
        digits_variant.passes((n, max(0, _ID_DIGITS - len(digits))))
        n, digit = divmod(n, 36)
        digits = _BASE36[digit] + digits
    return "call" + digits


def next_call_number(ids) -> int:
    """A number past every one among ``ids`` that :func:`call_id` writes, so that the ids numbered from it are new and nine characters long."""
    taken = {int(found.group(1), 36) for found in map(_CALL_ID.fullmatch, ids) if found}
    after = max(taken, default=-1) + 1
    if after + _FREE_RUN <= _ID_LIMIT:
        return after
    start = 0
    runs_variant = LoopVariant("finding a free run of call ids")
    while clash := [n for n in taken if start <= n < start + _FREE_RUN]:
        runs_variant.passes(max(taken) + 1 - start)
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


def result_message(name: str, tool_call_id: str, text: str) -> dict:
    """A tool message answering the call ``tool_call_id`` to ``name`` with ``text``."""
    return {"role": "tool", "name": name, "tool_call_id": tool_call_id, "content": text}


def with_result_text(message: dict, text: str) -> dict:
    """The result ``message`` holding ``text`` instead."""
    return {**message, "content": text}


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


def history_count(messages: list[dict]) -> int:
    """How many messages ``messages`` hold after the system message."""
    return sum(message["role"] != "system" for message in messages)


def exchange_starts(messages: list[dict]) -> list[int]:
    """Where each exchange begins: every user message, where kept history may open.

    See docs/decisions/0022-eviction-drops-whole-oldest-exchanges.md.
    """
    return [i for i, m in enumerate(messages) if m["role"] == "user"]


def exchange_count(messages: list[dict]) -> int:
    """How many exchanges ``messages`` hold: one per user message, and one more for what comes before the first after the system message."""
    history = messages[len(system_part(messages)):]
    return len(exchange_starts(history)) + (1 if history and history[0]["role"] != "user" else 0)


def system_part(messages: list[dict]) -> list[dict]:
    """The system message ``messages`` open with, alone in a list, or an empty list where they open otherwise."""
    return [m for m in messages[:1] if m["role"] == "system"]


def unanswered(messages: list[dict]) -> list[tuple[ToolCall, str]]:
    """The calls of the newest turn that no result after them answers yet."""
    calling = None
    for i in range(len(messages) - 1, -1, -1):          # back to the newest call, at most
        if messages[i].get("tool_calls"):
            calling = i
            break
        if messages[i]["role"] != "tool":
            return []                   # a message after the newest call: nothing is owed
    if calling is None:
        return []
    answered = len(messages) - calling - 1
    return [(ToolCall("", {}, c["id"], "unreadable") if UNREADABLE_CALL in c else
             ToolCall(c["function"]["name"], copy.deepcopy(c["function"]["arguments"]), c["id"]),
             c["id"]) for c in messages[calling]["tool_calls"][answered:]]
