"""The two forms a conversation's history takes: the Chat Completions messages ``begin`` takes and ``conversation()`` gives, and beside them the turns kept as the tokens they were generated as.

Plain functions over message lists, testable without a model. Inside, the
conversation keeps the messages a template is given: a call's arguments as
the object they encode, a call the model wrote unreadably marked as one, and
a turn kept as generated on the message it ends with. See design_history.md,
"Two history forms".
"""

from __future__ import annotations

from typing import Any

import collections
import copy
import hashlib
import json
import logging

from .call_parsing import object_once
from .errors import ArgumentTypeError, InvalidArgumentError
from .messages import GENERATED_TURN, THOUGHT_KEY, UNKNOWN_FUNCTION, UNREADABLE_CALL, result_message
from .tools import ensure_readable, json_value_fault
from .rendering import GeneratedTurn, generated_turns_in
from .loop_variant import LoopVariant

_log = logging.getLogger(__name__)

# The keys each role's message takes in the Chat Completions form this library takes, ``role`` included.
_COMPLETIONS_KEYS = {"system": {"role", "content"}, "user": {"role", "content"},
                "assistant": {"role", "content", "tool_calls", THOUGHT_KEY},
                "tool": {"role", "tool_call_id", "content"}}
_CALL_KEYS = {"id", "type", "function"}
_FUNCTION_KEYS = {"name", "arguments"}


# ----- the Chat Completions form, taken -----------------------------------------------------------
def read_messages(messages: list) -> list[dict]:
    """``begin``'s messages, in the Chat Completions form, as the messages a template is given.

    A call whose arguments are no JSON object, or that names no function, is
    marked as one that could not be read; ``""`` is no arguments. A reply's
    ``content`` may be ``None`` where it makes calls, and is then empty.

    Raises:
        TypeError: If a message or a call is no dictionary, or a text, a
            function's name or a call's arguments is of another type.
        ValueError: If a message holds a key its role does not take here, or
            a list of content parts; its role is none of ``system``, ``user``,
            ``assistant`` and ``tool``; a system message is not the first; a
            call lacks its id, is of another type than ``function``, or
            shares its id with another; or a result lacks its
            ``tool_call_id`` or is no text.
    """
    read = []
    for i, item in enumerate(messages):
        where = f"messages[{i}]"
        if not isinstance(item, dict):
            raise ArgumentTypeError(f"{where} is a chat message dictionary, such as "
                            f"{{'role': 'user', 'content': text}}, not the "
                            f"{type(item).__name__} {item!r}")
        role = item.get("role")
        if role not in _COMPLETIONS_KEYS:
            raise InvalidArgumentError("messages", f"{where}'s role is 'system', 'user', "
                                                   f"'assistant' or 'tool', not {role!r}")
        extra = sorted(set(item) - _COMPLETIONS_KEYS[role])
        if extra:
            raise InvalidArgumentError("messages", f"{where}, a {role} message, holds {extra[0]!r}, "
                                                   f"which it does not take here; it takes "
                                                   f"{sorted(_COMPLETIONS_KEYS[role])}")
        if "content" not in item:
            raise InvalidArgumentError("messages", f"{where} needs its 'content'")
        if role == "system" and i:
            raise InvalidArgumentError("messages", f"{where} is a system message, which may only "
                                                   "open the messages")
        read.append(_reply(item, where) if role == "assistant"
                    else _result(item, where) if role == "tool"
                    else {"role": role, "content": _text(item["content"], f"{where}.content")})
    ids = [c["id"] for m in read for c in m.get("tool_calls") or ()]
    repeated = next((given for given, n in collections.Counter(ids).items() if n > 1), None)
    if repeated is not None:
        raise InvalidArgumentError("messages", f"two calls of the history carry the id "
                                               f"{repeated!r}; give each call an id of its own")
    return read


def _text(value, where: str) -> str:
    """``value``, a message's text.

    Raises:
        TypeError: If it is no ``str``; a list of parts is refused as a form not taken.
    """
    if isinstance(value, list):
        raise InvalidArgumentError("messages", f"{where} is a list of parts, which is not taken "
                                               "here; give its text as one string")
    if not isinstance(value, str):
        raise ArgumentTypeError(f"{where} must be a str, not {type(value).__name__}")
    ensure_readable(value, "messages", where)
    return value


def _reply(item: dict, where: str) -> dict:
    """The assistant message ``item`` as a template is given it."""
    calls = item.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ArgumentTypeError(f"{where}.tool_calls must be a list, not {type(calls).__name__}")
    content = item["content"]
    if content is None and not calls:
        raise InvalidArgumentError("messages", f"{where}.content is null, which only a reply "
                                               "that makes calls may be")
    message: dict[str, object] = {
        "role": "assistant", "content": "" if content is None else _text(content, f"{where}.content")}
    thought = item.get(THOUGHT_KEY)
    if thought:
        message[THOUGHT_KEY] = _text(thought, f"{where}.{THOUGHT_KEY}")
    if calls:
        message["tool_calls"] = [_call(call, f"{where}.tool_calls[{k}]")
                                 for k, call in enumerate(calls)]
    return message


def _call(call, where: str) -> dict:
    """One call of a reply's ``tool_calls`` as a template is given it, marked as one that could not be read where its arguments are no JSON object."""
    if not isinstance(call, dict):
        raise ArgumentTypeError(f"{where} must be a dictionary, not {type(call).__name__}")
    extra = sorted(set(call) - _CALL_KEYS)
    if extra:
        raise InvalidArgumentError("messages", f"{where} holds {extra[0]!r}, which a call does "
                                               f"not take; it takes {sorted(_CALL_KEYS)}")
    if call.get("type", "function") != "function":
        raise InvalidArgumentError("messages", f"{where} is of type {call['type']!r}; only "
                                               "'function' calls are taken")
    given_id, function = call.get("id"), call.get("function")
    if not isinstance(given_id, str) or not given_id:
        raise InvalidArgumentError("messages", f"{where} needs its 'id', a non-empty string")
    if not isinstance(function, dict) or set(function) - _FUNCTION_KEYS:
        raise InvalidArgumentError("messages", f"{where}.function takes 'name' and 'arguments', "
                                               f"not {function!r}")
    name, given = function.get("name"), function.get("arguments", "")
    if not isinstance(name, str):
        raise ArgumentTypeError(f"{where}.function.name must be a str, not {type(name).__name__}")
    if not isinstance(given, str):
        raise ArgumentTypeError(f"{where}.function.arguments must be the arguments as a JSON string, "
                        f"not {type(given).__name__}")
    arguments = _arguments(given)
    if arguments is None or not name:
        _log.info("%s could not be read (%s), so it is left out with its result", where,
                  "it names no function" if not name
                  else "its arguments are no JSON object, each key once")
        return unreadable_call(given_id)
    return {"id": given_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _arguments(given: str) -> dict | None:
    """The object ``given`` encodes, ``{}`` for ``""``; ``None`` where it encodes none."""
    if given == "":
        return {}
    try:
        # A key written twice leaves the value meant untold, as in a call the model writes.
        arguments = json.loads(given, object_pairs_hook=object_once)
    except (ValueError, RecursionError):
        return None
    # As a call the model writes: one past the depth limit, or holding what no model reads, is unreadable.
    return arguments if isinstance(arguments, dict) and json_value_fault(arguments) is None else None


def unreadable_call(given_id: str) -> dict:
    """A call that could not be read, under ``given_id``, as a template is given it."""
    return {"id": given_id, "type": "function",
            "function": {"name": UNKNOWN_FUNCTION, "arguments": {}}, UNREADABLE_CALL: True}


def _result(item: dict, where: str) -> dict:
    """The tool message ``item`` as a template is given it, the name of its call filled in once the calls are read."""
    call = item.get("tool_call_id")
    if not isinstance(call, str) or not call:
        raise InvalidArgumentError("messages", f"{where} needs its 'tool_call_id', a non-empty "
                                               "string")
    content = item["content"]
    if not isinstance(content, str):
        raise InvalidArgumentError("messages", f"{where}.content, the result of {call!r}, must "
                                               f"be text, not {type(content).__name__}")
    return {"role": "tool", "tool_call_id": call, "content": content}


# ----- the order of calls and results ---------------------------------------------------
def answer_in_order(messages: list[dict], unreadable_result: str) -> list[dict]:
    """``messages`` with each result named for the call it answers, checked against the order of the calls.

    A result answers the oldest call still owed one since the newest
    question. A call that could not be read and is owed one when no result
    of it follows is answered with ``unreadable_result``, as it was when it
    was written; one behind a call still owed waits for it.

    Raises:
        ValueError: If two calls carry one id, a result names no call owed
            one or another than the oldest, or a reply follows a call before
            its result.
    """
    ids = [c["id"] for m in messages for c in m.get("tool_calls") or ()]
    repeated = next((given for given, n in collections.Counter(ids).items() if n > 1), None)
    if repeated is not None:
        raise InvalidArgumentError("messages", f"two calls of the history carry the id "
                                               f"{repeated!r}; give each call an id of its own")
    answered: list[dict] = []
    owed: list[dict] = []           # the calls still to be answered, oldest first

    def answer_unreadable(upto: str | None) -> None:
        owing_variant = LoopVariant("answering unreadable calls")
        while owed and UNREADABLE_CALL in owed[0] and owed[0]["id"] != upto:
            owing_variant.passes(len(owed))
            answered.append(result_message(UNKNOWN_FUNCTION, owed.pop(0)["id"], unreadable_result))

    for message in messages:
        if message["role"] == "tool":
            answer_unreadable(message["tool_call_id"])
            answered.append(_answering_oldest(message, owed))
            continue
        answer_unreadable(None)
        if message["role"] == "user":
            owed = []
        elif owed:
            raise InvalidArgumentError(
                "messages", f"a reply follows the call {owed[0]['id']!r} before its result; give "
                            "each call's result before the next reply")
        owed += message.get("tool_calls") or []
        answered.append(message)
    answer_unreadable(None)
    return answered


def _answering_oldest(result: dict, owed: list[dict]) -> dict:
    """``result``, named for the oldest of the ``owed`` calls, which it answers and which goes off them.

    Raises:
        ValueError: If no call is owed a result, or the result names another.
    """
    if not owed:
        raise InvalidArgumentError("messages", f"the result for {result['tool_call_id']!r} answers "
                                               "no call made before it since the newest question")
    call = owed.pop(0)
    if result["tool_call_id"] != call["id"]:
        raise InvalidArgumentError(
            "messages", f"the result for {result['tool_call_id']!r} is given before the result "
                        f"of the call {call['id']!r}; give results in the order of their calls")
    return {**result, "name": call["function"]["name"]}


# ----- the Chat Completions form, given ---------------------------------------------------------
def completions_messages(messages: list[dict], joins_questions: bool) -> list[dict]:
    """``messages`` in the Chat Completions form ``begin`` takes: every call that could not be read left out with its result, as :func:`without_unreadable` leaves them out."""
    kept, _gone = without_unreadable(messages, set(), joins_questions)
    return [_as_completions(message) for message in kept]


def _as_completions(message: dict) -> dict:
    """One message in the Chat Completions form."""
    role = message["role"]
    if role == "tool":
        return {"role": "tool", "tool_call_id": message["tool_call_id"],
                "content": message["content"]}
    if role != "assistant":
        return {"role": role, "content": message["content"]}
    calls = message.get("tool_calls") or []
    written = {"role": "assistant", "content": None if calls and not message["content"]
               else message["content"]}
    if message.get(THOUGHT_KEY):
        written[THOUGHT_KEY] = message[THOUGHT_KEY]
    if calls:
        written["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["function"]["name"],
                          "arguments": arguments_text(c["function"]["arguments"])}}
            for c in calls]
    return written


def arguments_text(arguments: dict) -> str:
    """A call's ``arguments`` as the Chat Completions form writes them: compact JSON, in their own order."""
    return json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)


# ----- calls that could not be read ---------------------------------------------------
def without_unreadable(messages: list[dict], held: set[int],
                       joins_questions: bool) -> tuple[list[dict], set[str]]:
    """``messages`` without each call that could not be read and its result, except in the messages at ``held``; and the ids of the calls that went.

    A reply left with no call goes on in the next reply of its turn, where
    that one is not held, its text and thought before that reply's; else it
    stays where it says something, and goes where it says nothing. With
    ``joins_questions``, two questions this leaves in a row are joined, as a
    question after an unanswered one is.
    """
    kept, _sources, gone = _without_unreadable(messages, held, joins_questions)
    return kept, gone


def _without_unreadable(messages: list[dict], held: set[int], joins_questions: bool
                        ) -> tuple[list[dict], list[frozenset[int]], set[str]]:
    """:func:`without_unreadable`, and for each message kept, the places in ``messages`` of the messages it stands for."""
    kept: list[dict] = []
    sources: list[frozenset[int]] = []
    gone: set[str] = set()
    # A reply whose calls all went, held back with the places it stands for; a later one
    # whose calls all went too joins it.
    emptied: tuple[dict, frozenset[int]] | None = None

    def keep(message: dict, at: int) -> None:
        nonlocal emptied
        if emptied is not None and message["role"] == "assistant" and at not in held:
            said, said_at = emptied
            kept.append(_joined_replies(said, message))
            sources.append(said_at | {at})
        else:
            keep_saying()
            kept.append(message)
            sources.append(frozenset((at,)))
        emptied = None

    def keep_saying() -> None:
        if emptied is not None and (emptied[0]["content"] or emptied[0].get(THOUGHT_KEY)):
            kept.append(emptied[0])
            sources.append(emptied[1])

    for at, message in enumerate(messages):
        calls = message.get("tool_calls") or ()
        if at not in held and any(UNREADABLE_CALL in c for c in calls):
            gone |= {c["id"] for c in calls if UNREADABLE_CALL in c}
            left = [c for c in calls if UNREADABLE_CALL not in c]
            if left:
                keep({**message, "tool_calls": left}, at)
                continue
            said = {k: v for k, v in message.items() if k != "tool_calls"}
            emptied = ((_joined_replies(emptied[0], said), emptied[1] | {at})
                       if emptied is not None else (said, frozenset((at,))))
            continue
        if message["role"] == "tool" and message["tool_call_id"] in gone:
            continue
        keep(message, at)
    keep_saying()
    if joins_questions and gone:
        kept, sources = _questions_joined(kept, sources)
    return kept, sources, gone


def _joined_replies(said: dict, then: dict) -> dict:
    """``then``, a reply, with the text and thought of ``said``, an earlier one, before its own."""
    joined = {**then, "content": _joined_texts(said["content"], then["content"])}
    thought = _joined_texts(said.get(THOUGHT_KEY) or "", then.get(THOUGHT_KEY) or "")
    if thought:
        joined[THOUGHT_KEY] = thought
    return joined


def _joined_texts(first: str, then: str) -> str:
    """Two texts as one, a blank line between them where both say something."""
    return "\n\n".join(text for text in (first, then) if text)


def _questions_joined(messages: list[dict], sources: list[frozenset[int]]
                      ) -> tuple[list[dict], list[frozenset[int]]]:
    """``messages``, each standing for the places ``sources`` gives, with each question joined to the one before it where what stood between them is gone."""
    joined: list[dict] = []
    joined_sources: list[frozenset[int]] = []
    for message, source in zip(messages, sources):
        if joined and message["role"] == joined[-1]["role"] == "user" \
                and min(source) - max(joined_sources[-1]) > 1:
            joined[-1] = joined_question(joined[-1], message)
            joined_sources[-1] |= source
        else:
            joined.append(message)
            joined_sources.append(source)
    return joined, joined_sources


def with_questions_joined(messages: list[dict]) -> list[dict]:
    """``messages`` with each question that follows another, no reply between them, joined to it, as a template that writes no two in a row is given them."""
    joined: list[dict] = []
    for message in messages:
        if joined and message["role"] == joined[-1]["role"] == "user":
            joined[-1] = joined_question(joined[-1], message)
        else:
            joined.append(message)
    return joined


def joined_question(question: dict, later: dict) -> dict:
    """``question`` with the ``later`` question after it, as one."""
    return {**question, "content": f"{question['content']}\n\n{later['content']}"}


# ----- turns kept as generated ----------------------------------------------------------
def generated_turn_records(messages: list[dict], joins_questions: bool) -> list[dict]:
    """The turns ``messages`` keep as generated, JSON-ready, each with the messages it is put back as: those messages, where each turn's tokens begin among them and how many they stand for, and what they are found again by.

    They are found by the Chat Completions messages ``messages`` give for
    them, with the question before them, and with the message after a turn
    that gives none of its own; they are put back as every message those
    stand for, so that a question joined across a turn, or a reply that took
    in its text, comes back as it was. Turns whose messages overlap so are
    one record; a question may close one record and open the next.
    """
    kept, sources, _gone = _without_unreadable(messages, set(), joins_questions)
    portable = [_portable(_as_completions(message)) for message in kept]
    records: list[dict] = []
    for first, last, turn in generated_turns_in(messages):
        end = last
        results_variant = LoopVariant("finding an open turn's results")
        while end + 1 < len(messages) and messages[end + 1]["role"] == "tool":
            results_variant.passes(len(messages) - end)
            end += 1                # the results written so far in an open turn
        own = set(range(first, end + 1))
        window = [k for k, source in enumerate(sources) if source & (own | {first - 1})]
        if not any(sources[k] & own for k in window) and window and window[-1] + 1 < len(kept):
            window.append(window[-1] + 1)
        if not window:
            continue                # nothing of it, or before it, is given back to find it by
        lo = min(min(sources[k]) for k in window)
        hi = max(end, max(max(sources[k]) for k in window))
        part: dict[str, Any] = {"from": window[0], "to": window[-1], "lo": lo, "hi": hi,
                                "kept": [(first, turn)]}
        shared = records[-1]["to"] - part["from"] + 1 if records else 0
        if shared > 1 or shared == 1 and not portable[part["from"]].startswith('["user"'):
            older = records.pop()
            part = {"from": older["from"], "to": max(older["to"], part["to"]),
                    "lo": min(older["lo"], lo), "hi": max(older["hi"], hi),
                    "kept": older["kept"] + part["kept"]}
        records.append(part)
    made = [{"messages": [_exact(m) for m in messages[r["lo"]:r["hi"] + 1]],
             "kept": [{"tokens": list(turn.tokens), "start": first - r["lo"], "span": turn.span}
                      for first, turn in r["kept"]],
             "count": r["to"] - r["from"] + 1, "after": len(kept) - r["to"] - 1,
             "digest": _digest_of(portable[r["from"]:r["to"] + 1])} for r in records]
    return [{**record, "check": _check_of(record)} for record in made]


def _check_of(record: dict) -> str:
    """What tells a record as :func:`generated_turn_records` made it from one changed since: a digest of everything it holds but this."""
    held = {key: value for key, value in record.items() if key != "check"}
    return hashlib.sha256(json.dumps(held, sort_keys=True, ensure_ascii=False).encode(
        "utf-8", "surrogatepass")).hexdigest()


def intact_records(turns: list[dict]) -> list[dict]:
    """The records of ``turns`` that hold what :func:`generated_turn_records` made them with, tokens included; one changed since is left out."""
    return [record for record in turns if record["check"] == _check_of(record)]


def _exact(message: dict) -> dict:
    """``message`` as a generated turn holds it: as the conversation holds it, without the turn itself."""
    return copy.deepcopy({k: v for k, v in message.items() if k != GENERATED_TURN})


def _portable(message: dict) -> str:
    """A Chat Completions message as a caller keeps it, as text: no thought, an empty text and a null one alike, arguments as the objects they encode, in their own order."""
    calls = [[c["id"], c["function"]["name"], _arguments(c["function"]["arguments"])]
             for c in message.get("tool_calls") or ()]
    return json.dumps([message["role"], message.get("content") or "", message.get("tool_call_id"),
                       calls], ensure_ascii=False, separators=(",", ":"))


def _digest_of(portable: list[str]) -> str:
    """A digest of messages, each given as :func:`_portable` writes it."""
    return hashlib.sha256(("[" + ",".join(portable) + "]").encode()).hexdigest()


def with_generated_turns(messages: list[dict], turns: list[dict],
                     joins_questions: bool) -> tuple[list[dict], int]:
    """``messages``, read from the Chat Completions form, with each record of ``turns`` that they still hold put back as the conversation held it, each turn's tokens on its message; and how many turns were.

    The newest record is placed first, where it stood when it was saved
    among places that read alike, else at the newest, and each older one
    before the record placed after it, the question that opens that
    record's messages shared; its messages take the place of the Chat
    Completions ones it was found by. A turn found nowhere is written as the
    template writes it.
    """
    portable = [_portable(_as_completions(message)) for message in messages]
    by_width: dict[int, dict[str, list[int]]] = {}
    places: list[tuple[int, dict]] = []
    before = len(messages)          # where the record placed last begins
    for record in reversed(turns):
        width = record["count"]
        if width not in by_width:
            found: dict[str, list[int]] = {}
            for i in range(len(portable) - width + 1):
                found.setdefault(_digest_of(portable[i:i + width]), []).append(i)
            by_width[width] = found
        fitting = [i for i in by_width[width].get(record["digest"], ()) if i + width <= before]
        # Where it stood when it was saved, among messages that read alike; else the newest.
        at = next((i for i in fitting if len(portable) - i - width == record["after"]),
                  max(fitting, default=None))
        if at is not None:
            places.append((at, record))
            # A question may close one record's messages and open the next record's.
            before = at + (portable[at].startswith('["user"'))
    put = list(messages)
    for at, record in places:       # the newest first, so that no place moves before it is used
        held = copy.deepcopy(record["messages"])
        for turn in record["kept"]:
            last = turn["start"] + turn["span"] - 1
            held[last] = {**held[last], GENERATED_TURN: GeneratedTurn(
                tuple(turn["tokens"]), turn["span"], bool(held[last].get("tool_calls")))}
        put[at:at + record["count"]] = held
    return put, sum(len(record["kept"]) for _, record in places)


def ensure_generated_turns(given) -> None:
    """Refuse generated turns that are not what :func:`generated_turn_records` and the model's fingerprint make, as a conversation gives them.

    Raises:
        TypeError: If ``given`` is neither ``None`` nor a ``dict``.
        ValueError: Naming ``generated_turns``, if it lacks the model, or a
            record lacks its messages, its turns' tokens, start or span, its
            count, its place or its digest, or holds one of another form.
    """
    if given is None:
        return
    if not isinstance(given, dict):
        raise ArgumentTypeError(f"generated_turns must be what conversation() gives, a dict or None, "
                        f"not {type(given).__name__}")
    turns = given.get("turns")
    if not isinstance(given.get("model"), str) or not isinstance(turns, list) \
            or not all(_is_generated_turn_record(turn) for turn in turns):
        raise InvalidArgumentError("generated_turns", "generated_turns is not what "
                                   "conversation() gives; pass that, or None to write every "
                                   "turn as the template writes it")


def _is_generated_turn_record(record) -> bool:
    """Whether ``record`` is one record as :func:`generated_turn_records` gives it."""
    if not isinstance(record, dict) \
            or set(record) != {"messages", "kept", "count", "after", "digest", "check"}:
        return False
    held, kept = record["messages"], record["kept"]
    return (isinstance(held, list) and all(_is_held(m) for m in held)
            and isinstance(kept, list) and bool(kept) and all(
                isinstance(turn, dict) and set(turn) == {"tokens", "start", "span"}
                and isinstance(turn["tokens"], list)
                and all(type(token) is int for token in turn["tokens"])
                and type(turn["start"]) is int and type(turn["span"]) is int
                and turn["span"] > 0 and 0 <= turn["start"]
                and turn["start"] + turn["span"] <= len(held) for turn in kept)
            and type(record["count"]) is int and record["count"] > 0
            and type(record["after"]) is int and record["after"] >= 0
            and isinstance(record["digest"], str) and isinstance(record["check"], str))


def _is_held(message) -> bool:
    """Whether ``message`` is a question, a reply or a result as a generated turn holds one."""
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return False
    if message.get("role") == "user":
        return set(message) == {"role", "content"}
    if message.get("role") == "tool":
        return set(message) == {"role", "tool_call_id", "name", "content"} \
            and isinstance(message["tool_call_id"], str) and isinstance(message["name"], str)
    if message.get("role") != "assistant" or set(message) - {"role", "content", "tool_calls",
                                                             THOUGHT_KEY}:
        return False
    if not isinstance(message.get(THOUGHT_KEY, ""), str):
        return False
    return all(isinstance(c, dict) and isinstance(c.get("id"), str)
               and isinstance(c.get("function"), dict)
               and isinstance(c["function"].get("name"), str)
               and isinstance(c["function"].get("arguments"), dict)
               for c in message.get("tool_calls") or ())
