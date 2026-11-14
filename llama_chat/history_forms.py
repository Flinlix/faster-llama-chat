"""The two forms a conversation's history takes: the Chat Completions messages ``begin`` takes and ``conversation()`` gives, and beside them the turns kept as the tokens they were generated as.

Plain functions over message lists, testable without a model. Inside, the
conversation keeps the messages a template is given: a call's arguments as
the object they encode, a call the model wrote unreadably marked as one, and
a turn kept as generated on the message it ends with. See design_history.md,
"Two history forms".
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import logging

from .errors import InvalidArgumentError
from .messages import KEPT_TURN, THOUGHT_KEY, UNKNOWN_FUNCTION, UNREADABLE_CALL, result_message
from .rendering import KeptTurn, kept_turns

_log = logging.getLogger(__name__)

# The keys each role's message holds in the Chat Completions form this library takes, ``role`` included.
_COMPLETIONS_KEYS = {"system": {"role", "content"}, "user": {"role", "content"},
                "assistant": {"role", "content", "tool_calls", THOUGHT_KEY},
                "tool": {"role", "tool_call_id", "content"}}
_CALL_KEYS = {"id", "type", "function"}
_FUNCTION_KEYS = {"name", "arguments"}


# ----- the Chat Completions form, taken -----------------------------------------------------------
def read_messages(messages: list) -> list[dict]:
    """``begin``'s messages, in the Chat Completions form, as the messages a template is given.

    A call whose arguments are no JSON object is marked as one that could
    not be read; ``""`` is no arguments. A reply's ``content`` may be
    ``None`` where it makes calls, and is then empty.

    Raises:
        TypeError: If a message is no dictionary, or a text, an id or a
            call's arguments is no ``str``.
        ValueError: If a message holds a key its role does not take here,
            its role is none of ``system``, ``user``, ``assistant`` and
            ``tool``, a system message is not the first, or a call lacks its
            id or is of another type than ``function``.
    """
    read = []
    for i, item in enumerate(messages):
        where = f"messages[{i}]"
        if not isinstance(item, dict):
            raise TypeError(f"{where} is a chat message dictionary, such as "
                            f"{{'role': 'user', 'content': text}}, not the "
                            f"{type(item).__name__} {item!r}")
        role = item.get("role")
        if role not in _COMPLETIONS_KEYS:
            raise InvalidArgumentError("messages", f"{where}'s role is 'system', 'user', "
                                                   f"'assistant' or 'tool', not {role!r}")
        extra = sorted(set(item) - _COMPLETIONS_KEYS[role])
        if extra:
            raise InvalidArgumentError("messages", f"{where}, a {role} message, holds {extra[0]!r}, "
                                                   f"which it does not take here; it holds "
                                                   f"{sorted(_COMPLETIONS_KEYS[role])}")
        if "content" not in item:
            raise InvalidArgumentError("messages", f"{where} needs its 'content'")
        if role == "system" and i:
            raise InvalidArgumentError("messages", f"{where} is a system message, which may only "
                                                   "open the messages")
        read.append(_reply(item, where) if role == "assistant"
                    else _result(item, where) if role == "tool"
                    else {"role": role, "content": _text(item["content"], f"{where}.content")})
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
        raise TypeError(f"{where} must be a str, not {type(value).__name__}")
    return value


def _reply(item: dict, where: str) -> dict:
    """The assistant message ``item`` as a template is given it."""
    calls = item.get("tool_calls") or []
    if not isinstance(calls, list):
        raise TypeError(f"{where}.tool_calls must be a list, not {type(calls).__name__}")
    content = item["content"]
    if content is None and not calls:
        raise InvalidArgumentError("messages", f"{where}.content is null, which only a reply "
                                               "that makes calls may be")
    message = {"role": "assistant",
               "content": "" if content is None else _text(content, f"{where}.content")}
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
        raise TypeError(f"{where} must be a dictionary, not {type(call).__name__}")
    extra = sorted(set(call) - _CALL_KEYS)
    if extra:
        raise InvalidArgumentError("messages", f"{where} holds {extra[0]!r}, which a call does "
                                               f"not take; it holds {sorted(_CALL_KEYS)}")
    if call.get("type", "function") != "function":
        raise InvalidArgumentError("messages", f"{where} is of type {call['type']!r}; only "
                                               "'function' calls are taken")
    given_id, function = call.get("id"), call.get("function")
    if not isinstance(given_id, str) or not given_id:
        raise InvalidArgumentError("messages", f"{where} needs its 'id', a non-empty string")
    if not isinstance(function, dict) or set(function) - _FUNCTION_KEYS:
        raise InvalidArgumentError("messages", f"{where}.function holds 'name' and 'arguments', "
                                               f"not {function!r}")
    name, given = function.get("name"), function.get("arguments", "")
    if not isinstance(name, str):
        raise TypeError(f"{where}.function.name must be a str, not {type(name).__name__}")
    if not isinstance(given, str):
        raise TypeError(f"{where}.function.arguments must be the arguments as a JSON string, "
                        f"not {type(given).__name__}")
    arguments = _arguments(given)
    if arguments is None or not name:
        _log.info("%s could not be read (%s), so it is left out with its result", where,
                  "it names no function" if not name else "its arguments are no JSON object")
        return unreadable_call(given_id)
    return {"id": given_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _arguments(given: str) -> dict | None:
    """The object ``given`` encodes, ``{}`` for ``""``; ``None`` where it encodes none."""
    if given == "":
        return {}
    try:
        arguments = json.loads(given)
    except ValueError:
        return None
    return arguments if isinstance(arguments, dict) else None


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
        while owed and UNREADABLE_CALL in owed[0] and owed[0]["id"] != upto:
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
    that one is not held; else it stays where it says something, and goes
    where it says nothing. With ``joins_questions``, two questions this
    leaves in a row are joined, as a question after an unanswered one is.
    """
    kept, gone, emptied, placed = [], set(), None, []
    for at, message in enumerate(messages):
        calls = message.get("tool_calls") or ()
        if at not in held and any(UNREADABLE_CALL in c for c in calls):
            gone |= {c["id"] for c in calls if UNREADABLE_CALL in c}
            left = [c for c in calls if UNREADABLE_CALL not in c]
            if left:
                kept.append({**message, "tool_calls": left})
                placed.append(at)
                continue
            _keep_saying(emptied, kept, placed)
            emptied = ({k: v for k, v in message.items() if k != "tool_calls"}, at)
            continue
        if message["role"] == "tool" and message["tool_call_id"] in gone:
            continue
        if emptied is not None and message["role"] == "assistant" and at not in held:
            said = emptied[0]
            message = {**message, "content": "\n\n".join(
                text for text in (said["content"], message["content"]) if text),
                **({THOUGHT_KEY: said[THOUGHT_KEY]} if said.get(THOUGHT_KEY)
                   and not message.get(THOUGHT_KEY) else {})}
        else:
            _keep_saying(emptied, kept, placed)
        emptied = None
        kept.append(message)
        placed.append(at)
    _keep_saying(emptied, kept, placed)
    if joins_questions and gone:
        kept = _questions_joined(kept, placed)
    return kept, gone


def _keep_saying(emptied: tuple[dict, int] | None, kept: list[dict], placed: list[int]) -> None:
    """Keep ``emptied``, a reply whose calls were all left out, as a reply of its own where it says something."""
    if emptied is not None and (emptied[0]["content"] or emptied[0].get(THOUGHT_KEY)):
        kept.append(emptied[0])
        placed.append(emptied[1])


def _questions_joined(messages: list[dict], placed: list[int]) -> list[dict]:
    """``messages``, each from where ``placed`` says, with each question joined to the one before it where what stood between them is gone."""
    joined: list[dict] = []
    before = None
    for message, place in zip(messages, placed):
        if joined and message["role"] == joined[-1]["role"] == "user" and place - before > 1:
            joined[-1] = joined_question(joined[-1], message)
        else:
            joined.append(message)
        before = place
    return joined


def joined_question(question: dict, later: dict) -> dict:
    """``question`` with the ``later`` question after it, as one."""
    return {**question, "content": f"{question['content']}\n\n{later['content']}"}


# ----- turns kept as generated ----------------------------------------------------------
def saved_turns(messages: list[dict], joins_questions: bool) -> list[dict]:
    """Each turn ``messages`` keep as generated, JSON-ready: its tokens, its messages as the conversation holds them through the end of the turn, how many of them its tokens stand for, and the key it is found again by."""
    turns = []
    for first, last, turn in kept_turns(messages):
        end = last
        while end + 1 < len(messages) and messages[end + 1]["role"] == "tool":
            end += 1                # the results written so far in an open turn
        held = [_exact(m) for m in messages[first:end + 1]]
        turns.append({"tokens": list(turn.tokens), "span": turn.span, "messages": held,
                      **_key(messages[first - 1:first], held, joins_questions)})
    return turns


def _exact(message: dict) -> dict:
    """``message`` as a saved turn holds it: as the conversation holds it, without the turn itself."""
    return copy.deepcopy({k: v for k, v in message.items() if k != KEPT_TURN})


def _key(before: list[dict], held: list[dict], joins_questions: bool) -> dict:
    """What a saved turn of the messages ``held`` after ``before`` is found again by: how many Chat Completions messages it stands for, the one before them counted, and their digest."""
    found = completions_messages(before + held, joins_questions)
    return {"count": len(found), "digest": _digest(found)}


def _digest(messages: list[dict]) -> str:
    """A digest of Chat Completions messages as a caller keeps them: no thought, an empty text and a null one alike, arguments as the objects they encode, in their own order."""
    def portable(message: dict) -> list:
        calls = [[c["id"], c["function"]["name"], _arguments(c["function"]["arguments"])]
                 for c in message.get("tool_calls") or ()]
        return [message["role"], message.get("content") or "", message.get("tool_call_id"), calls]
    return hashlib.sha256(json.dumps([portable(m) for m in messages], ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def with_saved_turns(messages: list[dict], turns: list[dict],
                     joins_questions: bool) -> tuple[list[dict], int]:
    """``messages``, read from the Chat Completions form, with each of ``turns`` that they still hold put back as the conversation held it, its tokens on its message; and how many were.

    A turn is found by its Chat Completions messages and the question before them, the
    newest place first, and its messages take the place of its Chat Completions ones;
    one found nowhere is written as the template writes it.
    """
    completions = [_as_completions(message) for message in messages]
    places: list[tuple[int, int]] = []      # where a turn's window begins, and which turn
    taken: set[int] = set()
    for n, turn in enumerate(turns):
        width = turn["count"]
        at = next((i for i in range(len(completions) - width, -1, -1)
                   if not taken.intersection(range(i, i + width))
                   and _digest(completions[i:i + width]) == turn["digest"]), None)
        if at is not None:
            places.append((at, n))
            taken.update(range(at, at + width))
    put = list(messages)
    for at, n in sorted(places, reverse=True):
        turn = turns[n]
        held = copy.deepcopy(turn["messages"])
        last = turn["span"] - 1
        held[last] = {**held[last], KEPT_TURN: KeptTurn(
            tuple(turn["tokens"]), turn["span"], bool(held[last].get("tool_calls")))}
        put[at + 1:at + turn["count"]] = held
    return put, len(places)


def ensure_saved_turns(given) -> None:
    """Refuse saved turns that are not what :func:`saved_turns` and the model's fingerprint make, as a conversation gives them.

    Raises:
        TypeError: If ``given`` is neither ``None`` nor a ``dict``.
        ValueError: Naming ``generated_turns``, if it lacks the model or a
            turn's tokens, messages, span, count or digest, or holds one of
            another form.
    """
    if given is None:
        return
    if not isinstance(given, dict):
        raise TypeError(f"generated_turns must be what conversation() gives, a dict or None, "
                        f"not {type(given).__name__}")
    turns = given.get("turns")
    if not isinstance(given.get("model"), str) or not isinstance(turns, list) \
            or not all(_is_saved_turn(turn) for turn in turns):
        raise InvalidArgumentError("generated_turns", "generated_turns is not what "
                                   "conversation() gives; pass that, or None to write every "
                                   "turn as the template writes it")


def _is_saved_turn(turn) -> bool:
    """Whether ``turn`` is one turn as :func:`saved_turns` gives it."""
    if not isinstance(turn, dict) or set(turn) != {"tokens", "span", "messages", "count", "digest"}:
        return False
    tokens, span, held = turn["tokens"], turn["span"], turn["messages"]
    return (isinstance(tokens, list) and all(type(token) is int for token in tokens)
            and isinstance(held, list) and type(span) is int and 0 < span <= len(held)
            and type(turn["count"]) is int and turn["count"] > 0
            and isinstance(turn["digest"], str) and all(_is_held(m) for m in held))


def _is_held(message) -> bool:
    """Whether ``message`` is a reply or a result as a saved turn holds one."""
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return False
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
