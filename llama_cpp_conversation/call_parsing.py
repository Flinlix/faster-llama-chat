"""Reading a model's tool calls back out of the text it generated.

A call is read in the one shape the model's template writes it in: a JSON
object, read by the keys the template names the function and its arguments
with, or the delimiters recovered from that template. Nothing here runs
anything; see :mod:`llama_cpp_conversation.tools` for that half.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from typing import Any

from .template import CallBody, json_arguments
from .tools import MAX_ARGUMENT_DEPTH, ToolCall, json_value_fault, nesting_depth
from .loop_variant import LoopVariant



def parse_calls(body: str, call_body: CallBody | None,
                lead_in: str = "", parameters: dict[str, dict] | None = None,
                json_keys: tuple[str, ...] | None = None,
                json_id_key: str | None = None) -> list[ToolCall]:
    """Read the calls out of the text a model generated between its call tags.

    Args:
        body: What the model generated, tags excluded.
        call_body: This model's argument spelling, where its calls are not
            JSON; ``None`` where they are, or cannot be read.
        lead_in: Fixed text the model writes before the function name.
        parameters: The parameter schema of each declared tool, by its name.
            A value written without a type mark is text where its schema asks
            for text and not for the literal it spells, and that literal
            otherwise.
        json_keys: Where the model writes a call as a JSON object, the keys
            its template writes: the one naming the function, the one holding
            the arguments, then any others. Such a body is read as JSON
            alone, and an object with another key is not a call.
        json_id_key: Of ``json_keys``, the one a call's id is written under,
            where the template writes one.

    Returns:
        One :class:`ToolCall` per call, in the order the model wrote them,
        with the id the model wrote for it where the format holds one.
        Usually one; a JSON array holds a whole batch.

    Raises:
        ValueError: If ``body`` is not a call this can read, among them a call
            whose arguments are no object or nest deeper than
            :data:`MAX_ARGUMENT_DEPTH`.
    """
    try:
        read = _read_calls(body, call_body, lead_in, json_keys, json_id_key)
    except RecursionError as e:
        raise ValueError("the call's arguments nest deeper than can be read") from e
    calls = []
    for call, written in read:
        # Each entry of a batch is judged on its own: one that cannot be read is unreadable alone.
        try:
            calls.append(call if call.status != "complete" else _checked(call, parameters or {}))
        except ValueError:
            if len(read) == 1:
                raise
            calls.append(ToolCall("", {}, status="unreadable", text=written))
    return calls


def _checked(call: ToolCall, schemas: dict[str, dict]) -> ToolCall:
    """``call`` with its arguments read against its tool's schema.

    Raises:
        ValueError: If its arguments nest deeper than :data:`MAX_ARGUMENT_DEPTH`
            or are no JSON every model can be given.
    """
    try:
        typed = replace(call, arguments=_typed(call.arguments, schemas.get(call.name),
                                               schemas.get(call.name)))
    except RecursionError as e:
        raise ValueError("the call's arguments nest deeper than can be read") from e
    if nesting_depth(typed.arguments) > MAX_ARGUMENT_DEPTH:
        raise ValueError(f"the call's arguments nest deeper than {MAX_ARGUMENT_DEPTH} levels")
    # Held to what a history's call is held to, so that a conversation begun again keeps it.
    if (fault := json_value_fault(typed.arguments)) is not None:
        raise ValueError(f"the arguments of the call to {call.name!r} are no JSON every "
                         f"model can be given: {fault}")
    return typed


def _read_calls(body: str, call_body: CallBody | None, lead_in: str,
                json_keys: tuple[str, ...] | None, json_id_key: str | None
                ) -> list[tuple[ToolCall, str]]:
    """:func:`parse_calls` before the arguments are checked, each call with the text the model wrote for it."""
    # Unstripped: the recovered delimiters carry their own whitespace.
    text = body
    if lead_in and text.startswith(lead_in):
        text = text[len(lead_in):]
    if json_keys is not None:
        # Whole first: a model whose call *is* JSON would have the lead-in cut out
        # of that JSON. One that opens with a tag then writes JSON needs the second.
        candidates = [body.strip()] + ([text.strip()] if text is not body else [])
        for candidate in candidates:
            calls = _as_json_calls(candidate, json_keys, json_id_key)
            if calls:
                return calls
        raise ValueError(f"not a call as this model writes one in JSON: {body!r}")
    if call_body is None:
        raise ValueError(f"no call_body to read this call with: {body!r}")
    return [(_as_delimited_call(text, call_body), body)]


def _as_json_calls(text: str, keys: tuple[str, ...], id_key: str | None
                   ) -> list[tuple[ToolCall, str]]:
    """Read the calls of a JSON body by ``keys``, each with its entry's text, or nothing where none of them is a call."""
    try:
        decoded = _SPELLED.decode(text)
    except ValueError:
        return []
    batch = decoded if isinstance(decoded, list) else [decoded]
    written = _entry_texts(text) if isinstance(decoded, list) else [text.strip(_JSON_SPACE)]
    calls = []
    for item, entry in zip(batch, written):
        try:
            calls.append(_one_json_call(item, keys, id_key))
        except ValueError:
            if len(batch) == 1:
                raise
            calls.append(ToolCall("", {}, status="unreadable", text=entry))
    if not calls or all(call is None for call in calls):
        return []
    # Each entry of a batch is judged on its own: one that is no call is unreadable alone.
    return [(call or ToolCall("", {}, status="unreadable", text=entry), entry)
            for call, entry in zip(calls, written)]


def _entry_texts(text: str) -> list[str]:
    """The text of each entry of the JSON list ``text``, as the model wrote it."""
    at, entries = text.index("[") + 1, list[str]()
    entries_variant = LoopVariant("finding the entries of a batch")
    while True:
        entries_variant.passes(len(text) - at)
        at = _skip_space(text, at)
        if text[at] == "]":
            return entries
        _value, end = _SPELLED.raw_decode(text, at)
        entries.append(text[at:end])
        at = _skip_space(text, end)
        at += text[at] == ","


def _skip_space(text: str, at: int) -> int:
    """Where the first character from ``at`` on that is no JSON whitespace stands."""
    return len(text) - len(text[at:].lstrip(_JSON_SPACE))


def _one_json_call(decoded, keys: tuple[str, ...], id_key: str | None) -> ToolCall | None:
    """Read one entry of a batch by the keys its template writes, or ``None`` if it is not a call.

    Raises:
        ValueError: If it names a function but its arguments are no object.
    """
    if not isinstance(decoded, dict) or not set(decoded) <= set(keys):
        return None
    name_key, arguments_key = keys[:2]
    name = decoded.get(name_key)
    if not isinstance(name, str) or isinstance(name, _Unmarked) or not name:
        return None
    arguments = decoded.get(arguments_key, {})
    if not isinstance(arguments, dict):
        raise ValueError(f"the arguments of the call to {name!r} are no object: {arguments!r}")
    written_id = decoded.get(id_key) if id_key is not None else None
    return ToolCall(name=name, arguments=arguments,
                    id=written_id if isinstance(written_id, str) else "")


def _as_delimited_call(text: str, call_body: CallBody) -> ToolCall:
    """Read a call written with this model's own delimiters."""
    written_id = ""
    if call_body.id_open and call_body.id_open in text.partition(call_body.name_close)[0]:
        name, _, rest = text.partition(call_body.id_open)
        written_id, separator, rest = rest.partition(call_body.name_close)
        written_id, found = written_id.strip(), bool(separator)
    elif call_body.name_close:
        # A call written without its id is read as one too.
        name, separator, rest = text.partition(call_body.name_close)
        found = bool(separator)
    else:
        # The name runs up to the first key's tag, or to the end.
        name, separator, rest = text.partition(call_body.key_open)
        rest, found = separator + rest, True
    name = name.strip()
    if not found or not name:
        raise ValueError(f"not a call: {text!r}")
    if call_body.arguments_json:
        if rest.strip() == call_body.empty_close.strip():
            return ToolCall(name=name, arguments={}, id=written_id)
        whole = _as_json_arguments(rest, call_body)
        if whole is None:
            raise ValueError(f"arguments that are no JSON object: {rest!r}")
        return ToolCall(name=name, arguments=whole, id=written_id)
    if call_body.key_in_close:
        return ToolCall(name=name, arguments=_parse_keyed_pairs(rest, call_body), id=written_id)
    # Compared whole, not by its ending: a model that wraps each argument closes
    # the last one with the very same text.
    if rest == call_body.empty_close:
        return ToolCall(name=name, arguments={}, id=written_id)
    # A model may write whitespace after the last argument's closing tag.
    closing = call_body.body_close.rstrip()
    if closing and not rest.rstrip().endswith(closing):
        raise ValueError(f"an argument that never closes: {rest!r}")
    if closing:
        rest = rest.rstrip()[:len(rest.rstrip()) - len(closing)]
    if call_body.key_open and rest.startswith(call_body.key_open):
        rest = rest[len(call_body.key_open):]
    return ToolCall(name=name, arguments=_parse_pairs(rest, call_body), id=written_id)


def _parse_keyed_pairs(text: str, call_body: CallBody) -> dict:
    """Read pairs whose values each close with a tag naming their own key."""
    parsed: dict[str, object] = {}
    at = len(call_body.key_open) if text.startswith(call_body.key_open) else 0
    pairs_variant = LoopVariant("reading keyed pairs")
    while at < len(text):
        pairs_variant.passes(len(text) - at)
        key_end = text.find(call_body.key_close, at)
        if key_end < 0:
            if text[at:].strip():
                raise ValueError(f"text after the last argument: {text[at:]!r}")
            break
        key, at = text[at:key_end], key_end + len(call_body.key_close)
        separator = call_body.pair_sep.replace("{key}", key)
        closing = call_body.body_close.replace("{key}", key).rstrip()
        if call_body.member_open:
            # A nested member may carry this key too, so its closing is matched by depth.
            end = _matching_close(text, at, call_body.member_open.replace("{key}", key),
                                  call_body.member_close.replace("{key}", key))
            following = end if end >= 0 and text.startswith(separator, end) else -1
        else:
            following = text.find(separator, at)
            end = following if following >= 0 else text.find(closing, at)
        if end < 0:
            raise ValueError(f"an argument that never closes: {text[at:]!r}")
        after = end + len(closing) if text.startswith(closing, end) else end
        if following < 0 and (after == end and closing or text[after:].strip()):
            raise ValueError(f"text after the last argument: {text[after:]!r}")
        _add_once(parsed, _unquote(key.strip(), call_body),
                  _parse_typed_value(text[at:end], call_body))
        if following < 0:
            return parsed
        at = following + len(separator)
    if parsed:
        raise ValueError(f"a separator after the last argument: {text[-40:]!r}")
    return parsed


def _as_json_arguments(rest: str, call_body: CallBody) -> dict | None:
    """The arguments after a call's name as the JSON object they are, followed by nothing but what the template writes after one; ``None`` otherwise."""
    text = json_arguments(call_body.name_close, rest)
    try:
        decoded, end = _SPELLED.raw_decode(text)
    except ValueError:
        return None
    close = call_body.body_close
    after_object = close[close.find("}") + 1:].strip() if "}" in close else ""
    if not isinstance(decoded, dict) or text[end:].strip() not in ("", after_object):
        return None
    return decoded


def _parse_pairs(text: str, call_body: CallBody) -> dict:
    """Read ``key`` and value pairs separated by this model's own delimiter."""
    parsed: dict[str, object] = {}
    for chunk in _split_on(text, call_body.pair_sep, call_body):
        key, found, value = chunk.partition(call_body.key_close)
        if not found:
            raise ValueError(f"an argument with no value: {chunk!r}")
        _add_once(parsed, _unquote(key.strip(), call_body), _parse_typed_value(value, call_body))
    return parsed


def _add_once(arguments: dict, key: str, value) -> None:
    """Add ``value`` under ``key``, which ``arguments`` must not hold yet.

    Raises:
        ValueError: If the model wrote ``key`` twice, so that the value it
            meant cannot be told.
    """
    if key in arguments:
        raise ValueError(f"the argument {key!r} written twice")
    arguments[key] = value


def _parse_typed_value(text: str, call_body: CallBody) -> Any:
    """Read one value, after the type the model writes before it, where it writes one."""
    if not call_body.type_close:
        return _parse_value(text, call_body)
    written_type, found, value = text.partition(call_body.type_close)
    if not found:
        raise ValueError(f"a value with no type: {text!r}")
    return value if written_type == call_body.string_type else _parse_value(value, call_body)


def _parse_value(text: str, call_body: CallBody) -> Any:
    """Read one argument value, keeping its type where the model marked one."""
    if call_body.member_open and text.startswith(call_body.member_open.split("{key}")[0]):
        try:
            return _parse_elements(text, call_body)
        except ValueError:
            pass                    # a string that opens like a member
    text = text.strip()
    if text[:1] == '"' and text[-1:] == '"':
        # Read as JSON so the escapes come back as the characters they stand for.
        try:
            return json.loads(text)
        except ValueError:
            pass
    unquoted = _unquote(text, call_body)
    if unquoted is not text:
        return unquoted             # the model marked it as a string
    if text[:1] in ("{", "[") and text[-1:] in ("}", "]"):
        nested = _nested_value(text, call_body)
        return text if nested is None else _Bracketed(text, nested)
    return _Unmarked(text)


def _nested_value(text: str, call_body: CallBody) -> Any:
    """The object or list ``text`` spells, or ``None`` where it spells neither."""
    # Nested values are JSON in some models, the Python literal a template
    # interpolates in others, and these delimiters in the rest.
    try:
        return _SPELLED.decode(text)
    except ValueError:
        pass
    try:
        found = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        found = None
    if isinstance(found, (dict, list)):
        return found
    try:
        if text[0] == "{":
            return _parse_pairs(text[1:-1], call_body)
        return [_parse_value(item, call_body)
                for item in _split_on(text[1:-1], call_body.pair_sep, call_body)]
    except ValueError:
        return None                 # a string a template wrote as it stands


class _Unmarked(str):
    """A value a call format wrote without marking its type, read by :func:`_typed`."""


class _Bracketed(_Unmarked):
    """An object or a list a call format wrote without marking its type: that value, or its text where the schema asks for text alone."""

    value: Any

    def __new__(cls, text: str, value):
        found = super().__new__(cls, text)
        found.value = value
        return found


def object_once(pairs: list[tuple[str, Any]]) -> dict:
    """A JSON object from its ``pairs``.

    Raises:
        ValueError: If a key comes twice, so that the value meant cannot be told.
    """
    found: dict = {}
    for key, value in pairs:
        _add_once(found, key, value)
    return found


# Reads JSON with each number kept as the text the model wrote, read by its tool's
# schema afterwards: a number where text is asked is that text, as written.
_SPELLED = json.JSONDecoder(parse_int=_Unmarked, parse_float=_Unmarked,
                            object_pairs_hook=object_once)
_JSON_SPACE = " \t\n\r"


def _typed(value, schema, root) -> Any:
    """``value`` with each :class:`_Unmarked` text in it read against the part of ``schema`` it fills, ``root`` holding what a ``$ref`` names."""
    branches = _branches(schema, root, set())
    if isinstance(value, _Bracketed):
        if _asks_text_alone(branches):
            return str(value)
        value = value.value
    if isinstance(value, _Unmarked):
        return _read_unmarked(str(value), branches)
    if isinstance(value, dict):
        def part(key):
            return {"anyOf": [_properties(branch).get(key, branch.get("additionalProperties"))
                              for branch in branches]}
        return {key: _typed(item, part(key), root) for key, item in value.items()}
    if isinstance(value, list):
        items = {"anyOf": [branch.get("items") for branch in branches]}
        return [_typed(item, items, root) for item in value]
    return value


def _branches(schema, root, seen: set[int]) -> list[dict]:
    """``schema`` and every schema its ``anyOf``, ``oneOf`` and ``$ref`` into ``root`` lead to, each once; ``seen`` holds those found so far."""
    if not isinstance(schema, dict) or id(schema) in seen:
        return []
    seen.add(id(schema))
    found = [schema]
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        target = root
        for name in ref[2:].split("/"):
            name = name.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict):
                target = target.get(name)
            elif isinstance(target, list) and name.isascii() and name.isdigit() and int(name) < len(target):
                target = target[int(name)]
            else:
                target = None
        found += _branches(target, root, seen)
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            found += [branch for item in schema[key] for branch in _branches(item, root, seen)]
    return found


def _properties(schema: dict) -> dict:
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _asks_text_alone(branches: list[dict]) -> bool:
    """Whether ``branches`` ask for text and for no object or list."""
    allowed = set()
    for branch in branches:
        kind = branch.get("type")
        allowed |= {kind} if isinstance(kind, str) else set(kind) if isinstance(kind, list) else set()
    return "string" in allowed and not allowed & {"array", "object"}


def _read_unmarked(text: str, branches: list[dict]) -> Any:
    """``text`` as the literal it spells, unless one of ``branches`` asks for text and none for that literal."""
    allowed = set()
    for branch in branches:
        choices = branch.get("enum")
        if isinstance(choices, list) and text in choices:
            return text
        kind = branch.get("type")
        allowed |= {kind} if isinstance(kind, str) else set(kind) if isinstance(kind, list) else set()
    if not text and "string" not in allowed and allowed & {"object", "array"}:
        # A format that writes a value between tags writes an empty object or list as nothing.
        return {} if "object" in allowed else []
    literal = _literal(text)
    if "string" in allowed and not allowed & _json_types(literal):
        return text
    return literal


def _literal(text: str) -> Any:
    """The JSON or Python literal ``text`` spells, or ``text`` itself where it spells none."""
    # Both spellings: JSON literals, and a template interpolating Python ones.
    if text in ("true", "false", "True", "False"):
        return text.lower() == "true"
    if text in ("null", "None"):
        return None
    for number in (int, float):
        try:
            value = number(text)
        except ValueError:
            continue
        if number is float and not math.isfinite(value) and text.strip().lstrip("+-").isdigit():
            raise ValueError(f"a number too long to be read: {text[:40]}...")
        return value
    return text


def _json_types(value) -> set[str]:
    """The JSON Schema types ``value`` satisfies."""
    if value is None:
        return {"null"}
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, int):
        return {"integer", "number"}
    return {"number"} if isinstance(value, float) else {"string"}


def _parse_elements(text: str, call_body: CallBody) -> Any:
    """Read a nested value written in this model's own member and item delimiters.

    A run of items is a list, and anything else an object; an object whose
    only key is the items' own name is written the same way and reads as a list.
    """
    if text.startswith(call_body.item_open):
        try:
            return [_parse_value(value, call_body) for value in
                    _elements(text, lambda _: call_body.item_open, lambda _: call_body.item_close)]
        except ValueError:
            pass
    head, tail = call_body.member_open.split("{key}")
    keys: list[str] = []

    def opening(at: int) -> str:
        key_end = text.find(tail, at + len(head)) if text.startswith(head, at) else -1
        if key_end < 0:
            raise ValueError(f"not a member: {text[at:]!r}")
        keys.append(text[at + len(head):key_end])
        return head + keys[-1] + tail

    values = _elements(text, opening, lambda _: call_body.member_close.replace("{key}", keys[-1]))
    return {key: _parse_value(value, call_body) for key, value in zip(keys, values)}


def _elements(text: str, opening, closing) -> list[str]:
    """The values of the elements ``text`` consists of, each between ``opening(at)`` and its closing."""
    values, at = [], 0
    elements_variant = LoopVariant("reading elements")
    while at < len(text):
        elements_variant.passes(len(text) - at)
        opened = opening(at)
        if not text.startswith(opened, at):
            raise ValueError(f"not an element: {text[at:]!r}")
        closed = closing(at)
        start = at + len(opened)
        end = _matching_close(text, start, opened, closed)
        if end < 0:
            raise ValueError(f"an element that never closes: {text[at:]!r}")
        values.append(text[start:end])
        at = end + len(closed)
    if not values:
        raise ValueError("no elements")
    return values


def _matching_close(text: str, at: int, opening: str, closing: str) -> int:
    """Where the ``closing`` that matches an ``opening`` just before ``at`` starts, or -1."""
    depth = 0
    closes_variant = LoopVariant("finding the matching close")
    while True:
        closes_variant.passes(len(text) - at)
        next_open, next_close = text.find(opening, at), text.find(closing, at)
        if next_close < 0:
            return -1
        if 0 <= next_open < next_close:
            depth, at = depth + 1, next_open + len(opening)
        elif depth:
            depth, at = depth - 1, next_close + len(closing)
        else:
            return next_close


def _unquote(text: str, call_body: CallBody) -> str:
    """``text`` with the model's string wrapping removed, or ``text`` itself."""
    opener, closer = call_body.quote_open, call_body.quote_close
    if not opener or not closer:
        return text
    if len(text) >= len(opener) + len(closer) \
            and text.startswith(opener) and text.endswith(closer):
        return text[len(opener):len(text) - len(closer)]
    return text


def _starts_a_value(before: str, call_body: CallBody) -> bool:
    """Whether a key or a value may start after ``before``, the text of its part so far: a quote anywhere else is a character of the value."""
    before = before.rstrip()
    return not before or before.endswith(call_body.key_close) or before[-1] in "[{,:"


def _split_on(text: str, separator: str, call_body: CallBody) -> list[str]:
    """Split on ``separator`` where it is not nested inside something.

    Objects and lists count as nesting, and so does a quoted string unless this
    model's quote also occurs inside its own structural delimiters.
    """
    if not separator:
        return [text] if text.strip() else []
    opener, closer = call_body.quote_open, call_body.quote_close
    # Such a quote cannot mark nesting: the delimiters would open and close
    # strings of their own. The separators are then the only reliable signal.
    if opener and any(opener in delimiter for delimiter in
                      (call_body.key_open, call_body.key_close, separator,
                       call_body.body_close)):
        opener = closer = ""
    parts, start, depth, quoted, at = [], 0, 0, False, 0
    splits_variant = LoopVariant("splitting on a separator")
    while at < len(text):
        splits_variant.passes(len(text) - at)
        if opener and not quoted and text.startswith(opener, at) \
                and _starts_a_value(text[start:at], call_body):
            quoted, at = True, at + len(opener)
            continue
        if quoted:
            if closer and text.startswith(closer, at):
                quoted, at = False, at + len(closer)
                follows = text[at:].lstrip()
                bare_separator = separator.strip()
                if depth == 0 and bare_separator and follows.startswith(bare_separator) \
                        and not text.startswith(separator, at):
                    # The separator written without its spaces after a quoted value.
                    parts.append(text[start:at])
                    at = start = len(text) - len(follows) + len(bare_separator)
                elif call_body.escapes_quotes and follows and not (
                        follows.startswith((",", ":", "]", "}")) or not separator.strip()
                        or follows.startswith(bare_separator)):
                    # Where the template escapes a quote, an unescaped one ends the value.
                    raise ValueError(f"text after a quoted value: {follows[:40]!r}")
                continue
            # A quote the template escapes, as JSON and Python write it, is part of the value.
            at += 2 if call_body.escapes_quotes and text[at] == "\\" else 1
            continue
        if depth == 0 and text.startswith(separator, at):
            parts.append(text[start:at])
            at += len(separator)
            start = at
            continue
        character = text[at]
        if character in "{[":
            depth += 1
        elif character in "}]":
            # Never below zero: a delimiter may carry an unmatched bracket.
            depth = max(0, depth - 1)
        at += 1
    parts.append(text[start:])
    if not any(part.strip() for part in parts):
        return []
    if not all(part.strip() for part in parts):
        raise ValueError(f"an empty argument between separators: {text!r}")
    return parts
