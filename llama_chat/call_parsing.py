"""Reading a model's tool calls back out of the text it generated.

A call is read in the one shape the model's template writes it in: a JSON
object, read by the keys the template names the function and its arguments
with, or the delimiters recovered from that template. Nothing here runs
anything; see :mod:`llama_chat.tools` for that half.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from typing import Any

from .template import CallBody, json_arguments
from .tools import ToolCall

# How many levels of lists and objects a call's arguments may nest, the arguments'
# own object counted. Well inside the interpreter's recursion limit, so that every
# later step that walks the arguments, copying and rendering them, completes.
MAX_ARGUMENT_DEPTH = 100


def parse_calls(body: str, call_body: CallBody | None,
                lead_in: str = "", parameters: dict[str, dict] | None = None,
                json_keys: tuple[str, ...] | None = None) -> list[ToolCall]:
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

    Returns:
        One :class:`ToolCall` per call, in the order the model wrote them.
        Usually one; a JSON array holds a whole batch.

    Raises:
        ValueError: If ``body`` is not a call this can read, among them a call
            whose arguments are no object or nest deeper than
            :data:`MAX_ARGUMENT_DEPTH`.
    """
    try:
        calls = _read_calls(body, call_body, lead_in, json_keys)
    except RecursionError as e:
        raise ValueError("the call's arguments nest deeper than can be read") from e
    for call in calls:
        if _nesting_depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            raise ValueError(f"the call's arguments nest deeper than {MAX_ARGUMENT_DEPTH} levels")
    schemas = parameters or {}
    return [replace(call, arguments=_typed(call.arguments, schemas.get(call.name),
                                           schemas.get(call.name)))
            for call in calls]


def _read_calls(body: str, call_body: CallBody | None, lead_in: str,
                json_keys: tuple[str, ...] | None) -> list[ToolCall]:
    """:func:`parse_calls` before the depth of the arguments is checked."""
    # Unstripped: the recovered delimiters carry their own whitespace.
    text = body
    if lead_in and text.startswith(lead_in):
        text = text[len(lead_in):]
    if json_keys is not None:
        # Whole first: a model whose call *is* JSON would have the lead-in cut out
        # of that JSON. One that opens with a tag then writes JSON needs the second.
        candidates = [body.strip()] + ([text.strip()] if text is not body else [])
        for candidate in candidates:
            calls = _as_json_calls(candidate, json_keys)
            if calls:
                return calls
        raise ValueError(f"not a call as this model writes one in JSON: {body!r}")
    if call_body is None:
        raise ValueError(f"no call_body to read this call with: {body!r}")
    return [_as_delimited_call(text, call_body)]


def _nesting_depth(value) -> int:
    """How many levels of lists and objects ``value`` nests, counted without recursion; past :data:`MAX_ARGUMENT_DEPTH` it stops counting."""
    deepest, pending = 0, [(value, 0)]
    while pending and deepest <= MAX_ARGUMENT_DEPTH:
        item, depth = pending.pop()
        if isinstance(item, (dict, list)):
            deepest = max(deepest, depth + 1)
            pending += [(child, depth + 1) for child in
                        (item.values() if isinstance(item, dict) else item)]
    return deepest


def _as_json_calls(text: str, keys: tuple[str, ...]) -> list[ToolCall]:
    """Read the calls of a JSON body by ``keys``, or nothing where any of them is not a call."""
    try:
        decoded = _SPELLED.decode(text)
    except ValueError:
        return []
    batch = decoded if isinstance(decoded, list) else [decoded]
    calls = [_one_json_call(item, keys) for item in batch]
    return calls if calls and None not in calls else []


def _one_json_call(decoded, keys: tuple[str, ...]) -> ToolCall | None:
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
    return ToolCall(name=name, arguments=arguments)


def _as_delimited_call(text: str, call_body: CallBody) -> ToolCall:
    """Read a call written with this model's own delimiters."""
    if call_body.name_close:
        name, found, rest = text.partition(call_body.name_close)
    else:
        # The name runs up to the first key's tag, or to the end.
        name, found, rest = text.partition(call_body.key_open)
        rest, found = found + rest, True
    name = name.strip()
    if not found or not name:
        raise ValueError(f"not a call: {text!r}")
    whole = _as_json_arguments(rest, call_body) if call_body.arguments_json else None
    if whole is not None:
        return ToolCall(name=name, arguments=whole)
    if call_body.key_in_close:
        return ToolCall(name=name, arguments=_parse_keyed_pairs(rest, call_body))
    # Compared whole, not by its ending: a model that wraps each argument closes
    # the last one with the very same text.
    if rest == call_body.empty_close:
        return ToolCall(name=name, arguments={})
    # A model may write whitespace after the last argument's closing tag.
    closing = call_body.body_close.rstrip()
    if closing and rest.rstrip().endswith(closing):
        rest = rest.rstrip()[:len(rest.rstrip()) - len(closing)]
    if call_body.key_open and rest.startswith(call_body.key_open):
        rest = rest[len(call_body.key_open):]
    return ToolCall(name=name, arguments=_parse_pairs(rest, call_body))


def _parse_keyed_pairs(text: str, call_body: CallBody) -> dict:
    """Read pairs whose values each close with a tag naming their own key."""
    parsed, at = {}, len(call_body.key_open) if text.startswith(call_body.key_open) else 0
    while at < len(text):
        key_end = text.find(call_body.key_close, at)
        if key_end < 0:
            break
        key, at = text[at:key_end], key_end + len(call_body.key_close)
        separator = call_body.pair_sep.replace("{key}", key)
        if call_body.member_open:
            # A nested member may carry this key too, so its closing is matched by depth.
            end = _matching_close(text, at, call_body.member_open.replace("{key}", key),
                                  call_body.member_close.replace("{key}", key))
            following = end if end >= 0 and text.startswith(separator, end) else -1
        else:
            closing = call_body.body_close.replace("{key}", key).rstrip()
            following = text.find(separator, at)
            end = following if following >= 0 else text.find(closing, at)
        if end < 0:
            raise ValueError(f"an argument that never closes: {text[at:]!r}")
        parsed[_unquote(key.strip(), call_body)] = _parse_typed_value(text[at:end], call_body)
        if following < 0:
            break
        at = following + len(separator)
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
    parsed = {}
    for chunk in _split_on(text, call_body.pair_sep, call_body):
        key, found, value = chunk.partition(call_body.key_close)
        if not found:
            raise ValueError(f"an argument with no value: {chunk!r}")
        parsed[_unquote(key.strip(), call_body)] = _parse_typed_value(value, call_body)
    return parsed


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
        # Nested values are JSON in some models, the Python literal a template
        # interpolates in others, and these delimiters in the rest.
        try:
            return json.loads(text)
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
            return text             # a string a template wrote as it stands
    return _Unmarked(text)


class _Unmarked(str):
    """A value a call format wrote without marking its type, read by :func:`_typed`."""


# Reads JSON with each number kept as the text the model wrote, read by its tool's
# schema afterwards: a number where text is asked is that text, as written.
_SPELLED = json.JSONDecoder(parse_int=_Unmarked, parse_float=_Unmarked)


def _typed(value, schema, root) -> Any:
    """``value`` with each :class:`_Unmarked` text in it read against the part of ``schema`` it fills, ``root`` holding what a ``$ref`` names."""
    branches = _branches(schema, root, frozenset())
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


def _branches(schema, root, seen: frozenset[int]) -> list[dict]:
    """``schema`` and every schema its ``anyOf``, ``oneOf`` and ``$ref`` into ``root`` lead to, each once along a path."""
    if not isinstance(schema, dict) or id(schema) in seen:
        return []
    seen = seen | {id(schema)}
    found = [schema]
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        target = root
        for name in ref[2:].split("/"):
            name = name.replace("~1", "/").replace("~0", "~")
            target = target.get(name) if isinstance(target, dict) else None
        found += _branches(target, root, seen)
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            found += [branch for item in schema[key] for branch in _branches(item, root, seen)]
    return found


def _properties(schema: dict) -> dict:
    properties = schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _read_unmarked(text: str, branches: list[dict]) -> Any:
    """``text`` as the literal it spells, unless one of ``branches`` asks for text and none for that literal."""
    allowed = set()
    for branch in branches:
        choices = branch.get("enum")
        if isinstance(choices, list) and text in choices:
            return text
        kind = branch.get("type")
        allowed |= {kind} if isinstance(kind, str) else set(kind) if isinstance(kind, list) else set()
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
            return number(text)
        except ValueError:
            pass
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
    while at < len(text):
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
    while True:
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
    while at < len(text):
        if opener and not quoted and text.startswith(opener, at):
            quoted, at = True, at + len(opener)
            continue
        if quoted:
            if closer and text.startswith(closer, at):
                quoted, at = False, at + len(closer)
                continue
            at += 1
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
    return [part for part in parts if part.strip()]
