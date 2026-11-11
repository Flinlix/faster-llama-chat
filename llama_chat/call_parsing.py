"""Reading a model's tool calls back out of the text it generated.

Two shapes cover the field: a body that is JSON names itself, and anything else
is read with the delimiters recovered from that model's own template. Nothing
here runs anything; see :mod:`llama_chat.tools` for that half.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from .template import CallBody
from .tools import ToolCall

# How many levels of lists and objects a call's arguments may nest, the arguments'
# own object counted. Well inside the interpreter's recursion limit, so that every
# later step that walks the arguments, copying and rendering them, completes.
MAX_ARGUMENT_DEPTH = 100


def parse_calls(body: str, call_body: CallBody | None,
                lead_in: str = "") -> list[ToolCall]:
    """Read the calls out of the text a model generated between its call tags.

    A body that is JSON names itself and needs no ``call_body``; anything else
    is read with the delimiters recovered from this model's own template.

    Args:
        body: What the model generated, tags excluded.
        call_body: This model's argument spelling, or ``None`` where only the
            JSON shape can be read.
        lead_in: Fixed text the model writes before the function name.

    Returns:
        One :class:`ToolCall` per call, in the order the model wrote them.
        Usually one; a JSON array holds a whole batch.

    Raises:
        ValueError: If ``body`` is not a call this can read, among them a call
            whose arguments are no object or nest deeper than
            :data:`MAX_ARGUMENT_DEPTH`.
    """
    try:
        calls = _read_calls(body, call_body, lead_in)
    except RecursionError as e:
        raise ValueError("the call's arguments nest deeper than can be read") from e
    for call in calls:
        if _nesting_depth(call.arguments) > MAX_ARGUMENT_DEPTH:
            raise ValueError(f"the call's arguments nest deeper than {MAX_ARGUMENT_DEPTH} levels")
    return calls


def _read_calls(body: str, call_body: CallBody | None, lead_in: str) -> list[ToolCall]:
    """:func:`parse_calls` before the depth of the arguments is checked."""
    # Unstripped: the recovered delimiters carry their own whitespace.
    text = body
    if lead_in and text.startswith(lead_in):
        text = text[len(lead_in):]
    # Whole first: a model whose call *is* JSON would have the lead-in cut out
    # of that JSON. One that opens with a tag then writes JSON needs the second.
    candidates = [body.strip()]
    if text is not body:
        candidates.append(text.strip())
    for candidate in candidates:
        calls = _as_json_calls(candidate)
        if calls:
            return calls
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


def _as_json_calls(text: str) -> list[ToolCall]:
    """Read the calls that name themselves, or nothing if these are not calls."""
    try:
        decoded = json.loads(text)
    except ValueError:
        return []
    batch = decoded if isinstance(decoded, list) else [decoded]
    calls = [_one_json_call(item) for item in batch]
    # All or nothing: half a batch would silently drop what the model asked for.
    return calls if calls and None not in calls else []


def _one_json_call(decoded) -> ToolCall | None:
    """Read one entry of a batch, or ``None`` if it is not a call.

    Raises:
        ValueError: If it names a function but its arguments are no object.
    """
    if not isinstance(decoded, dict):
        return None
    name = decoded.get("name", decoded.get("tool_name"))      # both are in the wild
    if not isinstance(name, str) or not name:
        return None
    arguments = decoded.get("arguments", decoded.get("parameters", {}))  # both are in the wild
    if isinstance(arguments, str):          # some models nest it as JSON text
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
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
    whole = _as_json_object(rest)
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


def _as_json_object(text: str) -> dict | None:
    """``text`` as the JSON object it holds, or ``None`` where it holds none."""
    try:
        decoded = json.loads(text)
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


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
