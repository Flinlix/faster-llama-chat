"""Declaring tools to a model, reading back its calls, and running them.

Reading a call and running one are separate: an engine reached over a wire
emits calls it cannot run. See the README, "Tool calling".
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

from .template import CallBody

__all__ = ["ToolCall", "ToolRegistry", "ToolSpec", "tool"]

# Anything not here has to be spelled out by passing `parameters` explicitly.
_JSON_TYPES = {str: "string", int: "integer", float: "number",
               bool: "boolean", list: "array", dict: "object"}


@dataclass(frozen=True)
class ToolSpec:
    """One function, as the model is told about it.

    Attributes:
        name: What the model calls to reach it.
        description: What it does, for the model to read.
        parameters: A JSON Schema object describing the arguments.
    """

    name: str
    description: str
    parameters: dict

    def as_declaration(self) -> dict:
        """This spec in the shape a chat template expects to be handed."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


@dataclass(frozen=True)
class ToolCall:
    """One call a model asked for.

    Attributes:
        name: The function named. Nothing here checks that it exists.
        arguments: What it was called with, by keyword.
    """

    name: str
    arguments: dict


@dataclass
class _RegisteredTool:
    """One registered tool. ``should_declare`` is an intent, not a record."""

    spec: ToolSpec
    function: Callable
    should_declare: bool


class ToolRegistry:
    """The tools this process can run, by the name the model calls them.

    Registering makes a tool runnable at once; telling the model about it is
    separate, because that costs a prefill. Nothing here writes to a model, so
    nothing here learns when a written declaration is evicted again. See the
    README, "Registering and declaring are two things".
    """

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredTool] = {}

    def register(self, function: Callable, description: str | None = None, *,
                 name: str | None = None, parameters: dict | None = None,
                 declare: bool = False) -> str:
        """Make ``function`` callable by the model, and return the name it uses.

        A name already registered is replaced, ``declare`` state included.

        Args:
            function: What runs. Called by keyword with the model's arguments.
            description: What it does, for the model to read. Taken from a
                :func:`tool` declaration on ``function`` when omitted.
            name: The name the model calls, where that is not the function's own.
            parameters: An explicit JSON Schema object, for a signature
                :func:`tool` cannot describe.
            declare: Whether the model should be told about it. ``False`` leaves
                it registered and silent, callable the moment something names it.

        Returns:
            The name the model calls it by, which is also how :meth:`declare`
            and :meth:`call` refer to it.

        Raises:
            TypeError: If neither a description nor a :func:`tool` declaration
                is given, or the signature cannot be described and no
                ``parameters`` were given.
        """
        declaration = getattr(function, "spec", None)
        if description is None and declaration is None:
            raise TypeError(
                f"{getattr(function, '__name__', function)!r} carries no tool() "
                f"declaration, so register() needs a description")
        spec = ToolSpec(
            name=name or function.__name__,
            description=(description if description is not None
                         else declaration.description),
            parameters=(parameters if parameters is not None
                        else declaration.parameters if declaration is not None
                        else _parameters_of(function)))
        self._entries[spec.name] = _RegisteredTool(spec, function, declare)
        return spec.name

    def declare(self, name: str) -> ToolSpec:
        """Record that the model should be told about ``name``, and say what to tell.

        Raises:
            KeyError: If nothing is registered under that name.
        """
        entry = self._entries[name]
        entry.should_declare = True
        return entry.spec

    def specs_to_declare(self) -> list[ToolSpec]:
        """The declarations of every tool the model should be told about.

        What ought to be declared, not what the model currently knows.
        """
        return [e.spec for e in self._entries.values() if e.should_declare]

    def call(self, call: ToolCall) -> Any:
        """Run what ``call`` asked for, turning every failure into a result.

        Nothing raises: a failure belongs back in the conversation. The model's
        mistake and the tool's own are told apart before the call, so a
        ``TypeError`` raised inside a tool is not reported as a bad call.

        Returns:
            Whatever the tool returned, or a dict carrying ``error`` and
            ``message``.
        """
        entry = self._entries.get(call.name)
        if entry is None:
            return {"error": "unknown_tool",
                    "message": f"no tool named {call.name!r} is registered"}
        try:
            inspect.signature(entry.function).bind(**call.arguments)
        except TypeError as e:
            return {"error": "wrong_arguments", "message": str(e)}
        except ValueError:
            pass        # some builtins have no signature to check against
        try:
            return entry.function(**call.arguments)
        except Exception as e:
            return {"error": "tool_raised", "message": f"{type(e).__name__}: {e}"}

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def tool(description: str, parameters: dict | None = None) -> Callable:
    """Declare a function as a tool, deriving its parameters from its signature.

    Attached as ``fn.spec``; the function is returned unchanged.

    Args:
        description: What the tool does, for the model to read.
        parameters: An explicit JSON Schema object, for a signature this cannot
            describe. Derived from the annotations when omitted.

    Raises:
        TypeError: If a parameter has no annotation this can turn into a schema
            and none was supplied.
    """
    def attach(fn):
        fn.spec = ToolSpec(name=fn.__name__, description=description,
                           parameters=parameters if parameters is not None
                           else _parameters_of(fn))
        return fn
    return attach


def _parameters_of(fn) -> dict:
    """A JSON Schema object for ``fn``'s keyword arguments.

    Raises:
        TypeError: If the signature cannot be read or cannot be described.
    """
    try:
        signature = inspect.signature(fn)
    except ValueError as e:
        raise TypeError(f"{fn!r} has no signature this can read ({e}); "
                        f"pass parameters=") from e
    properties, required = {}, []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        annotation = parameter.annotation
        if annotation not in _JSON_TYPES:
            raise TypeError(
                f"{fn.__name__}({name}): {annotation!r} is not a type this can "
                f"describe; annotate it as one of "
                f"{sorted(t.__name__ for t in _JSON_TYPES)} or pass parameters=")
        properties[name] = {"type": _JSON_TYPES[annotation]}
        if parameter.default is parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


# ----- reading a call back ------------------------------------------------
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
        ValueError: If ``body`` is not a call this can read.
    """
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
    """Read one entry of a batch, or ``None`` if it is not a call."""
    if not isinstance(decoded, dict):
        return None
    name = decoded.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = decoded.get("arguments", decoded.get("parameters", {}))  # both are in the wild
    if isinstance(arguments, str):          # some models nest it as JSON text
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    return ToolCall(name=name,
                    arguments=arguments if isinstance(arguments, dict) else {})


def _as_delimited_call(text: str, call_body: CallBody) -> ToolCall:
    """Read a call written with this model's own delimiters."""
    name, found, rest = text.partition(call_body.name_close)
    name = name.strip()
    if not found or not name:
        raise ValueError(f"not a call: {text!r}")
    # Compared whole, not by its ending: a model that wraps each argument closes
    # the last one with the very same text.
    if rest == call_body.empty_close:
        return ToolCall(name=name, arguments={})
    if call_body.body_close and rest.endswith(call_body.body_close):
        rest = rest[:len(rest) - len(call_body.body_close)]
    if call_body.key_open and rest.startswith(call_body.key_open):
        rest = rest[len(call_body.key_open):]
    return ToolCall(name=name, arguments=_parse_pairs(rest, call_body))


def _parse_pairs(text: str, call_body: CallBody) -> dict:
    """Read ``key`` and value pairs separated by this model's own delimiter."""
    parsed = {}
    for chunk in _split_on(text, call_body.pair_sep, call_body):
        key, found, value = chunk.partition(call_body.key_close)
        if not found:
            raise ValueError(f"an argument with no value: {chunk!r}")
        parsed[_unquote(key.strip(), call_body)] = _parse_value(value, call_body)
    return parsed


def _parse_value(text: str, call_body: CallBody) -> Any:
    """Read one argument value, keeping its type where the model marked one."""
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
        # Nested values are JSON in some models and these delimiters in others.
        try:
            return json.loads(text)
        except ValueError:
            pass
        if text[0] == "{":
            return _parse_pairs(text[1:-1], call_body)
        return [_parse_value(item, call_body)
                for item in _split_on(text[1:-1], call_body.pair_sep, call_body)]
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
