"""What a tool is declared as, and what a model's call to one holds.

Running calls is :mod:`llama_chat.tool_registry`'s, and reading a call back
from what the model wrote is :mod:`llama_chat.call_parsing`'s. See
docs/decisions/0038-a-call-is-read-only-where-all-its-text-is-read.md.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Callable, Literal

from .errors import ArgumentTypeError, InvalidArgumentError

__all__ = ["ToolCall", "ToolCallStatus", "ToolSpec"]

# How many levels of lists and objects a call's arguments may nest, the arguments'
# own object counted. Well inside the interpreter's recursion limit, so that every
# later step that walks the arguments, copying and rendering them, completes.
MAX_ARGUMENT_DEPTH = 100

# A half of a UTF-16 surrogate pair standing alone, which no UTF-8 text holds.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def lone_surrogate_at(text: str) -> int | None:
    """Where ``text`` holds a lone surrogate, which no model can be given; ``None`` where it holds none."""
    if text.isascii():
        return None
    found = _LONE_SURROGATE.search(text)
    return None if found is None else found.start()


def ensure_readable(text: str, field: str, where: str) -> None:
    """Refuse a text no model can be given, one holding a lone surrogate.

    Raises:
        InvalidArgumentError: Naming ``field``, and ``where`` the text is.
    """
    at = lone_surrogate_at(text)
    if at is not None:
        raise InvalidArgumentError(
            field, f"{where} holds a lone surrogate (U+{ord(text[at]):04X}) at character {at}, "
            f"which is no character and no model can read; give it as valid Unicode text")


def nesting_depth(value) -> int:
    """How many levels of lists and objects ``value`` nests, counted without recursion; past :data:`MAX_ARGUMENT_DEPTH` it stops counting."""
    deepest, pending = 0, [(value, 0)]
    while pending and deepest <= MAX_ARGUMENT_DEPTH:
        item, depth = pending.pop()
        if isinstance(item, (dict, list)):
            deepest = max(deepest, depth + 1)
            pending += [(child, depth + 1) for child in
                        (item.values() if isinstance(item, dict) else item)]
    return deepest


def json_value_fault(value) -> str | None:
    """What keeps ``value`` from being a JSON value every model can be given, ``None`` where nothing does.

    Objects with text keys, lists, text, finite numbers, truth values and
    ``None``, nested at most :data:`MAX_ARGUMENT_DEPTH` levels, no text
    holding a lone surrogate.
    """
    if nesting_depth(value) > MAX_ARGUMENT_DEPTH:
        return f"it nests deeper than {MAX_ARGUMENT_DEPTH} levels"
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for key in item:
                if not isinstance(key, str):
                    return f"it holds the key {key!r}, which is no text"
            pending += list(item) + list(item.values())
        elif isinstance(item, list):
            pending += item
        elif isinstance(item, str):
            if lone_surrogate_at(item) is not None:
                return f"the text {item!r} holds a lone surrogate"
        elif isinstance(item, float) and not math.isfinite(item):
            return f"it holds {item!r}, which JSON cannot write"
        elif item is not None and not isinstance(item, (bool, int, float)):
            return f"it holds {item!r}, which is no JSON value"
    return None


ToolCallStatus = Literal["complete", "unreadable", "unfinished", "result_dropped"]
"""Whether a call the model wrote was read: whole, not readable, cut short by the reply cap or a cancel, or run with its result left out for want of room."""


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

    def __post_init__(self) -> None:
        """Refuse a spec no template could declare.

        Raises:
            ArgumentTypeError: If a field is of another type than it states.
            InvalidArgumentError: If ``parameters`` is no JSON value every
                model can be given, or a text holds a lone surrogate.
        """
        if not isinstance(self.name, str) or not isinstance(self.description, str) \
                or not isinstance(self.parameters, dict):
            raise ArgumentTypeError(f"a ToolSpec holds a str name, a str description and a dict "
                                    f"of parameters, not {self!r}")
        ensure_readable(self.name, "name", "the tool's name")
        ensure_readable(self.description, "description", f"the description of {self.name!r}")
        fault = json_value_fault(self.parameters)
        if fault is not None:
            raise InvalidArgumentError("parameters", f"the parameters of {self.name!r} are no "
                                                     f"JSON Schema a model can be given: {fault}")

    def as_declaration(self) -> dict:
        """This spec in the shape a chat template expects to be handed."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


@dataclass(frozen=True)
class ToolCall:
    """One call a model asked for.

    It may be kept in a set or used as a dict key, and equals another call
    only where every field does.

    Attributes:
        name: The function named. Nothing here checks that it exists.
        arguments: What it was called with, by keyword.
        id: What tells it apart from every other call of its conversation, as
            the conversation's history holds it: the id the model wrote for
            it where its format holds one, else one the library gives. Empty
            only for a call that has none: an unfinished one, or one that
            could not be read where the reply stopped on it.
        status: ``"complete"`` for a call that was read. ``"unreadable"``
            for one that was written but could not be read, which the library
            answers itself, and ``"unfinished"`` for one the reply cap or a
            cancel cut short, or one whose plain opening ended a reply that
            withholds calls, which nothing answers: either is only reported,
            with no name or arguments, and an unfinished one with no id. A
            call read whole whose turn closed for want of room before its
            result was written is ``"result_dropped"`` where its tool ran, so
            the model never read what it returned, and ``"unfinished"`` where
            it was not run.
        text: What the model wrote for a call that could not be read, where
            the reply stopped on it, between the call's opening and its
            closing; empty for any other.
    """

    name: str
    arguments: dict
    id: str = ""
    status: ToolCallStatus = "complete"
    text: str = ""

    def __hash__(self) -> int:
        """By name and id, which two equal calls share."""
        return hash((self.name, self.id))


def ensure_declarable(specs: list[ToolSpec]) -> None:
    """Refuse a declaration of anything but tool specs, or one that names a tool twice.

    Raises:
        TypeError: If an entry is no :class:`ToolSpec`, or one of its fields
            is of another type than it states.
        InvalidArgumentError: Naming the first tool named twice, or a tool
            without a name.
    """
    seen = set()
    for spec in specs:
        if not isinstance(spec, ToolSpec):
            raise ArgumentTypeError(f"a declaration holds ToolSpecs, such as ToolRegistry.specs() "
                            f"gives, not {spec!r}")
        if not isinstance(spec.name, str) or not isinstance(spec.description, str) \
                or not isinstance(spec.parameters, dict):
            raise ArgumentTypeError(f"a ToolSpec holds a str name, a str description and a dict of "
                            f"parameters, not {spec!r}")
        if not spec.name:
            raise InvalidArgumentError("tools", "a declared tool needs a name to be called by")
        if spec.name in seen:
            raise InvalidArgumentError("tools", f"the declaration names the tool {spec.name!r} "
                                                "twice; declare each name once")
        seen.add(spec.name)


def ensure_no_tags(specs: list[ToolSpec], holds_tag: Callable[[str], bool]) -> None:
    """Refuse a declaration that holds a tag where a template reads it as written: anywhere but in a description, which is written as text.

    Raises:
        InvalidArgumentError: Naming ``tools``, the tool and the place, where
            ``holds_tag`` holds for a name, a key or a value of a declaration.
    """
    def fault(value, path: str) -> str | None:
        if isinstance(value, str):
            return path if holds_tag(value) else None
        if isinstance(value, dict):
            for key, item in value.items():
                found = (path + f" key {key!r}" if isinstance(key, str) and holds_tag(key)
                         else None if key == "description" and isinstance(item, str)
                         else fault(item, f"{path}[{key!r}]"))
                if found:
                    return found
        if isinstance(value, list):
            return next((found for i, item in enumerate(value)
                         if (found := fault(item, f"{path}[{i}]"))), None)
        return None

    for spec in specs:
        found = fault(spec.name, "its name") or fault(spec.parameters, "parameters")
        if found:
            raise InvalidArgumentError(
                "tools", f"the tool {spec.name!r} holds one of the model's tags in {found}, "
                "which its template writes as the tag itself; a tag may stand only in a "
                "description")


def result_text(result) -> str:
    """One tool's return value as the text the model is handed.

    Raises:
        ValueError: If it is neither text nor anything JSON can write, ``NaN``
            and infinity included.
    """
    if isinstance(result, str):
        ensure_readable(result, "result", "the tool result")
        return result
    try:
        return json.dumps(result, allow_nan=False)
    except Exception as e:          # whatever the value's own methods raise
        raise InvalidArgumentError("result", f"a tool result must be text or what JSON can "
                                             f"write: {type(e).__name__}: {e}") from e


def unwritable_failure(result) -> dict | None:
    """The failure that answers a call in place of ``result`` where :func:`result_text` refuses it; ``None`` where it does not."""
    try:
        result_text(result)
    except ValueError as e:
        return {"error": "unwritable_result", "message": str(e)}
    return None
