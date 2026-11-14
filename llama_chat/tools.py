"""What a tool is declared as, and what a model's call to one holds.

Running calls is :mod:`llama_chat.tool_registry`'s, and reading a call back
from what the model wrote is :mod:`llama_chat.call_parsing`'s. See
design_history.md, "Reasons moved out of the code".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from .errors import InvalidArgumentError

__all__ = ["ToolCall", "ToolCallStatus", "ToolSpec"]

# Whether a call the model wrote was read: whole, not readable, or cut short by the
# reply cap or a cancel.
ToolCallStatus = Literal["complete", "unreadable", "unfinished"]


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

    It may be kept in a set or used as a dict key, and equals another call
    only where every field does.

    Attributes:
        name: The function named. Nothing here checks that it exists.
        arguments: What it was called with, by keyword.
        id: What tells it apart from every other call of its conversation, as
            the conversation's history holds it; empty for a call not written
            into one yet, such as one built for a history, which ``begin``
            gives an id.
        status: ``"complete"`` for a call that was read. ``"unreadable"``
            for one that was written but could not be read, and
            ``"unfinished"`` for one the reply cap or a cancel cut short:
            such a call is only reported, with no name, arguments or id, and
            nothing answers it.
        text: What the model wrote for a call that could not be read,
            between the call's opening and its closing; empty for any other.
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
            raise TypeError(f"a declaration holds ToolSpecs or is a ToolRegistry, not {spec!r}")
        if not isinstance(spec.name, str) or not isinstance(spec.description, str) \
                or not isinstance(spec.parameters, dict):
            raise TypeError(f"a ToolSpec holds a str name, a str description and a dict of "
                            f"parameters, not {spec!r}")
        if not spec.name:
            raise InvalidArgumentError("tools", "a declared tool needs a name to be called by")
        if spec.name in seen:
            raise InvalidArgumentError("tools", f"the declaration names the tool {spec.name!r} "
                                                "twice; declare each name once")
        seen.add(spec.name)


def result_text(result) -> str:
    """One tool's return value as the text the model is handed.

    Raises:
        ValueError: If it is neither text nor anything JSON can write, ``NaN``
            and infinity included.
    """
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise InvalidArgumentError("result", f"a tool result must be text or what JSON can "
                                             f"write: {e}") from e


def unwritable_failure(result) -> dict | None:
    """The failure that answers a call in place of ``result`` where :func:`result_text` refuses it; ``None`` where it does not."""
    try:
        result_text(result)
    except ValueError as e:
        return {"error": "unwritable_result", "message": str(e)}
    return None
