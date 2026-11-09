"""Declaring tools to a model and running the ones it calls.

Reading a call back is the other half and lives in
:mod:`llama_chat.call_parsing`: an engine reached over a wire emits calls it
cannot run, so the two are separate. See the README, "Tool calling".
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["ToolCall", "ToolRegistry", "ToolResult", "ToolSpec", "ToolTurn", "result_text"]

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


def result_text(result) -> str:
    """One tool's return value as the text the model is handed.

    Raises:
        ValueError: If it is neither text nor anything JSON can write.
    """
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except (TypeError, ValueError) as e:
        raise ValueError(f"a tool result must be text or what JSON can write: {e}") from e


@dataclass(frozen=True)
class ToolResult:
    """What one call returned, as the conversation already holds it.

    Attributes:
        name: The function that was called.
        result: What it returned: text as it is, anything else as JSON.
    """

    name: str
    result: object


@dataclass(frozen=True)
class ToolTurn:
    """One assistant turn of a history that called tools, for ``begin``.

    Attributes:
        steps: The turn in the order it was written: the model's text as a
            ``str``, each call as a :class:`ToolCall`, and each result as a
            :class:`ToolResult`, each result after the call it answers.

    Raises:
        ValueError: If a result answers no call made before it in the turn.
    """

    steps: tuple

    def __post_init__(self) -> None:
        owed = 0
        for step in self.steps:
            if isinstance(step, ToolCall):
                owed += 1
            elif isinstance(step, ToolResult):
                if not owed:
                    raise ValueError(f"the result for {step.name!r} answers no call made "
                                     "before it in its ToolTurn")
                owed -= 1

    @property
    def text(self) -> str:
        """What the model wrote in this turn, its calls and results left out."""
        return "".join(step for step in self.steps if isinstance(step, str))


@dataclass
class _RegisteredTool:
    """One registered tool, and when the model is to be told about it.

    ``declare_at_begin`` is an intent, not a record of what the model has been
    told.
    """

    spec: ToolSpec
    function: Callable
    declare_at_begin: bool


class ToolRegistry:
    """The tools this process can run, by the name the model calls them.

    Registering makes a tool runnable at once and says when the model is to be
    told about it: by ``begin``, or later by ``declare_tools``. Nothing
    here writes to a model, so nothing here learns what the conversation still
    holds. See the README, "Tool calling".
    """

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredTool] = {}

    def register(self, function: Callable, description: str, *,
                 name: str | None = None, parameters: dict | None = None,
                 declare_at_begin: bool = True) -> str:
        """Make ``function`` callable by the model, and return the name it uses.

        A name already registered is replaced, when it is declared included.

        Args:
            function: What runs. Called by keyword with the model's arguments.
            description: What it does, for the model to read.
            name: The name the model calls, where that is not the function's own.
            parameters: An explicit JSON Schema object, for a signature whose
                annotations cannot be described. Derived from them when omitted.
            declare_at_begin: When the model is told about it. ``True``
                declares it as :meth:`~llama_chat.wrapper.ChatWrapper.begin`
                writes the conversation; ``False`` leaves it for
                :meth:`~llama_chat.wrapper.ChatWrapper.declare_tools`.
                Either way the declaration goes where the template writes
                declarations. It is runnable either way, from the moment it is
                registered.

        Returns:
            The name the model calls it by, which is also how :meth:`call`
            refers to it.

        Raises:
            TypeError: If the signature cannot be described and no
                ``parameters`` were given, or if ``name`` is omitted and the
                function has none a model could call it by.
        """
        spec = ToolSpec(name=name or _callable_name(function), description=description,
                        parameters=(parameters if parameters is not None
                                    else _parameters_of(function)))
        self._entries[spec.name] = _RegisteredTool(spec, function, declare_at_begin)
        return spec.name

    def tool(self, description: str, *, name: str | None = None,
             parameters: dict | None = None,
             declare_at_begin: bool = True) -> Callable:
        """Register the function this decorates, and return it unchanged.

        :meth:`register` in the form the function is written in.

        Args:
            description, name, parameters, declare_at_begin: As
                :meth:`register`.

        Raises:
            TypeError: As :meth:`register`.
        """
        def attach(function: Callable) -> Callable:
            self.register(function, description, name=name,
                          parameters=parameters,
                          declare_at_begin=declare_at_begin)
            return function
        return attach

    def specs_declared_at_begin(self) -> list[ToolSpec]:
        """The declarations :meth:`~llama_chat.wrapper.ChatWrapper.begin` writes when it is handed this registry."""
        return [e.spec for e in self._entries.values() if e.declare_at_begin]

    def specs(self) -> list[ToolSpec]:
        """Every declaration, wherever it belongs.

        What ought to be declared, not what the model currently knows.
        """
        return [e.spec for e in self._entries.values()]

    def call(self, call: ToolCall) -> Any:
        """Run what ``call`` asked for, turning every failure into a result.

        Nothing raises: a failure belongs back in the conversation. The model's
        mistake and the tool's own are told apart before the call, so a
        ``TypeError`` raised inside a tool is not reported as a bad call.

        Returns:
            Whatever the tool returned, where it is text or what JSON can write,
            or a dict carrying ``error`` and ``message``.
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
            returned = entry.function(**call.arguments)
        except Exception as e:
            return {"error": "tool_raised", "message": f"{type(e).__name__}: {e}"}
        try:
            result_text(returned)
        except ValueError as e:
            return {"error": "unwritable_result", "message": str(e)}
        return returned

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)



def _callable_name(function) -> str:
    """The name a model would call ``function`` by, taken from the function.

    Raises:
        TypeError: If it carries none a model could name, as a lambda and a
            partial do not.
    """
    name = getattr(function, "__name__", "")
    if not name.isidentifier():
        raise TypeError(
            f"{function!r} carries no name the model could call it by; "
            f"pass name= to register it under one")
    return name


def _parameters_of(fn) -> dict:
    """A JSON Schema object for ``fn``'s keyword arguments.

    Raises:
        TypeError: If the signature cannot be read or cannot be described.
    """
    try:
        # Annotations written as strings, as under postponed evaluation, are
        # read as the types they name.
        signature = inspect.signature(fn, eval_str=True)
    except (ValueError, NameError) as e:
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
