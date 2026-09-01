"""Declaring tools to a model and running the ones it calls.

Reading a call back is the other half and lives in
:mod:`llama_chat.call_parsing`: an engine reached over a wire emits calls it
cannot run, so the two are separate. See the README, "Tool calling".
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

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
    """One registered tool, and where its declaration belongs.

    ``in_system_prompt`` is an intent about placement, not a record of what the
    model has been told.
    """

    spec: ToolSpec
    function: Callable
    in_system_prompt: bool


class ToolRegistry:
    """The tools this process can run, by the name the model calls them.

    Registering makes a tool runnable at once and says where the model is to be
    told about it: in the system turn, or in a message of its own later. Nothing
    here writes to a model, so nothing here learns what the conversation still
    holds. See the README, "Registering, and where the declaration lands".
    """

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredTool] = {}

    def register(self, function: Callable, description: str | None = None, *,
                 name: str | None = None, parameters: dict | None = None,
                 in_system_prompt: bool = True) -> str:
        """Make ``function`` callable by the model, and return the name it uses.

        A name already registered is replaced, its placement included.

        Args:
            function: What runs. Called by keyword with the model's arguments.
            description: What it does, for the model to read. Taken from a
                :func:`tool` declaration on ``function`` when omitted.
            name: The name the model calls, where that is not the function's own.
            parameters: An explicit JSON Schema object, for a signature
                :func:`tool` cannot describe.
            in_system_prompt: Where the model is told about it. ``True`` puts
                the declaration in the system turn, which
                :meth:`~llama_chat.wrapper.ChatWrapper.begin` writes at no extra
                prefill and no eviction can take back. ``False`` leaves it for
                :meth:`~llama_chat.wrapper.ChatWrapper.declare_unknown_tools`, a message
                of its own that costs a prefill and can be evicted later. It is
                runnable either way, from the moment it is registered.

        Returns:
            The name the model calls it by, which is also how :meth:`spec`
            and :meth:`call` refer to it.

        Raises:
            TypeError: If neither a description nor a :func:`tool` declaration
                is given, if the signature cannot be described and no
                ``parameters`` were given, or if ``name`` is omitted and the
                function has none a model could call it by.
        """
        declaration = getattr(function, "spec", None)
        if description is None and declaration is None:
            raise TypeError(
                f"{getattr(function, '__name__', function)!r} carries no tool() "
                f"declaration, so register() needs a description")
        spec = ToolSpec(
            name=name or _callable_name(function),
            description=(description if description is not None
                         else declaration.description),
            parameters=(parameters if parameters is not None
                        else declaration.parameters if declaration is not None
                        else _parameters_of(function)))
        self._entries[spec.name] = _RegisteredTool(spec, function, in_system_prompt)
        return spec.name

    def tool(self, description: str, *, name: str | None = None,
             parameters: dict | None = None,
             in_system_prompt: bool = True) -> Callable:
        """Register the function this decorates, and return it unchanged.

        :meth:`register` in the form the function is written in.

        Args:
            description, name, parameters, in_system_prompt: As
                :meth:`register`.

        Raises:
            TypeError: As :meth:`register`.
        """
        def attach(function: Callable) -> Callable:
            self.register(function, description, name=name,
                          parameters=parameters,
                          in_system_prompt=in_system_prompt)
            return function
        return attach

    def spec(self, name: str) -> ToolSpec:
        """What the model would be told about ``name``.

        For declaring one tool by hand rather than the whole registry.

        Raises:
            KeyError: If nothing is registered under that name.
        """
        return self._entries[name].spec

    def specs_for_system_prompt(self) -> list[ToolSpec]:
        """The declarations that belong in the system turn.

        What :meth:`~llama_chat.wrapper.ChatWrapper.begin` writes when it is
        handed this registry.
        """
        return [e.spec for e in self._entries.values() if e.in_system_prompt]

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
