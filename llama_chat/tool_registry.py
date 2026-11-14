"""Running the tools a model calls: plain Python functions registered by name, each declared from its signature."""

from __future__ import annotations

import copy
import enum
import inspect
import json
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .tools import ToolCall, ToolSpec, unwritable_failure

__all__ = ["ToolRegistry"]


@dataclass
class _RegisteredTool:
    """One registered tool: its declaration, the function that runs, and how each argument is read for it."""

    spec: ToolSpec
    function: Callable
    # By parameter name; empty where an explicit schema left the types unread.
    readers: dict[str, _Reader]


class ToolRegistry:
    """The tools this process can run, by the name the model calls them.

    Registering makes a tool runnable at once. :meth:`specs` gives the
    declarations that ``begin`` and ``change_settings`` tell the model. Nothing here writes to a model, so nothing here learns what
    the conversation still holds. See the README, "Tool calling".
    """

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredTool] = {}

    def tool(self, function_or_description: Callable | str | None = None, /, *,
             name: str | None = None, description: str | None = None,
             parameters: dict | None = None) -> Callable:
        """Make a function callable by the model, as a decorator or called on the function.

        ``@registry.tool`` describes the function by its docstring and its
        parameters by their annotations; ``@registry.tool("...")`` and
        ``description=`` give the description instead. ``registry.tool(function,
        ...)`` registers a function written elsewhere. A name already
        registered is replaced, its declaration included.

        Args:
            function_or_description: The function to register, or the
                description, where this decorates the function that follows.
            name: The name the model calls, where that is not the function's own.
            description: What it does, for the model to read; the function's
                docstring where omitted.
            parameters: An explicit JSON Schema object, for a signature whose
                annotations cannot be described. The model's arguments are then
                not checked against types before the call.

        Returns:
            The function unchanged, where it was given; else the decorator that
            registers the function it is applied to and returns it unchanged.

        Raises:
            TypeError: If the function is asynchronous, takes a parameter only
                by position, has no description, or has a parameter that cannot
                be described and no ``parameters`` were given; if ``name`` is
                omitted and the function has none a model could call it by; or
                if the description is given twice.
        """
        if callable(function_or_description):
            self._add(function_or_description, name, description, parameters)
            return function_or_description
        if function_or_description is not None:
            if not isinstance(function_or_description, str):
                raise TypeError(f"a tool is given a function or its description, not "
                                f"{function_or_description!r}")
            if description is not None:
                raise TypeError("the description is given twice; give it once")
            description = function_or_description

        def attach(function: Callable) -> Callable:
            self._add(function, name, description, parameters)
            return function
        return attach

    def _add(self, function: Callable, name: str | None, description: str | None,
             parameters: dict | None) -> None:
        """Register ``function``, as :meth:`tool` describes."""
        name = name or _callable_name(function)
        if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function) \
                or inspect.iscoroutinefunction(getattr(function, "__call__", None)):
            raise TypeError(f"the tool {name!r} is asynchronous, and a tool is called and its "
                            "result read at once; register a function that returns the result")
        if description is None:
            description = _docstring(function)
            if not description:
                raise TypeError(f"the tool {name!r} has no description; give the function a "
                                "docstring or pass description=")
        if parameters is None:
            parameters, readers = _parameters_of(function, name)
        else:
            parameters, readers = copy.deepcopy(parameters), {}
        self._entries[name] = _RegisteredTool(ToolSpec(name, description, parameters),
                                              function, readers)

    def specs(self) -> list[ToolSpec]:
        """Every registered tool's declaration, each a copy that is the caller's to change.

        What ought to be declared, not what the model currently knows.
        """
        return [replace(e.spec, parameters=copy.deepcopy(e.spec.parameters))
                for e in self._entries.values()]

    def call(self, call: ToolCall) -> Any:
        """Run what ``call`` asked for, turning every failure into a result.

        Nothing raises: a failure belongs back in the conversation. The model's
        mistake and the tool's own are told apart before the call: arguments
        that do not fit the signature or its types are the model's, so a
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
            arguments = _read_arguments(entry, call.arguments)
        except TypeError as e:
            return {"error": "wrong_arguments", "message": str(e)}
        try:
            returned = entry.function(**arguments)
        except Exception as e:
            return {"error": "tool_raised", "message": f"{type(e).__name__}: {e}"}
        failure = unwritable_failure(returned)
        return returned if failure is None else failure

    def __len__(self) -> int:
        return len(self._entries)


def _read_arguments(entry: _RegisteredTool, arguments) -> dict:
    """The model's arguments as the tool's function takes them.

    Raises:
        TypeError: Saying what does not fit the signature or a parameter's type.
    """
    if not isinstance(arguments, dict):
        raise TypeError(f"the arguments must be an object of named values, not {arguments!r}")
    try:
        inspect.signature(entry.function).bind(**arguments)
    except ValueError:
        pass        # some builtins have no signature to check against
    return {key: entry.readers[key](value, key) if key in entry.readers else value
            for key, value in arguments.items()}


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


def _docstring(function) -> str | None:
    """The docstring a function or method was written with; ``None`` for any other callable, whose docstring is its type's."""
    if inspect.isfunction(function) or inspect.ismethod(function):
        return inspect.getdoc(function)
    return None


def _parameters_of(function, name: str) -> tuple[dict, dict[str, _Reader]]:
    """A JSON Schema object for ``function``'s keyword arguments, and the reader of each.

    Raises:
        TypeError: If the signature cannot be read or cannot be described.
    """
    try:
        # Annotations written as strings, as under postponed evaluation, are
        # read as the types they name.
        signature = inspect.signature(function, eval_str=True)
    except (ValueError, NameError) as e:
        raise TypeError(f"the tool {name!r} has no signature this can read ({e}); "
                        f"pass parameters=") from e
    properties, required, readers = {}, [], {}
    for key, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise TypeError(f"{name}({key}): a model passes every argument by name, so a "
                            "parameter taken only by position could never be given")
        try:
            schema, readers[key] = _described(parameter.annotation)
        except TypeError as e:
            raise TypeError(f"{name}({key}): {e}; pass parameters= to describe it") from e
        if parameter.default is parameter.empty:
            required.append(key)
        else:
            default = parameter.default
            default = default.value if isinstance(default, enum.Enum) else default
            if _is_json(default):
                schema = {**schema, "default": default}
        properties[key] = schema
    return {"type": "object", "properties": properties, "required": required}, readers


# ----- one parameter's type, as a schema and as a check of the model's value ----------
# Reads one argument value against a parameter's type, the parameter's name or the
# value's place in it as the second argument; returns the value as the function takes
# it, or raises TypeError saying what was wrong.
_Reader = Callable[[Any, str], Any]

_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _described(annotation) -> tuple[dict, _Reader]:
    """The JSON Schema of one annotation, and the reader of a value against it.

    Raises:
        TypeError: If the annotation is not one this can describe.
    """
    origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Annotated:
        schema, read = _described(arguments[0])
        notes = [note for note in arguments[1:] if isinstance(note, str)]
        return ({**schema, "description": " ".join(notes)} if notes else schema), read
    if origin in (typing.Union, types.UnionType):
        return _nullable(arguments)
    if origin is typing.Literal:
        return _choices(list(arguments), lambda value: value)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _choices([member.value for member in annotation], annotation)
    if annotation is list or origin is list:
        return _array(arguments)
    if annotation is dict or origin is dict:
        return _mapping(arguments)
    if annotation in _SCALARS:
        return {"type": _SCALARS[annotation]}, _scalar_reader(annotation)
    shown = "no annotation" if annotation is inspect.Parameter.empty else repr(annotation)
    raise TypeError(f"{shown} is not a type this can describe; annotate it as text, a number, "
                    "a truth value, a list, a dict, an optional value, a Literal or an Enum")


def _nullable(arguments: tuple) -> tuple[dict, _Reader]:
    """An optional value: one type or ``None``."""
    given = [argument for argument in arguments if argument is not type(None)]
    if len(given) != 1 or len(arguments) != 2:
        raise TypeError(f"a union of {', '.join(map(repr, given))} is not a type this can "
                        "describe; only one type or None is")
    schema, read = _described(given[0])
    if isinstance(schema.get("type"), str):
        schema = {**schema, "type": [schema["type"], "null"]}
    if "enum" in schema:
        schema = {**schema, "enum": schema["enum"] + [None]}
    return schema, lambda value, where: None if value is None else read(value, where)


def _choices(values: list, chosen: Callable) -> tuple[dict, _Reader]:
    """One of ``values``, each a JSON scalar; the function is handed ``chosen(value)``."""
    if not values or not all(isinstance(v, (str, int, float, bool)) or v is None for v in values):
        raise TypeError(f"the choices {values!r} are not all text, numbers, truth values or None")
    kinds = {_json_type(v) for v in values}
    schema = {"type": kinds.pop(), "enum": values} if len(kinds) == 1 else {"enum": values}

    def read(value, where):
        for allowed in values:
            if value == allowed and isinstance(value, bool) == isinstance(allowed, bool):
                return chosen(allowed)
        raise TypeError(f"{where} must be one of {values!r}, not {value!r}")
    return schema, read


def _array(arguments: tuple) -> tuple[dict, _Reader]:
    """A list, of one type where it names one."""
    if not arguments:
        return {"type": "array"}, _scalar_reader(list)
    schema, read_item = _described(arguments[0])

    def read(value, where):
        if not isinstance(value, list):
            raise TypeError(f"{where} must be a list, not {value!r}")
        return [read_item(item, f"{where}[{i}]") for i, item in enumerate(value)]
    return {"type": "array", "items": schema}, read


def _mapping(arguments: tuple) -> tuple[dict, _Reader]:
    """A dict from text, of one value type where it names one."""
    if not arguments:
        return {"type": "object"}, _scalar_reader(dict)
    if arguments[0] is not str:
        raise TypeError(f"a dict whose keys are {arguments[0]!r} is not a type this can "
                        "describe; a JSON object's keys are text")
    schema, read_value = _described(arguments[1])

    def read(value, where):
        if not isinstance(value, dict):
            raise TypeError(f"{where} must be an object, not {value!r}")
        return {key: read_value(item, f"{where}[{key!r}]") for key, item in value.items()}
    return {"type": "object", "additionalProperties": schema}, read


def _scalar_reader(kind: type) -> _Reader:
    """The reader of a value of one plain type; ``True`` and ``False`` are never numbers."""
    def read(value, where):
        fits = (isinstance(value, (int, float)) if kind is float else isinstance(value, kind))
        if not fits or isinstance(value, bool) and kind is not bool:
            raise TypeError(f"{where} must be {_ARGUMENT_KIND_NAMES[kind]}, not {value!r}")
        return value
    return read


_ARGUMENT_KIND_NAMES = {str: "text", int: "an integer", float: "a number",
                        bool: "true or false", list: "a list", dict: "an object"}


def _json_type(value) -> str:
    """The JSON Schema type of one scalar value."""
    if value is None:
        return "null"
    return next(_SCALARS[kind] for kind in (bool, int, float, str) if isinstance(value, kind))


def _is_json(value) -> bool:
    """Whether JSON can write ``value`` as it is."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True
