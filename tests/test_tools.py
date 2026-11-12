"""Registering tools, the declarations registering makes, and the arguments a tool is run with."""

import enum
import functools
import re
from typing import Annotated, Literal, Optional

import pytest

from llama_chat.tools import ToolCall, ToolRegistry
from tests.wrappers import system_turn, tool_wrapper


# ----- a call --------------------------------------------------------------
def test_a_call_may_be_kept_in_a_set_or_as_a_key_and_equals_only_a_call_with_every_field_equal():
    call = ToolCall("weather", {"city": "Bonn"}, id="call00000")
    same = ToolCall("weather", {"city": "Bonn"}, id="call00000")
    other_arguments = ToolCall("weather", {"city": "Köln"}, id="call00000")
    assert {call, same} == {call} and {call: 1}[same] == 1
    assert hash(call) == hash(same)
    assert call != other_arguments and len({call, other_arguments}) == 2


# ----- the forms of the decorator ------------------------------------------
def test_a_bare_decorator_describes_the_tool_by_its_docstring():
    registry = ToolRegistry()

    @registry.tool
    def clock(city: str) -> str:
        """Sagt die Zeit in einer Stadt."""
        return "09:00"

    spec, = registry.specs()
    assert (spec.name, spec.description) == ("clock", "Sagt die Zeit in einer Stadt.")
    # The function is the caller's own still, and runs as it did.
    assert clock("Bonn") == "09:00"


def test_a_string_given_to_the_decorator_is_the_description():
    registry = ToolRegistry()

    @registry.tool("Current weather for a city.")
    def get_weather(city: str) -> dict:
        """Not what the model reads."""
        return {}

    assert registry.specs()[0].description == "Current weather for a city."
    assert get_weather("Bonn") == {}


def test_the_decorator_takes_a_name_a_description_and_a_schema_by_keyword():
    registry = ToolRegistry()
    schema = {"type": "object", "properties": {"zone": {"type": "string"}}}

    @registry.tool(name="clock", description="Sagt die Zeit.", parameters=schema)
    def _clock_v2(**zone):
        return "09:00"

    spec, = registry.specs()
    assert (spec.name, spec.description, spec.parameters) == ("clock", "Sagt die Zeit.", schema)


def test_a_function_written_elsewhere_is_registered_by_calling_tool_on_it():
    registry = ToolRegistry()

    def clock(city: str) -> str:
        return "09:00"

    assert registry.tool(clock, description="Sagt die Zeit.") is clock
    assert registry.call(ToolCall("clock", {"city": "Bonn"})) == "09:00"


def test_a_tool_without_a_description_is_refused():
    with pytest.raises(TypeError, match="no description"):
        ToolRegistry().tool(lambda: "12:00", name="clock")

    with pytest.raises(TypeError, match="no description"):
        @ToolRegistry().tool
        def clock() -> str:
            return "12:00"


def test_a_description_given_twice_is_refused():
    with pytest.raises(TypeError, match="given twice"):
        ToolRegistry().tool("Eins.", description="Zwei.")


def test_an_asynchronous_function_is_refused_when_it_is_registered():
    """A call would hand back a coroutine that never runs."""
    async def clock(city: str) -> str:
        return "09:00"

    with pytest.raises(TypeError, match="asynchronous"):
        ToolRegistry().tool(clock, description="Sagt die Zeit.")


def test_a_parameter_taken_only_by_position_is_refused():
    """A model passes every argument by name, so the tool could never run."""
    def clock(city: str, /) -> str:
        return "09:00"

    with pytest.raises(TypeError, match="only by position"):
        ToolRegistry().tool(clock, description="Sagt die Zeit.")


def test_a_registry_is_declared_as_it_is():
    registry = ToolRegistry()

    @registry.tool
    def clock() -> str:
        """Sagt die Zeit."""
        return "12:00"

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry)
    assert "<decl>clock</decl>" in system_turn(fake)
    assert list(registry) == registry.specs()


# ----- what a function declares --------------------------------------------
def test_a_decorated_function_declares_its_own_parameters():
    registry = ToolRegistry()

    @registry.tool("Trägt einen Termin ein.")
    def calendar_add(title: str, minutes: int, whole_day: bool = False) -> dict:
        ...

    spec, = registry.specs()
    assert spec.name == "calendar_add"
    assert spec.parameters["properties"] == {
        "title": {"type": "string"},
        "minutes": {"type": "integer"},
        "whole_day": {"type": "boolean", "default": False},
    }
    # A parameter with a default is one the model may leave out.
    assert spec.parameters["required"] == ["title", "minutes"]


class _Unit(enum.Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


def test_lists_dicts_optional_values_choices_enums_and_descriptions_are_described():
    registry = ToolRegistry()

    @registry.tool("Das Wetter.")
    def weather(city: Annotated[str, "the city"], days: list[int],
                hours: dict[str, float], note: Optional[str] = None,
                style: Literal["short", "long"] = "short", unit: _Unit = _Unit.CELSIUS,
                tags: list | None = None):
        ...

    properties = registry.specs()[0].parameters["properties"]
    assert properties == {
        "city": {"type": "string", "description": "the city"},
        "days": {"type": "array", "items": {"type": "integer"}},
        "hours": {"type": "object", "additionalProperties": {"type": "number"}},
        "note": {"type": ["string", "null"], "default": None},
        "style": {"type": "string", "enum": ["short", "long"], "default": "short"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        "tags": {"type": ["array", "null"], "default": None},
    }
    assert registry.specs()[0].parameters["required"] == ["city", "days", "hours"]


@pytest.mark.parametrize("annotation", [complex, "no annotation", int | str, dict[int, str],
                                        Literal[1.5j]])
def test_a_parameter_the_schema_cannot_describe_is_refused(annotation):
    def broken(when):
        ...
    if annotation != "no annotation":
        broken.__annotations__ = {"when": annotation}
    with pytest.raises(TypeError, match="pass parameters="):
        ToolRegistry().tool(broken, description="...")


def test_a_nameless_callable_with_a_parameter_that_cannot_be_described_is_refused_as_such():
    def broken(when: complex):
        ...
    with pytest.raises(TypeError, match=r"clock\(when\).*not a type this can describe"):
        ToolRegistry().tool(functools.partial(broken), description="...", name="clock")


def test_an_explicit_schema_and_the_declarations_handed_out_are_copies():
    schema = {"type": "object", "properties": {"zone": {"type": "string"}}}
    registry = ToolRegistry()
    registry.tool(lambda zone: zone, description="…", name="clock", parameters=schema)
    schema["properties"]["zone"]["type"] = "integer"
    registry.specs()[0].parameters["properties"].clear()
    assert registry.specs()[0].parameters == {
        "type": "object", "properties": {"zone": {"type": "string"}}}


def test_variadic_parameters_are_left_out_of_the_schema():
    registry = ToolRegistry()

    @registry.tool("Nimmt beliebige Argumente.")
    def spread(city: str, *rest, **extra):
        return city

    spec, = registry.specs()
    assert spec.parameters["properties"] == {"city": {"type": "string"}}
    assert spec.parameters["required"] == ["city"]


def test_a_function_whose_annotations_are_postponed_registers_as_its_types():
    from .postponed_tool import weather
    registry = ToolRegistry()
    registry.tool(weather, description="Weather for a city")
    assert registry.specs()[0].parameters == {
        "type": "object", "properties": {"city": {"type": "string"},
                                         "days": {"type": "integer", "default": 1}},
        "required": ["city"]}


# ----- the registry --------------------------------------------------------
def test_a_tool_may_be_called_something_other_than_its_function():
    registry = ToolRegistry()

    def _add_v2(title: str) -> dict:
        return {"ok": True}

    registry.tool(_add_v2, description="…", name="calendar_add")
    assert [spec.name for spec in registry.specs()] == ["calendar_add"]
    assert registry.call(ToolCall("calendar_add", {"title": "x"})) == {"ok": True}


def test_a_function_with_no_name_to_call_it_by_is_refused_at_registration():
    """Registered under a name no model would ever write, it never runs.

    Nothing later says so: the tool is declared, sits in the system turn for the
    whole conversation, and is never called.
    """
    with pytest.raises(TypeError, match="pass name="):
        ToolRegistry().tool(lambda: "12:00", description="Die Zeit.")
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    assert [spec.name for spec in registry.specs()] == ["clock"]


def test_registering_a_name_again_replaces_it():
    registry = ToolRegistry()
    registry.tool(lambda: "first", description="Erste.", name="clock")
    assert registry.specs()[0].description == "Erste."

    registry.tool(lambda: "second", description="Zweite.", name="clock")
    assert len(registry) == 1
    assert [s.description for s in registry.specs()] == ["Zweite."]
    assert registry.call(ToolCall("clock", {})) == "second"


# ----- running a tool ------------------------------------------------------
def test_a_tool_runs_with_the_arguments_the_model_gave():
    registry = ToolRegistry()

    @registry.tool("…")
    def add(title: str, minutes: int = 30) -> dict:
        return {"title": title, "minutes": minutes}

    # The default stands in for what the model left out.
    assert registry.call(ToolCall("add", {"title": "Zahnarzt"})) == {
        "title": "Zahnarzt", "minutes": 30}


@pytest.mark.parametrize("arguments, named", [
    ({"title": "x", "minutes": "five"}, "minutes must be an integer"),
    ({"title": "x", "minutes": True}, "minutes must be an integer"),
    ({"title": ["x"], "minutes": 5}, "title must be text"),
    ({"title": "x", "minutes": 5, "days": [1, "2"]}, r"days\[1\] must be an integer"),
    ({"title": "x", "minutes": 5, "style": "medium"}, "style must be one of"),
    ({"title": "x", "minutes": 5, "unit": "kelvin"}, "unit must be one of"),
    ({"title": "x", "minutes": 5, "hours": {"a": "b"}}, r"hours\['a'\] must be a number"),
    ({"title": "x"}, "missing a required argument: 'minutes'"),
    ({"title": "x", "minutes": 5, "nope": 1}, "unexpected keyword argument 'nope'"),
])
def test_arguments_that_do_not_fit_the_types_are_the_models_mistake(arguments, named):
    ran = []
    registry = ToolRegistry()

    @registry.tool("…")
    def add(title: str, minutes: int, days: list[int] | None = None,
            style: Literal["short", "long"] = "short", unit: _Unit = _Unit.CELSIUS,
            hours: dict[str, float] | None = None) -> dict:
        ran.append(1)
        return {}

    result = registry.call(ToolCall("add", arguments))
    assert result["error"] == "wrong_arguments"
    assert re.search(named, result["message"])
    assert ran == []


def test_arguments_that_fit_reach_the_tool_as_its_types():
    registry, given = ToolRegistry(), []

    @registry.tool("…")
    def add(title: str, minutes: float, unit: _Unit, note: str | None = None) -> str:
        given.append((title, minutes, unit, note))
        return "ok"

    assert registry.call(ToolCall("add", {"title": 42, "minutes": 5, "unit": "fahrenheit",
                                          "note": None})) == "ok"
    # A number where text is asked is the text it was written as, an integer is a
    # number, and an enum's value is its member.
    assert given == [("42", 5, _Unit.FAHRENHEIT, None)]


def test_an_unknown_tool_becomes_a_result_rather_than_an_exception():
    result = ToolRegistry().call(ToolCall("nope", {}))
    assert result["error"] == "unknown_tool"


def test_the_models_mistake_and_the_tools_failure_are_told_apart():
    registry = ToolRegistry()

    @registry.tool("…")
    def strict(title: str) -> dict:
        raise TypeError("this one is the tool's own")

    # The model passed an argument that does not exist.
    assert registry.call(ToolCall("strict", {"nope": 1}))["error"] == "wrong_arguments"
    # The tool raised TypeError itself, which is not the model's doing.
    assert registry.call(ToolCall("strict", {"title": "x"}))["error"] == "tool_raised"


# ----- builtins ------------------------------------------------------------
def test_a_builtin_with_no_readable_signature_is_still_usable():
    registry = ToolRegistry()
    # `dict.update` has no signature inspect can read, so the schema has to be
    # supplied and the argument check has to be skipped rather than raise.
    holder = {}
    registry.tool(holder.update, description="…", name="remember",
                  parameters={"type": "object", "properties": {}})
    assert registry.call(ToolCall("remember", {"a": 1})) is None
    assert holder == {"a": 1}


def test_a_builtin_registered_without_a_schema_says_what_to_do():
    with pytest.raises(TypeError, match="pass parameters="):
        ToolRegistry().tool({}.update, description="…", name="remember")
