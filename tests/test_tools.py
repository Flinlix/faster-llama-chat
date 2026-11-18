"""Registering tools, the declarations registering makes, and the arguments a tool is run with."""

import enum
import functools
import re
from typing import Annotated, Literal, Optional

import pytest

from tests.chat_messages import opened

from llama_chat.tool_registry import ToolRegistry

from llama_chat.tools import ToolCall
from tests.wrappers import system_turn, tool_wrapper
from llama_chat.result_cutting import _most_kept


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


def test_a_registry_is_declared_by_its_specs():
    registry = ToolRegistry()

    @registry.tool
    def clock() -> str:
        """Sagt die Zeit."""
        return "12:00"

    wrapper, fake = tool_wrapper()
    wrapper.begin(opened("persona"), tools=registry.specs())
    assert "<decl>clock</decl>" in system_turn(fake)
    with pytest.raises(TypeError, match="registry.specs|ToolRegistry.specs"):
        wrapper.begin(opened("persona"), tools=registry)


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

    assert registry.call(ToolCall("add", {"title": "42", "minutes": 5, "unit": "fahrenheit",
                                          "note": None})) == "ok"
    # An integer is a number, and an enum's value is its member.
    assert given == [("42", 5, _Unit.FAHRENHEIT, None)]
    # A number where text is asked is the model's mistake, told to it rather than mended.
    assert registry.call(ToolCall("add", {"title": 42, "minutes": 5,
                                          "unit": "fahrenheit"}))["error"] == "wrong_arguments"


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


def test_a_result_cut_again_counts_the_whole_it_was_first_cut_from():
    from llama_chat.result_cutting import cut_result
    once = cut_result("x" * 85, 5)
    assert once.endswith("(cut: 5 of 85 characters kept)")
    assert cut_result(once, 4) == "xxxx\n(cut: 4 of 85 characters kept)"
    assert cut_result(once, 50) == once
    assert cut_result("Text\n(cut: 9 of 10 characters kept)", 3).endswith("of 35 characters kept)")


def test_a_tool_spec_of_the_wrong_types_is_refused_naming_what_each_must_be():
    from llama_chat.errors import ArgumentTypeError
    from llama_chat.tools import ToolSpec
    with pytest.raises(ArgumentTypeError, match="str name, a str description and a dict"):
        ToolSpec(1, "Sagt die Zeit.", {})


def test_an_asynchronous_generator_or_a_callable_object_with_an_asynchronous_call_is_refused_as_a_tool():
    class Clock:
        async def __call__(self, city: str) -> str:
            return "09:00"

    async def times(city: str):
        yield "09:00"
    for asynchronous in (Clock(), times):
        with pytest.raises(TypeError, match="asynchronous"):
            ToolRegistry().tool(asynchronous, name="clock", description="Sagt die Zeit.")


def _kept_by_every_guess(total, most):
    """What ``_most_kept`` finds for a ``fits`` that holds up to ``most`` characters, from no guess and from every guess."""
    def fits(n):
        return min(n, total) <= most      # a count past the text keeps the whole text, as a cut does
    return {guess: _most_kept(fits, total, guess) for guess in [None, *range(total + 1)]}


def test_a_cut_keeps_the_most_characters_that_fit_wherever_the_guess_lies():
    for total in [*range(1, 90), 300, 1000]:
        for most in sorted({*range(min(total, 90)), *range(0, total, 37)}):
            kept = _kept_by_every_guess(total, most)
            assert all(type(n) is int and n == most for n in kept.values()), (total, most, kept)


def test_a_cut_of_a_text_that_fits_whole_keeps_all_of_it_wherever_the_guess_lies():
    for total in [*range(90), 300, 1000]:
        kept = _kept_by_every_guess(total, total)
        assert all(type(n) is int and n == total for n in kept.values()), (total, kept)


def test_an_annotation_is_described_by_its_text_notes_alone_joined_by_spaces():
    registry = ToolRegistry()

    @registry.tool("Das Wetter.")
    def weather(city: Annotated[str, "the city", "as the post writes it"],
                days: Annotated[int, 7]):
        ...

    assert registry.specs()[0].parameters["properties"] == {
        "city": {"type": "string", "description": "the city as the post writes it"},
        "days": {"type": "integer"},
    }


def test_every_choice_of_a_literal_reaches_the_tool_as_it_was_written():
    choices = ["short", "long", 3, True, None]
    registry, given = ToolRegistry(), []

    @registry.tool("…")
    def pick(style: Literal["short", "long", 3, True, None]) -> str:
        given.append(style)
        return "ok"

    assert [registry.call(ToolCall("pick", {"style": c})) for c in choices] == ["ok"] * len(choices)
    assert given == choices and [type(g) for g in given] == [type(c) for c in choices]


def test_an_optional_choice_admits_null_among_its_choices():
    registry = ToolRegistry()

    @registry.tool("…")
    def pick(style: Optional[Literal["short", "long"]], unit: _Unit | None = None):
        ...

    properties = registry.specs()[0].parameters["properties"]
    assert properties["style"] == {"type": ["string", "null"], "enum": ["short", "long", None]}
    assert properties["unit"] == {"type": ["string", "null"],
                                  "enum": ["celsius", "fahrenheit", None], "default": None}


@pytest.mark.parametrize("annotation, shown", [
    (complex, repr(complex)), (set[int], repr(set[int])), (None, "no annotation")])
def test_a_parameter_that_cannot_be_described_is_refused_naming_its_annotation(annotation, shown):
    def broken(when):
        ...
    if annotation is not None:
        broken.__annotations__ = {"when": annotation}
    with pytest.raises(TypeError) as refused:
        ToolRegistry().tool(broken, description="...")
    message = str(refused.value)
    assert message.startswith("broken(when): ") and "is not a type this can describe" in message
    assert shown.lower() in message.lower()
    assert ("no annotation" in message.lower()) == (annotation is None)


@pytest.mark.parametrize("annotation", [int | str, Optional[int | str], int | list[str] | None])
def test_a_union_of_more_than_one_type_is_refused_naming_each_type_in_it(annotation):
    def broken(when):
        ...
    broken.__annotations__ = {"when": annotation}
    with pytest.raises(TypeError) as refused:
        ToolRegistry().tool(broken, description="...")
    message = str(refused.value)
    named = [a for a in annotation.__args__ if a is not type(None)]
    assert message.startswith("broken(when): a union of ")
    assert all(repr(a) in message for a in named) and "only one type or none" in message.lower()
    assert "NoneType" not in message


def test_a_tool_given_neither_a_function_nor_a_description_is_refused_naming_what_it_was_given():
    with pytest.raises(TypeError, match="a function or its description, not 42"):
        ToolRegistry().tool(42)


def test_every_failure_of_a_call_is_a_result_carrying_its_kind_and_a_message_that_names_it():
    registry = ToolRegistry()

    @registry.tool("…")
    def clock(city: str) -> str:
        raise ValueError(f"no clock in {city}")

    failures = [(ToolCall("nope", {}), "unknown_tool", "'nope'"),
                (ToolCall("clock", {"city": 7}), "wrong_arguments", "city must be text"),
                (ToolCall("clock", ["Bonn"]), "wrong_arguments", "['Bonn']"),
                (ToolCall("clock", {"city": "Bonn"}), "tool_raised", "ValueError: no clock in Bonn")]
    for call, kind, named in failures:
        result = registry.call(call)
        assert set(result) == {"error", "message"} and result["error"] == kind
        assert named in result["message"]


def test_a_callable_that_carries_no_name_at_all_is_refused_and_asked_for_one():
    class Clock:
        def __call__(self) -> str:
            return "12:00"

    for nameless in (functools.partial(lambda: "12:00"), Clock()):
        registry = ToolRegistry()
        with pytest.raises(TypeError, match="no name the model could call it by; pass name="):
            registry.tool(nameless, description="Die Zeit.")
        assert len(registry) == 0


def test_a_bound_method_is_described_by_its_own_docstring():
    class Calendar:
        def add(self, title: str) -> str:
            """Trägt einen Termin ein."""
            return title

    registry = ToolRegistry()
    registry.tool(Calendar().add)
    spec, = registry.specs()
    assert (spec.name, spec.description) == ("add", "Trägt einen Termin ein.")
    assert registry.call(ToolCall("add", {"title": "Zahnarzt"})) == "Zahnarzt"


def test_parameters_after_a_variadic_one_are_declared_and_checked():
    registry = ToolRegistry()

    @registry.tool("Nimmt beliebige Argumente.")
    def spread(*rest, city: str, days: int = 1):
        return city

    spec, = registry.specs()
    assert spec.parameters["properties"] == {"city": {"type": "string"},
                                             "days": {"type": "integer", "default": 1}}
    assert spec.parameters["required"] == ["city"]
    assert registry.call(ToolCall("spread", {"city": 5}))["error"] == "wrong_arguments"


def test_a_default_that_json_cannot_write_is_left_out_of_the_declaration():
    unset = object()
    registry = ToolRegistry()

    @registry.tool("Sagt das Wetter.")
    def weather(city: str, hours: float = float("nan"), note: str | None = unset) -> str:
        return city

    spec, = registry.specs()
    assert spec.parameters["properties"] == {"city": {"type": "string"},
                                             "hours": {"type": "number"},
                                             "note": {"type": ["string", "null"]}}
