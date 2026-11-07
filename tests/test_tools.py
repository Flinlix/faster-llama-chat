"""Registering tools, and the declarations registering makes."""

import pytest

from llama_chat.tools import ToolCall, ToolRegistry, tool


# ----- what a function declares --------------------------------------------
def test_a_decorated_function_declares_its_own_parameters():
    @tool("Trägt einen Termin ein.")
    def calendar_add(title: str, minutes: int, whole_day: bool = False) -> dict:
        ...

    spec = calendar_add.spec
    assert spec.name == "calendar_add"
    assert spec.parameters["properties"] == {
        "title": {"type": "string"},
        "minutes": {"type": "integer"},
        "whole_day": {"type": "boolean"},
    }
    # A parameter with a default is one the model may leave out.
    assert spec.parameters["required"] == ["title", "minutes"]


def test_a_parameter_the_schema_cannot_describe_is_refused():
    with pytest.raises(TypeError, match="not a type this can describe"):
        @tool("...")
        def broken(when: complex):
            ...


# ----- the registry --------------------------------------------------------
def test_registering_returns_the_name_the_model_calls():
    registry = ToolRegistry()

    def calendar_add(title: str) -> dict:
        return {"ok": True}

    assert registry.register(calendar_add, "Trägt ein.") == "calendar_add"
    assert "calendar_add" in registry


def test_a_tool_may_be_called_something_other_than_its_function():
    registry = ToolRegistry()

    def _add_v2(title: str) -> dict:
        return {"ok": True}

    assert registry.register(_add_v2, "…", name="calendar_add") == "calendar_add"
    assert "calendar_add" in registry and "_add_v2" not in registry


def test_the_flag_says_where_a_declaration_lands_not_whether():
    """Every registered tool is meant to be told; the flag picks where."""
    registry = ToolRegistry()
    registry.register(lambda: {}, "gleich", name="early")
    registry.register(lambda: {}, "später", name="late", in_system_prompt=False)

    assert [s.name for s in registry.specs_for_system_prompt()] == ["early"]
    assert sorted(s.name for s in registry.specs()) == ["early", "late"]
    assert registry.spec("late").description == "später"
    # Runnable from the moment it is registered, wherever it is declared.
    assert registry.call(ToolCall("late", {})) == {}


def test_a_tool_runs_with_the_arguments_the_model_gave():
    registry = ToolRegistry()

    def add(title: str, minutes: int = 30) -> dict:
        return {"title": title, "minutes": minutes}

    registry.register(add, "…")
    # The default stands in for what the model left out.
    assert registry.call(ToolCall("add", {"title": "Zahnarzt"})) == {
        "title": "Zahnarzt", "minutes": 30}


def test_a_parameter_the_schema_cannot_describe_is_refused_at_registration():
    # The same rule the decorator enforces, at the other entry point.
    with pytest.raises(TypeError, match="not a type this can describe"):
        ToolRegistry().register(lambda title: {}, "…", name="add")


def test_a_function_with_no_name_to_call_it_by_is_refused_at_registration():
    """Registered under a name no model would ever write, it never runs.

    Nothing later says so: the tool is declared, sits in the system turn for the
    whole conversation, and is never called.
    """
    with pytest.raises(TypeError, match="pass name="):
        ToolRegistry().register(lambda: "12:00", "Die Zeit.")
    assert ToolRegistry().register(lambda: "12:00", "Die Zeit.",
                                   name="clock") == "clock"


def test_an_unknown_tool_becomes_a_result_rather_than_an_exception():
    result = ToolRegistry().call(ToolCall("nope", {}))
    assert result["error"] == "unknown_tool"


def test_the_models_mistake_and_the_tools_failure_are_told_apart():
    registry = ToolRegistry()

    def strict(title: str) -> dict:
        raise TypeError("this one is the tool's own")

    registry.register(strict, "…")
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
    registry.register(holder.update, "…", name="remember",
                      parameters={"type": "object", "properties": {}})
    assert registry.call(ToolCall("remember", {"a": 1})) is None
    assert holder == {"a": 1}


def test_a_builtin_registered_without_a_schema_says_what_to_do():
    with pytest.raises(TypeError, match="pass parameters="):
        ToolRegistry().register({}.update, "…", name="remember")


# ----- registering ---------------------------------------------------------
def test_register_takes_what_the_tool_declaration_already_said():
    @tool("Sagt die Zeit.")
    def clock(city: str) -> str:
        return "09:00"

    registry = ToolRegistry()
    registry.register(clock)
    spec = registry.specs()[0]
    assert spec.description == "Sagt die Zeit."
    assert spec.parameters["properties"] == {"city": {"type": "string"}}
    assert len(registry) == 1 and "clock" in registry
    assert registry.call(ToolCall("clock", {"city": "Bonn"})) == "09:00"


def test_register_without_a_description_or_a_declaration_is_refused():
    def bare(x: int):
        pass

    with pytest.raises(TypeError, match="needs a description"):
        ToolRegistry().register(bare)


def test_registering_a_name_again_replaces_it_and_where_it_is_declared():
    registry = ToolRegistry()
    registry.register(lambda: "first", "Erste.", name="clock",
                      in_system_prompt=False)
    assert registry.specs()[0].description == "Erste."
    assert registry.specs_for_system_prompt() == []

    registry.register(lambda: "second", "Zweite.", name="clock")
    assert len(registry) == 1
    # The placement was replaced along with the rest of it.
    assert [s.name for s in registry.specs_for_system_prompt()] == ["clock"]
    assert registry.call(ToolCall("clock", {})) == "second"


def test_variadic_parameters_are_left_out_of_the_schema():
    @tool("Nimmt beliebige Argumente.")
    def spread(city: str, *rest, **extra):
        return city

    assert spread.spec.parameters["properties"] == {"city": {"type": "string"}}
    assert spread.spec.parameters["required"] == ["city"]


# ----- postponed annotations -----------------------------------------------
def test_a_function_whose_annotations_are_postponed_registers_as_its_types():
    from .postponed_tool import weather
    registry = ToolRegistry()
    registry.register(weather, "Weather for a city")
    assert registry.specs()[0].parameters == {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"]}
