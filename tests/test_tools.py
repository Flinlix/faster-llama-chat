"""Declaring tools, reading calls back, and the turn a call holds open.

Model-free throughout: :class:`~tests.fake_context.FakeContext` plays a second
model whose tool tags are deliberately spelled unlike the bundled one's, so
nothing here can pass by matching a real model's format.
"""

import json
import re

import pytest

from llama_chat.config import Config
from llama_chat.template import CallBody, TemplateFormatter
from llama_chat.template_extract import extract_call_body, extract_fragments
from llama_chat.tools import ToolCall, ToolRegistry, ToolSpec, parse_calls, tool
from llama_chat.wrapper import ChatWrapper, ContextOverflowError
from tests.fake_context import DEFAULT_FRAGMENTS, TOOL_FRAGMENTS, FakeContext

# The call_body the fake model's renders yield, spelled out so the parser tests
# read without a round trip through extraction.
FAKE = CallBody(name_close="{", key_open="", key_close=":", pair_sep=",",
                body_close="}", empty_close="}", quote_open="'", quote_close="'")


def _wrapper(**kw):
    fake = FakeContext(fragments=TOOL_FRAGMENTS, **kw)
    return ChatWrapper(Config(context_size=512, evict_above_fraction=0.9),
                       context=fake), fake


def _flatten(wrapper):
    toks = []
    for m in wrapper._table.messages:
        toks.extend(m.token_ids)
    return toks


# ----- declaring ----------------------------------------------------------
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


def test_the_declaration_reaches_the_system_turn():
    wrapper, fake = _wrapper()

    @tool("Sagt die Zeit.")
    def clock() -> dict:
        ...

    wrapper.begin("persona", tools=[clock.spec])

    # The fake template renders a declaration as <decl>NAME</decl>, and the
    # 1:1 char tokenizer lets the system turn be read straight back.
    system = wrapper._table.messages[0]
    written = "".join(chr(t) for t in system.token_ids if t >= 0)
    assert "<decl>clock</decl>" in written
    assert "persona" in written
    assert fake.cache == _flatten(wrapper)


def test_a_declaration_is_tokenized_as_tags_and_the_prompt_is_not():
    wrapper, fake = _wrapper()

    @tool("Sagt die Zeit.")
    def clock() -> dict:
        ...

    wrapper.begin("persona", tools=[clock.spec])

    # The prompt is a caller's text and must not be able to forge a tag; the
    # declarations are the model's own and have to parse as tags.
    assert ("persona", False) in fake.tokenize_calls
    assert any("<decl>clock</decl>" in text and special
               for text, special in fake.tokenize_calls)


# ----- reading a call back ------------------------------------------------
def test_a_call_parses_into_a_name_and_typed_arguments():
    call, = parse_calls("fn:calendar_add{title:'Zahnarzt',minutes:30,all_day:false}",
                      FAKE, "fn:")
    assert call == ToolCall(name="calendar_add",
                            arguments={"title": "Zahnarzt", "minutes": 30,
                                       "all_day": False})


def test_a_quoted_argument_may_hold_the_separators():
    call, = parse_calls("fn:say{text:'a,b:c{d}'}", FAKE, "fn:")
    assert call.arguments == {"text": "a,b:c{d}"}


def test_nested_objects_and_lists_survive():
    call, = parse_calls("fn:x{where:{city:'Bern'},days:[1,2]}", FAKE, "fn:")
    assert call.arguments == {"where": {"city": "Bern"}, "days": [1, 2]}


@pytest.mark.parametrize("body", ["", "fn:", "fn:name", "fn:{a:1}"])
def test_something_that_is_not_a_call_is_refused(body):
    with pytest.raises(ValueError):
        parse_calls(body, FAKE, "fn:")


# ----- the turn a call holds open -----------------------------------------
def test_a_call_ends_the_stream_and_holds_the_turn_open():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    events = list(wrapper.generate_events("wie spät?"))

    assert events == [ToolCall(name="clock", arguments={})]
    assert wrapper.awaiting_tool_result
    # Nothing of the call reached the reply text.
    assert wrapper.snapshot()[-1]["text"] == ""
    assert fake.cache == _flatten(wrapper)


def test_a_result_resumes_the_same_turn():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))

    wrapper.inject_tool_result("clock", {"time": "09:00"})
    fake.call_script = []
    events = list(wrapper.resume_events())

    assert "".join(e for e in events if isinstance(e, str))
    assert not wrapper.awaiting_tool_result
    # One assistant message for the whole exchange, and the cache still agrees.
    assert [m["role"] for m in wrapper.snapshot()] == ["system", "user", "assistant"]
    assert fake.cache == _flatten(wrapper)


def test_two_calls_in_a_row_stay_inside_one_turn():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:first{}"]
    assert list(wrapper.generate_events("los")) == [ToolCall("first", {})]

    wrapper.inject_tool_result("first", "ok")
    fake.call_script = ["fn:second{}"]
    assert list(wrapper.resume_events()) == [ToolCall("second", {})]

    wrapper.inject_tool_result("second", "ok")
    fake.call_script = []
    list(wrapper.resume_events())

    assert [m["role"] for m in wrapper.snapshot()] == ["system", "user", "assistant"]
    assert fake.cache == _flatten(wrapper)


def test_an_open_turn_outgrows_the_message_limit_and_is_trimmed_on_close():
    """The two rules meet here: nothing is evicted while a turn is open.

    A turn holding a call would be cut into by an eviction, so the message limit
    is allowed to be exceeded until the turn closes, and enforced again the
    moment it does.
    """
    fake = FakeContext(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=4000, max_history_messages=2),
                          context=fake)
    wrapper.begin("persona", [("user", "alt"), ("assistant", "älter")])
    fake.call_script = ["fn:first{}"]
    assert list(wrapper.generate_events("los")) == [ToolCall("first", {})]
    assert len(wrapper.snapshot()) - 1 > 2      # over the limit, turn still open

    wrapper.inject_tool_result("first", "ok")
    fake.call_script = []
    list(wrapper.resume_events())
    assert len(wrapper.snapshot()) - 1 == 2     # back inside it once closed
    assert wrapper.snapshot()[0]["role"] == "system"
    assert fake.cache == _flatten(wrapper)


def test_a_result_under_the_wrong_name_is_refused():
    """Results are owed in call order, so the name identifies which one."""
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']
    list(wrapper.generate_events("los"))

    with pytest.raises(ValueError, match="the waiting call is 'clock'"):
        wrapper.inject_tool_result("other", "b")

    # The refusal consumed nothing: both calls are still owed, in order.
    wrapper.inject_tool_result("clock", "a")
    wrapper.inject_tool_result("other", "b")
    fake.call_script = []
    list(wrapper.resume_events())
    assert fake.cache == _flatten(wrapper)


def test_a_result_whose_wrapping_cannot_fit_is_refused_whole():
    """The tags are never clipped, so a turn with no room for them is refused.

    Clipping them would leave the cache holding half a block no template ever
    produces, which is worse than a turn that cannot be continued.
    """
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))

    before = len(fake.cache)
    wrapper._config = Config(context_size=before + 1, evict_above_fraction=0.9)
    with pytest.raises(ContextOverflowError, match="result wrapping alone"):
        wrapper.inject_tool_result("clock", "ok")

    # Nothing reached the cache, and the call is still owed.
    assert len(fake.cache) == before
    assert wrapper.awaiting_tool_result
    assert fake.cache == _flatten(wrapper)


def test_a_result_with_no_call_waiting_is_refused():
    wrapper, _ = _wrapper()
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="no tool call is waiting"):
        wrapper.inject_tool_result("clock", "ok")


def test_resuming_with_no_open_reply_is_refused():
    wrapper, _ = _wrapper()
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="no reply is open"):
        list(wrapper.resume_events())


def test_a_new_turn_cannot_open_inside_a_reply_a_call_left_open():
    """A reply stays open after its last result, until it is resumed.

    Nothing owes a result any more, but the assistant turn has no closing tag
    yet. A message appended here would sit inside a model turn that never
    ended, which is a conversation no template renders.
    """
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))
    wrapper.inject_tool_result("clock", "ok")
    assert not wrapper.awaiting_tool_result

    for action in (lambda: wrapper.generate("etwas anderes"),
                   lambda: wrapper.inject("etwas anderes"),
                   lambda: wrapper.declare_tools([ToolSpec("x", "d", {})])):
        with pytest.raises(ValueError, match="while a reply is open"):
            action()

    # Refusing left the turn untouched: it still resumes and closes normally.
    fake.call_script = []
    list(wrapper.resume_events())
    assert [m["role"] for m in wrapper.snapshot()] == ["system", "user", "assistant"]
    assert fake.cache == _flatten(wrapper)


def test_begin_discards_a_reply_a_call_left_open():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))

    wrapper.begin("persona")
    wrapper.generate("etwas anderes")     # the open turn went with the reset
    assert [m["role"] for m in wrapper.snapshot()] == ["system", "user", "assistant"]
    assert fake.cache == _flatten(wrapper)


def test_a_model_without_tool_tags_cannot_call_tools():
    fake = FakeContext(fragments=DEFAULT_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    assert not wrapper.can_call_tools


def test_a_model_with_tool_tags_can():
    wrapper, _ = _wrapper()
    assert wrapper.can_call_tools


class _ConstantBodyContext(FakeContext):
    """Tags a caller supplied, over a template that never renders an argument."""

    def render_with_model_template(self, messages, tools=None):
        rendered = super().render_with_model_template(messages, tools)
        return re.sub(r"(<call>fn:\w+)\{[^}]*\}", r"\1{}", rendered)


def test_tags_without_a_readable_grammar_cannot_call_tools():
    """The tags resolve, so a call is still detected; nothing runnable comes out.

    Reporting this as callable would only be found out by making a call and
    getting an empty reply back.
    """
    fake = _ConstantBodyContext(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    assert wrapper._call_body is None           # the probes recovered nothing
    assert not wrapper.can_call_tools

    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    reply = wrapper.generate("los")
    assert reply.stop_reason == "unreadable_tool_call"
    assert reply.text == ""                      # the call never reaches the reply


def test_a_supplied_grammar_makes_such_a_model_callable_again():
    fake = _ConstantBodyContext(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake,
                          call_body=FAKE)
    assert wrapper.can_call_tools


def test_an_oversized_result_is_clipped_and_keeps_its_tags():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))

    before = len(fake.cache)
    wrapper.inject_tool_result("clock", "x" * 10_000)

    # The turn still fits, and the block still ends where the template ends one.
    assert len(fake.cache) <= wrapper._config.context_size
    written = "".join(chr(t) for t in fake.cache[before:] if t >= 0)
    assert written.startswith("<result>clock(")
    assert written.endswith(")</result>")
    assert fake.cache == _flatten(wrapper)


# ----- the registry -------------------------------------------------------
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


def test_only_declared_tools_are_offered_to_the_model():
    registry = ToolRegistry()
    registry.register(lambda: {}, "laut", name="loud", declare=True)
    registry.register(lambda: {}, "leise", name="quiet")

    assert [s.name for s in registry.specs_to_declare()] == ["loud"]
    # Registered and silent is still registered: it runs the moment it is named.
    assert registry.call(ToolCall("quiet", {})) == {}

    spec = registry.declare("quiet")
    assert spec.name == "quiet"
    assert sorted(s.name for s in registry.specs_to_declare()) == ["loud", "quiet"]


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


# ----- the local loop -----------------------------------------------------
def test_answer_stream_runs_the_tool_and_yields_only_prose():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    seen = []
    registry.register(lambda: seen.append("ran") or {"time": "09:00"},
                      "Sagt die Zeit.", name="clock")
    fake.call_script = ["fn:clock{}"]

    out = "".join(wrapper.answer_stream("wie spät?", registry))

    assert seen == ["ran"]
    assert out                        # the reply after the result
    assert not wrapper.awaiting_tool_result
    assert fake.cache == _flatten(wrapper)


def test_answer_stops_calling_at_the_ceiling_and_still_closes_the_turn():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    runs = []
    registry.register(lambda: runs.append(1) or {}, "…", name="clock")
    # The fake would call forever; the ceiling is what has to stop it.
    fake.call_script = ["fn:clock{}"] * 10

    list(wrapper.answer_stream("los", registry, max_calls=2))

    assert len(runs) == 2
    assert not wrapper.awaiting_tool_result
    assert [m["role"] for m in wrapper.snapshot()] == ["system", "user", "assistant"]
    assert fake.cache == _flatten(wrapper)


def test_a_declaration_added_later_reaches_the_model():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    registry.register(lambda: {}, "Sagt die Zeit.", name="clock")

    wrapper.declare_tools([registry.declare("clock")])

    written = "".join(chr(t) for t in fake.cache if t >= 0)
    assert "<decl>clock</decl>" in written
    assert fake.cache == _flatten(wrapper)


def test_declaring_nothing_writes_nothing():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    before = list(fake.cache)
    assert wrapper.declare_tools([]) == 0
    assert fake.cache == before


def test_declarations_too_large_to_fit_are_refused_rather_than_truncated():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    before = list(fake.cache)
    registry = ToolRegistry()
    # The fake renders one <decl>NAME</decl> per tool, so many long names
    # overrun the threshold.
    for i in range(200):
        registry.register(lambda: {}, "…", name=f"tool_with_a_long_name_{i}",
                          declare=True)

    with pytest.raises(ValueError, match="declare fewer tools"):
        wrapper.declare_tools(registry.specs_to_declare())

    # A refused declaration leaves the conversation exactly as it was.
    assert fake.cache == before
    assert fake.cache == _flatten(wrapper)


def test_an_empty_parameter_schema_is_a_caller_saying_no_parameters():
    # Not a caller saying nothing: the annotations must not be read instead.
    @tool("…", parameters={})
    def anything(title: str):
        ...

    assert anything.spec.parameters == {}
    registry = ToolRegistry()
    registry.register(anything, "…", parameters={})
    assert registry.specs_to_declare() == []


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


def test_declaring_before_a_conversation_is_refused():
    wrapper, fake = _wrapper()
    registry = ToolRegistry()
    registry.register(lambda: {}, "…", name="clock", declare=True)

    # `begin` would reset the table and throw the declaration away, so writing
    # it here would be a silent loss rather than a declaration.
    with pytest.raises(ValueError, match="no conversation to declare into"):
        wrapper.declare_tools(registry.specs_to_declare())

    assert fake.cache == []


def test_a_model_without_tool_support_still_reports_the_missing_conversation():
    # The caller's own mistake is named before the model's limits are.
    fake = FakeContext(fragments=DEFAULT_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    spec = ToolSpec("clock", "…", {"type": "object", "properties": {}})
    with pytest.raises(ValueError, match="no conversation to declare into"):
        wrapper.declare_tools([spec])


def test_a_call_this_parser_cannot_read_still_closes_the_turn():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    # A body in another model's format, which this model's fragments cannot read.
    fake.call_script = ["\n<function=clock>\n<parameter=x>\n1\n</parameter>\n</function>\n"]

    list(wrapper.generate_events("wie spät?"))

    # Left open, nothing would ever close it and the next request would prefill
    # into a model turn that never ended.
    assert not wrapper.awaiting_tool_result
    assert fake.cache == _flatten(wrapper)


def test_an_unreadable_call_says_why():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["<function=clock>"]

    events = wrapper.generate_events("wie spät?")
    while True:
        try:
            next(events)
        except StopIteration as done:
            reply = done.value
            break

    assert reply.stop_reason == "unreadable_tool_call"
    assert reply.tool_calls == []


def test_a_reset_ends_the_turn_a_call_was_holding_open():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))
    assert wrapper.awaiting_tool_result

    wrapper.begin("neue persona")

    # The turn went with the table. Left waiting, the result below would be
    # appended to whatever message now sits last, which is the persona.
    assert not wrapper.awaiting_tool_result
    with pytest.raises(ValueError, match="no tool call is waiting"):
        wrapper.inject_tool_result("clock", {"time": "09:00"})
    assert [m["role"] for m in wrapper.snapshot()] == ["system"]
    assert fake.cache == _flatten(wrapper)


def test_reply_open_outlasts_the_calls_that_are_owed():
    """The window the two properties disagree in is the one that refuses work.

    Reading ``awaiting_tool_result`` as "may I start a new turn?" is wrong
    exactly here, which is why ``reply_open`` answers it instead.
    """
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))
    assert wrapper.awaiting_tool_result and wrapper.reply_open

    wrapper.inject_tool_result("clock", {"time": "09:00"})
    assert not wrapper.awaiting_tool_result     # nothing is owed any more
    assert wrapper.reply_open                   # but the turn has not closed
    for action in (lambda: wrapper.generate("weiter"),
                   lambda: wrapper.inject("kontext"),
                   lambda: wrapper.declare_tools(
                       [ToolSpec("x", "d", {"type": "object", "properties": {}})])):
        with pytest.raises(ValueError, match="while a reply is open"):
            action()

    list(wrapper.resume_events())
    assert not wrapper.reply_open               # closed, and work is allowed again
    wrapper.generate("weiter")


# ----- what the whole exchange reports back -------------------------------
def test_the_exchange_reports_what_the_reply_alone_cannot():
    """A reply that ended on an unreadable call reads as empty text.

    Only the ``stop_reason`` tells that apart from a model that said nothing,
    which is why the exchange carries one.
    """
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["nonsense with no name"]
    reply = wrapper.answer("los")
    assert reply.text == ""
    assert reply.stop_reason == "unreadable_tool_call"


def test_the_exchange_says_when_the_tool_ceiling_bound():
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "The time.", name="clock")
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"] * 6
    reply = wrapper.answer("los", registry, max_calls=2)
    assert reply.tool_limit_reached
    # Two ran, and the ceiling then withheld the call tokens rather than
    # letting a third be asked for.
    assert [c.name for c in reply.tool_calls] == ["clock", "clock"]
    assert reply.stop_reason == "end_of_turn"


def test_an_exchange_that_never_called_anything_says_so():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    reply = wrapper.answer("los")
    assert reply.tool_calls == []
    assert not reply.tool_limit_reached
    assert reply.stop_reason == "end_of_turn"


def test_a_ceiling_of_zero_does_not_bind_a_reply_that_asked_for_nothing():
    """The flag reports a call that was refused, not a budget that went unused."""
    wrapper, _ = _wrapper()
    wrapper.begin("persona")
    reply = wrapper.answer("los", max_calls=0)
    assert reply.tool_calls == []
    assert not reply.tool_limit_reached


def test_the_exchange_sums_every_generation_it_spanned():
    """The counts are sums, so a tool exchange costs more than its last generation."""
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "The time.", name="clock")
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    with_call = wrapper.answer("los", registry)

    wrapper2, fake2 = _wrapper()
    wrapper2.begin("persona")
    without = wrapper2.answer("los", registry)

    assert with_call.n_generated_tokens > without.n_generated_tokens
    assert with_call.n_prefilled_tokens >= without.n_prefilled_tokens


def test_the_streamed_deltas_join_into_the_exchange_text():
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "The time.", name="clock")
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    stream = wrapper.answer_stream("los", registry)
    deltas = []
    while True:
        try:
            deltas.append(next(stream))
        except StopIteration as ended:
            reply = ended.value
            break
    assert "".join(deltas) == reply.text


# ----- a call that names itself -------------------------------------------
ELEMENT = CallBody(name_close=">\n<", key_open="parameter=", key_close=">\n",
                   pair_sep="\n</parameter>\n<parameter=",
                   body_close="\n</parameter>\n</function>\n",
                   empty_close="/function>\n")


def test_a_json_call_needs_no_grammar_at_all():
    call, = parse_calls('{"name": "calendar_add", '
                      '"arguments": {"title": "Zahnarzt", "minutes": 30}}', None)
    assert call == ToolCall("calendar_add", {"title": "Zahnarzt", "minutes": 30})


@pytest.mark.parametrize("raw", [
    '{"name": "x", "parameters": {"a": 1}}',        # the other spelling
    '{"name": "x", "arguments": "{\\"a\\": 1}"}',   # arguments nested as text
])
def test_the_shapes_the_json_family_comes_in(raw):
    assert parse_calls(raw, None)[0].arguments == {"a": 1}


def test_without_a_grammar_anything_but_json_is_refused():
    with pytest.raises(ValueError, match="no call_body"):
        parse_calls("fn:clock{}", None, "fn:")


# ----- the same parser, an unlike model -----------------------------------
def _element_body(name, arguments):
    """What an element-shaped model generates between its call tags."""
    parts = [f"\n<function={name}>\n"]
    for key, value in arguments.items():
        parts.append(f"<parameter={key}>\n{value}\n</parameter>\n")
    parts.append("</function>\n")
    return "".join(parts)


@pytest.mark.parametrize("arguments", [
    {},
    {"title": "Zahnarzt"},
    {"title": "Zahnarzt", "minutes": 30},
])
def test_an_element_shaped_model_reads_with_the_same_parser(arguments):
    call, = parse_calls(_element_body("calendar_add", arguments), ELEMENT,
                      "\n<function=")
    assert call == ToolCall("calendar_add", arguments)


# ----- the shapes a model may write a call in -----------------------------
def _json_object(name, arguments):
    return json.dumps({"name": name, "arguments": arguments})


def _json_array(name, arguments):
    return json.dumps([{"name": name, "arguments": arguments}])


def _python_call(name, arguments):
    return f"{name}(" + ", ".join(
        f"{k}={json.dumps(v)}" for k, v in arguments.items()) + ")"


def _attribute_xml(name, arguments):
    body = "".join(f'<parameter name="{k}">{json.dumps(v)}</parameter>\n'
                   for k, v in arguments.items())
    return f'<invoke name="{name}">\n{body}</invoke>'


SHAPES = {
    "json object": _json_object,
    "json array": _json_array,
    "python call": _python_call,
    "attribute xml": _attribute_xml,
    "delimited": lambda n, a: f"call:{n}{{" + ",".join(
        f"{k}:" + (f"<|\"|>{v}<|\"|>" if isinstance(v, str) else json.dumps(v))
        for k, v in a.items()) + "}",
}

ARGUMENTS = [{}, {"title": "Zahnarzt"}, {"title": "Zahnarzt", "minutes": 30},
             {"where": {"city": "Bern"}}, {"days": [1, 2]},
             {"title": "a,b:c"}, {"flag": True}, {"nothing": None}]


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("arguments", ARGUMENTS)
def test_every_shape_round_trips_through_recovery_and_parsing(shape, arguments):
    """One model family's call shape survives recovery and parsing intact.

    Render a call the way that family does, recover the call body from the
    rendering, and read the call back equal to what went in.
    """
    emit = SHAPES[shape]

    def render(messages, tools=None):
        out = []
        for m in messages:
            calls = m.get("tool_calls", ())
            inner = "".join("<tc>" + emit(c["function"]["name"],
                                          c["function"]["arguments"]) + "</tc>"
                            for c in calls)
            out.append(f"<|turn>{m['role']}\n{inner}{m['content'].strip()}"
                       + ("" if calls else "<turn|>\n"))
        return "".join(out)

    tools = extract_fragments(render).tools
    call_body = extract_call_body(render, tools.call_open, "</tc>")
    lead_in = tools.call_open[len("<tc>"):]

    call, = parse_calls(emit("calendar_add", arguments), call_body, lead_in)
    assert call == ToolCall("calendar_add", arguments)


def test_a_grammar_may_be_supplied_where_the_probes_cannot_recover_one():
    # The same escape hatch `fragments=` is for a template that cannot be read.
    fake = FakeContext(fragments=TOOL_FRAGMENTS)
    spelled = CallBody(name_close="(", key_open="", key_close="=",
                       pair_sep=",", body_close=")", empty_close=")",
                       quote_open='"', quote_close='"')
    wrapper = ChatWrapper(Config(context_size=512), context=fake,
                          call_body=spelled)
    assert wrapper._call_body is spelled


# ----- a model that writes a batch -----------------------------------------
def test_a_json_array_yields_every_call_it_holds():
    calls = parse_calls('[{"name": "clock", "arguments": {}}, '
                        '{"name": "calendar_add", "arguments": {"title": "x"}}]',
                        None)
    assert calls == [ToolCall("clock", {}),
                     ToolCall("calendar_add", {"title": "x"})]


def test_an_array_holding_something_that_is_not_a_call_is_not_a_batch():
    # All or nothing: a half-read batch would silently drop work.
    with pytest.raises(ValueError):
        parse_calls('[{"name": "clock", "arguments": {}}, 42]', None)


def test_every_call_of_a_batch_owes_a_result_before_the_reply_resumes():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']

    asked = list(wrapper.generate_events("los"))
    assert asked == [ToolCall("clock", {}), ToolCall("other", {})]
    assert wrapper.awaiting_tool_result

    wrapper.inject_tool_result("clock", "a")
    # One down, one still owed: resuming now would leave a call unanswered.
    with pytest.raises(ValueError, match="1 call"):
        list(wrapper.resume_events())

    wrapper.inject_tool_result("other", "b")
    fake.call_script = []
    list(wrapper.resume_events())
    assert not wrapper.awaiting_tool_result
    assert fake.cache == _flatten(wrapper)


def test_answer_stream_answers_every_call_of_a_batch():
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    ran = []
    registry.register(lambda: ran.append("clock") or {}, "…", name="clock")
    registry.register(lambda: ran.append("other") or {}, "…", name="other")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']

    list(wrapper.answer_stream("los", registry))

    assert ran == ["clock", "other"]
    assert fake.cache == _flatten(wrapper)


def test_a_resumed_turn_gets_back_what_the_generation_prompt_carries():
    """A generation prompt holding more than the turn tag is restored on resume.

    The tag must not repeat, but the rest was never prefilled, so without this
    the model writes it and it reaches the caller as reply text.
    """
    class Channelled(TemplateFormatter):
        def resume_open(self) -> str:
            return "<chan>"

    fake = FakeContext(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=1024, evict_above_fraction=0.9),
                          context=fake, formatter=Channelled)
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))
    wrapper.inject_tool_result("clock", "ok")

    before = len(fake.cache)
    fake.call_script = []
    list(wrapper.resume_events())

    written = "".join(chr(c) for c in fake.cache[before:] if c >= 0)
    assert written.startswith("<chan>")
    assert fake.cache == _flatten(wrapper)


def test_a_model_needing_nothing_extra_gets_nothing():
    """The default ``resume_open`` is empty, so a resumed reply prefills nothing."""
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))
    wrapper.inject_tool_result("clock", "ok")

    before = len(fake.cache)
    fake.call_script = []
    list(wrapper.resume_events())

    # What the generation holds is the sampled tokens and the turn's closing
    # tag, and nothing in front of them.
    generated = fake._gen_len
    closing = len("<turn|>\n")
    assert len(fake.cache) - before == generated + closing


# ----- reading a body the model got wrong ---------------------------------
def test_an_empty_argument_value_reads_as_an_empty_string():
    """A key the model wrote no value for is an empty string, not a crash.

    The container test used to be ``text[:1] in "{["``, and an empty string is
    a substring of every string, so an empty value took the object branch and
    indexed past its own end.
    """
    assert parse_calls("fn:clock{city:}", FAKE, "fn:") == [
        ToolCall("clock", {"city": ""})]


def test_an_argument_with_no_separator_at_all_is_refused():
    with pytest.raises(ValueError, match="an argument with no value"):
        parse_calls("fn:clock{city}", FAKE, "fn:")


def test_a_batch_entry_that_is_not_a_call_voids_the_whole_batch():
    # All or nothing: half a batch would silently drop what the model asked for.
    with pytest.raises(ValueError):
        parse_calls('[{"name":"clock","arguments":{}}, 7]', None)


def test_a_batch_entry_with_no_usable_name_voids_the_whole_batch():
    with pytest.raises(ValueError):
        parse_calls('[{"name":"clock","arguments":{}}, {"arguments":{}}]', None)


def test_a_value_that_only_looks_like_a_json_string_is_read_as_written():
    # It opens and closes with a quote, so JSON is tried first; it is not valid
    # JSON, so the model's own quoting decides instead.
    assert parse_calls("""fn:say{text:"a"b"}""", FAKE, "fn:") == [
        ToolCall("say", {"text": '"a"b"'})]


def test_nested_arguments_that_are_not_json_are_not_a_call():
    # Some models nest the arguments as JSON text; text that is not JSON means
    # this was never a call worth acting on.
    with pytest.raises(ValueError):
        parse_calls('{"name":"clock","arguments":"not json"}', None)


def test_a_nested_object_written_in_the_model_s_own_delimiters():
    # The value is neither JSON nor quoted, so it is read with the same
    # delimiters again, one level down.
    assert parse_calls("fn:plan{where:{city:'Bonn',day:2}}", FAKE, "fn:") == [
        ToolCall("plan", {"where": {"city": "Bonn", "day": 2}})]


def test_a_list_value_written_in_the_model_s_own_delimiters():
    assert parse_calls("fn:plan{days:['Mo','Di',3]}", FAKE, "fn:") == [
        ToolCall("plan", {"days": ["Mo", "Di", 3]})]


def test_a_grammar_with_no_pair_separator_reads_one_argument():
    # A model that never writes two arguments has no separator to recover, and
    # the whole body is then a single pair.
    single = CallBody(name_close="(", key_open="", key_close="=", pair_sep="",
                      body_close=")", empty_close=")",
                      quote_open='"', quote_close='"')
    assert parse_calls('fn:clock(city="Bonn")', single, "fn:") == [
        ToolCall("clock", {"city": "Bonn"})]


# ----- registering --------------------------------------------------------
def test_register_takes_what_the_tool_declaration_already_said():
    @tool("Sagt die Zeit.")
    def clock(city: str) -> str:
        return "09:00"

    registry = ToolRegistry()
    registry.register(clock, declare=True)
    spec = registry.specs_to_declare()[0]
    assert spec.description == "Sagt die Zeit."
    assert spec.parameters["properties"] == {"city": {"type": "string"}}
    assert len(registry) == 1 and "clock" in registry
    assert registry.call(ToolCall("clock", {"city": "Bonn"})) == "09:00"


def test_register_without_a_description_or_a_declaration_is_refused():
    def bare(x: int):
        pass

    with pytest.raises(TypeError, match="needs a description"):
        ToolRegistry().register(bare)


def test_registering_a_name_again_replaces_it_and_its_declare_state():
    registry = ToolRegistry()
    registry.register(lambda: "first", "Erste.", name="clock", declare=True)
    assert registry.specs_to_declare()[0].description == "Erste."

    registry.register(lambda: "second", "Zweite.", name="clock")
    assert len(registry) == 1
    assert registry.specs_to_declare() == []          # declare went with it
    assert registry.call(ToolCall("clock", {})) == "second"


def test_variadic_parameters_are_left_out_of_the_schema():
    @tool("Nimmt beliebige Argumente.")
    def spread(city: str, *rest, **extra):
        return city

    assert spread.spec.parameters["properties"] == {"city": {"type": "string"}}
    assert spread.spec.parameters["required"] == ["city"]


def test_a_parser_that_fails_unexpectedly_still_closes_the_turn(monkeypatch):
    """A reply that stopped on a call must close, whatever reading it does.

    ``_read_calls`` runs with the turn still open. A failure escaping it would
    skip the closing tag while leaving ``_reply_open`` False, so nothing would
    ever close the turn and no guard could see that it was open.
    """
    import llama_chat.wrapper as wrapper_module

    def explode(*_a, **_k):
        raise IndexError("string index out of range")

    monkeypatch.setattr(wrapper_module, "parse_calls", explode)
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    reply = wrapper.generate("wie spät?")
    assert reply.stop_reason == "unreadable_tool_call"
    assert reply.tool_calls == []
    assert not wrapper.awaiting_tool_result

    # The turn really closed: an ordinary request follows it, and the cache
    # still equals the table.
    wrapper.generate("etwas anderes")
    assert [m["role"] for m in wrapper.snapshot()] == [
        "system", "user", "assistant", "user", "assistant"]
    assert fake.cache == _flatten(wrapper)


def test_the_registry_keeps_naming_a_declaration_the_cache_dropped():
    """``specs_to_declare`` is intent, not a record of what the model knows.

    A declaration prefilled mid-conversation is an ordinary message, so a long
    conversation evicts it. Nothing tells the registry, and nothing should: it
    answers what ought to be declared, which is what a later ``begin`` needs.
    """
    def clock(city: str) -> str:
        return "09:00"

    registry = ToolRegistry()
    registry.register(clock, "Sagt die Zeit.")

    fake = FakeContext(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=200, evict_above_fraction=0.5),
                          context=fake)
    wrapper.begin("persona")
    wrapper.declare_tools([registry.declare("clock")])
    assert [m["role"] for m in wrapper.snapshot()] == ["system", "system"]

    for i in range(6):
        wrapper.generate(f"frage {i}")

    # The declaration went with the oldest messages; only the prompt is pinned.
    assert [m["role"] for m in wrapper.snapshot()][:2] == ["system", "user"]
    # And the registry still names it, so another begin() would declare it again.
    assert [s.name for s in registry.specs_to_declare()] == ["clock"]
    assert fake.cache == _flatten(wrapper)


# ----- detecting a call as it is generated --------------------------------
# These drive the real KVContext.generate over a stubbed backend, because the
# fake context reimplements this contract and so cannot test it.
def _generating(pieces: dict, order: list, eog: int | None = None):
    """A KVContext that samples ``order`` and renders each token from ``pieces``."""
    from types import SimpleNamespace

    from llama_chat.context import KVContext

    sampled = iter(order)
    ctx = object.__new__(KVContext)
    ctx._backend = SimpleNamespace(
        llama_cpp=SimpleNamespace(llama_sampler_sample=lambda s, c, i: next(sampled)),
        is_eog=lambda vocab, token: token == eog,
        decode=lambda *_a: None,
        token_to_piece_bytes=lambda vocab, token: pieces[token])
    ctx._sampler = ctx._vocab = ctx._llama_context = None
    ctx._max_batch_tokens = 8
    return ctx


_PIECES = {0: b"hi ", 1: b"<call>", 2: b"fn:clock{}", 3: b"</call>", 4: b" done"}
_TAGS = (1, 3)


def test_a_call_is_recognised_by_its_token_and_never_reaches_the_reply():
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = _generating(_PIECES, [0, 1, 2, 3]).generate(
        0, 10, [], out=acc, tool_tokens=_TAGS)

    assert list(stream) == ["hi "]      # the call's own text is never yielded
    assert acc.stop_reason == "tool_call"
    assert acc.tool_call == "fn:clock{}"
    assert acc.text == "hi "
    assert acc.token_ids == [0, 1, 2, 3]   # yet all of it reached the cache


def test_the_held_back_tail_is_released_when_a_call_opens():
    """A stop string holds text back; the call is not what it was waiting for."""
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    # "ZZZZ" is longer than "hi ", so nothing can be emitted before the call.
    stream = _generating(_PIECES, [0, 1, 2, 3]).generate(
        0, 10, ["ZZZZ"], out=acc, tool_tokens=_TAGS)
    assert list(stream) == ["hi "]
    assert acc.text == "hi "            # nothing of the reply was lost


def test_closing_the_stream_as_a_call_opens_still_reads_as_interrupted():
    """The release above is a yield like any other, so barge-in counts there.

    It used to sit outside the ``GeneratorExit`` handler, and a consumer that
    closed exactly then saw ``stop_reason == "token_limit"``.
    """
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = _generating(_PIECES, [0, 1, 2, 3]).generate(
        0, 10, ["ZZZZ"], out=acc, tool_tokens=_TAGS)
    assert next(stream) == "hi "
    stream.close()
    assert acc.stop_reason == "interrupted"


def test_a_call_the_token_cap_cuts_short_is_dropped():
    """Truncation mid-call loses the call, not the reply, and closes normally."""
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = _generating(_PIECES, [0, 1, 2]).generate(
        0, 3, [], out=acc, tool_tokens=_TAGS)
    assert list(stream) == ["hi "]
    assert acc.stop_reason == "token_limit"
    assert acc.tool_call == ""          # half a call is no call
    assert acc.text == "hi "            # and none of it leaked into the reply


def test_without_tool_tokens_a_call_tag_is_ordinary_text():
    """A model that cannot call tools has its tags yielded like any other text."""
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = _generating(_PIECES, [0, 1, 2, 3]).generate(0, 4, [], out=acc)
    assert "".join(stream) == "hi <call>fn:clock{}</call>"
    assert acc.stop_reason == "token_limit"


# The two ways a stream ends that the fake context answers for itself. Pinned
# here for the same reason as the call detection above: only the real loop
# decides what reaches the cache before it stops.
def test_end_of_generation_stops_before_the_token_reaches_the_cache():
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = _generating(_PIECES, [0, 4], eog=4).generate(0, 10, [], out=acc)
    assert "".join(stream) == "hi "
    assert acc.stop_reason == "end_of_turn"
    assert acc.token_ids == [0]     # the end token is recognised, never decoded


def test_cancelling_ends_the_stream_within_one_token():
    import threading

    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    cancel = threading.Event()
    ctx = _generating(_PIECES, [0, 4, 4])
    # Set it while the first token renders; the loop checks before sampling the
    # next one, so exactly one token makes it in.
    render = ctx._backend.token_to_piece_bytes
    ctx._backend.token_to_piece_bytes = lambda v, t: (cancel.set(), render(v, t))[1]

    stream = ctx.generate(0, 10, [], out=acc, cancel=cancel)
    assert "".join(stream) == "hi "
    assert acc.stop_reason == "interrupted"
    assert acc.token_ids == [0]


def test_a_stop_string_ends_the_stream_and_trims_the_reply():
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    pieces = {0: b"hello ", 1: b"STOP", 2: b"never sampled"}
    stream = _generating(pieces, [0, 1, 2]).generate(0, 10, ["STOP"], out=acc)
    assert "".join(stream) == "hello "
    assert acc.stop_reason == "stop_string"
    # The stop string is trimmed from the reply but still reached the cache, so
    # the table can go on describing what the cache holds.
    assert acc.token_ids == [0, 1]


def test_the_held_back_tail_is_flushed_when_generation_runs_out():
    from llama_chat.context import GenerationAccumulator

    acc = GenerationAccumulator()
    # A stop string this long holds back three characters at every step, so
    # without the final flush the reply would end short.
    stream = _generating({0: b"abc", 1: b"def"}, [0, 1]).generate(
        0, 2, ["ZZZZ"], out=acc)
    assert "".join(stream) == "abcdef"
    assert acc.stop_reason == "token_limit"
    assert acc.text == "abcdef"


def test_a_resumed_reply_gathers_its_text_into_one_message():
    """One turn's text is what a reader saw, across both generations of it.

    ``on_message`` fires once, when the turn finally closes, so whatever
    persists the conversation is handed the resumed prose and not only what
    came before the call.
    """
    seen = []
    fake = FakeContext(fragments=TOOL_FRAGMENTS, gen_text="neun Uhr")
    wrapper = ChatWrapper(Config(context_size=512), context=fake,
                          on_message=lambda role, text: seen.append((role, text)))
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))
    assert wrapper.snapshot()[-1]["text"] == ""     # nothing visible yet

    wrapper.inject_tool_result("clock", {"time": "09:00"})
    fake.call_script = []
    resumed = "".join(e for e in wrapper.resume_events() if isinstance(e, str))

    row = wrapper.snapshot()[-1]
    assert resumed and row["text"] == resumed       # every visible delta, once
    assert "clock" not in row["text"]               # the call is not reply text
    assert "09:00" not in row["text"]               # nor is the result
    assert row["n_tokens"] > len(row["text"])       # yet both reached the cache
    assert ("assistant", resumed) in seen           # and that is what is handed on
    assert fake.cache == _flatten(wrapper)


# ----- what a model that cannot do this looks like ------------------------
def test_a_batch_past_the_ceiling_is_answered_rather_than_run():
    """With one call per round the ceiling withholds the call tokens instead.

    A model that asks for a batch can outrun the ceiling inside a single round,
    and the calls past it are answered with an error the model can read.
    """
    ran = []
    registry = ToolRegistry()
    registry.register(lambda: ran.append(1) or "09:00", "Zeit.", name="clock")

    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        ' {"name":"clock","arguments":{}},'
                        ' {"name":"clock","arguments":{}}]']
    list(wrapper.answer_stream("frage", registry, max_calls=1))

    assert len(ran) == 1                       # only the permitted one ran
    refusals = [text for text, _ in fake.tokenize_calls
                if "tool_limit_reached" in text]
    assert len(refusals) == 2                  # the other two were answered
    assert "the limit is 1" in refusals[0]     # and the count reads correctly
    assert fake.cache == _flatten(wrapper)


def test_tags_that_resolve_to_no_token_leave_tool_calling_off():
    """TemplateFragments can name tags this model's vocabulary does not have."""
    from llama_chat.template import TemplateFragments, ToolCallFragments

    unknown = TemplateFragments(
        system_prefix="<|turn>user\n", system_suffix="<turn|>\n",
        user_prefix="<|turn>user\n", user_suffix="<turn|>\n",
        assistant_prefix="<|turn>model\n", assistant_suffix="<turn|>\n",
        tools=ToolCallFragments(call_open="OPEN:", call_close=":CLOSE"))
    wrapper = ChatWrapper(Config(context_size=512),
                          context=FakeContext(fragments=unknown))
    assert not wrapper.can_call_tools


class _NoDeclarations(FakeContext):
    """A template that renders the same system turn whether tools are given."""

    def render_with_model_template(self, messages, tools=None):
        return super().render_with_model_template(messages, tools=None)


def test_a_template_that_declares_nothing_refuses_to_be_told_about_tools():
    fake = _NoDeclarations(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    spec = ToolSpec("clock", "Zeit.", {"type": "object", "properties": {}})

    # At begin, where the declarations would join the system turn.
    with pytest.raises(ValueError, match="declares no tools"):
        wrapper.begin("persona", tools=[spec])
    # And afterwards, where they would be a message of their own.
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="declares no tools"):
        wrapper.declare_tools([spec])


class _NoResultBlock(FakeContext):
    """A template that renders a call but has nowhere to put its result."""

    def render_with_model_template(self, messages, tools=None):
        kept = [m for m in messages if m["role"] != "tool"]
        return super().render_with_model_template(kept, tools=tools)


def test_a_template_with_no_result_block_refuses_the_result():
    fake = _NoResultBlock(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.generate_events("wie spät?"))

    with pytest.raises(ValueError, match="renders no tool result"):
        wrapper.inject_tool_result("clock", "ok")
    # Refused before anything was written, so the call is still owed.
    assert wrapper.awaiting_tool_result
    assert fake.cache == _flatten(wrapper)


class _UnsliceableDeclarations(FakeContext):
    """Declarations in a system turn that is rebuilt around them.

    The turn with tools shares neither its opening nor its whole ending with
    the turn without, so what the two differ by cannot be cut out of it.
    """

    def render_with_model_template(self, messages, tools=None):
        plain = super().render_with_model_template(messages, tools=None)
        if not tools:
            return plain
        return plain.replace("<|turn>", "<|TURN>")


def test_declarations_that_cannot_be_cut_out_are_refused():
    fake = _UnsliceableDeclarations(fragments=TOOL_FRAGMENTS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin("persona")
    spec = ToolSpec("clock", "Zeit.", {"type": "object", "properties": {}})

    # The two renders differ, so the template does declare something, but the
    # difference is not a block that can be lifted out and prefilled on its own.
    with pytest.raises(ValueError, match="declares no tools"):
        wrapper.declare_tools([spec])
    assert [m["role"] for m in wrapper.snapshot()] == ["system"]
    assert fake.cache == _flatten(wrapper)


def test_no_render_sentinel_contains_another():
    """A split on one has to land on that one, whatever else the render holds.

    ``PROBE`` used to sit inside ``RESULTPROBE`` and ``TAILPROBE``, so a render
    carrying all three could be cut six characters into the wrong sentinel.
    """
    from llama_chat import wrapper

    sentinels = {name: getattr(wrapper, name)
                 for name in ("_PROMPT", "_RESULT", "_TAIL")}
    assert len(set(sentinels.values())) == len(sentinels)
    for name, value in sentinels.items():
        others = [v for n, v in sentinels.items() if n != name]
        assert not any(value in other for other in others), \
            f"{name}={value!r} occurs inside another sentinel"


# ----- the registry the wrapper remembers ---------------------------------
def _weather_registry():
    registry = ToolRegistry()
    registry.register(lambda city: {"city": city, "celsius": 18},
                      "Wetter.", name="weather",
                      parameters={"type": "object",
                                  "properties": {"city": {"type": "string"}}})
    registry.declare("weather")
    return registry


def test_begin_keeps_the_registry_it_declared_from():
    """One registry, handed over once: what is declared is what can run."""
    registry = _weather_registry()
    wrapper, fake = _wrapper()
    wrapper.begin("persona", tools=registry)
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    assert "".join(wrapper.answer_stream("wetter?")) == "reply"
    assert fake.cache == _flatten(wrapper)


def test_answer_is_answer_stream_drained():
    registry = _weather_registry()
    wrapper, fake = _wrapper()
    wrapper.begin("persona", tools=registry)
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    assert wrapper.answer("wetter?").text == "reply"


def test_a_registry_passed_here_beats_the_remembered_one():
    other = ToolRegistry()
    other.register(lambda city: "aus der anderen", "Wetter.", name="weather",
                   parameters={"type": "object",
                               "properties": {"city": {"type": "string"}}})
    wrapper, fake = _wrapper()
    wrapper.begin("persona", tools=_weather_registry())
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    wrapper.answer("wetter?", other)
    assert any("aus der anderen" in text for text, _ in fake.tokenize_calls)


def test_a_conversation_with_no_tools_needs_no_registry():
    """The easy path is the easy path whether or not tools are involved."""
    wrapper, fake = _wrapper()
    wrapper.begin("persona")
    assert wrapper.answer("hallo").text == "reply"
    assert fake.cache == _flatten(wrapper)


def test_a_call_invented_with_no_tools_declared_is_answered_not_raised():
    """Raising here would leave the turn open with nothing able to close it."""
    wrapper, fake = _wrapper()
    wrapper.begin("persona")            # nothing declared, no registry
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    assert wrapper.answer("wetter?").text == "reply"     # the reply still arrives
    assert not wrapper._reply_open                  # and the turn closed
    assert any("no_tools" in text for text, _ in fake.tokenize_calls)
    assert fake.cache == _flatten(wrapper)


def test_begin_forgets_the_registry_of_the_conversation_it_replaced():
    ran = []
    registry = ToolRegistry()
    registry.register(lambda city: ran.append(1) or "09:00", "Wetter.",
                      name="weather",
                      parameters={"type": "object",
                                  "properties": {"city": {"type": "string"}}})
    registry.declare("weather")

    wrapper, fake = _wrapper()
    wrapper.begin("persona", tools=registry)
    wrapper.begin("persona")            # a reset drops it with everything else
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    wrapper.answer("wetter?")
    assert ran == []                    # the tool the old conversation had is gone
    assert any("no_tools" in text for text, _ in fake.tokenize_calls)
