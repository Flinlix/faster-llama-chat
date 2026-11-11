"""Declaring tools to a conversation, the turn a call holds open, and running the tools."""

import json
import re
import threading

import pytest

from llama_chat.addon_files import template_fingerprint
from llama_chat.config import Config
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tools import ToolCall, ToolRegistry, ToolResult, ToolSpec, ToolTurn
from llama_chat import ChatWrapper, ContextOverflowError
from tests.fake_context import FakeContext, ScriptedCall
from tests.invented_templates import (KEYLESS_CALLS, NO_DECLARATIONS, NO_RESULT_BLOCK, PLAIN,
                                      RESULT_LINES, TOOLS, TOOLS_DECLARED_IN_THE_NEWEST_TURN,
                                      TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN)
from tests.internals import held_tokens
from tests.wrappers import (assert_consistent, held_renders, held_text, newest_turn, system_turn,
                            tool_wrapper, weather_registry, rows)


def _result_of(fake):
    """The result block written into the newest turn, tags excluded."""
    turn = newest_turn(fake)
    return turn.split("<result>")[1].split("</result>")[0]


def _declared_in(fake, index):
    """The text of the system turn (``index`` 0) or of the newest user turn on, as the cache holds it."""
    return system_turn(fake) if index == 0 else newest_turn(fake)


def _moving(**fake_kw):
    fake = FakeContext(TOOLS_DECLARED_IN_THE_NEWEST_TURN, **fake_kw)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    return w, fake


def _as_rendered_with(w, fake, specs):
    """The cache equals the template's render of the conversation, ``specs`` declared."""
    assert_consistent(w, fake)
    rendered = held_renders(fake, [{"role": r["role"], "content": r["text"]} for r in rows(w)],
                            tools=[spec.as_declaration() for spec in specs])
    assert fake.text_of(fake.cache) in rendered


# ----- declaring -----------------------------------------------------------
def test_the_declaration_reaches_the_system_turn():
    wrapper, fake = tool_wrapper()

    clock = ToolSpec(name="clock", description="Sagt die Zeit.",
                     parameters={"type": "object", "properties": {}, "required": []})
    wrapper.begin("persona", tools=[clock])

    # The invented template renders a declaration as <decl>NAME</decl>, and the
    # 1:1 char tokenizer lets the system turn be read straight back.
    written = system_turn(fake)
    assert "<decl>clock</decl>" in written
    assert "persona" in written


def test_a_declaration_is_tokenized_as_tags_and_the_prompt_is_not():
    wrapper, fake = tool_wrapper()

    clock = ToolSpec(name="clock", description="Sagt die Zeit.",
                     parameters={"type": "object", "properties": {}, "required": []})
    wrapper.begin("persona", tools=[clock])

    # The prompt is a caller's text and must not be able to forge a tag; the
    # declarations are the model's own and have to parse as tags.
    assert any("persona" in text and not special for text, special in fake.tokenize_calls)
    assert {("<decl>", True), ("</decl>", True)} <= set(fake.tokenize_calls)


# ----- the turn a call holds open ------------------------------------------
def test_a_call_ends_the_stream_and_holds_the_turn_open():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    events = list(wrapper.reply("wie spät?", tool_call_policy="yield"))

    assert events == [ToolCall(name="clock", arguments={})]
    assert bool(wrapper.pending_calls)
    # Nothing of the call reached the reply text.
    assert rows(wrapper)[-1]["text"] == ""


def test_a_call_refused_without_tools_still_lets_the_reply_go_on():
    """The model is told there are no tools, not left with its call ignored.

    Dropping the call would leave a turn holding a call and no result, which no
    template produces and the model reads back on every later turn, and would
    hand the caller silence where it asked a question.
    """
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    reply = wrapper.reply("wie spät?").finish()

    assert reply.text == "reply"                 # it answered in words
    assert reply.tool_calls == [ToolCall(name="clock", arguments={})]
    assert not wrapper.turn_open
    # The turn holds the call, an error result for it, and the reply after.
    written = rows(wrapper)[-1]["text"]
    assert written == "reply"
    assert fake.cache == held_tokens(wrapper)
    # And the conversation goes on, which is what the result bought.
    assert wrapper.reply("und sonst?").finish().stop_reason == "end_of_turn"


def test_the_refusal_reaches_the_model_as_a_result_block():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    wrapper.reply("wie spät?").finish()

    turn = newest_turn(fake)
    assert "<result>clock(" in turn and "tools_not_run" in turn


def test_the_refusal_does_not_deny_a_tool_the_model_was_told_about():
    """The system turn declared it; saying it does not exist contradicts that.

    `reply(tool_call_policy="refuse")` runs no tool either way, so the refusal says that rather than
    denying the tool.
    """
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    wrapper.reply("wie spät?").finish()

    declared = system_turn(fake)
    turn = newest_turn(fake)
    assert "<decl>clock</decl>" in declared      # the model was told it exists
    assert "no tools" not in turn                # so it is not told otherwise
    assert "tools_not_run" in turn


def test_a_refusing_reply_runs_none_even_of_a_declared_tool():
    """The tool-free entry point runs nothing; a registry runs under the other."""
    registry = ToolRegistry()
    ran = []
    registry.register(lambda: ran.append("ran") or "12:00", "Die Zeit.",
                      name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    wrapper.reply("wie spät?").finish()
    assert ran == []

    fake.call_script = ["fn:clock{}"]
    wrapper.reply("wie spät?", tool_call_policy=registry).finish()
    assert ran == ["ran"]


def test_the_reported_calls_are_not_the_wrappers_own_list():
    """A Reply that aliased ``_pending_calls`` would empty as results arrive."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    events = wrapper.reply("wie spät?", tool_call_policy="yield")
    reply = None
    while True:
        try:
            next(events)
        except StopIteration as done:
            reply = done.value
            break
    wrapper.inject_tool_result(ToolCall("clock", {}), {"time": "09:00"})

    assert not bool(wrapper.pending_calls)          # the wrapper's list emptied
    assert [c.name for c in reply.tool_calls] == ["clock"]   # the Reply's did not


def test_a_result_resumes_the_same_turn():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))

    wrapper.inject_tool_result(ToolCall("clock", {}), {"time": "09:00"})
    fake.call_script = []
    events = list(wrapper.reply(tool_call_policy="yield"))

    assert "".join(e for e in events if isinstance(e, str))
    assert not bool(wrapper.pending_calls)
    # One assistant message for the whole exchange.
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]


def test_two_calls_in_a_row_stay_inside_one_turn():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:first{}"]
    assert list(wrapper.reply("los", tool_call_policy="yield")) == [ToolCall("first", {})]

    wrapper.inject_tool_result(ToolCall("first", {}), "ok")
    fake.call_script = ["fn:second{}"]
    assert list(wrapper.reply(tool_call_policy="yield")) == [ToolCall("second", {})]

    wrapper.inject_tool_result(ToolCall("second", {}), "ok")
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))

    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]


def test_an_open_turn_outgrows_the_exchange_limit_and_is_trimmed_on_close():
    """The two rules meet here: nothing is evicted while a turn is open.

    A turn holding a call would be cut into by an eviction, so the exchange
    limit is allowed to be exceeded until the turn closes, and enforced again
    the moment it does.
    """
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=4000, max_history_exchanges=1),
                          context=fake)
    wrapper.begin("persona", [("user", "alt"), ("assistant", "älter")])
    fake.call_script = ["fn:first{}"]
    assert list(wrapper.reply("los", tool_call_policy="yield")) == [ToolCall("first", {})]
    assert len(rows(wrapper)) - 1 == 4     # two exchanges, over the limit, turn still open

    wrapper.inject_tool_result(ToolCall("first", {}), "ok")
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))
    assert len(rows(wrapper)) - 1 == 2     # the newest exchange alone once closed
    assert rows(wrapper)[0]["role"] == "system"


def test_a_result_under_the_wrong_name_is_refused():
    """Results are owed in call order, so only the oldest call owed is answered."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ['fn:[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']
    list(wrapper.reply("los", tool_call_policy="yield"))

    with pytest.raises(ValueError, match=re.escape("the waiting call is ToolCall(name='clock'")):
        wrapper.inject_tool_result(ToolCall("other", {}), "b")

    # The refusal consumed nothing: both calls are still owed, in order.
    wrapper.inject_tool_result(ToolCall("clock", {}), "a")
    wrapper.inject_tool_result(ToolCall("other", {}), "b")
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))


@pytest.mark.parametrize("not_the_call", ["clock", ToolCall("clock", {"city": "Bonn"})])
def test_a_result_answers_the_call_itself_not_its_name(not_the_call):
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    before = list(fake.cache)

    with pytest.raises(ValueError, match="the waiting call is"):
        wrapper.inject_tool_result(not_the_call, "ok")

    assert fake.cache == before
    assert wrapper.pending_calls == [ToolCall("clock", {})]


def test_a_result_whose_wrapping_cannot_fit_is_refused_whole():
    """The tags are never clipped, so a turn with no room for them is refused.

    Clipping them would leave the cache holding half a block no template ever
    produces, which is worse than a turn that cannot be continued.
    """
    for size in range(40, 400):
        wrapper, fake = tool_wrapper(context_size=size)
        try:
            wrapper.begin("persona")
            fake.call_script = ["fn:clock{}"]
            list(wrapper.reply("wie spät?", tool_call_policy="yield"))
        except ContextOverflowError:
            continue                    # too small for the call itself
        before = list(fake.cache)
        try:
            wrapper.inject_tool_result(ToolCall("clock", {}), "ok")
        except ContextOverflowError as refused:
            assert "result wrapping alone" in str(refused)
            # Nothing reached the cache, and the call is still owed.
            assert fake.cache == before
            assert wrapper.pending_calls == [ToolCall("clock", {})]
            return
    pytest.fail("no context size left the call room but its result wrapping none")


def test_a_result_with_no_call_waiting_is_refused():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="no tool call is waiting"):
        wrapper.inject_tool_result(ToolCall("clock", {}), "ok")


def test_going_on_without_a_request_and_nothing_to_answer_is_refused():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="nothing to answer"):
        list(wrapper.reply(tool_call_policy="yield"))


def test_tags_without_a_readable_grammar_cannot_call_tools():
    """The tags resolve, so a call is still detected; nothing runnable comes out.

    Every call such a model makes is unreadable, so it is told so and answers in
    words. Reporting it as callable would promise a result that never arrives.
    """
    fake = FakeContext(KEYLESS_CALLS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    template = ResolvedTemplate(FakeContext(KEYLESS_CALLS))
    assert template.call_boundary().tool_tokens is not None  # the tags resolved
    with pytest.raises(ValueError, match="call format could not be recovered"):
        template.ensure_tools_usable()                        # the probes recovered nothing
    assert not wrapper.can_call_tools

    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    reply = wrapper.reply("los").finish()
    assert reply.text == "reply"                 # it answered instead
    assert not wrapper.turn_open


def test_an_oversized_result_is_clipped_and_keeps_its_tags():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))

    before = len(fake.cache)
    wrapper.inject_tool_result(ToolCall("clock", {}), "x" * 10_000)

    # The turn still fits, and the block still ends where the template ends one.
    assert len(fake.cache) <= wrapper.config.context_size
    written = fake.text_of(fake.cache[before:])
    assert written.startswith("<result>clock(")
    assert written.endswith(")</result>")


# ----- the local loop ------------------------------------------------------
def test_a_running_reply_runs_the_tool_and_yields_only_prose():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    seen = []
    registry.register(lambda: seen.append("ran") or {"time": "09:00"},
                      "Sagt die Zeit.", name="clock")
    fake.call_script = ["fn:clock{}"]

    out = "".join(wrapper.reply("wie spät?", tool_call_policy=registry))

    assert seen == ["ran"]
    assert out                        # the reply after the result
    assert not bool(wrapper.pending_calls)


@pytest.mark.parametrize("before_call, after_call, joined", [
    ("Ich sehe nach.", "Es ist neun.", "Ich sehe nach.\n\nEs ist neun."),
    ("Ich sehe nach. ", "Es ist neun.", "Ich sehe nach. Es ist neun."),
    ("Ich sehe nach.", " Es ist neun.", "Ich sehe nach. Es ist neun."),
    ("", "Es ist neun.", "Es ist neun."),
])
def test_the_texts_around_a_call_are_parted_by_a_blank_line_where_they_would_join(
        before_call, after_call, joined):
    fake = FakeContext(TOOLS, gen_text=after_call, gen_tokens=len(after_call))
    wrapper = ChatWrapper(Config(context_size=512, evict_above_fraction=0.9), context=fake)
    wrapper.begin("persona")
    registry = ToolRegistry()
    registry.register(lambda: {"time": "09:00"}, "Sagt die Zeit.", name="clock")
    fake.replies = [ScriptedCall("fn:clock{}", before=before_call)]

    stream = wrapper.reply("wie spät?", tool_call_policy=registry)
    shown = "".join(stream)

    assert shown == stream.reply.text == joined


def test_a_running_reply_stops_calling_at_the_ceiling_and_closes_the_turn():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    runs = []
    registry.register(lambda: runs.append(1) or {}, "…", name="clock")
    # The fake would call forever; the ceiling is what has to stop it.
    fake.call_script = ["fn:clock{}"] * 10

    list(wrapper.reply("los", tool_call_policy=registry, max_calls=2))

    assert len(runs) == 2
    assert not bool(wrapper.pending_calls)
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]


def test_a_declaration_added_later_reaches_the_model():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    registry.register(lambda: {}, "Sagt die Zeit.", name="clock")

    wrapper.declare_tools(registry.specs())

    written = fake.text_of(fake.cache)
    assert "<decl>clock</decl>" in written


def test_two_tools_of_one_name_in_one_declaration_are_refused():
    """The model could call only one of them, and which one runs is the caller's guess."""
    wrapper, fake = tool_wrapper()
    first = ToolSpec("clock", "Sagt die Zeit.", {"type": "object", "properties": {}})
    second = ToolSpec("clock", "Sagt die Zeit in Bonn.", {"type": "object", "properties": {}})
    with pytest.raises(ValueError, match="'clock' twice"):
        wrapper.begin("persona", tools=[first, second])
    assert fake.cache == []

    wrapper.begin("persona")
    before = list(fake.cache)
    with pytest.raises(ValueError, match="'clock' twice"):
        wrapper.declare_tools([first, second])
    assert fake.cache == before and wrapper.declare_tools([first]) == 0
    assert held_text(fake).count("<decl>clock</decl>") == 1


def test_declaring_nothing_writes_nothing():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    before = list(fake.cache)
    assert wrapper.declare_tools([]) == 0
    assert fake.cache == before


def test_declarations_too_large_to_fit_are_refused_rather_than_truncated():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    before = list(fake.cache)
    registry = ToolRegistry()
    # The invented template renders one <decl>NAME</decl> per tool, so many long names
    # overrun the threshold.
    for i in range(200):
        registry.register(lambda: {}, "…", name=f"tool_with_a_long_name_{i}")

    with pytest.raises(ValueError, match="declare fewer tools"):
        wrapper.declare_tools(registry.specs())

    # A refused declaration leaves the conversation exactly as it was.
    assert fake.cache == before


def test_an_empty_parameter_schema_is_a_caller_saying_no_parameters():
    # Not a caller saying nothing: the annotations must not be read instead.
    def anything(title: str):
        ...

    registry = ToolRegistry()
    registry.register(anything, "…", parameters={})
    assert registry.specs()[0].parameters == {}


def test_declaring_before_a_conversation_is_refused():
    wrapper, fake = tool_wrapper()
    registry = ToolRegistry()
    registry.register(lambda: {}, "…", name="clock")

    # `begin` would reset the conversation and throw the declaration away, so writing
    # it here would be a silent loss rather than a declaration.
    with pytest.raises(ValueError, match="no conversation to declare into"):
        wrapper.declare_tools(registry.specs())

    assert fake.cache == []


def test_a_model_without_tool_support_still_reports_the_missing_conversation():
    # The caller's own mistake is named before the model's limits are.
    fake = FakeContext(PLAIN)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    spec = ToolSpec("clock", "…", {"type": "object", "properties": {}})
    with pytest.raises(ValueError, match="no conversation to declare into"):
        wrapper.declare_tools([spec])


def test_a_call_this_parser_cannot_read_still_closes_the_turn():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    # A body in another model's format, which this model's fragments cannot read.
    fake.call_script = ["fn:\n<function=clock>\n<parameter=x>\n1\n</parameter>\n</function>\n"]

    list(wrapper.reply("wie spät?", tool_call_policy="yield"))

    # Left open, nothing would ever close it and the next request would prefill
    # into a model turn that never ended.
    assert not bool(wrapper.pending_calls)


def test_an_unreadable_call_says_why():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:<function=clock>"]

    events = wrapper.reply("wie spät?", tool_call_policy="yield")
    while True:
        try:
            next(events)
        except StopIteration as done:
            reply = done.value
            break

    assert reply.stop_reason == "unreadable_tool_call"
    assert reply.tool_calls == []


def test_a_reset_ends_the_turn_a_call_was_holding_open():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    assert bool(wrapper.pending_calls)

    wrapper.begin("neue persona")

    # The turn went with the conversation. Left waiting, the result below would be
    # appended to whatever message now sits last, which is the persona.
    assert not bool(wrapper.pending_calls)
    with pytest.raises(ValueError, match="no tool call is waiting"):
        wrapper.inject_tool_result(ToolCall("clock", {}), {"time": "09:00"})
    assert [m["role"] for m in rows(wrapper)] == ["system"]
    wrapper.reply("etwas anderes").finish()
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]


def test_turn_open_outlasts_the_pending_calls():
    """The window the two properties disagree in is the one that refuses work.

    Reading ``pending_calls`` as "may I start a new turn?" is wrong
    exactly here, which is why ``turn_open`` answers it instead.
    """
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    assert bool(wrapper.pending_calls) and wrapper.turn_open

    wrapper.inject_tool_result(ToolCall("clock", {}), {"time": "09:00"})
    assert not bool(wrapper.pending_calls)     # nothing is owed any more
    assert wrapper.turn_open                   # but the turn has not closed
    for action in (lambda: wrapper.reply("weiter").finish(),
                   lambda: wrapper.inject("kontext"),
                   lambda: wrapper.declare_tools(
                       [ToolSpec("x", "d", {"type": "object", "properties": {}})])):
        with pytest.raises(ValueError, match="while a reply is open"):
            action()

    list(wrapper.reply(tool_call_policy="yield"))
    assert not wrapper.turn_open               # closed, and work is allowed again
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]
    assert fake.cache == held_tokens(wrapper)
    wrapper.reply("weiter").finish()


# ----- what the whole exchange reports back --------------------------------
def test_an_unreadable_call_is_told_so_and_the_reply_goes_on():
    """The model is told its call could not be read, and answers in words.

    Dropping it would leave a call with no result in the history and hand the
    caller silence, which is what the readable refusal already avoids.
    """
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"]

    reply = wrapper.reply("los", tool_call_policy="refuse").finish()

    assert reply.text == "reply"
    assert not wrapper.turn_open
    turn = newest_turn(fake)
    assert "unreadable_call" in turn


def test_a_failed_call_is_named_as_one_and_told_to_try_again():
    """The reason alone reads as data; the model has to know the call failed."""
    registry = ToolRegistry()
    registry.register(lambda: 1 / 0, "Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=4).finish()

    told = _result_of(fake)
    assert "the tool call failed" in told
    assert "ZeroDivisionError: division by zero" in told   # the reason survives
    assert "try again" in told


def test_an_unreadable_call_may_be_written_again():
    """A malformed call is a mistake the model can correct, so let it.

    Bounded by ``max_calls`` like any other failure, or a model whose format was
    never recovered would retry for ever: nothing it writes can ever be read.
    """
    ran = []
    registry = ToolRegistry()
    registry.register(lambda: ran.append(1) or "12:00", "Die Zeit.", name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name", "fn:clock{}"]

    reply = wrapper.reply("wie spät?", tool_call_policy=registry).finish()

    assert ran == [1]                                # the second one ran
    assert [c.name for c in reply.tool_calls] == ["clock"]
    assert not wrapper.turn_open


def test_retrying_an_unreadable_call_is_bounded_by_the_ceiling():
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 20

    wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=3).finish()

    assert len(fake.call_script) == 16               # three tries, then one left as text
    assert not wrapper.turn_open


def test_a_refusing_reply_does_not_retry_an_unreadable_call():
    """A retry could only be refused, this path running no tool either way."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 5

    wrapper.reply("wie spät?").finish()

    assert len(fake.call_script) == 3                # one try, then one left as text
    assert not wrapper.turn_open


def test_a_ceiling_of_zero_binds_an_unreadable_call_too():
    """The budget is spent only while there is budget, or it goes negative.

    ``tool_limit_reached`` compares it to zero, so overshooting reports that the
    ceiling did not bind when it is the only reason nothing was retried.
    """
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 5

    reply = wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=0).finish()

    assert reply.tool_limit_reached
    assert len(fake.call_script) == 3                # no retry: the next is left as text
    assert not wrapper.turn_open


def test_the_instruction_matches_what_the_model_may_still_do():
    """The same failure ends differently once no call is left to make.

    Telling a model to try again where the tags are about to be withheld would
    ask for the one thing it cannot do.
    """
    registry = ToolRegistry()
    registry.register(lambda: 1 / 0, "Die Zeit.", name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    wrapper.reply("q", tool_call_policy=registry, max_calls=1).finish()   # the only call there was
    assert "do not call again" in _result_of(fake)

    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    wrapper.reply("q", tool_call_policy=registry, max_calls=2).finish()          # one still to spend
    assert "try again" in _result_of(fake)


def test_a_result_that_is_not_a_failure_reaches_the_model_untouched():
    registry = ToolRegistry()
    registry.register(lambda: {"time": "12:00"}, "Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    wrapper.reply("wie spät?", tool_call_policy=registry).finish()

    told = _result_of(fake)
    assert '{"time": "12:00"}' in told
    assert "failed" not in told


def test_the_exchange_says_when_the_tool_ceiling_bound():
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "The time.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"] * 6
    reply = wrapper.reply("los", tool_call_policy=registry, max_calls=2).finish()
    assert reply.tool_limit_reached
    # Two ran, and the ceiling then withheld the call tokens rather than
    # letting a third be asked for.
    assert [c.name for c in reply.tool_calls] == ["clock", "clock"]
    assert reply.stop_reason == "end_of_turn"


def test_an_exchange_that_never_called_anything_says_so():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    reply = wrapper.reply("los", tool_call_policy=ToolRegistry()).finish()
    assert reply.tool_calls == []
    assert not reply.tool_limit_reached
    assert reply.stop_reason == "end_of_turn"


def test_a_ceiling_of_zero_does_not_bind_a_reply_that_asked_for_nothing():
    """The flag reports a call that was refused, not a budget that went unused."""
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    reply = wrapper.reply("los", tool_call_policy=ToolRegistry(), max_calls=0).finish()
    assert reply.tool_calls == []
    assert not reply.tool_limit_reached


def test_the_exchange_sums_every_generation_it_spanned():
    """The counts are sums, so a tool exchange costs more than its last generation."""
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "The time.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    with_call = wrapper.reply("los", tool_call_policy=registry).finish()

    wrapper2, fake2 = tool_wrapper()
    wrapper2.begin("persona")
    without = wrapper2.reply("los", tool_call_policy=registry).finish()

    assert with_call.n_generated_tokens > without.n_generated_tokens
    assert with_call.n_prefilled_tokens >= without.n_prefilled_tokens


def test_the_streamed_deltas_join_into_the_exchange_text():
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "The time.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    stream = wrapper.reply("los", tool_call_policy=registry)
    deltas = []
    while True:
        try:
            deltas.append(next(stream))
        except StopIteration as ended:
            reply = ended.value
            break
    assert "".join(deltas) == reply.text


# ----- a model that writes a batch -----------------------------------------
def test_every_call_of_a_batch_owes_a_result_before_the_reply_resumes():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ['fn:[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']

    asked = list(wrapper.reply("los", tool_call_policy="yield"))
    assert asked == [ToolCall("clock", {}), ToolCall("other", {})]
    assert bool(wrapper.pending_calls)

    wrapper.inject_tool_result(ToolCall("clock", {}), "a")
    # One down, one still owed: resuming now would leave a call unanswered.
    with pytest.raises(ValueError, match="1 call"):
        list(wrapper.reply(tool_call_policy="yield"))

    wrapper.inject_tool_result(ToolCall("other", {}), "b")
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))
    assert not bool(wrapper.pending_calls)


def test_the_calls_owed_a_result_are_read_from_the_wrapper_oldest_first():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ['fn:[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{"city":"Bonn"}}]']
    asked = list(wrapper.reply("los", tool_call_policy="yield"))
    assert wrapper.pending_calls == asked

    wrapper.pending_calls[1].arguments["city"] = "Rom"          # a copy, the caller's to change
    wrapper.inject_tool_result(asked[0], "a")
    assert wrapper.pending_calls == [ToolCall("other", {"city": "Bonn"})]
    wrapper.inject_tool_result(asked[1], "b")
    assert wrapper.pending_calls == [] and wrapper.turn_open


def test_a_history_owes_the_calls_of_its_newest_turn_that_no_result_answers():
    wrapper, _ = tool_wrapper()
    clock, other = ToolCall("clock", {}), ToolCall("other", {})
    wrapper.begin("persona", [("user", "los"), ToolTurn((clock, other))])
    assert wrapper.pending_calls == [clock, other]
    wrapper.begin("persona", [("user", "los"), ToolTurn((clock, other, ToolResult("clock", "a")))])
    assert wrapper.pending_calls == [other]


def test_the_pending_calls_are_read_without_waiting_for_a_running_reply():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    stream = wrapper.reply("hallo")
    next(stream)                                # the reply holds the wrapper's lock
    read = []
    reader = threading.Thread(target=lambda: read.append(wrapper.pending_calls), daemon=True)
    reader.start()
    reader.join(timeout=5)
    stream.close()
    assert read == [[]]


def test_a_running_reply_answers_every_call_of_a_batch():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    ran = []
    registry.register(lambda: ran.append("clock") or {}, "…", name="clock")
    registry.register(lambda: ran.append("other") or {}, "…", name="other")
    fake.call_script = ['fn:[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']

    list(wrapper.reply("los", tool_call_policy=registry))

    assert ran == ["clock", "other"]


def test_the_results_of_a_batch_are_written_as_the_template_writes_a_batch():
    fake = FakeContext(RESULT_LINES)
    wrapper = ChatWrapper(Config(context_size=1024, evict_above_fraction=0.9),
                          context=fake)
    wrapper.begin("persona")
    fake.call_script = ['fn:[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']
    list(wrapper.reply("los", tool_call_policy="yield"))
    before = len(fake.cache)
    wrapper.inject_tool_result(ToolCall("clock", {}), "a")
    wrapper.inject_tool_result(ToolCall("other", {}), "b")
    written = fake.text_of(fake.cache[before:])
    assert written == "<result>clock(a)</result>\n<result>other(b)</result><done>"
    list(wrapper.reply(tool_call_policy="yield"))


def test_a_model_needing_nothing_extra_gets_nothing():
    """A model whose replies open with the turn tag alone resumes with nothing."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    wrapper.inject_tool_result(ToolCall("clock", {}), "ok")

    before = len(fake.cache)
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))

    # What the generation holds is the sampled tokens, then the turn's closing
    # tag and the next question's opening, decoded ahead, and nothing in front.
    generated = fake.gen_tokens
    closing = len(fake.tokenize("<turn]>\n<[turn>"))
    assert len(fake.cache) - before == generated + closing


# ----- a parser that fails, and a later declaration ------------------------
def test_a_declaration_written_later_joins_the_system_turn_and_outlives_eviction():
    def clock(city: str) -> str:
        return "09:00"

    registry = ToolRegistry()
    registry.register(clock, "Sagt die Zeit.")

    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=200, evict_above_fraction=0.5),
                          context=fake)
    wrapper.begin("persona")
    wrapper.declare_tools(registry.specs())
    assert [m["role"] for m in rows(wrapper)] == ["system"]

    for i in range(6):
        wrapper.reply(f"frage {i}").finish()

    # The system turn is pinned, so the declaration in it is never evicted.
    assert "<decl>clock</decl>" in _declared_in(fake, 0)
    assert wrapper.declare_tools(registry.specs()) == 0


# ----- what a model that cannot do this looks like -------------------------
def test_a_batch_past_the_ceiling_is_answered_rather_than_run():
    """With one call per round the ceiling withholds the call tokens instead.

    A model that asks for a batch can outrun the ceiling inside a single round,
    and the calls past it are answered with an error the model can read.
    """
    ran = []
    registry = ToolRegistry()
    registry.register(lambda: ran.append(1) or "09:00", "Zeit.", name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ['fn:[{"name":"clock","arguments":{}},'
                        ' {"name":"clock","arguments":{}},'
                        ' {"name":"clock","arguments":{}}]']
    list(wrapper.reply("frage", tool_call_policy=registry, max_calls=1))

    assert len(ran) == 1                       # only the permitted one ran
    written = [step.result for step in wrapper.conversation()[1][-1].steps
               if isinstance(step, ToolResult)]
    refusals = [str(result) for result in written if "tool_limit_reached" in str(result)]
    assert len(refusals) == 2                  # the other two were answered
    assert "the limit is 1" in refusals[0]     # and the count reads correctly


def test_an_add_on_files_call_format_gives_no_calls_to_a_template_that_writes_none(tmp_path):
    """An add-on file says how a call is written only where the template writes one."""
    fake = FakeContext(PLAIN)
    (tmp_path / "model.json").write_text(json.dumps(
        {"templates": [template_fingerprint(PLAIN["chat_template"])],
         "call": {"opening": "OPEN:", "name_close": "(", "closing": ":CLOSE"}}))
    wrapper = ChatWrapper(Config(model_path=str(tmp_path / "model.gguf"), context_size=512),
                          context=fake)
    assert not wrapper.can_call_tools


def test_a_template_that_declares_nothing_refuses_to_be_told_about_tools():
    fake = FakeContext(NO_DECLARATIONS)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    spec = ToolSpec("clock", "Zeit.", {"type": "object", "properties": {}})

    # At begin, where the declarations would join the system turn.
    with pytest.raises(ValueError, match="declares no tools"):
        wrapper.begin("persona", tools=[spec])
    # And afterwards, where they would be a message of their own.
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="declares no tools"):
        wrapper.declare_tools([spec])


def test_recording_a_reply_survives_a_template_with_nowhere_to_tell_it():
    """Telling the model is best effort; keeping the cache consistent is not.

    Recording runs in a ``finally`` and is what leaves the cache and the conversation
    agreeing, so a template that renders no result block must make it fall back
    to closing the turn, not raise out of it.
    """
    fake = FakeContext(NO_RESULT_BLOCK)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"]

    reply = wrapper.reply("wie spät?").finish()          # must not raise

    assert reply.stop_reason == "unreadable_tool_call"
    assert not wrapper.turn_open
    turn = newest_turn(fake)
    assert "</call><turn]>\n" in turn             # the terminator still landed
    assert fake.cache == held_tokens(wrapper)
    assert wrapper.reply("weiter").finish().stop_reason == "end_of_turn"


@pytest.mark.parametrize("build, can_call", [
    (lambda: ChatWrapper(Config(context_size=1024), context=FakeContext(PLAIN)), False),
    (lambda: ChatWrapper(Config(context_size=1024),
                         context=FakeContext(NO_DECLARATIONS)), False),
    (lambda: ChatWrapper(Config(context_size=1024),
                         context=FakeContext(KEYLESS_CALLS)), False),
    (lambda: ChatWrapper(Config(context_size=1024),
                         context=FakeContext(NO_RESULT_BLOCK)), False),
    (lambda: ChatWrapper(Config(context_size=1024),
                         context=FakeContext(TOOLS)), True),
], ids=["no-call-tags", "declares-nothing", "unreadable-call", "no-result-block", "calls"])
def test_can_call_tools_is_exactly_what_declaring_will_accept(build, can_call):
    """One question, one answer: reading it at startup must predict what begin does.

    Two predicates over the same four steps would drift, and the property is
    read precisely to avoid the raise.
    """
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")
    wrapper = build()

    assert wrapper.can_call_tools is can_call
    if can_call:
        wrapper.begin("persona", tools=registry.specs())
    else:
        with pytest.raises(ValueError):
            wrapper.begin("persona", tools=registry.specs())


@pytest.mark.parametrize("throws_on", [
    lambda messages, tools: any(m["role"] == "tool" for m in messages),
    lambda messages, tools: bool(tools)], ids=["result", "declaration"])
def test_a_context_that_throws_while_probing_still_loads(throws_on):
    """The probe decides an optional capability, so it cannot refuse the model.

    A cache context of your own may raise where the bundled one returns None.
    Letting that escape would deny a conversation that never wanted tools.
    """
    class Throws(FakeContext):
        def render_with_model_template(self, messages, tools=None, **_options):
            if throws_on(messages, tools):
                raise RuntimeError("template blew a gasket")
            return super().render_with_model_template(messages, tools, **_options)

    wrapper = ChatWrapper(Config(context_size=1024),
                          context=Throws(TOOLS))

    assert not wrapper.can_call_tools
    wrapper.begin("persona")
    assert wrapper.reply("hallo").finish().text == "reply"


def test_the_result_block_is_probed_once_not_per_result():
    """A property read must not render a template, so the answer is kept."""
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=1024), context=fake)
    after_construction = len(fake.rendered)
    for _ in range(5):
        assert wrapper.can_call_tools
    assert len(fake.rendered) == after_construction


def test_declaring_refuses_a_model_that_could_never_call():
    """A declaration is a promise, so the whole chain is checked when it is made.

    Failing at the first call instead would surface mid-reply, after the model
    was already told the tool exists.
    """
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")

    no_calls = FakeContext(PLAIN)
    wrapper = ChatWrapper(Config(context_size=1024), context=no_calls)
    with pytest.raises(ValueError, match="describes no tool call"):
        wrapper.begin("persona", tools=registry.specs())

    unreadable = FakeContext(KEYLESS_CALLS)
    wrapper = ChatWrapper(Config(context_size=1024), context=unreadable)
    with pytest.raises(ValueError, match="call format could not be recovered"):
        wrapper.begin("persona", tools=registry.specs())

    no_result = FakeContext(NO_RESULT_BLOCK)
    wrapper = ChatWrapper(Config(context_size=1024), context=no_result)
    with pytest.raises(ValueError, match="renders no tool result"):
        wrapper.begin("persona", tools=registry.specs())


def test_a_registry_handed_to_a_running_reply_is_checked_the_same_way():
    """Handing tools over to be run is the same promise as declaring them.

    Unchecked, the exchange runs to the end and never calls anything,
    with nothing anywhere saying why.
    """
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")
    fake = FakeContext(NO_RESULT_BLOCK)
    wrapper = ChatWrapper(Config(context_size=1024), context=fake)
    wrapper.begin("persona")
    before = rows(wrapper)

    with pytest.raises(ValueError, match="renders no tool result"):
        wrapper.reply("wie spät?", tool_call_policy=registry).finish()
    with pytest.raises(ValueError, match="renders no tool result"):
        list(wrapper.reply("wie spät?", tool_call_policy=registry))

    assert rows(wrapper) == before          # refused before the request
    assert fake.cache == held_tokens(wrapper)
    # A registry with nothing in it promises nothing, and the tool-free path
    # was never promising anything either.
    assert wrapper.reply("hallo", tool_call_policy=ToolRegistry()).finish().text == "reply"
    assert wrapper.reply("hallo").finish().text == "reply"


def test_declaring_later_is_checked_the_same_way():
    fake = FakeContext(NO_RESULT_BLOCK)
    wrapper = ChatWrapper(Config(context_size=1024), context=fake)
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="renders no tool result"):
        wrapper.declare_tools(
            [ToolSpec("clock", "Die Zeit.", {"type": "object", "properties": {}})])


def test_a_refused_declaration_leaves_the_conversation_alone():
    """begin validates before it resets, and that must include this check."""
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")
    fake = FakeContext(NO_RESULT_BLOCK)
    wrapper = ChatWrapper(Config(context_size=1024), context=fake)
    wrapper.begin("erste persona")
    wrapper.reply("hallo").finish()
    before = rows(wrapper)

    with pytest.raises(ValueError, match="renders no tool result"):
        wrapper.begin("zweite persona", tools=registry.specs())

    assert rows(wrapper) == before


def test_a_model_without_tools_is_still_fine_without_them():
    fake = FakeContext(PLAIN)
    wrapper = ChatWrapper(Config(context_size=1024), context=fake)
    wrapper.begin("persona")                          # no tools, no complaint
    assert wrapper.reply("hallo").finish().text == "reply"


def test_a_template_with_no_result_block_refuses_the_result():
    fake = FakeContext(NO_RESULT_BLOCK)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))

    with pytest.raises(ValueError, match="renders no tool result"):
        wrapper.inject_tool_result(ToolCall("clock", {}), "ok")
    # Refused before anything was written, so the call is still owed.
    assert bool(wrapper.pending_calls)


# ----- what a caller has to write ------------------------------------------
def test_the_decorator_registers_and_declares_in_one_step():
    """The two steps a tool always needed, in the form the function is written."""
    registry = ToolRegistry()

    @registry.tool("Sagt die Zeit.")
    def clock(city: str) -> dict:
        return {"time": "12:00", "city": city}

    assert [spec.name for spec in registry.specs()] == ["clock"]
    assert registry.specs()[0].parameters["properties"] == {
        "city": {"type": "string"}}
    # Returned unchanged, so it stays an ordinary function.
    assert clock("Bonn") == {"time": "12:00", "city": "Bonn"}
    assert registry.call(ToolCall("clock", {"city": "Bonn"}))["city"] == "Bonn"


def test_a_result_is_injected_under_the_call_it_answers():
    """The call you were handed is what you hand back, not its name spelled out."""
    registry = ToolRegistry()
    registry.register(lambda: "12:00", "Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    events = wrapper.reply("wie spät?", tool_call_policy="yield")
    call = [event for event in events if isinstance(event, ToolCall)][0]
    wrapper.inject_tool_result(call, registry.call(call))

    assert not bool(wrapper.pending_calls)
    assert "12:00" in _result_of(fake)


def test_a_tool_left_out_at_begin_is_declared_later_beside_the_others():
    """One registry for both: begin takes the specs it is given, declaring adds the rest."""
    registry = ToolRegistry()

    @registry.tool("Sagt die Zeit.")
    def clock() -> dict:
        return {}

    @registry.tool("Sagt das Wetter.")
    def weather(city: str) -> dict:
        return {}

    clock_spec, weather_spec = registry.specs()
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=[clock_spec])

    assert "<decl>clock</decl>" in _declared_in(fake, 0)
    assert "<decl>weather</decl>" not in _declared_in(fake, 0)

    wrapper.declare_tools([weather_spec])          # the rest, where the template puts them

    assert "<decl>clock</decl>" in _declared_in(fake, 0)
    assert "<decl>weather</decl>" in _declared_in(fake, 0)
    assert [m["role"] for m in rows(wrapper)] == ["system"]


def test_declaring_the_same_tools_again_writes_nothing():
    """What the conversation already says is not worth a second prefill."""
    registry = ToolRegistry()

    @registry.tool("Sagt die Zeit.")
    def clock() -> dict:
        return {}

    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    wrapper.declare_tools(registry.specs())
    once = rows(wrapper)

    wrapper.declare_tools(registry.specs())
    wrapper.declare_tools(registry.specs())

    assert rows(wrapper) == once


def test_declaring_skips_what_the_system_turn_already_holds():
    """Repeating those is pure cost: the system turn is never evicted."""
    registry = ToolRegistry()

    @registry.tool("Sagt die Zeit.")
    def clock() -> dict:
        return {}

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    before = rows(wrapper)

    assert wrapper.declare_tools(registry.specs()) == 0
    assert rows(wrapper) == before          # nothing was written at all

    @registry.tool("Sagt das Wetter.")
    def weather(city: str) -> dict:
        return {}

    wrapper.declare_tools(registry.specs())      # only the one it has not heard
    declared = _declared_in(fake, 0)
    assert declared.count("<decl>clock</decl>") == 1 and "<decl>weather</decl>" in declared

    wrapper.begin("persona")                     # a reset forgets what was said
    wrapper.declare_tools(registry.specs())
    assert "<decl>clock</decl>" in _declared_in(fake, 0)


def test_a_turn_may_call_as_often_as_it_likes_by_default():
    """No ceiling unless one is asked for, so nothing is refused for counting."""
    ran = []
    registry = ToolRegistry()
    registry.register(lambda: ran.append(1) or "12:00", "Die Zeit.", name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"] * 9

    reply = wrapper.reply("los", tool_call_policy=registry).finish()

    assert len(ran) == 9                         # every one of them ran
    assert not reply.tool_limit_reached
    assert not any("tool_limit_reached" in text for text, _ in fake.tokenize_calls)


def test_a_stream_of_calls_hands_back_its_reply_too():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    events = wrapper.reply("wie spät?", tool_call_policy="yield")
    calls = [event for event in events if isinstance(event, ToolCall)]

    assert events.reply.stop_reason == "tool_call"
    assert events.reply.tool_calls == calls


# ----- a registry handed to a reply ----------------------------------------
def test_a_running_reply_is_its_own_stream_drained():
    registry = weather_registry()
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    assert wrapper.reply("wetter?", tool_call_policy=registry).finish().text == "reply"


def test_a_conversation_with_no_tools_needs_no_registry():
    """The easy path is the easy path whether or not tools are involved."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    assert wrapper.reply("hallo", tool_call_policy=ToolRegistry()).finish().text == "reply"
    assert fake.cache == held_tokens(wrapper)


# ----- tools declared before the newest user turn --------------------------
@pytest.mark.parametrize("can_shift", [True, False])
def test_declarations_move_on_with_the_newest_user_turn(can_shift):
    w, fake = _moving(can_shift=can_shift)
    specs = [ToolSpec("uhr", "", {})]
    assert ResolvedTemplate(FakeContext(TOOLS_DECLARED_IN_THE_NEWEST_TURN)).declares_before_newest_turn
    assert w.can_call_tools
    w.begin("s", tools=specs)
    _as_rendered_with(w, fake, specs)
    for text in ("eins", "zwei", "drei"):
        w.reply(text).finish()
        _as_rendered_with(w, fake, specs)
    w.inject("vier")
    _as_rendered_with(w, fake, specs)


def test_a_history_carries_its_declarations_on_its_last_user_turn():
    w, fake = _moving()
    specs = [ToolSpec("uhr", "", {})]
    w.begin("s", [("user", "eins"), ("assistant", "zwei"), ("user", "drei"),
                  ("assistant", "vier")], tools=specs)
    _as_rendered_with(w, fake, specs)
    w.reply("fünf").finish()
    _as_rendered_with(w, fake, specs)


def test_a_later_declaration_joins_the_newest_user_turn():
    w, fake = _moving()
    w.begin("s", tools=[ToolSpec("uhr", "", {})])
    w.reply("eins").finish()
    assert w.declare_tools([ToolSpec("wetter", "", {})]) == 0
    both = [ToolSpec("uhr", "", {}), ToolSpec("wetter", "", {})]
    _as_rendered_with(w, fake, both)
    w.reply("zwei").finish()
    _as_rendered_with(w, fake, both)


def test_a_running_exchange_holds_the_wrapper_while_its_tool_runs():
    # Another thread's action used to slip in between the rounds of a running reply.
    import threading
    wrapper, fake = tool_wrapper(gen_tokens=6, gen_text="Sonnig")
    registry = ToolRegistry()
    seen = []

    def weather(city: str) -> dict:
        idle = threading.Thread(target=lambda: seen.append(wrapper.wait_idle(timeout_s=0.05)))
        idle.start()
        idle.join()
        return {"city": city}

    registry.register(weather, "Wetter.")
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    assert wrapper.reply("wetter?", tool_call_policy=registry).finish().text == "Sonnig"
    assert seen == [False]


def test_withheld_calls_cannot_be_written_and_nothing_is_told_about_them():
    wrapper, fake = tool_wrapper(gen_tokens=6, gen_text="Sonnig")
    wrapper.begin("persona", tools=weather_registry().specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    reply = wrapper.reply("wetter?", tool_call_policy="withhold").finish()
    assert (reply.text, reply.stop_reason, reply.tool_calls) == ("Sonnig", "end_of_turn", [])
    assert fake.call_script == ["fn:weather{city:'Bonn'}"]    # never offered the call tokens
    assert not wrapper.turn_open


def test_a_fault_while_a_call_is_read_is_raised_and_leaves_the_conversation_whole(monkeypatch):
    """Only what a written call can cause counts as an unreadable call; a fault in the library is raised."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ['fn:<function=clock>{}</function>']

    def faulty(*args, **kwargs):
        raise TypeError("a fault in the library")

    monkeypatch.setattr("llama_chat.resolved_template.parse_calls", faulty)
    with pytest.raises(TypeError):
        wrapper.reply("wie spät?", tool_call_policy="yield").finish()
    monkeypatch.undo()
    assert wrapper.reply("noch einmal?").finish().text


# ----- a template of its own for a conversation that declares tools -------------------
def _with_a_template_for_tools(fixture=TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN):
    fake = FakeContext(fixture, gen_tokens=6, gen_text="Sonnig")
    return ChatWrapper(Config(context_size=1024, evict_above_fraction=0.9), context=fake), fake


def test_a_model_whose_template_for_tools_calls_them_can_call_tools_though_its_default_cannot():
    assert _with_a_template_for_tools()[0].can_call_tools
    without = {**TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN, "tool_use_template": None}
    assert not _with_a_template_for_tools(without)[0].can_call_tools


def test_a_conversation_is_written_with_the_template_for_tools_once_it_declares_them():
    chat, fake = _with_a_template_for_tools()
    chat.begin("persona")
    chat.reply("hallo?").finish()
    assert "<[say>" in held_text(fake) and "<[turn>" not in held_text(fake)
    chat.declare_tools(weather_registry().specs())
    assert "<[say>" not in held_text(fake) and "<decl>weather</decl>" in held_text(fake)
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    reply = chat.reply("wetter?", tool_call_policy=weather_registry()).finish()
    assert reply.text == "Sonnig" and [call.name for call in reply.tool_calls] == ["weather"]
    chat.begin("persona")                       # without tools: the default template again
    assert "<[say>" in held_text(fake) and "<[turn>" not in held_text(fake)


def test_a_conversation_begun_with_tools_is_written_with_the_template_for_tools():
    chat, fake = _with_a_template_for_tools()
    specs = weather_registry().specs()
    chat.begin("persona", tools=specs)
    chat.reply("hallo?").finish()
    assert "<decl>weather</decl>" in held_text(fake) and "<[say>" not in held_text(fake)


def test_a_template_for_tools_that_fails_the_load_check_refuses_the_model():
    # Its closing after a reply is plain text, which a reply could never stop on.
    broken = {**TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN, "tool_use_template":
              "{%- for m in messages -%}{{ m.role }}: {{ m.content }} END<turn]>\n"
              "{%- endfor -%}{%- if add_generation_prompt -%}model: {%- endif -%}"}
    with pytest.raises(ValueError, match="generation would never stop"):
        _with_a_template_for_tools(broken)
