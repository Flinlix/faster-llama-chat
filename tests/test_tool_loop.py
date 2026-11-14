"""Declaring tools to a conversation, the turn a call holds open, and running the tools."""

import json
import re
import threading

import pytest

from llama_chat.addon_files import template_fingerprint
from llama_chat.config import Config
from llama_chat.rendered_cache import RenderedCache
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tool_registry import ToolRegistry
from llama_chat.tools import ToolCall, ToolSpec
from llama_chat import ChatWrapper, ContextOverflowError, OutOfOrderError, UnsupportedError
from llama_chat.wrapper import MAX_UNREADABLE_CALLS_IN_A_ROW
from tests.chat_messages import Result, tool_turn
from tests.chat_templates import STEMS, WEATHER, expected, fixture
from tests.fake_context import FakeContext, ScriptedCall, written_call
from tests.invented_templates import (JSON_RESULT_LINES, JSON_TOOLS, KEYLESS_CALLS,
                                      NO_DECLARATIONS, NO_RESULT_BLOCK, PLAIN, TOOLS,
                                      TOOLS_DECLARED_IN_THE_NEWEST_TURN,
                                      TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN)
from tests.internals import held_tokens
from tests.wrappers import (add_tools, assert_cache_reads_as_rendered, assert_consistent, held_text,
                            newest_turn, system_turn, tool_wrapper, weather_registry,
                            without_ids, rows)


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

    assert without_ids(events) == [ToolCall(name="clock", arguments={})]
    assert bool(wrapper.pending_calls)
    # Nothing of the call reached the reply text.
    assert rows(wrapper)[-1]["text"] == ""


def test_a_call_whose_turn_ends_before_its_closing_tag_is_handed_over_closed():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.replies = [ScriptedCall("fn:clock{}", closes=False)]

    events = list(wrapper.reply("wie spät?", tool_call_policy="yield"))

    assert without_ids(events) == [ToolCall(name="clock", arguments={})]
    assert newest_turn(fake).endswith("<call>fn:clock{}</call>")
    wrapper.inject_tool_result(events[0], "12:00")
    assert wrapper.reply(tool_call_policy="yield").finish().text == "reply"


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
    assert without_ids(reply.tool_calls) == [ToolCall(name="clock", arguments={})]
    assert not wrapper.turn_open
    # The turn holds the call, an error result for it, and the reply after.
    written = rows(wrapper)[-1]["text"]
    assert written == "reply"
    assert fake.cache == held_tokens(wrapper)
    # And the conversation goes on, which is what the result bought.
    assert wrapper.reply("und sonst?").finish().stop_reason == "end_of_turn"


@pytest.mark.parametrize("policy", ["refuse", "registry"])
def test_a_reply_that_may_call_no_more_cannot_open_a_call_with_its_tag(policy):
    """A call the reply would not recognize is never drawn, so its tags never reach the answer."""
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}", "fn:clock{}"]
    options = {"tool_call_policy": registry, "max_calls": 1} if policy == "registry" else {}
    shown = "".join(e for e in wrapper.reply("wie spät?", **options) if isinstance(e, str))
    assert shown == "reply"
    assert fake.call_script == ["fn:clock{}"]       # the second call was never drawn
    assert rows(wrapper)[-1]["text"] == "reply"


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
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")

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
    registry.tool(lambda: ran.append("ran") or "12:00", description="Die Zeit.",
                  name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    wrapper.reply("wie spät?").finish()
    assert ran == []

    fake.call_script = ["fn:clock{}"]
    wrapper.reply("wie spät?", tool_call_policy=registry).finish()
    assert ran == ["ran"]


def test_a_call_the_library_runs_is_handed_over_in_the_stream_before_it_runs():
    """An app can say what is being looked up while the tool runs."""
    seen = []
    registry = ToolRegistry()
    registry.tool(lambda: seen.append("ran") or "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper(template=JSON_TOOLS)
    wrapper.begin("persona", tools=registry.specs())
    # The second call is past the ceiling, so it does not run, and is not handed over.
    fake.call_script = ['[{"name":"clock","arguments":{}},{"name":"clock","arguments":{}}]']

    stream = wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=1)
    for event in stream:
        if not isinstance(event, str):
            seen.append(("handed", event.name))
        elif seen[-1:] != ["text"]:
            seen.append("text")

    assert seen == [("handed", "clock"), "ran", "text"]
    assert stream.reply.text == "reply" and len(stream.reply.tool_calls) == 2


def test_a_stream_closed_at_a_call_handed_over_leaves_it_owed_and_unrun():
    ran = []
    registry = ToolRegistry()
    registry.tool(lambda: ran.append(1) or "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]

    with wrapper.reply("wie spät?", tool_call_policy=registry) as stream:
        call = next(event for event in stream if isinstance(event, ToolCall))

    assert ran == [] and wrapper.pending_calls == [call]
    wrapper.inject_tool_result(call, "12:00")
    assert wrapper.reply().finish().text == "reply" and not wrapper.turn_open


def test_a_refusing_reply_hands_over_no_call():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    assert "".join(wrapper.reply("wie spät?")) == "reply"


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
    # The call, its result and the reply after it, in one exchange.
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant", "tool", "assistant"]


def test_a_reply_going_on_after_results_keeps_the_policy_its_turn_was_started_with():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}", "fn:other{}"]
    first, = list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    wrapper.inject_tool_result(first, "09:00")

    second, = [e for e in wrapper.reply() if isinstance(e, ToolCall)]

    assert second.name == "other" and wrapper.pending_calls == [second]
    wrapper.inject_tool_result(second, "Bonn")
    # Another policy, given, holds for that reply.
    fake.call_script = ["fn:third{}"]
    reply = wrapper.reply(tool_call_policy="refuse").finish()
    assert reply.text == "reply" and not wrapper.turn_open
    # A new turn starts with the default again.
    fake.call_script = ["fn:clock{}"]
    assert wrapper.reply("und jetzt?").finish().text == "reply" and not wrapper.turn_open


def test_a_turn_a_history_left_open_goes_on_under_the_default_policy():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", [("user", "los"), *tool_turn(ToolCall("clock", {}),
                                                         Result("clock", "09:00"))])
    fake.call_script = ["fn:other{}"]
    reply = wrapper.reply().finish()
    assert reply.text == "reply" and not wrapper.turn_open
    assert "tools_not_run" in newest_turn(fake)


def test_two_calls_in_a_row_stay_inside_one_turn():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:first{}"]
    assert without_ids(wrapper.reply("los", tool_call_policy="yield")) == [ToolCall("first", {})]

    wrapper.inject_tool_result(ToolCall("first", {}), "ok")
    fake.call_script = ["fn:second{}"]
    assert without_ids(wrapper.reply(tool_call_policy="yield")) == [ToolCall("second", {})]

    wrapper.inject_tool_result(ToolCall("second", {}), "ok")
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))

    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant", "tool",
                                                  "assistant", "tool", "assistant"]


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
    assert without_ids(wrapper.reply("los", tool_call_policy="yield")) == [ToolCall("first", {})]
    assert len(rows(wrapper)) - 1 == 4     # two exchanges, over the limit, turn still open

    wrapper.inject_tool_result(ToolCall("first", {}), "ok")
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))
    # the newest exchange alone once closed
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant", "tool", "assistant"]
    assert rows(wrapper)[0]["role"] == "system"


def test_a_result_under_the_wrong_name_is_refused():
    """Results are owed in call order, so only the oldest call owed is answered."""
    wrapper, fake = tool_wrapper(template=JSON_TOOLS)
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']
    list(wrapper.reply("los", tool_call_policy="yield"))

    with pytest.raises(ValueError, match=re.escape("the waiting call is ToolCall(name='clock'")):
        wrapper.inject_tool_result(ToolCall("other", {}), "b")

    # The refusal consumed nothing: both calls are still owed, in order.
    wrapper.inject_tool_result(ToolCall("clock", {}), "a")
    wrapper.inject_tool_result(ToolCall("other", {}), "b")
    fake.call_script = []
    list(wrapper.reply(tool_call_policy="yield"))


@pytest.mark.parametrize("not_the_call, refusal", [
    ("clock", TypeError), (ToolCall("clock", {"city": "Bonn"}), OutOfOrderError)])
def test_a_result_answers_the_call_itself_not_its_name(not_the_call, refusal):
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    before = list(fake.cache)

    with pytest.raises(refusal, match="the waiting call is|must be the ToolCall"):
        wrapper.inject_tool_result(not_the_call, "ok")

    assert fake.cache == before
    assert without_ids(wrapper.pending_calls) == [ToolCall("clock", {})]


def test_a_result_that_cannot_fit_even_cut_closes_the_turn_without_any_of_it():
    """The tags are never cut, so a turn with no room for them closes on what the model wrote.

    Cutting them would leave the cache holding half a block no template ever
    produces.
    """
    for size in range(40, 400):
        wrapper, fake = tool_wrapper(context_size=size)
        try:
            wrapper.begin("persona")
            fake.call_script = ["fn:clock{}"]
            list(wrapper.reply("wie spät?", tool_call_policy="yield"))
        except ContextOverflowError:
            continue                    # too small for the call itself
        wrapper.inject_tool_result(ToolCall("clock", {}), "ok")
        if wrapper.turn_open:
            continue                    # room for the result
        assert "<result>" not in held_text(fake) and wrapper.pending_calls == []
        assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]
        fake.call_script = []
        assert wrapper.reply("wie spät?").finish().stop_reason == "end_of_turn"
        return
    pytest.fail("no context size left the call room but its result none")


@pytest.mark.parametrize("policy", ["refuse", "registry"])
def test_a_running_reply_with_no_room_to_answer_its_call_closes_the_turn_as_the_cap_does(policy):
    """The turn closes on what the model wrote, as a reply the cap cut does, and the conversation goes on."""
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    for size in range(40, 400):
        wrapper, fake = tool_wrapper(context_size=size)
        try:
            wrapper.begin("persona")
        except ContextOverflowError:
            continue
        fake.call_script = ["fn:clock{}"]
        try:
            reply = wrapper.reply("wie spät?", tool_call_policy=(
                registry if policy == "registry" else policy)).finish()
        except ContextOverflowError as refused:
            if "the request needs" in str(refused) or "the reply needs" in str(refused):
                continue                # too small for the question itself
            raise
        if reply.stop_reason != "token_limit" or "<result>" in held_text(fake):
            continue                    # room for the result, or none for the call
        assert not wrapper.turn_open and wrapper.pending_calls == []
        assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]
        fake.call_script = []
        assert wrapper.reply("wie spät?").finish().stop_reason == "end_of_turn"
        return
    pytest.fail("no context size left the call room but its result none")


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
        template.ensure_tools_usable("tools")                # the probes recovered nothing
    assert not wrapper.capabilities(tools=True).can_call_tools

    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    reply = wrapper.reply("los").finish()
    assert reply.text == "reply"                 # it answered instead
    assert not wrapper.turn_open


def test_an_oversized_result_is_cut_keeps_its_tags_and_tells_the_model_and_the_caller():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in wrapper.reply("wie spät?", tool_call_policy="yield")
             if isinstance(e, ToolCall)]

    before = len(fake.cache)
    wrapper.inject_tool_result(call, "x" * 10_000)

    # The turn still fits, and the block still ends where the template ends one.
    assert len(fake.cache) <= wrapper.config.context_size
    written = fake.text_of(fake.cache[before:])
    assert written.startswith("<result>clock(x")
    kept = written.count("x")
    assert written.endswith(f"x\n(cut: {kept} of 10000 characters kept))</result>")
    fake.call_script = []
    assert wrapper.reply().finish().tool_results_cut == [call]
    assert wrapper.reply("und sonst?").finish().tool_results_cut == []


def test_a_result_cut_under_a_registry_is_listed_on_its_reply():
    registry = ToolRegistry()
    registry.tool(lambda: "x" * 10_000, description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    reply = wrapper.reply("wie spät?", tool_call_policy=registry).finish()
    assert without_ids(reply.tool_results_cut) == [ToolCall("clock", {})]
    assert reply.tool_results_cut == reply.tool_calls


def _two_calls_owed(context_size):
    wrapper, fake = tool_wrapper(context_size=context_size, template=JSON_TOOLS)
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},{"name":"other","arguments":{}}]']
    calls = [e for e in wrapper.reply("los", tool_call_policy="yield") if isinstance(e, ToolCall)]
    assert [call.name for call in calls] == ["clock", "other"]
    return wrapper, fake, calls


def test_a_result_cut_leaves_room_for_the_results_still_owed_after_it():
    wrapper, fake, (first, second) = _two_calls_owed(context_size=400)
    wrapper.inject_tool_result(first, "x" * 5000)
    wrapper.inject_tool_result(second, "y" * 10)
    assert wrapper.turn_open and not wrapper.pending_calls
    written = fake.text_of(fake.cache)
    assert "y" * 10 in written and " of 5000 characters kept)" in written
    fake.call_script = []
    assert wrapper.reply().finish().tool_results_cut == [first]


def test_short_results_that_fit_together_are_written_whole_and_keep_the_turn_open():
    wrapper, fake = tool_wrapper(context_size=400, template=JSON_TOOLS)
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},{"name":"other","arguments":{}}]']
    first, second = [e for e in wrapper.reply("q" * 230, tool_call_policy="yield")
                     if isinstance(e, ToolCall)]
    wrapper.inject_tool_result(first, "a")          # shorter than the note a cut leaves
    wrapper.inject_tool_result(second, "b")
    assert wrapper.turn_open and not wrapper.pending_calls
    fake.call_script = []
    assert wrapper.reply().finish().tool_results_cut == []


@pytest.mark.parametrize("length", range(200, 300))
def test_a_round_of_results_keeps_its_turn_however_long_the_first_result_is(length):
    wrapper, fake, (first, second) = _two_calls_owed(context_size=400)
    wrapper.inject_tool_result(first, "x" * length)
    wrapper.inject_tool_result(second, "y")
    assert wrapper.turn_open and not wrapper.pending_calls
    fake.call_script = []
    assert set(wrapper.reply().finish().tool_results_cut) <= {first, second}


def test_a_turn_closed_for_want_of_room_lists_no_result_it_dropped_as_cut():
    wrapper, fake, (first, second) = _two_calls_owed(context_size=400)
    wrapper.inject_tool_result(first, "x" * 5000)
    wrapper.inject_tool_result(second, "y" * 10 ** 7)       # its cut note longer than any room left
    assert not wrapper.turn_open
    assert "tool" not in [m["role"] for m in wrapper.conversation()["messages"]]
    fake.call_script = []
    assert wrapper.reply("und?").finish().tool_results_cut == []


@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot-shift"])
def test_a_result_too_long_for_the_room_left_evicts_the_oldest_exchanges_first(can_shift):
    wrapper, fake = tool_wrapper(context_size=600, can_shift=can_shift)
    wrapper.begin("persona", [item for n in range(6) for item in
                              (("user", f"Frage {n}?"), ("assistant", f"Antwort {n}."))])
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in wrapper.reply("wie spät?", tool_call_policy="yield")
             if isinstance(e, ToolCall)]
    room = wrapper.config.context_size - wrapper.config.min_reply_tokens
    result = "y" * (room - wrapper.total_tokens + 40)
    evicted = wrapper.inject_tool_result(call, result).n_evicted_messages
    # Whole, the turn kept, and on a cache that cannot shift well below the room left.
    assert evicted > 0 and _result_of(fake) == f"clock({result})"
    questions = [m["text"] for m in rows(wrapper) if m["role"] == "user"]
    assert questions[-1] == "wie spät?" and len(questions) < 7
    if not can_shift:
        assert len(questions) <= 3
    fake.call_script = []
    assert wrapper.reply().finish().tool_results_cut == []


# ----- the local loop ------------------------------------------------------
def test_a_running_reply_runs_the_tool_and_yields_its_call_then_prose():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    seen = []
    registry.tool(lambda: seen.append("ran") or {"time": "09:00"},
                  description="Sagt die Zeit.", name="clock")
    fake.call_script = ["fn:clock{}"]

    events = list(wrapper.reply("wie spät?", tool_call_policy=registry))

    assert seen == ["ran"]
    assert without_ids(events[:1]) == [ToolCall("clock", {})]
    assert "".join(events[1:])        # the reply after the result
    assert not bool(wrapper.pending_calls)


@pytest.mark.parametrize("before_call, after_call, joined", [
    ("Ich sehe nach.", "Es ist neun.", "Ich sehe nach.\n\nEs ist neun."),
    ("Ich sehe nach. ", "Es ist neun.", "Ich sehe nach. Es ist neun."),
    ("Ich sehe nach.", " Es ist neun.", "Ich sehe nach. Es ist neun."),
    ("", "Es ist neun.", "Es ist neun."),
])
@pytest.mark.parametrize("handed_over, going_on", [
    (False, None), (True, "yield"), (True, "refuse"), (True, "registry")],
    ids=["run", "yield", "yield-then-refuse", "yield-then-run"])
def test_the_texts_around_a_call_are_parted_by_a_blank_line_where_they_would_join(
        before_call, after_call, joined, handed_over, going_on):
    """Whether the library runs the call or hands it over, and whatever the reply after the result runs under, the same rule parts the texts."""
    fake = FakeContext(TOOLS, gen_text=after_call, gen_tokens=len(after_call))
    wrapper = ChatWrapper(Config(context_size=512, evict_above_fraction=0.9), context=fake)
    wrapper.begin("persona")
    registry = ToolRegistry()
    registry.tool(lambda: {"time": "09:00"}, description="Sagt die Zeit.", name="clock")
    fake.replies = [ScriptedCall("fn:clock{}", before=before_call)]

    if not handed_over:
        stream = wrapper.reply("wie spät?", tool_call_policy=registry)
        shown = "".join(event for event in stream if isinstance(event, str))
        assert shown == stream.reply.text == joined
        return
    first = wrapper.reply("wie spät?", tool_call_policy="yield")
    shown = [event for event in first if isinstance(event, str)]
    wrapper.inject_tool_result(wrapper.pending_calls[0], {"time": "09:00"})
    rest = wrapper.reply(tool_call_policy=registry if going_on == "registry" else going_on)
    shown += list(rest)
    assert "".join(shown) == first.reply.text + rest.reply.text == joined


def test_a_running_reply_stops_calling_at_the_ceiling_and_closes_the_turn():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    runs = []
    registry.tool(lambda: runs.append(1) or {}, description="…", name="clock")
    # The fake would call forever; the ceiling is what has to stop it.
    fake.call_script = ["fn:clock{}"] * 10

    list(wrapper.reply("los", tool_call_policy=registry, max_calls=2))

    assert len(runs) == 2
    assert not bool(wrapper.pending_calls)
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant", "tool",
                                                  "assistant", "tool", "assistant"]


def test_a_declaration_added_later_reaches_the_model():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    registry = ToolRegistry()
    registry.tool(lambda: {}, description="Sagt die Zeit.", name="clock")

    add_tools(wrapper, registry.specs())

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
        add_tools(wrapper, [first, second])
    assert fake.cache == before and add_tools(wrapper, [first]) == 0
    assert held_text(fake).count("<decl>clock</decl>") == 1


def test_declaring_nothing_writes_nothing():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    before = list(fake.cache)
    assert add_tools(wrapper, []) == 0
    assert fake.cache == before


def test_declarations_too_large_to_fit_are_refused_rather_than_truncated():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    before = list(fake.cache)
    registry = ToolRegistry()
    # The invented template renders one <decl>NAME</decl> per tool, so many long names
    # overrun the threshold.
    for i in range(200):
        registry.tool(lambda: {}, description="…", name=f"tool_with_a_long_name_{i}")

    with pytest.raises(ValueError, match="declare fewer tools"):
        add_tools(wrapper, registry.specs())

    # A refused declaration leaves the conversation exactly as it was.
    assert fake.cache == before


def test_an_empty_parameter_schema_is_a_caller_saying_no_parameters():
    # Not a caller saying nothing: the annotations must not be read instead.
    def anything(title: str):
        ...

    registry = ToolRegistry()
    registry.tool(anything, description="…", parameters={})
    assert registry.specs()[0].parameters == {}


def test_declaring_before_a_conversation_is_refused():
    wrapper, fake = tool_wrapper()
    registry = ToolRegistry()
    registry.tool(lambda: {}, description="…", name="clock")

    # `begin` would reset the conversation and throw the declaration away, so writing
    # it here would be a silent loss rather than a declaration.
    with pytest.raises(ValueError, match="no conversation to change the settings of"):
        add_tools(wrapper, registry.specs())

    assert fake.cache == []


def test_a_model_without_tool_support_still_reports_the_missing_conversation():
    # The caller's own mistake is named before the model's limits are.
    fake = FakeContext(PLAIN)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    spec = ToolSpec("clock", "…", {"type": "object", "properties": {}})
    with pytest.raises(ValueError, match="no conversation to change the settings of"):
        add_tools(wrapper, [spec])


def test_a_call_this_parser_cannot_read_still_closes_the_turn():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    # A body in another model's format, which this model's fragments cannot read.
    fake.call_script = ["fn:\n<fun=clock>\n<arg=x>\n1\n</arg>\n</fun>\n"]

    list(wrapper.reply("wie spät?", tool_call_policy="yield"))

    # Left open, nothing would ever close it and the next request would prefill
    # into a model turn that never ended.
    assert not bool(wrapper.pending_calls)


def test_an_unreadable_call_says_why():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:<fun=clock>"]

    events = wrapper.reply("wie spät?", tool_call_policy="yield")
    while True:
        try:
            next(events)
        except StopIteration as done:
            reply = done.value
            break

    assert reply.stop_reason == "end_of_turn"       # told so, the model answered
    # Reported with what the model wrote, though nothing of it is handed over to be answered.
    assert reply.tool_calls == [ToolCall("", {}, status="unreadable", text="fn:<fun=clock>")]


def test_a_refusal_while_an_unreadable_call_is_answered_is_raised_not_taken_for_no_room(
        monkeypatch):
    """Only no room, or no result block, records the reply without its call; any other refusal is raised."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"]

    fitting_prompt = RenderedCache._fitting_prompt

    def refused(self):
        if "nonsense" in held_text(fake):       # only where the call is answered
            raise ValueError("a kept turn cannot be placed")
        return fitting_prompt(self)

    monkeypatch.setattr(RenderedCache, "_fitting_prompt", refused)
    with pytest.raises(ValueError, match="cannot be placed"):
        wrapper.reply("los").finish()
    monkeypatch.undo()
    assert wrapper.reply("noch einmal?").finish().text == "reply"


@pytest.mark.parametrize("policy", ["refuse", "registry"])
def test_an_unreadable_call_is_reported_where_the_reply_answers_it_itself(policy):
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"]
    reply = wrapper.reply("los", tool_call_policy=(
        weather_registry() if policy == "registry" else policy)).finish()
    assert reply.tool_calls == [ToolCall("", {}, status="unreadable",
                                         text="fn:nonsense with no name")]
    assert reply.text == "reply"


@pytest.mark.parametrize("policy", ["refuse", "yield", "registry"])
@pytest.mark.parametrize("cut_by", ["token_limit", "interrupted"])
def test_a_call_cut_by_the_cap_or_a_cancel_is_reported_unfinished(policy, cut_by):
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    cancel = threading.Event()
    decode = fake.decode_generated

    def decode_and_cancel(tokens, start):
        decode(tokens, start)
        if "<call>fn" in held_text(fake):
            cancel.set()
    fake.decode_generated = decode_and_cancel
    options = ({"max_reply_tokens": 4} if cut_by == "token_limit" else {"cancel": cancel})

    reply = wrapper.reply("wie spät?", tool_call_policy=(
        weather_registry() if policy == "registry" else policy), **options).finish()

    assert reply.stop_reason == cut_by
    assert reply.tool_calls == [ToolCall("", {}, status="unfinished")]
    assert not wrapper.turn_open and reply.text == ""


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
    with pytest.raises(OutOfOrderError, match="while a reply is open"):
        add_tools(wrapper, [ToolSpec("x", "d", {"type": "object", "properties": {}})])

    list(wrapper.reply(tool_call_policy="yield"))
    assert not wrapper.turn_open               # closed, and work is allowed again
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant", "tool", "assistant"]
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


@pytest.mark.parametrize("body", [
    '{"name": "clock", "arguments": null}',
    '{"name": "clock", "arguments": ["Bonn"]}',
    "fn:clock{a:%s}" % ("[" * 600 + "]" * 600),
], ids=["null", "a-list", "nested-too-deeply"])
def test_a_call_whose_arguments_cannot_be_read_does_not_run_and_is_told_so(body):
    ran = []
    registry = ToolRegistry()
    registry.tool(lambda **given: ran.append(given) or "12:00", description="Die Zeit.",
                  name="clock", parameters={"type": "object", "properties": {}})
    wrapper, fake = tool_wrapper(context_size=8192)
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = [body]

    reply = wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=1).finish()

    assert ran == []
    assert "unreadable_call" in newest_turn(fake)
    assert reply.text == "reply" and not wrapper.turn_open


def test_a_failed_call_is_named_as_one_and_told_to_try_again():
    """The reason alone reads as data; the model has to know the call failed."""
    registry = ToolRegistry()
    registry.tool(lambda: 1 / 0, description="Die Zeit.", name="clock")
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
    registry.tool(lambda: ran.append(1) or "12:00", description="Die Zeit.", name="clock")

    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name", "fn:clock{}"]

    reply = wrapper.reply("wie spät?", tool_call_policy=registry).finish()

    assert ran == [1]                                # the second one ran
    assert [(c.name, c.status, c.text) for c in reply.tool_calls] == [
        ("", "unreadable", "fn:nonsense with no name"), ("clock", "complete", "")]
    assert not wrapper.turn_open


def test_an_unreadable_call_under_yield_is_written_again_in_the_same_reply():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name", "fn:clock{}"]

    stream = wrapper.reply("wie spät?", tool_call_policy="yield")
    handed_over = [(e.name, e.arguments) for e in stream if isinstance(e, ToolCall)]

    assert handed_over == [("clock", {})]
    assert [(c.name, c.status) for c in stream.reply.tool_calls] == [
        ("", "unreadable"), ("clock", "complete")]
    assert stream.reply.stop_reason == "tool_call"
    wrapper.inject_tool_result(wrapper.pending_calls[0], "12:00")
    assert wrapper.reply().finish().text == "reply" and not wrapper.turn_open


def test_an_unreadable_call_under_yield_is_answered_in_the_same_reply():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"]

    reply = wrapper.reply("wie spät?", tool_call_policy="yield").finish()

    assert (reply.text, reply.stop_reason) == ("reply", "end_of_turn")
    assert not wrapper.turn_open and wrapper.pending_calls == []


def test_retrying_an_unreadable_call_under_yield_is_bounded_by_the_reply_cap():
    wrapper, fake = tool_wrapper(context_size=2048)
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 20      # 26 tokens a call

    reply = wrapper.reply("wie spät?", tool_call_policy="yield", max_reply_tokens=60).finish()

    assert reply.n_generated_tokens <= 60
    assert len(fake.call_script) == 17                  # two tries, and a third cut by the cap
    assert reply.stop_reason == "token_limit" and not wrapper.turn_open


@pytest.mark.parametrize("cap_from", ["request", "config"])
@pytest.mark.parametrize("policy", ["refuse", "registry", "yield"])
def test_the_reply_cap_bounds_every_generation_of_a_reply_together(policy, cap_from):
    fake = FakeContext(TOOLS, gen_tokens=80)
    wrapper = ChatWrapper(Config(context_size=2048, evict_above_fraction=0.9,
                                 max_reply_tokens=40 if cap_from == "config" else None),
                          context=fake)
    wrapper.begin("persona")
    # Under "yield" only a call that cannot be read is answered within the reply.
    fake.call_script = ["fn:nonsense with no name" if policy == "yield" else 'fn:weather{"city":"Bonn"}']

    reply = wrapper.reply("wie spät?", tool_call_policy=(
        weather_registry() if policy == "registry" else policy),
        max_reply_tokens=40 if cap_from == "request" else None).finish()

    assert len(reply.tool_calls) == 1 and not fake.call_script     # a call, then text
    assert reply.n_generated_tokens == 40 and reply.stop_reason == "token_limit"


@pytest.mark.parametrize("policy", ["yield", "registry"])
def test_a_reply_reads_at_most_three_unreadable_calls_in_a_row_then_the_model_answers(policy):
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper(context_size=4096)
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 20

    reply = wrapper.reply("wie spät?", tool_call_policy=registry if policy == "registry" else policy
                          ).finish()

    assert len(fake.call_script) == 20 - MAX_UNREADABLE_CALLS_IN_A_ROW
    assert [c.status for c in reply.tool_calls] == ["unreadable"] * MAX_UNREADABLE_CALLS_IN_A_ROW
    told = held_text(fake)
    assert told.count("try again") == MAX_UNREADABLE_CALLS_IN_A_ROW - 1
    assert told.count("do not call again") == 1
    assert reply.stop_reason == "end_of_turn" and not wrapper.turn_open


def test_a_number_a_model_writes_for_a_text_parameter_reaches_the_tool_as_it_was_written():
    given = []
    registry = ToolRegistry()

    @registry.tool("Notes a version.")
    def note(version: str) -> str:
        given.append(version)
        return "ok"

    wrapper, fake = tool_wrapper(context_size=2048, template=JSON_TOOLS)
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ['[{"name": "note", "arguments": {"version": 3.10}}]']
    wrapper.reply("Notiere es.", tool_call_policy=registry).finish()
    assert given == ["3.10"]


def test_a_call_that_spends_the_reply_cap_is_not_run_and_its_turn_closes():
    probe, probe_fake = tool_wrapper(context_size=2048)
    probe.begin("persona")
    probe_fake.call_script = ["fn:clock{}"]
    call_tokens = probe.reply("wie spät?", tool_call_policy="yield").finish().n_generated_tokens
    ran = []
    registry = ToolRegistry()
    registry.tool(lambda: ran.append(1) or "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper(context_size=2048)
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    stream = wrapper.reply("wie spät?", tool_call_policy=registry, max_reply_tokens=call_tokens)
    handed = [e for e in stream if isinstance(e, ToolCall)]
    reply = stream.finish()

    assert ran == [] and handed == []
    assert reply.stop_reason == "token_limit" and not wrapper.turn_open
    assert [call.status for call in reply.tool_calls] == ["unfinished"]
    wrapper.reply("und?").finish()


def test_a_readable_call_starts_the_count_of_unreadable_calls_in_a_row_again():
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper(context_size=4096)
    wrapper.begin("persona")
    unreadable = "fn:nonsense with no name"
    fake.call_script = [unreadable, unreadable, "fn:clock{}"] + [unreadable] * 5

    reply = wrapper.reply("wie spät?", tool_call_policy=registry).finish()

    assert len(fake.call_script) == 2
    assert [c.status for c in reply.tool_calls].count("unreadable") == 5
    told = held_text(fake)
    assert told.count("try again") == 4 and told.count("do not call again") == 1
    assert told.rindex("do not call again") > told.rindex("try again")


def test_retrying_an_unreadable_call_is_bounded_by_the_ceiling():
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 20

    wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=3).finish()

    assert len(fake.call_script) == 17               # three tries, then none drawn
    assert not wrapper.turn_open


def test_a_refusing_reply_does_not_retry_an_unreadable_call():
    """A retry could only be refused, this path running no tool either way."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 5

    wrapper.reply("wie spät?").finish()

    assert len(fake.call_script) == 4                # one try, then none drawn
    assert not wrapper.turn_open


def test_a_ceiling_of_zero_binds_an_unreadable_call_too():
    """The budget is spent only while there is budget, or it goes negative.

    ``tool_limit_reached`` compares it to zero, so overshooting reports that the
    ceiling did not bind when it is the only reason nothing was retried.
    """
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"] * 5

    reply = wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=0).finish()

    assert reply.tool_limit_reached
    assert len(fake.call_script) == 4                # no retry: none drawn after it
    assert not wrapper.turn_open


def test_the_instruction_matches_what_the_model_may_still_do():
    """The same failure ends differently once no call is left to make.

    Telling a model to try again where the tags are about to be withheld would
    ask for the one thing it cannot do.
    """
    registry = ToolRegistry()
    registry.tool(lambda: 1 / 0, description="Die Zeit.", name="clock")

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
    registry.tool(lambda: {"time": "12:00"}, description="Die Zeit.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]

    wrapper.reply("wie spät?", tool_call_policy=registry).finish()

    told = _result_of(fake)
    assert '{"time": "12:00"}' in told
    assert "failed" not in told


def test_the_exchange_says_when_the_tool_ceiling_bound():
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="The time.", name="clock")
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
    registry.tool(lambda: "12:00", description="The time.", name="clock")
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
    registry.tool(lambda: "12:00", description="The time.", name="clock")
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
    assert "".join(d for d in deltas if isinstance(d, str)) == reply.text


# ----- an action on the thread whose reply is still being read ------------
def _within_seconds(action, seconds=10):
    """Run ``action`` on a thread of its own; what it returned or raised, or a failure where it hangs."""
    outcome = {}

    def run():
        try:
            outcome["returned"] = action()
        except BaseException as e:      # noqa: BLE001  handed back to the test
            outcome["raised"] = e
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        pytest.fail("the action hangs")
    if "raised" in outcome:
        raise outcome["raised"]
    return outcome["returned"]


@pytest.mark.parametrize("action", [lambda w: w.inject("Notiz"), lambda w: w.close()],
                         ids=["inject", "close"])
def test_a_tool_that_calls_back_into_the_wrapper_is_refused_rather_than_hanging(action):
    wrapper, fake = tool_wrapper()
    registry = ToolRegistry()

    @registry.tool("Notiert etwas.")
    def note() -> str:
        action(wrapper)
        return "ok"

    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:note{}"]
    reply = _within_seconds(lambda: wrapper.reply("los", tool_call_policy=registry).finish())
    told = _result_of(fake)
    assert "tool_raised" in told and "the reply this thread is reading" in told
    assert reply.text == "reply" and not wrapper.turn_open


@pytest.mark.parametrize("action", [
    lambda w: w.inject("Notiz"),
    lambda w: w.reply("noch was").finish(),
    lambda w: w.begin("neu"),
    lambda w: add_tools(w, [ToolSpec("clock", "…", {})]),
    lambda w: w.truncate_last_reply(""),
    lambda w: w.wait_idle(),
], ids=["inject", "reply", "begin", "declare_tools", "truncate_last_reply", "wait_idle"])
def test_an_action_on_the_thread_reading_a_reply_is_refused_and_names_the_reply(action):
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")

    def read_and_act():
        stream = wrapper.reply("hallo")
        next(stream)
        try:
            action(wrapper)
        finally:
            stream.close()

    with pytest.raises(RuntimeError, match="the reply this thread is reading"):
        _within_seconds(read_and_act)
    assert wrapper.reply("weiter").finish().text == "reply"


def test_an_action_on_the_thread_a_reply_was_handed_to_is_refused_rather_than_hanging():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    stream = wrapper.reply("hallo")
    first_read, release = threading.Event(), threading.Event()

    def read_first():
        next(stream)
        first_read.set()
        release.wait(10)        # still alive while another thread reads on

    first = threading.Thread(target=read_first, daemon=True)
    first.start()
    first_read.wait(10)

    def read_on_and_act():
        next(stream)
        try:
            wrapper.inject("Notiz")
        finally:
            stream.close()

    try:
        with pytest.raises(RuntimeError, match="the reply this thread is reading"):
            _within_seconds(read_on_and_act)
    finally:
        release.set()
        first.join(10)
    assert wrapper.reply("weiter").finish().text == "reply"


def test_a_thread_that_reads_no_reply_waits_for_one_rather_than_being_refused():
    """A thread started after the reader ended may carry the reader's number; it is not the reader."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    stream = wrapper.reply("hallo")
    _within_seconds(lambda: next(stream))
    for _ in range(3):
        assert _within_seconds(lambda: wrapper.wait_idle(timeout_s=0.01)) is False
    stream.close()


# ----- a model that writes a batch -----------------------------------------
def test_every_call_of_a_batch_owes_a_result_before_the_reply_resumes():
    wrapper, fake = tool_wrapper(template=JSON_TOOLS)
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']

    asked = list(wrapper.reply("los", tool_call_policy="yield"))
    assert without_ids(asked) == [ToolCall("clock", {}), ToolCall("other", {})]
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
    wrapper, fake = tool_wrapper(template=JSON_TOOLS)
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{"city":"Bonn"}}]']
    asked = list(wrapper.reply("los", tool_call_policy="yield"))
    assert wrapper.pending_calls == asked

    wrapper.pending_calls[1].arguments["city"] = "Rom"          # a copy, the caller's to change
    wrapper.inject_tool_result(asked[0], "a")
    assert wrapper.pending_calls == [asked[1]] and asked[1].arguments == {"city": "Bonn"}
    wrapper.inject_tool_result(asked[1], "b")
    assert wrapper.pending_calls == [] and wrapper.turn_open


def test_a_history_owes_the_calls_of_its_newest_turn_that_no_result_answers():
    wrapper, _ = tool_wrapper()
    clock, other = ToolCall("clock", {}), ToolCall("other", {})
    wrapper.begin("persona", [("user", "los"), *tool_turn(clock, other)])
    assert without_ids(wrapper.pending_calls) == [clock, other]
    wrapper.begin("persona", [("user", "los"), *tool_turn(clock, other, Result("clock", "a"))])
    assert without_ids(wrapper.pending_calls) == [other]


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
    wrapper, fake = tool_wrapper(template=JSON_TOOLS)
    wrapper.begin("persona")
    registry = ToolRegistry()
    ran = []
    registry.tool(lambda: ran.append("clock") or {}, description="…", name="clock")
    registry.tool(lambda: ran.append("other") or {}, description="…", name="other")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        '{"name":"other","arguments":{}}]']

    list(wrapper.reply("los", tool_call_policy=registry))

    assert ran == ["clock", "other"]


def test_the_results_of_a_batch_are_written_as_the_template_writes_a_batch():
    fake = FakeContext(JSON_RESULT_LINES)
    wrapper = ChatWrapper(Config(context_size=1024, evict_above_fraction=0.9),
                          context=fake)
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
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
    registry.tool(clock, description="Sagt die Zeit.")

    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=200, evict_above_fraction=0.5),
                          context=fake)
    wrapper.begin("persona")
    add_tools(wrapper, registry.specs())
    assert [m["role"] for m in rows(wrapper)] == ["system"]

    for i in range(6):
        wrapper.reply(f"frage {i}").finish()

    # The system turn is pinned, so the declaration in it is never evicted.
    assert "<decl>clock</decl>" in _declared_in(fake, 0)
    assert add_tools(wrapper, registry.specs()) == 0


# ----- what a model that cannot do this looks like -------------------------
def test_a_batch_past_the_ceiling_is_answered_rather_than_run():
    """With one call per round the ceiling withholds the call tokens instead.

    A model that asks for a batch can outrun the ceiling inside a single round,
    and the calls past it are answered with an error the model can read.
    """
    ran = []
    registry = ToolRegistry()
    registry.tool(lambda: ran.append(1) or "09:00", description="Zeit.", name="clock")

    wrapper, fake = tool_wrapper(template=JSON_TOOLS)
    wrapper.begin("persona")
    fake.call_script = ['[{"name":"clock","arguments":{}},'
                        ' {"name":"clock","arguments":{}},'
                        ' {"name":"clock","arguments":{}}]']
    list(wrapper.reply("frage", tool_call_policy=registry, max_calls=1))

    assert len(ran) == 1                       # only the permitted one ran
    written = [m["content"] for m in wrapper.conversation()["messages"] if m["role"] == "tool"]
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
    assert not wrapper.capabilities(tools=True).can_call_tools


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
        add_tools(wrapper, [spec])


def test_recording_a_reply_survives_a_template_with_nowhere_to_tell_it():
    """A template that renders no result block could never answer a call, so under the default none is read: the reply closes its turn."""
    fake = FakeContext(NO_RESULT_BLOCK)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin("persona")
    fake.call_script = ["fn:nonsense with no name"]

    reply = wrapper.reply("wie spät?").finish()          # must not raise

    assert reply.stop_reason not in ("tool_call", "unreadable_tool_call")
    assert not wrapper.turn_open
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
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    wrapper = build()

    assert wrapper.capabilities(tools=True).can_call_tools is can_call
    if can_call:
        wrapper.begin("persona", tools=registry.specs())
    else:
        with pytest.raises(ValueError):
            wrapper.begin("persona", tools=registry.specs())


@pytest.mark.parametrize("raises_on", [
    "{% for m in messages if m.role == 'tool' %}{{ raise_exception('no results') }}{% endfor %}",
    "{% if tools %}{{ raise_exception('no declarations') }}{% endif %}"],
    ids=["result", "declaration"])
def test_a_template_that_raises_while_probing_still_loads(raises_on):
    """The probe decides an optional capability, so a template that raises on it cannot refuse the model."""
    wrapper = ChatWrapper(Config(context_size=1024),
                          context=FakeContext(dict(TOOLS, chat_template=raises_on + TOOLS["chat_template"])))

    assert not wrapper.capabilities(tools=True).can_call_tools
    wrapper.begin("persona")
    assert wrapper.reply("hallo").finish().text == "reply"


def test_the_result_block_is_probed_once_not_per_result():
    """A property read must not render a template, so the answer is kept."""
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=1024), context=fake)
    after_construction = len(fake.rendered)
    for _ in range(5):
        assert wrapper.capabilities(tools=True).can_call_tools
    assert len(fake.rendered) == after_construction


def test_declaring_refuses_a_model_that_could_never_call():
    """A declaration is a promise, so the whole chain is checked when it is made.

    Failing at the first call instead would surface mid-reply, after the model
    was already told the tool exists.
    """
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")

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
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
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
        add_tools(wrapper, 
            [ToolSpec("clock", "Die Zeit.", {"type": "object", "properties": {}})])


def test_a_refused_declaration_leaves_the_conversation_alone():
    """begin validates before it resets, and that must include this check."""
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
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
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
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

    add_tools(wrapper, [weather_spec])          # the rest, where the template puts them

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
    add_tools(wrapper, registry.specs())
    once = rows(wrapper)

    add_tools(wrapper, registry.specs())
    add_tools(wrapper, registry.specs())

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

    assert add_tools(wrapper, registry.specs()) == 0
    assert rows(wrapper) == before          # nothing was written at all

    @registry.tool("Sagt das Wetter.")
    def weather(city: str) -> dict:
        return {}

    add_tools(wrapper, registry.specs())      # only the one it has not heard
    declared = _declared_in(fake, 0)
    assert declared.count("<decl>clock</decl>") == 1 and "<decl>weather</decl>" in declared

    wrapper.begin("persona")                     # a reset forgets what was said
    add_tools(wrapper, registry.specs())
    assert "<decl>clock</decl>" in _declared_in(fake, 0)


def test_a_turn_may_call_as_often_as_it_likes_by_default():
    """No ceiling unless one is asked for, so nothing is refused for counting."""
    ran = []
    registry = ToolRegistry()
    registry.tool(lambda: ran.append(1) or "12:00", description="Die Zeit.", name="clock")

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
    assert w.capabilities(tools=True).can_call_tools
    w.begin("s", tools=specs)
    assert_cache_reads_as_rendered(w, fake)
    for text in ("eins", "zwei", "drei"):
        w.reply(text).finish()
        assert_cache_reads_as_rendered(w, fake)
    w.inject("vier")
    assert_cache_reads_as_rendered(w, fake)


def test_a_history_carries_its_declarations_on_its_last_user_turn():
    w, fake = _moving()
    specs = [ToolSpec("uhr", "", {})]
    w.begin("s", [("user", "eins"), ("assistant", "zwei"), ("user", "drei"),
                  ("assistant", "vier")], tools=specs)
    assert_cache_reads_as_rendered(w, fake)
    w.reply("fünf").finish()
    assert_cache_reads_as_rendered(w, fake)


def test_a_later_declaration_joins_the_newest_user_turn():
    w, fake = _moving()
    w.begin("s", tools=[ToolSpec("uhr", "", {})])
    w.reply("eins").finish()
    assert add_tools(w, [ToolSpec("wetter", "", {})]) == 0
    assert_cache_reads_as_rendered(w, fake)
    w.reply("zwei").finish()
    assert_cache_reads_as_rendered(w, fake)


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

    registry.tool(weather, description="Wetter.")
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


@pytest.mark.parametrize("stem", [stem for stem in STEMS if expected(fixture(stem), "call_opening_text")])
def test_a_call_that_opens_with_plain_text_ends_a_reply_that_reads_no_call_where_it_opens(stem):
    """No token opens it, so none can be kept from being drawn; the reply ends there instead."""
    fake = FakeContext(fixture(stem))
    wrapper = ChatWrapper(Config(context_size=8192), context=fake)
    wrapper.begin("", tools=[WEATHER])
    reader = FakeContext(fixture(stem))
    _opener, body, _closer = written_call(ResolvedTemplate(reader), reader, WEATHER.name, {"city": "Bonn"})
    fake.replies = [ScriptedCall(body)]
    reply = wrapper.reply("Wetter?", tool_call_policy="withhold").finish()
    assert (reply.text, reply.stop_reason) == ("", "end_of_turn")
    assert (reply.tool_calls, wrapper.turn_open) == ([], False)
    assert WEATHER.name not in wrapper.conversation()["messages"][-1]["content"]


def _told_of_an_unreadable_call(policy, **options) -> str:
    """The result the model is told for an unreadable call under ``policy``."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=[ToolSpec("clock", "", {})])
    fake.call_script = ["fn:nonsense with no name"]
    wrapper.reply("los", tool_call_policy=policy, **options).finish()
    return _result_of(fake)


@pytest.mark.parametrize("policy, options", [
    ("yield", {}), (ToolRegistry(), {}), (ToolRegistry(), {"max_calls": 2})],
    ids=["yield", "registry", "registry-with-a-call-left"])
def test_the_answer_to_an_unreadable_call_says_why_and_to_try_again_where_another_call_is_read(
        policy, options):
    told = _told_of_an_unreadable_call(policy, **options)
    assert "the tool call failed: the call could not be read" in told
    assert "try again" in told and "do not call again" not in told


@pytest.mark.parametrize("policy, options", [("refuse", {}), (ToolRegistry(), {"max_calls": 1})],
                         ids=["refuse", "registry-at-its-last-call"])
def test_the_answer_to_an_unreadable_call_says_not_to_call_again_where_no_call_is_read_after_it(
        policy, options):
    told = _told_of_an_unreadable_call(policy, **options)
    assert "the call could not be read" in told and "do not call again" in told


def test_an_error_result_of_the_callers_own_is_told_as_a_registered_tools_is():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=weather_registry().specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    call, = [e for e in wrapper.reply("wetter?", tool_call_policy="yield")
             if isinstance(e, ToolCall)]
    own = {"error": "offline", "message": "kein Netz"}
    wrapper.inject_tool_result(call, own)
    assert "the tool call failed: kein Netz. try again" in newest_turn(fake)
    wrapper.reply(tool_call_policy="yield").finish()


def test_a_fault_while_a_call_is_read_is_raised_and_leaves_the_conversation_whole(monkeypatch):
    """Only what a written call can cause counts as an unreadable call; a fault in the library is raised."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ['fn:<fun=clock>{}</fun>']

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
    chat = _with_a_template_for_tools()[0]
    assert chat.capabilities(tools=True).can_call_tools
    assert not chat.capabilities(tools=False).can_call_tools
    without = {**TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN, "tool_use_template": None}
    assert not _with_a_template_for_tools(without)[0].capabilities(tools=True).can_call_tools


def test_a_tool_turn_in_a_history_begun_without_tools_is_refused_saying_to_pass_tools():
    chat, _fake = _with_a_template_for_tools()
    turn = tool_turn(ToolCall("clock", {}), Result("clock", "12:00"))
    history = [("user", "Wie spät?"), *turn, ("user", "Danke")]
    with pytest.raises(UnsupportedError, match="pass tools") as refused:
        chat.begin("persona", history)
    assert refused.value.field == "messages"
    chat.begin("persona", history, tools=[])


def test_a_change_of_tools_moves_the_conversation_between_its_templates_keeping_it():
    chat, fake = _with_a_template_for_tools()
    chat.begin("persona")
    chat.reply("hallo?").finish()
    messages = chat.conversation()["messages"]
    assert "<[say>" in held_text(fake) and "<[turn>" not in held_text(fake)
    chat.change_settings(tools=weather_registry().specs())
    assert "<[say>" not in held_text(fake) and "<decl>weather</decl>" in held_text(fake)
    assert chat.conversation()["messages"] == messages
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    reply = chat.reply("wetter?", tool_call_policy=weather_registry()).finish()
    assert reply.text == "Sonnig" and [call.name for call in reply.tool_calls] == ["weather"]
    chat.begin("persona")                       # without tools: the default template again
    assert "<[say>" in held_text(fake) and "<[turn>" not in held_text(fake)


def test_tools_handed_to_begin_select_the_template_for_tools_though_none_is_declared_yet():
    chat, fake = _with_a_template_for_tools()
    chat.begin("persona", tools=[])
    chat.reply("hallo?").finish()
    assert "<[say>" not in held_text(fake) and "<[turn>" in held_text(fake)
    assert chat.conversation()["tools"] == []
    resets = fake.resets
    add_tools(chat, weather_registry().specs())
    assert "<decl>weather</decl>" in held_text(fake) and "<[say>" not in held_text(fake)
    assert fake.resets == resets
    assert_consistent(chat, fake)


def test_the_template_a_conversation_is_written_with_survives_its_round_trip():
    chat, fake = _with_a_template_for_tools()
    chat.begin("persona", tools=[])
    assert chat.conversation()["tools"] == []
    assert_cache_reads_as_rendered(chat, fake)
    with_tools = held_text(fake)
    chat.begin(**chat.conversation())
    assert held_text(fake) == with_tools
    chat.begin("persona")
    assert chat.conversation()["tools"] is None
    without = held_text(fake)
    chat.begin(**chat.conversation())
    assert held_text(fake) == without != with_tools


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


def _counting_clock():
    """A registry whose one tool, ``clock``, notes each run in the list returned beside it."""
    ran = []
    registry = ToolRegistry()
    registry.tool(lambda: ran.append(1) or "12:00", description="Die Zeit.", name="clock")
    return registry, ran


def test_a_question_asked_while_a_turn_is_open_closes_it_and_runs_under_its_own_policy():
    """Two replies asked for before either runs: the second question closes the first one's turn, whose call is told it was not run."""
    registry, ran = _counting_clock()
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    opening = wrapper.reply("wie spät?", tool_call_policy="yield")
    later = wrapper.reply("und jetzt?", tool_call_policy=registry)
    call, = [event for event in opening if isinstance(event, ToolCall)]
    assert later.finish().text and not wrapper.turn_open and ran == []
    assert "not run" in held_text(fake)
    with pytest.raises(OutOfOrderError):
        wrapper.inject_tool_result(call, "12:00")       # the call was answered as not run


def test_a_turn_going_on_with_its_registry_keeps_what_is_left_of_its_max_calls():
    registry, ran = _counting_clock()
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}", "fn:clock{}", "fn:clock{}"]
    with wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=1) as stream:
        call = next(event for event in stream if isinstance(event, ToolCall))
    wrapper.inject_tool_result(call, "12:00")   # the one call the turn may use
    reply = wrapper.reply().finish()
    assert ran == [] and reply.tool_limit_reached and not wrapper.turn_open
    assert "the limit is 1" in held_text(fake)


def test_a_reply_given_no_policy_takes_the_policy_of_the_turn_open_when_it_runs():
    """Asked for before another reply opens a turn, it goes on with that turn under the turn's policy."""
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    wrapper.inject("wie spät?")
    waiting = wrapper.reply()
    fake.call_script = ["fn:clock{}"]
    call, = [event for event in wrapper.reply(tool_call_policy="yield") if isinstance(event, ToolCall)]
    wrapper.inject_tool_result(call, "12:00")
    fake.call_script = ["fn:clock{}"]
    handed = [event for event in waiting if isinstance(event, ToolCall)]
    assert handed and wrapper.pending_calls == handed and wrapper.turn_open


def test_a_turn_a_history_left_open_keeps_the_policy_of_the_first_reply_that_gives_one():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", [("user", "los"), *tool_turn(ToolCall("clock", {}))],
                  tools=[ToolSpec(name="clock", description="Die Zeit.",
                                       parameters={"type": "object", "properties": {}})])
    wrapper.inject_tool_result(wrapper.pending_calls[0], "12:00")
    fake.call_script = ["fn:clock{}"]
    first, = [event for event in wrapper.reply(tool_call_policy="yield")
              if isinstance(event, ToolCall)]
    wrapper.inject_tool_result(first, "12:01")
    fake.call_script = ["fn:clock{}"]
    handed = [event for event in wrapper.reply() if isinstance(event, ToolCall)]
    assert handed and wrapper.pending_calls == handed       # handed over, not refused


def test_a_failed_reply_that_gives_max_calls_leaves_the_turn_what_was_left_of_its_own():
    registry, ran = _counting_clock()
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    with wrapper.reply("wie spät?", tool_call_policy=registry, max_calls=1) as stream:
        call = next(event for event in stream if isinstance(event, ToolCall))
    wrapper.inject_tool_result(call, "12:00")   # the one call the turn may use
    failing = wrapper.reply(max_calls=5)
    fake.call_script = ["fn:clock{}"]
    yielded, = [event for event in wrapper.reply(tool_call_policy="yield")
                if isinstance(event, ToolCall)]
    with pytest.raises(ValueError):
        failing.finish()                        # a result is owed when it runs
    wrapper.inject_tool_result(yielded, "12:01")
    fake.call_script = ["fn:clock{}"]
    reply = wrapper.reply().finish()
    assert ran == [] and reply.tool_limit_reached and not wrapper.turn_open
    assert "the limit is 1" in held_text(fake)


def test_a_stream_closed_at_a_later_call_of_a_round_writes_the_results_of_those_that_ran():
    registry, ran = _counting_clock()
    wrapper, fake = tool_wrapper(context_size=1024, template=JSON_TOOLS)
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ['[{"name":"clock","arguments":{}},{"name":"clock","arguments":{}}]']
    with wrapper.reply("wie spät?", tool_call_policy=registry) as stream:
        handed = []
        for event in stream:
            if isinstance(event, ToolCall):
                handed.append(event)
                if len(handed) == 2:
                    break                       # closed before the second call runs
    assert ran == [1]
    _, second = handed
    assert wrapper.pending_calls == [second]    # the first ran, and its result is written
    wrapper.inject_tool_result(second, "12:01")
    assert wrapper.reply().finish().text == "reply" and ran == [1]
    assert_consistent(wrapper, fake)


def test_stopping_the_second_call_of_a_round_keeps_the_first_calls_result():
    ran = []
    registry = ToolRegistry()

    @registry.tool("Die Zeit.")
    def clock() -> str:
        ran.append(1)
        if len(ran) == 2:
            raise KeyboardInterrupt
        return "12:00"

    wrapper, fake = tool_wrapper(context_size=1024, template=JSON_TOOLS)
    wrapper.begin("persona", tools=registry.specs())
    fake.call_script = ['[{"name":"clock","arguments":{}},{"name":"clock","arguments":{}}]']
    with pytest.raises(KeyboardInterrupt):
        wrapper.reply("wie spät?", tool_call_policy=registry).finish()
    # The first call ran and keeps its result; the second is owed.
    assert [call.id for call in wrapper.pending_calls] == ["call00001"]
    assert wrapper.wait_idle(0)


def test_writing_a_result_renders_the_whole_conversation_only_to_measure_it():
    wrapper, fake = tool_wrapper(context_size=32768)
    history = [m for i in range(300) for m in (("user", f"question {i}"), ("assistant", f"answer {i}"))]
    wrapper.begin("persona", history, tools=weather_registry().specs())
    fake.call_script = ['fn:weather{"city": "Bonn"}']
    call, = [e for e in wrapper.reply("Wetter?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    before = len(fake.rendered)
    wrapper.inject_tool_result(call, "sonnig")
    # One split of the whole conversation, two renders, to see the result fits; the rest is the newest turn's.
    assert sum(1 for messages in fake.rendered[before:] if messages > 100) <= 2



def test_a_model_that_marks_calls_but_writes_no_result_reads_no_call_under_the_default():
    """Its call could never be answered, so it stays text, and the next question is answered."""
    fake = FakeContext(NO_RESULT_BLOCK)
    w = ChatWrapper(Config(context_size=512), context=fake)
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    reply = w.reply("wie spät?").finish()
    assert reply.stop_reason != "tool_call" and not w.turn_open and not w.pending_calls
    assert w.reply("und jetzt?").finish().stop_reason == "end_of_turn"


def test_a_call_that_spends_the_reply_cap_is_reported_by_its_name_though_not_run():
    clock = ToolRegistry()
    clock.tool(lambda: "12:00", description="Die Zeit.", name="uhr")

    def reply(policy, cap):
        fake = FakeContext(TOOLS)
        w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
        w.begin("s", tools=clock.specs())
        fake.call_script = ["fn:uhr{}"]
        return w, w.reply("Zeit?", tool_call_policy=policy, max_reply_tokens=cap).finish()
    # The cap the whole call takes: handed over complete under "yield" with nothing left.
    cap = next(cap for cap in range(4, 40)
               if (r := reply("yield", cap)[1]).tool_calls and r.tool_calls[0].status == "complete")
    w, run = reply(clock, cap)
    assert run.stop_reason == "token_limit" and not w.turn_open
    assert [(call.name, call.status) for call in run.tool_calls] == [("uhr", "unfinished")]


def test_a_batch_with_one_entry_that_is_no_call_runs_the_others_and_tells_the_model_of_that_one():
    """Each call of a batch is judged on its own: the good ones are answered, the bad one told it could not be read."""
    registry, ran = _counting_clock()
    fake = FakeContext(JSON_TOOLS)
    w = ChatWrapper(Config(context_size=2048), context=fake)
    w.begin("persona", tools=registry.specs())
    fake.call_script = ['[{"name": "clock", "arguments": {}}, {"no": "call"}, '
                        '{"name": "clock", "arguments": {}}]']
    reply = w.reply("wie spät?", tool_call_policy=registry).finish()
    assert len(ran) == 2
    assert [call.status for call in reply.tool_calls] == ["complete", "unreadable", "complete"]
    assert "could not be read" in held_text(fake) and not w.turn_open


def test_a_question_that_closes_a_batch_with_an_unreadable_entry_begins_again_the_same():
    fake = FakeContext(JSON_TOOLS)
    w = ChatWrapper(Config(context_size=2048), context=fake)
    w.begin("persona", tools=[])
    fake.call_script = ['[{"name": "clock", "arguments": {}}, {"no": "call"}]']
    list(w.reply("wie spät?", tool_call_policy="yield"))
    assert [call.name for call in w.pending_calls] == ["clock"]     # the unreadable one is not the caller's
    fake.call_script = []
    w.reply("egal, wie heißt du?").finish()
    saved = w.conversation()
    w.begin(**saved)
    assert w.conversation() == saved
