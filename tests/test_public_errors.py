"""After load, every public action raises only the library's own errors, each naming its cause, and a refused one changes nothing; every question still gets a reply.

The cases below include every input found to break this rule once.
"""

import re
import threading
import time

import pytest

from llama_chat import (ChatWrapper, Config, ContextOverflowError, InternalError, ToolRegistry,
                        ToolSpec)
from llama_chat.errors import (LIBRARY_ERRORS, ArgumentTypeError, InvalidArgumentError,
                               InvalidSettingError, OutOfOrderError, raising_library_errors_only)
from llama_chat.rendered_cache import RenderedCache
from tests.chat_messages import after_system, opened
from llama_chat.testing.fake_context import FakeContext, ScriptedCall
from llama_chat.testing.invented_templates import PLAIN, REASONING, TOOLS
from tests.wrappers import assert_cache_reads_as_rendered, tool_wrapper, weather_registry

_LONE = "a\ud800b"


def _public_members(cls):
    return [(name, member) for name, member in vars(cls).items() if not name.startswith("_")]


def test_every_public_method_and_property_of_the_wrapper_goes_through_the_one_entry_point():
    for name, member in _public_members(ChatWrapper):
        action = member.fget if isinstance(member, property) else member
        assert getattr(action, "raises_library_errors_only", False), \
            f"ChatWrapper.{name} does not go through raising_library_errors_only"


def test_a_fault_inside_an_action_is_an_internal_error_naming_it_and_changes_nothing(monkeypatch):
    wrapper, fake = tool_wrapper()
    wrapper.begin(opened("s"))
    wrapper.reply("eins").finish()
    before = wrapper.conversation()

    def broken(*_args, **_kwargs):
        raise KeyError("a fault of the library's own")
    monkeypatch.setattr(RenderedCache, "add", broken)
    with pytest.raises(InternalError, match=r"ChatWrapper\.inject failed inside the library"):
        wrapper.inject("Notiz")
    monkeypatch.undo()
    assert wrapper.conversation() == before
    assert wrapper.reply("zwei").finish().text


def test_a_fault_while_a_reply_streams_is_an_internal_error_and_the_next_question_is_answered(
        monkeypatch):
    """The reply keeps what it had produced, as any failed reply does (rule 6)."""
    wrapper, fake = tool_wrapper()
    wrapper.begin(opened("s"))
    monkeypatch.setattr("llama_chat.rendered_cache.ReplyText.feed",
                        lambda *_args: 1 / 0)
    with pytest.raises(InternalError, match="ZeroDivisionError"):
        wrapper.reply("eins").finish()
    monkeypatch.undo()
    assert wrapper.reply("zwei").finish().text
    assert wrapper.wait_idle(timeout_s=10)
    assert_cache_reads_as_rendered(wrapper, fake)


@pytest.mark.parametrize("call", [
    lambda w: w.reply("a", "b"), lambda w: w.inject(), lambda w: w.begin(nonsense=1),
    lambda w: w.capabilities(True), lambda w: w.wait_idle(1, 2)],
    ids=["reply", "inject", "begin", "capabilities", "wait_idle"])
def test_arguments_an_action_does_not_take_are_refused_by_name(call):
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("s"))
    with pytest.raises(ArgumentTypeError, match=r"ChatWrapper\.\w+"):
        call(wrapper)


# ----- inputs found to break the rule once ---------------------------------------
def _refusals():
    """``(id, action, error, field)``: an action on a begun conversation, and what it is refused with."""
    return [
        ("question-lone-surrogate", lambda w: w.reply(_LONE).finish(), InvalidArgumentError, "text"),
        ("inject-lone-surrogate", lambda w: w.inject(_LONE), InvalidArgumentError, "text"),
        ("begin-lone-surrogate", lambda w: w.begin(opened(_LONE)), InvalidArgumentError, "messages"),
        ("keep-text-lone-surrogate", lambda w: w.truncate_last_reply(_LONE),
         InvalidArgumentError, "keep_text"),
        ("capabilities-not-a-switch", lambda w: w.capabilities(tools="no"), ArgumentTypeError, None),
        ("reasoning-not-a-switch", lambda w: w.begin(opened("s"), reasoning="yes"),
         ArgumentTypeError, None),
        ("reply-cap-not-whole", lambda w: w.reply("a", max_reply_tokens=1.5),
         InvalidSettingError, "max_reply_tokens"),
    ]


@pytest.mark.parametrize("action, error, field", [case[1:] for case in _refusals()],
                         ids=[case[0] for case in _refusals()])
def test_an_input_no_model_can_read_is_refused_by_name_and_changes_nothing(action, error, field):
    wrapper, fake = tool_wrapper()
    wrapper.begin(opened("s"))
    wrapper.reply("eins").finish()
    before = wrapper.conversation()
    with pytest.raises(error) as refused:
        action(wrapper)
    assert isinstance(refused.value, LIBRARY_ERRORS)
    if field is not None:
        assert refused.value.field == field
    assert wrapper.conversation() == before
    assert wrapper.reply("zwei").finish().text


def test_the_reply_cap_refusal_says_what_the_cap_must_be():
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("s"))
    with pytest.raises(InvalidSettingError, match="whole number of at least 1"):
        wrapper.reply("a", max_reply_tokens=1.5)


class _BrokenMapping(dict):
    """A result whose own methods fail when JSON writes it."""

    def items(self):
        raise KeyError("broken")

    def __iter__(self):
        raise KeyError("broken")


def _deep(levels: int):
    value: list = []
    for _ in range(levels):
        value = [value]
    return value


@pytest.mark.parametrize("result", [_LONE, _BrokenMapping(a=1), _deep(50000), float("nan")],
                         ids=["lone-surrogate", "own-methods-fail", "deep", "nan"])
@pytest.mark.parametrize("how", ["handed over", "returned by a registered tool"])
def test_a_result_no_model_can_read_is_told_as_a_failed_call_and_the_turn_goes_on(result, how):
    wrapper, fake = tool_wrapper(context_size=2048)
    registry = ToolRegistry()
    registry.tool(lambda: result, description="Gibt etwas.", name="gib", parameters={})
    wrapper.begin(opened("s"), tools=registry.specs())
    fake.call_script = ["fn:gib{}"]
    if how == "handed over":
        calls = [step for step in wrapper.reply("los", tool_call_policy="yield")
                 if not isinstance(step, str)]
        wrapper.inject_tool_result(calls[0].id, result)
        reply = wrapper.reply().finish()
    else:
        reply = wrapper.reply("los", tool_call_policy=registry).finish()
    assert reply.text and not wrapper.turn_open
    told = fake.text_of(fake.cache)
    assert "unwritable_result" in told
    assert_cache_reads_as_rendered(wrapper, fake)


@pytest.mark.parametrize("parameters", [{1: "x"}, {"default": object()}, {"minimum": float("nan")},
                                        {"enum": ("a",)}, {"x": "\udfff"}])
def test_a_tool_spec_no_model_can_be_given_is_refused_where_it_is_built(parameters):
    with pytest.raises(InvalidArgumentError, match="parameters"):
        ToolSpec("f", "d", parameters)


def test_a_history_call_nested_past_the_limit_is_left_out_as_unreadable_not_raised():
    wrapper, fake = tool_wrapper(context_size=100000)
    arguments = '{"a":' + "[" * 600 + "]" * 600 + "}"
    wrapper.begin(opened("s", [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r"}]), tools=[])
    assert wrapper.reply("weiter").finish().text
    assert not any(m.get("tool_calls") for m in wrapper.conversation()["messages"])


def test_a_reply_asked_before_a_begin_replaced_its_conversation_is_refused_when_it_is_read():
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("s"))
    stream = wrapper.reply("q1")
    wrapper.begin(opened("ein anderes Gespräch"))
    with pytest.raises(OutOfOrderError, match="begin"):
        stream.finish()
    assert [m["content"] for m in after_system(wrapper.conversation()["messages"])] == []
    assert wrapper.reply("q2").finish().text


def test_a_reply_nobody_reads_on_is_ended_for_an_action_from_another_thread():
    fake = FakeContext(PLAIN, gen_tokens=20, gen_text="eine lange Antwort, Wort für Wort")
    wrapper = ChatWrapper(Config(context_size=512, unread_reply_timeout_s=0.2), context=fake)
    wrapper.begin(opened("s"))
    stream = wrapper.reply("hallo")
    next(stream)                            # read once, then left
    done = []
    acting = threading.Thread(target=lambda: done.append(wrapper.inject("Notiz")), daemon=True)
    started = time.monotonic()
    acting.start()
    acting.join(10)
    assert done and time.monotonic() - started < 5
    assert list(stream) == [] and stream.reply.stop_reason == "interrupted"
    assert wrapper.reply("weiter").finish().text


def test_a_tool_that_does_not_return_in_time_is_told_as_failed_and_the_reply_ends():
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=512, tool_timeout_s=0.2), context=fake)
    release, acted = threading.Event(), threading.Event()
    refusals = []

    def hangs():
        release.wait(30)
        try:
            wrapper.inject("spät")
        except OutOfOrderError as refused:
            refusals.append(str(refused))
        acted.set()
        return "spät"
    registry = ToolRegistry()
    registry.tool(hangs, description="Hängt.", name="haengt", parameters={})
    wrapper.begin(opened("s"), tools=registry.specs())
    fake.call_script = ["fn:haengt{}"]
    started = time.monotonic()
    reply = wrapper.reply("los", tool_call_policy=registry).finish()
    release.set()
    assert time.monotonic() - started < 10 and reply.text and not wrapper.turn_open
    assert "tool_timeout" in fake.text_of(fake.cache)
    # Acting on the wrapper once past its time, it is told why, not that the reply waits for it.
    assert acted.wait(10)
    assert len(refusals) == 1 and "did not return within tool_timeout_s" in refusals[0]


def test_a_model_that_calls_without_end_is_stopped_by_the_default_ceiling_and_answers():
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=8192, max_calls=3), context=fake)
    ran = []
    registry = ToolRegistry()
    registry.tool(lambda: ran.append(1) or "noch einmal", description="Zählt.", name="zaehl",
                  parameters={})
    wrapper.begin(opened("s"), tools=registry.specs())
    fake.replies = [ScriptedCall("fn:zaehl{}")] * 20 + ["Fertig."]
    reply = wrapper.reply("los", tool_call_policy=registry).finish()
    assert len(ran) == 3 and reply.text and not wrapper.turn_open



# ----- the review's finds -------------------------------------------------------
@pytest.mark.parametrize("name, template", [("plain", PLAIN), ("reasoning", REASONING)])
def test_a_long_question_is_answered_wherever_begin_left_room_for_a_short_one(name, template):
    for system in range(200, 250):
        wrapper = ChatWrapper(Config(context_size=256, min_reply_tokens=1, evict_above_fraction=1.0),
                              context=FakeContext(template))
        try:
            wrapper.begin(opened("s" * system))
        except ContextOverflowError:
            continue
        assert wrapper.reply("x" * 500 + " Frage?").finish().stop_reason, system


def test_without_a_time_limit_a_tool_runs_on_the_thread_that_reads_the_reply():
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=512, tool_timeout_s=None), context=fake)
    ran_on = []
    registry = ToolRegistry()
    registry.tool(lambda: ran_on.append(threading.current_thread()) or "ok",
                  description="Merkt sich den Faden.", name="faden", parameters={})
    wrapper.begin(opened("s"), tools=registry.specs())
    fake.call_script = ["fn:faden{}"]
    wrapper.reply("los", tool_call_policy=registry).finish()
    assert ran_on == [threading.current_thread()]


def test_a_huge_time_limit_is_waited_as_no_limit(monkeypatch):
    raised = []
    monkeypatch.setattr(threading, "excepthook", raised.append)
    fake = FakeContext(TOOLS, gen_tokens=20)
    wrapper = ChatWrapper(Config(context_size=512, tool_timeout_s=1e300,
                                 unread_reply_timeout_s=1e300), context=fake)
    registry = ToolRegistry()
    registry.tool(lambda: "ok", description="Gibt ok.", name="ok", parameters={})
    wrapper.begin(opened("s"), tools=registry.specs())
    fake.call_script = ["fn:ok{}"]
    assert wrapper.reply("los", tool_call_policy=registry).finish().text
    assert not wrapper.turn_open
    # An action that waits for a reply held unread arms the unread-reply timer with that limit.
    stream = wrapper.reply("noch was")
    next(stream)
    waiting = threading.Thread(target=lambda: wrapper.inject("Notiz"), daemon=True)
    waiting.start()
    deadline = time.monotonic() + 60
    while wrapper._unread_watch is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert wrapper._unread_watch is not None and waiting.is_alive()
    stream.finish()
    waiting.join(10)
    assert not waiting.is_alive() and raised == []


def test_closing_a_stream_another_thread_is_reading_is_refused_rather_than_waiting(settles_in_the_background):
    fake = FakeContext(PLAIN, gen_tokens=20)
    wrapper = ChatWrapper(Config(context_size=512, unread_reply_timeout_s=None), context=fake)
    wrapper.begin(opened("s"))
    first = wrapper.reply("eins")
    next(first)                             # holds the wrapper, unread
    second = wrapper.reply("zwei")
    reading = threading.Thread(target=lambda: next(second, None), daemon=True)
    reading.start()
    deadline = time.monotonic() + 60
    while not wrapper._blocked.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)                    # until its first step waits for the wrapper
    assert wrapper._blocked.is_set(), "the other thread never began its first step"
    with pytest.raises(OutOfOrderError, match="being read meanwhile"):
        second.close()
    first.close()
    reading.join(10)
    assert not reading.is_alive()
    second.close()
    assert wrapper.reply("drei").finish().text


def test_a_begin_after_a_fault_the_cache_check_found_writes_the_cache_anew(monkeypatch):
    fake = FakeContext(PLAIN)
    wrapper = ChatWrapper(Config(context_size=512, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    wrapper.reply("eins").finish()
    wrapper.wait_idle()
    wrapper._conversation._found_fault = InternalError("a fault found after the reply")
    wrapper.begin(opened("s"))              # the remedy, not refused for what it remedies
    assert wrapper.reply("zwei").finish().text


def test_a_refused_begin_leaves_the_fault_the_cache_check_found_to_be_reported():
    fake = FakeContext(PLAIN)
    wrapper = ChatWrapper(Config(context_size=512, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    wrapper.reply("eins").finish()
    wrapper.wait_idle()
    wrapper._conversation._found_fault = InternalError("a fault found after the reply")
    with pytest.raises(ContextOverflowError):
        wrapper.begin(opened("s" * 600))
    with pytest.raises(InternalError, match="a fault found after the reply"):
        wrapper.reply("zwei").finish()


@pytest.mark.parametrize("action", ["inject", "close"])
def test_an_action_on_the_thread_that_read_a_reply_another_thread_has_begun_to_step_waits_for_it(action):
    fake = FakeContext(PLAIN, gen_tokens=20)
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin(opened("s"))
    stream = wrapper.reply("eins")
    next(stream)                            # this thread holds the wrapper, waiting to read on
    stepping, finished = threading.Event(), []

    def read_on():
        with stream._stepping:              # a step of the other thread has begun
            stepping.set()
            time.sleep(0.2)
        finished.append(stream.finish())
    other = threading.Thread(target=read_on, daemon=True)
    other.start()
    assert stepping.wait(10)
    if action == "inject":
        wrapper.inject("Notiz")             # waits for the other thread's reply, not refused
    else:
        wrapper.close()
    other.join(10)
    assert finished and finished[0] is not None and finished[0].stop_reason != "interrupted"
    if action == "inject":
        assert after_system(wrapper.conversation()["messages"])[-1]["content"] == "Notiz"


def test_a_begin_that_fails_before_it_starts_leaves_the_fault_the_cache_check_found(monkeypatch):
    fake = FakeContext(PLAIN)
    wrapper = ChatWrapper(Config(context_size=512, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    wrapper.reply("eins").finish()
    wrapper.wait_idle()
    wrapper._conversation._found_fault = InternalError("a fault found after the reply")
    memory = wrapper._conversation._memory

    def failing():
        raise RuntimeError("the cache could not be brought up")
    monkeypatch.setattr(memory, "repair", failing)
    with pytest.raises(Exception, match="could not be brought up"):
        wrapper.begin(opened("s"))
    monkeypatch.undo()
    with pytest.raises(InternalError, match="a fault found after the reply"):
        wrapper.reply("zwei").finish()


def _before_begin():
    return ChatWrapper(Config(context_size=500), context=FakeContext(PLAIN))


def _on_an_open_turn():
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=1024), context=fake)
    wrapper.begin(opened("s"), tools=weather_registry().specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    wrapper.reply("Wetter?", tool_call_policy="yield").finish()
    return wrapper


@pytest.mark.parametrize("situation, role, first", [
    (_before_begin, "user", r"begin\(\)"),
    (_before_begin, "assistant", r"begin\(\)"),
    (_on_an_open_turn, "assistant", r"inject_tool_result\(\)"),
], ids=["question-before-begin", "assistant-before-begin", "assistant-on-an-open-turn"])
def test_an_injection_refused_for_its_order_names_the_injection_and_what_comes_first(
        situation, role, first):
    wrapper = situation()
    before = wrapper.conversation()
    with pytest.raises(OutOfOrderError) as refused:
        wrapper.inject("Notiz", role=role)
    message = str(refused.value)
    assert re.search(r"\binject", message, re.IGNORECASE) and re.search(first, message)
    assert "None" not in message
    assert wrapper.conversation() == before


def test_a_fault_inside_a_property_is_an_internal_error_naming_the_property(monkeypatch):
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("s"))
    before = wrapper.conversation()

    def broken(*_args, **_kwargs):
        raise KeyError("a fault of the library's own")
    monkeypatch.setattr(RenderedCache, "published_total_tokens", broken)
    with pytest.raises(InternalError, match=r"ChatWrapper\.total_tokens failed inside the library"):
        _ = wrapper.total_tokens
    monkeypatch.undo()
    assert wrapper.conversation() == before


def test_a_guarded_property_keeps_its_setter_its_deleter_and_its_own_text():
    class Holder:
        def __init__(self):
            self.stored = 1

        def _get(self):
            """The getter's own text."""
            return self.stored

        def _set(self, value):
            self.stored = value

        def _delete(self):
            self.stored = None

        held = property(_get, _set, _delete, "The property's own text.")

    guarded = raising_library_errors_only(Holder)
    holder = guarded()
    holder.held = 2
    assert holder.held == 2
    del holder.held
    assert holder.stored is None
    assert guarded.__dict__["held"].__doc__ == "The property's own text."
    assert guarded.__dict__["held"].fget.raises_library_errors_only


# ----- for tests/test_public_errors.py ----------------------------------------------
def test_a_fault_inside_a_tool_loop_is_an_internal_error_naming_the_step_the_caller_took(monkeypatch):
    """A method of the wrapper's own, not the caller's, is never named as the action."""
    wrapper, fake = tool_wrapper(context_size=2048)
    registry = weather_registry()
    wrapper.begin(opened("s"), tools=registry.specs())
    fake.call_script = ["fn:weather{city:'Bonn'}"]

    def broken(*_args, **_kwargs):
        raise KeyError("a fault of the library's own")
    monkeypatch.setattr(RenderedCache, "add_result", broken)
    with pytest.raises(InternalError) as failed:
        wrapper.reply("Wetter?", tool_call_policy=registry).finish()
    monkeypatch.undo()
    assert re.match(r"ReplyStream\.__next__ failed inside the library \(KeyError", str(failed.value))
    assert wrapper.reply("noch einmal").finish().text
    assert wrapper.wait_idle(timeout_s=10)
    assert_cache_reads_as_rendered(wrapper, fake)
