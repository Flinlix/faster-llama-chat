"""Every action that is refused or fails leaves the conversation and the cache as they were before it, a reply keeping only what it produced, and usable."""

import random
import dataclasses
import json
import datetime as real_datetime
import threading

import pytest

from llama_chat.config import generation_sampling
from llama_chat import (EngineError, InternalError, TOOL_CALL_POLICIES, ChatWrapper, Config, ContextOverflowError,
                        TemplateRefusedError, eviction, rendered_cache)
from llama_chat.rendered_cache import RenderedCache
from llama_chat.rendering import ConversationRendering
from llama_chat.tool_registry import ToolRegistry
from llama_chat.tools import ToolCall, ToolSpec
from tests.chat_messages import Result, tool_turn, opened, after_system
from tests.chat_templates import STEMS, expected, fixture
from llama_chat.testing.fake_context import FakeContext
from tests import internals
from tests.internals import generated_turns_held
from llama_chat.testing.invented_templates import (PLAIN, REASONING, REFUSES_TWO_USER_TURNS, TOOLS,
                                      TOOLS_REFUSING_TWO_USER_TURNS, NO_DECLARATIONS, WRITES_TWO_QUESTIONS_REFUSES_A_TEXT)
from tests.wrappers import (UNGUARDED, add_tools, assert_cache_reads_as_rendered, assert_consistent, held_text,
                            numbered_exchanges,
                            newest_turn, rows, tool_wrapper, weather_registry, with_check_first)
from llama_chat.cache_sync import CacheSync
from llama_chat.generation import GenerationAccumulator


class _Flaky(FakeContext):
    """A fake cache whose next decode, cut or generation fails once, or whose operation number ``fail_at`` fails; a failed decode lands some of its tokens first."""

    fail = None             # "decode", "cut" or "generate"
    fail_at = None          # counted over every operation from 0
    operations = 0

    def _fails(self, kind=None) -> bool:
        """Whether this operation fails: the next of ``kind`` asked to, or operation number ``fail_at``."""
        self.operations += 1
        if (kind is not None and self.fail == kind) or self.operations - 1 == self.fail_at:
            self.fail = self.fail_at = None
            return True
        return False

    def prefill(self, token_ids, start_pos, want_logits):
        if self._fails("decode"):
            self.cache.extend(token_ids[:len(token_ids) // 2])
            raise EngineError("decode failed")
        super().prefill(token_ids, start_pos, want_logits)

    def decode_generated(self, token_ids, start_pos):
        if self._fails():
            self.cache.extend(token_ids)
            raise EngineError("decode failed")
        super().decode_generated(token_ids, start_pos)

    def remove_from(self, pos):
        if self._fails("cut"):
            raise EngineError("the cache refused to remove its tokens")
        super().remove_from(pos)

    def apply_eviction(self, start, end):
        if self._fails():
            raise EngineError("the cache refused to shift")
        super().apply_eviction(start, end)

    def reset(self):
        if self._fails():
            raise EngineError("the cache refused to empty")
        super().reset()

    def save_state(self):
        if self._fails():
            raise EngineError("the cache refused to save its state")
        return super().save_state()

    def restore_state(self, state, pos):
        if self._fails():
            raise EngineError("the cache refused the saved state")
        super().restore_state(state, pos)

    def save_whole_state(self):
        if self._fails():
            raise EngineError("the cache refused to save its state")
        return super().save_whole_state()

    def restore_whole_state(self, state):
        if self._fails():
            self.cache = self.cache[:len(self.cache) // 2]      # left as no saved point holds it
            raise EngineError("the cache refused the saved state")
        super().restore_whole_state(state)

    def sampled_tokens(self, *args, **kwargs):
        if self.fail == "generate":
            self.fail = None
            self.cache.append(ord("x"))
            raise EngineError("generation failed")
        for token in super().sampled_tokens(*args, **kwargs):
            if self._fails():
                raise EngineError("generation failed")
            yield token


def _flaky(**kw):
    fake = _Flaky(PLAIN, **kw)
    return ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake), fake


# ----- every action, failing at every point ----------------------------------------
_CALL, _CALLED = "fn:uhr{}", ToolCall("uhr", {})
_CALENDAR = ToolSpec("kalender", "Die Termine.", {"type": "object", "properties": {}})
_CACHES = {"shifts": dict(can_shift=True), "cuts": dict(can_shift=False, can_cut_tail=True),
           "neither": dict(can_shift=False, can_cut_tail=False),
           # Each token's text shown before it is decoded, as on the processor alone.
           "shifts, shown first": dict(can_shift=True, decode_blocks=True),
           "neither, shown first": dict(can_shift=False, can_cut_tail=False, decode_blocks=True)}
_CLOCK = ToolRegistry()
_CLOCK.tool(lambda: "12:00", description="Die Zeit.", name="uhr")


def _begin(*history):
    """A step that begins the conversation on ``history``, a clock its one tool."""
    def begin(w, fake):
        w.begin(opened("persona", list(history)), tools=_CLOCK.specs(),
                reasoning=bool(getattr(fake, "reasons", False)))
    return begin


def _exchanged(w, fake):
    w.reply("eins").finish()


def _injected(w, fake):
    _exchanged(w, fake)
    w.inject("Lies das.")


def _owing(w, fake):
    _exchanged(w, fake)
    fake.call_script = [_CALL]
    list(w.reply("Wetter?", tool_call_policy="yield"))


def _answered(w, fake):
    _owing(w, fake)
    w.inject_tool_result(w.pending_calls[0].id, "sonnig")


def _replying(text=None, script=None, **options):
    def act(w, fake, shown):
        fake.call_script = [script] if script else []
        for event in w.reply(text, **options):
            if isinstance(event, str):
                shown.append(event)
    return act


# What each action is taken from, what it does, and for a reply the calls its model makes.
_ACTIONS = {
    "begin": (_exchanged, lambda w, f, s: w.begin(opened("neu", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])), None),
    "begin while a call is owed": (_owing, lambda w, f, s: w.begin(opened("neu")), None),
    "inject": (_exchanged, lambda w, f, s: w.inject("Notiz dazu."), None),
    "inject_tool_result": (_owing, lambda w, f, s: w.inject_tool_result(w.pending_calls[0].id, "sonnig"),
                           None),
    "truncate_last_reply": (_exchanged, lambda w, f, s: w.truncate_last_reply("Ge"), None),
    "declare_tools": (_exchanged, lambda w, f, s: add_tools(w, [_CALENDAR]), None),
    "reply": (_exchanged, _replying("zwei"), []),
    "reply to the first question": (_begin(), _replying("zwei"), []),
    "reply to an injected question": (_injected, _replying(), []),
    "reply to the question a history ends on": (
        _begin({"role": "user", "content": "alt"}, {"role": "assistant", "content": "älter"}, {"role": "user", "content": "Und?"}), _replying(), []),
    "reply refusing a call": (_exchanged, _replying("Wetter?", _CALL), [_CALLED]),
    "reply running a call": (_exchanged, _replying("Wetter?", _CALL, tool_call_policy=_CLOCK), [_CALLED]),
    "reply yielding a call": (_exchanged, _replying("Wetter?", _CALL, tool_call_policy="yield"), [_CALLED]),
    "reply going on after a result": (_answered, _replying(tool_call_policy="yield"), []),
    "a question closing an open turn": (_owing, _replying("Neu?"), []),
    "inject an assistant message": (_exchanged, lambda w, f, s: w.inject("Gesagt.", role="assistant"),
                                    None),
    "begin again from the conversation": (_answered, lambda w, f, s: w.begin(**w.conversation()),
                                          None),
    "a long question evicting the oldest": (_exchanged, _replying("Eine lange Frage: " + "x" * 30),
                                            []),
    "a long question after an unanswered one": (_injected, _replying("Lang: " + "y" * 40), []),
}


def _state(w):
    return w.conversation(), w.pending_calls, w.turn_open, w.total_tokens


def _steps(history):
    """A history as one list: each message, a reply that called tools as its text, then each call as a ToolCall."""
    return [step for message in history for step in (
        [{"role": "assistant", "content": message["content"] or ""},
         *(ToolCall(c["function"]["name"], json.loads(c["function"]["arguments"]), c["id"])
           for c in message["tool_calls"])]
        if message.get("tool_calls") else [message])]


def _assert_kept_only_what_it_produced(before, after, shown, calls):
    """A failed reply loses nothing but the oldest exchanges a result evicted for room, and adds nothing, or what it produced: the text it showed, then a call or a result."""
    assert before[0]["messages"][:1] == after[0]["messages"][:1]       # the system message
    held, kept = (_steps(after_system(before[0]["messages"])),
                  _steps(after_system(after[0]["messages"])))
    # A question the reply asked after an unanswered one may have joined it.
    if held and isinstance(held[-1], dict) and held[-1]["role"] == "user":
        held[-1] = next((step for step in kept if isinstance(step, dict) and step["role"] == "user"
                         and step["content"].startswith(held[-1]["content"] + "\n\n")), held[-1])
    starts = [i for i, step in enumerate(held) if isinstance(step, dict) and step["role"] == "user"]
    evicted = next((i for i in [0] + starts + [len(held)] if kept[:len(held) - i] == held[i:]),
                   None)
    assert evicted is not None
    held = held[evicted:]
    added = kept[len(held):]
    texts = [step["content"] for step in added
             if isinstance(step, dict) and step["role"] == "assistant"]
    assert "".join(texts) == "".join(shown)
    made = [dataclasses.replace(step, id="") for step in added if isinstance(step, ToolCall)]
    assert made == calls[:len(made)]
    last = added[-1] if added else None
    assert not isinstance(last, dict) or last["role"] == "tool" \
        or (last["role"] == "assistant" and last["content"]), \
        "the failed reply left a question or an empty reply behind"


def _go_on(w, fake):
    """Use the conversation as it was left: answer what is owed, close an open turn, then reply."""
    fake.call_script = []
    for call in w.pending_calls:
        w.inject_tool_result(call.id, "sonnig")
    if w.turn_open:
        w.reply(tool_call_policy="yield").finish()
    history = after_system(w.conversation()["messages"])
    waiting = bool(history) and isinstance(history[-1], dict) and history[-1]["role"] == "user"
    w.reply(None if waiting else "weiter").finish()


@pytest.mark.parametrize("action", sorted(_ACTIONS))
@pytest.mark.parametrize("cache", sorted(_CACHES))
def test_nothing_an_action_failing_at_any_point_leaves_blocks_a_question(cache, action):
    """Whatever state a failure leaves, owed calls and open turns included, the next question is answered at once."""
    prepare, act, _calls = _ACTIONS[action]
    for template in (TOOLS, TOOLS_REFUSING_TWO_USER_TURNS):
        def prepared(template=template):
            fake = _Flaky(template, gen_tokens=5, gen_text="Gern.", **_CACHES[cache])
            w = ChatWrapper(Config(context_size=160, evict_above_fraction=0.6,
                                   min_reply_tokens=20), context=fake)
            _begin({"role": "user", "content": "alt"}, {"role": "assistant", "content": "älter"})(w, fake)
            prepare(w, fake)
            fake.operations = 0
            return w, fake

        w, fake = prepared()
        act(w, fake, [])
        points = fake.operations
        for at in range(points + 1):
            w, fake = prepared()
            fake.fail_at = at if at < points else None
            try:
                act(w, fake, [])
            except RuntimeError:
                pass
            fake.call_script = []
            reply = w.reply("Und jetzt?").finish()
            assert reply.text == "Gern." and not w.turn_open, (template is TOOLS, at)
            assert_cache_reads_as_rendered(w, fake)
            w.close()


@pytest.mark.parametrize("action", sorted(_ACTIONS))
@pytest.mark.parametrize("cache", sorted(_CACHES))
def test_an_action_failing_at_any_point_leaves_everything_as_it_was_and_usable(cache, action):
    """A reply keeps what it produced before the failure and nothing else; any other action changes nothing."""
    prepare, act, calls = _ACTIONS[action]
    for refuses_two_user_turns, template, reasons in (
            (False, TOOLS, False), (True, TOOLS_REFUSING_TWO_USER_TURNS, False),
            (False, REASONING, True)):
        def prepared(template=template, reasons=reasons):
            fake = _Flaky(template, gen_tokens=5, gen_text="Gern.", **_CACHES[cache])
            fake.reasons = reasons
            w = ChatWrapper(Config(context_size=160, evict_above_fraction=0.6,
                                   min_reply_tokens=20), context=fake)
            _begin({"role": "user", "content": "alt"}, {"role": "assistant", "content": "älter"})(w, fake)
            prepare(w, fake)
            fake.operations = 0
            return w, fake

        w, fake = prepared()
        act(w, fake, [])
        done, points = _state(w), fake.operations
        for at in range(points):
            w, fake = prepared()
            before, shown = _state(w), []
            fake.fail_at = at
            try:
                act(w, fake, shown)
            except RuntimeError:
                assert fake.operations == at + 1, "the cache was used again after it failed"
                if calls is None:
                    assert _state(w) == before, (refuses_two_user_turns, at)
                else:
                    _assert_kept_only_what_it_produced(before, _state(w), shown, calls)
            else:
                assert _state(w) == done, (refuses_two_user_turns, at)      # recovered, as if nothing failed
            _go_on(w, fake)
            assert_cache_reads_as_rendered(w, fake)
            w.close()                   # checked here, not again after the test


# ----- restarts --------------------------------------------------------------------
def test_a_refused_restart_keeps_the_previous_conversation_usable():
    w, fake = _flaky(gen_tokens=2)
    w.begin(opened("s"))
    w.reply("eins").finish()
    before = after_system(w.conversation()["messages"])
    with pytest.raises(ContextOverflowError):
        w.begin(opened("x" * 600))
    assert w.conversation()["messages"][0] == {"role": "system", "content": "s"}
    assert after_system(w.conversation()["messages"]) == before
    w.reply("zwei").finish()
    assert [r["text"] for r in rows(w)][-2:] == ["zwei", "re"]


@pytest.mark.parametrize("stem, system", [
    pytest.param(stem, system, id=f"{stem}-system-prompt" if system else stem)
    for stem in STEMS if expected(fixture(stem), "date_format")
    for system in ("", "Be brief.")[:1 + expected(fixture(stem), "can_take_system_prompt")]])
def test_a_refused_restart_on_a_template_that_writes_the_date_keeps_the_conversation(
        stem, system, clock):
    context = FakeContext(fixture(stem), ["Eins.", "Zwei."])
    w = ChatWrapper(Config(context_size=8192, min_reply_tokens=8), context=context)
    w.begin(opened(system))
    w.reply("One?").finish()
    clock[0] = real_datetime.datetime(2026, 11, 8, 0, 1)
    with pytest.raises(ContextOverflowError):
        w.begin(opened("x" * 9000))
    assert after_system(w.conversation()["messages"]) == [{"role": "user", "content": "One?"},
                                            {"role": "assistant", "content": "Eins."}]
    assert w.reply("Two?").finish().text == "Zwei."


def test_a_failed_restart_keeps_the_call_its_conversation_owes_and_nothing_else():
    w, fake = tool_wrapper()
    w.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    with pytest.raises(ContextOverflowError):
        w.begin(opened("x" * 600))
    assert w.turn_open and w.pending_calls == [call]
    w.inject_tool_result(call.id, "12:00")
    w.reply().finish()
    assert not w.turn_open


# ----- results ---------------------------------------------------------------------
def test_a_result_that_leaves_no_room_for_the_reply_closes_the_turn_and_the_conversation_goes_on():
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=300, evict_above_fraction=1.0, min_reply_tokens=40),
                    context=fake)
    w.begin(opened("s"))
    fake.call_script = ["fn:clock{}"]
    for n in range(100, 300):
        w.begin(opened("s"))
        try:
            calls = [e for e in w.reply("q" * n, tool_call_policy="yield") if isinstance(e, ToolCall)]
        except ContextOverflowError:
            break
        w.inject_tool_result(calls[0].id, "12:00")
        if not w.turn_open:
            assert w.pending_calls == []
            with pytest.raises(ValueError, match="nothing to answer"):
                w.reply().finish()           # never resumed without its result
            fake.call_script = []
            w.reply("weiter").finish()
            return
        fake.call_script = ["fn:clock{}"]
    pytest.fail("no question length left the result without room")


def test_a_turn_closing_for_want_of_room_that_fails_part_way_leaves_the_call_owed():
    def owing():
        fake = _Flaky(TOOLS)
        w = ChatWrapper(Config(context_size=300, evict_above_fraction=1.0, min_reply_tokens=40),
                        context=fake)
        w.begin(opened("s"))
        fake.call_script = ["fn:clock{}"]
        call, = [e for e in w.reply("q" * 210, tool_call_policy="yield") if isinstance(e, ToolCall)]
        fake.operations = 0
        return w, fake, call

    w, fake, call = owing()
    w.inject_tool_result(call.id, "12:00")
    assert not w.turn_open and fake.operations > 0     # the turn closes, the cache settled after it
    for at in range(fake.operations):
        w, fake, call = owing()
        before = _state(w)
        fake.fail_at = at
        with pytest.raises(RuntimeError):
            w.inject_tool_result(call.id, "12:00")
        assert _state(w) == before
        _go_on(w, fake)
        assert_cache_reads_as_rendered(w, fake)
        w.close()


@pytest.mark.parametrize("next_action", ["going-on", "question", "result", "injected question",
                                         "refused result"])
def test_a_result_the_engine_failed_to_write_is_written_by_the_next_action_and_not_lost(next_action):
    fake = _Flaky(TOOLS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    ran = []

    def clock():
        ran.append("clock")
        fake.fail = "decode"            # writing its result fails
        return "12:00"
    registry = ToolRegistry()
    registry.tool(clock, description="Die Zeit.", name="clock")
    fake.call_script = ["fn:clock{}"]
    with pytest.raises(EngineError):
        w.reply("wie spät?", tool_call_policy=registry).finish()
    assert ran == ["clock"] and w.turn_open
    fake.call_script = []
    if next_action == "refused result":
        before = _state(w)
        with pytest.raises(ValueError):
            w.inject_tool_result("no-such-call", "13:00")
        assert _state(w) == before                      # rule 6: the kept answer waits on
        w.reply().finish()
    elif next_action == "result":
        w.inject_tool_result(w.pending_calls[0].id, "13:00")    # the caller's own answer wins
    elif next_action == "injected question":
        w.inject("und jetzt?")
    else:
        w.reply(None if next_action == "going-on" else "und jetzt?").finish()
    assert [m["content"] for m in w.conversation()["messages"] if m["role"] == "tool"] == [
        "13:00" if next_action == "result" else "12:00"]
    assert ran == ["clock"]
    assert_cache_reads_as_rendered(w, fake)


@pytest.mark.parametrize("cache", sorted(_CACHES))
def test_a_fault_of_the_library_partway_through_a_drop_leaves_the_cache_as_it_was(cache):
    """Rule 6 holds for an InternalError too: the edit the drop made before it is undone at once."""
    fake = FakeContext(PLAIN, gen_tokens=5, **_CACHES[cache])
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.9), context=fake)
    w.begin(opened("s", numbered_exchanges(14)))
    memory, faults = w._conversation._memory, []
    removing = memory.remove

    def remove_then_fault(start, end):
        removing(start, end)
        if faults:
            raise InternalError("a fault forced after the cache was edited")
    memory.remove = remove_then_fault
    for i in range(10):
        before = _state(w)
        faults.append(True)
        try:
            w.inject(f"Frage {i} " + "A" * 40)
            w.inject("B" * 40, role="assistant")
        except InternalError:
            faults.clear()
            assert _state(w) == before and fake.cache == internals.held_tokens(w)
            _go_on(w, fake)
            return
        faults.clear()
    pytest.fail("no inject evicted")


_UNWRITABLE = [{"when": real_datetime.date(2026, 11, 8)}, {"t": float("nan")},
               [float("inf")]]


@pytest.mark.parametrize("result", _UNWRITABLE, ids=["date", "nan", "infinity"])
def test_a_result_handed_over_that_cannot_be_written_reaches_the_model_as_a_failure(
        result, caplog):
    w, fake = tool_wrapper()
    w.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    with caplog.at_level("WARNING", logger="llama_chat"):
        w.inject_tool_result(call.id, result)
    assert "unwritable_result" in held_text(fake) and "NaN" not in held_text(fake)
    assert w.pending_calls == [] and "unwritable_result" in caplog.text
    w.reply().finish()


@pytest.mark.parametrize("returned", _UNWRITABLE, ids=["date", "nan", "infinity"])
def test_a_tool_that_returns_what_cannot_be_written_is_answered_with_an_error(returned):
    registry = ToolRegistry()
    registry.tool(lambda: returned, description="Das Datum.", name="clock",
                  parameters={"type": "object", "properties": {}})
    w, fake = tool_wrapper()
    w.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    reply = w.reply("welcher Tag?", tool_call_policy=registry).finish()
    assert reply.stop_reason == "end_of_turn" and not w.turn_open
    assert "unwritable_result" in held_text(fake) and "NaN" not in held_text(fake)


def test_a_resumed_turn_cancelled_before_it_starts_stays_open_and_goes_on_later():
    w, fake = tool_wrapper()
    w.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call.id, "12:00")
    before = _state(w)
    cancel = threading.Event()
    cancel.set()
    reply = w.reply(cancel=cancel).finish()
    assert (reply.text, reply.stop_reason) == ("", "interrupted") and _state(w) == before
    fake.call_script = []
    w.reply(tool_call_policy="yield").finish()
    assert not w.turn_open and generated_turns_held(w) and "fn:clock{}" in held_text(fake)


def test_an_action_whose_render_cannot_place_a_generated_turn_is_refused_and_changes_nothing(
        monkeypatch):
    w, fake = tool_wrapper()
    w.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call.id, "12:00")
    w.reply().finish()
    before, held = w.conversation(), list(fake.cache)
    with monkeypatch.context() as patched:
        # A template that writes the turn where it cannot be told apart, as none of the test
        # templates does.
        patched.setattr(ConversationRendering, "place", lambda *_args: None)
        with pytest.raises(ValueError, match="turn that called tools"):
            add_tools(w, [ToolSpec("uhr", "Die Uhrzeit.", {})])
    assert w.conversation() == before and fake.cache == held
    add_tools(w, [ToolSpec("uhr", "Die Uhrzeit.", {})])


# ----- questions --------------------------------------------------------------------
def test_a_begin_that_would_leave_no_question_room_for_its_reply_is_refused_and_changes_nothing():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=86, evict_above_fraction=1.0, min_reply_tokens=60),
                    context=fake)
    w.begin(opened(""))
    w.reply("hi").finish()
    before = w.conversation()
    with pytest.raises(ContextOverflowError, match="room beside the reply's"):
        w.begin(opened("ein System, das zu lang ist"))
    assert w.conversation() == before
    assert w.reply("hello").finish().text


# ----- a cache that cannot shift ----------------------------------------------------
def test_one_turn_that_drops_twice_on_a_cache_that_cannot_shift_brings_it_back_once(
        monkeypatch):
    monkeypatch.setattr(eviction, "EVICT_TO_FRACTION_WITHOUT_SHIFT", 0.2)
    fake = FakeContext(PLAIN, gen_tokens=2, can_shift=False)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5), context=fake)
    w.begin(opened("s"))
    for i in range(8):
        before = fake.resets + fake.edit_count("restore")
        w.reply(f"message number {i} with some length").finish()
        assert fake.resets + fake.edit_count("restore") - before <= 1
        assert_consistent(w, fake)


# ----- reading from another thread ------------------------------------------------
def test_the_history_read_from_another_thread_is_never_half_built(monkeypatch):
    w, fake = tool_wrapper()
    w.begin(opened("persona", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]))
    seen = []
    split = ConversationRendering.split

    def reading(self, messages, generation_prompt):
        seen.append(len(after_system(w.conversation()["messages"])))           # as another thread reads it, mid-action
        return split(self, messages, generation_prompt)

    monkeypatch.setattr(ConversationRendering, "split", reading)
    w.reply("c").finish()
    # Before the question, with the question until its reply is recorded, then with both.
    assert set(seen) <= {2, 3, 4}


def test_the_token_count_and_the_open_turn_read_from_another_thread_are_a_finished_changes(
        monkeypatch):
    """Read in the middle of a begin that fails, they are what the conversation it leaves standing holds."""
    w, fake = tool_wrapper()
    w.begin(opened("persona", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]))
    before, seen = (w.total_tokens, w.turn_open), []
    split = ConversationRendering.split

    def reading(self, messages, generation_prompt):
        seen.append((w.total_tokens, w.turn_open))            # as another thread reads them
        return split(self, messages, generation_prompt)

    def failing(token_ids, start_pos, want_logits):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(ConversationRendering, "split", reading)
    fake.prefill = failing
    with pytest.raises(RuntimeError):
        w.begin(opened("persona", [{"role": "user", "content": "x"}, *tool_turn(ToolCall("weather", {"city": "Bonn"}))]), tools=weather_registry().specs())
    del fake.prefill
    assert seen and set(seen) == {before}
    assert (w.total_tokens, w.turn_open) == before


@pytest.mark.parametrize("fault", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("ending, step", [("answer", "_holds_thought_tags"),
                                          ("call", "_drop_unplaceable"),
                                          ("going on after a result", "_drop_unplaceable")])
def test_a_reply_whose_recording_fails_is_taken_back_with_its_question(monkeypatch, fault, ending,
                                                                       step):
    w, fake = tool_wrapper()
    w.begin(opened("persona"), tools=weather_registry().specs())
    w.reply("eins").finish()
    if ending == "going on after a result":
        fake.call_script = ['fn:weather{"city": "Bonn"}']
        list(w.reply("Wetter?", tool_call_policy="yield"))
        w.inject_tool_result(w.pending_calls[0].id, "sonnig")
    before = _state(w)
    recorded = getattr(rendered_cache.RenderedCache, step)
    failing = iter([True])

    def recording(self, *args):
        if next(failing, False):
            raise fault("recording failed")
        return recorded(self, *args)

    monkeypatch.setattr(rendered_cache.RenderedCache, step, recording)
    fake.call_script = ['fn:weather{"city": "Bonn"}'] if ending == "call" else []
    with pytest.raises(fault, match="recording failed"):
        w.reply(None if ending == "going on after a result" else "zwei",
                tool_call_policy="yield").finish()
    assert _state(w) == before
    assert_cache_reads_as_rendered(w, fake)
    fake.call_script = []
    if ending == "going on after a result":
        w.reply(tool_call_policy="yield").finish()
    w.reply("drei").finish()
    assert after_system(w.conversation()["messages"])[-2] == {"role": "user", "content": "drei"}


def test_a_call_that_cannot_be_read_is_never_published_as_owed(monkeypatch):
    w, fake = tool_wrapper()
    w.begin(opened("persona"), tools=weather_registry().specs())
    owed = []
    publish = rendered_cache.RenderedCache._publish

    def publishing(self):
        publish(self)
        owed.append(self._published.pending_calls)

    monkeypatch.setattr(rendered_cache.RenderedCache, "_publish", publishing)
    fake.call_script, fake.replies = ["@@garbled((("], ["Sorry."]
    w.reply("Wetter?", tool_call_policy="yield").finish()
    assert owed and not any(owed)


# ----- what a request asks for ------------------------------------------------------
def test_a_call_ceiling_where_no_call_is_run_is_refused():
    w, fake = tool_wrapper()
    w.begin(opened("persona"))
    for tool_call_policy in TOOL_CALL_POLICIES:
        with pytest.raises(ValueError, match="max_calls"):
            w.reply("x", tool_call_policy=tool_call_policy, max_calls=2)
    assert after_system(w.conversation()["messages"]) == []


def test_a_failed_decode_of_the_answer_to_an_unreadable_call_closes_its_turn():
    fake = _Flaky(TOOLS, gen_tokens=3)
    w = ChatWrapper(Config(context_size=800, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"), tools=weather_registry().specs())
    fake.call_script.append("fn:@@@ kein Aufruf @@@")
    sampled = fake.sampled_tokens

    def then_failing_decode(*args, **kwargs):
        try:
            yield from sampled(*args, **kwargs)
        finally:
            fake.fail = "decode"        # the decode of the error result

    fake.sampled_tokens = then_failing_decode
    with pytest.raises(RuntimeError, match="decode failed"):
        w.reply("Wetter?", tool_call_policy="yield").finish()
    fake.sampled_tokens = sampled
    assert not w.turn_open and w.pending_calls == []
    assert after_system(w.conversation()["messages"])[-2:] == [{"role": "user", "content": "Wetter?"},
                                                 {"role": "assistant", "content": ""}]
    w.reply("weiter").finish()


def test_a_reply_kept_without_text_holds_the_replys_opening_once():
    for size in range(100, 300):
        fake = FakeContext(TOOLS, gen_tokens=6, gen_text="Hallo.")
        w = ChatWrapper(Config(context_size=size, evict_above_fraction=1.0, min_reply_tokens=60),
                        context=fake)
        try:
            w.begin(opened("s"), tools=weather_registry().specs())
            w.reply("eins").finish()
            fake.call_script.append("fn:@@@ kein Aufruf @@@")
            reply = w.reply("Wetter?", tool_call_policy="yield").finish()
        except ContextOverflowError:
            continue
        if reply.stop_reason == "unreadable_tool_call" and not w.turn_open:
            assert newest_turn(fake).count("<[turn>model") == 1
            return
    pytest.fail("no context size left the error result without room")


def test_a_refused_action_leaves_a_cache_that_cannot_cut_ready_to_generate():
    fake = FakeContext(REFUSES_TWO_USER_TURNS, can_shift=False, can_cut_tail=False, gen_tokens=3)
    w = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}] * 10 + [{"role": "user", "content": "q"}]))
    with pytest.raises(ContextOverflowError):
        w.inject("x" * 3000, role="assistant")  # refused before anything is dropped
    resets, decoded = fake.resets, fake.total_prefilled()
    w.reply().finish()
    assert fake.resets == resets and fake.total_prefilled() - decoded < 100


def test_a_decode_that_fails_before_the_pinned_state_is_saved_loses_none_of_its_tokens():
    from llama_chat.cache_sync import CacheSync
    fake = _Flaky(PLAIN, can_shift=False, can_cut_tail=False)
    memory = CacheSync(fake)
    memory.pin([3], "opening")
    fake.fail = "decode"            # the first half, up to the pin
    with pytest.raises(RuntimeError):
        memory.decode(list(range(1, 11)))
    assert memory.tokens == list(range(1, 11))
    memory.repair()
    assert fake.cache == list(range(1, 11))


@pytest.mark.parametrize("can_shift, can_cut_tail", [(True, False), (False, True), (False, False)],
                         ids=["shifts", "cuts", "neither"])
def test_every_change_to_the_tokens_a_cache_should_hold_changes_their_version(can_shift,
                                                                             can_cut_tail):
    import random

    from llama_chat.generation import GenerationAccumulator
    from llama_chat.cache_sync import CacheSync
    fake = _Flaky(PLAIN, gen_tokens=2, can_shift=can_shift, can_cut_tail=can_cut_tail)
    memory, draw = CacheSync(fake), random.Random(0)

    def some_tokens():
        return [draw.randrange(1, 5) for _ in range(draw.randrange(4))]

    def generate(closed_early):
        pieces = memory.generate(draw.randrange(3), GenerationAccumulator(),
                                 sampling=generation_sampling(Config(), None))
        if closed_early:
            next(pieces, None)
            pieces.close()
        else:
            list(pieces)

    actions = {
        "decode": lambda: memory.decode(some_tokens(), want_logits=draw.random() < 0.5),
        "remove": lambda: memory.remove(*sorted(draw.choices(range(len(memory.tokens) + 1), k=2))),
        "cut_to": lambda: memory.cut_to(draw.randrange(len(memory.tokens) + 1)),
        "replace": lambda: memory.replace(
            draw.choice([memory.tokens, memory.tokens[:-1] + some_tokens()])),
        "generate": lambda: memory.tokens and generate(closed_early=False),
        "generate closed early": lambda: memory.tokens and generate(closed_early=True),
        "pin": lambda: memory.pin(draw.sample(range(len(memory.tokens) + 2), k=2),
                                  draw.choice(["opening", "exchanges"])),
        "rest": memory.rest, "repair": memory.repair,
    }
    for _ in range(3000):
        name = draw.choice(sorted(actions))
        fake.fail = draw.choice([None] * 8 + ["decode", "cut", "generate"])
        tokens, version = list(memory.tokens), memory.version
        try:
            actions[name]()
        except RuntimeError:
            pass
        assert memory.tokens == tokens or memory.version != version, name


@pytest.mark.parametrize("cut", ["right after the closing", "a word later"])
@pytest.mark.parametrize("stem", [stem for stem in STEMS if fixture(stem)["marks_thought"]])
def test_a_generation_failing_after_an_answer_that_spells_its_thought_closing_leaves_no_empty_answer(
        stem, cut):
    fake = _Flaky(fixture(stem))
    w = ChatWrapper(Config(context_size=2048), context=fake)
    w.begin(opened(""))
    closing = internals.template(w).properties.reasoning.thought_close.strip()
    said = f"Close it with {closing} and more words follow here."
    fake.replies = [said, "Gern."]
    sampled = fake.sampled_tokens

    def failing_after_the_closing(*args, **kwargs):
        text = ""
        for token in sampled(*args, **kwargs):
            yield token
            text += fake.token_text(token)
            if closing in text and text.endswith(closing if cut == "right after the closing" else " an"):
                raise RuntimeError("generation failed")

    fake.sampled_tokens = failing_after_the_closing
    with pytest.raises(RuntimeError, match="generation failed"):
        w.reply("Q?").finish()
    fake.sampled_tokens = sampled
    messages = after_system(w.conversation()["messages"])
    # Kept with a part of what it wrote, or taken back whole with its question.
    assert messages == [] or (messages[0] == {"role": "user", "content": "Q?"}
                              and messages[1]["content"].strip()
                              and messages[1]["content"] in said)
    w.reply("Weiter?").finish()


@pytest.mark.parametrize("template", [TOOLS, TOOLS_REFUSING_TWO_USER_TURNS], ids=["plain", "refusing two"])
def test_a_reply_failing_before_it_generates_is_taken_back_with_its_question(monkeypatch, template):
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext(template, gen_tokens=3))
    w.begin(opened("persona", [{"role": "user", "content": "alt"}, {"role": "assistant", "content": "älter"}]))
    w.reply("eins").finish()
    before = _state(w)

    def failing(*args, **kwargs):
        raise RuntimeError("set-up failed")

    monkeypatch.setattr(RenderedCache, "_turn_end_test", failing)
    with pytest.raises(RuntimeError, match="set-up failed"):
        w.reply("zwei").finish()
    monkeypatch.undo()
    assert _state(w) == before
    w.reply("drei").finish()


def test_a_reply_that_fails_before_writing_anything_leaves_its_turns_policy_as_it_was():
    def turn_opened_by_its_history():
        fake = _Flaky(TOOLS, gen_tokens=3)
        w = ChatWrapper(Config(context_size=800, evict_above_fraction=1.0), context=fake)
        w.begin(opened("s", [{"role": "user", "content": "x"}, *tool_turn(ToolCall("weather", {"city": "Bonn"}),
                                               Result("weather", "18"))]), tools=weather_registry().specs())
        return w, fake

    w, fake = turn_opened_by_its_history()
    fake.fail = "generate"
    with pytest.raises(RuntimeError, match="generation failed"):
        w.reply(tool_call_policy="yield").finish()
    fake.call_script = ['fn:weather{"city":"Köln"}']
    after_failure = [type(event).__name__ for event in w.reply()]
    w, fake = turn_opened_by_its_history()
    fake.call_script = ['fn:weather{"city":"Köln"}']
    assert after_failure == [type(event).__name__ for event in w.reply()]


@pytest.mark.parametrize("template", [PLAIN, REFUSES_TWO_USER_TURNS], ids=["plain", "refusing two"])
def test_a_request_cancelled_before_it_starts_changes_nothing(template):
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5, min_reply_tokens=20),
                    context=FakeContext(template, gen_tokens=5))
    w.begin(opened("s"))
    for n in range(8):
        w.reply(f"frage {n} " + "x" * 20).finish()
    before = _state(w)
    cancel = threading.Event()
    cancel.set()
    reply = w.reply("eine lange neue frage " + "y" * 60, cancel=cancel).finish()
    assert (reply.stop_reason, reply.n_evicted_messages, reply.n_prefilled_tokens) == ("interrupted", 0, 0)
    assert _state(w) == before


def test_a_long_question_stops_decoding_within_a_part_once_cancelled():
    fake = FakeContext(PLAIN, gen_tokens=5)
    w = ChatWrapper(Config(context_size=8192), context=fake)
    w.begin(opened("s"))
    w.reply("eins").finish()
    before, cancel, decoded = _state(w), threading.Event(), []
    prefill = fake.prefill

    def cancelling_after_the_first_part(token_ids, start_pos, want_logits):
        decoded.append(len(token_ids))
        prefill(token_ids, start_pos, want_logits)
        cancel.set()                            # as another thread would, mid-decode

    fake.prefill = cancelling_after_the_first_part
    reply = w.reply("z" * 3000, cancel=cancel).finish()
    del fake.prefill
    assert reply.stop_reason == "interrupted" and len(decoded) == 1 and decoded[0] <= 512
    assert _state(w) == before


@pytest.mark.parametrize("cache", sorted(_CACHES))
def test_a_cancel_set_while_the_question_is_decoded_changes_nothing_and_counts_what_was_decoded(cache):
    fake = FakeContext(PLAIN, gen_tokens=5, **_CACHES[cache])
    w = ChatWrapper(Config(context_size=8192, evict_above_fraction=0.75), context=fake)
    w.begin(opened("s"))
    for n in range(40):
        w.reply(f"frage {n} " + "x" * 120).finish()
    before, cancel = _state(w), threading.Event()
    prefill = fake.prefill

    def cancelling(token_ids, start_pos, want_logits):
        prefill(token_ids, start_pos, want_logits)
        cancel.set()

    fake.prefill = cancelling
    reply = w.reply("z" * 3000, cancel=cancel).finish()
    del fake.prefill
    assert reply.stop_reason == "interrupted" and _state(w) == before
    assert reply.n_prefilled_tokens > 0


def test_a_cancel_set_during_a_short_questions_decode_changes_nothing():
    fake = FakeContext(PLAIN, gen_tokens=5)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5, min_reply_tokens=20), context=fake)
    w.begin(opened("s"))
    for n in range(8):
        w.reply(f"frage {n} " + "x" * 20).finish()
    before, cancel = _state(w), threading.Event()
    prefill = fake.prefill

    def cancelling(token_ids, start_pos, want_logits):
        prefill(token_ids, start_pos, want_logits)
        cancel.set()

    fake.prefill = cancelling
    reply = w.reply("eine neue frage " + "y" * 60, cancel=cancel).finish()
    del fake.prefill
    assert (reply.stop_reason, reply.n_evicted_messages) == ("interrupted", 0) and _state(w) == before


# ----- one undo path: every exit before the first word puts the attempt back ---------
def _evicting_conversation(fake):
    """A wrapper whose history a long question pushes out at once, and that history."""
    w = ChatWrapper(Config(context_size=300, evict_above_fraction=1.0, min_reply_tokens=32),
                    context=fake)
    w.begin(opened("s", [{"role": "user", "content": "a" * 60}, {"role": "assistant", "content": "b" * 60}, {"role": "user", "content": "c" * 60},
                  {"role": "assistant", "content": "d" * 30}]))
    return w, after_system(w.conversation()["messages"])


def test_a_generation_failing_before_its_first_word_puts_back_the_history_its_question_evicted():
    fake = _Flaky(PLAIN, gen_tokens=5)
    w, history = _evicting_conversation(fake)
    fake.fail = "generate"
    with pytest.raises(RuntimeError):
        w.reply(" " + "q" * 150).finish()       # not spliced, so it evicts as it is asked
    assert after_system(w.conversation()["messages"]) == history
    assert_cache_reads_as_rendered(w, fake)


class _CancelAtTheFirstDraw(FakeContext):
    """A fake model that sets ``cancel`` as its first token is drawn, once the question is decoded."""

    cancel = None

    def sampled_tokens(self, *args, **kwargs):
        if self.cancel is not None:
            self.cancel.set()
        return super().sampled_tokens(*args, **kwargs)


@pytest.mark.parametrize("template", [PLAIN, REFUSES_TWO_USER_TURNS],
                         ids=["writes-two-user-turns", "refuses-two-user-turns"])
def test_a_cancel_at_the_first_draw_changes_nothing(template):
    fake = _CancelAtTheFirstDraw(template)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]))
    before = w.conversation()
    fake.cancel = threading.Event()
    reply = w.reply("q", cancel=fake.cancel).finish()
    assert reply.stop_reason == "interrupted" and reply.text == "" \
        and reply.n_evicted_messages == 0
    assert w.conversation() == before


def test_a_cancel_at_the_first_draw_after_an_injected_question_keeps_that_question():
    fake = _CancelAtTheFirstDraw(REFUSES_TWO_USER_TURNS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]))
    w.inject("q")
    before = w.conversation()
    fake.cancel = threading.Event()
    assert w.reply(cancel=fake.cancel).finish().stop_reason == "interrupted"
    assert w.conversation() == before


def test_a_cancel_at_the_first_draw_of_a_turn_going_on_leaves_it_open_with_its_results():
    fake = _CancelAtTheFirstDraw(TOOLS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "wie spät?"}, *tool_turn(ToolCall("clock", {}),
                                                   Result("clock", "12:00"))]))
    assert w.turn_open
    before = w.conversation()
    fake.cancel = threading.Event()
    assert w.reply(cancel=fake.cancel).finish().stop_reason == "interrupted"
    assert w.conversation() == before and w.turn_open


@pytest.mark.parametrize("stem", [stem for stem in STEMS
                                  if not expected(fixture(stem), "writes_empty_reply")])
def test_a_reply_taken_back_with_its_question_puts_back_what_the_question_evicted(stem):
    fake = FakeContext(fixture(stem), gen_tokens=0)
    w = ChatWrapper(Config(context_size=4096, evict_above_fraction=1.0, min_reply_tokens=32,
                           **UNGUARDED), context=fake)
    w.begin(opened("", [{"role": "user", "content": "a" * 500}, {"role": "assistant", "content": "b" * 500}, {"role": "user", "content": "c" * 500},
                 {"role": "assistant", "content": "d" * 250}]))
    history = after_system(w.conversation()["messages"])
    # Not spliced, and too long to fit beside the whole history: it evicts as it is asked.
    question = " " + "q" * (4096 - w.total_tokens + 100)
    reply = w.reply(question).finish()          # ends at once: no empty reply is written
    assert reply.stop_reason == "end_of_turn"
    assert after_system(w.conversation()["messages"]) == history and reply.n_evicted_messages == 0
    assert_cache_reads_as_rendered(w, fake)


def test_a_cache_whose_repair_fails_once_is_decoded_whole_again_by_the_next_action():
    fake = _Flaky(PLAIN, gen_tokens=5)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]))
    fake.fail = "generate"
    with pytest.raises(RuntimeError):
        w.reply("eins").finish()
    fake.fail = "decode"                        # the repair at the next action fails once
    assert w.reply("zwei").finish().text
    assert_cache_reads_as_rendered(w, fake)


def test_a_refusal_carries_what_the_template_itself_says():
    says = dict(PLAIN, chat_template="{% if messages[-1].content == 'verboten' %}"
                                     "{{ raise_exception('this word is not written') }}{% endif %}"
                + PLAIN["chat_template"])
    w = ChatWrapper(Config(context_size=500), context=FakeContext(says))
    w.begin(opened("s"))
    with pytest.raises(TemplateRefusedError, match="the template says: this word is not written"):
        w.inject("verboten")
    with pytest.raises(TemplateRefusedError, match="the template says: this word is not written"):
        w.begin(opened("s", [{"role": "user", "content": "verboten"}]))


@pytest.mark.parametrize("policy", ["refuse", "yield", "withhold"])
def test_a_failure_while_a_stream_closed_early_records_its_reply_reaches_the_caller(policy):
    fake = _Flaky(PLAIN, gen_tokens=8, gen_text="Eine lange Antwort.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    stream = w.reply("hallo", tool_call_policy=policy)
    next(stream)
    fake.fail = "decode"            # the decode ahead, once the reply is recorded
    with pytest.raises(RuntimeError, match="decode failed"):
        stream.close()
    assert "".join(w.reply("noch einmal")) == "Eine lan"
    assert_cache_reads_as_rendered(w, fake)


_SAYS_VERBOTEN = ("{% if messages[-1].content == 'verboten' %}"
                  "{{ raise_exception('this word is not written') }}{% endif %}")


def test_a_history_the_template_refuses_names_messages_as_its_field():
    says = dict(PLAIN, chat_template=_SAYS_VERBOTEN + PLAIN["chat_template"])
    w = ChatWrapper(Config(context_size=500), context=FakeContext(says))
    with pytest.raises(TemplateRefusedError) as refused:
        w.begin(opened("s", [{"role": "user", "content": "verboten"}]))
    assert refused.value.field == "messages"


def test_a_history_ending_on_a_result_is_refused_with_what_the_template_says_before_its_reply():
    # The template refuses only the render a reply follows, which a history ending on a result needs.
    says = dict(TOOLS, chat_template="{% if add_generation_prompt and messages[-1].role == 'tool' "
                                     "and 'verboten' in messages[-1].content %}"
                                     "{{ raise_exception('no reply after this result') }}{% endif %}"
                + TOOLS["chat_template"])
    w = ChatWrapper(Config(context_size=1024), context=FakeContext(says))
    history = opened("s", [{"role": "user", "content": "Wetter?"},
                           *tool_turn(ToolCall("weather", {"city": "Bonn"}),
                                      Result("weather", "verboten"))])
    # The result is the last message, and neither the first after the system turn nor the one before.
    assert history[-1]["role"] == "tool" and history[1]["role"] != "tool" \
        and history[-2]["role"] != "tool"
    with pytest.raises(TemplateRefusedError,
                       match="the template says: no reply after this result") as refused:
        w.begin(history, tools=weather_registry().specs())
    assert refused.value.field == "messages"


class _FailingAfterADecode(FakeContext):
    """A fake cache that sets :attr:`cancel` as a decode of :attr:`cancel_after` ends, and fails every operation after it until :attr:`fails` is cleared."""

    cancel = None
    cancel_after = None
    fails = False

    def _refuse(self):
        if self.fails:
            raise EngineError("the cache fails")

    def prefill(self, *args, **kwargs):
        self._refuse()
        super().prefill(*args, **kwargs)
        if self.cancel is not None and self.cancel_after in self.text_of(args[0]):
            self.cancel.set()
            self.cancel, self.fails = None, True

    def remove_from(self, pos):
        self._refuse()
        super().remove_from(pos)

    def apply_eviction(self, start, end):
        self._refuse()
        super().apply_eviction(start, end)

    def reset(self):
        self._refuse()
        super().reset()

    def restore_state(self, state, pos):
        self._refuse()
        super().restore_state(state, pos)


@pytest.mark.parametrize("cache", _CACHES)
def test_a_question_cancelled_where_the_cache_cannot_be_brought_back_is_taken_back_and_the_next_is_answered(
        settles_in_the_background, cache):
    fake = _FailingAfterADecode(PLAIN, **_CACHES[cache])
    wrapper = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    wrapper.begin(opened("persona"))
    wrapper.reply("eins").finish()
    before = wrapper.conversation()
    cancel = threading.Event()
    fake.cancel, fake.cancel_after = cancel, "zwei"
    reply = wrapper.reply("zwei", cancel=cancel).finish()
    assert (reply.text, reply.stop_reason) == ("", "interrupted")
    assert wrapper.conversation() == before
    assert wrapper.wait_idle()          # the cache is brought up after it, which fails and is logged
    fake.fails = False
    assert wrapper.reply("drei").finish().text


# ----- for tests/test_failures.py ---------------------------------------------------
@pytest.mark.parametrize("refusing", [
    "{{ raise_exception('') }}", "{{ (1 / 0) }}"], ids=["empty-reason", "fails-otherwise"])
def test_a_refusal_quotes_no_reason_where_the_template_gives_none(refusing):
    says = dict(PLAIN, chat_template="{% if messages[-1].content == 'verboten' %}" + refusing
                + "{% endif %}" + PLAIN["chat_template"])
    w = ChatWrapper(Config(context_size=500), context=FakeContext(says))
    w.begin(opened("s"))
    with pytest.raises(TemplateRefusedError) as refused:
        w.inject("verboten")
    assert str(refused.value) == "this model's template does not write a user message here"


@pytest.mark.parametrize("can_shift, can_cut_tail", [(True, False), (False, True), (False, False)],
                         ids=["shifts", "cuts", "neither"])
def test_a_version_of_the_tokens_a_cache_should_hold_never_comes_back_for_other_tokens(can_shift,
                                                                                     can_cut_tail):
    fake = _Flaky(PLAIN, gen_tokens=2, can_shift=can_shift, can_cut_tail=can_cut_tail)
    memory, draw = CacheSync(fake), random.Random(1)

    def some_tokens():
        return [draw.randrange(1, 5) for _ in range(draw.randrange(4))]

    def generate():
        list(memory.generate(draw.randrange(3), GenerationAccumulator(),
                             sampling=generation_sampling(Config(), None)))

    actions = {
        "decode": lambda: memory.decode(some_tokens(), want_logits=draw.random() < 0.5),
        "remove": lambda: memory.remove(*sorted(draw.choices(range(len(memory.tokens) + 1), k=2))),
        "cut_to": lambda: memory.cut_to(draw.randrange(len(memory.tokens) + 1)),
        "replace": lambda: memory.replace(
            draw.choice([memory.tokens, memory.tokens[:-1] + some_tokens()])),
        "generate": lambda: memory.tokens and generate(),
        "rest": memory.rest, "repair": memory.repair,
    }
    seen = {memory.version: list(memory.tokens)}
    for _ in range(2000):
        name = draw.choice(sorted(actions))
        fake.fail = draw.choice([None] * 8 + ["decode", "cut", "generate"])
        try:
            actions[name]()
        except RuntimeError:
            pass
        assert seen.setdefault(memory.version, list(memory.tokens)) == memory.tokens, name


@pytest.mark.parametrize("can_shift, can_cut_tail", [(True, False), (False, True), (False, False)],
                         ids=["shifts", "cuts", "neither"])
def test_every_draw_follows_a_decode_that_kept_the_logits_of_the_cache_as_it_ends(can_shift,
                                                                                 can_cut_tail):
    class _DrawsAfterFreshLogits(_Flaky):
        """Records whether the newest decode kept logits for the cache's last token, with no edit since."""

        fresh = False

        def prefill(self, token_ids, start_pos, want_logits):
            self.fresh = False
            super().prefill(token_ids, start_pos, want_logits)
            self.fresh = bool(want_logits)

        def decode_generated(self, token_ids, start_pos):
            self.fresh = False
            super().decode_generated(token_ids, start_pos)
            self.fresh = True

        def remove_from(self, pos):
            self.fresh = False
            super().remove_from(pos)

        def apply_eviction(self, start, end):
            self.fresh = False
            super().apply_eviction(start, end)

        def restore_state(self, state, pos):
            self.fresh = False
            super().restore_state(state, pos)

        def reset(self):
            self.fresh = False
            super().reset()

        def sampled_tokens(self, sampling, withheld_tokens):
            assert self.fresh, "a draw from logits no decode kept for the cache as it ends"
            yield from super().sampled_tokens(sampling, withheld_tokens)

    fake = _DrawsAfterFreshLogits(PLAIN, gen_tokens=2, can_shift=can_shift,
                                  can_cut_tail=can_cut_tail)
    memory, draw = CacheSync(fake), random.Random(2)

    def some_tokens():
        return [draw.randrange(1, 5) for _ in range(draw.randrange(1, 4))]

    def generate():
        list(memory.generate(draw.randrange(1, 3), GenerationAccumulator(),
                             sampling=generation_sampling(Config(), None)))

    actions = {
        "decode": lambda: memory.decode(some_tokens(), want_logits=draw.random() < 0.5),
        "remove": lambda: memory.remove(*sorted(draw.choices(range(len(memory.tokens) + 1), k=2))),
        "cut_to": lambda: memory.cut_to(draw.randrange(1, len(memory.tokens) + 1)),
        "replace": lambda: memory.replace(draw.choice(
            [memory.tokens[:draw.randrange(1, len(memory.tokens) + 1)],
             memory.tokens[:-1] + some_tokens()])),
        "rest": memory.rest, "repair": memory.repair,
    }
    memory.decode(some_tokens())
    for _ in range(1500):
        actions[draw.choice(sorted(actions))]()
        if memory.tokens and draw.random() < 0.5:
            generate()
        if not memory.tokens:
            memory.decode(some_tokens())


def test_a_cache_whose_saved_states_cannot_be_ordered_goes_back_to_the_newest_that_serves():
    class _StatesWithoutOrder(FakeContext):
        """Saves each state as an object that compares by identity alone, as an engine's buffer does."""

        def save_state(self):
            saved = super().save_state()
            return lambda: saved

        def restore_state(self, state, pos):
            super().restore_state(state(), pos)

    fake = _StatesWithoutOrder(PLAIN, can_shift=False, can_cut_tail=False)
    memory, draw = CacheSync(fake), random.Random(3)

    def some_tokens():
        return [draw.randrange(1, 5) for _ in range(draw.randrange(1, 4))]

    actions = {
        "decode": lambda: memory.decode(some_tokens()),
        "cut_to": lambda: memory.cut_to(draw.randrange(len(memory.tokens) + 1)),
        "hold": lambda: memory.hold(memory.tokens[:draw.randrange(len(memory.tokens) + 1)]
                                    + some_tokens(), rest_at_cut=draw.random() < 0.7),
        "pin": lambda: memory.pin(draw.sample(range(len(memory.tokens) + 2), k=2),
                                  draw.choice(["opening", "exchanges"])),
        "rest": memory.rest, "repair": memory.repair,
    }
    for _ in range(1500):
        name = draw.choice(sorted(actions))
        actions[name]()
        if memory.is_current:
            assert fake.cache == memory.tokens, name
    assert fake.edit_count("restore") > 0


@pytest.mark.parametrize("handed_over", [True, False], ids=["handed-over", "registered"])
@pytest.mark.parametrize("result", _UNWRITABLE, ids=["date", "nan", "infinity"])
def test_a_result_that_cannot_be_written_is_told_as_a_failed_call_naming_why(result, handed_over):
    registry = ToolRegistry()
    registry.tool(lambda: result, description="Das Datum.", name="clock",
                  parameters={"type": "object", "properties": {}})
    w, fake = tool_wrapper()
    w.begin(opened("persona"), tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    if handed_over:
        call, = [e for e in w.reply("welcher Tag?", tool_call_policy="yield")
                 if isinstance(e, ToolCall)]
        w.inject_tool_result(call.id, result)
    else:
        w.reply("welcher Tag?", tool_call_policy=registry).finish()
    told, = [json.loads(m["content"]) for m in w.conversation()["messages"] if m["role"] == "tool"]
    assert told["error"] == "unwritable_result"
    assert told["message"].startswith("the tool call failed: a tool result must be text or what "
                                      "JSON can write: ")


# Templates that refuse a text only where the render is given, in turn, the tools, the reasoning
# switch on, or the variable that puts the declarations into the system turn.
_REFUSES_ONLY_WITH_TOOLS = ("{% if messages[-1].content == 'verboten' and tools %}"
                            "{{ raise_exception('this word is not written') }}{% endif %}")
_REFUSES_ONLY_WITH_REASONING = ("{% if messages[-1].content == 'verboten' and enable_thinking %}"
                                "{{ raise_exception('this word is not written') }}{% endif %}")
_DECLARED_BY_A_VARIABLE = "{%- set declares = none -%}"
assert _DECLARED_BY_A_VARIABLE in NO_DECLARATIONS["chat_template"]
_REFUSES_ONLY_WITH_THE_VARIABLE = (
    "{% if messages[-1].content == 'verboten' and tools_in_system %}"
    "{{ raise_exception('this word is not written') }}{% endif %}")


_REFUSING = {
    "tools": {**REASONING, "chat_template": _REFUSES_ONLY_WITH_TOOLS
              + REASONING["chat_template"]},
    "reasoning": {**REASONING, "chat_template": _REFUSES_ONLY_WITH_REASONING
                  + REASONING["chat_template"]},
    "variable": {**NO_DECLARATIONS, "render_variables": {"off": {"tools_in_system": True}},
                 "chat_template": _REFUSES_ONLY_WITH_THE_VARIABLE
                 + NO_DECLARATIONS["chat_template"].replace(
                     _DECLARED_BY_A_VARIABLE,
                     "{%- set declares = 'in the system turn' if tools_in_system else none -%}")},
}


@pytest.mark.parametrize("given", list(_REFUSING))
def test_a_refusal_carries_what_the_template_says_where_it_refuses_only_as_the_conversation_renders(given):
    """The template's cause is read with the tools, the reasoning switch and the declaration variable the conversation renders with."""
    fake = FakeContext(_REFUSING[given])
    w = ChatWrapper(Config(context_size=1024), context=fake)
    w.begin(opened("s"), tools=weather_registry().specs(), reasoning=bool(given == "reasoning"))
    with pytest.raises(TemplateRefusedError, match="the template says: this word is not written"):
        w.inject("verboten")


@pytest.mark.parametrize("role_after", ["user", "assistant"])
def test_a_refused_history_carries_the_template_s_cause_for_the_refused_message_not_for_the_one_after_it(
        role_after):
    """The template's own cause is read from the history up to the message it refuses, not one message further."""
    says = with_check_first(WRITES_TWO_QUESTIONS_REFUSES_A_TEXT,
                       "{% if messages[-1].content == 'verboten' %}"
                       "{{ raise_exception('this word is not written last') }}{% endif %}")
    chat = ChatWrapper(Config(context_size=500), context=FakeContext(says))
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
               {"role": "user", "content": "refused"}, {"role": role_after, "content": "verboten"},
               {"role": "user", "content": "c"}]
    with pytest.raises(TemplateRefusedError) as refused:
        chat.begin(opened("s", history))
    message = str(refused.value)
    assert refused.value.field == "messages"
    assert "the user message 'refused' where it stands" in message
    assert "this text is refused" in message and "not written last" not in message


@pytest.mark.parametrize("cache", sorted(_CACHES))
@pytest.mark.parametrize("goes_back", [False, True], ids=["switching away", "going back"])
def test_a_begin_that_keeps_or_takes_back_a_conversation_failing_at_any_point_changes_nothing(
        cache, goes_back):
    """Rule 6 on the kept conversations: a begin that saves the cache it leaves, or restores one kept, failing at any point leaves the conversation as it was and the cache usable."""
    def history(name):
        return opened(name, [m for n in range(12) for m in (
            {"role": "user", "content": f"{name} {n} " + "x" * 30},
            {"role": "assistant", "content": "Antwort " + "y" * 30})])

    def prepared():
        fake = _Flaky(PLAIN, gen_tokens=5, gen_text="Gern.", **_CACHES[cache])
        w = ChatWrapper(Config(context_size=4096, kept_conversation_bytes=1 << 24), context=fake)
        w.begin(history("Eins"))
        first = w.conversation()
        if goes_back:
            w.begin(history("Zwei"))
        fake.operations = 0
        return w, fake, first

    w, fake, first = prepared()
    target = first if goes_back else w.conversation() | {"messages": history("Zwei")}
    saved = fake.saved_states
    w.begin(**target)
    points = fake.operations
    # The begin swept saves the cache it leaves, and going back restores the one kept; a cache
    # that goes back to saved states keeps none.
    keeps = _CACHES[cache].get("can_shift") or _CACHES[cache].get("can_cut_tail")
    assert (fake.saved_states > saved) == bool(keeps) or not keeps
    assert any(edit[0] == "restore whole" for edit in fake.edits) == (goes_back and bool(keeps))
    for at in range(points):
        w, fake, first = prepared()
        before = w.conversation()
        fake.fail_at = at
        try:
            w.begin(**target)
        except RuntimeError:
            assert w.conversation() == before, at
        else:
            assert w.conversation()["messages"] == target["messages"], at
        w.reply("Und jetzt?").finish()
        assert_cache_reads_as_rendered(w, fake)
        w.close()
