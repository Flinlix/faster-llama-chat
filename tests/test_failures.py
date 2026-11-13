"""Every action that is refused or fails leaves the conversation and the cache as they were before it, a reply keeping only what it produced, and usable."""

import dataclasses
import datetime as real_datetime
import threading

import pytest

from llama_chat import TOOL_CALL_POLICIES, ChatWrapper, Config, ContextOverflowError, rendered_cache
from llama_chat.rendering import ConversationRendering
from llama_chat.tools import ToolCall, ToolRegistry, ToolSpec, ToolTurn
from tests.chat_templates import STEMS, expected, fixture
from tests.fake_context import FakeContext
from tests.internals import kept_turns
from tests.invented_templates import PLAIN, REFUSES_TWO_USER_TURNS, TOOLS, TOOLS_REFUSING_TWO_USER_TURNS
from tests.wrappers import (assert_cache_reads_as_rendered, assert_consistent, held_text,
                            newest_turn, rows, tool_wrapper, weather_registry)


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
            raise RuntimeError("decode failed")
        super().prefill(token_ids, start_pos, want_logits)

    def decode_generated(self, token_ids, start_pos):
        if self._fails():
            self.cache.extend(token_ids)
            raise RuntimeError("decode failed")
        super().decode_generated(token_ids, start_pos)

    def remove_from(self, pos):
        if self._fails("cut"):
            raise RuntimeError("the cache refused to remove its tokens")
        super().remove_from(pos)

    def apply_eviction(self, start, end):
        if self._fails():
            raise RuntimeError("the cache refused to shift")
        super().apply_eviction(start, end)

    def reset(self):
        if self._fails():
            raise RuntimeError("the cache refused to empty")
        super().reset()

    def save_state(self):
        if self._fails():
            raise RuntimeError("the cache refused to save its state")
        return super().save_state()

    def restore_state(self, state, pos):
        if self._fails():
            raise RuntimeError("the cache refused the saved state")
        super().restore_state(state, pos)

    def sampled_tokens(self, *args, **kwargs):
        if self.fail == "generate":
            self.fail = None
            self.cache.append(ord("x"))
            raise RuntimeError("generation failed")
        for token in super().sampled_tokens(*args, **kwargs):
            if self._fails():
                raise RuntimeError("generation failed")
            yield token


def _flaky(**kw):
    fake = _Flaky(PLAIN, **kw)
    return ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake), fake


# ----- every action, failing at every point ----------------------------------------
_CALL, _CALLED = "fn:uhr{}", ToolCall("uhr", {})
_CALENDAR = ToolSpec("kalender", "Die Termine.", {"type": "object", "properties": {}})
_CACHES = {"shifts": dict(can_shift=True), "cuts": dict(can_shift=False, can_cut_tail=True),
           "neither": dict(can_shift=False, can_cut_tail=False)}
_CLOCK = ToolRegistry()
_CLOCK.tool(lambda: "12:00", description="Die Zeit.", name="uhr")


def _begin(*history):
    """A step that begins the conversation on ``history``, a clock its one tool."""
    def begin(w, fake):
        w.begin("persona", list(history), tools=_CLOCK.specs())
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
    w.inject_tool_result(w.pending_calls[0], "sonnig")


def _replying(text=None, script=None, **options):
    def act(w, fake, shown):
        fake.call_script = [script] if script else []
        for event in w.reply(text, **options):
            if isinstance(event, str):
                shown.append(event)
    return act


# What each action is taken from, what it does, and for a reply the calls its model makes.
_ACTIONS = {
    "begin": (_exchanged, lambda w, f, s: w.begin("neu", [("user", "a"), ("assistant", "b")]), None),
    "begin while a call is owed": (_owing, lambda w, f, s: w.begin("neu"), None),
    "inject": (_exchanged, lambda w, f, s: w.inject("Notiz dazu."), None),
    "inject_tool_result": (_owing, lambda w, f, s: w.inject_tool_result(w.pending_calls[0], "sonnig"),
                           None),
    "truncate_last_reply": (_exchanged, lambda w, f, s: w.truncate_last_reply("Ge"), None),
    "declare_tools": (_exchanged, lambda w, f, s: w.declare_tools([_CALENDAR]), None),
    "reply": (_exchanged, _replying("zwei"), []),
    "reply to the first question": (_begin(), _replying("zwei"), []),
    "reply to an injected question": (_injected, _replying(), []),
    "reply to the question a history ends on": (
        _begin(("user", "alt"), ("assistant", "älter"), ("user", "Und?")), _replying(), []),
    "reply refusing a call": (_exchanged, _replying("Wetter?", _CALL), [_CALLED]),
    "reply running a call": (_exchanged, _replying("Wetter?", _CALL, tool_call_policy=_CLOCK), [_CALLED]),
    "reply yielding a call": (_exchanged, _replying("Wetter?", _CALL, tool_call_policy="yield"), [_CALLED]),
    "reply going on after a result": (_answered, _replying(tool_call_policy="yield"), []),
}


def _state(w):
    return w.conversation(), w.pending_calls, w.turn_open, w.total_tokens


def _steps(history):
    """A history as one list: each message, and each step of a tool turn, its text as the assistant's."""
    return [step for item in history for step in (
        [{"role": "assistant", "content": s} if isinstance(s, str) else s for s in item.steps]
        if isinstance(item, ToolTurn) else [item])]


def _assert_kept_only_what_it_produced(before, after, shown, calls):
    """A failed reply loses nothing but the oldest exchanges a result evicted for room, and adds nothing, or what it produced: the text it showed, then a call or a result."""
    held, kept = _steps(before[0]["messages"]), _steps(after[0]["messages"])
    starts = [i for i, step in enumerate(held) if isinstance(step, dict) and step["role"] == "user"]
    evicted = next((i for i in [0] + starts + [len(held)] if kept[:len(held) - i] == held[i:]),
                   None)
    assert after[0]["system_prompt"] == before[0]["system_prompt"] and evicted is not None
    held = held[evicted:]
    added = kept[len(held):]
    texts = [step["content"] for step in added
             if isinstance(step, dict) and step["role"] == "assistant"]
    assert "".join(texts) == "".join(shown)
    made = [dataclasses.replace(step, id="") for step in added if isinstance(step, ToolCall)]
    assert made == calls[:len(made)]
    last = added[-1] if added else None
    assert not isinstance(last, dict) or (last["role"] == "assistant" and last["content"]), \
        "the failed reply left a question or an empty reply behind"


def _go_on(w, fake):
    """Use the conversation as it was left: answer what is owed, close an open turn, then reply."""
    fake.call_script = []
    for call in w.pending_calls:
        w.inject_tool_result(call, "sonnig")
    if w.turn_open:
        w.reply(tool_call_policy="yield").finish()
    history = w.conversation()["messages"]
    waiting = bool(history) and isinstance(history[-1], dict) and history[-1]["role"] == "user"
    w.reply(None if waiting else "weiter").finish()


@pytest.mark.parametrize("action", sorted(_ACTIONS))
@pytest.mark.parametrize("cache", sorted(_CACHES))
def test_an_action_failing_at_any_point_leaves_everything_as_it_was_and_usable(cache, action):
    """A reply keeps what it produced before the failure and nothing else; any other action changes nothing."""
    prepare, act, calls = _ACTIONS[action]
    for refuses_two_user_turns, template in ((False, TOOLS), (True, TOOLS_REFUSING_TWO_USER_TURNS)):
        def prepared():
            fake = _Flaky(template, gen_tokens=5, gen_text="Gern.", **_CACHES[cache])
            w = ChatWrapper(Config(context_size=160, evict_above_fraction=0.6,
                                   min_reply_tokens=20), context=fake)
            _begin(("user", "alt"), ("assistant", "älter"))(w, fake)
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
    w.begin("s")
    w.reply("eins").finish()
    before = w.conversation()["messages"]
    with pytest.raises(ContextOverflowError):
        w.begin("x" * 600)
    assert w.conversation()["system_prompt"] == "s" and w.conversation()["messages"] == before
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
    w.begin(system)
    w.reply("One?").finish()
    clock[0] = real_datetime.datetime(2026, 11, 8, 0, 1)
    with pytest.raises(ContextOverflowError):
        w.begin("x" * 9000)
    assert w.conversation()["messages"] == [{"role": "user", "content": "One?"},
                                            {"role": "assistant", "content": "Eins."}]
    assert w.reply("Two?").finish().text == "Zwei."


def test_a_failed_restart_keeps_the_call_its_conversation_owes_and_nothing_else():
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    with pytest.raises(ContextOverflowError):
        w.begin("x" * 600)
    assert w.turn_open and w.pending_calls == [call]
    w.inject_tool_result(call, "12:00")
    w.reply().finish()
    assert not w.turn_open


# ----- results ---------------------------------------------------------------------
def test_a_result_that_leaves_no_room_for_the_reply_closes_the_turn_and_the_conversation_goes_on():
    fake = FakeContext(TOOLS)
    w = ChatWrapper(Config(context_size=300, evict_above_fraction=1.0, min_reply_tokens=40),
                    context=fake)
    w.begin("s")
    fake.call_script = ["fn:clock{}"]
    for n in range(100, 300):
        w.begin("s")
        try:
            calls = [e for e in w.reply("q" * n, tool_call_policy="yield") if isinstance(e, ToolCall)]
        except ContextOverflowError:
            break
        w.inject_tool_result(calls[0], "12:00")
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
        w.begin("s")
        fake.call_script = ["fn:clock{}"]
        call, = [e for e in w.reply("q" * 210, tool_call_policy="yield") if isinstance(e, ToolCall)]
        fake.operations = 0
        return w, fake, call

    w, fake, call = owing()
    w.inject_tool_result(call, "12:00")
    assert not w.turn_open and fake.operations > 0     # the turn closes, the cache settled after it
    for at in range(fake.operations):
        w, fake, call = owing()
        before = _state(w)
        fake.fail_at = at
        with pytest.raises(RuntimeError):
            w.inject_tool_result(call, "12:00")
        assert _state(w) == before
        _go_on(w, fake)
        assert_cache_reads_as_rendered(w, fake)
        w.close()


_UNWRITABLE = [{"when": real_datetime.date(2026, 11, 8)}, {"t": float("nan")},
               [float("inf")]]


@pytest.mark.parametrize("result", _UNWRITABLE, ids=["date", "nan", "infinity"])
def test_a_result_handed_over_that_cannot_be_written_reaches_the_model_as_a_failure(
        result, caplog):
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    with caplog.at_level("WARNING", logger="llama_chat"):
        w.inject_tool_result(call, result)
    assert "unwritable_result" in held_text(fake) and "NaN" not in held_text(fake)
    assert w.pending_calls == [] and "unwritable_result" in caplog.text
    w.reply().finish()


@pytest.mark.parametrize("returned", _UNWRITABLE, ids=["date", "nan", "infinity"])
def test_a_tool_that_returns_what_cannot_be_written_is_answered_with_an_error(returned):
    registry = ToolRegistry()
    registry.tool(lambda: returned, description="Das Datum.", name="clock",
                  parameters={"type": "object", "properties": {}})
    w, fake = tool_wrapper()
    w.begin("persona", tools=registry.specs())
    fake.call_script = ["fn:clock{}"]
    reply = w.reply("welcher Tag?", tool_call_policy=registry).finish()
    assert reply.stop_reason == "end_of_turn" and not w.turn_open
    assert "unwritable_result" in held_text(fake) and "NaN" not in held_text(fake)


def test_a_resumed_turn_whose_answer_is_cut_at_once_closes_and_keeps_its_call():
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call, "12:00")
    cancel = threading.Event()
    cancel.set()
    assert w.reply(cancel=cancel).finish().text == ""
    assert not w.turn_open and kept_turns(w)
    assert "fn:clock{}" in held_text(fake)
    w.reply("und jetzt?").finish()
    assert "fn:clock{}" in held_text(fake)


def test_an_action_whose_render_cannot_place_a_kept_turn_is_refused_and_changes_nothing(
        monkeypatch):
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call, "12:00")
    w.reply().finish()
    before, held = w.conversation(), list(fake.cache)
    with monkeypatch.context() as patched:
        # A template that writes the turn where it cannot be told apart, as none of the test
        # templates does.
        patched.setattr(ConversationRendering, "place", lambda *_args: None)
        with pytest.raises(ValueError, match="turn that called tools"):
            w.declare_tools([ToolSpec("uhr", "Die Uhrzeit.", {})])
    assert w.conversation() == before and fake.cache == held
    w.declare_tools([ToolSpec("uhr", "Die Uhrzeit.", {})])


# ----- questions --------------------------------------------------------------------
def test_a_question_without_room_for_its_reply_is_not_recorded():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=80, evict_above_fraction=1.0, min_reply_tokens=60),
                    context=fake)
    w.begin("s")
    with pytest.raises(ContextOverflowError):
        w.reply("hello").finish()
    assert w.conversation()["messages"] == []


# ----- a cache that cannot shift ----------------------------------------------------
def test_one_turn_that_drops_twice_on_a_cache_that_cannot_shift_brings_it_back_once(
        monkeypatch):
    monkeypatch.setattr(rendered_cache, "EVICT_TO_FRACTION_WITHOUT_SHIFT", 0.2)
    fake = FakeContext(PLAIN, gen_tokens=2, can_shift=False)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5), context=fake)
    w.begin("s")
    for i in range(8):
        before = fake.resets + fake.edit_count("restore")
        w.reply(f"message number {i} with some length").finish()
        assert fake.resets + fake.edit_count("restore") - before <= 1
        assert_consistent(w, fake)


# ----- reading from another thread ------------------------------------------------
def test_the_history_read_from_another_thread_is_never_half_built(monkeypatch):
    w, fake = tool_wrapper()
    w.begin("persona", [("user", "a"), ("assistant", "b")])
    seen = []
    split = ConversationRendering.split

    def reading(self, messages, generation_prompt):
        seen.append(len(w.conversation()["messages"]))           # as another thread reads it, mid-action
        return split(self, messages, generation_prompt)

    monkeypatch.setattr(ConversationRendering, "split", reading)
    w.reply("c").finish()
    # Before the question, with the question until its reply is recorded, then with both.
    assert set(seen) <= {2, 3, 4}


def test_the_token_count_and_the_open_turn_read_from_another_thread_are_a_finished_changes(
        monkeypatch):
    """Read in the middle of a begin that fails, they are what the conversation it leaves standing holds."""
    w, fake = tool_wrapper()
    w.begin("persona", [("user", "a"), ("assistant", "b")])
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
        w.begin("persona", [("user", "x"), ToolTurn((ToolCall("weather", {"city": "Bonn"}),))],
                tools=weather_registry().specs())
    del fake.prefill
    assert seen and set(seen) == {before}
    assert (w.total_tokens, w.turn_open) == before


@pytest.mark.parametrize("fault", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("ending", ["answer", "call"])
def test_a_reply_whose_recording_fails_is_taken_back_with_its_question(monkeypatch, fault, ending):
    w, fake = tool_wrapper()
    w.begin("persona", tools=weather_registry().specs())
    w.reply("eins").finish()
    before = _state(w)
    recorded = rendered_cache.RenderedCache._drop_unplaceable
    failing = iter([True])

    def recording(self):
        if next(failing, False):
            raise fault("recording failed")
        return recorded(self)

    monkeypatch.setattr(rendered_cache.RenderedCache, "_drop_unplaceable", recording)
    fake.call_script = ['fn:weather{"city": "Bonn"}'] if ending == "call" else []
    with pytest.raises(fault, match="recording failed"):
        w.reply("zwei", tool_call_policy="yield").finish()
    assert _state(w) == before
    assert_cache_reads_as_rendered(w, fake)
    fake.call_script = []
    w.reply("drei").finish()
    assert w.conversation()["messages"][-2] == {"role": "user", "content": "drei"}


def test_a_call_that_cannot_be_read_is_never_published_as_owed(monkeypatch):
    w, fake = tool_wrapper()
    w.begin("persona", tools=weather_registry().specs())
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
    w.begin("persona")
    for tool_call_policy in TOOL_CALL_POLICIES:
        with pytest.raises(ValueError, match="max_calls"):
            w.reply("x", tool_call_policy=tool_call_policy, max_calls=2)
    assert w.conversation()["messages"] == []


def test_a_failed_decode_of_the_answer_to_an_unreadable_call_closes_its_turn():
    fake = _Flaky(TOOLS, gen_tokens=3)
    w = ChatWrapper(Config(context_size=800, evict_above_fraction=1.0), context=fake)
    w.begin("s", tools=weather_registry().specs())
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
    assert w.conversation()["messages"][-2:] == [{"role": "user", "content": "Wetter?"},
                                                 {"role": "assistant", "content": ""}]
    w.reply("weiter").finish()


def test_a_reply_kept_without_text_holds_the_replys_opening_once():
    for size in range(100, 300):
        fake = FakeContext(TOOLS, gen_tokens=6, gen_text="Hallo.")
        w = ChatWrapper(Config(context_size=size, evict_above_fraction=1.0, min_reply_tokens=60),
                        context=fake)
        try:
            w.begin("s", tools=weather_registry().specs())
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
    w.begin("s", [("user", "a"), ("assistant", "b")] * 10 + [("user", "q")])
    with pytest.raises(ValueError, match="two user turns"):
        w.inject("noch eine Frage")
    resets, decoded = fake.resets, fake.total_prefilled()
    w.reply().finish()
    assert fake.resets == resets and fake.total_prefilled() - decoded < 100


def test_a_decode_that_fails_before_the_pinned_state_is_saved_loses_none_of_its_tokens():
    from llama_chat.cache_sync import CacheSync
    fake = _Flaky(PLAIN, can_shift=False, can_cut_tail=False)
    memory = CacheSync(fake)
    memory.pin(3)
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

    from llama_chat.cache_context import GenerationAccumulator
    from llama_chat.cache_sync import CacheSync
    fake = _Flaky(PLAIN, gen_tokens=2, can_shift=can_shift, can_cut_tail=can_cut_tail)
    memory, draw = CacheSync(fake), random.Random(0)

    def some_tokens():
        return [draw.randrange(1, 5) for _ in range(draw.randrange(4))]

    def generate(closed_early):
        pieces = memory.generate(draw.randrange(3), GenerationAccumulator())
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
        "pin": lambda: memory.pin(draw.randrange(len(memory.tokens) + 2)),
        "reset": memory.reset, "rest": memory.rest, "repair": memory.repair,
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
