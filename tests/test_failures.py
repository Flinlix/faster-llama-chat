"""Every action that is refused or fails leaves the conversation and the cache as they were before it, and usable."""

import datetime as real_datetime
import threading

import pytest

from llama_chat import ChatWrapper, Config, ContextOverflowError
from llama_chat.tools import ToolCall, ToolRegistry
from tests.fake_context import FakeContext
from tests.wrappers import (assert_consistent, held_text, newest_turn, rows, tool_wrapper,
                            weather_registry)


class _Flaky(FakeContext):
    """A fake cache whose next decode, cut or generation fails once, a decode part-way."""

    fail = None             # "decode", "cut" or "generate"

    def prefill(self, token_ids, start_pos, want_logits):
        if self.fail == "decode":
            self.fail = None
            self.cache.extend(token_ids[:len(token_ids) // 2])
            raise RuntimeError("decode failed")
        super().prefill(token_ids, start_pos, want_logits)

    def remove_from(self, pos):
        if self.fail == "cut":
            self.fail = None
            raise RuntimeError("the cache refused to remove its tokens")
        super().remove_from(pos)

    def generate(self, *args, **kwargs):
        if self.fail == "generate":
            self.fail = None
            self.cache.append(ord("x"))
            raise RuntimeError("generation failed")
        return (yield from super().generate(*args, **kwargs))


def _flaky(**kw):
    fake = _Flaky(**kw)
    return ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake), fake


# ----- restarts --------------------------------------------------------------------
def test_a_refused_restart_keeps_the_previous_conversation_usable():
    w, fake = _flaky(gen_tokens=2)
    w.begin("s")
    w.reply("eins").finish()
    before = w.history()
    with pytest.raises(ContextOverflowError):
        w.begin("x" * 600)
    assert w.history() == before and w.system_prompt == "s"
    w.reply("zwei").finish()
    assert [r["text"] for r in rows(w)][-2:] == ["zwei", "re"]


def test_a_restart_whose_decode_fails_puts_the_previous_conversation_back():
    w, fake = _flaky(gen_tokens=2)
    w.begin("s")
    w.reply("eins").finish()
    before = w.history()
    fake.fail = "decode"
    with pytest.raises(RuntimeError):
        w.begin("t", [("user", "a"), ("assistant", "b")])
    assert w.history() == before and w.system_prompt == "s"
    w.reply("zwei").finish()                   # the cache is decoded again, then goes on
    assert_consistent(w, fake)


def test_a_refused_restart_on_a_template_that_writes_the_date_keeps_the_conversation(monkeypatch):
    import json
    from pathlib import Path

    import llama_cpp.llama_chat_format as chat_format

    from tests.template_contexts import ConversationContext
    today = [real_datetime.datetime(2026, 11, 7, 23, 59)]
    monkeypatch.setattr(chat_format, "datetime", type("_Clock", (), {"now": staticmethod(lambda: today[0])}))
    fixture = json.loads((Path(__file__).parent / "templates" / "llama-3.2.json").read_text())
    context = ConversationContext(fixture, ["Eins.", "Zwei."])
    w = ChatWrapper(Config(context_size=400, min_reply_tokens=8), context=context)
    w.begin("Be brief.")
    w.reply("One?").finish()
    today[0] = real_datetime.datetime(2026, 11, 8, 0, 1)
    with pytest.raises(ContextOverflowError):
        w.begin("x" * 400)
    assert w.history() == [("user", "One?"), ("assistant", "Eins.")]
    assert w.reply("Two?").finish().text == "Zwei."


def test_a_failed_restart_keeps_the_call_its_conversation_owes_and_nothing_else():
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", calls="yield") if isinstance(e, ToolCall)]
    with pytest.raises(ContextOverflowError):
        w.begin("x" * 600)
    assert w.reply_open and w._conversation.owed_calls == [call]
    w.inject_tool_result(call, "12:00")
    w.reply().finish()
    assert not w.reply_open


# ----- results ---------------------------------------------------------------------
def test_a_result_that_leaves_no_room_for_the_reply_is_refused_and_the_call_stays_owed():
    fake = FakeContext(calls_tools=True)
    w = ChatWrapper(Config(context_size=300, evict_above_fraction=1.0, min_reply_tokens=40),
                    context=fake)
    w.begin("s")
    fake.call_script = ["fn:clock{}"]
    for n in range(100, 300):
        w.begin("s")
        try:
            calls = [e for e in w.reply("q" * n, calls="yield") if isinstance(e, ToolCall)]
        except ContextOverflowError:
            break
        held = list(fake.cache)
        try:
            w.inject_tool_result(calls[0], "12:00")
        except ContextOverflowError:
            assert fake.cache == held and w._conversation.owed_calls == calls
            with pytest.raises(ValueError, match="owe a result"):
                w.reply().finish()           # never resumed without its result
            return
        fake.call_script = ["fn:clock{}"]
    pytest.fail("no question length left the result without room")


def test_a_result_that_cannot_be_written_as_text_is_refused_and_the_call_stays_owed():
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", calls="yield") if isinstance(e, ToolCall)]
    with pytest.raises(ValueError, match="what JSON can write"):
        w.inject_tool_result(call, {"when": real_datetime.date(2026, 11, 8)})
    w.inject_tool_result(call, "12:00")
    w.reply().finish()


def test_a_tool_that_returns_what_cannot_be_written_is_answered_with_an_error():
    registry = ToolRegistry()
    registry.register(lambda: real_datetime.date(2026, 11, 8), "Das Datum.", name="clock",
                      parameters={"type": "object", "properties": {}})
    w, fake = tool_wrapper()
    w.begin("persona", tools=registry)
    fake.call_script = ["fn:clock{}"]
    reply = w.reply("welcher Tag?", calls="run").finish()
    assert reply.stop_reason == "end_of_turn" and not w.reply_open
    assert "unwritable_result" in held_text(w)


def test_a_resumed_turn_whose_answer_is_cut_at_once_closes_and_keeps_its_call():
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", calls="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call, "12:00")
    cancel = threading.Event()
    cancel.set()
    assert w.reply(cancel=cancel).finish().text == ""
    assert not w.reply_open and w._conversation._kept
    assert "fn:clock{}" in held_text(w)
    w.reply("und jetzt?").finish()
    assert "fn:clock{}" in held_text(w)


# ----- questions --------------------------------------------------------------------
@pytest.mark.parametrize("join", ["", None], ids=["writes-two-questions", "refuses-two-questions"])
def test_a_question_whose_decode_fails_is_not_recorded(join):
    w, fake = _flaky(user_join=join, gen_tokens=2)
    w.begin("s")
    w.reply("eins").finish()
    before = w.history()
    fake.fail = "decode"
    with pytest.raises(RuntimeError):
        w.reply("zwei").finish()
    assert w.history() == before
    w.reply("drei").finish()
    assert [r["text"] for r in rows(w)][-2] == "drei"
    assert_consistent(w, fake)


def test_a_question_without_room_for_its_reply_is_not_recorded():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=60),
                    context=fake)
    w.begin("s")
    with pytest.raises(ContextOverflowError):
        w.reply("hello").finish()
    assert w.history() == []


# ----- the cache's own failures -----------------------------------------------------
def test_a_cut_the_cache_refuses_is_noticed_and_repaired_by_the_next_action():
    w, fake = _flaky(gen_tokens=12, gen_text="hello there!", can_cut_tail=True)
    w.begin("s")
    w.reply("eins").finish()
    fake.fail = "cut"
    with pytest.raises(RuntimeError):
        w.truncate_last_reply("hello")
    w.reply("zwei").finish()
    assert_consistent(w, fake)


def test_after_a_failed_generation_nothing_is_decoded_into_the_cache_until_it_is_repaired():
    w, fake = _flaky(gen_tokens=2)
    w.begin("s")
    fake.fail = "generate"
    decoded = len(fake.prefill_calls)
    with pytest.raises(RuntimeError, match="generation failed"):
        w.reply("eins").finish()
    assert len(fake.prefill_calls) == decoded + 1          # the question, nothing after it
    w.reply("zwei").finish()
    assert_consistent(w, fake)


def test_one_turn_that_drops_twice_on_a_cache_that_cannot_shift_brings_it_back_once():
    fake = FakeContext(gen_tokens=2, can_shift=False)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5,
                           evict_to_fraction_without_shift=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        before = fake.rebuilds + fake.restores
        w.reply(f"message number {i} with some length").finish()
        assert fake.rebuilds + fake.restores - before <= 1
        assert_consistent(w, fake)


# ----- reading from another thread ------------------------------------------------
def test_the_history_read_from_another_thread_is_never_half_built(monkeypatch):
    w, fake = tool_wrapper()
    w.begin("persona", [("user", "a"), ("assistant", "b")])
    seen = []
    split = type(w._conversation)._split

    def reading(self, messages, generation_prompt):
        seen.append(len(w.history()))           # as another thread reads it, mid-action
        return split(self, messages, generation_prompt)

    monkeypatch.setattr(type(w._conversation), "_split", reading)
    w.reply("c").finish()
    assert set(seen) <= {2, 4}


# ----- what a request asks for ------------------------------------------------------
def test_a_call_ceiling_where_no_call_is_run_is_refused():
    w, fake = tool_wrapper()
    w.begin("persona")
    for calls in ("refuse", "yield", "withhold"):
        with pytest.raises(ValueError, match="max_calls"):
            w.reply("x", calls=calls, max_calls=2)
    assert w.history() == []


def test_a_generation_that_fails_before_it_says_anything_takes_its_question_back():
    w, fake = _flaky(gen_tokens=2)
    w.begin("s")
    w.reply("eins").finish()
    before = w.history()
    fake.fail = "generate"
    with pytest.raises(RuntimeError):
        w.reply("zwei").finish()
    assert w.history() == before
    w.reply("zwei").finish()
    assert w.history()[-2:] == [("user", "zwei"), ("assistant", "re")]
    assert_consistent(w, fake)



def test_a_failed_generation_keeps_a_question_its_request_did_not_ask():
    w, fake = _flaky(gen_tokens=2)
    w.begin("s")
    w.reply("eins").finish()
    w.inject("Lies diese Notiz.")
    before = w.history()
    fake.fail = "generate"
    with pytest.raises(RuntimeError):
        w.reply().finish()
    assert w.history() == before
    w.reply().finish()
    assert w.history()[-2:] == [("user", "Lies diese Notiz."), ("assistant", "re")]


def test_a_failed_generation_keeps_the_question_a_history_ends_on():
    w, fake = _flaky(gen_tokens=2)
    w.begin("s", [("user", "aus der Geschichte")])
    fake.fail = "generate"
    with pytest.raises(RuntimeError):
        w.reply().finish()
    assert w.history() == [("user", "aus der Geschichte")]


def test_a_question_taken_back_after_a_failed_generation_has_no_reply_reported():
    seen = []
    fake = _Flaky(gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake,
                    on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    fake.fail = "generate"
    with pytest.raises(RuntimeError):
        w.reply("eins").finish()
    assert seen == [("user", "eins")]


def test_a_failed_decode_of_the_answer_to_an_unreadable_call_closes_its_turn():
    fake = _Flaky(calls_tools=True, gen_tokens=3)
    w = ChatWrapper(Config(context_size=800, evict_above_fraction=1.0), context=fake)
    w.begin("s", tools=weather_registry())
    fake.call_script.append("@@@ kein Aufruf @@@")
    generate = fake.generate

    def then_failing_decode(*args, **kwargs):
        result = yield from generate(*args, **kwargs)
        fake.fail = "decode"            # the decode of the error result
        return result

    fake.generate = then_failing_decode
    with pytest.raises(RuntimeError, match="decode failed"):
        w.reply("Wetter?", calls="yield").finish()
    fake.generate = generate
    assert not w.reply_open and w._conversation.owed_calls == []
    assert w.history()[-2:] == [("user", "Wetter?"), ("assistant", "")]
    w.reply("weiter").finish()
    assert_consistent(w, fake)


def test_a_reply_kept_without_text_holds_the_replys_opening_once():
    for size in range(200, 300):
        fake = FakeContext(calls_tools=True, gen_tokens=6, gen_text="Hallo.")
        w = ChatWrapper(Config(context_size=size, evict_above_fraction=1.0, min_reply_tokens=60),
                        context=fake)
        try:
            w.begin("s", tools=weather_registry())
            w.reply("eins").finish()
            fake.call_script.append("@@@ kein Aufruf @@@")
            reply = w.reply("Wetter?", calls="yield").finish()
        except ContextOverflowError:
            continue
        if reply.stop_reason == "unreadable_tool_call" and not w.reply_open:
            assert newest_turn(w).count("<[turn>model") == 1
            return
    pytest.fail("no context size left the error result without room")


def test_a_refused_action_leaves_a_cache_that_cannot_cut_ready_to_generate():
    fake = FakeContext(can_shift=False, can_cut_tail=False, user_join=None, gen_tokens=3)
    w = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "a"), ("assistant", "b")] * 10 + [("user", "q")])
    with pytest.raises(ValueError, match="two user turns"):
        w.inject("noch eine Frage")
    rebuilds, decoded = fake.rebuilds, fake.total_prefilled()
    w.reply().finish()
    assert fake.rebuilds == rebuilds and fake.total_prefilled() - decoded < 100


def test_a_decode_that_fails_before_the_pinned_state_is_saved_loses_none_of_its_tokens():
    from llama_chat.cache_sync import CacheSync
    fake = _Flaky(can_shift=False, can_cut_tail=False)
    memory = CacheSync(fake)
    memory.pin(3)
    fake.fail = "decode"            # the first half, up to the pin
    with pytest.raises(RuntimeError):
        memory.decode(list(range(1, 11)))
    assert memory.tokens == list(range(1, 11))
    memory.repair()
    assert fake.cache == list(range(1, 11))
