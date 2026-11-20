"""Rule 9: actions take turns, a reply asked for while another runs waits for no lock, a cancel from elsewhere stops a reply, and closing ends what the closing thread reads."""

import threading

import pytest

from llama_cpp_conversation.errors import OutOfOrderError

from llama_cpp_conversation.config import Config, generation_sampling
from llama_cpp_conversation import ChatWrapper, InvalidArgumentError
from llama_cpp_conversation.generation import generate

from tests.chat_messages import opened
from llama_cpp_conversation.testing.fake_context import FakeContext
from llama_cpp_conversation.testing.invented_templates import PLAIN, TOOLS
from llama_cpp_conversation.tool_registry import ToolRegistry
from llama_cpp_conversation.tools import ToolCall, ToolSpec
from tests.wrappers import add_tools, assert_consistent, numbered_exchanges, tool_wrapper, rows

# Every sampling value set, as a generation is given them.
SAMPLING = generation_sampling(Config(), None)


def test_asking_for_a_reply_while_another_runs_does_not_wait_for_it():
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("persona"))
    running = wrapper.reply("hallo")
    next(running)                               # holds the wrapper's lock
    asked = []
    asker = threading.Thread(target=lambda: asked.append(wrapper.reply("noch eine")),
                             daemon=True)
    asker.start()
    asker.join(timeout=5)
    running.close()
    assert len(asked) == 1 and asked[0].finish().text


def test_cancel_event_interrupts_midstream():
    fake = FakeContext(PLAIN, gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))

    cancel = threading.Event()
    stream = w.reply("hi", cancel=cancel)
    assert next(stream) == "a"
    cancel.set()  # e.g. from another thread
    try:
        while True:
            next(stream)
    except StopIteration as done:
        reply = done.value

    # The stream ends normally, so the Reply is delivered, unlike stream.close().
    assert reply.stop_reason == "interrupted"
    assert reply.n_generated_tokens == 1
    assert rows(w)[-1]["role"] == "assistant"
    assert_consistent(w, fake)
    # The spent event is scoped to that call; the next generation runs normally.
    assert w.reply("again").finish().stop_reason == "end_of_turn"


def test_cancel_set_before_start_yields_an_empty_interrupted_reply():
    fake = FakeContext(PLAIN, gen_tokens=5)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    cancel = threading.Event()
    cancel.set()
    before = w.conversation(), w.total_tokens
    reply = w.reply("hi", cancel=cancel).finish()
    assert reply.stop_reason == "interrupted"
    assert reply.n_generated_tokens == 0
    # Cancelled before it started: the request changes nothing, its question included.
    assert (w.conversation(), w.total_tokens) == before


def test_a_cancel_set_before_a_reply_also_stops_the_work_the_reply_before_left(monkeypatch):
    """A drop on a cache that cannot shift leaves the kept history to decode again; a cancelled request leaves that to the next action."""
    monkeypatch.setattr(ChatWrapper, "_start_settling", lambda self: None)    # left to the next action
    fake = FakeContext(PLAIN, gen_tokens=5, gen_text="Antwort", can_shift=False)
    w = ChatWrapper(Config(context_size=8000, evict_above_fraction=0.5), context=fake)
    w.begin(opened("s" * 50) + numbered_exchanges(220))
    w.reply("Frage? " + "x" * 60).finish()
    assert w._conversation.settle_due        # a drop, whose decode waits for the next action
    prefilled = fake.total_prefilled()
    cancel = threading.Event()
    cancel.set()
    assert w.reply("noch eine?", cancel=cancel).finish().stop_reason == "interrupted"
    assert fake.total_prefilled() - prefilled <= 512
    assert "noch eine?" not in [m["content"] for m in w.conversation()["messages"]]
    assert w.reply("und jetzt?").finish().stop_reason == "end_of_turn"


def test_a_reply_waiting_for_one_another_thread_holds_ends_at_its_cancel():
    fake = FakeContext(PLAIN, gen_tokens=50)
    w = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    first = w.reply("eins")
    next(first)                         # holds the wrapper, unread
    cancel, ended = threading.Event(), []
    waiting = threading.Thread(target=lambda: ended.append(w.reply("zwei", cancel=cancel).finish()),
                               daemon=True)
    waiting.start()
    assert w._waiting.wait(timeout=5)
    cancel.set()
    waiting.join(2)
    assert not waiting.is_alive() and ended[0].stop_reason == "interrupted"
    first.finish()
    assert "zwei" not in [m["content"] for m in w.conversation()["messages"]]


_REFUSED_FOR_ITS_ARGUMENTS = {
    "begin, a tool twice": lambda w: w.begin(opened("s"), tools=[_CLOCK, _CLOCK]),
    "begin, reasoning": lambda w: w.begin(opened("s"), reasoning=True),
    "begin, a message": lambda w: w.begin([{"role": "bogus", "content": "x"}]),
    "inject, a role": lambda w: w.inject("x", role="tool"),
}
_CLOCK = ToolSpec(name="clock", description="Die Zeit.", parameters={"type": "object", "properties": {}})


@pytest.mark.parametrize("action", sorted(_REFUSED_FOR_ITS_ARGUMENTS))
def test_an_action_refused_for_its_own_arguments_leaves_the_reply_its_thread_reads_unended(action):
    fake = FakeContext(TOOLS, gen_tokens=20)
    w = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    stream = w.reply("eins")
    next(stream)
    with pytest.raises(ValueError):
        _REFUSED_FOR_ITS_ARGUMENTS[action](w)
    assert stream.reply is None and list(stream)        # read on to its end


def test_generator_close_marks_interrupted():
    # stream.close() records the same stop reason on the accumulator (the Reply
    # itself is unreachable after close(): Python offers no return channel).
    from llama_cpp_conversation.generation import GenerationAccumulator

    fake = FakeContext(PLAIN, gen_tokens=5, gen_text="abcde")
    acc = GenerationAccumulator()
    g = generate(fake, 0, 5, sampling=SAMPLING, out=acc)
    next(g)
    g.close()
    assert acc.stop_reason == "interrupted"


def test_close_is_idempotent_and_blocks_actions():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    w.close()
    w.close()  # second close is a no-op
    with pytest.raises(OutOfOrderError, match="closed"):
        w.begin(opened("s"))
    with pytest.raises(OutOfOrderError, match="closed"):
        w.inject("ctx")
    with pytest.raises(OutOfOrderError, match="closed"):
        w.reply("hi").finish()
    with pytest.raises(OutOfOrderError, match="closed"):
        add_tools(w, [])     # even with nothing to declare


def test_context_manager_closes_on_exit():
    fake = FakeContext(PLAIN, gen_tokens=2)
    with ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake) as w:
        w.begin(opened("s"))
        w.reply("hi").finish()
    with pytest.raises(OutOfOrderError, match="closed"):
        w.reply("again").finish()


def test_closing_on_the_thread_that_reads_an_unfinished_reply_ends_it_and_keeps_the_callers_error():
    fake, closed = FakeContext(PLAIN, gen_tokens=5), []
    fake.close = lambda: closed.append(True)
    with pytest.raises(ValueError, match="the reader failed") as raised:
        with ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake) as chat:
            chat.begin(opened("persona"))
            for _ in chat.reply("hallo"):
                raise ValueError("the reader failed")
    assert raised.value.__context__ is None
    assert closed == [True]


def test_closing_ends_the_reply_this_thread_reads_though_a_newer_stream_was_made_since():
    fake, closed = FakeContext(PLAIN, gen_tokens=5), []
    fake.close = lambda: closed.append(True)
    chat = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    chat.begin(opened("persona"))
    reading = chat.reply("hallo")
    next(reading)
    chat.reply("noch was")
    chat.close()
    assert closed == [True]


@pytest.mark.parametrize("timeout_s, refused", [(float("inf"), False), (0, False), (0.01, False),
                                                (1e300, False), (2**63, False), (10**400, False),
                                                (float("nan"), True), (-1, True), ("1", True),
                                                (True, True)])
def test_waiting_for_idle_takes_infinity_as_no_timeout_and_refuses_what_is_no_time(timeout_s, refused):
    chat = ChatWrapper(Config(context_size=500), context=FakeContext(PLAIN))
    if refused:
        with pytest.raises((InvalidArgumentError, TypeError), match="timeout_s"):
            chat.wait_idle(timeout_s)
    else:
        assert chat.wait_idle(timeout_s) is True


# ----- the work after a reply, on a thread of its own ---------------------------------
class _HeldAhead(FakeContext):
    """A fake cache whose decodes after a reply's generation wait until ``go`` is set, noting in ``waiting`` that one does."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.go, self.waiting, self.holds = threading.Event(), threading.Event(), False

    def prefill(self, token_ids, start_pos, want_logits):
        if self.holds and not want_logits:
            self.waiting.set()
            assert self.go.wait(timeout=5)
        super().prefill(token_ids, start_pos, want_logits)


def test_a_reply_s_stream_ends_before_the_work_after_it_and_the_next_action_waits_for_it(
        settles_in_the_background):
    fake = _HeldAhead(PLAIN, gen_tokens=4, gen_text="Gut.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    fake.holds = True
    assert w.reply("Wie geht es?").finish().text == "Gut."     # the decode ahead still waits
    assert fake.waiting.wait(timeout=5) and not w.wait_idle(timeout_s=0.05)
    injected = threading.Thread(target=lambda: w.inject("Noch etwas."), daemon=True)
    injected.start()
    injected.join(timeout=0.1)
    assert injected.is_alive()                  # it waits for the work after the reply
    fake.holds = False
    fake.go.set()
    injected.join(timeout=5)
    assert not injected.is_alive() and w.wait_idle(timeout_s=5)
    assert w.reply().finish().text == "Gut."
    assert_consistent(w, fake)


def test_a_long_decode_after_a_reply_gives_way_to_an_action_that_waits(settles_in_the_background):
    """On a cache that cannot shift, the decode after an eviction stops between its parts once an action waits, and that action does what it needs."""
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="Gut.", can_shift=False, can_cut_tail=True)
    w = ChatWrapper(Config(context_size=8192, evict_above_fraction=0.5, min_reply_tokens=8),
                    context=fake)
    w.begin(opened("s", [m for n in range(60) for m in (
        {"role": "user", "content": f"Frage {n} " + "x" * 60},
        {"role": "assistant", "content": "Antwort " + "y" * 60})]))
    decode, parts, started = fake.prefill, [], threading.Event()

    def watched(token_ids, start_pos, want_logits):
        if threading.current_thread().name == "llama_cpp_conversation after a reply":
            parts.append(len(token_ids))
            if len(parts) == 1:
                started.set()
                assert w._waiting.wait(timeout=5)   # an action waits for the wrapper now
        decode(token_ids, start_pos, want_logits)

    fake.prefill = watched
    assert w.reply("frage " + "z" * 200).finish().text == "Gut."
    assert started.wait(timeout=5), "no eviction after the reply decoded the cache again"
    w.inject("Gleich weiter.")
    assert len(parts) == 1                      # it gave way after its first part
    assert w.wait_idle(timeout_s=5) and w.reply().finish().text == "Gut."
    assert_consistent(w, fake)


def test_a_failure_of_the_work_after_a_reply_is_logged_and_the_next_action_decodes_again(
        settles_in_the_background, caplog):
    fake = _HeldAhead(PLAIN, gen_tokens=4, gen_text="Gut.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    decode = fake.prefill

    def failing(token_ids, start_pos, want_logits):
        if not want_logits:
            fake.prefill = decode
            raise RuntimeError("the decode failed")
        decode(token_ids, start_pos, want_logits)

    fake.prefill = failing
    with caplog.at_level("WARNING", logger="llama_cpp_conversation"):
        assert w.reply("Wie geht es?").finish().text == "Gut."
        assert w.wait_idle(timeout_s=5)
    assert any("after a reply failed" in r.getMessage() for r in caplog.records)
    assert w.reply("Und jetzt?").finish().text == "Gut."


def test_the_work_after_a_reply_is_waited_for_when_the_process_ends(settles_in_the_background):
    """Its thread is no daemon, so that a program that ends right after a reply does not tear the model down under its decode."""
    fake = _HeldAhead(PLAIN, gen_tokens=4, gen_text="Gut.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    fake.holds = True
    w.reply("Wie geht es?").finish()
    assert fake.waiting.wait(timeout=5)
    working = [t for t in threading.enumerate() if t.name == "llama_cpp_conversation after a reply"]
    assert working and not any(t.daemon for t in working)
    fake.go.set()
    assert w.wait_idle(timeout_s=5)


def test_the_work_after_a_reply_is_left_to_the_next_action_where_the_wrapper_stays_held(
        monkeypatch):
    from llama_cpp_conversation import wrapper as module
    monkeypatch.setattr(module, "_AFTER_REPLY_WAIT_S", 0.05)
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="Gut.")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    w._conversation._settle_due = True          # as a reply leaves it
    with w._lock:
        waiting = threading.Thread(target=w._settle_when_free)
        waiting.start()
        waiting.join(timeout=5)
        assert not waiting.is_alive() and w._conversation.settle_due
    assert w.reply("Und?").finish().text == "Gut."   # which does it first
    assert not w._conversation.settle_due


@pytest.mark.parametrize("cache", [dict(can_shift=False, can_cut_tail=True), dict(can_shift=False),
                                   dict(can_shift=True)], ids=["cuts", "saves states", "shifts"])
@pytest.mark.parametrize("goes_on", [True, False], ids=["next question", "other conversation"])
def test_an_action_that_interrupts_the_work_after_a_reply_decodes_no_more_than_waiting(
        settles_in_the_background, cache, goes_on):
    """What the interrupted work did stands, so the action only finishes the rest of it, and a begin of another conversation not even that."""
    def decoded(interrupts: bool) -> int:
        fake = FakeContext(PLAIN, gen_tokens=4, gen_text="Gut.", **cache)
        w = ChatWrapper(Config(context_size=8192, evict_above_fraction=0.5, min_reply_tokens=8),
                        context=fake)
        w.begin(opened("s", [m for n in range(60) for m in (
            {"role": "user", "content": f"Frage {n} " + "x" * 60},
            {"role": "assistant", "content": "Antwort " + "y" * 60})]))
        decode, parts, started = fake.prefill, [], threading.Event()

        def watched(token_ids, start_pos, want_logits):
            if interrupts and threading.current_thread().name == "llama_cpp_conversation after a reply":
                parts.append(len(token_ids))
                if len(parts) == 1:
                    started.set()
                    assert w._waiting.wait(timeout=5)
            decode(token_ids, start_pos, want_logits)

        fake.prefill = watched
        before = fake.total_prefilled()
        w.reply("frage " + "z" * 200).finish()
        if interrupts:
            assert started.wait(timeout=5)
        else:
            w.wait_idle()
        if goes_on:
            w.inject("Gleich weiter.")
        else:
            w.begin(opened("ein ganz anderer Anfang", [{"role": "user", "content": "Hallo?"}]))
        w.reply().finish()
        w.wait_idle()
        assert_consistent(w, fake)
        return fake.total_prefilled() - before

    if goes_on or cache["can_shift"]:
        assert decoded(interrupts=True) <= decoded(interrupts=False) + 8
    else:
        # The rest of the conversation it replaces, decoded again after a drop, is never
        # decoded: at least a part of it saved.
        assert decoded(interrupts=True) + 256 <= decoded(interrupts=False)


def test_a_stream_closed_before_it_started_finishes_with_no_reply():
    w, fake = tool_wrapper()
    w.begin(opened("s"))
    stream = w.reply("hi")
    stream.close()
    assert stream.finish() is None


def test_close_from_another_thread_waits_for_the_reply_being_read():
    import time
    from tests.wrappers import begun_wrapper
    text = "Ein langer Satz, der viele Token braucht."
    wrapper, _ = begun_wrapper(gen_tokens=len(text), gen_text=text)
    stream = wrapper.reply("hi")
    first = next(stream)
    closing = threading.Thread(target=wrapper.close, daemon=True)
    closing.start()
    time.sleep(0.3)
    assert closing.is_alive(), "close() did not wait for the reply being read"
    assert first + "".join(stream) == text
    closing.join(10)
    assert not closing.is_alive()


def test_a_call_handed_to_the_reader_is_a_copy_whose_change_the_tool_never_sees():
    w, fake = tool_wrapper()
    w.begin(opened("persona"))
    given = []
    registry = ToolRegistry()
    def weather(city: str) -> str:
        given.append(city)
        return "sonnig"
    registry.tool(weather, description="Wetter.")
    fake.call_script = ["fn:weather{city:'Bonn'}"]
    for step in w.reply("wetter?", tool_call_policy=registry):
        if isinstance(step, ToolCall):
            step.arguments["city"] = "Paris"        # the reader's own copy
    assert given == ["Bonn"]


def test_a_stream_left_half_read_as_the_program_ends_prints_no_error():
    import subprocess
    import sys
    script = ("from llama_cpp_conversation import ChatWrapper, Config\n"
              "from llama_cpp_conversation.testing.fake_context import FakeContext\n"
              "from llama_cpp_conversation.testing.invented_templates import PLAIN\n"
              "w = ChatWrapper(Config(context_size=2000), context=FakeContext(PLAIN, gen_tokens=20))\n"
              "w.begin([{'role': 'system', 'content': 's'}])\n"
              "stream = w.reply('eins')\n"
              "next(stream)\n")
    ended = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert ended.returncode == 0 and "Traceback" not in ended.stderr, ended.stderr


def test_a_failure_of_the_watch_on_a_reply_nobody_reads_is_logged(monkeypatch, caplog):
    w = ChatWrapper(Config(context_size=500), context=FakeContext(PLAIN))

    def failing():
        raise RuntimeError("the watch failed")
    monkeypatch.setattr(w, "_end_unread_reply_now", failing)
    with caplog.at_level("ERROR", logger="llama_cpp_conversation"):
        w._end_unread_reply()
    assert "ending a reply nobody read failed" in caplog.text
