"""Rule 9: actions take turns, a reply asked for while another runs waits for no lock, a cancel from elsewhere stops a reply, and closing ends what the closing thread reads."""

import threading

import pytest

from llama_chat.config import Config
from llama_chat import ChatWrapper, InvalidArgumentError
from llama_chat.generation import generate
from tests.chat_messages import opened
from tests.fake_context import FakeContext
from tests.invented_templates import PLAIN
from tests.wrappers import add_tools, assert_consistent, tool_wrapper, rows


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


def test_generator_close_marks_interrupted():
    # stream.close() records the same stop reason on the accumulator (the Reply
    # itself is unreachable after close(): Python offers no return channel).
    from llama_chat.generation import GenerationAccumulator

    fake = FakeContext(PLAIN, gen_tokens=5, gen_text="abcde")
    acc = GenerationAccumulator()
    g = generate(fake, 0, 5, out=acc)
    next(g)
    g.close()
    assert acc.stop_reason == "interrupted"


def test_close_is_idempotent_and_blocks_actions():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    w.close()
    w.close()  # second close is a no-op
    with pytest.raises(RuntimeError, match="closed"):
        w.begin(opened("s"))
    with pytest.raises(RuntimeError, match="closed"):
        w.inject("ctx")
    with pytest.raises(RuntimeError, match="closed"):
        w.reply("hi").finish()
    with pytest.raises(RuntimeError, match="closed"):
        add_tools(w, [])     # even with nothing to declare


def test_context_manager_closes_on_exit():
    fake = FakeContext(PLAIN, gen_tokens=2)
    with ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake) as w:
        w.begin(opened("s"))
        w.reply("hi").finish()
    with pytest.raises(RuntimeError, match="closed"):
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
        if threading.current_thread().name == "llama_chat after a reply":
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
    with caplog.at_level("WARNING", logger="llama_chat"):
        assert w.reply("Wie geht es?").finish().text == "Gut."
        assert w.wait_idle(timeout_s=5)
    assert any("after a reply failed" in r.getMessage() for r in caplog.records)
    assert w.reply("Und jetzt?").finish().text == "Gut."
