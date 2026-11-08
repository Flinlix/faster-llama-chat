"""End-to-end wrapper tests using the model-free FakeContext.

These verify the two properties that matter most: (1) the wrapper only ever
prefills *new* tokens (never re-prefills survivors), and (2) the physical cache
stays exactly what the conversation holds it to be after every action,
including across evictions.
"""

import dataclasses
import threading

import pytest

from llama_chat.config import Config, Sampling
from llama_chat import ChatWrapper, ContextOverflowError
from tests.fake_context import BOS, DEFAULT_FRAGMENTS, FakeContext
from tests.wrappers import (assert_consistent, begun_wrapper, cache_tokens, held_renders,
                            tool_wrapper, rows)
def _n_kept(wrapper):
    """Messages the limit counts: everything but the system prompt."""
    return len(rows(wrapper)) - 1


class _TemplatingFake(FakeContext):
    """FakeContext that also renders a model chat template for the fragment check."""

    def __init__(self, render, fragments=DEFAULT_FRAGMENTS):
        super().__init__(fragments=fragments)
        self._render = render

    def render_with_model_template(self, messages, tools=None, **_options):
        return self._render(messages)


def _as_rendered(w, fake):
    """The cache equals the template's own render of the conversation."""
    assert_consistent(w, fake)
    found = rows(w)
    rendered = held_renders(w._template, [{"role": r["role"], "content": r["text"]} for r in found])
    assert "".join(chr(t) for t in fake.cache if t >= 0) in rendered


# ----- a reply without text, and the call policy ---------------------------
def test_a_reply_without_text_answers_the_newest_user_message():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    wrapper.inject("Wie spät ist es?")
    reply = wrapper.reply().finish()
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]
    assert reply.text == rows(wrapper)[-1]["text"]
    assert fake.cache == cache_tokens(wrapper)


def test_an_unknown_call_policy_is_refused():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="calls must be"):
        wrapper.reply("x", calls="sometimes")


# ----- begin ---------------------------------------------------------------
def test_begin_prefills_system_and_history_once():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=200, evict_above_fraction=1.0), context=fake)
    w.begin("sys", [("user", "hi"), ("assistant", "yo")])

    roles = [s["role"] for s in rows(w)]
    assert roles == ["system", "user", "assistant"]
    # One prefill call for the whole conversation == cheapest prefill.
    assert len(fake.prefill_calls) == 1
    assert_consistent(w, fake)


def test_begin_drops_oldest_history_beyond_threshold():
    fake = FakeContext()
    # system fragment "<[turn>user\ns<turn]>\n" + BOS is well under budget;
    # pick a tiny threshold so only the newest exchange fits.
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.08), context=fake)
    w.begin("s", [("user", "old message that is long"), ("assistant", "x"), ("user", "y")])

    # The oldest exchange is dropped whole: the kept history opens on a user turn.
    assert [s["text"] for s in rows(w)] == ["s", "y"]
    assert_consistent(w, fake)


def test_begin_has_bos_only_at_start():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "u")])
    # Exactly one BOS sentinel, at position 0.
    assert fake.cache[0] == BOS
    assert fake.cache.count(BOS) == 1


def test_begin_warms_up_once():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "u")])
    w.begin("s", [("user", "u")])
    # Warmed on the first begin only; later begins skip the redundant decode.
    assert fake.warmups == 1
    assert_consistent(w, fake)


# ----- inject --------------------------------------------------------------
def test_inject_appends_without_generating():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    before = w.total_tokens
    n_evicted_messages = w.inject("some context", role="user")
    assert n_evicted_messages == 0
    assert rows(w)[-1]["role"] == "user"
    assert w.total_tokens > before
    # No generation happened.
    assert fake.samplings == []
    assert_consistent(w, fake)


def test_inject_evicts_oldest_to_fit():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.1), context=fake)
    w.begin("s")
    w.inject("first injected chunk", role="user")
    n_evicted_messages = w.inject("second injected chunk", role="user")
    assert n_evicted_messages >= 1
    roles = [s["role"] for s in rows(w)]
    assert roles[0] == "system"  # system never evicted
    assert_consistent(w, fake)


# ----- replying without tools ----------------------------------------------
def test_a_refusing_reply_prefills_only_new_tokens_and_reuses_context():
    fake = FakeContext(gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, max_reply_tokens=50), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "hello")])
    prefilled_after_begin, held = fake.total_prefilled(), len(fake.cache)
    mark = len(fake.prefill_calls)

    reply = w.reply("what's up?").finish()

    # Every decode is counted, and none reaches back into what was held.
    new_prefill = fake.total_prefilled() - prefilled_after_begin
    assert new_prefill == reply.n_prefilled_tokens
    assert all(start >= held for start, _, _ in fake.prefill_calls[mark:])
    assert reply.n_generated_tokens == 4
    assert reply.stop_reason == "end_of_turn"
    # Assistant turn recorded for reuse next turn.
    assert rows(w)[-1]["role"] == "assistant"
    assert_consistent(w, fake)


def test_the_request_and_its_opening_tag_are_one_decode():
    """The prompt reaches the context in a single decode, closer aside.

    The request text and the assistant-open tag used to be two prefills, each
    paying a decode's fixed overhead on the one prefill a user is actually
    waiting through. Splitting them again would be invisible to every other
    assertion here, so it gets its own.
    """
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    mark = len(fake.prefill_calls)

    reply = w.reply("hi").finish()

    # The question and its generation prompt, then after the reply the turn's
    # closing and the next question's opening, decoded ahead.
    prompt_call, ahead_call = fake.prefill_calls[mark:]
    assert prompt_call[2] is True                 # logits: generation follows
    assert ahead_call[2] is False
    assert prompt_call[1] + ahead_call[1] == reply.n_prefilled_tokens
    assert_consistent(w, fake)


def test_a_second_generation_does_not_reprefill_history():
    fake = FakeContext(gen_tokens=3)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    w.reply("first").finish()
    mark, held, calls = fake.total_prefilled(), len(fake.cache), len(fake.prefill_calls)
    reply = w.reply("second").finish()
    # The second generation decodes only its own prompt and what follows its reply.
    assert fake.total_prefilled() - mark == reply.n_prefilled_tokens
    assert all(start >= held - len("reply<turn]>\n<[turn>") for start, _, _ in
               fake.prefill_calls[calls:])
    assert_consistent(w, fake)


# ----- streaming -----------------------------------------------------------
def test_a_refusing_reply_stream_yields_incremental_deltas():
    fake = FakeContext(gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    deltas = list(w.reply("hi"))
    assert "".join(deltas) == "abcd"
    assert rows(w)[-1]["role"] == "assistant"
    assert_consistent(w, fake)


def test_a_refusing_reply_drains_its_own_stream():
    fake = FakeContext(gen_tokens=3, gen_text="xyz")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    reply = w.reply("hi").finish()
    assert reply.text == "xyz"
    assert reply.n_generated_tokens == 3
    assert_consistent(w, fake)


def test_an_early_close_keeps_cache_consistent():
    # Consume only the first delta, then abandon the stream (close the generator).
    fake = FakeContext(gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")

    stream = w.reply("hi")
    first = next(stream)
    assert first == "a"
    stream.close()  # stop early

    # The assistant turn was still recorded, and the cache equals what is recorded:
    # exactly the tokens that reached the cache were recorded, no more.
    assert rows(w)[-1]["role"] == "assistant"
    assert_consistent(w, fake)

    # A following generation proceeds normally on the consistent cache.
    reply = w.reply("again").finish()
    assert reply.stop_reason == "end_of_turn"
    assert_consistent(w, fake)


def test_cancel_event_interrupts_midstream():
    fake = FakeContext(gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")

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
    assert_consistent(w, fake)


def test_cancel_set_before_start_yields_an_empty_interrupted_reply():
    fake = FakeContext(gen_tokens=5)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    cancel = threading.Event()
    cancel.set()
    reply = w.reply("hi", cancel=cancel).finish()
    assert reply.stop_reason == "interrupted"
    assert reply.n_generated_tokens == 0
    # A reply with nothing in it is no reply: only the request stays.
    assert rows(w)[-1]["role"] == "user"
    assert_consistent(w, fake)


def test_a_stop_string_ends_the_reply_and_is_trimmed_from_it():
    fake = FakeContext(gen_tokens=10, gen_text="helloSTOPx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin("s")
    with w.reply("hi", stop=["STOP"]) as stream:
        shown = "".join(stream)
    assert shown == stream.reply.text == "hello"
    assert stream.reply.stop_reason == "stop_string"   # not "interrupted"
    # What reached the cache stays recorded, so the conversation describes it.
    assert_consistent(w, fake)


def test_generator_close_marks_interrupted():
    # stream.close() records the same stop reason on the accumulator (the Reply
    # itself is unreachable after close(): Python offers no return channel).
    from llama_chat.cache_context import GenerationAccumulator

    fake = FakeContext(gen_tokens=5, gen_text="abcde")
    acc = GenerationAccumulator()
    g = fake.generate(0, 5, out=acc)
    next(g)
    g.close()
    assert acc.stop_reason == "interrupted"


def test_generation_is_capped_to_the_context_size():
    fake = FakeContext(gen_tokens=100)  # would generate 100 if uncapped
    w = ChatWrapper(Config(context_size=110, evict_above_fraction=1.0, max_reply_tokens=100), context=fake)
    w.begin("s")
    reply = w.reply("hello").finish()
    assert w.total_tokens <= 110
    assert reply.n_generated_tokens < 100
    assert_consistent(w, fake)


def test_a_refusing_reply_evicts_then_stays_within_context():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
        roles = [s["role"] for s in rows(w)]
        assert roles[0] == "system"
        assert w.total_tokens <= w._config.context_size
        assert_consistent(w, fake)


def test_the_cache_rests_under_the_threshold_after_a_generation():
    # The reply is appended first, then older turns are trimmed, so the cache
    # rests at or below threshold once the generation returns.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
        assert w.total_tokens <= w._config.evict_above_tokens  # rests under threshold
        assert rows(w)[0]["role"] == "system"        # system survives
        assert [r["role"] for r in rows(w)][1:2] in ([], ["user"])  # opens on a user turn
        assert_consistent(w, fake)


def test_turn_closes_after_the_eviction_not_before():
    # An eviction's K-shift lands on whatever decodes next, so the closer is
    # decoded last and carries it instead of the next generation's prefill.
    # The assertion is what pins that ordering. See the README,
    # "How eviction works".
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin("s")

    evicting = 0
    for i in range(8):
        n_evictions = len(fake.evictions)
        reply = w.reply(f"message number {i} with some length").finish()
        if reply.n_evicted_messages:
            evicting += 1
            # The eviction landed, and the closer is the turn's last prefill.
            assert len(fake.evictions) > n_evictions
            last = fake.prefill_calls[-1]
            assert last[0] + last[1] == w.total_tokens   # closer at the tail
            assert last[2] is False                      # no logits wanted
        assert_consistent(w, fake)
    assert evicting


def test_rebuild_mode_does_not_decode_the_closer_twice():
    # can_shift=False brings the cache back from a saved state or decodes it
    # again whole, closer included, so closing the turn afterwards must be a
    # no-op rather than a duplicate decode.
    fake = FakeContext(gen_tokens=2, can_shift=False)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin("s")
    brought_back = 0
    for i in range(8):
        before = fake.rebuilds + fake.restores
        w.reply(f"message number {i} with some length").finish()
        brought_back += fake.rebuilds + fake.restores > before
        assert_consistent(w, fake)          # would double-count the closer
    assert brought_back


# ----- the message limit, OR'd with the token threshold --------------------
def test_a_generation_evicts_down_to_the_message_limit():
    # Far under the token threshold, so the message count is the only trigger.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_messages=4), context=fake)
    w.begin("s")
    for i in range(8):
        w.reply(f"q{i}").finish()
        assert _n_kept(w) <= 4
        assert w.total_tokens < w._config.evict_above_tokens  # the tokens never bound
        assert rows(w)[0]["role"] == "system"       # system still survives
        assert rows(w)[-1]["role"] == "assistant"   # newest turn kept
        assert_consistent(w, fake)


def test_the_tighter_of_the_two_limits_is_the_one_that_binds():
    # Same conversation under each limit alone, then under both: whichever
    # would evict more is what the pair evicts, because either one triggers.
    def kept_under(**limits):
        fake = FakeContext(gen_tokens=2)
        w = ChatWrapper(Config(context_size=400, **limits), context=fake)
        w.begin("s")
        for i in range(8):
            w.reply(f"message number {i} with some length").finish()
        assert_consistent(w, fake)
        return _n_kept(w)

    by_tokens = kept_under(evict_above_fraction=0.5)
    by_messages = kept_under(evict_above_fraction=1.0, max_history_messages=2)
    both = kept_under(evict_above_fraction=0.5, max_history_messages=2)
    assert both == min(by_tokens, by_messages)


def test_a_limit_that_never_binds_changes_nothing():
    def snapshot_under(**limits):
        fake = FakeContext(gen_tokens=2)
        w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5, **limits),
                        context=fake)
        w.begin("s")
        for i in range(8):
            w.reply(f"message number {i} with some length").finish()
        return rows(w)

    assert snapshot_under() == snapshot_under(max_history_messages=100)


def test_inject_evicts_to_leave_room_under_the_message_limit():
    # inject evicts before appending, so the arriving message counts against
    # the limit while it is still absent from the conversation.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_messages=3), context=fake)
    w.begin("s", [("user", "a"), ("assistant", "b"), ("user", "c")])
    assert _n_kept(w) == 3
    # The oldest exchange goes whole, so the survivors open on a user turn.
    assert w.inject("reference material") == 2
    assert [r["text"] for r in rows(w)][1:] == ["c", "reference material"]
    assert_consistent(w, fake)


def test_begin_prefills_no_more_than_the_message_limit():
    # Without the cap, begin would prefill history the first turn then evicts.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_messages=2), context=fake)
    w.begin("s", [("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")])
    assert [r["text"] for r in rows(w)] == ["s", "c", "d"]  # newest two
    assert_consistent(w, fake)


def test_a_limit_of_one_keeps_only_the_system_prompt_between_turns():
    # An exchange is two messages and goes whole, so under a limit of one the
    # reply leaves with its request; the turn still closes cleanly.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_messages=1), context=fake)
    w.begin("s")
    for i in range(3):
        w.reply(f"q{i}").finish()
        assert [r["role"] for r in rows(w)] == ["system"]
        assert_consistent(w, fake)


def test_the_limit_holds_across_every_action_that_appends():
    # The property, not three examples: whatever the mix of actions, the count
    # never exceeds the limit and the system prompt is never the one dropped.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_messages=3), context=fake)
    w.begin("s", [("user", "old"), ("assistant", "older")])
    for i in range(6):
        w.inject(f"context {i}")
        assert _n_kept(w) <= 3
        w.reply(f"q{i}").finish()
        assert _n_kept(w) <= 3
        assert rows(w)[0]["role"] == "system"
        assert_consistent(w, fake)


def test_eviction_shift_mode_does_not_rebuild():
    # Default cache supports shifting: eviction uses apply_eviction, never rebuild.
    fake = FakeContext(gen_tokens=2)  # can_shift=True
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin("s")
    rebuilds_after_begin = fake.rebuilds  # begin() resets once
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
    assert len(fake.evictions) > 0          # eviction did happen
    assert fake.rebuilds == rebuilds_after_begin  # but never via rebuild
    assert_consistent(w, fake)


def test_eviction_rebuild_mode_when_cache_cannot_shift():
    # Cache cannot shift: eviction drops oldest from bookkeeping and re-prefills.
    fake = FakeContext(gen_tokens=2, can_shift=False)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin("s")
    rebuilds_after_begin = fake.rebuilds
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
        roles = [s["role"] for s in rows(w)]
        assert roles[0] == "system"             # system survives
        assert w.total_tokens <= w._config.context_size   # never overflows
        assert_consistent(w, fake)             # the cache as recorded after it is brought back
    assert len(fake.evictions) == 0             # shift path never used
    assert fake.rebuilds + fake.restores > rebuilds_after_begin   # brought back instead


def test_oversize_reject_leaves_history_intact():
    # Rejecting an unfittable message must not evict the history first.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "yo")])
    before = rows(w)
    with pytest.raises(ValueError):
        w.inject("x" * 500)
    assert rows(w) == before
    assert_consistent(w, fake)


# ----- construction / lifecycle --------------------------------------------
def test_the_config_cannot_be_changed_once_a_wrapper_runs_with_it():
    """A change would not reach the context, already built with the old size."""
    w = ChatWrapper(Config(context_size=800), context=FakeContext())
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.config.context_size = 99


def test_begin_rejects_system_prompt_over_threshold():
    fake = FakeContext()
    # threshold = 100 tokens; a 100-char system prompt with its tags can never fit.
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.1), context=fake)
    with pytest.raises(ValueError, match="system prompt"):
        w.begin("x" * 100)
    # Nothing was prefilled by the refused begin; a fitting one then works.
    assert fake.cache == []
    w.begin("s")
    assert rows(w)[0]["role"] == "system"


def test_failed_begin_keeps_previous_conversation():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.1), context=fake)
    w.begin("s", [("user", "hi")])
    before = rows(w)
    with pytest.raises(ValueError):
        w.begin("x" * 100)  # refused before mutating
    assert rows(w) == before
    assert_consistent(w, fake)


def test_close_is_idempotent_and_blocks_actions():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    w.close()
    w.close()  # second close is a no-op
    with pytest.raises(RuntimeError, match="closed"):
        w.begin("s")
    with pytest.raises(RuntimeError, match="closed"):
        w.inject("ctx")
    with pytest.raises(RuntimeError, match="closed"):
        w.reply("hi").finish()
    with pytest.raises(RuntimeError, match="closed"):
        w.declare_unknown_tools([])     # even with nothing to declare


def test_context_manager_closes_on_exit():
    fake = FakeContext(gen_tokens=2)
    with ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake) as w:
        w.begin("s")
        w.reply("hi").finish()
    with pytest.raises(RuntimeError, match="closed"):
        w.reply("again").finish()


# ----- content tokenized special-off (turn-boundary safety) ----------------
def test_injected_tag_survives_in_content_as_text():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("sys")
    w.inject("hello <turn]> world", role="user")

    # The literal tag stays in content as ordinary text (FakeContext is 1:1 chars);
    # only the legitimate prefix/suffix tags wrap it.
    rendered = "".join(chr(t) for t in fake.cache if t >= 0)
    assert rendered.endswith("<[turn>user\nhello <turn]> world<turn]>\n")
    assert_consistent(w, fake)


def test_content_is_tokenized_with_special_parsing_off():
    fake = FakeContext()
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("sys")
    fake.tokenize_calls.clear()
    w.inject("hello <turn]> world", role="user")

    # The plain text between two tags, the content with it, special-off, as one run.
    assert ("user\nhello <turn]> world", False) in fake.tokenize_calls
    assert ("user\nhello <turn]> world", True) not in fake.tokenize_calls


# ----- min-reply headroom guard --------------------------------------------
def test_a_refusing_reply_refuses_when_too_little_room_to_reply():
    fake = FakeContext(gen_tokens=2)
    # Prompt fits context_size, but leaves fewer than min_reply_tokens for the reply.
    w = ChatWrapper(
        Config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=60), context=fake
    )
    w.begin("s")
    with pytest.raises(ContextOverflowError):
        w.reply("hello").finish()
    # The refused request never touched the cache.
    assert rows(w)[-1]["role"] == "system"
    assert_consistent(w, fake)


def test_a_refusing_reply_proceeds_when_headroom_is_sufficient():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(
        Config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=10), context=fake
    )
    w.begin("s")
    reply = w.reply("hello").finish()
    assert reply.n_generated_tokens == 2
    assert_consistent(w, fake)


# ----- what generation reads from the template, checked at load -----------
def test_a_template_whose_closing_after_a_reply_is_not_where_its_tags_say_is_refused(monkeypatch):
    from llama_chat.resolved_template import ResolvedTemplate
    read = ResolvedTemplate._resolve_fragments
    monkeypatch.setattr(ResolvedTemplate, "_resolve_fragments",
                        lambda self: dataclasses.replace(read(self), assistant_suffix="<end>\n"))
    with pytest.raises(ValueError, match="closing after a reply"):
        ChatWrapper(Config(), context=FakeContext())


def test_a_template_whose_generation_prompt_is_not_where_its_tags_say_is_refused(monkeypatch):
    from llama_chat.resolved_template import ResolvedTemplate
    read = ResolvedTemplate._resolve_fragments
    monkeypatch.setattr(ResolvedTemplate, "_resolve_fragments",
                        lambda self: dataclasses.replace(read(self), generation_prompt="<x>"))
    with pytest.raises(ValueError, match="generation prompt"):
        ChatWrapper(Config(), context=FakeContext())


def test_a_restart_says_how_much_history_it_kept():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000), context=fake)
    assert w.begin("s", [("user", "a"), ("assistant", "b"), ("user", "c")]) == 3
    assert w.begin("s") == 0


# ----- sampling per request ------------------------------------------------
def test_a_seed_the_config_refuses_is_refused_for_one_request_too():
    with pytest.raises(ValueError, match="seed"):
        Sampling(seed=-1).applied_to(Config())


@pytest.mark.parametrize("calls", ["refuse", "registry", "yield"])
def test_every_call_policy_hands_its_sampling_to_the_generation(calls):
    from llama_chat.tools import ToolRegistry
    w, fake = begun_wrapper()
    sampling = Sampling(temperature=0.1)
    w.reply("hi", calls=ToolRegistry() if calls == "registry" else calls,
            sampling=sampling).finish()
    assert fake.samplings == [sampling]


def test_a_stream_and_a_resume_hand_it_over_as_well():
    w, fake = begun_wrapper()
    sampling = Sampling(top_p=0.5)
    list(w.reply("hi", sampling=sampling))
    list(w.reply("again", calls="yield", sampling=sampling))
    assert fake.samplings == [sampling, sampling]


def test_a_seed_out_of_range_is_refused_before_anything_is_prefilled():
    with pytest.raises(ValueError, match="seed"):
        Sampling(seed=2**40)


# ----- a system turn where the template has one or none --------------------
def test_a_system_prompt_is_refused_where_the_template_has_no_system_turn():
    from tests.fake_context import DEFAULT_FRAGMENTS
    import dataclasses
    folded = dataclasses.replace(DEFAULT_FRAGMENTS, system_prefix="<[turn>user\n")
    w = ChatWrapper(Config(context_size=500), context=FakeContext(fragments=folded))
    with pytest.raises(ValueError, match="no system turn of its own"):
        w.begin("persona")
    w.begin("")                         # no system prompt is still fine


def test_a_system_message_later_on_has_the_tags_of_a_later_system_turn():
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext())
    w.begin("s")
    w.inject("hi", "user")
    w.inject("Notiz", "system")
    assert "".join(chr(t) for t in w._context.cache if t >= 0).endswith(
        "<[turn>system\nNotiz<turn]>\n")


# ----- two user turns in a row ---------------------------------------------
def test_a_request_after_a_reply_without_an_answer_closes_the_user_turn_as_the_template_does():
    w, fake = begun_wrapper(user_join="<sep>", gen_tokens=0)
    w.reply("eins").finish()
    assert [m["role"] for m in rows(w)] == ["system", "user"]
    fake._gen_tokens = 5
    w.reply("zwei").finish()
    _as_rendered(w, fake)


def test_a_template_that_refuses_two_user_turns_keeps_a_reply_without_an_answer():
    w, fake = begun_wrapper(user_join=None, gen_tokens=0)
    w.reply("eins").finish()
    found = [(m["role"], m["text"]) for m in rows(w)]
    assert found[1:] == [("user", "eins"), ("assistant", "")]
    fake._gen_tokens = 5
    w.reply("zwei").finish()
    _as_rendered(w, fake)


def test_an_injected_user_turn_after_a_user_turn_is_joined_or_refused():
    w, fake = begun_wrapper(user_join="<sep>")
    w.inject("eins")
    w.inject("zwei")
    _as_rendered(w, fake)
    w, fake = begun_wrapper(user_join=None)
    w.inject("eins")
    before = list(fake.cache)
    with pytest.raises(ValueError, match="two user turns"):
        w.inject("zwei")
    assert fake.cache == before and len(rows(w)) == 2


def test_a_history_with_two_user_turns_in_a_row_is_joined_or_refused():
    fake = FakeContext(user_join="<sep>")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "eins"), ("user", "zwei"), ("assistant", "drei")])
    _as_rendered(w, fake)
    fake = FakeContext(user_join=None)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    with pytest.raises(ValueError, match="two user turns"):
        w.begin("s", [("user", "eins"), ("user", "zwei")])


def test_a_refused_history_is_explained_by_what_would_be_kept_of_it():
    """Two user turns in a row that would be dropped are not why the rest is refused."""
    class _RefusesOneText(FakeContext):
        def render_with_model_template(self, messages, tools=None, **options):
            if any(m.get("content") == "refused" for m in messages):
                return None
            return super().render_with_model_template(messages, tools, **options)

    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=_RefusesOneText(user_join=None))
    with pytest.raises(ValueError, match="refuses to write this conversation"):
        w.begin("s", [("user", "eins"), ("user", "zwei"), ("assistant", "drei"),
                      ("user", "refused"), ("assistant", "vier")])


def test_evicting_the_first_of_two_user_turns_takes_its_join_with_it():
    fake = FakeContext(user_join="<sep>")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, max_history_messages=1),
                    context=fake)
    w.begin("s")
    w.inject("eins")
    w.inject("zwei")
    assert [m["text"] for m in rows(w)] == ["s", "zwei"]
    _as_rendered(w, fake)


def test_a_config_larger_than_the_cache_context_is_refused_at_the_start():
    small = FakeContext()
    small.context_size = 256
    with pytest.raises(ValueError, match="holds 256 tokens"):
        ChatWrapper(Config(context_size=512), context=small)


def test_a_decode_that_fails_halfway_is_repaired_by_the_next_action():
    # It used to leave the cache and the conversation apart until the next begin.
    fake = FakeContext(gen_tokens=4)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.9), context=fake)
    w.begin("persona")
    w.reply("hi").finish()
    prefill = fake.prefill

    def fails_halfway(token_ids, start_pos, want_logits):
        prefill(token_ids[:len(token_ids) // 2], start_pos, want_logits)
        fake.prefill = prefill
        raise RuntimeError("the engine refused the batch")

    fake.prefill = fails_halfway
    with pytest.raises(RuntimeError, match="refused the batch"):
        w.inject("lost on the way")
    w.reply("again").finish()
    assert_consistent(w, fake)
    assert [m["role"] for m in rows(w)][-2:] == ["user", "assistant"]


# ----- the message log --------------------------------------------------------
def test_the_message_log_holds_only_what_was_recorded_an_early_closed_reply_included():
    seen = []
    fake = FakeContext(gen_tokens=5)
    w = ChatWrapper(Config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=10),
                    context=fake, on_message=lambda role, text: seen.append((role, text)))
    w.begin("s")
    with pytest.raises(ContextOverflowError):
        w.inject("x" * 200)                      # refused: never reported
    with pytest.raises(ContextOverflowError):
        w.reply("y" * 200).finish()
    stream = w.reply("hi")
    next(stream)
    stream.close()                               # closed early: recorded, so reported
    assert seen == [("user", "hi"), ("assistant", "r")]
    assert rows(w)[-1] == {"role": "assistant", "text": "r"}


def test_config_holds_the_sampling_the_model_supplies_where_the_wrapper_loads_it(monkeypatch):
    given = Config(context_size=512)
    in_use = dataclasses.replace(given, temperature=0.6, top_p=0.9)

    class _Loaded(FakeContext):
        def __init__(self, config, on_load_progress=None):
            super().__init__()
            self.config = in_use

    monkeypatch.setattr("llama_chat.wrapper.KVContext", _Loaded)
    assert ChatWrapper(given).config == in_use
    assert ChatWrapper(given, context=FakeContext()).config == given
