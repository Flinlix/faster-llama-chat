"""End-to-end wrapper tests using the model-free FakeContext.

These verify the two properties that matter most: (1) the wrapper only ever
prefills *new* tokens (never re-prefills survivors), and (2) the physical cache
stays exactly equal to the concatenation of the message table after every
action, including across evictions.
"""

import pytest

import dataclasses
import threading

from llama_chat.config import Config
from llama_chat.template import TemplateFormatter
from llama_chat.wrapper import ChatWrapper, ContextOverflowError
from tests.fake_context import BOS, DEFAULT_FRAGMENTS, FakeContext


def _config(**kw):
    return Config(**kw)


def _fragments(**kw):
    """The default fragments with the given fields overridden."""
    return dataclasses.replace(DEFAULT_FRAGMENTS, **kw)


def _flatten(wrapper):
    toks = []
    for m in wrapper._table.messages:
        toks.extend(m.token_ids)
    return toks


def _assert_consistent(wrapper, fake):
    """Physical cache == concatenation of the message table."""
    assert fake.cache == _flatten(wrapper)
    snap = wrapper.snapshot()
    if snap:
        assert snap[-1]["pos_end"] == wrapper.total_tokens == len(fake.cache)


# ----- begin -------------------------------------------------------------
def test_begin_prefills_system_and_history_once():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=200, evict_above_fraction=1.0), context=fake)
    w.begin("sys", [("user", "hi"), ("assistant", "yo")])

    roles = [s["role"] for s in w.snapshot()]
    assert roles == ["system", "user", "assistant"]
    # One prefill call for the whole conversation == cheapest prefill.
    assert len(fake.prefill_calls) == 1
    _assert_consistent(w, fake)


def test_begin_drops_oldest_history_beyond_threshold():
    fake = FakeContext()
    # system fragment "<|turn>user\ns<turn|>\n" + BOS is well under budget;
    # pick a tiny threshold so only the newest message fits.
    w = ChatWrapper(_config(context_size=1000, evict_above_fraction=0.08), context=fake)
    w.begin("s", [("user", "old message that is long"), ("assistant", "x")])

    roles = [s["role"] for s in w.snapshot()]
    # Oldest user message dropped; only system + newest assistant kept.
    assert roles == ["system", "assistant"]
    _assert_consistent(w, fake)


def test_begin_has_bos_only_at_start():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "u")])
    # Exactly one BOS sentinel, at position 0.
    assert fake.cache[0] == BOS
    assert fake.cache.count(BOS) == 1


def test_begin_warms_up_once():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "u")])
    w.begin("s", [("user", "u")])
    # Warmed on the first begin only; later begins skip the redundant decode.
    assert fake.warmups == 1
    _assert_consistent(w, fake)


# ----- inject ------------------------------------------------------------
def test_inject_appends_without_generating():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    before = w.total_tokens
    n_evicted_messages = w.inject("some context", role="user")
    assert n_evicted_messages == 0
    assert w.snapshot()[-1]["role"] == "user"
    assert w.total_tokens > before
    # No generation happened.
    assert all(not want_logits for *_, want_logits in fake.prefill_calls)
    _assert_consistent(w, fake)


def test_inject_evicts_oldest_to_fit():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=1000, evict_above_fraction=0.1), context=fake)
    w.begin("s")
    w.inject("first injected chunk", role="user")
    n_evicted_messages = w.inject("second injected chunk", role="user")
    assert n_evicted_messages >= 1
    roles = [s["role"] for s in w.snapshot()]
    assert roles[0] == "system"  # system never evicted
    _assert_consistent(w, fake)


# ----- replying without tools --------------------------------------------
def test_reply_without_tools_prefills_only_new_tokens_and_reuses_context():
    fake = FakeContext(gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0, max_reply_tokens=50), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "hello")])
    prefilled_after_begin = fake.total_prefilled()

    reply = w.reply_without_tools("what's up?")

    # The only new prefill is the request prompt (user fragment + assistant-open)
    # plus the turn-closing tag, never any survivor.
    close_len = len(fake.tokenize(w._formatter.assistant_close()))
    new_prefill = fake.total_prefilled() - prefilled_after_begin
    assert new_prefill == reply.n_prefilled_tokens + close_len
    assert reply.n_generated_tokens == 4
    assert reply.stop_reason == "end_of_turn"
    # Assistant turn recorded for reuse next turn.
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)


def test_the_request_and_its_opening_tag_are_one_decode():
    """The prompt reaches the context in a single decode, closer aside.

    The request text and the assistant-open tag used to be two prefills, each
    paying a decode's fixed overhead on the one prefill a user is actually
    waiting through. Splitting them again would be invisible to every other
    assertion here, so it gets its own.
    """
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    close_len = len(fake.tokenize(w._formatter.assistant_close()))
    mark = len(fake.prefill_calls)

    reply = w.reply_without_tools("hi")

    prompt_call, close_call = fake.prefill_calls[mark:]
    assert prompt_call[1] == reply.n_prefilled_tokens     # user fragment + assistant-open
    assert prompt_call[2] is True                 # logits: generation follows
    assert close_call[1] == close_len             # the closer, decoded last
    assert close_call[2] is False
    _assert_consistent(w, fake)


def test_a_second_generation_does_not_reprefill_history():
    fake = FakeContext(gen_tokens=3)
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    w.reply_without_tools("first")
    mark = fake.total_prefilled()
    reply = w.reply_without_tools("second")
    # The second generation prefills only its own prompt and closer; the fake's
    # append-only assertion already guarantees no survivor was re-prefilled.
    close_len = len(fake.tokenize(w._formatter.assistant_close()))
    assert fake.total_prefilled() - mark == reply.n_prefilled_tokens + close_len
    _assert_consistent(w, fake)


# ----- streaming ---------------------------------------------------------
def test_reply_without_tools_stream_yields_incremental_deltas():
    fake = FakeContext(gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    deltas = list(w.reply_without_tools_stream("hi"))
    assert "".join(deltas) == "abcd"
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)


def test_reply_without_tools_drains_its_own_stream():
    fake = FakeContext(gen_tokens=3, gen_text="xyz")
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    reply = w.reply_without_tools("hi")
    assert reply.text == "xyz"
    assert reply.n_generated_tokens == 3
    _assert_consistent(w, fake)


def test_an_early_close_keeps_cache_consistent():
    # Consume only the first delta, then abandon the stream (close the generator).
    fake = FakeContext(gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")

    stream = w.reply_without_tools_stream("hi")
    first = next(stream)
    assert first == "a"
    stream.close()  # stop early

    # The assistant turn was still recorded, and the cache equals the table:
    # exactly the tokens that reached the cache were recorded, no more.
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)

    # A following generation proceeds normally on the consistent cache.
    reply = w.reply_without_tools("again")
    assert reply.stop_reason == "end_of_turn"
    _assert_consistent(w, fake)


def test_cancel_event_interrupts_midstream():
    fake = FakeContext(gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")

    cancel = threading.Event()
    stream = w.reply_without_tools_stream("hi", cancel=cancel)
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
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)
    # The spent event is scoped to that call; the next generation runs normally.
    assert w.reply_without_tools("again").stop_reason == "end_of_turn"
    _assert_consistent(w, fake)


def test_cancel_set_before_start_yields_an_empty_interrupted_reply():
    fake = FakeContext(gen_tokens=5)
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    cancel = threading.Event()
    cancel.set()
    reply = w.reply_without_tools("hi", cancel=cancel)
    assert reply.stop_reason == "interrupted"
    assert reply.n_generated_tokens == 0
    # The empty assistant turn is still closed cleanly and recorded.
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)


def test_close_after_stop_hit_keeps_stop_reason():
    # A stop string sets stop_reason="stop_string" before the final delta is yielded;
    # a consumer that then closes the generator must not reclassify the
    # completed reply as "interrupted". Drives the real KVContext.generate
    # with a stubbed backend (no model needed).
    from types import SimpleNamespace
    from llama_chat.context import GenerationAccumulator, KVContext

    pieces = {0: b"hello ", 1: b"STOP"}
    toks = iter([0, 1])
    ctx = object.__new__(KVContext)
    ctx._backend = SimpleNamespace(
        sample=lambda sampler, ctx: next(toks),
        is_eog=lambda vocab, tok: False,
        decode=lambda ctx_, ids, pos, seq, logits, batch: None,
        token_to_piece_bytes=lambda vocab, tok: pieces[tok],
    )
    ctx._sampler = ctx._vocab = ctx._llama_context = None
    ctx._max_batch_tokens = 8

    acc = GenerationAccumulator()
    stream = ctx.generate(0, 10, ["STOP"], out=acc)
    assert next(stream) == "hel"   # held back by the stop-string window
    assert next(stream) == "lo "   # final delta; "stop_string" already recorded
    stream.close()            # consumer closes after seeing the reply end
    assert acc.stop_reason == "stop_string"  # not misreported as "interrupted"


def test_generator_close_marks_interrupted():
    # stream.close() records the same stop reason on the accumulator (the Reply
    # itself is unreachable after close(): Python offers no return channel).
    from llama_chat.context import GenerationAccumulator

    fake = FakeContext(gen_tokens=5, gen_text="abcde")
    acc = GenerationAccumulator()
    g = fake.generate(0, 5, [], out=acc)
    next(g)
    g.close()
    assert acc.stop_reason == "interrupted"


def test_generation_is_capped_to_the_context_size():
    fake = FakeContext(gen_tokens=100)  # would generate 100 if uncapped
    w = ChatWrapper(_config(context_size=110, evict_above_fraction=1.0, max_reply_tokens=100), context=fake)
    w.begin("s")
    reply = w.reply_without_tools("hello")
    assert w.total_tokens <= 110
    assert reply.n_generated_tokens < 100
    _assert_consistent(w, fake)


def test_reply_without_tools_evicts_then_stays_within_context():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        w.reply_without_tools(f"message number {i} with some length")
        roles = [s["role"] for s in w.snapshot()]
        assert roles[0] == "system"
        assert w.total_tokens <= w._config.context_size
        _assert_consistent(w, fake)


def test_the_cache_rests_under_the_threshold_after_a_generation():
    # The reply is appended first, then older turns are trimmed, so the cache
    # rests at or below threshold once the generation returns.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        w.reply_without_tools(f"message number {i} with some length")
        assert w.total_tokens <= w._config.evict_above_tokens  # rests under threshold
        assert w.snapshot()[0]["role"] == "system"        # system survives
        assert w.snapshot()[-1]["role"] == "assistant"    # newest turn kept
        _assert_consistent(w, fake)


def test_turn_closes_after_the_eviction_not_before():
    # An eviction's K-shift lands on whatever decodes next, so the closer is
    # decoded last and carries it instead of the next generation's prefill.
    # The assertion is what pins that ordering. See the README,
    # "How eviction works".
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")

    evicting = 0
    for i in range(8):
        n_evictions = len(fake.evictions)
        reply = w.reply_without_tools(f"message number {i} with some length")
        if reply.n_evicted_messages:
            evicting += 1
            # The eviction landed, and the closer is the turn's last prefill.
            assert len(fake.evictions) > n_evictions
            last = fake.prefill_calls[-1]
            assert last[0] + last[1] == w.total_tokens   # closer at the tail
            assert last[2] is False                      # no logits wanted
        _assert_consistent(w, fake)
    assert evicting


def test_rebuild_mode_does_not_decode_the_closer_twice():
    # can_shift=False re-prefills the whole table, closer included, so closing
    # the turn afterwards must be a no-op rather than a duplicate decode.
    fake = FakeContext(gen_tokens=2, can_shift=False)
    w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    rebuilt = 0
    for i in range(8):
        before = fake.rebuilds
        w.reply_without_tools(f"message number {i} with some length")
        rebuilt += fake.rebuilds > before
        _assert_consistent(w, fake)          # would double-count the closer
    assert rebuilt


# ----- the message limit, OR'd with the token threshold -------------------
def _n_kept(wrapper):
    """Messages the limit counts: everything but the system prompt."""
    return len(wrapper.snapshot()) - 1


def test_a_generation_evicts_down_to_the_message_limit():
    # Far under the token threshold, so the message count is the only trigger.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=4000, max_history_messages=4), context=fake)
    w.begin("s")
    for i in range(8):
        w.reply_without_tools(f"q{i}")
        assert _n_kept(w) <= 4
        assert w.total_tokens < w._config.evict_above_tokens  # the tokens never bound
        assert w.snapshot()[0]["role"] == "system"       # system still survives
        assert w.snapshot()[-1]["role"] == "assistant"   # newest turn kept
        _assert_consistent(w, fake)


def test_the_tighter_of_the_two_limits_is_the_one_that_binds():
    # Same conversation under each limit alone, then under both: whichever
    # would evict more is what the pair evicts, because either one triggers.
    def kept_under(**limits):
        fake = FakeContext(gen_tokens=2)
        w = ChatWrapper(_config(context_size=400, **limits), context=fake)
        w.begin("s")
        for i in range(8):
            w.reply_without_tools(f"message number {i} with some length")
        _assert_consistent(w, fake)
        return _n_kept(w)

    by_tokens = kept_under(evict_above_fraction=0.5)
    by_messages = kept_under(evict_above_fraction=1.0, max_history_messages=2)
    both = kept_under(evict_above_fraction=0.5, max_history_messages=2)
    assert both == min(by_tokens, by_messages)


def test_a_limit_that_never_binds_changes_nothing():
    def snapshot_under(**limits):
        fake = FakeContext(gen_tokens=2)
        w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.5, **limits),
                        context=fake)
        w.begin("s")
        for i in range(8):
            w.reply_without_tools(f"message number {i} with some length")
        return w.snapshot()

    assert snapshot_under() == snapshot_under(max_history_messages=100)


def test_inject_evicts_to_leave_room_under_the_message_limit():
    # inject evicts before appending, so the arriving message counts against
    # the limit while it is still absent from the table.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=4000, max_history_messages=3), context=fake)
    w.begin("s", [("user", "a"), ("assistant", "b"), ("user", "c")])
    assert _n_kept(w) == 3
    assert w.inject("reference material") == 1   # one dropped to make room
    assert _n_kept(w) == 3
    assert [r["text"] for r in w.snapshot()][1:] == ["b", "c", "reference material"]
    _assert_consistent(w, fake)


def test_begin_prefills_no_more_than_the_message_limit():
    # Without the cap, begin would prefill history the first turn then evicts.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=4000, max_history_messages=2), context=fake)
    w.begin("s", [("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")])
    assert [r["text"] for r in w.snapshot()] == ["s", "c", "d"]  # newest two
    _assert_consistent(w, fake)


def test_a_limit_of_one_keeps_the_newest_message_and_the_system_prompt():
    # The floor: eviction stops with one message left, so a turn still closes
    # and the reply it just produced is the message that survives.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=4000, max_history_messages=1), context=fake)
    w.begin("s")
    for i in range(3):
        w.reply_without_tools(f"q{i}")
        assert [r["role"] for r in w.snapshot()] == ["system", "assistant"]
        _assert_consistent(w, fake)


def test_the_limit_holds_across_every_action_that_appends():
    # The property, not three examples: whatever the mix of actions, the count
    # never exceeds the limit and the system prompt is never the one dropped.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=4000, max_history_messages=3), context=fake)
    w.begin("s", [("user", "old"), ("assistant", "older")])
    for i in range(6):
        w.inject(f"context {i}")
        assert _n_kept(w) <= 3
        w.reply_without_tools(f"q{i}")
        assert _n_kept(w) <= 3
        assert w.snapshot()[0]["role"] == "system"
        _assert_consistent(w, fake)


def test_eviction_shift_mode_does_not_rebuild():
    # Default cache supports shifting: eviction uses apply_eviction, never rebuild.
    fake = FakeContext(gen_tokens=2)  # can_shift=True
    w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    rebuilds_after_begin = fake.rebuilds  # begin() resets once
    for i in range(8):
        w.reply_without_tools(f"message number {i} with some length")
    assert len(fake.evictions) > 0          # eviction did happen
    assert fake.rebuilds == rebuilds_after_begin  # but never via rebuild
    _assert_consistent(w, fake)


def test_eviction_rebuild_mode_when_cache_cannot_shift():
    # Cache cannot shift: eviction drops oldest from bookkeeping and re-prefills.
    fake = FakeContext(gen_tokens=2, can_shift=False)
    w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    rebuilds_after_begin = fake.rebuilds
    for i in range(8):
        w.reply_without_tools(f"message number {i} with some length")
        roles = [s["role"] for s in w.snapshot()]
        assert roles[0] == "system"             # system survives
        assert w.total_tokens <= w._config.context_size   # never overflows
        _assert_consistent(w, fake)             # cache == table after rebuild
    assert len(fake.evictions) == 0             # shift path never used
    assert fake.rebuilds > rebuilds_after_begin  # rebuild path was exercised


def test_oversize_reject():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=80, evict_above_fraction=0.5, oversize_policy="reject"), context=fake)
    w.begin("s")
    with pytest.raises(ValueError):
        w.inject("x" * 100)


def test_oversize_reject_leaves_history_intact():
    # Rejecting an unfittable message must not evict the history first.
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=400, evict_above_fraction=0.5, oversize_policy="reject"), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "yo")])
    before = w.snapshot()
    with pytest.raises(ValueError):
        w.inject("x" * 500)
    assert w.snapshot() == before
    _assert_consistent(w, fake)


def test_oversize_truncate():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=200, evict_above_fraction=0.3, oversize_policy="truncate"), context=fake)
    w.begin("s")
    w.inject("y" * 500)
    assert w.total_tokens <= w._config.evict_above_tokens
    # The clipped message still ends with its turn-terminator tag, so the
    # cache never holds a malformed turn.
    suffix = fake.tokenize(DEFAULT_FRAGMENTS.user_suffix)
    assert fake.cache[-len(suffix):] == suffix
    _assert_consistent(w, fake)


def test_oversize_truncate_rejects_when_no_room_for_suffix():
    fake = FakeContext()
    # Threshold barely above the system prompt: the remaining budget is smaller
    # than the turn-terminator tag, so truncation cannot produce a valid turn.
    w = ChatWrapper(_config(context_size=200, evict_above_fraction=0.15, oversize_policy="truncate"), context=fake)
    w.begin("s")
    with pytest.raises(ValueError):
        w.inject("y" * 100)
    _assert_consistent(w, fake)


# ----- construction / lifecycle -------------------------------------------
def test_config_and_kwargs_are_mutually_exclusive():
    with pytest.raises(TypeError, match="not both"):
        ChatWrapper(_config(), context=FakeContext(), context_size=8192)


def test_begin_rejects_system_prompt_over_threshold():
    fake = FakeContext()
    # threshold = 50 tokens; a 100-char system prompt can never fit.
    w = ChatWrapper(_config(context_size=1000, evict_above_fraction=0.05), context=fake)
    with pytest.raises(ValueError, match="system prompt"):
        w.begin("x" * 100)
    # Nothing was prefilled by the refused begin; a fitting one then works.
    assert fake.cache == []
    w.begin("s")
    assert w.snapshot()[0]["role"] == "system"


def test_failed_begin_keeps_previous_conversation():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=1000, evict_above_fraction=0.05), context=fake)
    w.begin("s", [("user", "hi")])
    before = w.snapshot()
    with pytest.raises(ValueError):
        w.begin("x" * 100)  # refused before mutating
    assert w.snapshot() == before
    _assert_consistent(w, fake)


def test_close_is_idempotent_and_blocks_actions():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    w.close()
    w.close()  # second close is a no-op
    with pytest.raises(RuntimeError, match="closed"):
        w.begin("s")
    with pytest.raises(RuntimeError, match="closed"):
        w.inject("ctx")
    with pytest.raises(RuntimeError, match="closed"):
        w.reply_without_tools("hi")


def test_context_manager_closes_on_exit():
    fake = FakeContext(gen_tokens=2)
    with ChatWrapper(_config(context_size=500, evict_above_fraction=1.0), context=fake) as w:
        w.begin("s")
        w.reply_without_tools("hi")
    with pytest.raises(RuntimeError, match="closed"):
        w.reply_without_tools("again")


# ----- template formatting -----------------------------------------------
def test_template_formatter_renders_gemma_tags():
    fmt = TemplateFormatter(_fragments())
    assert fmt.fragment("user", "hi") == "<|turn>user\nhi<turn|>\n"
    assert fmt.assistant_open() == "<|turn>model\n"
    assert fmt.assistant_close() == "<turn|>\n"


def test_template_formatter_combines_fragments():
    fmt = TemplateFormatter(
        _fragments(user_prefix="<u>", user_suffix="</u>",
               assistant_prefix="<a>", assistant_suffix="</a>")
    )
    assert fmt.fragment("user", "hi") == "<u>hi</u>"
    assert fmt.fragment("assistant", "yo") == "<a>yo</a>"
    assert fmt.assistant_open() == "<a>"
    assert fmt.assistant_close() == "</a>"
    with pytest.raises(ValueError):
        fmt.fragment("unknown-role", "x")


def test_trim_content_strips_by_default():
    fmt = TemplateFormatter(_fragments())
    assert fmt.fragment("user", "  hi\n") == "<|turn>user\nhi<turn|>\n"


def test_trim_content_disabled_keeps_whitespace():
    fmt = TemplateFormatter(_fragments(trim_content=False))
    assert fmt.fragment("user", "  hi\n") == "<|turn>user\n  hi\n<turn|>\n"


# ----- content tokenized special-off (turn-boundary safety) --------------
def test_fragment_keeps_literal_tag_in_content():
    # Content is no longer stripped: the literal tag is preserved verbatim and
    # rendered safe instead by tokenizing content with special parsing off.
    fmt = TemplateFormatter(_fragments())
    assert fmt.fragment("user", "hi <turn|> there") == "<|turn>user\nhi <turn|> there<turn|>\n"


def test_injected_tag_survives_in_content_as_text():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0),
                    context=fake, fragments=DEFAULT_FRAGMENTS)
    w.begin("sys")
    w.inject("hello <turn|> world", role="user")

    # The literal tag stays in content as ordinary text (FakeContext is 1:1 chars);
    # only the legitimate prefix/suffix tags wrap it.
    rendered = "".join(chr(t) for t in w._table.messages[-1].token_ids)
    assert rendered == "<|turn>user\nhello <turn|> world<turn|>\n"
    _assert_consistent(w, fake)


def test_content_is_tokenized_with_special_parsing_off():
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=500, evict_above_fraction=1.0),
                    context=fake, fragments=DEFAULT_FRAGMENTS)
    w.begin("sys")
    fake.tokenize_calls.clear()
    w.inject("hello <turn|> world", role="user")

    # The fragment's three parts: prefix (special-on), content (special-off),
    # suffix (special-on).
    assert fake.tokenize_calls == [
        ("<|turn>user\n", True),
        ("hello <turn|> world", False),
        ("<turn|>\n", True),
    ]


# ----- min-reply headroom guard ------------------------------------------
def test_reply_without_tools_refuses_when_too_little_room_to_reply():
    fake = FakeContext(gen_tokens=2)
    # Prompt fits context_size, but leaves fewer than min_reply_tokens for the reply.
    w = ChatWrapper(
        _config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=60), context=fake
    )
    w.begin("s")
    with pytest.raises(ContextOverflowError):
        w.reply_without_tools("hello")
    # The refused request never touched the cache.
    assert w.snapshot()[-1]["role"] == "system"
    _assert_consistent(w, fake)


def test_reply_without_tools_proceeds_when_headroom_is_sufficient():
    fake = FakeContext(gen_tokens=2)
    w = ChatWrapper(
        _config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=10), context=fake
    )
    w.begin("s")
    reply = w.reply_without_tools("hello")
    assert reply.n_generated_tokens == 2
    _assert_consistent(w, fake)


# ----- model-template fragment validation --------------------------------
class _TemplatingFake(FakeContext):
    """FakeContext that also renders a model chat template for the fragment check."""

    def __init__(self, render, fragments=DEFAULT_FRAGMENTS):
        super().__init__(fragments=fragments)
        self._render = render

    def render_with_model_template(self, messages):
        return self._render(messages)


def test_fragment_check_passes_when_fragments_match_model():
    fmt = TemplateFormatter(_fragments())
    render = lambda msgs: "".join(fmt.fragment(m["role"], m["content"]) for m in msgs)
    # Constructs without error: model render == fragment render.
    ChatWrapper(_config(), context=_TemplatingFake(render))


def test_fragment_check_rejects_mismatched_fragments():
    render = lambda msgs: "".join(
        f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n" for m in msgs
    )
    with pytest.raises(ValueError, match="do not match the model"):
        ChatWrapper(_config(), context=_TemplatingFake(render))


def test_fragment_check_catches_trim_mismatch():
    # Model trims content; the wrapper's fragments say not to -> the whitespace
    # probe diverges and construction must fail.
    trimming = TemplateFormatter(_fragments())
    render = lambda msgs: "".join(trimming.fragment(m["role"], m["content"]) for m in msgs)
    with pytest.raises(ValueError, match="do not match the model"):
        ChatWrapper(context=_TemplatingFake(render, fragments=_fragments(trim_content=False)))


# ----- KV-cache / flash-attn config --------------------------------------
def test_the_unquantized_kv_cache_type_needs_no_flash_attention():
    # f16 is llama.cpp's own default and the non-flash path holds it, so
    # naming it explicitly must not be refused.
    from llama_chat.config import UNQUANTIZED_KV_CACHE_TYPE
    config = _config(kv_cache_type=UNQUANTIZED_KV_CACHE_TYPE, flash_attention=False)
    assert config.kv_cache_type == UNQUANTIZED_KV_CACHE_TYPE


@pytest.mark.parametrize("quantized", ["q8_0", "q5_1", "q5_0", "q4_1", "q4_0"])
def test_every_quantized_kv_cache_type_requires_flash_attention(quantized):
    with pytest.raises(ValueError, match="flash_attention"):
        _config(kv_cache_type=quantized, flash_attention=False)


def test_kv_cache_type_requires_flash_attn():
    with pytest.raises(ValueError, match="flash_attention"):
        Config(kv_cache_type="q8_0")


def test_kv_cache_type_rejects_unknown_name():
    with pytest.raises(ValueError, match="kv_cache_type"):
        Config(kv_cache_type="bogus", flash_attention=True)


def test_kv_cache_type_accepts_known_type_with_flash_attn():
    cfg = Config(kv_cache_type="q8_0", flash_attention=True)
    assert cfg.kv_cache_type == "q8_0"


def _set_flash_attn(cparams, enable, llama_cpp=None):
    """Drive Backend._set_flash_attn with fakes, bypassing llama_cpp import."""
    from types import SimpleNamespace
    from llama_chat._backend import Backend

    Backend._set_flash_attn(SimpleNamespace(_llama_cpp=llama_cpp), cparams, enable)


@pytest.mark.parametrize("enable", [True, False])
def test_set_flash_attn_bool_build(enable):
    from types import SimpleNamespace

    cparams = SimpleNamespace(flash_attn=None)
    _set_flash_attn(cparams, enable)
    assert cparams.flash_attn is enable


@pytest.mark.parametrize("enable,expected", [(True, 1), (False, 0)])
def test_set_flash_attn_enum_build(enable, expected):
    from types import SimpleNamespace

    cparams = SimpleNamespace(flash_attn_type=None)  # no bool field
    llama_cpp = SimpleNamespace(LLAMA_FLASH_ATTN_TYPE_ENABLED=1,
                         LLAMA_FLASH_ATTN_TYPE_DISABLED=0)
    _set_flash_attn(cparams, enable, llama_cpp=llama_cpp)
    assert cparams.flash_attn_type == expected


@pytest.mark.parametrize("enable,expected", [(True, 1), (False, 0)])
def test_set_flash_attn_enum_build_without_constants(enable, expected):
    from types import SimpleNamespace

    cparams = SimpleNamespace(flash_attn_type=None)
    _set_flash_attn(cparams, enable, llama_cpp=SimpleNamespace())  # constants absent
    assert cparams.flash_attn_type == expected


def test_set_flash_attn_disable_on_unsupported_build_is_noop():
    from types import SimpleNamespace

    cparams = SimpleNamespace()  # neither field present
    _set_flash_attn(cparams, False)  # must not raise


def test_set_flash_attn_enable_on_unsupported_build_raises():
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="flash-attention"):
        _set_flash_attn(SimpleNamespace(), True)


def _set_offload(cparams, enable):
    """Drive Backend._set_kv_cache_on_gpu with fakes, bypassing llama_cpp import."""
    from types import SimpleNamespace
    from llama_chat._backend import Backend

    Backend._set_kv_cache_on_gpu(SimpleNamespace(_llama_cpp=None), cparams, enable)


def test_offload_kqv_to_gpu_defaults_off():
    # KV cache stays in host RAM unless explicitly offloaded to the GPU.
    assert Config().kv_cache_on_gpu is False


@pytest.mark.parametrize("enable", [True, False])
def test_set_offload_kqv_bool_build(enable):
    from types import SimpleNamespace

    cparams = SimpleNamespace(offload_kqv=None)
    _set_offload(cparams, enable)
    assert cparams.offload_kqv is enable


def test_set_offload_kqv_enable_on_unsupported_build_is_noop():
    from types import SimpleNamespace

    # GPU placement is llama.cpp's own default, so a missing param is harmless.
    _set_offload(SimpleNamespace(), True)  # must not raise


def test_set_offload_kqv_disable_on_unsupported_build_raises():
    from types import SimpleNamespace

    with pytest.raises(RuntimeError, match="KV-offload"):
        _set_offload(SimpleNamespace(), False)


@pytest.mark.parametrize("field,value", [
    ("max_batch_tokens", 0),
    ("max_reply_tokens", 0),
    ("min_reply_tokens", -1),
    ("context_size", 0),
    ("evict_above_fraction", 0.0),
    ("max_history_messages", 0),
    ("max_history_messages", -1),
    ("repetition_penalty", 0.0),
    ("seed", -1),
    ("seed", 0xFFFFFFFF),
])
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


# ----- stop-string scanning ------------------------------------------------
def test_stop_hit_finds_earliest_match():
    from llama_chat.context import _stop_hit

    assert _stop_hit("abcSTOPdef", ["STOP"]) == 3
    assert _stop_hit("aXbYc", ["Y", "X"]) == 1          # earliest of several
    assert _stop_hit("abc", ["STOP"]) is None
    assert _stop_hit("abc", ["", "b"]) == 1             # empty strings ignored


def test_stop_hit_respects_search_offset():
    from llama_chat.context import _stop_hit

    # A match strictly before the offset is skipped; one spanning or after the
    # offset is found, which is the contract the emit cursor relies on.
    assert _stop_hit("STOPxxSTOP", ["STOP"], search_from=1) == 6
    assert _stop_hit("xxSTOP", ["STOP"], search_from=2) == 2
    assert _stop_hit("xxSTOP", ["STOP"], search_from=3) is None


# ----- public surface for callers that schedule around the lock ----------
# The assistant that drives this wrapper needs three things the lock already
# knows, and used to reach into privates for all three.
def test_can_shift_reports_the_contexts_capability():
    assert ChatWrapper(_config(), context=FakeContext(can_shift=True)).can_shift is True
    assert ChatWrapper(_config(), context=FakeContext(can_shift=False)).can_shift is False


def test_the_eviction_limits_are_readable_beside_the_total():
    """total_tokens alone has no denominator, and kwargs hand back no Config."""
    w = ChatWrapper(context=FakeContext(), fragments=DEFAULT_FRAGMENTS,
                    context_size=800, evict_above_fraction=0.5,
                    max_history_messages=7)
    assert w.context_size == 800
    # Multiplied out, so a caller does not redo the arithmetic to know the trigger.
    assert w.evict_above_tokens == 400
    assert w.max_history_messages == 7


def test_no_message_limit_reads_back_as_none():
    w = ChatWrapper(context=FakeContext(), fragments=DEFAULT_FRAGMENTS)
    assert w.max_history_messages is None


def test_the_limits_cannot_be_assigned_through_the_wrapper():
    """Assignment would not reach the context, already built with the old size."""
    w = ChatWrapper(context=FakeContext(), fragments=DEFAULT_FRAGMENTS,
                    context_size=800)
    for name in ("context_size", "evict_above_tokens", "max_history_messages"):
        with pytest.raises(AttributeError):
            setattr(w, name, 99)
    assert w.context_size == 800


def test_wait_idle_is_true_when_nothing_holds_the_wrapper():
    w = ChatWrapper(_config(context_size=200), context=FakeContext())
    assert w.busy is False
    assert w.wait_idle(timeout_s=0.1) is True


def test_wait_idle_blocks_until_a_generation_releases_the_lock():
    fake = FakeContext(gen_tokens=3, gen_text="abc")
    w = ChatWrapper(_config(context_size=200), context=fake)
    w.begin("sys")

    entered, release = threading.Event(), threading.Event()

    def generate():
        stream = w.reply_without_tools_stream("hi")
        next(stream)                 # takes the lock inside the generator body
        entered.set()
        release.wait(5)
        for _ in stream:
            pass

    t = threading.Thread(target=generate, daemon=True)
    t.start()
    entered.wait(5)

    assert w.busy is True
    assert w.wait_idle(timeout_s=0.05) is False   # the turn still holds it
    release.set()
    t.join(5)
    assert w.wait_idle(timeout_s=1.0) is True
    assert w.busy is False


# ----- template customisation -------------------------------------------
def test_a_formatter_factory_customises_how_generation_opens():
    class NoThink(TemplateFormatter):
        def assistant_open(self) -> str:
            return super().assistant_open() + "<empty/>"

    fake = FakeContext()
    w = ChatWrapper(_config(context_size=200), context=fake, formatter=NoThink)
    w.begin("sys")
    w.reply_without_tools("hi")
    # The suffix is tokenized into the generation prompt ...
    assert any("<empty/>" in text for text, _special in fake.tokenize_calls)
    # ... and only there: the history-wrapping fragments are untouched, so the
    # cache layout and the template validation see exactly what they did before.
    plain = ChatWrapper(_config(context_size=200), context=FakeContext())
    assert "<empty/>" not in plain._formatter.fragment("assistant", "yo")


def test_a_formatter_that_loses_the_models_turn_tags_is_refused():
    """A generation prompt is an addition to the turn tag, never a replacement.

    Left unchecked, the first encodes every reply outside the turn it belongs
    to, the second leaves that turn unterminated, and the last opens a second
    turn inside the first. All three are shapes no template produces, and all
    three surface as a worse reply rather than as an error.
    """
    class Replaces(TemplateFormatter):
        def assistant_open(self) -> str:
            return "<oops>"

    class Drops(TemplateFormatter):
        def assistant_close(self) -> str:
            return "\n"

    class Reopens(TemplateFormatter):
        def resume_open(self) -> str:
            return DEFAULT_FRAGMENTS.assistant_prefix + "<empty/>"

    for broken, complaint in ((Replaces, "assistant_open"),
                              (Drops, "assistant_close"),
                              (Reopens, "resume_open")):
        with pytest.raises(ValueError, match=complaint):
            ChatWrapper(_config(), context=FakeContext(), formatter=broken)


def test_a_cache_context_missing_an_edit_is_refused_where_it_is_adopted():
    """Eviction is the last thing a conversation reaches, and the first to fail.

    A context that answers everything else runs for a whole conversation before
    the threshold is crossed, so the check belongs where the context is taken.
    """
    class WithoutEviction:
        """Everything the fake answers, minus the one edit eviction performs."""

        def __init__(self) -> None:
            self._real = FakeContext()

        def __getattr__(self, name):
            if name == "apply_eviction":
                raise AttributeError(name)
            return getattr(self._real, name)

    with pytest.raises(TypeError, match="apply_eviction"):
        ChatWrapper(_config(), context=WithoutEviction())


# ----- a conversation has to be begun ------------------------------------
def test_generating_before_begin_is_refused():
    """Only begin prefills the sequence opener and the system prompt.

    Generating without one asks the model to continue a conversation that has
    no beginning, and nothing about the reply says so.
    """
    fake = FakeContext()
    w = ChatWrapper(_config(context_size=200), context=fake)

    with pytest.raises(ValueError, match="no conversation to generate into"):
        w.reply_without_tools("hi")
    with pytest.raises(ValueError, match="no conversation to inject into"):
        w.inject("context")
    assert fake.cache == []                 # and nothing was prefilled first

    w.begin("sys")
    assert w.reply_without_tools("hi").text == "reply"


# ----- configurations that could never work ------------------------------
def test_min_reply_tokens_at_the_context_size_is_refused():
    """Every generate would raise, and the first one is too late to find out."""
    with pytest.raises(ValueError, match="min_reply_tokens"):
        _config(context_size=64, min_reply_tokens=64)
    _config(context_size=64, min_reply_tokens=63)        # one token to reply in


def test_an_eviction_threshold_that_rounds_to_nothing_is_refused():
    with pytest.raises(ValueError, match="rounds down to no tokens"):
        _config(context_size=8, evict_above_fraction=0.1)


# ----- the stream and the reply it produced ------------------------------
def test_a_stream_hands_back_the_reply_it_produced():
    """A plain for loop leaves the summary behind, with no StopIteration to catch."""
    fake = FakeContext(gen_tokens=3, gen_text="abc")
    w = ChatWrapper(_config(context_size=200), context=fake)
    w.begin("sys")

    stream = w.reply_without_tools_stream("hi")
    assert stream.reply is None                  # lazy: nothing has run yet
    assert "".join(stream) == "abc"

    assert stream.reply.n_generated_tokens == 3
    assert stream.reply.stop_reason == "end_of_turn"
    _assert_consistent(w, fake)


def test_a_second_pass_over_a_finished_stream_keeps_the_reply():
    """A spent generator returns ``None``, which must not replace the summary."""
    fake = FakeContext(gen_tokens=3, gen_text="abc")
    w = ChatWrapper(_config(context_size=200), context=fake)
    w.begin("sys")

    stream = w.reply_without_tools_stream("hi")
    list(stream)
    produced = stream.reply

    assert list(stream) == []
    assert stream.reply is produced


def test_a_stream_closed_early_ends_the_turn_and_reports_no_reply():
    """There is no summary of a reply nobody read to the end.

    The turn is closed all the same, which is what the next request needs.
    """
    fake = FakeContext(gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(_config(context_size=200), context=fake)
    w.begin("sys")

    with w.reply_without_tools_stream("hi") as stream:
        assert next(stream) == "a"               # read one delta, then leave

    assert stream.reply is None
    assert not w.reply_open
    _assert_consistent(w, fake)
    assert w.reply_without_tools("weiter").stop_reason == "end_of_turn"
