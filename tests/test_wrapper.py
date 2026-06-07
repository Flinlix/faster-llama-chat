"""End-to-end wrapper tests using the model-free FakeContext.

These verify the two properties that matter most: (1) the wrapper only ever
prefills *new* tokens (never re-prefills survivors), and (2) the physical cache
stays exactly equal to the concatenation of the message table after every
action, including across evictions.
"""

import pytest

from llama_chat.config import Config
from llama_chat.template import TemplateFormatter
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import BOS, FakeContext


def _cfg(**kw):
    return Config(**kw)


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
    w = ChatWrapper(_cfg(n_ctx=200, threshold_pct=1.0), context=fake)
    w.begin("sys", [("user", "hi"), ("assistant", "yo")])

    roles = [s["role"] for s in w.snapshot()]
    assert roles == ["system", "user", "assistant"]
    # One prefill call for the whole conversation == cheapest prefill.
    assert len(fake.prefill_calls) == 1
    _assert_consistent(w, fake)


def test_begin_drops_oldest_history_beyond_threshold():
    fake = FakeContext()
    # system fragment "<|turn>system\nsys<turn|>\n" + BOS is well under budget;
    # pick a tiny threshold so only the newest message fits.
    w = ChatWrapper(_cfg(n_ctx=1000, threshold_pct=0.07), context=fake)
    w.begin("s", [("user", "old message that is long"), ("assistant", "x")])

    roles = [s["role"] for s in w.snapshot()]
    # Oldest user message dropped; only system + newest assistant kept.
    assert roles == ["system", "assistant"]
    _assert_consistent(w, fake)


def test_begin_has_bos_only_at_start():
    fake = FakeContext()
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s", [("user", "u")])
    # Exactly one BOS sentinel, at position 0.
    assert fake.cache[0] == BOS
    assert fake.cache.count(BOS) == 1


# ----- inject ------------------------------------------------------------
def test_inject_appends_without_generating():
    fake = FakeContext()
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s")
    before = w.total_tokens
    n_evicted = w.inject("some context", role="user")
    assert n_evicted == 0
    assert w.snapshot()[-1]["role"] == "user"
    assert w.total_tokens > before
    # No generation happened.
    assert all(not want_logits for *_, want_logits in fake.prefill_calls)
    _assert_consistent(w, fake)


def test_inject_evicts_oldest_to_fit():
    fake = FakeContext()
    w = ChatWrapper(_cfg(n_ctx=1000, threshold_pct=0.085), context=fake)
    w.begin("s")
    w.inject("first injected chunk", role="user")
    n_evicted = w.inject("second injected chunk", role="user")
    assert n_evicted >= 1
    roles = [s["role"] for s in w.snapshot()]
    assert roles[0] == "system"  # system never evicted
    _assert_consistent(w, fake)


# ----- request -----------------------------------------------------------
def test_request_prefills_only_new_tokens_and_reuses_context():
    fake = FakeContext(gen_len=4, gen_text="abcd")
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0, max_tokens=50), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "hello")])
    prefilled_after_begin = fake.total_prefilled()

    turn = w.request("what's up?")

    # The only new prefill is the request prompt (user fragment + assistant-open)
    # plus the turn-closing tag — never any survivor.
    close_len = len(fake.tokenize(w._fmt.assistant_close()))
    new_prefill = fake.total_prefilled() - prefilled_after_begin
    assert new_prefill == turn.n_prefilled + close_len
    assert turn.n_generated == 4
    assert turn.stop_reason == "eog"
    # Assistant turn recorded for reuse next turn.
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)


def test_second_request_does_not_reprefill_history():
    fake = FakeContext(gen_len=3)
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s")
    w.request("first")
    mark = fake.total_prefilled()
    turn = w.request("second")
    # Second turn only prefills its own prompt + closer; the FakeContext's
    # append-only assertion already guarantees no survivor was re-prefilled.
    close_len = len(fake.tokenize(w._fmt.assistant_close()))
    assert fake.total_prefilled() - mark == turn.n_prefilled + close_len
    _assert_consistent(w, fake)


# ----- streaming ---------------------------------------------------------
def test_stream_yields_incremental_deltas():
    fake = FakeContext(gen_len=4, gen_text="abcd")
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s")
    deltas = list(w.stream("hi"))
    assert "".join(deltas) == "abcd"
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)


def test_request_drains_stream_and_returns_turn():
    fake = FakeContext(gen_len=3, gen_text="xyz")
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s")
    turn = w.request("hi")
    assert turn.text == "xyz"
    assert turn.n_generated == 3
    _assert_consistent(w, fake)


def test_bargein_keeps_cache_consistent():
    # Consume only the first delta, then abandon the stream (close the generator).
    fake = FakeContext(gen_len=8, gen_text="abcdefgh")
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s")

    gen = w.stream("hi")
    first = next(gen)
    assert first == "a"
    gen.close()  # barge-in: stop early

    # The assistant turn was still recorded, and the cache equals the table —
    # i.e. exactly the tokens that reached the cache were recorded, no more.
    assert w.snapshot()[-1]["role"] == "assistant"
    _assert_consistent(w, fake)

    # A following turn proceeds normally on the consistent cache.
    turn = w.request("again")
    assert turn.stop_reason == "eog"
    _assert_consistent(w, fake)


def test_request_caps_generation_to_context_size():
    fake = FakeContext(gen_len=100)  # would generate 100 if uncapped
    w = ChatWrapper(_cfg(n_ctx=110, threshold_pct=1.0, max_tokens=100), context=fake)
    w.begin("s")
    turn = w.request("hello")
    assert w.total_tokens <= 110
    assert turn.n_generated < 100
    _assert_consistent(w, fake)


def test_request_evicts_then_stays_within_context():
    fake = FakeContext(gen_len=2)
    w = ChatWrapper(_cfg(n_ctx=400, threshold_pct=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        w.request(f"message number {i} with some length")
        roles = [s["role"] for s in w.snapshot()]
        assert roles[0] == "system"
        assert w.total_tokens <= w._cfg.n_ctx
        _assert_consistent(w, fake)


def test_eviction_shift_mode_does_not_rebuild():
    # Default cache supports shifting: eviction uses apply_eviction, never rebuild.
    fake = FakeContext(gen_len=2)  # can_shift=True
    w = ChatWrapper(_cfg(n_ctx=400, threshold_pct=0.2), context=fake)
    w.begin("s")
    rebuilds_after_begin = fake.rebuilds  # begin() resets once
    for i in range(8):
        w.request(f"message number {i} with some length")
    assert len(fake.evictions) > 0          # eviction did happen
    assert fake.rebuilds == rebuilds_after_begin  # but never via rebuild
    _assert_consistent(w, fake)


def test_eviction_rebuild_mode_when_cache_cannot_shift():
    # Cache cannot shift: eviction drops oldest from bookkeeping and re-prefills.
    fake = FakeContext(gen_len=2, can_shift=False)
    w = ChatWrapper(_cfg(n_ctx=400, threshold_pct=0.2), context=fake)
    w.begin("s")
    rebuilds_after_begin = fake.rebuilds
    for i in range(8):
        w.request(f"message number {i} with some length")
        roles = [s["role"] for s in w.snapshot()]
        assert roles[0] == "system"             # system survives
        assert w.total_tokens <= w._cfg.n_ctx   # never overflows
        _assert_consistent(w, fake)             # cache == table after rebuild
    assert len(fake.evictions) == 0             # shift path never used
    assert fake.rebuilds > rebuilds_after_begin  # rebuild path was exercised


def test_oversize_reject():
    fake = FakeContext()
    w = ChatWrapper(_cfg(n_ctx=40, threshold_pct=0.5, oversize_policy="reject"), context=fake)
    w.begin("s")
    with pytest.raises(ValueError):
        w.inject("x" * 100)


def test_oversize_truncate():
    fake = FakeContext()
    w = ChatWrapper(_cfg(n_ctx=200, threshold_pct=0.3, oversize_policy="truncate"), context=fake)
    w.begin("s")
    w.inject("y" * 500)
    assert w.total_tokens <= w._cfg.threshold_tokens
    _assert_consistent(w, fake)


# ----- template formatting -----------------------------------------------
def test_template_defaults_to_gemma4():
    fmt = TemplateFormatter(_cfg())
    assert fmt.fragment("user", "hi") == "<|turn>user\nhi<turn|>\n"
    assert fmt.assistant_open() == "<|turn>model\n"
    assert fmt.assistant_close() == "<turn|>\n"


def test_template_formatter_combines_fragments():
    fmt = TemplateFormatter(
        _cfg(user_prefix="<u>", user_suffix="</u>",
             assistant_prefix="<a>", assistant_suffix="</a>")
    )
    assert fmt.fragment("user", "hi") == "<u>hi</u>"
    assert fmt.fragment("assistant", "yo") == "<a>yo</a>"
    assert fmt.assistant_open() == "<a>"
    assert fmt.assistant_close() == "</a>"
    with pytest.raises(ValueError):
        fmt.fragment("unknown-role", "x")
