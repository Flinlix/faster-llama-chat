"""End-to-end wrapper tests using the model-free FakeContext.

These verify the two properties that matter most: (1) the wrapper only ever
prefills *new* tokens (never re-prefills survivors), and (2) the physical cache
stays exactly equal to the concatenation of the message table after every
action, including across evictions.
"""

import pytest

import dataclasses

from llama_chat.config import Config
from llama_chat.context import _collect_special_token_texts
from llama_chat.template import TemplateFormatter
from llama_chat.wrapper import ChatWrapper, ContextOverflowError
from tests.fake_context import BOS, GEMMA_FRAGMENTS, FakeContext


def _cfg(**kw):
    return Config(**kw)


def _frags(**kw):
    """Gemma-default fragments with the given fields overridden."""
    return dataclasses.replace(GEMMA_FRAGMENTS, **kw)


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
    # system fragment "<|turn>user\ns<turn|>\n" + BOS is well under budget;
    # pick a tiny threshold so only the newest message fits.
    w = ChatWrapper(_cfg(n_ctx=1000, threshold_pct=0.08), context=fake)
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


def test_begin_warms_up_once():
    fake = FakeContext()
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s", [("user", "u")])
    w.begin("s", [("user", "u")])
    # Warmed on the first begin only; later begins skip the redundant decode.
    assert fake.warmups == 1
    _assert_consistent(w, fake)


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
    w = ChatWrapper(_cfg(n_ctx=1000, threshold_pct=0.1), context=fake)
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
    # plus the turn-closing tag - never any survivor.
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

    # The assistant turn was still recorded, and the cache equals the table -
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


def test_request_rests_under_threshold_after_turn():
    # The reply is appended first, then older turns are trimmed, so the cache
    # rests at or below threshold once the turn returns.
    fake = FakeContext(gen_len=2)
    w = ChatWrapper(_cfg(n_ctx=400, threshold_pct=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        w.request(f"message number {i} with some length")
        assert w.total_tokens <= w._cfg.threshold_tokens  # rests under threshold
        assert w.snapshot()[0]["role"] == "system"        # system survives
        assert w.snapshot()[-1]["role"] == "assistant"    # newest turn kept
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
    w = ChatWrapper(_cfg(n_ctx=80, threshold_pct=0.5, oversize_policy="reject"), context=fake)
    w.begin("s")
    with pytest.raises(ValueError):
        w.inject("x" * 100)


def test_oversize_reject_leaves_history_intact():
    # Rejecting an unfittable message must not evict the history first.
    fake = FakeContext(gen_len=2)
    w = ChatWrapper(_cfg(n_ctx=400, threshold_pct=0.5, oversize_policy="reject"), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "yo")])
    before = w.snapshot()
    with pytest.raises(ValueError):
        w.inject("x" * 500)
    assert w.snapshot() == before
    _assert_consistent(w, fake)


def test_oversize_truncate():
    fake = FakeContext()
    w = ChatWrapper(_cfg(n_ctx=200, threshold_pct=0.3, oversize_policy="truncate"), context=fake)
    w.begin("s")
    w.inject("y" * 500)
    assert w.total_tokens <= w._cfg.threshold_tokens
    # The clipped message still ends with its turn-terminator tag, so the
    # cache never holds a malformed turn.
    suffix = fake.tokenize(GEMMA_FRAGMENTS.user_suffix)
    assert fake.cache[-len(suffix):] == suffix
    _assert_consistent(w, fake)


def test_oversize_truncate_rejects_when_no_room_for_suffix():
    fake = FakeContext()
    # Threshold barely above the system prompt: the remaining budget is smaller
    # than the turn-terminator tag, so truncation cannot produce a valid turn.
    w = ChatWrapper(_cfg(n_ctx=200, threshold_pct=0.15, oversize_policy="truncate"), context=fake)
    w.begin("s")
    with pytest.raises(ValueError):
        w.inject("y" * 100)
    _assert_consistent(w, fake)


# ----- construction / lifecycle -------------------------------------------
def test_config_and_kwargs_are_mutually_exclusive():
    with pytest.raises(TypeError, match="not both"):
        ChatWrapper(_cfg(), context=FakeContext(), n_ctx=8192)


def test_begin_rejects_system_prompt_over_threshold():
    fake = FakeContext()
    # threshold = 50 tokens; a 100-char system prompt can never fit.
    w = ChatWrapper(_cfg(n_ctx=1000, threshold_pct=0.05), context=fake)
    with pytest.raises(ValueError, match="system prompt"):
        w.begin("x" * 100)
    # Nothing was prefilled by the refused begin; a fitting one then works.
    assert fake.cache == []
    w.begin("s")
    assert w.snapshot()[0]["role"] == "system"


def test_failed_begin_keeps_previous_conversation():
    fake = FakeContext(gen_len=2)
    w = ChatWrapper(_cfg(n_ctx=1000, threshold_pct=0.05), context=fake)
    w.begin("s", [("user", "hi")])
    before = w.snapshot()
    with pytest.raises(ValueError):
        w.begin("x" * 100)  # refused before mutating
    assert w.snapshot() == before
    _assert_consistent(w, fake)


def test_close_is_idempotent_and_blocks_actions():
    fake = FakeContext(gen_len=2)
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake)
    w.begin("s")
    w.close()
    w.close()  # second close is a no-op
    with pytest.raises(RuntimeError, match="closed"):
        w.begin("s")
    with pytest.raises(RuntimeError, match="closed"):
        w.inject("ctx")
    with pytest.raises(RuntimeError, match="closed"):
        w.request("hi")


def test_context_manager_closes_on_exit():
    fake = FakeContext(gen_len=2)
    with ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0), context=fake) as w:
        w.begin("s")
        w.request("hi")
    with pytest.raises(RuntimeError, match="closed"):
        w.request("again")


# ----- template formatting -----------------------------------------------
def test_template_formatter_renders_gemma_tags():
    fmt = TemplateFormatter(_frags())
    assert fmt.fragment("user", "hi") == "<|turn>user\nhi<turn|>\n"
    assert fmt.assistant_open() == "<|turn>model\n"
    assert fmt.assistant_close() == "<turn|>\n"


def test_template_formatter_combines_fragments():
    fmt = TemplateFormatter(
        _frags(user_prefix="<u>", user_suffix="</u>",
               assistant_prefix="<a>", assistant_suffix="</a>")
    )
    assert fmt.fragment("user", "hi") == "<u>hi</u>"
    assert fmt.fragment("assistant", "yo") == "<a>yo</a>"
    assert fmt.assistant_open() == "<a>"
    assert fmt.assistant_close() == "</a>"
    with pytest.raises(ValueError):
        fmt.fragment("unknown-role", "x")


def test_trim_content_strips_by_default():
    fmt = TemplateFormatter(_frags())
    assert fmt.fragment("user", "  hi\n") == "<|turn>user\nhi<turn|>\n"


def test_trim_content_disabled_keeps_whitespace():
    fmt = TemplateFormatter(_frags(trim_content=False))
    assert fmt.fragment("user", "  hi\n") == "<|turn>user\n  hi\n<turn|>\n"


# ----- special-token sanitization ----------------------------------------
def test_collect_special_token_texts_filters_whitespace_and_non_round_trip():
    pieces = {1: "<end>", 2: " ", 3: ""}  # whitespace + empty must be dropped
    out = _collect_special_token_texts(
        5,
        lambda t: t in pieces,           # ids 1,2,3 are special; 0,4 are not
        lambda t: pieces[t],
        lambda s: s == "<end>",          # only "<end>" round-trips to a special
    )
    assert out == ["<end>"]


def test_collect_special_token_texts_sorts_longest_first():
    pieces = {0: "<a>", 1: "<longer>"}
    out = _collect_special_token_texts(
        2, lambda t: True, lambda t: pieces[t], lambda s: True
    )
    assert out == ["<longer>", "<a>"]


def test_fragment_strips_special_tokens_from_content_only():
    fmt = TemplateFormatter(_frags(), special_tokens=["<turn|>", "<|turn>"])
    # The content tag is removed; the wrapping prefix/suffix tags (which contain
    # the same strings) survive intact.
    assert fmt.fragment("user", "hi <turn|> there") == "<|turn>user\nhi  there<turn|>\n"


def test_fragment_without_special_tokens_is_unchanged():
    fmt = TemplateFormatter(_frags())
    assert fmt.fragment("user", "hi <turn|> there") == "<|turn>user\nhi <turn|> there<turn|>\n"


def test_fragment_strips_longest_token_first():
    fmt = TemplateFormatter(_frags(), special_tokens=["of", "of_turn"])
    # Stripping "of" first would leave "_turn"; longest-first removes "of_turn".
    assert fmt.fragment("user", "x of_turn y") == "<|turn>user\nx  y<turn|>\n"


def test_injected_special_token_does_not_reach_cache_as_tag():
    fake = FakeContext(special_tokens=("<turn|>",))
    w = ChatWrapper(_cfg(n_ctx=500, threshold_pct=1.0),
                    context=fake, fragments=GEMMA_FRAGMENTS)
    w.begin("sys")
    w.inject("hello <turn|> world", role="user")

    # Reconstruct the injected fragment from its tokens (FakeContext is 1:1 chars).
    rendered = "".join(chr(t) for t in w._table.messages[-1].token_ids)
    # Content tag stripped; the legitimate suffix tag is preserved.
    assert rendered == "<|turn>user\nhello  world<turn|>\n"
    _assert_consistent(w, fake)


def test_default_fake_context_reports_no_special_tokens():
    assert FakeContext().special_token_texts() == []


# ----- min-answer headroom guard -----------------------------------------
def test_request_refuses_when_too_little_room_to_answer():
    fake = FakeContext(gen_len=2)
    # Prompt fits n_ctx, but leaves fewer than min_answer_tokens for the reply.
    w = ChatWrapper(
        _cfg(n_ctx=120, threshold_pct=1.0, min_answer_tokens=60), context=fake
    )
    w.begin("s")
    with pytest.raises(ContextOverflowError):
        w.request("hello")
    # The refused request never touched the cache.
    assert w.snapshot()[-1]["role"] == "system"
    _assert_consistent(w, fake)


def test_request_proceeds_when_headroom_sufficient():
    fake = FakeContext(gen_len=2)
    w = ChatWrapper(
        _cfg(n_ctx=120, threshold_pct=1.0, min_answer_tokens=10), context=fake
    )
    w.begin("s")
    turn = w.request("hello")
    assert turn.n_generated == 2
    _assert_consistent(w, fake)


# ----- model-template fragment validation --------------------------------
class _TemplatingFake(FakeContext):
    """FakeContext that also renders a model chat template for the fragment check."""

    def __init__(self, render, fragments=GEMMA_FRAGMENTS):
        super().__init__(fragments=fragments)
        self._render = render

    def render_with_model_template(self, messages):
        return self._render(messages)


def test_fragment_check_passes_when_fragments_match_model():
    fmt = TemplateFormatter(_frags())
    render = lambda msgs: "".join(fmt.fragment(m["role"], m["content"]) for m in msgs)
    # Constructs without error: model render == fragment render.
    ChatWrapper(_cfg(), context=_TemplatingFake(render))


def test_fragment_check_rejects_mismatched_fragments():
    render = lambda msgs: "".join(
        f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n" for m in msgs
    )
    with pytest.raises(ValueError, match="do not match the model"):
        ChatWrapper(_cfg(), context=_TemplatingFake(render))


def test_fragment_check_catches_trim_mismatch():
    # Model trims content; the wrapper's fragments say not to -> the whitespace
    # probe diverges and construction must fail.
    trimming = TemplateFormatter(_frags())
    render = lambda msgs: "".join(trimming.fragment(m["role"], m["content"]) for m in msgs)
    with pytest.raises(ValueError, match="do not match the model"):
        ChatWrapper(context=_TemplatingFake(render, fragments=_frags(trim_content=False)))


# ----- KV-cache / flash-attn config --------------------------------------
def test_kv_cache_type_requires_flash_attn():
    with pytest.raises(ValueError, match="flash_attn"):
        Config(kv_cache_type="q8_0")


def test_kv_cache_type_rejects_unknown_name():
    with pytest.raises(ValueError, match="kv_cache_type"):
        Config(kv_cache_type="bogus", flash_attn=True)


def test_kv_cache_type_accepts_known_type_with_flash_attn():
    cfg = Config(kv_cache_type="q8_0", flash_attn=True)
    assert cfg.kv_cache_type == "q8_0"


@pytest.mark.parametrize("field,value", [
    ("n_batch", 0),
    ("max_tokens", 0),
    ("min_answer_tokens", -1),
    ("n_ctx", 0),
    ("threshold_pct", 0.0),
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
    # offset is found - the contract the emit cursor relies on.
    assert _stop_hit("STOPxxSTOP", ["STOP"], search_from=1) == 6
    assert _stop_hit("xxSTOP", ["STOP"], search_from=2) == 2
    assert _stop_hit("xxSTOP", ["STOP"], search_from=3) is None
