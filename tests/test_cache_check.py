"""The check every test on a model-free cache ends with: it fails on a cache that does not hold what the wrapper holds it to be, or does not read as the template renders the conversation."""

import pytest

from llama_chat.tools import ToolCall
from tests.internals import held_tokens, leave_cache_behind, overwrite_held_tokens
from tests.wrappers import (assert_cache_reads_as_rendered, begun_wrapper, reasoning_wrapper,
                            tool_wrapper)

_MARK = ord("#")        # a token that spells a character no render here holds


def _plain():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi").finish()
    return w, fake


def _reasoning():
    w, fake = reasoning_wrapper()
    w.reply("hi").finish()
    return w, fake


def _with_a_kept_turn():
    w, fake = tool_wrapper()
    w.begin("persona")
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", calls="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call, "12:00")
    w.reply(calls="yield").finish()
    w.reply("und?").finish()
    return w, fake


def _one_token_off(tokens):
    """Every list one token away from ``tokens``: one inserted, removed or replaced at each position."""
    for at in range(len(tokens) + 1):
        yield tokens[:at] + [_MARK] + tokens[at:]
        if at < len(tokens):
            yield tokens[:at] + tokens[at + 1:]
            yield tokens[:at] + [_MARK] + tokens[at + 1:]


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_kept_turn],
                         ids=["plain", "reasoning", "kept-turn"])
def test_a_cache_one_token_off_what_the_wrapper_holds_is_caught(
        conversation, _the_cache_reads_as_the_template_renders_the_conversation):
    w, fake = conversation()
    assert w in _the_cache_reads_as_the_template_renders_the_conversation
    sound = list(fake.cache)
    for broken in _one_token_off(sound):
        fake.cache = broken
        with pytest.raises(AssertionError, match="no longer holds what the wrapper holds"):
            assert_cache_reads_as_rendered(w, fake)
    fake.cache = sound


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_kept_turn],
                         ids=["plain", "reasoning", "kept-turn"])
def test_a_cache_held_one_token_off_the_render_is_caught_unless_it_reads_the_same_or_stops_short(
        conversation):
    w, fake = conversation()
    sound = list(fake.cache)
    for broken in _one_token_off(sound):
        fake.cache = broken
        overwrite_held_tokens(w, broken)
        read = fake.text_of(broken)
        if read != fake.text_of(sound) and not fake.text_of(sound).startswith(read):
            with pytest.raises(AssertionError, match="does not read as the template renders"):
                assert_cache_reads_as_rendered(w, fake)
    fake.cache = sound
    overwrite_held_tokens(w, sound)


def test_a_cache_left_behind_its_tokens_with_nothing_failed_is_caught_not_brought_back():
    w, fake = _plain()
    held = held_tokens(w)
    leave_cache_behind(w, held[:-1])
    with pytest.raises(AssertionError, match="no longer holds what the wrapper holds"):
        assert_cache_reads_as_rendered(w, fake)
    leave_cache_behind(w, held)


def test_a_cache_a_failed_decode_left_behind_is_brought_back_as_the_next_action_would():
    w, fake = _plain()
    prefill = fake.prefill

    def fails_halfway(token_ids, start_pos, want_logits):
        prefill(token_ids[:len(token_ids) // 2], start_pos, want_logits)
        fake.prefill = prefill
        raise RuntimeError("the engine refused the batch")

    fake.prefill = fails_halfway
    with pytest.raises(RuntimeError, match="refused the batch"):
        w.inject("lost on the way")
    assert fake.cache != held_tokens(w)
    assert_cache_reads_as_rendered(w, fake)
