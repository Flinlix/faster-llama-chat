"""The check every test on a model-free cache ends with: it fails on a cache that does not hold what the wrapper holds it to be, or does not read as the template renders the conversation."""

import pytest

from llama_chat.tools import ToolCall
from tests.chat_messages import tool_turn
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
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call, "12:00")
    w.reply(tool_call_policy="yield").finish()
    w.reply("und?").finish()
    return w, fake


def _owed_from_a_history():
    w, fake = tool_wrapper()
    w.begin("persona", [("user", "wie spät?"), *tool_turn(ToolCall("clock", {}))])
    assert w.pending_calls
    return w, fake


def _one_token_off(tokens):
    """Every list one token away from ``tokens``: one inserted, removed or replaced at each position."""
    for at in range(len(tokens) + 1):
        yield tokens[:at] + [_MARK] + tokens[at:]
        if at < len(tokens):
            yield tokens[:at] + tokens[at + 1:]
            yield tokens[:at] + [_MARK] + tokens[at + 1:]


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_kept_turn,
                                          _owed_from_a_history],
                         ids=["plain", "reasoning", "kept-turn", "owed-from-a-history"])
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


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_kept_turn,
                                          _owed_from_a_history],
                         ids=["plain", "reasoning", "kept-turn", "owed-from-a-history"])
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


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_kept_turn,
                                          _owed_from_a_history],
                         ids=["plain", "reasoning", "kept-turn", "owed-from-a-history"])
def test_a_cache_that_reads_the_same_with_its_tags_held_as_their_letters_is_caught(conversation):
    w, fake = conversation()
    sound = list(fake.cache)
    spelled = [letter for token in sound for letter in (
        [ord(c) for c in fake.token_text(token)] if fake.is_special(token) else [token])]
    assert fake.text_of(spelled) == fake.text_of(sound)
    fake.cache = spelled
    overwrite_held_tokens(w, spelled)
    with pytest.raises(AssertionError, match="does not read as the template renders"):
        assert_cache_reads_as_rendered(w, fake)
    fake.cache = sound
    overwrite_held_tokens(w, sound)


def test_a_cache_that_reads_the_same_with_a_typed_tag_held_as_the_models_is_caught():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi <turn]> there").finish()
    sound = list(fake.cache)
    forged = fake.tokenize(fake.text_of(sound), add_special=True, parse_special=True)
    assert forged != sound and fake.text_of(forged) == fake.text_of(sound)
    fake.cache = forged
    overwrite_held_tokens(w, forged)
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


def test_a_cache_holding_a_token_that_spells_the_renders_text_but_is_not_its_token_is_caught():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi there").finish()
    sound = list(fake.cache)
    fake._tokens_spell_text = False             # what a generation hands out from here on
    at = sound.index(ord("h"))
    broken = sound[:at] + [fake._generated("h")] + sound[at + 1:]
    assert fake.text_of(broken) == fake.text_of(sound)
    fake.cache = broken
    overwrite_held_tokens(w, broken)
    with pytest.raises(AssertionError, match="does not read as the template renders"):
        assert_cache_reads_as_rendered(w, fake)
    fake.cache = sound
    overwrite_held_tokens(w, sound)


def test_a_cache_missing_an_exchange_after_a_kept_turn_is_caught_where_replies_repeat_their_text():
    w, fake = _with_a_kept_turn()
    w.reply("noch?").finish()
    sound = list(fake.cache)
    replies = [m["content"] for m in w.conversation()["messages"]
               if m["role"] == "assistant" and "tool_calls" not in m]
    assert len(replies) >= 2 and len(set(replies)) == 1     # every reply repeats one text
    # The cache without the exchange the question "und?" opened.
    starts, text = [], ""
    for token in sound:
        starts.append(len(text))
        text += fake.token_text(token)
    begin, end = text.index("und?"), text.index("noch?")
    begin, end = text.rfind("<[turn>", 0, begin), text.rfind("<[turn>", 0, end)
    first, last = starts.index(begin), starts.index(end)
    broken = sound[:first] + sound[last:]
    fake.cache = broken
    overwrite_held_tokens(w, broken)
    with pytest.raises(AssertionError, match="does not read as the template renders"):
        assert_cache_reads_as_rendered(w, fake)
    fake.cache = sound
    overwrite_held_tokens(w, sound)
