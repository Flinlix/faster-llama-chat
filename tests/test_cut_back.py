"""Cutting the newest reply back, and the saved states a cache that cannot shift returns to."""

import pytest

from llama_chat.config import Config
from llama_chat.tools import ToolSpec
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import FakeContext
from tests.wrappers import (assert_consistent, begun_wrapper, held_text, newest_turn,
                            reasoning_wrapper, rendered_reply, rows)
def _without_restore(fake):
    class Wrapped:
        def __getattr__(self, name):
            if name == "restore_state":
                raise AttributeError(name)
            return getattr(fake, name)
    return Wrapped()


# ----- cutting the newest reply --------------------------------------------
def test_a_cut_keeps_the_prefix_and_the_next_request_appends_at_the_cache_end():
    w, _fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply("hi").finish()
    w.truncate_last_reply("Eins.")
    assert rows(w)[-1]["text"] == "Eins."
    w.reply("weiter").finish()


def test_a_reply_generated_in_other_pieces_than_its_text_tokenizes_into_is_kept_as_its_text():
    """Same text, other tokens: the template's past reply is the text, tokenized."""
    w, fake = begun_wrapper(gen_tokens=4, gen_text="ok!!", tokens_spell_text=False)
    reply = w.reply("hi").finish()
    assert rendered_reply(fake, "ok!!") in held_text(w)
    assert all(t < 1000 for t in fake.cache)          # none of the generated ids is kept
    assert reply.text == "ok!!"


def test_a_model_that_ships_no_chat_template_is_refused_at_load():
    class _NoTemplate(FakeContext):
        def render_with_model_template(self, messages, tools=None, **options):
            return None

    with pytest.raises(ValueError, match="no chat template"):
        ChatWrapper(Config(context_size=500), context=_NoTemplate())


def test_the_tokens_that_stay_are_not_decoded_again():
    w, fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply("hi").finish()
    before = list(fake.prefill_calls)
    w.truncate_last_reply("Eins.")
    (start, n, _logits), = fake.prefill_calls[len(before):]
    # From the end of what stays: the closing and the next question's opening.
    assert fake.text_of(fake.cache[:start]).endswith("Eins.")
    assert n == len(fake.tokenize("<turn]>\n<[turn>"))


def test_keeping_the_whole_text_changes_nothing():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi").finish()
    cache = list(fake.cache)
    w.truncate_last_reply("Hallo")
    assert fake.cache == cache


def test_a_text_that_is_not_a_prefix_is_refused_and_changes_nothing():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi").finish()
    cache = list(fake.cache)
    with pytest.raises(ValueError, match="prefix"):
        w.truncate_last_reply("Tschüss")
    assert fake.cache == cache


def test_only_an_assistant_reply_can_be_cut():
    w, _fake = begun_wrapper()
    w.inject("nebenbei", "user")
    with pytest.raises(ValueError, match="assistant"):
        w.truncate_last_reply("")


def test_a_prefilled_reply_keeps_the_templates_opening_when_cut():
    w, fake = begun_wrapper()
    w.inject("Eins. Zwei.", "assistant")
    w.truncate_last_reply("Eins.")
    assert held_text(w).endswith(rendered_reply(fake, "Eins.").rstrip("\n") + "\n<[turn>")
    assert "Zwei" not in held_text(w)


def test_a_reply_that_called_a_tool_cannot_be_cut():
    w, fake = begun_wrapper(calls_tools=True)
    fake.call_script.append("uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})])
    for _event in w.reply("wie spät?", calls="yield"):
        pass
    with pytest.raises(ValueError, match="tool call"):
        w.truncate_last_reply("")


@pytest.mark.parametrize("saves_state", [True, False])
def test_a_cut_on_a_cache_that_cannot_shift_restores_or_rebuilds(saves_state):
    w, fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw", can_shift=False)
    if not saves_state:
        fake.save_state = lambda: None
        w.begin("s")
    w.reply("hi").finish()
    resets = fake.resets
    w.truncate_last_reply("Eins.")
    # From the state saved before the reply where there is one, else rebuilt.
    assert fake.resets == resets + (not saves_state)
    assert rows(w)[-1]["text"] == "Eins."


# ----- a past reply on a cache that cannot shift ---------------------------
def test_a_past_reply_on_a_cache_that_cannot_shift_comes_back_from_its_saved_state():
    w, fake = reasoning_wrapper(can_shift=False)
    resets = fake.resets
    prefilled = fake.total_prefilled()
    w.reply("hi").finish()
    # No rebuild: the request and the past reply are all that is decoded again.
    assert fake.resets == resets
    assert fake.total_prefilled() - prefilled < 2 * len(newest_turn(w)) + 20
    assert_consistent(w, fake)
    w.truncate_last_reply("Ant")
    assert fake.resets == resets


def test_a_cache_that_saves_no_state_is_rebuilt_instead():
    w, fake = reasoning_wrapper(can_shift=False)
    fake.save_state = lambda: None
    w.begin("s", reasoning=True)
    resets = fake.resets
    w.reply("hi").finish()
    assert fake.resets == resets + 1


# ----- saved states of a cache that cannot shift ---------------------------
def test_a_saved_state_the_cache_refuses_is_rebuilt_from_instead():
    w, fake = reasoning_wrapper(can_shift=False)
    w.reply("hi").finish()

    def refuse(state, pos):
        raise RuntimeError("the cache refused the saved state")
    fake.restore_state = refuse
    resets = fake.resets
    w.truncate_last_reply("Ant")
    assert fake.resets == resets + 1


def test_a_cache_that_takes_a_cut_needs_no_saved_state():
    w, fake = reasoning_wrapper(can_shift=False)
    fake.can_cut_tail = True
    w.begin("s", reasoning=True)
    saved, resets = fake.saved_states, fake.resets
    w.reply("hi").finish()
    w.truncate_last_reply("Ant")
    assert fake.saved_states == saved and fake.resets == resets


def test_a_cut_leaves_a_state_for_the_next_reply():
    w, fake = reasoning_wrapper(can_shift=False)
    w.reply("hi").finish()
    saved = fake.saved_states
    w.truncate_last_reply("Ant")
    assert fake.saved_states == saved + 1


@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot-shift"])
def test_what_a_reply_decodes_again_is_counted(can_shift):
    w, fake = reasoning_wrapper(can_shift=can_shift)
    before = fake.total_prefilled()
    reply = w.reply("hi").finish()
    assert reply.n_prefilled_tokens == fake.total_prefilled() - before


def test_a_context_that_saves_states_must_restore_them():
    with pytest.raises(TypeError, match="restore_state"):
        ChatWrapper(Config(context_size=500),
                    context=_without_restore(FakeContext(can_shift=False)))


# ----- a reply ended by its closing text -----------------------------------
def test_a_reply_ended_by_its_closing_text_is_cut_back_as_its_cache_allows():
    """A cache that cannot cut at its end goes back to a saved state instead."""
    from llama_chat.template import CallBoundary

    class _ClosingText(FakeContext):
        def remove_from(self, pos):
            assert self.can_shift, "a cache that cannot cut was cut"
            super().remove_from(pos)

    for can_shift in (True, False):
        # Three tokens of the reply, then two of the closing, which leave the reply again.
        fake = _ClosingText(gen_tokens=5, gen_text="rep<|", can_shift=can_shift)
        w = ChatWrapper(Config(context_size=2000), context=fake)
        # A template whose reply ends where the text its closing opens with begins.
        w._template.call_boundary = lambda *args, **kwargs: CallBoundary(turn_end_text="<|")
        w.begin("s")
        assert w.reply("hi").finish().text == "rep"
        w.reply("again").finish()
        assert_consistent(w, fake)
