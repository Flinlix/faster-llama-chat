"""Cutting the newest reply back, and the saved states a cache that cannot shift returns to."""

import pytest

from llama_chat.config import Config
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tools import ToolSpec
from llama_chat.wrapper import ChatWrapper
from tests.chat_templates import STEMS, expected, fixture
from tests.fake_context import FakeContext
from tests.internals import token_id_range
from tests.invented_templates import PLAIN, TOOLS
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
    assert rendered_reply(fake, "ok!!") in held_text(fake)
    assert not any(fake.is_generated_only(token) for token in fake.cache)   # none is kept
    assert reply.text == "ok!!"


@pytest.mark.parametrize("spelled", [True, False], ids=["spelling-text", "other-pieces"])
def test_every_token_id_the_stand_in_hands_out_lies_where_a_context_may_have_one(spelled):
    w, fake = begun_wrapper(gen_tokens=40, gen_text="Ein langer Satz, der viele Token braucht.",
                            tokens_spell_text=spelled)
    sampled, drawn = fake.sampled_tokens, []

    def recorded(*args, **kwargs):
        for token in sampled(*args, **kwargs):
            drawn.append(token)
            yield token

    fake.sampled_tokens = recorded
    for question in ("eins", "zwei", "drei"):
        w.reply(question).finish()
    handed_out = set(drawn) | set(fake.cache) | set(fake.special_tokens())
    assert handed_out <= set(token_id_range())
    assert spelled or any(fake.is_generated_only(token) for token in drawn)


def test_a_model_whose_template_renders_nothing_is_refused_at_load_naming_why():
    class _RendersNothing(FakeContext):
        def render_with_model_template(self, messages, tools=None, **options):
            return None

    with pytest.raises(ValueError, match="reply can be told from the question"):
        ChatWrapper(Config(context_size=500), context=_RendersNothing(PLAIN))


def test_a_model_whose_template_drops_every_reply_is_refused_at_load_naming_why():
    drops = dict(PLAIN, tags=["<|user_says|>", "<|assistant_says|>", "<|said|>"],
                 eos_token="<|said|>",
                 chat_template="{%- for m in messages if m.role != 'assistant' -%}"
                               "<|{{ m.role }}_says|>{{ m.content }}<|said|>{%- endfor -%}"
                               "{%- if add_generation_prompt -%}<|assistant_says|>{%- endif -%}")
    with pytest.raises(ValueError, match="reply can be told from the question"):
        ChatWrapper(Config(context_size=500), context=FakeContext(drops))


def test_a_context_of_your_own_that_reports_no_template_text_is_refused_at_load():
    """Its renders alone would pass every check, but no add-on file could name its template."""
    class _NoTemplateText(FakeContext):
        def chat_template_text(self, template_name=None):
            return None

    with pytest.raises(ValueError, match="no chat template"):
        ChatWrapper(Config(context_size=500), context=_NoTemplateText(PLAIN))


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
    assert held_text(fake).endswith(rendered_reply(fake, "Eins.").rstrip("\n") + "\n<[turn>")
    assert "Zwei" not in held_text(fake)


@pytest.mark.parametrize("stem", [stem for stem in STEMS
                                  if not expected(fixture(stem), "writes_empty_reply")])
def test_a_cut_to_a_reply_the_template_does_not_write_is_refused(stem):
    """These templates write no empty reply, so a reply cut to nothing could be neither held nor followed."""
    fake = FakeContext(fixture(stem), gen_tokens=7, gen_text="Antwort")
    w = ChatWrapper(Config(context_size=8192, evict_above_fraction=1.0), context=fake)
    w.begin("")
    w.reply("Hallo").finish()
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(ValueError, match="keep more"):
        w.truncate_last_reply("")
    assert w.conversation() == before and fake.cache == held
    w.truncate_last_reply("Ant")
    assert w.reply("Weiter").finish().text == "Antwort"


def test_a_cut_to_nothing_stands_where_the_template_writes_an_empty_reply():
    w, fake = begun_wrapper(gen_tokens=8, gen_text="Eins. Zw")
    w.reply("hi").finish()
    w.truncate_last_reply("")
    assert rows(w)[-1] == {"role": "assistant", "text": ""}
    w.reply("weiter").finish()


def test_a_reply_that_called_a_tool_cannot_be_cut():
    w, fake = begun_wrapper(TOOLS)
    fake.call_script.append("fn:uhr{}")
    w.begin("s", tools=[ToolSpec("uhr", "", {})])
    for _event in w.reply("wie spät?", tool_call_policy="yield"):
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
    assert fake.total_prefilled() - prefilled < 2 * len(newest_turn(fake)) + 20
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


@pytest.mark.parametrize("refusal", ["none", "raises"])
def test_a_context_of_your_own_that_can_neither_shift_cut_nor_save_is_refused(refusal):
    """It could go back to an earlier point only by decoding the whole conversation again."""
    class _NoWayBack(FakeContext):
        def save_state(self):
            if refusal == "raises":
                raise RuntimeError("no state to save")

    with pytest.raises(ValueError, match="neither shift"):
        ChatWrapper(Config(context_size=500),
                    context=_NoWayBack(PLAIN, can_shift=False, can_cut_tail=False))
    for can_shift, can_cut_tail in ((True, False), (False, True)):
        ChatWrapper(Config(context_size=500), context=_NoWayBack(
            PLAIN, can_shift=can_shift, can_cut_tail=can_cut_tail)).begin("s")


def test_a_context_that_saves_states_must_restore_them():
    with pytest.raises(TypeError, match="restore_state"):
        ChatWrapper(Config(context_size=500),
                    context=_without_restore(FakeContext(PLAIN, can_shift=False)))


# ----- a reply ended by its closing text -----------------------------------
@pytest.mark.parametrize("stem", [stem for stem in STEMS if expected(fixture(stem), "turn_end_text")])
def test_a_reply_ended_by_its_closing_text_is_cut_back_as_its_cache_allows(stem):
    """A cache that cannot cut at its end goes back to a saved state instead."""
    # A template whose reply ends where a text its closing opens with is written.
    known = fixture(stem)
    closing = ResolvedTemplate(FakeContext(known)).call_boundary().turn_end_text
    for can_shift in (True, False):
        # The reply, then its closing, which leaves the reply again.
        fake = FakeContext(known, ["rep" + closing, "Fine."], can_shift=can_shift)
        w = ChatWrapper(Config(context_size=2000, min_reply_tokens=8), context=fake)
        w.begin("s")
        assert w.reply("hi").finish().text == "rep"
        assert fake.edit_count("cut" if can_shift else "restore") == 1
        w.reply("again").finish()
        assert_consistent(w, fake)
