"""The check every test on a model-free cache ends with: it fails on a cache that does not hold what the wrapper holds it to be, or does not read as the template renders the conversation."""

import re
import pytest

from llama_chat import ChatWrapper, Config, InternalError
from llama_chat.tools import ToolCall, ToolSpec
from tests.chat_messages import tool_turn, opened, after_system
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import PLAIN, TOOLS, REASONING
from tests.internals import held_tokens, leave_cache_behind, overwrite_held_tokens, generated_turns_held
from tests.wrappers import (assert_cache_reads_as_rendered, begun_wrapper, reasoning_wrapper,
                            tool_wrapper, ask_through_a_tool_call, THOUGHTFUL)
from tests import internals
from tests.chat_templates import ANSWERS, QUESTIONS, STEMS, fixture

_MARK = ord("#")        # a token that spells a character no render here holds


def _plain():
    w, fake = begun_wrapper(gen_tokens=5, gen_text="Hallo")
    w.reply("hi").finish()
    return w, fake


def _reasoning():
    w, fake = reasoning_wrapper()
    w.reply("hi").finish()
    return w, fake


def _with_a_generated_turn():
    w, fake = tool_wrapper()
    w.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    w.inject_tool_result(call.id, "12:00")
    w.reply(tool_call_policy="yield").finish()
    w.reply("und?").finish()
    return w, fake


def _owed_from_a_history():
    w, fake = tool_wrapper()
    w.begin(opened("persona", [{"role": "user", "content": "wie spät?"}, *tool_turn(ToolCall("clock", {}))]))
    assert w.pending_calls
    return w, fake


def _one_token_off(tokens):
    """Every list one token away from ``tokens``: one inserted, removed or replaced at each position."""
    for at in range(len(tokens) + 1):
        yield tokens[:at] + [_MARK] + tokens[at:]
        if at < len(tokens):
            yield tokens[:at] + tokens[at + 1:]
            yield tokens[:at] + [_MARK] + tokens[at + 1:]


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_generated_turn,
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


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_generated_turn,
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


@pytest.mark.parametrize("conversation", [_plain, _reasoning, _with_a_generated_turn,
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


def test_a_cache_missing_an_exchange_after_a_generated_turn_is_caught_where_replies_repeat_their_text():
    w, fake = _with_a_generated_turn()
    w.reply("noch?").finish()
    sound = list(fake.cache)
    replies = [m["content"] for m in after_system(w.conversation()["messages"])
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


# ----- the check_cache setting ----------------------------------------------------
@pytest.mark.parametrize("cache", [dict(can_shift=True), dict(can_shift=False, can_cut_tail=True),
                                   dict(can_shift=False)], ids=["shifts", "cuts", "saves states"])
def test_check_cache_stops_a_reply_the_cache_no_longer_holds_the_render_for(cache):
    from llama_chat import InternalError
    fake = FakeContext(PLAIN, replies=["Gut.", "Ja.", "Auch."], **cache)
    wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    wrapper.reply("Wie geht es?").finish()
    memory = wrapper._conversation._memory
    hold = memory.hold

    def holding_one_token_wrong(target, **options):
        # A fault behind the sync: the cache holds another token than it was told to.
        held = hold(target, **options)
        at = len(memory.tokens) // 2
        memory._tokens[at] = fake.cache[at] = memory.tokens[at - 1]
        return held
    memory.hold = holding_one_token_wrong
    with pytest.raises(InternalError, match="differs from a fresh render of its conversation"):
        wrapper.reply("Und dir?").finish()
        wrapper.reply("Und sonst?").finish()
    wrapper.close()                             # its cache holds the wrong token on purpose


def test_check_cache_finds_nothing_in_a_conversation_kept_right():
    fake = FakeContext(PLAIN, replies=["Gut.", "Ja.", "Auch."])
    wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    for question in ("Wie geht es?", "Und dir?", "Schön."):
        wrapper.reply(question).finish()


def test_check_cache_logs_a_fault_it_finds_after_a_reply(caplog):
    fake = FakeContext(PLAIN, replies=["Gut.", "Ja."])
    wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    memory = wrapper._conversation._memory
    hold = memory.hold

    def holding_one_token_wrong(target, **options):
        held = hold(target, **options)
        at = len(memory.tokens) // 2
        memory._tokens[at] = fake.cache[at] = memory.tokens[at - 1]
        return held
    memory.hold = holding_one_token_wrong
    with caplog.at_level("ERROR", logger="llama_chat"):
        wrapper.reply("Wie geht es?").finish()
    assert any("check_cache" in r.getMessage() for r in caplog.records)
    wrapper.close()                             # its cache holds the wrong token on purpose


@pytest.mark.parametrize("cache", [dict(can_shift=True), dict(can_shift=False, can_cut_tail=True),
                                   dict(can_shift=False)], ids=["shifts", "cuts", "saves states"])
def test_check_cache_stops_a_reply_whose_cache_lacks_the_end_of_its_generation_prompt(cache):
    """A cache one token short of the render the model reads is a fault there, though it would pass as work still to do after a reply."""
    fake = FakeContext(PLAIN, replies=["Gut.", "Ja.", "Auch."], **cache)
    wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    wrapper.reply("Wie geht es?").finish()
    conversation = wrapper._conversation
    memory, ask = conversation._memory, conversation.ask

    def asking_one_token_short(text, cancel=None):
        # A fault behind the question: the cache gives up the last token it was told to hold.
        ask(text, cancel)
        memory.hold(list(memory.tokens)[:-1], want_logits=True)
    conversation.ask = asking_one_token_short
    with pytest.raises(InternalError, match="where the model reads it"):
        wrapper.reply("Und dir?").finish()
    wrapper.close()                             # its cache lacks a token on purpose


def test_check_cache_finds_nothing_in_a_conversation_that_keeps_a_turn_as_generated():
    fake = FakeContext(TOOLS)
    wrapper = ChatWrapper(Config(context_size=1024, evict_above_fraction=0.9, check_cache=True),
                          context=fake)
    ask_through_a_tool_call(wrapper, fake)
    assert internals.generated_turns_held(wrapper)
    wrapper.reply("und?").finish()
    wrapper.reply("noch?").finish()
    assert wrapper._conversation._found_fault is None


def _checked_after_a_tag_typed_into_a_question():
    fake = FakeContext(PLAIN, replies=["Gut.", "Ja."])
    wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    wrapper.reply("hi <turn]> there").finish()
    return wrapper


def _checked_past_half_its_context():
    fake = FakeContext(PLAIN, gen_tokens=20, gen_text="Eine Antwort, lang genug.")
    config = Config(context_size=256, check_cache=True)
    wrapper = ChatWrapper(config, context=fake)
    wrapper.begin(opened("s"))
    for n in range(6):
        wrapper.reply(f"Frage Nummer {n}, auch lang genug?").finish()
        wrapper.wait_idle()
    assert wrapper.total_tokens > config.context_size // 2     # more than a begin keeps of it
    return wrapper


def _checked_at_the_exchange_limit():
    fake = FakeContext(PLAIN, gen_tokens=2)
    wrapper = ChatWrapper(Config(context_size=4000, max_history_exchanges=2, check_cache=True),
                          context=fake)
    wrapper.begin(opened("s"))
    for n in range(4):
        wrapper.reply(f"q{n}").finish()
        wrapper.wait_idle()
    return wrapper


def _checked_with_a_declared_tool():
    fake = FakeContext(TOOLS, replies=["Gut.", "Ja."])
    wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
    wrapper.begin(opened("s"), tools=[ToolSpec("clock", "Die Zeit.", {})])
    wrapper.reply("Wie geht es?").finish()
    return wrapper


def _checked_with_reasoning(reasoning):
    def checked():
        fake = FakeContext(REASONING, gen_tokens=len(THOUGHTFUL), gen_text=THOUGHTFUL)
        wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
        wrapper.begin(opened("s"), reasoning=reasoning)
        wrapper.reply("Wie geht es?").finish()
        return wrapper
    return checked


def _checked_with_a_turn_generated_as_other_tokens():
    fake = FakeContext(TOOLS, tokens_spell_text=False)
    wrapper = ChatWrapper(Config(context_size=1024, evict_above_fraction=0.9, check_cache=True),
                          context=fake)
    wrapper.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in wrapper.reply("wie spät?", tool_call_policy="yield") if isinstance(e, ToolCall)]
    wrapper.inject_tool_result(call.id, "12:00")
    wrapper.reply(tool_call_policy="yield").finish()
    wrapper.wait_idle()
    assert generated_turns_held(wrapper)
    wrapper.reply("und?").finish()
    return wrapper


@pytest.mark.parametrize("conversation", [
    _checked_after_a_tag_typed_into_a_question, _checked_past_half_its_context,
    _checked_at_the_exchange_limit, _checked_with_a_declared_tool, _checked_with_reasoning(True),
    _checked_with_reasoning(False), _checked_with_a_turn_generated_as_other_tokens],
    ids=["typed-tag", "past-half-the-context", "exchange-limit", "declared-tool", "reasoning-on", "reasoning-off",
         "turn-generated-as-other-tokens"])
def test_check_cache_finds_nothing_in_any_conversation_kept_right(conversation):
    wrapper = conversation()
    wrapper.reply("Und sonst?").finish()        # checked where the model reads it
    wrapper.wait_idle()
    assert wrapper._conversation._found_fault is None


def test_check_cache_names_where_the_cache_first_differs_and_shows_both_from_there():
    fake = FakeContext(PLAIN, replies=["Gut.", "Ja.", "Auch."])
    wrapper = ChatWrapper(Config(context_size=1024, check_cache=True), context=fake)
    wrapper.begin(opened("s"))
    wrapper.reply("Wie geht es?").finish()
    wrapper.wait_idle()
    memory = wrapper._conversation._memory
    hold = memory.hold
    sound = list(memory.tokens)
    at = len(sound) // 2

    def holding_one_token_wrong(target, **options):
        held = hold(target, **options)
        memory._tokens[at] = fake.cache[at] = memory.tokens[at - 1]
        return held
    memory.hold = holding_one_token_wrong
    with pytest.raises(InternalError) as found:
        wrapper.reply("Und dir?").finish()
        wrapper.reply("Und sonst?").finish()
    wrapper.close()                             # its cache holds the wrong token on purpose
    said = re.search(r"from token (\d+) on: it holds '(.*)' \(\d+ tokens\) where the render has "
                     r"'(.*)' \(\d+ tokens\)", str(found.value))
    assert said is not None, str(found.value)
    assert int(said[1]) == at
    assert said[2] != said[3]
    assert said[3].startswith(fake.token_text(sound[at]))


@pytest.mark.parametrize("stem", STEMS)
def test_check_cache_finds_nothing_in_a_conversation_kept_right_on_every_test_template(stem):
    wrapper = ChatWrapper(Config(context_size=8192, check_cache=True),
                          context=FakeContext(fixture(stem), ANSWERS))
    wrapper.begin(opened("Be brief." if wrapper.capabilities(tools=False).can_take_system_prompt
                         else ""))
    for question in QUESTIONS[:3]:
        wrapper.reply(question).finish()        # checked where the model reads it
        wrapper.wait_idle()                     # and once the work after the reply has ended
        assert wrapper._conversation._found_fault is None
