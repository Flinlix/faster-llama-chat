"""The two eviction limits, the token threshold and the exchange limit: which exchanges go, when, and what a cache that cannot shift does instead."""


import math
import random
import pytest

from llama_chat.config import Config
from llama_chat import ChatWrapper, ContextOverflowError
from llama_chat.messages import QUESTION_CUT_MARK
from tests.chat_messages import opened, after_system
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import PLAIN, TOOLS
from tests.wrappers import assert_consistent, held_text, rows
from llama_chat.conversation_state import ConversationState
from llama_chat.eviction import Eviction, EVICT_TO_FRACTION_WITHOUT_SHIFT
from llama_chat.tools import ToolCall
from tests import internals


def _n_kept(wrapper):
    """Messages kept: everything but the system prompt."""
    return len(rows(wrapper)) - 1


def _n_exchanges(wrapper):
    """Exchanges the limit counts: each user message opens one, and what comes before the first is one more."""
    history = [row["role"] for row in rows(wrapper) if row["role"] != "system"]
    return history.count("user") + (1 if history and history[0] != "user" else 0)


def test_a_generation_evicts_down_to_the_exchange_limit():
    # Far under the token threshold, so the exchange count is the only trigger.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=2), context=fake)
    w.begin(opened("s"))
    for i in range(8):
        w.reply(f"q{i}").finish()
        assert _n_exchanges(w) == min(i + 1, 2) and _n_kept(w) == 2 * _n_exchanges(w)
        assert w.total_tokens < w.config.evict_above_tokens  # the tokens never bound
        assert rows(w)[0]["role"] == "system"       # system still survives
        assert rows(w)[-1]["role"] == "assistant"   # newest turn kept
        assert_consistent(w, fake)


def test_a_question_not_prepared_ahead_shifts_the_exchange_the_limit_drops():
    """A question ending in a space cannot take the stand-in's place, and its eviction still shifts the cache."""
    fake = FakeContext(PLAIN, gen_tokens=5, gen_text="Gern.")
    w = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0, max_history_exchanges=2),
                    context=fake)
    w.begin(opened("s"))
    for question in ("Eins?", "Zwei?", "Drei?"):
        w.reply(question).finish()
    fake.edits.clear()
    reply = w.reply("Vier? ").finish()
    assert [edit[0] for edit in fake.edits] == ["shift"]
    newest = held_text(fake)[held_text(fake).rfind("<[turn>user"):]
    assert reply.n_prefilled_tokens <= len(fake.tokenize(newest))


def test_the_tighter_of_the_two_limits_is_the_one_that_binds():
    # Same conversation under each limit alone, then under both: whichever
    # would evict more is what the pair evicts, because either one triggers.
    def kept_under(**limits):
        fake = FakeContext(PLAIN, gen_tokens=2)
        w = ChatWrapper(Config(context_size=400, **limits), context=fake)
        w.begin(opened("s"))
        for i in range(8):
            w.reply(f"message number {i} with some length").finish()
        assert_consistent(w, fake)
        return _n_kept(w)

    by_tokens = kept_under(evict_above_fraction=0.5)
    by_exchanges = kept_under(evict_above_fraction=1.0, max_history_exchanges=1)
    both = kept_under(evict_above_fraction=0.5, max_history_exchanges=1)
    assert both == min(by_tokens, by_exchanges)


def test_a_limit_that_never_binds_changes_nothing():
    def snapshot_under(**limits):
        fake = FakeContext(PLAIN, gen_tokens=2)
        w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5, **limits),
                        context=fake)
        w.begin(opened("s"))
        for i in range(8):
            w.reply(f"message number {i} with some length").finish()
        return rows(w)

    assert snapshot_under() == snapshot_under(max_history_exchanges=100)


def test_inject_evicts_to_leave_room_under_the_exchange_limit():
    # inject evicts before appending, so the exchange the arriving message opens
    # counts against the limit while it is still absent from the conversation.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=2), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]))
    assert _n_exchanges(w) == 2
    # The oldest exchange goes whole, so the survivors open on a user turn.
    assert w.inject("reference material").n_evicted_messages == 2
    assert [r["text"] for r in rows(w)][1:] == ["c", "reference material"]


@pytest.mark.parametrize("history, kept", [
    ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}], ["c", "d"]),
    ([{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}, {"role": "assistant", "content": "d"}],
     ["b", "c", "d"]),
    ([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}], ["c"]),
])
def test_begin_prefills_no_more_than_the_exchange_limit(history, kept):
    # Without the cap, begin would prefill history the first turn then evicts.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=1), context=fake)
    assert w.begin(opened("s", history)).n_kept_messages == len(kept)
    assert [r["text"] for r in rows(w)] == ["s"] + kept       # the newest exchange, whole


def test_a_limit_of_one_keeps_the_newest_exchange_between_turns():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=1), context=fake)
    w.begin(opened("s"))
    for i in range(3):
        w.reply(f"q{i}").finish()
        assert [(r["role"], r["text"]) for r in rows(w)][:2] == [("system", "s"), ("user", f"q{i}")]
        assert _n_kept(w) == 2
        assert_consistent(w, fake)


def test_the_limit_holds_across_every_action_that_appends():
    # The property, not three examples: whatever the mix of actions, the count
    # never exceeds the limit and the system prompt is never the one dropped.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=2), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "old"}, {"role": "assistant", "content": "older"}]))
    for i in range(6):
        w.inject(f"context {i}")
        assert _n_exchanges(w) <= 2
        w.reply(f"q{i}").finish()
        assert _n_exchanges(w) <= 2
        assert rows(w)[0]["role"] == "system"
        assert_consistent(w, fake)


def test_eviction_shift_mode_does_not_rebuild():
    # Default cache supports shifting: eviction uses apply_eviction, never rebuild.
    fake = FakeContext(PLAIN, gen_tokens=2)  # can_shift=True
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin(opened("s"))
    resets_after_begin = fake.resets  # begin() resets once
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
    assert fake.edit_count("shift") > 0          # eviction did happen
    assert fake.resets == resets_after_begin  # but never via rebuild


def test_a_dropped_stretch_is_found_among_token_ids_up_to_the_bound_the_readme_names():
    from llama_chat.cache_sync import CacheSync
    fake = FakeContext(PLAIN)
    memory = CacheSync(fake)
    low, high = -0x10000, 0xFFFFF                   # -65,536 and 1,048,575
    dropped, kept = [high] * 4, [low, high] * 5
    memory.decode([1] + dropped + kept)
    memory.hold([1] + kept, find_dropped_stretch=True)
    assert fake.edits == [("shift", 1, 4)] and fake.cache == [1] + kept


def test_eviction_rebuild_mode_when_cache_cannot_shift():
    # A cache that cannot shift goes back to a saved state or is decoded again.
    fake = FakeContext(PLAIN, gen_tokens=2, can_shift=False)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin(opened("s"))
    resets_after_begin = fake.resets
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
        roles = [s["role"] for s in rows(w)]
        assert roles[0] == "system"
        assert w.total_tokens <= w.config.context_size
        assert_consistent(w, fake)             # the cache as recorded after it is brought back
    assert fake.edit_count("shift") == 0
    assert fake.resets + fake.edit_count("restore") > resets_after_begin


def test_a_message_that_cannot_fit_is_refused_before_anything_is_evicted():
    # Rejecting an unfittable message must not evict the history first.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]))
    before = rows(w)
    with pytest.raises(ValueError):
        w.inject("x" * 500, role="assistant")
    assert rows(w) == before


def test_every_refusal_for_want_of_room_says_what_to_change():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, min_reply_tokens=32),
                    context=fake)
    w.begin(opened("s"))
    with pytest.raises(ContextOverflowError, match="shorten the message.*evict_above_fraction"):
        w.inject("x" * 600, role="assistant")
    w.inject("x" * 470, role="assistant")       # what is answered by none needs no such room


@pytest.mark.parametrize("asks", ["reply", "inject"])
def test_a_question_longer_than_the_room_keeps_its_end_and_is_answered(asks, caplog):
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, min_reply_tokens=32),
                    context=fake)
    w.begin(opened("s", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]))
    question = "Anfang " + "x" * 2000 + " und die eigentliche Frage?"
    with caplog.at_level("WARNING", logger="llama_chat"):
        if asks == "reply":
            assert w.reply(question).finish().text
        else:
            w.inject(question)
            assert w.reply().finish().text
    asked = [m for m in after_system(w.conversation()["messages"]) if m["role"] == "user"][-1]
    assert asked["content"].startswith(QUESTION_CUT_MARK)
    assert asked["content"].endswith(" und die eigentliche Frage?") and "Anfang" not in asked["content"]
    assert any("characters were cut" in record.message for record in caplog.records)
    assert w.total_tokens <= w.config.context_size


class _CostlyLastCharacter(FakeContext):
    """A stand-in on which one character takes three tokens, more than the ``?`` a begin measures its room with."""

    COSTLY = "😀"

    def _plain(self, text):
        return [token for c in text for token in ([ord(c), 0, 0] if c == self.COSTLY else [ord(c)])]


def test_a_question_whose_last_character_does_not_fit_is_asked_as_the_cut_mark_and_answered(caplog):
    """At the edge a begin leaves, the cut may keep nothing of the question; the model still answers the mark."""
    cases = set()
    for system_length in range(150, 260):
        w = ChatWrapper(Config(context_size=256, min_reply_tokens=8, evict_above_fraction=1.0),
                        context=_CostlyLastCharacter(PLAIN, gen_tokens=2))
        try:
            w.begin(opened("s" * system_length))
        except ContextOverflowError:
            continue
        reply = w.reply("x" * 500 + " Frage " + _CostlyLastCharacter.COSTLY).finish()
        asked = [m for m in after_system(w.conversation()["messages"]) if m["role"] == "user"][-1]
        assert reply.text
        cases.add(asked["content"] == QUESTION_CUT_MARK)
    assert cases == {True, False}


def test_a_history_no_question_follows_is_refused_where_it_cannot_fit():
    """Nothing in it can be evicted, since an eviction drops up to the next user message."""
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=0.8), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]))
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(ContextOverflowError, match="no user message follows"):
        w.begin(opened("s", [{"role": "assistant", "content": "x" * 200}] * 3))
    assert w.conversation() == before and fake.cache == held
    # With a later question, the replies before it are evicted instead.
    w.begin(opened("s", [{"role": "assistant", "content": "x" * 200}] * 3 + [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}]))
    assert after_system(w.conversation()["messages"]) == [{"role": "user", "content": "q"},
                                            {"role": "assistant", "content": "r"}]


@pytest.mark.parametrize("history", [[], [{"role": "user", "content": "q"}]], ids=["no-question", "after-a-question"])
def test_replies_injected_in_a_row_are_refused_once_they_no_longer_fit(history):
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=0.8), context=fake)
    w.begin(opened("s", history))
    with pytest.raises(ContextOverflowError, match="no user message follows"):
        for _ in range(8):
            before, held = w.conversation(), list(fake.cache)
            w.inject("y" * 150, role="assistant")
            assert w.total_tokens <= w.config.evict_above_tokens
    assert w.conversation() == before and fake.cache == held
    w.inject("Und jetzt?")                      # a question lets them go
    w.inject("y" * 150, role="assistant")
    assert w.total_tokens <= w.config.evict_above_tokens


def test_a_reply_over_the_threshold_alone_stays_until_the_next_question_so_it_can_be_cut():
    """A reader that heard part of a long reply cuts it back; the next question evicts it."""
    fake = FakeContext(PLAIN, gen_tokens=250)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5), context=fake)
    w.begin(opened("s"))
    text = w.reply("Erzähl lange.").finish().text
    assert w.conversation()["messages"][-1]["content"] == text
    w.truncate_last_reply(text[:20])
    assert w.reply("Und kurz?").finish().text
    assert w.conversation()["messages"][-2]["content"] == "Und kurz?"


@pytest.mark.counts_work
@pytest.mark.parametrize("can_shift", [True, False], ids=["shifts", "cannot shift"])
def test_the_search_for_how_many_exchanges_to_drop_measures_a_few_times(can_shift):
    """Each guess is placed by how far over the limit the last measure was, so 300 exchanges take a few measures, not one per halving."""
    from llama_chat import eviction
    from llama_chat.testing.invented_templates import NUMBERED
    measures = []
    search = eviction.Eviction._drop_oldest_exchanges

    def counted(self, length, bound, at_least, newest_too):
        count = [0]

        def measured():
            count[0] += 1
            return length()
        went = search(self, measured, bound, at_least, newest_too)
        measures.append(count[0])
        return went
    eviction.Eviction._drop_oldest_exchanges = counted
    try:
        fake = FakeContext(NUMBERED, gen_tokens=2, gen_text="Ja.", can_shift=can_shift)
        history = [m for n in range(300) for m in (
            {"role": "user", "content": f"Frage {n} " + "x" * 10}, {"role": "assistant", "content": "y" * 10})]
        w = ChatWrapper(Config(context_size=16000, evict_above_fraction=0.9), context=fake)
        w.begin(opened("s", history))
        w.reply("Q " + "x" * (15000 - w.total_tokens)).finish()
    finally:
        eviction.Eviction._drop_oldest_exchanges = search
    assert measures and max(measures) <= 7


@pytest.mark.parametrize("cache", [dict(can_shift=True), dict(can_shift=False, can_cut_tail=True),
                                   dict(can_shift=False)], ids=["shifts", "cuts", "saves states"])
def test_a_message_refused_after_the_evictions_it_made_leaves_the_cache_as_it_was(cache):
    """Rule 6: an eviction a refused message made on the cache itself is undone at once."""
    fake = FakeContext(PLAIN, **cache)
    w = ChatWrapper(Config(context_size=1024, min_reply_tokens=16, evict_above_fraction=0.5),
                    context=fake)
    w.begin(opened("s", [m for n in range(5) for m in (
        {"role": "user", "content": f"q{n} " * 10}, {"role": "assistant", "content": f"a{n} " * 10})]))
    w.inject("y" * 5000)                        # cut to fill the room
    w.wait_idle()
    before = w.conversation()
    with pytest.raises(ContextOverflowError):
        w.inject(" " * 30, role="assistant")
    assert w.conversation() == before
    assert_consistent(w, fake)
    assert w.reply().finish().text


@pytest.mark.parametrize("cache", [dict(can_shift=True), dict(can_shift=False, can_cut_tail=True),
                                   dict(can_shift=False)], ids=["shifts", "cuts", "saves states"])
def test_a_reply_injected_where_it_cannot_fit_once_the_older_exchanges_go_is_refused_before_any_goes(cache):
    fake = FakeContext(PLAIN, **cache)
    w = ChatWrapper(Config(context_size=1024, min_reply_tokens=16, evict_above_fraction=0.5),
                    context=fake)
    w.begin(opened("s", [m for n in range(5) for m in (
        {"role": "user", "content": f"q{n} " * 10}, {"role": "assistant", "content": f"a{n} " * 10})]))
    w.inject("y" * 5000)                        # cut to fill the room
    w.wait_idle()
    before, decoded, edits = w.conversation(), fake.total_prefilled(), len(fake.edits)
    with pytest.raises(ContextOverflowError, match="no user message follows"):
        w.inject(" " * 30, role="assistant")
    assert w.conversation() == before
    assert (fake.total_prefilled(), len(fake.edits)) == (decoded, edits)


def test_a_long_question_is_cut_to_the_room_under_the_threshold_not_to_the_whole_context():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=0.5, min_reply_tokens=32),
                    context=fake)
    w.begin(opened("s"))
    assert w.reply("x" * 2000 + " Frage?").finish().text
    asked = [m for m in after_system(w.conversation()["messages"]) if m["role"] == "user"][-1]
    beside = len(fake.tokenize(fake.stated_render(opened("s", [asked]), generation_prompt=True)))
    assert w.config.evict_above_tokens - 8 <= beside <= w.config.evict_above_tokens


@pytest.mark.parametrize("template", [PLAIN, TOOLS], ids=["plain", "tools"])
@pytest.mark.parametrize("length", [40, 60, 80, 100])
def test_an_injected_message_that_evicts_decodes_no_more_than_one_that_does_not(template, length):
    """On a cache that can shift, the exchanges kept are moved, not decoded again."""
    fake = FakeContext(template)
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.3), context=fake)
    w.begin(opened("s"))
    evicting, keeping = [], []
    for i in range(20):
        for role in ("user", "assistant"):
            before = fake.total_prefilled()
            report = w.inject(f"{i:03d}" + "x" * length, role=role)
            decoded = fake.total_prefilled() - before
            (evicting if report.n_evicted_messages else keeping).append(decoded)
    assert evicting and keeping
    assert max(evicting) <= max(keeping), (evicting, keeping)
    assert_consistent(w, fake)


def test_an_eviction_drops_the_fewest_oldest_exchanges_that_bring_the_conversation_within_its_bound():
    """Whatever each exchange takes: at least ``at_least`` go, no more than the bound needs, or every one that can go."""
    shapes = random.Random(14)
    for _ in range(100):
        sizes = [shapes.randint(1, 80) for _ in range(shapes.randint(2, 10))]
        messages = [{"role": "system", "content": "s"}] + [
            message for n, size in enumerate(sizes) for message in (
                {"role": "user", "content": f"q{n}"}, {"role": "assistant", "content": "y" * size})]
        state = ConversationState(messages=list(messages))
        eviction = Eviction(None, Config(), True, state, measure=lambda kept, prompted: 0)

        def length(state=state):
            return sum(len(message["content"]) for message in state.messages)

        for newest_too in (False, True):
            can_go = len(sizes) - (0 if newest_too else 1)
            after = [sum(len(m["content"]) for m in messages[:1] + messages[1 + 2 * k:])
                     for k in range(can_go + 1)]
            # Each length an eviction can leave, and one below it: every place the fewest changes.
            for bound in sorted({at + step for at in after for step in (-1, 0)}):
                for at_least in range(can_go + 2):
                    if not at_least and after[0] <= bound:
                        continue            # no caller searches where nothing is over
                    if after[-1] > bound or can_go <= at_least:
                        fewest = can_go
                    else:
                        fewest = min(k for k in range(at_least, can_go + 1) if after[k] <= bound)
                    state.messages[:] = messages
                    went = eviction._drop_oldest_exchanges(length, bound, at_least, newest_too)
                    assert state.messages == messages[:1] + messages[1 + 2 * fewest:], \
                        (sizes, newest_too, bound, at_least)
                    assert went == (fewest > 0)


@pytest.mark.counts_work
def test_injected_messages_that_evict_turns_that_called_tools_render_only_the_newest_exchanges():
    """Each drop in a window leaves the places of the newest exchanges standing, so the next render is of them alone."""
    fake = FakeContext(TOOLS, gen_tokens=6, gen_text="Answer.")
    w = ChatWrapper(Config(context_size=3000, evict_above_fraction=0.5, min_reply_tokens=8),
                    context=fake)
    w.begin(opened("s"))
    largest, evicted = [], 0
    for i in range(30):
        fake.call_script = ["fn:clock{}"]
        call, = [step for step in w.reply(f"Wie spät {i}?", tool_call_policy="yield")
                 if isinstance(step, ToolCall)]
        w.inject_tool_result(call.id, "12:00")
        w.reply(tool_call_policy="yield").finish()
        before = len(fake.rendered)
        evicted += w.inject(f"Notiz {i} " + "n" * 20).n_evicted_messages
        evicted += w.inject(f"Gesagt {i} " + "g" * 20, role="assistant").n_evicted_messages
        if evicted:
            largest.append(max(fake.rendered[before:], default=0))
    assert internals.renders_tool_turns_in_windows(w)
    assert all(largest), "no render was recorded after an eviction"
    assert largest and len(w.conversation()["messages"]) > 40
    # The system turn, the newest two exchanges of four messages each, and the stand-in.
    assert max(largest) <= 10, largest


def _render_length_of(fake, messages, generation_prompt=False):
    """How many tokens ``fake`` takes for the template's render of ``messages``."""
    return len(fake.tokenize(fake.stated_render(messages, generation_prompt=generation_prompt)))


def test_an_eviction_on_a_cache_that_shifts_drops_the_fewest_oldest_exchanges_that_bring_it_within_the_threshold():
    for padding in range(0, 40, 3):
        fake = FakeContext(PLAIN, gen_tokens=5, gen_text="Gern.")
        w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5), context=fake)
        w.begin(opened("s"))
        limit = w.config.evict_above_tokens
        before = w.conversation()["messages"]
        for i in range(10):
            w.reply(f"Frage {i}? " + "w" * padding + ".").finish()
            w.wait_idle()
            kept = w.conversation()["messages"]
            whole = before + kept[-2:]
            assert kept[1:] == whole[len(whole) - len(kept) + 1:]
            assert _render_length_of(fake, kept) <= limit
            if len(kept) < len(whole):
                # The newest exchange that went, put back, would be over the threshold.
                newest_gone = whole[len(whole) - len(kept) - 1:len(whole) - len(kept) + 1]
                assert _render_length_of(fake, kept[:1] + newest_gone + kept[1:]) > limit, \
                    (padding, i)
            before = kept


def test_an_eviction_after_a_reply_on_a_cache_that_cannot_shift_goes_below_the_threshold_as_far_as_the_newest_two_exchanges_allow():
    rng = random.Random(4)
    # Exchanges of mixed lengths, and short ones before one far longer than the rest.
    runs = [[rng.choice([10, 30, 60, 160]) for _ in range(14)] for _ in range(6)]
    runs.append([10] * 12 + [200, 10, 10, 200])
    for sizes in runs:
        fake = FakeContext(PLAIN, can_shift=False)
        w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.8, min_reply_tokens=8),
                        context=fake)
        w.begin(opened("s"))
        system = w.conversation()["messages"][:1]
        limit = w.config.evict_above_tokens
        shortest = _render_length_of(fake, system + [{"role": "user", "content": QUESTION_CUT_MARK + "?"}],
                                     generation_prompt=True)
        bound = shortest + int((limit - shortest) * EVICT_TO_FRACTION_WITHOUT_SHIFT)
        n_before = 1
        for i, size in enumerate(sizes):
            fake.gen_tokens, fake.gen_text = size, "r" * size
            w.reply(f"Frage {i} " + "w" * size).finish()
            w.wait_idle()
            kept = w.conversation()["messages"]
            if len(kept) < n_before + 2:
                assert _render_length_of(fake, kept) <= bound or len(kept) <= 5, (i, len(kept))
            n_before = len(kept)


def test_a_question_within_the_room_its_reply_needs_is_answered_from_all_history_and_evicts_after_its_reply():
    fake = FakeContext(PLAIN, gen_tokens=5, gen_text="Gern.")
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.5, min_reply_tokens=8),
                    context=fake)
    w.begin(opened("s"))
    for i in range(12):
        w.reply(f"Frage {i}?").finish()
    w.wait_idle()
    oldest = rows(w)[1]["text"]
    seen = []
    fake.before_reply = seen.append
    # A question that cannot take the stand-in's place, past the threshold but within the room.
    reply = w.reply("lang " * 70 + " ").finish()
    assert w.total_tokens < w.config.context_size - w.config.min_reply_tokens
    assert reply.n_evicted_messages == 0 and oldest in seen[-1]
    w.wait_idle()
    assert w.reply("Und?").finish().n_evicted_messages > 0


def test_the_search_for_how_many_exchanges_to_drop_takes_no_more_than_twice_the_measures_of_halving():
    """However uneven the exchanges, a guess that does not halve the range is followed by one that does."""
    shapes = random.Random(15)
    for _ in range(30):
        n = shapes.randint(2, 64)
        sizes = [shapes.choice((1, 1, 1, shapes.randint(1, 2000))) for _ in range(n)]
        messages = [{"role": "system", "content": "s"}] + [
            message for k, size in enumerate(sizes) for message in (
                {"role": "user", "content": f"q{k}"}, {"role": "assistant", "content": "y" * size})]
        state = ConversationState(messages=list(messages))
        eviction = Eviction(None, Config(), True, state, measure=lambda kept, prompted: 0)
        measures = [0]

        def length(state=state, measures=measures):
            measures[0] += 1
            return sum(len(message["content"]) for message in state.messages)

        after = [sum(len(m["content"]) for m in messages[:1] + messages[1 + 2 * k:])
                 for k in range(n)]
        for bound in sorted({at + step for at in after for step in (-1, 0)}):
            if after[0] <= bound or after[-1] > bound:
                continue                # nothing over, or every one that can go is not enough
            for at_least in (0, shapes.randint(0, n - 1)):
                state.messages[:] = messages
                measures[0] = 0
                eviction._drop_oldest_exchanges(length, bound, at_least, False)
                assert measures[0] <= 2 + 2 * math.ceil(math.log2(n)), (sizes, bound, at_least)


def test_beside_an_open_turn_an_eviction_brings_the_conversation_down_to_what_its_docstring_names():
    """Where the cache cannot shift, the bound is the room under ``limit`` after a short question, or after the open turn where that alone takes more, both measured with the generation prompt the next reply reads."""
    from types import SimpleNamespace
    from llama_chat.eviction import EVICT_TO_FRACTION_WITHOUT_SHIFT
    from llama_chat.messages import QUESTION_CUT_MARK

    def stub_length(messages, generation_prompt):
        return sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
                   for m in messages) + (5 if generation_prompt else 0)

    stub_rendering = SimpleNamespace(length=stub_length)
    call = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    shapes = random.Random(16)
    for _ in range(60):
        system = [{"role": "system", "content": "s" * shapes.randint(1, 30)}] if shapes.random() < 0.8 else []
        history = [message for k in range(shapes.randint(0, 3)) for message in (
            {"role": "user", "content": f"q{k}"}, {"role": "assistant", "content": "y" * shapes.randint(0, 40)})]
        asked = [{"role": "user", "content": "u" * shapes.randint(1, 40)}] if history or shapes.random() < 0.5 else []
        open_turn = asked + [{"role": "assistant", "content": "", "tool_calls": call},
                             {"role": "tool", "tool_call_id": "c1", "content": "r" * shapes.randint(0, 200)}]
        state = ConversationState(messages=system + history + open_turn)
        eviction = Eviction(stub_rendering, Config(), False, state)
        short = stub_length(system + [{"role": "user", "content": QUESTION_CUT_MARK + "?"}], True)
        for limit in range(1, 3 * stub_length(system + open_turn, True)):
            under = min(short, limit)
            bound = under + int((limit - under) * EVICT_TO_FRACTION_WITHOUT_SHIFT)
            alone = min(stub_length(system + open_turn, True), limit)
            beside = alone + int((limit - alone) * EVICT_TO_FRACTION_WITHOUT_SHIFT) if alone > bound else bound
            assert eviction.drop_bound(limit, False) == bound
            assert eviction.drop_bound(limit, True) == beside, (state.messages, limit)
