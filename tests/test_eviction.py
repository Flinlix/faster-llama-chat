"""The two eviction limits, the token threshold and the exchange limit: which exchanges go, when, and what a cache that cannot shift does instead."""


import pytest

from llama_chat.config import Config
from llama_chat import ChatWrapper, ContextOverflowError
from llama_chat.messages import QUESTION_CUT_MARK
from tests.chat_messages import opened, after_system
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import PLAIN
from tests.wrappers import assert_consistent, held_text, rows


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


def test_a_long_question_is_cut_to_the_room_under_the_threshold_not_to_the_whole_context():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=0.5, min_reply_tokens=32),
                    context=fake)
    w.begin(opened("s"))
    assert w.reply("x" * 2000 + " Frage?").finish().text
    asked = [m for m in after_system(w.conversation()["messages"]) if m["role"] == "user"][-1]
    beside = len(fake.tokenize(fake.stated_render(opened("s", [asked]), generation_prompt=True)))
    assert w.config.evict_above_tokens - 8 <= beside <= w.config.evict_above_tokens
