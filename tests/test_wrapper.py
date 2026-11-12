"""End-to-end wrapper tests using the model-free FakeContext.

These verify the two properties that matter most: (1) the wrapper only ever
prefills *new* tokens (never re-prefills survivors), and (2) the physical cache
stays exactly what the conversation holds it to be after every action,
including across evictions.
"""

import dataclasses
import json
import threading
import typing

import pytest

from llama_chat.config import Config, Sampling, generation_sampling
from llama_chat.tools import ToolCall, ToolRegistry
from llama_chat import TOOL_CALL_POLICIES, ChatWrapper, ContextOverflowError, ToolCallPolicy
from llama_chat.addon_files import SHIPPED_ADDONS, template_fingerprint
from llama_chat.generation import generate
from tests.chat_templates import STEMS, WEATHER, expected, fixture
from tests.fake_context import BOS, FakeContext
from tests.internals import read_fragments_as
from tests.invented_templates import (JOINS_USER_TURNS, PLAIN, REFUSES_A_TEXT,
                                      REFUSES_TWO_USER_TURNS)
from tests.wrappers import (assert_consistent, begun_wrapper, held_renders, held_text,
                            tool_wrapper, rows)
def _n_kept(wrapper):
    """Messages kept: everything but the system prompt."""
    return len(rows(wrapper)) - 1


def _n_exchanges(wrapper):
    """Exchanges the limit counts: each user message opens one, and what comes before the first is one more."""
    history = [row["role"] for row in rows(wrapper) if row["role"] != "system"]
    return history.count("user") + (1 if history and history[0] != "user" else 0)


def _as_rendered(w, fake):
    """The cache equals the template's own render of the conversation."""
    assert_consistent(w, fake)
    found = rows(w)
    rendered = held_renders(fake, [{"role": r["role"], "content": r["text"]} for r in found])
    assert fake.text_of(fake.cache) in rendered


# ----- a reply without text, and the tool call policy ----------------------
def test_a_reply_without_text_answers_the_newest_user_message():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    wrapper.inject("Wie spät ist es?")
    reply = wrapper.reply().finish()
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]
    assert reply.text == rows(wrapper)[-1]["text"]


def test_an_unknown_tool_call_policy_is_refused():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="tool_call_policy must be"):
        wrapper.reply("x", tool_call_policy="sometimes")


# ----- a reply the conversation cannot take ---------------------------------
def test_a_reply_the_conversation_cannot_take_is_refused_when_it_is_asked_for():
    """Refused by the call itself, before a stream is returned or read."""
    wrapper, fake = tool_wrapper()
    with pytest.raises(ValueError, match="call begin"):
        wrapper.reply("hallo")
    wrapper.begin("persona")
    with pytest.raises(ValueError, match="nothing to answer"):
        wrapper.reply()
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    for tool_call_policy in ("yield", "refuse"):
        with pytest.raises(ValueError, match="1 call"):
            wrapper.reply(tool_call_policy=tool_call_policy)
    with pytest.raises(ValueError, match="while a reply is open"):
        wrapper.reply("etwas anderes")
    wrapper.close()
    with pytest.raises(RuntimeError, match="closed"):
        wrapper.reply("hallo")


def test_a_conversation_changed_between_the_call_and_the_first_read_is_checked_again():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    wrapper.inject("frage")
    stream = wrapper.reply()                    # a question waits: taken
    wrapper.begin("persona")                    # as another thread might, before it is read
    with pytest.raises(ValueError, match="nothing to answer"):
        next(stream)


def test_asking_for_a_reply_while_another_runs_does_not_wait_for_it():
    wrapper, _ = tool_wrapper()
    wrapper.begin("persona")
    running = wrapper.reply("hallo")
    next(running)                               # holds the wrapper's lock
    asked = []
    asker = threading.Thread(target=lambda: asked.append(wrapper.reply("noch eine")),
                             daemon=True)
    asker.start()
    asker.join(timeout=5)
    running.close()
    assert len(asked) == 1 and asked[0].finish().text


# ----- begin ---------------------------------------------------------------
def test_begin_prefills_system_and_history_once():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=200, evict_above_fraction=1.0), context=fake)
    w.begin("sys", [("user", "hi"), ("assistant", "yo")])

    roles = [s["role"] for s in rows(w)]
    assert roles == ["system", "user", "assistant"]
    # One prefill call for the whole conversation == cheapest prefill.
    assert len(fake.prefill_calls) == 1


def test_begin_drops_oldest_history_beyond_threshold():
    fake = FakeContext(PLAIN)
    # The system turn "<[turn>system\ns<turn]>\n" and BOS are well under budget;
    # pick a tiny threshold so only the newest exchange fits.
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.05), context=fake)
    w.begin("s", [("user", "old message that is long"), ("assistant", "x"), ("user", "y")])

    # The oldest exchange is dropped whole: the kept history opens on a user turn.
    assert [s["text"] for s in rows(w)] == ["s", "y"]


def test_begin_has_bos_only_at_start():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "u")])
    # Exactly one BOS sentinel, at position 0.
    assert fake.cache[0] == BOS
    assert fake.cache.count(BOS) == 1


# ----- inject --------------------------------------------------------------
def test_inject_appends_without_generating():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    before = w.total_tokens
    n_evicted_messages = w.inject("some context", role="user")
    assert n_evicted_messages == 0
    assert rows(w)[-1]["role"] == "user"
    assert w.total_tokens > before
    # No generation happened.
    assert fake.samplings == []


def test_inject_evicts_oldest_to_fit():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.06), context=fake)
    w.begin("s")
    w.inject("first injected chunk", role="user")
    n_evicted_messages = w.inject("second injected chunk", role="user")
    assert n_evicted_messages >= 1
    roles = [s["role"] for s in rows(w)]
    assert roles[0] == "system"  # system never evicted


# ----- replying without tools ----------------------------------------------
def test_a_refusing_reply_prefills_only_new_tokens_and_reuses_context():
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, max_reply_tokens=50), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "hello")])
    prefilled_after_begin, held = fake.total_prefilled(), len(fake.cache)
    mark = len(fake.prefill_calls)

    reply = w.reply("what's up?").finish()

    # Every decode is counted, and none reaches back into what was held.
    new_prefill = fake.total_prefilled() - prefilled_after_begin
    assert new_prefill == reply.n_prefilled_tokens
    assert all(start >= held for start, _, _ in fake.prefill_calls[mark:])
    assert reply.n_generated_tokens == 4
    assert reply.stop_reason == "end_of_turn"
    # Assistant turn recorded for reuse next turn.
    assert rows(w)[-1]["role"] == "assistant"


def test_the_request_and_its_opening_tag_are_one_decode():
    """The prompt reaches the context in a single decode, closer aside.

    The request text and the assistant-open tag used to be two prefills, each
    paying a decode's fixed overhead on the one prefill a user is actually
    waiting through. Splitting them again would be invisible to every other
    assertion here, so it gets its own.
    """
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    mark = len(fake.prefill_calls)

    reply = w.reply("hi").finish()

    # The question and its generation prompt, then after the reply the turn's
    # closing and the next question's opening, decoded ahead.
    prompt_call, ahead_call = fake.prefill_calls[mark:]
    assert prompt_call[2] is True                 # logits: generation follows
    assert ahead_call[2] is False
    assert prompt_call[1] + ahead_call[1] == reply.n_prefilled_tokens


def test_a_second_generation_does_not_reprefill_history():
    fake = FakeContext(PLAIN, gen_tokens=3)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    w.reply("first").finish()
    mark, held, calls = fake.total_prefilled(), len(fake.cache), len(fake.prefill_calls)
    reply = w.reply("second").finish()
    # The second generation decodes only its own prompt and what follows its reply.
    assert fake.total_prefilled() - mark == reply.n_prefilled_tokens
    assert all(start >= held - len("reply<turn]>\n<[turn>") for start, _, _ in
               fake.prefill_calls[calls:])


# ----- the tokens a reply was generated from --------------------------------
def _held_at_each_generation(fake):
    """The cache's length and the decodes made so far, recorded before each generation's first token."""
    held = []
    fake.before_reply = lambda _: held.append((len(fake.cache), len(fake.prefill_calls)))
    return held


def test_a_reply_reports_the_tokens_its_question_left_in_the_cache():
    w, fake = begun_wrapper()
    held = _held_at_each_generation(fake)
    reply = w.reply("what's up?").finish()
    assert reply.n_prompt_tokens == held[-1][0]


def test_a_reply_to_an_injected_question_reports_the_tokens_the_question_left():
    w, fake = begun_wrapper()
    w.inject("what's up?")
    decodes = len(fake.prefill_calls)
    held = _held_at_each_generation(fake)
    reply = w.reply().finish()
    assert held[-1][1] == decodes           # its generation prompt was decoded with it
    assert reply.n_prompt_tokens == held[-1][0]


def test_a_reply_that_decodes_its_own_opening_reports_the_opening_too():
    w, fake = begun_wrapper()
    w.inject(" what's up? ")                # space around it: decoded without the opening
    decodes = len(fake.prefill_calls)
    held = _held_at_each_generation(fake)
    reply = w.reply().finish()
    assert held[-1][1] > decodes            # the reply decoded its opening itself
    assert reply.n_prompt_tokens == held[-1][0]


def test_a_reply_resumed_after_a_result_reports_the_tokens_the_result_left():
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    wrapper.inject_tool_result(ToolCall("clock", {}), {"time": "09:00"})
    held = _held_at_each_generation(fake)
    reply = wrapper.reply(tool_call_policy="yield").finish()
    assert reply.n_prompt_tokens == held[-1][0]


def test_a_request_that_ran_calls_reports_its_last_generations_tokens():
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="The time.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    held = _held_at_each_generation(fake)
    reply = wrapper.reply("los", tool_call_policy=registry).finish()
    assert len(held) == 2 and held[0][0] < held[1][0]
    assert reply.n_prompt_tokens == held[-1][0]


# ----- streaming -----------------------------------------------------------
def test_a_refusing_reply_stream_yields_incremental_deltas():
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    deltas = list(w.reply("hi"))
    assert "".join(deltas) == "abcd"
    assert rows(w)[-1]["role"] == "assistant"


def test_a_refusing_reply_drains_its_own_stream():
    fake = FakeContext(PLAIN, gen_tokens=3, gen_text="xyz")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    reply = w.reply("hi").finish()
    assert reply.text == "xyz"
    assert reply.n_generated_tokens == 3


def test_an_early_close_keeps_cache_consistent():
    # Consume only the first delta, then abandon the stream (close the generator).
    fake = FakeContext(PLAIN, gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")

    stream = w.reply("hi")
    first = next(stream)
    assert first == "a"
    stream.close()  # stop early

    # The assistant turn was still recorded, and the cache equals what is recorded:
    # exactly the tokens that reached the cache were recorded, no more.
    assert rows(w)[-1]["role"] == "assistant"
    assert_consistent(w, fake)

    # A following generation proceeds normally on the consistent cache.
    reply = w.reply("again").finish()
    assert reply.stop_reason == "end_of_turn"


def test_cancel_event_interrupts_midstream():
    fake = FakeContext(PLAIN, gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")

    cancel = threading.Event()
    stream = w.reply("hi", cancel=cancel)
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
    assert rows(w)[-1]["role"] == "assistant"
    assert_consistent(w, fake)
    # The spent event is scoped to that call; the next generation runs normally.
    assert w.reply("again").finish().stop_reason == "end_of_turn"


def test_cancel_set_before_start_yields_an_empty_interrupted_reply():
    fake = FakeContext(PLAIN, gen_tokens=5)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    cancel = threading.Event()
    cancel.set()
    reply = w.reply("hi", cancel=cancel).finish()
    assert reply.stop_reason == "interrupted"
    assert reply.n_generated_tokens == 0
    # A reply with nothing in it is no reply: only the request stays.
    assert rows(w)[-1]["role"] == "user"


def test_a_stop_string_ends_the_reply_and_is_trimmed_from_it():
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="helloSTOPx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin("s")
    with w.reply("hi", stop=["STOP"]) as stream:
        shown = "".join(stream)
    assert shown == stream.reply.text == "hello"
    assert stream.reply.stop_reason == "stop_string"   # not "interrupted"


def test_only_text_that_could_begin_a_stop_string_is_held_back():
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="heSlSTOPx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin("s")
    starts = []
    fake.before_reply = lambda text: starts.append(len(fake.cache))
    # Each delta, and how many tokens had been generated when it came.
    shown = [(delta, len(fake.cache) - starts[0]) for delta in w.reply("hi", stop=["STOP", "lo"])]
    assert shown == [("h", 1), ("e", 2), ("S", 4), ("l", 5)]


def test_the_stop_texts_are_the_ones_given_when_the_reply_was_asked_for():
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="helloSTOPx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin("s")
    stop = ["STOP"]
    stream = w.reply("hi", stop=stop)
    stop.clear()                    # before the first read, which starts the generation
    assert stream.finish().text == "hello"


@pytest.mark.parametrize("options", [
    {"sampling": {"temperature": 0.1}}, {"stop": [1]}, {"stop": "END"}, {"stop": 5},
    {"cancel": True}], ids=["sampling-dict", "stop-number", "stop-str", "stop-int", "cancel-bool"])
def test_a_request_option_of_the_wrong_kind_is_refused_before_the_question_is_written(options):
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="helloENDx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin("s")
    before = w.conversation()
    with pytest.raises(TypeError):
        w.reply("hi", **options)
    assert w.conversation() == before
    assert w.reply("hi", stop=["END"]).finish().text == "hello"


def test_stop_texts_are_a_request_option_and_no_setting():
    assert "stop_strings" not in {field.name for field in dataclasses.fields(Config)}


def test_generator_close_marks_interrupted():
    # stream.close() records the same stop reason on the accumulator (the Reply
    # itself is unreachable after close(): Python offers no return channel).
    from llama_chat.cache_context import GenerationAccumulator

    fake = FakeContext(PLAIN, gen_tokens=5, gen_text="abcde")
    acc = GenerationAccumulator()
    g = generate(fake, 0, 5, out=acc)
    next(g)
    g.close()
    assert acc.stop_reason == "interrupted"


def test_generation_is_capped_to_the_context_size():
    fake = FakeContext(PLAIN, gen_tokens=100)  # would generate 100 if uncapped
    w = ChatWrapper(Config(context_size=110, evict_above_fraction=1.0, max_reply_tokens=100), context=fake)
    w.begin("s")
    reply = w.reply("hello").finish()
    assert w.total_tokens <= 110
    assert reply.n_generated_tokens < 100


def test_a_reply_runs_until_the_context_is_full_unless_a_cap_is_set():
    """A long thought counts against the cap too, so a default cap could leave no answer."""
    fake = FakeContext(PLAIN, gen_tokens=3000, gen_text="x" * 3000)
    w = ChatWrapper(Config(context_size=4096), context=fake)
    w.begin("s")
    reply = w.reply("hello").finish()
    assert reply.n_generated_tokens == 3000 and reply.stop_reason == "end_of_turn"
    assert w.reply("again", max_reply_tokens=10).finish().n_generated_tokens == 10


def test_the_cache_rests_under_the_threshold_after_a_generation():
    # The reply is appended first, then older turns are trimmed, so the cache
    # rests at or below threshold once the generation returns.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin("s")
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
        assert w.total_tokens <= w.config.evict_above_tokens  # rests under threshold
        assert rows(w)[0]["role"] == "system"        # system survives
        assert [r["role"] for r in rows(w)][1:2] in ([], ["user"])  # opens on a user turn
        assert_consistent(w, fake)


def test_turn_closes_after_the_eviction_not_before():
    # An eviction's K-shift lands on whatever decodes next, so the closer is
    # decoded last and carries it instead of the next generation's prefill.
    # The assertion is what pins that ordering. See design_history.md,
    # "How eviction works".
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin("s")

    evicting = 0
    for i in range(8):
        n_evictions = fake.edit_count("shift")
        reply = w.reply(f"message number {i} with some length").finish()
        if reply.n_evicted_messages:
            evicting += 1
            # The eviction landed, and the closer is the turn's last prefill.
            assert fake.edit_count("shift") > n_evictions
            last = fake.prefill_calls[-1]
            assert last[0] + last[1] == w.total_tokens   # closer at the tail
            assert last[2] is False                      # no logits wanted
        assert_consistent(w, fake)
    assert evicting


# ----- the exchange limit, OR'd with the token threshold -------------------
def test_a_generation_evicts_down_to_the_exchange_limit():
    # Far under the token threshold, so the exchange count is the only trigger.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=2), context=fake)
    w.begin("s")
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
    w.begin("s")
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
        w.begin("s")
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
        w.begin("s")
        for i in range(8):
            w.reply(f"message number {i} with some length").finish()
        return rows(w)

    assert snapshot_under() == snapshot_under(max_history_exchanges=100)


def test_inject_evicts_to_leave_room_under_the_exchange_limit():
    # inject evicts before appending, so the exchange the arriving message opens
    # counts against the limit while it is still absent from the conversation.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=2), context=fake)
    w.begin("s", [("user", "a"), ("assistant", "b"), ("user", "c")])
    assert _n_exchanges(w) == 2
    # The oldest exchange goes whole, so the survivors open on a user turn.
    assert w.inject("reference material") == 2
    assert [r["text"] for r in rows(w)][1:] == ["c", "reference material"]


@pytest.mark.parametrize("history, kept", [
    ([("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")], ["c", "d"]),
    ([("assistant", "a"), ("user", "b"), ("assistant", "c"), ("assistant", "d")],
     ["b", "c", "d"]),
    ([("user", "a"), ("assistant", "b"), ("user", "c")], ["c"]),
])
def test_begin_prefills_no_more_than_the_exchange_limit(history, kept):
    # Without the cap, begin would prefill history the first turn then evicts.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=1), context=fake)
    assert w.begin("s", history) == len(kept)
    assert [r["text"] for r in rows(w)] == ["s"] + kept       # the newest exchange, whole


def test_a_limit_of_one_keeps_the_newest_exchange_between_turns():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000, max_history_exchanges=1), context=fake)
    w.begin("s")
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
    w.begin("s", [("user", "old"), ("assistant", "older")])
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
    w.begin("s")
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
    # Cache cannot shift: eviction drops oldest from bookkeeping and re-prefills.
    fake = FakeContext(PLAIN, gen_tokens=2, can_shift=False)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.35), context=fake)
    w.begin("s")
    resets_after_begin = fake.resets
    for i in range(8):
        w.reply(f"message number {i} with some length").finish()
        roles = [s["role"] for s in rows(w)]
        assert roles[0] == "system"             # system survives
        assert w.total_tokens <= w.config.context_size   # never overflows
        assert_consistent(w, fake)             # the cache as recorded after it is brought back
    assert fake.edit_count("shift") == 0             # shift path never used
    assert fake.resets + fake.edit_count("restore") > resets_after_begin   # brought back instead


def test_a_message_that_cannot_fit_is_refused_before_anything_is_evicted():
    # Rejecting an unfittable message must not evict the history first.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.5), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "yo")])
    before = rows(w)
    with pytest.raises(ValueError):
        w.inject("x" * 500)
    assert rows(w) == before


def test_every_refusal_for_want_of_room_says_what_to_change():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, min_reply_tokens=32),
                    context=fake)
    w.begin("s")
    with pytest.raises(ContextOverflowError, match="shorten the request.*min_reply_tokens"):
        w.reply("x" * 600).finish()
    with pytest.raises(ContextOverflowError, match="shorten the message.*evict_above_fraction"):
        w.inject("x" * 600)
    w.inject("x" * 470)                         # under the threshold, not under room for a reply
    with pytest.raises(ContextOverflowError,
                       match="lower evict_above_fraction or min_reply_tokens"):
        w.reply().finish()


def test_a_history_no_question_follows_is_refused_where_it_cannot_fit():
    """Nothing in it can be evicted, since an eviction drops up to the next user message."""
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=0.8), context=fake)
    w.begin("s", [("user", "hi"), ("assistant", "yo")])
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(ContextOverflowError, match="no user message follows"):
        w.begin("s", [("assistant", "x" * 200)] * 3)
    assert w.conversation() == before and fake.cache == held
    # With a later question, the replies before it are evicted instead.
    w.begin("s", [("assistant", "x" * 200)] * 3 + [("user", "q"), ("assistant", "r")])
    assert w.conversation()["messages"] == [{"role": "user", "content": "q"},
                                            {"role": "assistant", "content": "r"}]


@pytest.mark.parametrize("history", [[], [("user", "q")]], ids=["no-question", "after-a-question"])
def test_replies_injected_in_a_row_are_refused_once_they_no_longer_fit(history):
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=0.8), context=fake)
    w.begin("s", history)
    with pytest.raises(ContextOverflowError, match="no user message follows"):
        for _ in range(8):
            before, held = w.conversation(), list(fake.cache)
            w.inject("y" * 150, role="assistant")
            assert w.total_tokens <= w.config.evict_above_tokens
    assert w.conversation() == before and fake.cache == held
    w.inject("Und jetzt?")                      # a question lets them go
    w.inject("y" * 150, role="assistant")
    assert w.total_tokens <= w.config.evict_above_tokens


# ----- construction / lifecycle --------------------------------------------
def test_the_config_cannot_be_changed_once_a_wrapper_runs_with_it():
    """A change would not reach the context, already built with the old size."""
    w = ChatWrapper(Config(context_size=800), context=FakeContext(PLAIN))
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.config.context_size = 99


def test_the_package_exports_the_one_protocol_a_wrapper_takes():
    """The model reader is a part of the cache context, which no wrapper takes alone."""
    import llama_chat
    assert "CacheContext" in llama_chat.__all__
    assert "ModelReader" not in llama_chat.__all__ and not hasattr(llama_chat, "ModelReader")


def test_every_member_of_the_cache_context_protocol_says_what_a_caller_relies_on():
    """An implementer reads the protocol, not the stand-in."""
    from llama_chat.cache_context import CACHE_CONTEXT, CacheContext
    member = lambda name: getattr(CacheContext, name)
    undocumented = [name for name in CACHE_CONTEXT
                    if not ((member(name).fget if isinstance(member(name), property)
                             else member(name)).__doc__ or "").strip()]
    assert undocumented == []


def test_the_readme_imports_only_exported_names_and_names_every_one():
    """Every name a caller is told to take from the package is exported, and every export is told of."""
    import re
    from pathlib import Path
    import llama_chat
    readme = (Path(llama_chat.__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    told = set(re.findall(r"`(\w+)`,? from `(llama_chat[\w.]*)`", readme))
    for module, imported in re.findall(r"from (llama_chat[\w.]*) import ([\w, ]+)", readme):
        told |= {(name.strip(), module) for name in imported.split(",")}
    assert told and all(module == "llama_chat" and name in llama_chat.__all__
                        for name, module in told)
    assert [name for name in llama_chat.__all__ if f"`{name}" not in readme] == []


def test_begin_rejects_system_prompt_over_threshold():
    fake = FakeContext(PLAIN)
    # threshold = 100 tokens; a 100-char system prompt with its tags can never fit.
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.1), context=fake)
    with pytest.raises(ValueError, match="system prompt"):
        w.begin("x" * 100)
    # Nothing was prefilled by the refused begin; a fitting one then works.
    assert fake.cache == []
    w.begin("s")
    assert rows(w)[0]["role"] == "system"


def test_close_is_idempotent_and_blocks_actions():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s")
    w.close()
    w.close()  # second close is a no-op
    with pytest.raises(RuntimeError, match="closed"):
        w.begin("s")
    with pytest.raises(RuntimeError, match="closed"):
        w.inject("ctx")
    with pytest.raises(RuntimeError, match="closed"):
        w.reply("hi").finish()
    with pytest.raises(RuntimeError, match="closed"):
        w.declare_tools([])     # even with nothing to declare


@pytest.mark.parametrize("failing", ["ResolvedTemplates", "RenderedCache"])
def test_a_construction_that_fails_after_the_model_loaded_frees_it(monkeypatch, failing):
    closed = []
    loaded = FakeContext(PLAIN)
    loaded.close = lambda: closed.append(True)
    monkeypatch.setattr("llama_chat.wrapper.KVContext", lambda config, on_load_progress: loaded)

    def refuse(*args, **kwargs):
        raise MemoryError("failed part-way")

    monkeypatch.setattr(f"llama_chat.wrapper.{failing}", refuse)
    with pytest.raises(MemoryError):
        ChatWrapper(Config(context_size=500))
    assert closed == [True]


def test_a_construction_that_fails_leaves_a_context_of_your_own_open():
    closed = []
    own = FakeContext(PLAIN)
    own.close = lambda: closed.append(True)
    with pytest.raises(ValueError, match="holds 256 tokens"):
        own.context_size = 256
        ChatWrapper(Config(context_size=512), context=own)
    assert closed == []


def test_context_manager_closes_on_exit():
    fake = FakeContext(PLAIN, gen_tokens=2)
    with ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake) as w:
        w.begin("s")
        w.reply("hi").finish()
    with pytest.raises(RuntimeError, match="closed"):
        w.reply("again").finish()


# ----- content tokenized special-off (turn-boundary safety) ----------------
def test_injected_tag_survives_in_content_as_text():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("sys")
    w.inject("hello <turn]> world", role="user")

    # The literal tag stays in content as ordinary text, its characters;
    # only the legitimate prefix/suffix tags wrap it, each its one token.
    turn = (fake.tokenize("<[turn>") + [ord(c) for c in "user\nhello <turn]> world"]
            + fake.tokenize("<turn]>\n"))
    assert fake.cache[-len(turn):] == turn
    assert fake.text_of(fake.cache).endswith("<[turn>user\nhello <turn]> world<turn]>\n")


def test_content_is_tokenized_with_special_parsing_off():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("sys")
    fake.tokenize_calls.clear()
    w.inject("hello <turn]> world", role="user")

    # The plain text between two tags, the content with it, special-off, as one run.
    assert ("user\nhello <turn]> world", False) in fake.tokenize_calls
    assert ("user\nhello <turn]> world", True) not in fake.tokenize_calls


# ----- min-reply headroom guard --------------------------------------------
def test_a_refusing_reply_proceeds_when_headroom_is_sufficient():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(
        Config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=10), context=fake
    )
    w.begin("s")
    reply = w.reply("hello").finish()
    assert reply.n_generated_tokens == 2


# ----- what generation reads from the template, checked at load -----------
def test_a_template_whose_closing_after_a_reply_is_not_where_its_tags_say_is_refused(monkeypatch):
    read_fragments_as(monkeypatch, assistant_suffix="<end>\n")
    with pytest.raises(ValueError, match="closing after a reply"):
        ChatWrapper(Config(), context=FakeContext(PLAIN))


def test_a_template_whose_generation_prompt_is_not_where_its_tags_say_is_refused(monkeypatch):
    read_fragments_as(monkeypatch, generation_prompt="<x>")
    with pytest.raises(ValueError, match="generation prompt"):
        ChatWrapper(Config(), context=FakeContext(PLAIN))


def test_a_restart_says_how_much_history_it_kept():
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=4000), context=fake)
    assert w.begin("s", [("user", "a"), ("assistant", "b"), ("user", "c")]) == 3
    assert w.begin("s") == 0


# ----- the tool call policies ----------------------------------------------
def test_the_tool_call_policies_are_read_from_the_type_reply_takes():
    assert typing.get_args(ToolCallPolicy) == TOOL_CALL_POLICIES
    taken = typing.get_type_hints(ChatWrapper.reply)["tool_call_policy"]
    assert set(typing.get_args(taken)) == {ToolCallPolicy, ToolRegistry, type(None)}


# ----- sampling per request ------------------------------------------------
@pytest.mark.parametrize("tool_call_policy", [*TOOL_CALL_POLICIES, "registry"])
def test_every_tool_call_policy_hands_its_sampling_to_the_generation(tool_call_policy):
    w, fake = begun_wrapper()
    sampling = Sampling(temperature=0.1)
    w.reply("hi", tool_call_policy=ToolRegistry() if tool_call_policy == "registry" else tool_call_policy,
            sampling=sampling).finish()
    assert fake.samplings == [generation_sampling(w.config, sampling)]
    assert fake.samplings[0].temperature == 0.1


def test_a_stream_and_a_resume_hand_it_over_as_well():
    w, fake = begun_wrapper()
    sampling = Sampling(top_p=0.5)
    list(w.reply("hi", sampling=sampling))
    list(w.reply("again", tool_call_policy="yield", sampling=sampling))
    assert fake.samplings == [generation_sampling(w.config, sampling)] * 2


def _shipped_sampling() -> dict:
    """The sampling each shipped add-on file sets, by the test template it names."""
    by_fingerprint = {}
    for path in SHIPPED_ADDONS.glob("*.json"):
        found = json.loads(path.read_text())
        by_fingerprint.update(dict.fromkeys(found["templates"], found.get("sampling")))
    return {stem: by_fingerprint[fingerprint] for stem in STEMS
            if by_fingerprint.get(fingerprint := template_fingerprint(fixture(stem)["chat_template"]))}


@pytest.mark.parametrize("stem, sampling", sorted(_shipped_sampling().items()))
def test_a_context_of_your_own_gets_the_add_on_file_that_names_its_template(stem, sampling):
    # The shipped file names this template; nothing the model states is used beside it.
    fake = FakeContext(fixture(stem), ["Fine."])
    w = ChatWrapper(Config(context_size=4096, min_reply_tokens=8), context=fake)
    assert {key: getattr(w.config, key) for key in sampling} == sampling
    w.begin("")
    w.reply("hi").finish()
    assert {key: getattr(fake.samplings[0], key) for key in sampling} == sampling


def test_a_context_of_your_own_gets_what_the_model_recommends():
    # A model that states its own and has no add-on file.
    stating = FakeContext(PLAIN)
    stating.recommended = {"temperature": 0.4, "min_p": 0.07}
    w = ChatWrapper(Config(context_size=512), context=stating)
    assert (w.config.temperature, w.config.min_p) == (0.4, 0.07)


def test_every_generation_is_handed_every_sampling_value():
    w, fake = begun_wrapper()
    w.reply("hi").finish()
    assert all(getattr(fake.samplings[0], f.name) is not None
               for f in dataclasses.fields(Sampling) if f.name != "seed")


def test_what_the_model_can_do_is_read_from_the_conversation_as_the_newest_finished_change_left_it():
    from tests.invented_templates import REASONING, TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN

    # The template for tools can reason, the default template cannot.
    fake = FakeContext({**TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN,
                        "tool_use_template": REASONING["chat_template"]})
    w = ChatWrapper(Config(context_size=512), context=fake)
    w.begin("s")
    seen = []
    prefill = fake.prefill

    def read_while_beginning(*args, **kwargs):
        seen.append((w.can_reason, w.can_turn_off_reasoning, w.can_take_system_prompt))
        raise RuntimeError("the cache refused the batch")

    fake.prefill = read_while_beginning
    with pytest.raises(RuntimeError, match="refused the batch"):
        w.begin("s", tools=[])
    fake.prefill = prefill
    assert seen and seen[0] == (w.can_reason, w.can_turn_off_reasoning, w.can_take_system_prompt)
    assert w.can_reason is False


# ----- what an action is given, refused where it enters ---------------------
def test_a_message_that_is_not_text_is_refused_where_it_enters_naming_it():
    w, fake = begun_wrapper()
    w.reply("hi").finish()
    before, held = w.conversation(), list(fake.cache)
    for action, what in ((lambda: w.inject(42), "injected message"),
                         (lambda: w.reply(42), "request"),
                         (lambda: w.begin(None), "system prompt"),
                         (lambda: w.begin("s", [("user", 42)]), "history message"),
                         (lambda: w.begin("s", [{"role": "user", "content": None}]),
                          "history message")):
        with pytest.raises(TypeError, match=f"{what}.*str, not"):
            action()
    assert w.conversation() == before and fake.cache == held


@pytest.mark.parametrize("max_calls", [-1, 1.5, True])
def test_a_ceiling_on_calls_that_is_no_count_is_refused(max_calls):
    w, _ = tool_wrapper()
    w.begin("s")
    with pytest.raises(ValueError, match="max_calls"):
        w.reply("x", tool_call_policy=ToolRegistry(), max_calls=max_calls)


def test_a_context_of_your_own_whose_methods_take_other_arguments_is_refused_at_the_start():
    class _OtherArguments(FakeContext):
        def prefill(self, tokens, start, want_logits):
            super().prefill(tokens, start, want_logits)

        def sampled_tokens(self, sampling):
            return super().sampled_tokens(sampling, ())

    with pytest.raises(TypeError, match=r"prefill.*sampled_tokens"):
        ChatWrapper(Config(context_size=500), context=_OtherArguments(PLAIN))

    class _Generous(FakeContext):
        def prefill(self, *args, **kwargs):
            super().prefill(*args, **kwargs)

    ChatWrapper(Config(context_size=500), context=_Generous(PLAIN)).begin("s")


@pytest.mark.parametrize("role, cause", [("system", "begin"), ("tool", "inject_tool_result"),
                                         ("human", "'human'")])
def test_inject_refuses_a_message_of_another_role_naming_where_it_goes(role, cause):
    """The system prompt belongs to begin, a result to inject_tool_result, and no other role is written."""
    w, fake = begun_wrapper()
    w.reply("hi").finish()
    before, held = w.conversation(), list(fake.cache)
    with pytest.raises(ValueError, match=cause):
        w.inject("Notiz", role)
    assert w.conversation() == before and fake.cache == held


# ----- two user turns in a row ---------------------------------------------
def test_a_request_after_a_reply_without_an_answer_closes_the_user_turn_as_the_template_does():
    w, fake = begun_wrapper(JOINS_USER_TURNS, gen_tokens=0)
    w.reply("eins").finish()
    assert [m["role"] for m in rows(w)] == ["system", "user"]
    fake.gen_tokens = 5
    w.reply("zwei").finish()
    _as_rendered(w, fake)


def test_a_template_that_refuses_two_user_turns_keeps_a_reply_without_an_answer():
    w, fake = begun_wrapper(REFUSES_TWO_USER_TURNS, gen_tokens=0)
    w.reply("eins").finish()
    found = [(m["role"], m["text"]) for m in rows(w)]
    assert found[1:] == [("user", "eins"), ("assistant", "")]
    fake.gen_tokens = 5
    w.reply("zwei").finish()
    _as_rendered(w, fake)


@pytest.mark.parametrize("stem", [stem for stem in STEMS
                                  if not expected(fixture(stem), "writes_empty_reply")])
@pytest.mark.parametrize("nothing", ["cancelled-at-once", "ends-at-once", "cancelled-by-a-tool"])
def test_a_reply_that_produced_nothing_the_template_cannot_write_takes_its_question_back(
        stem, nothing):
    """These templates write no empty reply, so the request is taken back whole and the next question is answered."""
    from tests.fake_context import ScriptedCall
    fake = FakeContext(fixture(stem), gen_tokens=7, gen_text="Antwort")
    w = ChatWrapper(Config(context_size=8192, evict_above_fraction=1.0), context=fake)
    cancel, options = threading.Event(), {}
    if nothing == "cancelled-by-a-tool":
        registry = ToolRegistry()
        registry.tool(lambda city: (cancel.set(), "18 Grad")[1], description="Wetter.",
                      name=WEATHER.name, parameters=WEATHER.parameters)
        w.begin("", tools=registry.specs())
        fake.replies = [ScriptedCall('{"name": "get_weather", "arguments": {"city": "Bonn"}}')]
        options = dict(tool_call_policy=registry, cancel=cancel)
    else:
        w.begin("")
        if nothing == "cancelled-at-once":
            cancel.set()
            options = dict(cancel=cancel)
        else:
            fake.gen_tokens = 0
    w.reply("Erste Frage", **options).finish()
    assert w.conversation()["messages"] == [] and not w.turn_open
    fake.gen_tokens = 7
    assert w.reply("Zweite Frage").finish().text == "Antwort"
    assert [m["text"] for m in rows(w)] == ["Zweite Frage", "Antwort"]


def test_a_reply_the_template_refuses_to_write_takes_its_question_back():
    fake = FakeContext(REFUSES_A_TEXT, replies=["refused", "fine"])
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "eins"), ("assistant", "zwei")])
    before = w.conversation()
    assert w.reply("hi").finish().text == "refused"
    assert w.conversation() == before
    assert w.reply("next?").finish().text == "fine"
    assert [m["text"] for m in rows(w)][-2:] == ["next?", "fine"]


def test_an_injected_user_turn_after_a_user_turn_is_joined_or_refused():
    w, fake = begun_wrapper(JOINS_USER_TURNS)
    w.inject("eins")
    w.inject("zwei")
    _as_rendered(w, fake)
    w, fake = begun_wrapper(REFUSES_TWO_USER_TURNS)
    w.inject("eins")
    before = list(fake.cache)
    with pytest.raises(ValueError, match="two user turns"):
        w.inject("zwei")
    assert fake.cache == before and len(rows(w)) == 2


def test_a_history_with_two_user_turns_in_a_row_is_joined_or_refused():
    fake = FakeContext(JOINS_USER_TURNS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin("s", [("user", "eins"), ("user", "zwei"), ("assistant", "drei")])
    _as_rendered(w, fake)
    fake = FakeContext(REFUSES_TWO_USER_TURNS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    with pytest.raises(ValueError, match="two user turns"):
        w.begin("s", [("user", "eins"), ("user", "zwei")])


def test_a_refused_history_is_explained_by_what_would_be_kept_of_it():
    """Two user turns in a row that would be dropped are not why the rest is refused."""
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext(REFUSES_A_TEXT))
    with pytest.raises(ValueError, match="refuses to write this conversation"):
        w.begin("s", [("user", "eins"), ("user", "zwei"), ("assistant", "drei"),
                      ("user", "refused"), ("assistant", "vier")])


def test_evicting_the_first_of_two_user_turns_takes_its_join_with_it():
    fake = FakeContext(JOINS_USER_TURNS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, max_history_exchanges=1),
                    context=fake)
    w.begin("s")
    w.inject("eins")
    w.inject("zwei")
    assert [m["text"] for m in rows(w)] == ["s", "zwei"]
    _as_rendered(w, fake)


def test_a_config_larger_than_the_cache_context_is_refused_at_the_start():
    small = FakeContext(PLAIN)
    small.context_size = 256
    with pytest.raises(ValueError, match="holds 256 tokens"):
        ChatWrapper(Config(context_size=512), context=small)


# ----- what the conversation records -----------------------------------------
def test_the_conversation_holds_only_what_was_recorded_an_early_closed_reply_included():
    fake = FakeContext(PLAIN, gen_tokens=5)
    w = ChatWrapper(Config(context_size=120, evict_above_fraction=1.0, min_reply_tokens=10),
                    context=fake)
    w.begin("s")
    with pytest.raises(ContextOverflowError):
        w.inject("x" * 200)                      # refused: never recorded
    with pytest.raises(ContextOverflowError):
        w.reply("y" * 200).finish()
    stream = w.reply("hi")
    next(stream)
    stream.close()                               # closed early: recorded
    assert w.conversation()["messages"] == [{"role": "user", "content": "hi"},
                                            {"role": "assistant", "content": "r"}]
    assert rows(w)[-1] == {"role": "assistant", "text": "r"}


def test_config_holds_the_sampling_the_model_supplies_whoever_built_the_context(monkeypatch):
    given = Config(context_size=512)

    class _Loaded(FakeContext):
        def __init__(self, config, on_load_progress=None):
            super().__init__(PLAIN)
            self.recommended = {"temperature": 0.6, "top_p": 0.9}

    monkeypatch.setattr("llama_chat.wrapper.KVContext", _Loaded)
    loaded = ChatWrapper(given).config
    own = ChatWrapper(given, context=_Loaded(given)).config
    assert loaded == own
    assert (own.temperature, own.top_p, own.context_size) == (0.6, 0.9, 512)


@pytest.mark.parametrize("question", ["zwei", " zwei "], ids=["prepared-ahead", "not-prepared"])
def test_a_question_sees_the_same_history_whether_or_not_it_was_prepared_ahead(question):
    """At the exchange limit a question opens one more exchange while its reply runs, and the limit holds again once it settles."""
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=2000, evict_above_fraction=1.0, max_history_exchanges=1),
                    context=fake)
    w.begin("s")
    w.reply("eins").finish()
    seen = []
    fake.before_reply = seen.append
    w.reply(question).finish()
    assert "eins" in seen[0]
    assert _n_exchanges(w) == 1 and rows(w)[1]["text"] == question
    assert_consistent(w, fake)
