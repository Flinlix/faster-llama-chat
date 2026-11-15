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
from llama_chat.tool_registry import ToolRegistry
from llama_chat.tools import ToolCall, ToolSpec
from llama_chat import (TOOL_CALL_POLICIES, ChatWrapper, ContextOverflowError,
                        InvalidArgumentError, ToolCallPolicy, UnsupportedError)
from llama_chat.addon_files import SHIPPED_ADDONS, template_fingerprint
from tests.chat_messages import Result, tool_turn, opened, after_system
from tests.chat_templates import STEMS, WEATHER, expected, fixture
from tests.fake_context import BOS, FakeContext
from tests.internals import read_fragments_as
from tests.invented_templates import (JOINS_USER_TURNS, JSON_TOOLS, PLAIN, REFUSES_A_TEXT,
                                      REFUSES_TWO_USER_TURNS)
from tests.wrappers import (add_tools, assert_cache_reads_as_rendered, assert_consistent,
                            begun_wrapper, tool_wrapper, rows)
def _n_exchanges(wrapper):
    """Exchanges the limit counts: each user message opens one, and what comes before the first is one more."""
    history = [row["role"] for row in rows(wrapper) if row["role"] != "system"]
    return history.count("user") + (1 if history and history[0] != "user" else 0)


# ----- a reply without text, and the tool call policy ----------------------
def test_a_reply_without_text_answers_the_newest_user_message():
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("persona"))
    wrapper.inject("Wie spät ist es?")
    reply = wrapper.reply().finish()
    assert [m["role"] for m in rows(wrapper)] == ["system", "user", "assistant"]
    assert reply.text == rows(wrapper)[-1]["text"]


def test_an_unknown_tool_call_policy_is_refused():
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("persona"))
    with pytest.raises(ValueError, match="tool_call_policy must be"):
        wrapper.reply("x", tool_call_policy="sometimes")


# ----- a reply the conversation cannot take ---------------------------------
def test_a_reply_the_conversation_cannot_take_is_refused_when_it_is_asked_for():
    """Refused by the call itself, before a stream is returned or read."""
    wrapper, fake = tool_wrapper()
    with pytest.raises(ValueError, match="call begin"):
        wrapper.reply("hallo")
    wrapper.begin(opened("persona"))
    with pytest.raises(ValueError, match="nothing to answer"):
        wrapper.reply()
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    for tool_call_policy in ("yield", "refuse"):
        with pytest.raises(ValueError, match="1 call"):
            wrapper.reply(tool_call_policy=tool_call_policy)
    wrapper.reply("etwas anderes").finish()     # a question closes the open turn itself
    assert not wrapper.turn_open and not wrapper.pending_calls
    wrapper.close()
    with pytest.raises(RuntimeError, match="closed"):
        wrapper.reply("hallo")


def test_a_conversation_changed_between_the_call_and_the_first_read_is_checked_again():
    wrapper, _ = tool_wrapper()
    wrapper.begin(opened("persona"))
    wrapper.inject("frage")
    stream = wrapper.reply()                    # a question waits: taken
    wrapper.begin(opened("persona"))                    # as another thread might, before it is read
    with pytest.raises(ValueError, match="nothing to answer"):
        next(stream)


# ----- begin ---------------------------------------------------------------
def test_begin_prefills_system_and_history_once():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=200, evict_above_fraction=1.0), context=fake)
    w.begin(opened("sys", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]))

    roles = [s["role"] for s in rows(w)]
    assert roles == ["system", "user", "assistant"]
    # One prefill call for the whole conversation == cheapest prefill.
    assert len(fake.prefill_calls) == 1


def test_begin_drops_oldest_history_beyond_threshold():
    fake = FakeContext(PLAIN)
    # The system turn "<[turn>system\ns<turn]>\n" and BOS are well under budget;
    # pick a tiny threshold so only the newest exchange fits.
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.05), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "old message that is long"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]))

    # The oldest exchange is dropped whole: the kept history opens on a user turn.
    assert [s["text"] for s in rows(w)] == ["s", "y"]


def test_begin_has_bos_only_at_start():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "u"}]))
    # Exactly one BOS sentinel, at position 0.
    assert fake.cache[0] == BOS
    assert fake.cache.count(BOS) == 1


# ----- inject --------------------------------------------------------------
def test_inject_appends_without_generating():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    before = w.total_tokens
    n_evicted_messages = w.inject("some context", role="user").n_evicted_messages
    assert n_evicted_messages == 0
    assert rows(w)[-1]["role"] == "user"
    assert w.total_tokens > before
    # No generation happened.
    assert fake.samplings == []


def test_inject_evicts_oldest_to_fit():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.06), context=fake)
    w.begin(opened("s"))
    w.inject("first injected chunk", role="user")
    n_evicted_messages = w.inject("second injected chunk", role="user").n_evicted_messages
    assert n_evicted_messages >= 1
    roles = [s["role"] for s in rows(w)]
    assert roles[0] == "system"  # system never evicted


# ----- replying without tools ----------------------------------------------
def test_a_refusing_reply_prefills_only_new_tokens_and_reuses_context():
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, max_reply_tokens=50), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]))
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
    w.begin(opened("s"))
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
    w.begin(opened("s"))
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
    wrapper.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    wrapper.inject_tool_result(wrapper.pending_calls[0].id, {"time": "09:00"})
    held = _held_at_each_generation(fake)
    reply = wrapper.reply(tool_call_policy="yield").finish()
    assert reply.n_prompt_tokens == held[-1][0]


def test_a_request_that_ran_calls_reports_its_last_generations_tokens():
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="The time.", name="clock")
    wrapper, fake = tool_wrapper()
    wrapper.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    held = _held_at_each_generation(fake)
    reply = wrapper.reply("los", tool_call_policy=registry).finish()
    assert len(held) == 2 and held[0][0] < held[1][0]
    assert reply.n_prompt_tokens == held[-1][0]


# ----- streaming -----------------------------------------------------------
def test_a_refusing_reply_stream_yields_incremental_deltas():
    fake = FakeContext(PLAIN, gen_tokens=4, gen_text="abcd")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    deltas = list(w.reply("hi"))
    assert "".join(deltas) == "abcd"
    assert rows(w)[-1]["role"] == "assistant"


def test_a_refusing_reply_drains_its_own_stream():
    fake = FakeContext(PLAIN, gen_tokens=3, gen_text="xyz")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))
    reply = w.reply("hi").finish()
    assert reply.text == "xyz"
    assert reply.n_generated_tokens == 3


def test_an_early_close_keeps_cache_consistent():
    # Consume only the first delta, then abandon the stream (close the generator).
    fake = FakeContext(PLAIN, gen_tokens=8, gen_text="abcdefgh")
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s"))

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


def test_a_stop_string_ends_the_reply_and_is_trimmed_from_it():
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="helloSTOPx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin(opened("s"))
    with w.reply("hi", stop=["STOP"]) as stream:
        shown = "".join(stream)
    assert shown == stream.reply.text == "hello"
    assert stream.reply.stop_reason == "stop_string"   # not "interrupted"


def test_only_text_that_could_begin_a_stop_string_is_held_back():
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="heSlSTOPx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin(opened("s"))
    starts = []
    fake.before_reply = lambda text: starts.append(len(fake.cache))
    # Each delta, and how many tokens had been generated when it came.
    shown = [(delta, len(fake.cache) - starts[0]) for delta in w.reply("hi", stop=["STOP", "lo"])]
    assert shown == [("h", 1), ("e", 2), ("S", 4), ("l", 5)]


def test_the_stop_texts_are_the_ones_given_when_the_reply_was_asked_for():
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="helloSTOPx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin(opened("s"))
    stop = ["STOP"]
    stream = w.reply("hi", stop=stop)
    stop.clear()                    # before the first read, which starts the generation
    assert stream.finish().text == "hello"


@pytest.mark.parametrize("options", [
    {"sampling": {"temperature": 0.1}}, {"stop": [1]}, {"stop": 5},
    {"cancel": True}], ids=["sampling-dict", "stop-number", "stop-int", "cancel-bool"])
def test_a_request_option_of_the_wrong_kind_is_refused_before_the_question_is_written(options):
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="helloENDx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin(opened("s"))
    before = w.conversation()
    with pytest.raises(TypeError):
        w.reply("hi", **options)
    assert w.conversation() == before
    assert w.reply("hi", stop=["END"]).finish().text == "hello"


def test_a_single_stop_text_is_one_stop_text_and_not_its_letters():
    fake = FakeContext(PLAIN, gen_tokens=10, gen_text="helloENDx")
    w = ChatWrapper(Config(context_size=400), context=fake)
    w.begin(opened("s"))
    reply = w.reply("hi", stop="END").finish()
    assert (reply.text, reply.stop_reason) == ("hello", "stop_string")


def test_generation_is_capped_to_the_context_size():
    fake = FakeContext(PLAIN, gen_tokens=100)  # would generate 100 if uncapped
    w = ChatWrapper(Config(context_size=110, evict_above_fraction=1.0, max_reply_tokens=100), context=fake)
    w.begin(opened("s"))
    reply = w.reply("hello").finish()
    assert w.total_tokens <= 110
    assert reply.n_generated_tokens < 100


def test_a_reply_runs_until_the_context_is_full_unless_a_cap_is_set():
    """A long thought counts against the cap too, so a default cap could leave no answer."""
    fake = FakeContext(PLAIN, gen_tokens=3000, gen_text="x" * 3000)
    w = ChatWrapper(Config(context_size=4096), context=fake)
    w.begin(opened("s"))
    reply = w.reply("hello").finish()
    assert reply.n_generated_tokens == 3000 and reply.stop_reason == "end_of_turn"
    assert w.reply("again", max_reply_tokens=10).finish().n_generated_tokens == 10


def test_the_cache_rests_under_the_threshold_after_a_generation():
    # The reply is appended first, then older turns are trimmed, so the cache
    # rests at or below threshold once the generation returns.
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=400, evict_above_fraction=0.2), context=fake)
    w.begin(opened("s"))
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
    w.begin(opened("s"))

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


@pytest.mark.parametrize("timeout_s", [-1, -5, -0.5])
def test_a_negative_wait_is_refused_naming_its_setting(timeout_s):
    """Threading reads -1 as waiting for ever and other negatives as an error of its own."""
    w = ChatWrapper(Config(context_size=800), context=FakeContext(PLAIN))
    with pytest.raises(InvalidArgumentError, match="timeout_s must be a number of seconds of at least 0") \
            as refused:
        w.wait_idle(timeout_s)
    assert refused.value.field == "timeout_s"
    assert w.wait_idle(0) is True


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
        w.begin(opened("x" * 100))
    # Nothing was prefilled by the refused begin; a fitting one then works.
    assert fake.cache == []
    w.begin(opened("s"))
    assert rows(w)[0]["role"] == "system"


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


# ----- content tokenized special-off (turn-boundary safety) ----------------
def test_injected_tag_survives_in_content_as_text():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("sys"))
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
    w.begin(opened("sys"))
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
    w.begin(opened("s"))
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
    assert w.begin(opened("s", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}])).n_kept_messages == 3
    assert w.begin(opened("s")).n_kept_messages == 0


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
    w.begin(opened(""))
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


def test_what_the_model_can_do_is_answered_per_kind_of_conversation_and_never_changes():
    from tests.invented_templates import REASONING, TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN

    # The template for tools can reason, the default template cannot.
    fake = FakeContext({**TOOLS_IN_A_TEMPLATE_OF_THEIR_OWN,
                        "tool_use_template": REASONING["chat_template"],
                        "render_variables": REASONING["render_variables"]})
    w = ChatWrapper(Config(context_size=512), context=fake)
    without, with_tools = w.capabilities(tools=False), w.capabilities(tools=True)
    assert not without.can_reason and with_tools.can_reason
    with pytest.raises(UnsupportedError) as refused:
        w.begin(opened("s"), reasoning=True)
    assert refused.value.field == "reasoning"
    w.begin(opened("s"), tools=[], reasoning=True)      # what the record says is accepted, is
    assert (w.capabilities(tools=False), w.capabilities(tools=True)) == (without, with_tools)
    with pytest.raises(TypeError):
        w.capabilities(True)                    # the kind is always named


def test_what_the_model_cannot_do_is_refused_naming_the_parameter_that_asked_for_it():
    plain = ChatWrapper(Config(context_size=512), context=FakeContext(PLAIN))
    assert plain.capabilities(tools=False) == plain.capabilities(tools=True)
    assert not plain.capabilities(tools=True).can_call_tools
    registry = ToolRegistry()
    registry.tool(lambda: "12:00", description="Die Zeit.", name="clock")
    turn = tool_turn(ToolCall("clock", {}), Result("clock", "12:00"))
    refusals = [
        ("tools", lambda: plain.begin(opened("s"), tools=registry.specs())),
        ("messages", lambda: plain.begin(opened("s", [{"role": "user", "content": "hi"}, *turn]))),
        ("reasoning", lambda: plain.begin(opened("s"), reasoning=True)),
    ]
    plain.begin(opened("s"))
    refusals += [
        ("tools", lambda: add_tools(plain, registry.specs())),
        ("tool_call_policy", lambda: plain.reply("hi", tool_call_policy=registry).finish()),
    ]
    for field, refused_action in refusals:
        with pytest.raises(UnsupportedError) as refused:
            refused_action()
        assert refused.value.field == field
        assert isinstance(refused.value, ValueError)
    assert [m["text"] for m in rows(plain)] == ["s"]

    stem = next(stem for stem in STEMS if not expected(fixture(stem), "can_take_system_prompt"))
    no_system = ChatWrapper(Config(context_size=4096), context=FakeContext(fixture(stem)))
    assert not no_system.capabilities(tools=False).can_take_system_prompt
    with pytest.raises(UnsupportedError) as refused:
        no_system.begin(opened("persona"))
    assert refused.value.field == "messages"


# ----- what an action is given, refused where it enters ---------------------
def test_a_message_that_is_not_text_is_refused_where_it_enters_naming_it():
    w, fake = begun_wrapper()
    w.reply("hi").finish()
    before, held = w.conversation(), list(fake.cache)
    for action, what in ((lambda: w.inject(42), "injected message"),
                         (lambda: w.reply(42), "request"),
                         (lambda: w.begin([{"role": "system", "content": None}]),
                          "messages\\[0\\].content"),
                         (lambda: w.begin(opened("s", [{"role": "user", "content": 42}])),
                          "messages\\[1\\].content"),
                         (lambda: w.begin(opened("s", [{"role": "user", "content": None}])),
                          "messages\\[1\\].content")):
        with pytest.raises(TypeError, match=f"{what}.*str, not"):
            action()
    assert w.conversation() == before and fake.cache == held


@pytest.mark.parametrize("max_calls, refusal", [(-1, InvalidArgumentError), (1.5, TypeError),
                                                (True, TypeError)])
def test_a_ceiling_on_calls_that_is_no_count_is_refused(max_calls, refusal):
    w, _ = tool_wrapper()
    w.begin(opened("s"))
    with pytest.raises(refusal, match="max_calls"):
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

    ChatWrapper(Config(context_size=500), context=_Generous(PLAIN)).begin(opened("s"))


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
    assert_cache_reads_as_rendered(w, fake)


def test_a_template_that_refuses_two_user_turns_keeps_a_reply_without_an_answer():
    w, fake = begun_wrapper(REFUSES_TWO_USER_TURNS, gen_tokens=0)
    w.reply("eins").finish()
    found = [(m["role"], m["text"]) for m in rows(w)]
    assert found[1:] == [("user", "eins"), ("assistant", "")]
    fake.gen_tokens = 5
    w.reply("zwei").finish()
    assert_cache_reads_as_rendered(w, fake)


@pytest.mark.parametrize("stem", [stem for stem in STEMS
                                  if not expected(fixture(stem), "writes_empty_reply")])
@pytest.mark.parametrize("nothing", ["cancelled-at-once", "ends-at-once", "cancelled-by-a-tool"])
def test_a_reply_that_produced_nothing_the_template_cannot_write_takes_its_question_back(
        stem, nothing):
    """These templates write no empty reply, so the request is taken back whole and the next question is answered; a later round cancelled before its first word leaves its turn open."""
    from llama_chat.resolved_template import ResolvedTemplate
    from tests.fake_context import ScriptedCall, written_call
    fake = FakeContext(fixture(stem), gen_tokens=7, gen_text="Antwort")
    w = ChatWrapper(Config(context_size=8192, evict_above_fraction=1.0), context=fake)
    cancel, options = threading.Event(), {}
    if nothing == "cancelled-by-a-tool":
        registry = ToolRegistry()
        registry.tool(lambda city: (cancel.set(), "18 Grad")[1], description="Wetter.",
                      name=WEATHER.name, parameters=WEATHER.parameters)
        w.begin(opened(""), tools=registry.specs())
        reader = FakeContext(fixture(stem))
        _opener, body, _closer = written_call(ResolvedTemplate(reader), reader, WEATHER.name,
                                              {"city": "Bonn"})
        fake.replies = [ScriptedCall(body)]
        options = dict(tool_call_policy=registry, cancel=cancel)
    else:
        w.begin(opened(""))
        if nothing == "cancelled-at-once":
            cancel.set()
            options = dict(cancel=cancel)
        else:
            fake.gen_tokens = 0
    w.reply("Erste Frage", **options).finish()
    fake.gen_tokens = 7
    if nothing == "cancelled-by-a-tool":
        # A later round cancelled before its first word changes nothing: the turn stays open
        # with its result, and the reply goes on with it.
        assert w.turn_open and after_system(w.conversation()["messages"])[0]["content"] == "Erste Frage"
        assert w.reply().finish().text == "Antwort"
        return
    assert after_system(w.conversation()["messages"]) == [] and not w.turn_open
    assert w.reply("Zweite Frage").finish().text == "Antwort"
    assert [m["text"] for m in rows(w)] == ["Zweite Frage", "Antwort"]


def test_a_reply_the_template_refuses_to_write_takes_its_question_back():
    fake = FakeContext(REFUSES_A_TEXT, replies=["refused", "fine"])
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "eins"}, {"role": "assistant", "content": "zwei"}]))
    before = w.conversation()
    assert w.reply("hi").finish().text == "refused"
    assert w.conversation() == before
    assert w.reply("next?").finish().text == "fine"
    assert [m["text"] for m in rows(w)][-2:] == ["next?", "fine"]


def test_an_injected_user_turn_after_a_user_turn_is_written_or_joined():
    w, fake = begun_wrapper(JOINS_USER_TURNS)
    w.inject("eins")
    w.inject("zwei")
    assert_cache_reads_as_rendered(w, fake)
    w, fake = begun_wrapper(REFUSES_TWO_USER_TURNS)
    w.inject("eins")
    w.inject("zwei")
    assert [r["text"] for r in rows(w)][1:] == ["eins\n\nzwei"]
    assert w.reply().finish().text


def test_a_history_with_two_user_turns_in_a_row_is_joined_or_refused():
    fake = FakeContext(JOINS_USER_TURNS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "eins"}, {"role": "user", "content": "zwei"}, {"role": "assistant", "content": "drei"}]))
    assert_cache_reads_as_rendered(w, fake)
    fake = FakeContext(REFUSES_TWO_USER_TURNS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0), context=fake)
    with pytest.raises(ValueError, match="two user turns"):
        w.begin(opened("s", [{"role": "user", "content": "eins"}, {"role": "user", "content": "zwei"}]))


def test_a_refused_history_is_explained_by_what_would_be_kept_of_it():
    """Two user turns in a row that would be dropped are not why the rest is refused."""
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0),
                    context=FakeContext(REFUSES_A_TEXT))
    with pytest.raises(ValueError, match="refuses to write this conversation"):
        w.begin(opened("s", [{"role": "user", "content": "eins"}, {"role": "user", "content": "zwei"}, {"role": "assistant", "content": "drei"},
                      {"role": "user", "content": "refused"}, {"role": "assistant", "content": "vier"}]))


def test_evicting_the_first_of_two_user_turns_takes_its_join_with_it():
    fake = FakeContext(JOINS_USER_TURNS)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, max_history_exchanges=1),
                    context=fake)
    w.begin(opened("s"))
    w.inject("eins")
    w.inject("zwei")
    assert [m["text"] for m in rows(w)] == ["s", "zwei"]
    assert_cache_reads_as_rendered(w, fake)


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
    w.begin(opened("s"))
    with pytest.raises(ContextOverflowError):
        w.inject("x" * 200)                      # refused: never recorded
    with pytest.raises(ContextOverflowError):
        w.reply("y" * 200).finish()
    stream = w.reply("hi")
    next(stream)
    stream.close()                               # closed early: recorded
    assert after_system(w.conversation()["messages"]) == [{"role": "user", "content": "hi"},
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
    w.begin(opened("s"))
    w.reply("eins").finish()
    seen = []
    fake.before_reply = seen.append
    w.reply(question).finish()
    assert "eins" in seen[0]
    assert _n_exchanges(w) == 1 and rows(w)[1]["text"] == question
    assert_consistent(w, fake)


@pytest.mark.parametrize("stem", STEMS)
def test_the_readmes_quick_start_runs_on_every_template(stem):
    chat = ChatWrapper(Config(context_size=8192), context=FakeContext(fixture(stem)))
    takes_system = chat.capabilities(tools=False).can_take_system_prompt
    chat.begin(opened("You are a helpful assistant." if takes_system else "", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]))
    chat.inject("Reference: the deadline is Friday. What's the deadline?")
    for _ in chat.reply():
        pass
    chat.reply("Summarize the plan.").finish()


# ----- what does not have the shape asked for is refused where it enters ----------
@pytest.mark.parametrize("tools", [[{"name": "clock"}], ["clock"], [ToolSpec(1, None, None)], 5],
                         ids=["dict", "str", "fields", "no-list"])
def test_tools_of_another_shape_are_refused_before_anything_changes(tools):
    w, fake = tool_wrapper()
    w.begin(opened("s"))
    before = list(fake.cache)
    with pytest.raises(TypeError):
        w.begin(opened("t"), tools=tools)
    assert fake.cache == before


def test_a_history_given_as_a_generator_is_read_once():
    w, _ = begun_wrapper()
    w.begin(opened("s", (item for item in [{"role": "user", "content": "hallo"}, {"role": "assistant", "content": "hi"}])))
    assert [m["content"] for m in after_system(w.conversation()["messages"])] == ["hallo", "hi"]


def test_a_reply_shortened_to_no_text_is_refused_naming_keep_text():
    w, _ = begun_wrapper()
    w.reply("hi").finish()
    with pytest.raises(TypeError, match="keep_text"):
        w.truncate_last_reply(("tuple",))


def test_a_reply_to_an_injected_question_evicts_down_to_the_room_its_reply_needs():
    """Where the eviction threshold lies above the room a reply needs, the reply still gets that room."""
    fake = FakeContext(PLAIN, gen_tokens=5)
    w = ChatWrapper(Config(context_size=500, evict_above_fraction=1.0, min_reply_tokens=100),
                    context=fake)
    w.begin(opened("s", [{"role": "user", "content": "a" * 80}, {"role": "assistant", "content": "b" * 80}, {"role": "user", "content": "c" * 80},
                  {"role": "assistant", "content": "d" * 80}]))
    w.inject("q" * 230)
    assert 400 < w.total_tokens <= 500          # over the room, under the threshold
    reply = w.reply().finish()
    assert reply.n_evicted_messages > 0 and reply.stop_reason == "end_of_turn"


def test_awaits_reply_says_whether_a_reply_without_text_answers_something():
    w, fake = tool_wrapper()
    assert not w.awaits_reply                   # nothing begun
    w.begin(opened("persona"))
    assert not w.awaits_reply
    w.inject("wie spät?")
    assert w.awaits_reply
    fake.call_script = ["fn:clock{}"]
    call, = [e for e in w.reply(tool_call_policy="yield") if isinstance(e, ToolCall)]
    assert not w.awaits_reply                   # a result is owed
    w.inject_tool_result(call.id, "12:00")
    assert w.awaits_reply                       # every result is in: the reply goes on
    w.reply(tool_call_policy="yield").finish()
    assert not w.awaits_reply


# ----- one report of what an action changed beyond what was asked ------------------
def test_begin_says_how_many_history_messages_it_left_out_and_why(caplog):
    fake = FakeContext(PLAIN, gen_tokens=2)
    w = ChatWrapper(Config(context_size=400), context=fake)
    history = [{"role": "user", "content": "q" * 60}, {"role": "assistant", "content": "a" * 60}] * 4
    with caplog.at_level("INFO", logger="llama_chat"):
        report = w.begin(opened("s", history))
    assert report.n_kept_messages < len(history) and report.evicted_for == "half_context"
    assert report.n_evicted_messages == len(history) - report.n_kept_messages
    said, = [r.getMessage() for r in caplog.records]
    assert said == (f"the {report.n_evicted_messages} oldest history message(s) went so that "
                    "half of the 400-token context stays free at a begin")
    caplog.clear()
    with caplog.at_level("INFO", logger="llama_chat"):
        w.begin(opened("s", history[-2:]))          # kept whole: nothing to say
    assert not caplog.records


def test_generated_turns_of_another_model_are_reported_as_unused(caplog):
    w, _ = begun_wrapper()
    with caplog.at_level("INFO", logger="llama_chat"):
        report = w.begin(opened("s", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]), generated_turns={"model": "another", "turns": []})
    assert report.generated_turns_ignored
    assert any("another model's" in r.getMessage() for r in caplog.records)


def test_a_result_that_cannot_fit_reports_the_turn_closed_not_a_success():
    w, fake = tool_wrapper(context_size=160)
    w.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    for size in range(1, 400):
        w, fake = tool_wrapper(context_size=60 + size)
        try:
            w.begin(opened("persona"))
        except ContextOverflowError:
            continue
        fake.call_script = ["fn:clock{}"]
        try:
            calls = [e for e in w.reply("wie?", tool_call_policy="yield") if isinstance(e, ToolCall)]
        except ContextOverflowError:
            continue
        if not calls or not w.pending_calls:
            continue
        report = w.inject_tool_result(calls[0].id, "x" * 2000)
        if not w.turn_open:
            assert report.turn_closed
            return
    pytest.fail("no context size left the call room but its result none")


def test_a_stream_closed_early_still_reports_its_generation():
    w, fake = begun_wrapper(gen_tokens=8, gen_text="Antwort.")
    stream = w.reply("hallo")
    next(stream)
    stream.close()
    assert stream.reply is not None and stream.reply.n_generated_tokens >= 1


def test_an_injected_message_far_below_the_limit_is_counted_alone_and_one_at_the_limit_is_measured():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.5), context=fake)
    w.begin(opened("s", [{"role": "user", "content": "hi"}]))
    seen = []
    measure = w._conversation._rendering.length
    w._conversation._rendering.length = lambda messages, *a, **k: (
        seen.append([m["content"] for m in messages]), measure(messages, *a, **k))[1]
    w.inject("short", role="assistant")
    assert ["s", "short"] not in seen
    limit = w._config.evict_above_tokens
    with pytest.raises(ContextOverflowError) as refused:
        w.inject("x " * limit, role="assistant")
    assert ["s", "x " * limit] in seen
    assert refused.value.available_tokens == limit and refused.value.needed_tokens > limit


# ----- what only some templates need --------------------------------------------------
# Every function a plain reply may run, by file: on a template without reasoning, tools or a
# clock. Code that serves only some templates is never among them; a function new to a plain
# reply fails the test below until it is judged to serve every template and listed here.
_PLAIN_REPLY_CODE = {
    "cache_sync.py": {
        "CacheSync._bring_up",
        "CacheSync._can_cut",
        "CacheSync._decode_at_end",
        "CacheSync._decode_at_end.<locals>.<genexpr>",
        "CacheSync._pins_to_save",
        "CacheSync._save_pinned_state",
        "CacheSync.cut_to",
        "CacheSync.decode",
        "CacheSync.generate",
        "CacheSync.hold",
        "CacheSync.is_current",
        "CacheSync.is_known",
        "CacheSync.keeps_saved_states",
        "CacheSync.repair",
        "CacheSync.rest",
        "CacheSync.tokens",
    },
    "config.py": {
        "Config.__post_init__",
        "Config.evict_above_tokens",
        "Sampling.__post_init__",
        "_is_of_kind",
        "ensure_reply_cap",
        "ensure_types",
        "ensure_types.<locals>.<genexpr>",
        "generation_sampling",
        "with_request_sampling",
    },
    "conversation_state.py": {
        "ConversationState.copied",
        "copied_value",
        "goes_on_from",
        "goes_on_from.<locals>.<genexpr>",
    },
    "eviction.py": {
        "Eviction._exchanges_over",
        "Eviction.drop_while_over",
        "Eviction.drop_while_over.<locals>.length",
        "Eviction.drop_while_over.<locals>.too_many",
        "Eviction.too_many",
    },
    "generation.py": {
        "_TextBuffer.__init__",
        "_TextBuffer.__len__",
        "_TextBuffer.append",
        "_TextBuffer.emit",
        "_append",
        "_sample_into",
        "generate",
    },
    "messages.py": {
        "ensure_role",
        "ensure_text",
        "exchange_starts",
        "system_part",
    },
    "model_template.py": {
        "ModelTemplate._rendered",
        "ModelTemplate.render",
    },
    "publishing.py": {
        "published",
        "published.<locals>.<genexpr>",
    },
    "render_checks.py": {
        "question_splices",
    },
    "rendered_cache.py": {
        "RenderedCache.__init__.<locals>.<lambda>",
        "RenderedCache._action",
        "RenderedCache._after_reply",
        "RenderedCache._append_reply",
        "RenderedCache._ask",
        "RenderedCache._drop_unplaceable",
        "RenderedCache._drop_while_over",
        "RenderedCache._ends_on_result",
        "RenderedCache._ensure_writes",
        "RenderedCache._holds_thought_tags",
        "RenderedCache._is_current",
        "RenderedCache._joins_questions",
        "RenderedCache._length_alone",
        "RenderedCache._made_for_now",
        "RenderedCache._make_way_for",
        "RenderedCache._max_prompt_tokens",
        "RenderedCache._moves_declarations",
        "RenderedCache._newest_exchange",
        "RenderedCache._note_ahead",
        "RenderedCache._note_prompt",
        "RenderedCache._publish",
        "RenderedCache._publishing",
        "RenderedCache._read_calls",
        "RenderedCache._record_generated",
        "RenderedCache._render_ahead",
        "RenderedCache._reply_opening",
        "RenderedCache._settle",
        "RenderedCache._settle.<locals>.<lambda>",
        "RenderedCache._splice",
        "RenderedCache._sync",
        "RenderedCache._system_part",
        "RenderedCache._turn_end_test",
        "RenderedCache._writes",
        "RenderedCache.add",
        "RenderedCache.ask",
        "RenderedCache.begun",
        "RenderedCache.decoded",
        "RenderedCache.ensure_can_reply",
        "RenderedCache.generate",
        "RenderedCache.generation",
        "RenderedCache.newest_role",
        "RenderedCache.open_attempt",
        "RenderedCache.pending_calls",
        "RenderedCache.prompt",
        "RenderedCache.prompt_text",
        "RenderedCache.published_turn_open",
        "RenderedCache.reasoning",
        "RenderedCache.record_reply",
        "RenderedCache.report",
        "RenderedCache.results_cut",
        "RenderedCache.template",
        "RenderedCache.total_tokens",
        "RenderedCache.turn_budget",
        "RenderedCache.turn_open",
        "RenderedCache.turn_policy",
        "_ensure_begun",
        "_ensure_can_reply",
        "_ensure_ready",
        "_reported_calls",
        "_unanswered",
    },
    "rendering.py": {
        "ConversationRendering._as_given",
        "ConversationRendering._as_given.<locals>.<genexpr>",
        "ConversationRendering._call_tags",
        "ConversationRendering._current_memo",
        "ConversationRendering._declarations",
        "ConversationRendering._render_given",
        "ConversationRendering._template",
        "ConversationRendering.count_tokens",
        "ConversationRendering.forget_renders",
        "ConversationRendering.is_plain",
        "ConversationRendering.is_plain.<locals>.<genexpr>",
        "ConversationRendering.length",
        "ConversationRendering.render",
        "ConversationRendering.run_start",
        "ConversationRendering.split",
        "ConversationRendering.split.<locals>.<lambda>",
        "ConversationRendering.tag_after_content",
        "ConversationRendering.tag_from",
        "ConversationRendering.tag_from.<locals>.<genexpr>",
        "ConversationRendering.target",
        "ConversationRendering.template_writes",
        "ConversationRendering.template_writes.<locals>.<genexpr>",
        "ConversationRendering.tokenize",
        "kept_turns",
    },
    "reply.py": {
        "ReplyStream.__init__",
        "ReplyStream.__iter__",
        "ReplyStream.__next__",
        "ReplyStream.finish",
    },
    "reply_text.py": {
        "ReplyText.__init__",
        "ReplyText.answer",
        "ReplyText.feed",
        "ReplyText.finish",
        "ReplyText.in_thought",
        "ReplyText.met_stop_string",
        "ReplyText.opens_in_thought",
        "ReplyText.thought",
        "ReplyText.thought_pieces",
        "ReplyText.turn_end_test",
        "ReplyText.turn_end_test.<locals>.<lambda>",
        "_AnswerOpening.__init__",
        "_AnswerOpening._unpadded",
        "_AnswerOpening.feed",
        "_AnswerOpening.flush",
        "_AnswerStops.__init__",
        "_AnswerStops.feed",
        "_AnswerStops.flush",
        "_AnswerStops.text",
        "reply_header",
        "without_tags",
    },
    "resolved_template.py": {
        "ResolvedTemplate._as_template_reads",
        "ResolvedTemplate.answer_opening",
        "ResolvedTemplate.call_boundary",
        "ResolvedTemplate.generation_prompt",
        "ResolvedTemplate.render",
        "ResolvedTemplate.reply_header",
        "ResolvedTemplates.for_conversation",
    },
    "sequences.py": {
        "common_prefix_length",
    },
    "split_render.py": {
        "SplitRender.pieces",
        "_forms",
        "_marked",
        "_marked.<locals>.mark",
        "_marked.<locals>.mark_descriptions",
        "_placed",
        "_placed.<locals>.<genexpr>",
        "_walk",
        "_walk.<locals>.matched_from",
        "_walk.<locals>.matched_from.<locals>.<genexpr>",
        "_walk.<locals>.rest",
        "split_render",
    },
    "template.py": {
        "TemplateFragments.content",
        "TemplateFragments.generation_prompt_without_reasoning",
    },
    "template_extract.py": {
        "reasoning_switch.<locals>.<lambda>",
    },
    "tokenizing.py": {
        "MemoTokenizer.is_special",
        "MemoTokenizer.token_text",
        "MemoTokenizer.tokenize",
        "split_at_tags",
        "tokenize_pieces",
        "tokenize_pieces.<locals>.end_plain",
        "tokenize_pieces.<locals>.tokenize",
    },
    "windows.py": {
        "Windows.ahead",
        "Windows.drop_in",
    },
    "wrapper.py": {
        "ChatWrapper._ensure_not_holding",
        "ChatWrapper._ensure_open",
        "ChatWrapper._generate",
        "ChatWrapper._generate_until_answered",
        "ChatWrapper._generate_until_answered.<locals>.<genexpr>",
        "ChatWrapper._generate_until_answered.<locals>.may_call_after_unreadable",
        "ChatWrapper._held",
        "ChatWrapper._keep_turn",
        "ChatWrapper._noting_each_reader",
        "ChatWrapper._open_turn_text",
        "ChatWrapper._policy_for",
        "ChatWrapper._read_generation",
        "ChatWrapper._reply_cap",
        "ChatWrapper._reply_cap.<locals>.<genexpr>",
        "ChatWrapper._replying",
        "ChatWrapper._replying.<locals>.<lambda>",
        "ChatWrapper._template",
        "ChatWrapper.inject",
        "ChatWrapper.reply",
        "ChatWrapper.reply.<locals>.<lambda>",
        "_CallBudget.__post_init__",
        "_answerable",
        "_ensure_cancel_and_sampling",
        "_ensure_max_calls_runs",
        "_explain_failure",
        "_stop_texts",
        "_sum_generations",
        "_sum_generations.<locals>.<genexpr>",
    },
}


def test_a_reply_on_a_template_without_reasoning_tools_or_a_clock_never_enters_their_code():
    """Invariant 2: no question, reply or token pays for a property the loaded template does not have."""
    import sys
    w = ChatWrapper(Config(context_size=1000), context=FakeContext(PLAIN, replies=["Gut.", "Ja."]))
    w.begin(opened("s"))
    entered = set()

    def note(frame, event, _arg):
        if event == "call" and "llama_chat/" in frame.f_code.co_filename:
            entered.add(frame.f_code.co_filename.split("llama_chat/")[1] + ":"
                        + frame.f_code.co_qualname)

    sys.setprofile(note)
    try:
        w.reply("Wie geht es?").finish()
        w.inject("Noch etwas.")
        w.reply().finish()
    finally:
        sys.setprofile(None)
    assert entered                  # the profile saw the library at work
    allowed = {f"{file}:{name}" for file, names in _PLAIN_REPLY_CODE.items() for name in names}
    assert sorted(entered - allowed) == [], "code new to a plain reply"


def test_the_render_memory_holds_one_actions_renders_however_long_the_conversation():
    fake = FakeContext(PLAIN)
    w = ChatWrapper(Config(context_size=1000, evict_above_fraction=0.5), context=fake)
    w.begin(opened("s"))
    sizes = []
    for n in range(60):
        w.reply(f"Frage {n}?").finish()
        sizes.append(len(w._conversation._rendering._memo))
    assert max(sizes[30:]) <= max(sizes[:30])


def test_every_action_reports_no_eviction_where_nothing_went():
    """Evictions are recorded where they happen, so an action that drops nothing reports none."""
    clock = ToolSpec("clock", "time", {"type": "object", "properties": {}})
    fake = FakeContext(JSON_TOOLS, replies=["Gut.", "Es ist zwölf.", "Ja."])
    w = ChatWrapper(Config(context_size=4096), context=fake)
    reports = [w.begin(opened("s", [{"role": "user", "content": "Hallo?"},
                             {"role": "assistant", "content": "Hallo."}]), tools=[clock])]
    reports.append(w.reply("Wie geht es?").finish())
    fake.call_script = ['[{"name": "clock", "arguments": {}}]']
    reports.append(w.reply("Wie spät?", tool_call_policy="yield").finish())
    reports.append(w.inject_tool_result(w.pending_calls[0].id, "12:00"))
    reports.append(w.reply().finish())
    reports.append(w.inject("Danke.", role="user"))
    reports.append(w.begin(**{**w.conversation(), "reasoning": False}))
    assert [(r.n_evicted_messages, r.evicted_for) for r in reports] == [(0, None)] * len(reports)


def test_a_reply_publishes_once_per_action_and_builds_no_history_copy_until_one_is_read(monkeypatch):
    from llama_chat import publishing, rendered_cache
    w, _fake = begun_wrapper()
    w.reply("eins").finish()
    made, built = [], []
    with monkeypatch.context() as patched:
        patched.setattr(rendered_cache, "published",
                        lambda *a, **k: made.append(1) or publishing.published(*a, **k))
        patched.setattr(publishing, "completions_messages",
                        lambda *a, **k: built.append(1) or [])
        w.reply("zwei").finish()
    assert len(made) <= 2 and not built        # the question, then the recorded reply


def test_a_stream_never_read_reports_no_other_streams_reply():
    w, _ = begun_wrapper(gen_tokens=8, gen_text="Antwort.")
    unread = w.reply("erste")
    read = w.reply("zweite").finish()
    unread.close()
    assert read.text == "Antwort." and unread.reply is None
