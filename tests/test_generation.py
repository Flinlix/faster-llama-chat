"""Generating on the engine's cache context: calls found as they are generated, cancelling."""

import dataclasses
import pytest
from llama_chat.config import Config, generation_sampling
from llama_chat.generation import generate, AnswerGuard, GenerationAccumulator

from llama_chat.template import CallBoundary
from llama_chat.wrapper import ChatWrapper
from tests.chat_messages import opened, after_system
from llama_chat.testing.fake_context import FakeContext
from llama_chat.testing.invented_templates import PLAIN, TOOLS
from tests.wrappers import rows

# Every sampling value set, as a generation is given them.
SAMPLING = generation_sampling(Config(), None)


# These drive the generation loop over KVContext's own steps on a stubbed
# backend, so that what reaches the engine is pinned along with the loop.
def _generating(pieces: dict, order: list, eog: int | None = None):
    """A KVContext that samples ``order`` and renders each token from ``pieces``."""
    from types import SimpleNamespace

    from llama_chat.context import KVContext

    sampled = iter(order)
    ctx = object.__new__(KVContext)
    ctx._decode_blocks = False
    ctx._backend = SimpleNamespace(
        sample=lambda sampler, ctx: next(sampled),
        is_eog=lambda vocab, token: token == eog,
        is_special=lambda vocab, token: False,
        decode=lambda *_a: None,
        token_to_piece_bytes=lambda vocab, token: pieces[token],
        free_sampler=lambda sampler: None)
    ctx._vocab = ctx._engine_context = None
    ctx._config = Config()
    ctx._build_sampler = lambda config, withheld=(): None
    ctx._max_batch_tokens = 8
    return ctx


_PIECES = {0: b"hi ", 1: b"<call>", 2: b"fn:clock{}", 3: b"</call>", 4: b" done"}


_TAGS = (1, 3)


# ----- detecting a call as it is generated ---------------------------------
def test_a_call_is_recognised_by_its_token_and_never_reaches_the_reply():
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 10, sampling=SAMPLING, out=acc, boundary=CallBoundary(tool_tokens=_TAGS))

    assert list(stream) == ["hi "]      # the call's own text is never yielded
    assert acc.stop_reason == "tool_call"
    assert acc.tool_call == "fn:clock{}"
    assert acc.text == "hi "
    assert acc.token_ids == [0, 1, 2, 3]   # yet all of it reached the cache


def test_the_held_back_tail_is_released_when_a_call_opens():
    """The text a reply ends with holds text back; the call is not what it was waiting for."""
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    # "ZZZZ" is longer than "hi ", so nothing can be emitted before the call.
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 10, sampling=SAMPLING, out=acc,
                      boundary=CallBoundary(tool_tokens=_TAGS, turn_end_text="ZZZZ"))
    assert list(stream) == ["hi "]
    assert acc.text == "hi "            # nothing of the reply was lost


def test_closing_the_stream_as_a_call_opens_still_reads_as_interrupted():
    """The release above is a yield like any other, so an early close counts there.

    It used to sit outside the ``GeneratorExit`` handler, and a consumer that
    closed exactly then saw ``stop_reason == "token_limit"``.
    """
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 10, sampling=SAMPLING, out=acc,
                      boundary=CallBoundary(tool_tokens=_TAGS, turn_end_text="ZZZZ"))
    assert next(stream) == "hi "
    stream.close()
    assert acc.stop_reason == "interrupted"


def test_a_call_the_token_cap_cuts_short_is_dropped():
    """Truncation mid-call loses the call, not the reply, and closes normally."""
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2]), 0, 3, sampling=SAMPLING, out=acc, boundary=CallBoundary(tool_tokens=_TAGS))
    assert list(stream) == ["hi "]
    assert acc.stop_reason == "token_limit"
    assert acc.tool_call == ""          # half a call is no call
    assert acc.text == "hi "            # and none of it leaked into the reply


def test_a_turn_that_ends_inside_a_call_closed_by_a_tag_closes_the_call_with_that_tag():
    """The model ends its turn before the call's closing tag: the call is closed as the template closes one."""
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2, 5], eog=5), 0, 10, sampling=SAMPLING, out=acc,
                      boundary=CallBoundary(tool_tokens=_TAGS))
    assert list(stream) == ["hi "]
    assert acc.stop_reason == "tool_call"
    assert acc.tool_call == "fn:clock{}"
    assert acc.token_ids == [0, 1, 2, 3]       # the closing tag reached the cache


def test_without_tool_tokens_a_call_tag_is_ordinary_text():
    """A model that cannot call tools has its tags yielded like any other text."""
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 4, sampling=SAMPLING, out=acc)
    assert "".join(stream) == "hi <call>fn:clock{}</call>"
    assert acc.stop_reason == "token_limit"


# The two ways a stream ends besides a call, and what reaches the cache
# before it stops.
def test_end_of_generation_stops_before_the_token_reaches_the_cache():
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 4], eog=4), 0, 10, sampling=SAMPLING, out=acc)
    assert "".join(stream) == "hi "
    assert acc.stop_reason == "end_of_turn"
    assert acc.token_ids == [0]     # the end token is recognised, never decoded


def test_cancelling_ends_the_stream_within_one_token():
    import threading

    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    cancel = threading.Event()
    ctx = _generating(_PIECES, [0, 4, 4])
    # Set it while the first token renders; the loop checks before sampling the
    # next one, so exactly one token makes it in.
    render = ctx._backend.token_to_piece_bytes
    ctx._backend.token_to_piece_bytes = lambda v, t: (cancel.set(), render(v, t))[1]

    stream = generate(ctx, 0, 10, sampling=SAMPLING, out=acc, cancel=cancel)
    assert "".join(stream) == "hi "
    assert acc.stop_reason == "interrupted"
    assert acc.token_ids == [0]


def test_only_a_tail_that_could_begin_a_mark_is_held_back():
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating({0: b"ab", 1: b"cZ", 2: b"Zq", 3: b"Z"}, [0, 1, 2, 3]), 0, 4,
                      sampling=SAMPLING, out=acc, boundary=CallBoundary(turn_end_text="ZZZZ"))
    assert list(stream) == ["ab", "c", "ZZq", "Z"]
    assert acc.text == "abcZZqZ"


def test_a_tail_that_could_begin_a_plain_text_opening_is_held_only_where_a_call_may_open():
    """A call that opens with plain text opens only at the reply's start or after a tag."""
    pieces = {0: b"{", 1: b"x", 2: b"a {", 3: b"b"}
    boundary = CallBoundary(call_texts=("{\"n\"", "}"), opens_as_text=True)
    assert list(generate(_generating(pieces, [0, 1]), 0, 2, sampling=SAMPLING, boundary=boundary)) == ["{x"]
    assert list(generate(_generating(pieces, [2, 3]), 0, 2, sampling=SAMPLING, boundary=boundary)) == ["a {", "b"]


def test_the_held_back_tail_is_flushed_when_generation_runs_out():
    from llama_chat.generation import GenerationAccumulator

    acc = GenerationAccumulator()
    # The reply ends on text that could still begin the closing, so without the
    # final flush it would end short.
    stream = generate(_generating({0: b"abc", 1: b"deZZ"}, [0, 1]), 0, 2, sampling=SAMPLING, out=acc,
                      boundary=CallBoundary(turn_end_text="ZZZZ"))
    assert list(stream) == ["abc", "de", "ZZ"]
    assert acc.stop_reason == "token_limit"
    assert acc.text == "abcdeZZ"


def test_a_resumed_reply_gathers_its_text_into_one_message():
    """One turn's text is what a reader saw, across both generations of it.

    Whatever persists the conversation reads it back with the resumed prose,
    not only what came before the call.
    """
    fake = FakeContext(TOOLS, gen_text="neun Uhr")
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin(opened("persona"))
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", tool_call_policy="yield"))
    assert rows(wrapper)[-1]["text"] is None   # nothing visible yet: a reply that only calls

    wrapper.inject_tool_result(wrapper.pending_calls[0].id, {"time": "09:00"})
    fake.call_script = []
    resumed = "".join(e for e in wrapper.reply(tool_call_policy="yield") if isinstance(e, str))

    row = rows(wrapper)[-1]
    assert resumed and row["text"] == resumed       # every visible delta, once
    assert "clock" not in row["text"]               # the call is not reply text
    assert "09:00" not in row["text"]               # nor is the result
    held = fake.text_of(fake.cache)
    assert "clock" in held and "09:00" in held      # yet both reached the cache
    assert after_system(wrapper.conversation()["messages"])[-1]["content"] == resumed   # and that is what is read back


# The plain-text call opening the generation tests below look for.
_PLAIN_OPENING = CallBoundary(call_texts=("{\"n\"", "}"), opens_as_text=True)


def _drawn_generation(pieces: dict, order: list, special: set, boundary: CallBoundary,
                     guard: AnswerGuard | None = None) -> tuple[str, GenerationAccumulator]:
    """What a generation of ``order`` yields, and its accumulator, the tokens in ``special`` being the model's tags."""
    ctx = _generating(pieces, order)
    ctx._backend.is_special = lambda vocab, token: token in special
    acc = GenerationAccumulator()
    text = "".join(generate(ctx, 0, len(order), sampling=SAMPLING, out=acc, boundary=boundary,
                            guard=guard or AnswerGuard()))
    return text, acc


@pytest.mark.parametrize("follows_text", [False, True], ids=["after-a-tag-only", "after-text-too"])
def test_a_reply_cut_after_text_on_part_of_a_plain_opening_keeps_that_part_as_its_answer(follows_text):
    boundary = dataclasses.replace(_PLAIN_OPENING, follows_text=follows_text)
    text, acc = _drawn_generation({0: b"a ", 1: b"{"}, [0, 1], set(), boundary)
    assert text == "a {" and not acc.cut_inside_call and acc.stop_reason == "token_limit"


@pytest.mark.parametrize("follows_text", [False, True], ids=["after-a-tag-only", "after-text-too"])
def test_a_reply_cut_on_part_of_a_plain_opening_where_a_call_opens_ends_inside_that_call(follows_text):
    boundary = dataclasses.replace(_PLAIN_OPENING, follows_text=follows_text)
    text, acc = _drawn_generation({0: b"{"}, [0], set(), boundary)
    assert text == "" and acc.cut_inside_call


def test_a_plain_text_opening_written_inside_the_reply_text_stays_reply_text():
    text, acc = _drawn_generation({0: b"a ", 1: b"{\"n\"", 2: b" b"}, [0, 1, 2], set(), _PLAIN_OPENING)
    assert text == "a {\"n\" b"
    assert not acc.tool_call and acc.stop_reason == "token_limit"


@pytest.mark.parametrize("tag, boundary", [
    (b"<t>", _PLAIN_OPENING),
    (b"</thought>", CallBoundary(call_texts=("{\"n\"", "}"), opens_as_text=True,
                               thought_tags=("<thought>", "</thought>"))),
], ids=["after-a-tag-held-back", "after-a-thought-tag-shown"])
def test_a_plain_text_opening_after_a_tag_and_whitespace_opens_a_call_with_all_its_arguments(
        tag, boundary):
    _, acc = _drawn_generation({0: tag, 1: b" ", 2: b"{\"n\": 1", 3: b"}"}, [0, 1, 2, 3], {0}, boundary)
    assert acc.stop_reason == "tool_call"
    assert acc.tool_call == ": 1"


def test_a_tag_without_text_between_shown_text_and_more_text_leaves_the_reply_whole():
    text, acc = _drawn_generation({0: b"a", 1: b"", 2: b"b"}, [0, 1, 2], {1}, _PLAIN_OPENING)
    assert text == "ab"
    assert acc.stop_reason == "token_limit"


@pytest.mark.parametrize("tags, pieces, order, open_at_the_limit", [
    (("<thought>", "</thought>"), {0: b"<thought>", 1: b"a", 2: b"</thought>"},
     [0, 1, 1, 1, 1, 1, 1, 1], True),
    (("<thought>", "</thought>"), {0: b"<thought>", 1: b"a", 2: b"</thought>", 3: b"b"},
     [0, 1, 2, 3, 3, 3, 3, 3], False),
    (("X", "Y"), {0: b"a", 2: b"Y"}, [0] * 8, False),
    (("\u00a7", "\u00a7"), {0: "\u00a7a\u00a7".encode(), 1: b"b", 2: "\u00a7".encode()},
     [0, 1, 1, 1, 1, 1, 1, 1], False),
], ids=["still-open", "closed-in-its-own-pieces", "one-letter-tags-never-written",
        "same-tag-twice-in-one-piece"])
def test_a_thought_is_closed_for_the_answer_exactly_where_it_is_still_open_at_the_limit(
        tags, pieces, order, open_at_the_limit):
    # Eight draws with two kept for the answer: the closing, token 2, takes the sixth place.
    guard = AnswerGuard(thought_tags=tags, thought_closing=(2,), answer_room_tokens=2)
    _, acc = _drawn_generation(pieces, order, set(), CallBoundary(), guard)
    assert acc.token_ids == (order[:5] + [2] + order[5:7] if open_at_the_limit else order)


@pytest.mark.parametrize("closes_early", [False, True])
def test_where_a_decode_blocks_each_token_is_shown_before_it_is_decoded_and_decoded_before_the_next(
        closes_early):
    fake = FakeContext(PLAIN, ["Hallo Welt"], decode_blocks=True)
    chat = ChatWrapper(Config(), context=fake)
    chat.begin([{"role": "system", "content": "S"}])
    stream = chat.reply("Frage")
    held = []
    for delta in stream:
        held.append((delta, fake.text_of(fake.cache)))
        if closes_early and len(held) == 3:
            stream.close()
            break
    # Each piece reached the reader while the cache did not hold it yet.
    assert all(not cache.endswith("".join(d for d, _ in held[:i + 1]))
               for i, (_, cache) in enumerate(held))
    # Before the next piece, or once the stream ended, it did.
    assert all(held[i + 1][1].endswith("".join(d for d, _ in held[:i + 1]))
               for i in range(len(held) - 1))
    chat.wait_idle(timeout_s=5)
    assert "".join(d for d, _ in held) in fake.text_of(fake.cache)


@pytest.mark.parametrize("cap, closed_at", [(1, 2), (3, 4), (6, 7), (20, None), (None, None)])
def test_a_thought_that_reaches_its_cap_is_closed_there_and_the_answer_follows(cap, closed_at):
    # The opening tag is the thought's first token; ``cap`` tokens inside it pass before the
    # closing, token 2, is written in place of the next draw.
    pieces = {0: b"<thought>", 1: b"a", 2: b"</thought>", 3: b"b"}
    order = [0, 1, 1, 1, 1, 1, 1, 1, 3, 3]
    guard = AnswerGuard(thought_tags=("<thought>", "</thought>"), thought_closing=(2,),
                        max_thought_tokens=cap)
    text, acc = _drawn_generation(pieces, order, set(), CallBoundary(), guard)
    if closed_at is None:
        assert 2 not in acc.token_ids
    else:
        assert acc.token_ids[:closed_at + 1] == order[:closed_at] + [2]
        assert acc.token_ids[closed_at + 1:] == order[closed_at:len(acc.token_ids) - 1]


def test_each_thought_of_a_generation_is_capped_on_its_own():
    pieces = {0: b"<thought>", 1: b"a", 2: b"</thought>", 3: b"b"}
    order = [0, 1, 1, 1, 3, 0, 1, 1, 1, 1, 3, 3]
    guard = AnswerGuard(thought_tags=("<thought>", "</thought>"), thought_closing=(2,),
                        max_thought_tokens=2)
    _, acc = _drawn_generation(pieces, order, set(), CallBoundary(), guard)
    # Two tokens into each thought its closing is written: once for each of the two.
    assert acc.token_ids.count(2) == 2
    first, second = (i for i, token in enumerate(acc.token_ids) if token == 2)
    assert acc.token_ids[first - 2:first] == [1, 1] and acc.token_ids[second - 2:second] == [1, 1]


def test_a_thought_the_cap_closed_leaves_the_rooms_closing_for_a_later_thought():
    pieces = {0: b"<thought>", 1: b"a", 2: b"</thought>", 3: b"b"}
    order = ([0] + [1] * 25 + [3, 0] + [1] * 60)[:40]
    guard = AnswerGuard(thought_tags=("<thought>", "</thought>"), thought_closing=(2,),
                        answer_room_tokens=10, max_thought_tokens=20)
    _, acc = _drawn_generation(pieces, order, set(), CallBoundary(), guard)
    # The first thought is closed at its cap; the second at the room the answer keeps.
    assert acc.token_ids.count(2) == 2
