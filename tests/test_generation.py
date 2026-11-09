"""Generating on the engine's cache context: calls found as they are generated, cancelling."""

from llama_chat.config import Config
from llama_chat.generation import generate
from llama_chat.template import CallBoundary
from llama_chat.tools import ToolCall
from llama_chat.wrapper import ChatWrapper
from tests.fake_context import FakeContext
from tests.wrappers import cache_tokens, rows


# These drive the generation loop over KVContext's own steps on a stubbed
# backend, so that what reaches the engine is pinned along with the loop.
def _generating(pieces: dict, order: list, eog: int | None = None):
    """A KVContext that samples ``order`` and renders each token from ``pieces``."""
    from types import SimpleNamespace

    from llama_chat.context import KVContext

    sampled = iter(order)
    ctx = object.__new__(KVContext)
    ctx._backend = SimpleNamespace(
        sample=lambda sampler, ctx: next(sampled),
        is_eog=lambda vocab, token: token == eog,
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
    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 10, out=acc, boundary=CallBoundary(tool_tokens=_TAGS))

    assert list(stream) == ["hi "]      # the call's own text is never yielded
    assert acc.stop_reason == "tool_call"
    assert acc.tool_call == "fn:clock{}"
    assert acc.text == "hi "
    assert acc.token_ids == [0, 1, 2, 3]   # yet all of it reached the cache


def test_the_held_back_tail_is_released_when_a_call_opens():
    """A stop string holds text back; the call is not what it was waiting for."""
    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    # "ZZZZ" is longer than "hi ", so nothing can be emitted before the call.
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 10, out=acc, boundary=CallBoundary(tool_tokens=_TAGS))
    assert list(stream) == ["hi "]
    assert acc.text == "hi "            # nothing of the reply was lost


def test_closing_the_stream_as_a_call_opens_still_reads_as_interrupted():
    """The release above is a yield like any other, so an early close counts there.

    It used to sit outside the ``GeneratorExit`` handler, and a consumer that
    closed exactly then saw ``stop_reason == "token_limit"``.
    """
    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 10, out=acc, boundary=CallBoundary(tool_tokens=_TAGS))
    assert next(stream) == "hi "
    stream.close()
    assert acc.stop_reason == "interrupted"


def test_a_call_the_token_cap_cuts_short_is_dropped():
    """Truncation mid-call loses the call, not the reply, and closes normally."""
    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2]), 0, 3, out=acc, boundary=CallBoundary(tool_tokens=_TAGS))
    assert list(stream) == ["hi "]
    assert acc.stop_reason == "token_limit"
    assert acc.tool_call == ""          # half a call is no call
    assert acc.text == "hi "            # and none of it leaked into the reply


def test_without_tool_tokens_a_call_tag_is_ordinary_text():
    """A model that cannot call tools has its tags yielded like any other text."""
    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 1, 2, 3]), 0, 4, out=acc)
    assert "".join(stream) == "hi <call>fn:clock{}</call>"
    assert acc.stop_reason == "token_limit"


# The two ways a stream ends besides a call, and what reaches the cache
# before it stops.
def test_end_of_generation_stops_before_the_token_reaches_the_cache():
    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    stream = generate(_generating(_PIECES, [0, 4], eog=4), 0, 10, out=acc)
    assert "".join(stream) == "hi "
    assert acc.stop_reason == "end_of_turn"
    assert acc.token_ids == [0]     # the end token is recognised, never decoded


def test_cancelling_ends_the_stream_within_one_token():
    import threading

    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    cancel = threading.Event()
    ctx = _generating(_PIECES, [0, 4, 4])
    # Set it while the first token renders; the loop checks before sampling the
    # next one, so exactly one token makes it in.
    render = ctx._backend.token_to_piece_bytes
    ctx._backend.token_to_piece_bytes = lambda v, t: (cancel.set(), render(v, t))[1]

    stream = generate(ctx, 0, 10, out=acc, cancel=cancel)
    assert "".join(stream) == "hi "
    assert acc.stop_reason == "interrupted"
    assert acc.token_ids == [0]


def test_the_held_back_tail_is_flushed_when_generation_runs_out():
    from llama_chat.cache_context import GenerationAccumulator

    acc = GenerationAccumulator()
    # A stop string this long holds back three characters at every step, so
    # without the final flush the reply would end short.
    stream = generate(_generating({0: b"abc", 1: b"def"}, [0, 1]), 0, 2, out=acc)
    assert "".join(stream) == "abcdef"
    assert acc.stop_reason == "token_limit"
    assert acc.text == "abcdef"


def test_a_resumed_reply_gathers_its_text_into_one_message():
    """One turn's text is what a reader saw, across both generations of it.

    Whatever persists the conversation reads it back with the resumed prose,
    not only what came before the call.
    """
    fake = FakeContext(calls_tools=True, gen_text="neun Uhr")
    wrapper = ChatWrapper(Config(context_size=512), context=fake)
    wrapper.begin("persona")
    fake.call_script = ["fn:clock{}"]
    list(wrapper.reply("wie spät?", calls="yield"))
    assert rows(wrapper)[-1]["text"] == ""     # nothing visible yet

    wrapper.inject_tool_result(ToolCall("clock", {}), {"time": "09:00"})
    fake.call_script = []
    resumed = "".join(e for e in wrapper.reply(calls="yield") if isinstance(e, str))

    row = rows(wrapper)[-1]
    assert resumed and row["text"] == resumed       # every visible delta, once
    assert "clock" not in row["text"]               # the call is not reply text
    assert "09:00" not in row["text"]               # nor is the result
    held = fake.text_of(fake.cache)
    assert "clock" in held and "09:00" in held      # yet both reached the cache
    assert wrapper.conversation()[1][-1].steps[-1] == resumed   # and that is what is read back
    assert fake.cache == cache_tokens(wrapper)
