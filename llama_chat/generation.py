"""A generation on a cache context, token by token: its text streamed, a call told apart, and where it stops.

The loop draws, decodes and reads each token through the steps every cache
context answers, so it runs the same over any of them.
"""

from __future__ import annotations

import codecs
import threading
from collections.abc import Callable, Generator, Iterator

from .cache_context import CacheContext, GenerationAccumulator
from .config import Sampling
from .reply_text import earliest
from .template import CallBoundary


class _TextBuffer:
    """Decoded text with a cursor tracking how much has been emitted."""

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0

    def append(self, text: str) -> None:
        self._buf += text

    def emit(self, upto: int) -> str | None:
        if upto > self._emitted:
            delta = self._buf[self._emitted:upto]
            self._emitted = upto
            return delta
        return None

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def text(self) -> str:
        return self._buf

    def __len__(self) -> int:
        return len(self._buf)


def generate(
    context: CacheContext, start_pos: int, max_generated_tokens: int,
    out: GenerationAccumulator | None = None,
    cancel: threading.Event | None = None,
    sampling: Sampling | None = None,
    ends_turn: Callable[[int], bool] | None = None,
    boundary: CallBoundary = CallBoundary(),
) -> Iterator[str]:
    """Stream a reply, yielding text deltas as tokens are sampled.

    The caller must have prefilled the generation prompt with
    ``want_logits=True``. Bytes run through an incremental UTF-8 decoder, so
    a codepoint split across tokens is never mangled, and the trailing
    characters are held back while they could still begin a call's opening
    or the text that ends a reply.

    Args:
        context: The cache context drawn from and decoded into.
        start_pos: Cache position of the first generated token.
        max_generated_tokens: Hard cap on tokens to generate.
        out: Accumulator to fill. Pass one in to read what reached the cache
            after abandoning the stream; a fresh one is used otherwise.
        cancel: Checked once per token. Setting it, from any thread, ends
            the generation with ``stop_reason == "interrupted"``.
        sampling: Every value this generation draws with, handed to the
            context.
        ends_turn: Whether a sampled token ends the reply as an
            end-of-generation token does, neither cached nor yielded: the
            first of a closing the model writes in several tags.
        boundary: Where a call opens and closes and where a reply ends in
            text. A call is recognized where it opens, none of it is
            yielded, and generation ends where it closes with
            ``stop_reason == "tool_call"``; one the token cap cuts short is
            dropped, and the reply ends with ``"token_limit"``.

    Yields:
        Visible reply text deltas; their concatenation equals ``out.text``.
    """
    tokens = context.sampled_tokens(sampling, boundary.withheld_tokens)
    try:
        yield from _sample_into(
            context, tokens, start_pos, max_generated_tokens, out, cancel, ends_turn, boundary)
    finally:
        tokens.close()


def _sample_into(context: CacheContext, tokens: Generator[int, None, None], start_pos: int,
                 max_generated_tokens: int, out: GenerationAccumulator | None,
                 cancel: threading.Event | None, ends_turn: Callable[[int], bool] | None,
                 boundary: CallBoundary):
    """The generation loop of :func:`generate`, drawing from ``tokens``."""
    generated = out if out is not None else GenerationAccumulator()
    pos = start_pos
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    visible = _TextBuffer()
    call_open, call_close = boundary.tool_tokens or (None, None)
    call = None            # pieces of the call being generated, once one is
    # A call's texts are matched without the whitespace around them, as a thought's tags are.
    open_text, close_text = ((text.strip() for text in boundary.call_texts)
                             if boundary.call_texts else (None, None))
    call_opens_as_text = boundary.opens_as_text and open_text is not None
    marks = [mark for mark in (open_text, boundary.turn_end_text) if mark]
    held_chars = max([0] + [len(mark) for mark in (*(boundary.call_texts or ()),
                                                   boundary.turn_end_text) if mark])
    starts = []            # where each token's text begins in ``visible``
    ended_at_text = False
    marks_from = 0         # where the next search for a mark begins
    held_tag = None        # where a tag a call may follow begins and ends, not shown yet

    for _ in range(max(0, max_generated_tokens)):
        if cancel is not None and cancel.is_set():
            generated.stop_reason = "interrupted"
            break
        tok = next(tokens)
        ends = context.ends_generation(tok) or (
            ends_turn is not None and ends_turn(tok))
        # A call may close with the token that otherwise ends a turn.
        closes_call = call is not None and (tok == call_close or (
            close_text is not None and close_text in "".join(call)
            + context.token_bytes(tok).decode("utf-8", errors="replace")))
        if ends and not closes_call and call is not None and close_text is not None:
            # Ended inside a call: the call ends here, closed as the template closes one.
            closing = context.tokenize(boundary.call_texts[1], add_special=False,
                                       parse_special=True)
            context.decode_generated(closing, pos)
            pos += len(closing)
            generated.token_ids.extend(closing)
            generated.stop_reason = "tool_call"
            generated.tool_call = "".join(call).strip()
            break
        if ends and not closes_call:
            generated.stop_reason = "end_of_turn"
            break
        # Cache, then record, then surface: this order is what keeps
        # token_ids equal to the cache across an early close.
        context.decode_generated([tok], pos)
        pos += 1
        generated.token_ids.append(tok)
        piece = decoder.decode(context.token_bytes(tok))

        if call is not None:
            if tok == call_close:
                generated.stop_reason = "tool_call"
                generated.tool_call = "".join(call)
                break
            call.append(piece)
            if close_text is not None and close_text in "".join(call):
                generated.stop_reason = "tool_call"
                generated.tool_call = "".join(call).split(close_text, 1)[0].strip()
                break
            continue
        if tok == call_open:
            # The held-back tail is ordinary reply text; release it.
            call = []
            upto = len(visible)
        elif marks:
            starts.append(len(visible))
            visible.append(piece)
            if call_opens_as_text and context.is_special(tok) \
                    and piece.strip() not in boundary.thought_tags:
                held_tag = (starts[-1], len(visible))
            mark_at = earliest(visible.text, marks,
                               search_from=max(visible.emitted, marks_from))
            while (mark_at is not None and call_opens_as_text
                   and visible.text.startswith(open_text, mark_at)
                   and not _after_tag(context, visible.text, starts, generated.token_ids,
                                      mark_at)):
                marks_from = mark_at + 1        # a mention, not a call
                mark_at = earliest(visible.text, marks, search_from=marks_from)
            if mark_at is not None:
                upto = mark_at
                if open_text is not None and visible.text.startswith(open_text, mark_at):
                    call = [visible.text[mark_at + len(open_text):]]
                    if boundary.call_texts[0][:1].isspace():
                        upto = len(visible.text[:mark_at].rstrip())     # the opening's own
                    if held_tag is not None:
                        upto = held_tag[0]          # the tag the call follows is the call's
                else:
                    generated.stop_reason = "end_of_turn"
                    ended_at_text = True
                    # Its tokens leave the reply, and the caller cuts them
                    # from the cache: the closing is written whole.
                    if mark_at in starts:
                        kept = starts.index(mark_at)
                        generated.cached_beyond = len(generated.token_ids) - kept
                        del generated.token_ids[kept:]
                        pos = start_pos + kept
            else:
                upto = len(visible) - (held_chars - 1)
            if held_tag is not None and call is None:
                following = visible.text[held_tag[1]:].lstrip()
                if following and not open_text.startswith(following[:len(open_text)]):
                    held_tag = None             # no call follows it: it is the reply's
                else:
                    upto = min(upto, held_tag[0])
        else:
            visible.append(piece)
            upto = len(visible)
        delta = visible.emit(upto)
        if delta:
            generated.text += delta
            try:
                yield delta
            except GeneratorExit:
                generated.stop_reason = "interrupted"
                raise
        if ended_at_text:
            break
    else:
        generated.stop_reason = "token_limit"

    # Flush the held-back tail unless the reply ended in a call or at its text.
    in_text_call = call is not None and close_text is not None
    if generated.stop_reason != "tool_call" and not ended_at_text and not in_text_call:
        visible.append(decoder.decode(b"", final=True))
        delta = visible.emit(len(visible))
        if delta:
            generated.text += delta
            yield delta


def _after_tag(context: CacheContext, text: str, starts: list[int], tokens: list[int],
               at: int) -> bool:
    """Whether ``at`` is the reply's start or follows a tag, whitespace between."""
    before = len(text[:at].rstrip())
    if not before:
        return True
    last = max(i for i, start in enumerate(starts) if start < before)
    return context.is_special(tokens[last])
