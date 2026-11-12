"""A generation on a cache context, token by token: its text streamed, a call told apart, and where it stops.

The loop draws, decodes and reads each token through the steps every cache
context answers, so it runs the same over any of them.
"""

from __future__ import annotations

import bisect
import codecs
import threading
from collections.abc import Callable, Generator, Iterator

from .cache_context import CacheContext, GenerationAccumulator
from .config import Sampling
from .reply_text import earliest, partial_match_at
from .template import CallBoundary


class _TextBuffer:
    """Decoded text with a cursor tracking how much has been emitted.

    Positions count from the text's start, but only the text not yet emitted
    is kept, so no step costs more as the text grows.
    """

    def __init__(self) -> None:
        self._kept = ""             # the text from ``_emitted`` on
        self._emitted = 0
        self._content_end = 0       # where the whole text ends, trailing whitespace left out
        self._emitted_content_end = 0   # the same for the emitted text

    def append(self, text: str) -> None:
        stripped = text.rstrip()
        if stripped:
            self._content_end = len(self) + len(stripped)
        self._kept += text

    def emit(self, upto: int) -> str | None:
        if upto <= self._emitted:
            return None
        delta, self._kept = self._kept[:upto - self._emitted], self._kept[upto - self._emitted:]
        stripped = delta.rstrip()
        if stripped:
            self._emitted_content_end = self._emitted + len(stripped)
        self._emitted = upto
        return delta

    def find(self, needles: list[str], search_from: int) -> int | None:
        """Where the earliest of ``needles`` begins at or after ``search_from``, or ``None``."""
        found = earliest(self._kept, needles, max(0, search_from - self._emitted))
        return None if found is None else self._emitted + found

    def startswith(self, prefix: str, at: int) -> bool:
        return self._kept.startswith(prefix, at - self._emitted)

    def since(self, at: int) -> str:
        """The text from ``at`` on, which must not be emitted yet."""
        assert at >= self._emitted, "the text before the cursor is no longer kept"
        return self._kept[at - self._emitted:]

    def partial_match(self, needle: str) -> int:
        """Where the longest end of the text not yet emitted that begins ``needle`` starts, ``len(self)`` where none does."""
        return self._emitted + partial_match_at(self._kept, needle)

    def content_end(self, at: int) -> int:
        """Where the text before ``at`` ends, trailing whitespace left out; ``at`` must not be emitted yet."""
        if at == len(self):
            return self._content_end
        assert at >= self._emitted, "the text before the cursor is no longer kept"
        stripped = self._kept[:at - self._emitted].rstrip()
        return self._emitted + len(stripped) if stripped else self._emitted_content_end

    @property
    def emitted(self) -> int:
        return self._emitted

    def __len__(self) -> int:
        return self._emitted + len(self._kept)


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
            ``stop_reason == "tool_call"``, or where the turn ends inside it,
            closed as the template closes one; one the reply cap cuts short is
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
    call_tail = ""         # the end of the call's text a closing may begin in
    # A call's texts are matched without the whitespace around them, as a thought's tags are.
    open_text, close_text = ((text.strip() for text in boundary.call_texts)
                             if boundary.call_texts else (None, None))
    call_opens_as_text = boundary.opens_as_text and open_text is not None
    marks = [mark for mark in (open_text, boundary.turn_end_text) if mark]
    opening_owns_space = open_text is not None and boundary.call_texts[0][:1].isspace()
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
            close_text is not None and close_text in call_tail
            + context.token_bytes(tok).decode("utf-8", errors="replace")))
        if ends and not closes_call and call is not None:
            # Ended inside a call: the call ends here, closed as the template closes one.
            closing = ([call_close] if close_text is None else
                       context.tokenize(boundary.call_texts[1], add_special=False,
                                        parse_special=True))
            context.decode_generated(closing, pos)
            pos += len(closing)
            generated.token_ids.extend(closing)
            generated.stop_reason = "tool_call"
            generated.tool_call = "".join(call) if close_text is None else "".join(call).strip()
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
            if close_text is not None:
                call_tail += piece
                if close_text in call_tail:
                    generated.stop_reason = "tool_call"
                    generated.tool_call = "".join(call).split(close_text, 1)[0].strip()
                    break
                call_tail = call_tail[max(0, len(call_tail) - len(close_text) + 1):]
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
            mark_at = visible.find(marks, marks_from)
            while (mark_at is not None and call_opens_as_text
                   and visible.startswith(open_text, mark_at)
                   and not _after_tag(context, starts, generated.token_ids,
                                      visible.content_end(mark_at))):
                marks_from = mark_at + 1        # a mention, not a call
                mark_at = visible.find(marks, marks_from)
            if mark_at is not None:
                upto = mark_at
                if open_text is not None and visible.startswith(open_text, mark_at):
                    call = [visible.since(mark_at + len(open_text))]
                    call_tail = call[0]
                    if opening_owns_space:
                        upto = visible.content_end(mark_at)     # the opening's own
                    if held_tag is not None:
                        upto = held_tag[0]          # the tag the call follows is the call's
                else:
                    generated.stop_reason = "end_of_turn"
                    ended_at_text = True
                    # Its tokens leave the reply, and the caller cuts them
                    # from the cache: the closing is written whole.
                    kept = bisect.bisect_left(starts, mark_at)
                    if kept < len(starts) and starts[kept] == mark_at:
                        generated.cached_beyond = len(generated.token_ids) - kept
                        del generated.token_ids[kept:]
                        pos = start_pos + kept
            else:
                upto = len(visible)
                if boundary.turn_end_text:
                    upto = visible.partial_match(boundary.turn_end_text)
                if open_text is not None:
                    opens_at = visible.partial_match(open_text)
                    before = visible.content_end(opens_at)
                    if opening_owns_space:
                        opens_at = before
                    if opens_at < len(visible) and (not call_opens_as_text or _after_tag(
                            context, starts, generated.token_ids, before)):
                        upto = min(upto, opens_at)
            if held_tag is not None and call is None:
                if visible.content_end(len(visible)) > held_tag[1]:
                    following = visible.since(held_tag[1]).lstrip()
                    if not open_text.startswith(following[:len(open_text)]):
                        held_tag = None             # no call follows it: it is the reply's
                if held_tag is not None:
                    upto = min(upto, held_tag[0])
        else:
            visible.append(piece)
            upto = len(visible)
        delta = visible.emit(upto)
        if delta:
            generated.text_pieces.append(delta)
            try:
                yield delta
            except GeneratorExit:
                generated.stop_reason = "interrupted"
                raise
        if ended_at_text:
            break
    else:
        generated.stop_reason = "token_limit"
    generated.cut_inside_call = call is not None and generated.stop_reason != "tool_call"

    # Flush the held-back tail unless the reply ended in a call or at its text.
    in_text_call = call is not None and close_text is not None
    if generated.stop_reason != "tool_call" and not ended_at_text and not in_text_call:
        visible.append(decoder.decode(b"", final=True))
        # A reply cut before a call could follow its tag ends before the tag, as a cut
        # thought's tag is dropped.
        cut = generated.stop_reason in ("token_limit", "interrupted") and held_tag is not None
        delta = visible.emit(held_tag[0] if cut else len(visible))
        if delta:
            generated.text_pieces.append(delta)
            yield delta


def _after_tag(context: CacheContext, starts: list[int], tokens: list[int],
               content_end: int) -> bool:
    """Whether text ending at ``content_end``, whitespace after it left out, is empty or ends in a tag."""
    if not content_end:
        return True
    return context.is_special(tokens[bisect.bisect_left(starts, content_end) - 1])
