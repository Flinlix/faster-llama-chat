"""A generation on a cache context, token by token: its text streamed, a call told apart, and where it stops.

The loop draws, decodes and reads each token through the steps every cache
context answers, so it runs the same over any of them.
"""

from __future__ import annotations

import bisect
import codecs
import threading
from dataclasses import dataclass, field
from collections.abc import Callable, Generator

from .cache_context import CacheContext
from .config import DrawnSampling
from .reply import StopReason
from .reply_text import earliest, partial_match_at
from .template import CallBoundary
from .loop_variant import LoopVariant


@dataclass
class GenerationAccumulator:
    """What one generation has produced so far, filled as tokens stream in.

    Every token is decoded into the cache before its text is yielded, so a
    caller that abandons the stream still reads exactly what reached the cache.

    Attributes:
        token_ids: Every token decoded into the cache.
        text_pieces: Visible reply text so far, as it was yielded; :attr:`text`
            joins it.
        stop_reason: One of :data:`~llama_chat.reply.STOP_REASONS`.
        tool_call: What the model generated between its call tags, tags
            excluded, when ``stop_reason`` is ``"tool_call"``. Never part of
            ``text``.
        cached_beyond: Tokens decoded into the cache after ``token_ids`` that
            belong to no reply, such as the first tokens of a closing written
            whole later; the caller cuts them away again.
        cut_inside_call: Whether the reply cap or a cancel ended it inside a
            call, or a call's plain opening ended it where calls are withheld;
            the call is then dropped.
        tag_spans: Where each of the model's own tags it wrote stands in the
            visible text, as its start and end there.
    """

    token_ids: list[int] = field(default_factory=list)
    text_pieces: list[str] = field(default_factory=list)
    stop_reason: StopReason = "token_limit"
    tool_call: str = ""
    cached_beyond: int = 0
    cut_inside_call: bool = False
    tag_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Visible reply text so far."""
        return "".join(self.text_pieces)


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


@dataclass(frozen=True)
class AnswerGuard:
    """What keeps a generation from ending without an answer.

    Attributes:
        end_tokens: Every token that ends the turn where it is drawn.
        min_answer_tokens: Until the generation holds this many tokens of
            answer, or a call, no token of ``end_tokens`` is drawn; ``None``
            for no such floor.
        thought_tags: The opening and closing tags of a thought, whitespace
            around them left out; empty where the model marks none.
        thought_closing: The tokens that close a thought as the template
            writes it.
        answer_room_tokens: The tokens kept for the answer: a thought that
            reaches the rest of the generation's room, at most half of it,
            beside its closing, is closed with ``thought_closing``; ``None``
            keeps none.
    """

    end_tokens: tuple[int, ...] = ()
    min_answer_tokens: int | None = None
    thought_tags: tuple[str, ...] = ()
    thought_closing: tuple[int, ...] = ()
    answer_room_tokens: int | None = None

    def thought_limit(self, max_generated_tokens: int) -> int | None:
        """How many tokens of a generation of at most ``max_generated_tokens`` may pass before a thought still open is closed; ``None`` for no limit."""
        if self.answer_room_tokens is None or len(self.thought_tags) != 2 \
                or not self.thought_closing:
            return None
        kept = min(self.answer_room_tokens, max_generated_tokens // 2) + len(self.thought_closing)
        return max(0, max_generated_tokens - kept)


def generate(
    context: CacheContext, start_pos: int, max_generated_tokens: int, *,
    sampling: DrawnSampling,
    out: GenerationAccumulator | None = None,
    cancel: threading.Event | None = None,
    ends_turn: Callable[[int], bool] | None = None,
    boundary: CallBoundary = CallBoundary(),  # noqa: B008 (frozen, so one instance serves)
    guard: AnswerGuard = AnswerGuard(),  # noqa: B008 (frozen, so one instance serves)
    answer_so_far: Callable[[], str] | None = None,
) -> Generator[str, None, None]:
    """Stream a reply, yielding text deltas as tokens are sampled.

    The caller must have prefilled the generation prompt with
    ``want_logits=True``. The trailing characters are held back while they
    could still begin a call's opening or the text that ends a reply.

    Args:
        start_pos: Cache position of the first generated token.
        out: Accumulator to fill. Pass one in to read what reached the cache
            after abandoning the stream; a fresh one is used otherwise.
        cancel: Checked once per token. Setting it, from any thread, ends
            the generation with ``stop_reason == "interrupted"``.
        ends_turn: Whether a sampled token ends the reply as an
            end-of-generation token does, neither cached nor yielded: the
            first of a closing the model writes in several tags.
        boundary: Where a call opens and closes and where a reply ends in
            text. No part of a call is yielded; generation ends where it
            closes with ``stop_reason == "tool_call"``, or where the turn ends
            inside it, closed as the template closes one; one the reply cap
            cuts short is dropped, and the reply ends with ``"token_limit"``.
        guard: Keeps the generation from ending before it answers, and
            closes a thought in time to leave the answer its room.
        answer_so_far: The answer the text yielded so far holds, as its
            reader reads it, which ``guard``'s floor counts; ``None`` sets no
            floor.

    Yields:
        Visible reply text deltas; their concatenation equals ``out.text``.
    """
    draws = _Draws(context, sampling, boundary.withheld_tokens, guard.end_tokens)
    try:
        yield from _sample_into(
            context, draws, start_pos, max_generated_tokens, out, cancel, ends_turn, boundary,
            guard, answer_so_far)
    finally:
        draws.close()


class _Draws:
    """The tokens a generation draws: from its own sampler, and, where the turn may not end yet, from one that withholds the tokens ending it."""

    def __init__(self, context: CacheContext, sampling: DrawnSampling,
                 withheld: tuple[int, ...], end_tokens: tuple[int, ...]) -> None:
        self._open = lambda extra: context.sampled_tokens(sampling, withheld + extra)
        self._tokens = self._open(())
        self._end_tokens = end_tokens
        self._without_end: Generator[int, None, None] | None = None

    def next(self, may_end: bool, ends: Callable[[int], bool]) -> int:
        """The next token; where not ``may_end``, one that does not end the turn."""
        tok = next(self._tokens)
        if may_end or not self._end_tokens or not ends(tok):
            return tok
        if self._without_end is None:
            self._without_end = self._open(self._end_tokens)
        return next(self._without_end)

    def close(self) -> None:
        self._tokens.close()
        if self._without_end is not None:
            self._without_end.close()


def _sample_into(context: CacheContext, draws: _Draws, start_pos: int,
                 max_generated_tokens: int, out: GenerationAccumulator | None,
                 cancel: threading.Event | None, ends_turn: Callable[[int], bool] | None,
                 boundary: CallBoundary, guard: AnswerGuard,
                 answer_so_far: Callable[[], str] | None):
    """The generation loop of :func:`generate`, drawing from ``draws``."""
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
    close_written = boundary.call_texts[1] if boundary.call_texts else ""
    opens_as_text = boundary.opens_as_text and open_text is not None
    # Where the template writes a call after text, one inside a thought is still a mention.
    thought = (_ThoughtState(*boundary.thought_tags, boundary.starts_in_thought)
               if opens_as_text and boundary.follows_text and len(boundary.thought_tags) == 2
               else None)
    # An opening in plain text counts only after a tag, unless the template writes a call after text.
    call_opens_as_text = opens_as_text and not boundary.follows_text
    marks = [mark for mark in (open_text, boundary.turn_end_text) if mark]
    opening_owns_space = boundary.call_texts is not None and boundary.call_texts[0][:1].isspace()
    starts = []            # where each token's text begins in ``visible``
    ended_at_text = False
    withheld_call = False           # whether a withheld call's plain opening ended the reply
    marks_from = 0         # where the next search for a mark begins
    held_tag = None        # where a tag a call may follow begins and ends, not shown yet
    # The answer guard: whether the generation is inside a thought, and how far it got.
    thinking = (_ThoughtState(*guard.thought_tags, boundary.starts_in_thought)
                if len(guard.thought_tags) == 2 else None)
    thought_limit = guard.thought_limit(max_generated_tokens)
    answer_tokens = 0                   # draws made since the answer holds text
    floor = guard.min_answer_tokens if answer_so_far is not None else None
    closing_thought: list[int] = []     # the closing still to be written in place of draws

    def ends_here(token: int) -> bool:
        return context.ends_generation(token) or (ends_turn is not None and ends_turn(token))

    for _ in range(max(0, max_generated_tokens)):
        if cancel is not None and cancel.is_set():
            generated.stop_reason = "interrupted"
            break
        if len(generated.token_ids) >= max_generated_tokens:
            generated.stop_reason = "token_limit"
            break
        in_thought = thinking is not None and thinking.inside
        if in_thought and call is None and not closing_thought and thought_limit is not None \
                and len(generated.token_ids) >= thought_limit:
            closing_thought = list(guard.thought_closing)
            thought_limit = None            # one closing per generation
        forced = bool(closing_thought)
        if forced:
            tok = closing_thought.pop(0)
        else:
            if floor is not None and answer_tokens < floor and answer_so_far is not None \
                    and (answer_tokens or answer_so_far().strip()):
                answer_tokens += 1
            may_end = call is not None or floor is None or answer_tokens >= floor
            tok = draws.next(may_end, ends_here)
        # A closing written for the model ends nothing, though a tag of it may end a turn.
        ends = not forced and ends_here(tok)
        # A call may close with the token that otherwise ends a turn.
        closes_call = call is not None and (tok == call_close or (
            close_text is not None and close_text in call_tail
            + context.token_bytes(tok).decode("utf-8", errors="replace")))
        if ends and not closes_call and call is not None:
            # Ended inside a call: the call ends here, closed as the template closes one.
            if close_text is None:
                assert call_close is not None, "a call its tag opened closes with its tag"
                closing = [call_close]
            else:
                closing = context.tokenize(close_written, add_special=False, parse_special=True)
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
        if call is None and thinking is not None:
            thinking.read(piece)
        if tok == call_open and floor is not None:
            answer_tokens = floor               # a call is an answer

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
        if tok == call_open and not forced:
            # The held-back tail is ordinary reply text; release it.
            call = []
            upto = len(visible)
        elif marks:
            starts.append(len(visible))
            is_tag = context.is_special(tok)
            _append(visible, piece, is_tag, generated)
            if thought is not None:
                call_opens_as_text = thought.read(piece)
            if call_opens_as_text and is_tag \
                    and piece.strip() not in boundary.thought_tags:
                held_tag = (starts[-1], len(visible))
            mark_at = visible.find(marks, marks_from)
            mentions_variant = LoopVariant("skipping mentions of a call")
            while (mark_at is not None and call_opens_as_text and open_text is not None
                   and visible.startswith(open_text, mark_at)
                   and not _after_tag(context, starts, generated.token_ids,
                                      visible.content_end(mark_at))):
                mentions_variant.passes(len(visible) + 1 - marks_from)
                marks_from = mark_at + 1        # a mention, not a call
                mark_at = visible.find(marks, marks_from)
            if mark_at is not None:
                upto = mark_at
                opens_call = open_text is not None and visible.startswith(open_text, mark_at)
                if opens_call:
                    if opening_owns_space:
                        upto = visible.content_end(mark_at)     # the opening's own
                    if held_tag is not None:
                        upto = held_tag[0]          # the tag the call follows is the call's
                if opens_call and open_text is not None and not boundary.ends_at_call:
                    call = [visible.since(mark_at + len(open_text))]
                    call_tail = call[0]
                else:
                    generated.stop_reason = "end_of_turn"
                    ended_at_text, withheld_call = True, opens_call
                    # Its tokens leave the reply, and the caller cuts them
                    # from the cache: the closing is written whole.
                    kept = bisect.bisect_left(starts, upto)
                    if kept < len(starts) and starts[kept] == upto:
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
            _append(visible, piece, context.is_special(tok), generated)
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
    generated.cut_inside_call = withheld_call or (call is not None
                                                  and generated.stop_reason != "tool_call")

    # Flush the held-back tail unless the reply ended in a call or at its text.
    in_text_call = call is not None and close_text is not None
    if generated.stop_reason != "tool_call" and not ended_at_text and not in_text_call:
        visible.append(decoder.decode(b"", final=True))
        # A reply cut before a call could follow its tag ends before the tag, as a cut
        # thought's tag is dropped.
        cut = generated.stop_reason in ("token_limit", "interrupted") and held_tag is not None
        upto = held_tag[0] if cut and held_tag is not None else len(visible)
        if generated.stop_reason == "token_limit" and call_open is None and open_text is not None:
            # Where no token of its own opens a call, a reply cut at a call's opening text
            # ends inside that call: the opening is the call's, not the answer's.
            # Only part of the opening counts where it stands where a call opens: first in
            # the reply, or after a tag.
            opens_at = visible.partial_match(open_text)
            written = visible.since(opens_at).strip()
            if opens_at < len(visible) and written and (written == open_text or _after_tag(
                    context, starts, generated.token_ids, visible.content_end(opens_at))):
                upto, generated.cut_inside_call = min(upto, opens_at), True
        delta = visible.emit(upto)
        if delta:
            generated.text_pieces.append(delta)
            yield delta


class _ThoughtState:
    """Whether the text generated so far ends inside a thought, read piece by piece."""

    def __init__(self, opening: str, closing: str, inside: bool) -> None:
        self._opening, self._closing, self._inside = opening, closing, inside
        self._tail = ""             # text a tag may still begin in

    @property
    def inside(self) -> bool:
        """Whether the text read so far ends inside a thought."""
        return self._inside

    def read(self, piece: str) -> bool:
        """Read ``piece``; whether the text now ends inside a thought."""
        self._tail += piece
        tags_variant = LoopVariant("reading thought tags")
        while (at := self._tail.find(self._closing if self._inside else self._opening)) >= 0:
            tags_variant.passes(len(self._tail))
            self._tail = self._tail[at + len(self._closing if self._inside else self._opening):]
            self._inside = not self._inside
        keep = max(len(self._opening), len(self._closing)) - 1
        self._tail = self._tail[max(0, len(self._tail) - keep):] if keep else ""
        return self._inside


def _append(visible: _TextBuffer, piece: str, is_tag: bool, generated: GenerationAccumulator) -> None:
    """Add a token's text to ``visible``, noting where it stands where the token is one of the model's tags."""
    if is_tag and piece:
        generated.tag_spans.append((len(visible), len(visible) + len(piece)))
    visible.append(piece)


def _after_tag(context: CacheContext, starts: list[int], tokens: list[int],
               content_end: int) -> bool:
    """Whether text ending at ``content_end``, whitespace after it left out, is empty or ends in a tag."""
    if not content_end:
        return True
    return context.is_special(tokens[bisect.bisect_left(starts, content_end) - 1])
