"""A reply's text as it streams: its thought, a header meant for another reader, and stop strings."""

from __future__ import annotations

import bisect
import codecs
from collections.abc import Callable

from .reply import StopReason
from .template import Reasoning, common_prefix_length, common_suffix_length


class ReplyText:
    """A reply's answer as it streams: its thought and any header for another reader taken out, then its stop strings matched.

    Args:
        reasoning: The model's thought tags; ``None`` for a model that has none.
        answer_opening: What a past reply's tag holds that the generation
            prompt leaves to the model.
        header: The start and end of a header naming the reply's reader, as
            :func:`reply_header` reads them; ``None`` where there is none.
        prompt_text: The template text the generation follows, which says
            whether it opens in a thought.
        stops: Stop strings, matched against the answer only.
        tag_spans: Where each of the model's own tags stands in the text fed,
            as it streams; ``None`` where that is not known, and every text a
            tag spells counts as the tag.

    Once answer text has passed, a thought opens only at the model's own tag,
    not at its letters. A tag the model writes is never shown, except, before
    answer text has passed, one that is part of a thought's tags or of the
    answer's opening, which the filters read.
    """

    def __init__(self, reasoning: Reasoning | None, answer_opening: str,
                 header: tuple[str, str] | None, prompt_text: str, stops: list[str],
                 tag_spans: list[tuple[int, int]] | None = None) -> None:
        self._reasoning = reasoning
        self._opens_in_thought = reasoning is not None and begins_in_thought(reasoning, prompt_text)
        framing = [text.strip() for text in (answer_opening, *(header or ()),
                                             *((reasoning.thought_open, reasoning.thought_close)
                                               if reasoning else ())) if text.strip()]
        self._frames = lambda tag: any(tag.strip() in text for text in framing)
        self._thought = (_ThoughtFilter(reasoning, self._opens_in_thought, tag_spans, self._frames)
                         if reasoning is not None else None)
        self._opening = _AnswerOpening(answer_opening, header)
        self._stops = _AnswerStops(stops)
        self._tag_spans = tag_spans
        self._fed = 0               # how much text was fed
        self._hides_tags = False    # set once answer text has passed

    def feed(self, piece: str) -> str:
        """The part of ``piece`` that is certainly answer, possibly empty."""
        at, self._fed = self._fed, self._fed + len(piece)
        if self._thought:
            text = self._thought.feed(piece)
        else:
            text = without_tags(piece, at, self._tag_spans, None if self._hides_tags else self._frames)
        shown = self._stops.feed(self._opening.feed(text))
        if not self._hides_tags and self._tag_spans is not None and self.answer.strip():
            self._hides_tags = True
            if self._thought:
                self._thought.hides_tags = True
        return shown

    def finish(self, stop_reason: StopReason) -> str:
        """The answer still held back, once the generation has ended for ``stop_reason``."""
        # A cut reply may end on the start of a thought's tag.
        complete = stop_reason not in ("interrupted", "token_limit")
        last = self._opening.feed(self._thought.flush(complete) if self._thought else "")
        # A header held back at a call is the call's, not the answer's.
        header_done = complete and stop_reason != "tool_call"
        return self._stops.flush(last + self._opening.flush(header_done))

    @property
    def answer(self) -> str:
        """The answer passed on so far."""
        return self._stops.text

    @property
    def opens_in_thought(self) -> bool:
        """Whether the text fed begins inside a thought, its opening written with the prompt."""
        return self._opens_in_thought

    @property
    def thought(self) -> str:
        """The thought fed so far, its tags and surrounding whitespace left out."""
        return self._thought.thought if self._thought else ""

    @property
    def thought_pieces(self) -> int:
        """How many of the pieces fed held thought."""
        return self._thought.thought_pieces if self._thought else 0

    @property
    def met_stop_string(self) -> bool:
        """Whether a stop string ended the answer."""
        return self._stops.stopped

    def turn_end_test(self, turn_end: int, turn_end_text: str, generated: list[int],
                      token_bytes: Callable[[int], bytes],
                      is_tag: Callable[[int], bool]) -> Callable[[int], bool]:
        """Whether a sampled token ends the reply, besides the end-of-generation tokens.

        Where a thought's closing begins with the turn's end, that token is the
        closing's while ``generated``, every token generated before it, ends
        inside a thought, whether or not its text was fed yet.
        """
        tags = self._reasoning
        if tags is None or not tags.thought_close.strip().startswith(turn_end_text):
            return lambda token: token == turn_end
        spans: list[tuple[int, int]] = []
        thought = _ThoughtFilter(tags, self._opens_in_thought, spans)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        read = fed = 0

        def ends(token: int) -> bool:
            nonlocal read, fed
            for earlier in generated[read:]:
                text = decoder.decode(token_bytes(earlier))
                if text and is_tag(earlier):
                    spans.append((fed, fed + len(text)))
                fed += len(text)
                thought.feed(text)
            read = len(generated)
            return token == turn_end and not thought.in_thought
        return ends


class _ThoughtFilter:
    """Passes on a reply's text as it streams, all but its thoughts.

    The tags are matched without their surrounding whitespace, which a model
    does not always reproduce, and the answer after a thought starts unpadded.
    Whitespace at the reply's very start is held until what follows says whose
    it is: a thought's tag makes it the tag's, answer text the answer's.

    Once answer text has begun, a thought opens only where a tag of the
    model's stands in its opening's text, where ``tag_spans`` says where they
    stand.

    Args:
        reasoning: The model's thought tags.
        inside: Whether the text begins inside a thought, its opening tag
            having been decoded with the generation prompt.
        tag_spans: As :class:`ReplyText` takes them.
        frames: Whether a tag's text is part of the tags that frame a thought
            or an answer, kept in the answer until ``hides_tags``.

    Attributes:
        thought_pieces: How many of the pieces fed held thought.
        hides_tags: Whether every tag of the model's in the answer is left
            out, a framing one too.
    """

    def __init__(self, reasoning: Reasoning, inside: bool = False,
                 tag_spans: list[tuple[int, int]] | None = None,
                 frames: Callable[[str], bool] = lambda tag: True) -> None:
        self._open = reasoning.thought_open.strip()
        self._close = reasoning.thought_close.strip()
        self._tag_spans = tag_spans
        self._frames = frames
        self._held = ""             # text not yet known to be thought or answer
        self._held_at = 0           # where the held text starts in the text fed
        self._answered = False      # whether answer text has begun
        self.hides_tags = False
        self._in_thought = inside
        self._uncounted_held = 0    # pieces the held text came from, not yet counted
        self._strips_answer_start = False   # set after a thought
        self._started = inside      # whether an answer or a thought has begun
        self._leading_space = ""    # whitespace at the reply's start, held
        self._thought: list[str] = []
        self.thought_pieces = 0

    def feed(self, text: str) -> str:
        """The part of ``text`` that is answer, possibly empty."""
        self._held += text
        self._uncounted_held += 1
        shown, counted = [], self._in_thought
        while True:
            if self._in_thought:
                end = self._held.find(self._close)
                if end < 0:
                    keep = len(self._close)                      # a close it may begin
                    self._thought.append(self._take(len(self._held) - keep))
                    break
                self._thought.append(self._take(end))
                self._take(len(self._close))
                self._in_thought, self._strips_answer_start = False, True
                continue
            start = self._opening_at()
            if start < 0:
                keep = self._open_prefix_length()
                shown.append(self._answer(len(self._held) - keep))
                break
            shown.append(self._answer(start, before_thought=True))
            self._take(len(self._open))
            if not counted:
                # The earlier pieces the opening tag began in; this one below.
                self.thought_pieces += self._uncounted_held - 1
            self._in_thought = counted = True
        if counted:
            self.thought_pieces += 1
            self._uncounted_held = 0
        elif not self._held:
            self._uncounted_held = 0
        return "".join(shown)

    @property
    def thought(self) -> str:
        """The thought text fed so far, its tags and surrounding whitespace left out."""
        return "".join(self._thought).strip()

    @property
    def in_thought(self) -> bool:
        """Whether the text fed so far ends inside a thought, its closing included."""
        return self._in_thought

    def flush(self, complete: bool = True) -> str:
        """Text held back as a possible start of a thought that never came.

        Args:
            complete: Whether the reply ended of its own accord. One that was
                cut keeps none of it, the held text being the start of a tag.
        """
        if self._in_thought:
            self._thought.append(self._take(len(self._held)))
        if self._in_thought or not complete:
            self._take(len(self._held))
        return self._answer(len(self._held)) + (self._leading_space if complete else "")

    def _take(self, n: int) -> str:
        """The first ``n`` characters of the held text, no longer held."""
        n = max(0, n)
        taken, self._held = self._held[:n], self._held[n:]
        self._held_at += len(taken)
        return taken

    def _opening_at(self) -> int:
        """Where in the held text a thought opens, -1 where none does."""
        at = self._held.find(self._open)
        while at >= 0 and self._answered and self._tag_spans is not None \
                and not _holds_tag(self._tag_spans, self._held_at + at,
                                   self._held_at + at + len(self._open)):
            at = self._held.find(self._open, at + 1)        # its letters, written as text
        return at

    def _answer(self, n: int, before_thought: bool = False) -> str:
        """The first ``n`` held characters as answer, with the whitespace rules of the reply's start."""
        at = self._held_at
        text = self._take(n)
        text = without_tags(text, at, self._tag_spans, None if self.hides_tags else self._frames)
        if not self._started:
            if before_thought and not text.strip():
                self._leading_space, self._started = "", True   # the tag's
                return ""
            if not text.strip():
                self._leading_space += text
                return ""
            text, self._leading_space = self._leading_space + text, ""
            self._started = True
        if self._strips_answer_start:
            text = text.lstrip()
            self._strips_answer_start = not text
        self._answered = self._answered or bool(text.strip())
        return text

    def _open_prefix_length(self) -> int:
        """How long an end of the held text is that could begin a thought."""
        for n in range(min(len(self._held), len(self._open) - 1), 0, -1):
            if self._open.startswith(self._held[-n:]):
                return n
        return 0


class _AnswerOpening:
    """Removes the header a model writes ahead of its answer, where it writes one.

    Only the answer's start is read: until the text is known to begin with the
    header, or not to, it is held back. The answer after a header starts
    unpadded. Given ``header``, the start and end every reply's header shares,
    a header naming another reader is held until it ends and never shown.
    """

    def __init__(self, opening: str, header: tuple[str, str] | None = None) -> None:
        self._opening = opening.strip()
        self._header = header
        self._held = ""
        self._done = not self._opening
        self._strips = False        # set once a header was removed

    def feed(self, text: str) -> str:
        """The part of ``text`` that is answer, possibly empty."""
        if self._done:
            return self._unpadded(text)
        self._held += text
        start = self._held.lstrip()
        if start.startswith(self._opening):
            self._done, self._strips = True, True
            return self._unpadded(start[len(self._opening):])
        if self._opening.startswith(start):
            return ""
        if self._header and start.startswith(self._header[0]):
            if self._header[1] not in start:
                return ""           # a header for another reader, not yet ended
            self._done, self._strips = True, True
            return self._unpadded(start.split(self._header[1], 1)[1])
        self._done = True
        return self._held

    def flush(self, complete: bool = True) -> str:
        """What was held, once the answer has ended; nothing of a reply that was cut."""
        held = "" if self._done or not complete else self._held
        self._held, self._done = "", True
        return held

    def _unpadded(self, text: str) -> str:
        if self._strips:
            text = text.lstrip()
            self._strips = not text
        return text


def reply_header(answer_opening: str, reasoning: Reasoning | None,
                 tags: tuple[str, ...] = ()) -> tuple[str, str] | None:
    """The start and end a reply's header shares, where the answer and a thought each have one.

    Neither end divides one of the model's ``tags``: an end that would is
    narrowed to the tags whole. ``None`` where a template routes no reply
    through a header naming its reader.
    """
    if reasoning is None or not answer_opening.strip():
        return None
    answer, thought = answer_opening.strip(), reasoning.thought_open.strip()
    start, end = common_prefix_length(answer, thought), common_suffix_length(answer, thought)
    start = _outside_tags(answer, start, tags, backwards=True)
    end = len(answer) - _outside_tags(answer, len(answer) - end, tags, backwards=False)
    if not start or not end or start + end >= min(len(answer), len(thought)):
        return None     # either opening leaves no reader between the two
    return answer[:start], answer[len(answer) - end:]


def _outside_tags(text: str, at: int, tags: tuple[str, ...], backwards: bool) -> int:
    """``at``, moved to the start (``backwards``) or the end of a tag of ``tags`` that it falls inside in ``text``."""
    for tag in tags:
        found = text.find(tag)
        while found >= 0:
            if found < at < found + len(tag):
                return found if backwards else found + len(tag)
            found = text.find(tag, found + 1)
    return at


def without_tags(text: str, at: int, tag_spans: list[tuple[int, int]] | None,
                 keeps: Callable[[str], bool] | None = None) -> str:
    """``text``, which starts at ``at`` in the text fed, without the model's tags ``tag_spans`` places in it, but those whose text ``keeps``."""
    if not tag_spans or not text:
        return text
    end = at + len(text)
    kept, cursor = [], at
    for start, stop in tag_spans[max(0, bisect.bisect_left(tag_spans, (at,)) - 1):]:
        if start >= end:
            break
        if stop <= at or keeps is not None and keeps(text[max(start, at) - at:min(stop, end) - at]):
            continue
        kept.append(text[cursor - at:max(start, cursor) - at])
        cursor = max(cursor, min(stop, end))
    kept.append(text[cursor - at:])
    return "".join(kept)


def _holds_tag(tag_spans: list[tuple[int, int]], start: int, end: int) -> bool:
    """Whether a tag of ``tag_spans`` overlaps the text from ``start`` to ``end``."""
    before = bisect.bisect_left(tag_spans, (end,))
    return before > 0 and tag_spans[before - 1][1] > start


def begins_in_thought(reasoning: Reasoning, decoded: str) -> bool:
    """Whether generation after ``decoded`` starts inside a thought."""
    return decoded.rstrip().endswith(reasoning.thought_open.strip())


def earliest(text: str, needles: list[str], search_from: int = 0) -> int | None:
    """Where the earliest of ``needles`` begins in ``text``, or ``None``.

    ``search_from`` skips text already known to begin none of them.
    """
    found: int | None = None
    for needle in needles:
        if not needle:
            continue
        at = text.find(needle, search_from)
        if at != -1 and (found is None or at < found):
            found = at
    return found


def partial_match_at(text: str, needle: str) -> int:
    """Where the longest end of ``text`` that begins ``needle`` starts, ``len(text)`` where none does."""
    for at in range(max(0, len(text) - len(needle) + 1), len(text)):
        if needle.startswith(text[at:]):
            return at
    return len(text)


class _AnswerStops:
    """Collects an answer as it streams, ending it at the earliest stop string.

    Text that could still begin a stop string is held back until it is known
    not to. Without stop strings everything passes at once.
    """

    def __init__(self, stops: list[str]) -> None:
        self._stops = [s for s in stops if s]
        self._held = ""
        self._passed: list[str] = []
        self.stopped = False

    @property
    def text(self) -> str:
        """The answer passed on so far."""
        return "".join(self._passed)

    def feed(self, piece: str) -> str:
        """The part of ``piece`` that is certainly answer, possibly empty."""
        if not self._stops:
            self._passed.append(piece)
            return piece
        self._held += piece
        hit = earliest(self._held, self._stops)
        if hit is not None:
            self.stopped = True
            return self._pass(self._held[:hit], "")
        cut = min(partial_match_at(self._held, stop) for stop in self._stops)
        return self._pass(self._held[:cut], self._held[cut:])

    def flush(self, last: str = "") -> str:
        """Everything still held, with ``last`` appended, once the answer has ended."""
        if not self._stops:
            return self.feed(last)
        shown = self.feed(last)
        if self.stopped:
            return shown
        return shown + self._pass(self._held, "")

    def _pass(self, shown: str, held: str) -> str:
        self._held = held
        self._passed.append(shown)
        return shown
