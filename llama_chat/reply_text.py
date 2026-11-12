"""A reply's text as it streams: its thought, a header meant for another reader, and stop strings."""

from __future__ import annotations

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
        prompt_text: The template text the generation follows, which says
            whether it opens in a thought.
        stops: Stop strings, matched against the answer only.
    """

    def __init__(self, reasoning: Reasoning | None, answer_opening: str, prompt_text: str,
                 stops: list[str]) -> None:
        self._reasoning = reasoning
        self._opens_in_thought = reasoning is not None and begins_in_thought(reasoning, prompt_text)
        self._thought = (_ThoughtFilter(reasoning, self._opens_in_thought)
                         if reasoning is not None else None)
        self._opening = _AnswerOpening(answer_opening, reply_header(answer_opening, reasoning))
        self._stops = _AnswerStops(stops)

    def feed(self, piece: str) -> str:
        """The part of ``piece`` that is certainly answer, possibly empty."""
        text = self._thought.feed(piece) if self._thought else piece
        return self._stops.feed(self._opening.feed(text))

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
                      token_bytes: Callable[[int], bytes]) -> Callable[[int], bool]:
        """Whether a sampled token ends the reply, besides the end-of-generation tokens.

        Where a thought's closing begins with the turn's end, that token is the
        closing's while ``generated``, every token generated before it, ends
        inside a thought, whether or not its text was fed yet.
        """
        tags = self._reasoning
        if tags is None or not tags.thought_close.strip().startswith(turn_end_text):
            return lambda token: token == turn_end
        thought = _ThoughtFilter(tags, self._opens_in_thought)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        read = 0

        def ends(token: int) -> bool:
            nonlocal read
            for earlier in generated[read:]:
                thought.feed(decoder.decode(token_bytes(earlier)))
            read = len(generated)
            return token == turn_end and not thought.in_thought
        return ends


class _ThoughtFilter:
    """Passes on a reply's text as it streams, all but its thoughts.

    The tags are matched without their surrounding whitespace, which a model
    does not always reproduce, and the answer after a thought starts unpadded.
    Whitespace at the reply's very start is held until what follows says whose
    it is: a thought's tag makes it the tag's, answer text the answer's.

    Args:
        reasoning: The model's thought tags.
        inside: Whether the text begins inside a thought, its opening tag
            having been decoded with the generation prompt.

    Attributes:
        thought_pieces: How many of the pieces fed held thought.
    """

    def __init__(self, reasoning: Reasoning, inside: bool = False) -> None:
        self._open = reasoning.thought_open.strip()
        self._close = reasoning.thought_close.strip()
        self._held = ""             # text not yet known to be thought or answer
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
                    self._thought.append(self._held[:-keep])
                    self._held = self._held[-keep:]
                    break
                self._thought.append(self._held[:end])
                self._held = self._held[end + len(self._close):]
                self._in_thought, self._strips_answer_start = False, True
                continue
            start = self._held.find(self._open)
            if start < 0:
                keep = self._open_prefix_length()
                shown.append(self._answer(self._held[:len(self._held) - keep]))
                self._held = self._held[len(self._held) - keep:]
                break
            shown.append(self._answer(self._held[:start], before_thought=True))
            self._held = self._held[start + len(self._open):]
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
            self._thought.append(self._held)
        held, self._held = ("" if self._in_thought or not complete else self._held), ""
        return self._answer(held) + (self._leading_space if complete else "")

    def _answer(self, text: str, before_thought: bool = False) -> str:
        """``text`` as answer, with the whitespace rules of the reply's start."""
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


def reply_header(answer_opening: str, reasoning: Reasoning | None) -> tuple[str, str] | None:
    """The start and end a reply's header shares, where the answer and a thought each have one.

    ``None`` where a template routes no reply through a header naming its reader.
    """
    if reasoning is None or not answer_opening.strip():
        return None
    answer, thought = answer_opening.strip(), reasoning.thought_open.strip()
    start, end = common_prefix_length(answer, thought), common_suffix_length(answer, thought)
    if not start or not end or start + end >= min(len(answer), len(thought)):
        return None     # either opening leaves no reader between the two
    return answer[:start], answer[len(answer) - end:]


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


def partial_match_at(text: str, needle: str, search_from: int = 0) -> int:
    """Where the longest end of ``text`` from ``search_from`` on that begins ``needle`` starts, ``len(text)`` where none does."""
    for at in range(max(search_from, len(text) - len(needle) + 1), len(text)):
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
        self.text = ""
        self.stopped = False

    def feed(self, piece: str) -> str:
        """The part of ``piece`` that is certainly answer, possibly empty."""
        if not self._stops:
            self.text += piece
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
        self.text += shown
        return shown
