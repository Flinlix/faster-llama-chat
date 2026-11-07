"""Telling a reply's thought from its answer while the reply streams."""

from __future__ import annotations

from collections.abc import Callable

from .template import Reasoning


class ThoughtFilter:
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
                    self._held = self._held[-len(self._close):]  # a close it may begin
                    break
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
    def in_thought(self) -> bool:
        """Whether the text fed so far ends inside a thought, its closing included."""
        return self._in_thought

    def flush(self, complete: bool = True) -> str:
        """Text held back as a possible start of a thought that never came.

        Args:
            complete: Whether the reply ended of its own accord. One that was
                cut keeps none of it, the held text being the start of a tag.
        """
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


class AnswerOpening:
    """Removes the header a model writes ahead of its answer, where it writes one.

    Only the answer's start is read: until the text is known to begin with the
    header, or not to, it is held back. The answer after a header starts
    unpadded.
    """

    def __init__(self, opening: str) -> None:
        self._opening = opening.strip()
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


def turn_end_test(turn_end: int, turn_end_text: str, reasoning: Reasoning | None,
                  thought: ThoughtFilter | None) -> Callable[[int], bool]:
    """Whether a sampled token ends the reply, besides the end-of-generation tokens.

    Where a thought's closing begins with the turn's end, that token is the
    closing's while the reply is inside a thought.
    """
    if thought is not None and reasoning.thought_close.strip().startswith(turn_end_text):
        return lambda token: token == turn_end and not thought.in_thought
    return lambda token: token == turn_end


def begins_in_thought(reasoning: Reasoning, decoded: str) -> bool:
    """Whether generation after ``decoded`` starts inside a thought."""
    return decoded.rstrip().endswith(reasoning.thought_open.strip())
