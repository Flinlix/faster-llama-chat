"""Writing a tool result within the room a reply leaves: whole where it fits, else it or an earlier result of the same round cut, the cut told to the model in a note.

:class:`ResultCutting` holds no state of its own: it changes the
conversation's state, and records there each result it cut.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Callable

from .conversation_state import ConversationState
from .errors import RAISE_THE_ROOM, ContextOverflowError
from .messages import UNKNOWN_FUNCTION, UNREADABLE_CALL, result_message, with_result_text
from .tools import ToolCall, result_text

_log = logging.getLogger(__name__)

# What a result cut to fit the room left ends with, so that the model reads how much of it it got.
RESULT_CUT_NOTE = "\n(cut: {kept} of {total} characters kept)"


# A note a cut left, read back so that a result cut again counts the text it was cut from.
_A_CUT_NOTE = re.compile(r"\n\(cut: (\d+) of (\d+) characters kept\)\Z")


def cut_result(text: str, kept: int) -> str:
    """``text`` cut to its first ``kept`` characters, :data:`RESULT_CUT_NOTE` after them; a text a cut left is cut from what it kept, the note counting the whole it was cut from."""
    note = _A_CUT_NOTE.search(text)
    if note is not None and int(note.group(1)) == note.start():
        text, total = text[:note.start()], int(note.group(2))
    else:
        total = len(text)
    kept = min(kept, len(text))
    return text[:kept] + RESULT_CUT_NOTE.format(kept=kept, total=total)


def _most_kept(fits: Callable[[int], bool], total: int, guess: int | None = None) -> int:
    """The most characters of ``total`` that a result keeps while ``fits`` holds, ``fits(0)`` holding.

    Searched outward from ``guess`` first, by steps that double, then
    halved between the last two tried, so that a close guess needs few tries.
    """
    low, high = 0, total
    if guess is not None and 0 < guess < total:
        step = 16
        if fits(guess):
            low = guess
            while low + step < total and fits(low + step):
                low, step = low + step, step * 2
            high = min(total, low + step) - 1
            if high == total - 1 and fits(total):
                return total
        else:
            high = guess - 1
            while high - step > 0 and not fits(high - step):
                high, step = high - step, step * 2
            low = max(0, high - step)
    while low < high:
        middle = (low + high + 1) // 2
        low, high = (middle, high) if fits(middle) else (low, middle - 1)
    return low


class ResultCutting:
    """Writes results into the conversation state it was made with, measured with ``measure``, which counts the tokens of messages as :meth:`~llama_chat.rendering.ConversationRendering.length` does."""

    def __init__(self, measure: Callable[[list[dict], bool], int], state: ConversationState,
                 text_within: Callable[[str, int], int]) -> None:
        """``text_within(text, n)`` is how many characters of ``text`` its own first ``n`` tokens spell."""
        self._measure, self._state = measure, state
        self._text_within = text_within

    def write(self, message: dict, owed_after: list[tuple[ToolCall, str]], room: int,
              make_room: Callable[[], bool]) -> bool:
        """Add ``message``, a result, so that the conversation fits in ``room`` with the generation prompt; whether ``make_room`` dropped exchanges for it.

        It is written whole where each result still owed after it, the calls
        and ids ``owed_after``, still fits empty. Otherwise ``make_room`` drops
        the oldest exchanges first; where that is not enough, the room keeps
        space for each later result as the note with nothing kept, unless not
        even this one would fit then, and an earlier result of the round is
        cut where this one cut to nothing does not fit, before this one is cut.

        Raises:
            ContextOverflowError: If not even the result cut to nothing fits.
        """
        messages, text = self._state.messages, message["content"]
        messages += [message] + [self._held_for(later, later_id, "") for later, later_id in owed_after]
        at = len(messages) - len(owed_after) - 1
        dropped = False
        if not self._fits(at, text, room):
            # Held cut to nothing instead, so that what this one is cut to leaves room for
            # each of them, its note included.
            del messages[at + 1:]
            messages += [self._held_for(later, later_id, cut_result(text, 0))
                         for later, later_id in owed_after]
            dropped = make_room()
            messages = self._state.messages
            at = len(messages) - len(owed_after) - 1
            if not self._fits(at, text, room):
                empty = cut_result(text, 0)
                if text and not self._fits(at, empty, room):
                    del messages[at + 1:]       # this one first, if only it fits
                if text and not self._fits(at, empty, room):
                    self._cut_an_earlier_result(at, empty, room)
                if not text or not self._fits(at, empty, room):
                    raise ContextOverflowError(
                        "not even the result cut to nothing leaves room for the reply in what "
                        f"is left of the context; {RAISE_THE_ROOM}")
                kept = _most_kept(lambda n: self._fits(at, cut_result(text, n), room), len(text),
                                  self._guess(at, text, room))
                messages[at] = with_result_text(message, cut_result(text, kept))
                self._note(self._call_answered(at), kept, len(text), "the result", "to fit")
        del messages[at + 1:]
        return dropped

    def cut_to_fit(self, room: int, held: set[int]) -> None:
        """Cut the results of the newest exchange, the newest first, until the conversation fits in ``room`` with the generation prompt, as a result written in a running conversation is cut; none of those at ``held``, which a generated turn's tokens stand for.

        It is measured with the generation prompt, as a reply follows only
        where the conversation ends on a question or a result.
        """
        messages = self._state.messages
        asked = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=0)
        for at in range(len(messages) - 1, asked, -1):
            message = messages[at]
            if message["role"] != "tool" or at in held or not message["content"]:
                continue
            text = message["content"]
            if self._fits(at, text, room):
                return
            if not self._fits(at, cut_result(text, 0), room):
                messages[at] = with_result_text(message, cut_result(text, 0))
                self._note(self._call_answered(at), 0, len(text), "a result", "at the begin")
                continue
            kept = _most_kept(lambda n: self._fits(at, cut_result(text, n), room), len(text),
                              self._guess(at, text, room))
            messages[at] = with_result_text(message, cut_result(text, kept))
            self._note(self._call_answered(at), kept, len(text), "a result", "at the begin")
            return

    def _held_for(self, later: ToolCall, later_id: str, text: str) -> dict:
        """The result room is kept for, for a call owed after the one written: ``text``, or for a call that could not be read the failure it is answered with, which is not cut."""
        if later.status == "unreadable" and self._state.unreadable_result is not None:
            return result_message(UNKNOWN_FUNCTION, later_id,
                                  result_text(self._state.unreadable_result))
        return result_message(later.name, later_id, text)

    def _fits(self, at: int, content: str, room: int) -> bool:
        """Whether the conversation, its message at ``at``, a result, holding ``content``, fits in ``room`` with the generation prompt."""
        return self._length_with(at, content) <= room

    def _length_with(self, at: int, content: str) -> int:
        """How many tokens the conversation takes with the generation prompt, its message at ``at``, a result, holding ``content``."""
        messages = self._state.messages
        messages[at] = {**messages[at], "content": content}
        return self._measure(messages, True)

    def _guess(self, at: int, text: str, room: int) -> int:
        """How many characters of ``text``, the result at ``at``, fit in ``room`` as its own tokens count, with the note of a cut to nothing."""
        return self._text_within(text, max(0, room - self._length_with(at, cut_result(text, 0))))

    def _cut_an_earlier_result(self, at: int, content: str, room: int) -> None:
        """Cut a result written whole earlier in this round, the newest one that makes room for the one at ``at`` holding ``content``, where one does."""
        messages = self._state.messages
        calling = max(i for i in range(at) if messages[i].get("tool_calls"))
        cut_ids = {call.id for call in self._state.results_cut}
        for j in range(at - 1, calling, -1):
            earlier = messages[j]
            if earlier["role"] != "tool" or earlier["tool_call_id"] in cut_ids:
                continue
            whole = earlier["content"]

            def fits(kept: int) -> bool:
                messages[j] = with_result_text(earlier, cut_result(whole, kept))
                return self._fits(at, content, room)

            if not fits(0):
                messages[j] = earlier
                continue
            kept = _most_kept(fits, len(whole))
            messages[j] = with_result_text(earlier, cut_result(whole, kept))
            self._note(self._call_answered(j), kept, len(whole), "an earlier result",
                       "to make room")
            return

    def _call_answered(self, at: int) -> ToolCall:
        """The call the result at ``at`` answers."""
        messages = self._state.messages
        answered = messages[at]["tool_call_id"]
        calling = max(i for i in range(at) if messages[i].get("tool_calls"))
        call, = [c for c in messages[calling]["tool_calls"] if c["id"] == answered]
        if UNREADABLE_CALL in call:
            return ToolCall("", {}, call["id"], "unreadable")
        return ToolCall(call["function"]["name"], copy.deepcopy(call["function"]["arguments"]),
                        call["id"])

    def _note(self, call: ToolCall, kept: int, total: int, what: str, why: str) -> None:
        """Record that ``call``'s result was cut to ``kept`` of its ``total`` characters."""
        self._state.results_cut.append(call)
        self._state.changes.tool_results_cut.append(call)
        _log.info("%s of %s was cut to %d of %d characters %s", what, call.name, kept, total, why)
