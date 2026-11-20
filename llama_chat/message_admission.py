"""Admitting a new message: refusing one the template does not write next or that cannot fit once the older exchanges go, making way for a question by closing an open turn or joining an unanswered question, and cutting a question too long to its end."""

from __future__ import annotations

import logging
from collections.abc import Callable

from .cache_sync import CacheSync
from .conversation_state import ConversationState
from .history_forms import (joined_question,
                            with_questions_joined)
from .messages import (QUESTION_CUT_MARK, UNKNOWN_FUNCTION, UNREADABLE_CALL,
                       exchange_starts, result_message, system_part, unanswered)
from .rendering import ConversationRendering
from .errors import RAISE_THE_ROOM, ContextOverflowError, TemplateRefusedError, with_template_reason
from .tools import result_text
from .loop_variant import LoopVariant
from .cache_targets import CacheTargets

_log = logging.getLogger(__name__)


# How far the count of a message alone may fall short of its length in the system turn's render.
NEAR_THE_ROOM_TOKENS = 64


INJECT_A_QUESTION_FIRST = "shorten the message or inject a user message first"


def no_room_once_dropped(needed: int, limit: int, remedy: str) -> ContextOverflowError:
    """The refusal of a conversation that renders ``needed`` tokens, over ``limit``, once every exchange that can go is dropped."""
    return ContextOverflowError(
        f"the conversation needs {needed} tokens but {limit} fit, and the messages no user "
        f"message follows cannot be evicted; {remedy}, or {RAISE_THE_ROOM}", needed, limit)


# What a call still owed is told when a question closes its turn.
NOT_RUN = {"error": "not_run", "message": "the call was not run: the user asked something new"}


def open_turns(messages: list[dict]) -> list[tuple[int, int, list[dict]]]:
    """Each turn a call left open before a later question: where its question stands, where the later question stands, and the calls still owed a result."""
    turns: list[tuple[int, int, list[dict]]] = []
    owed: list[dict] = []
    asked, is_open = None, False
    for at, message in enumerate(messages):
        if message["role"] == "user":
            if is_open and asked is not None:
                turns.append((asked, at, owed))
            asked, owed, is_open = at, [], False
        elif message.get("tool_calls"):
            owed, is_open = list(message["tool_calls"]), True
        elif message["role"] == "tool":
            owed = [c for c in owed if c["id"] != message["tool_call_id"]]
        elif message["role"] == "assistant":
            is_open = False
    return turns


class MessageAdmission:
    """Makes way for a new message and checks that it can be written and fit."""

    def __init__(self, rendering: ConversationRendering, state: ConversationState, memory: CacheSync, targets: CacheTargets, joins_questions: Callable[[], bool], question_room: Callable[[], int], turn_open: Callable[[], bool], newest_exchange: Callable[[], list[dict]]) -> None:
        self._rendering = rendering
        self._state = state
        self._memory = memory
        self._targets = targets
        self._joins_questions = joins_questions
        self._question_room = question_room
        self._turn_open = turn_open
        self._newest_exchange = newest_exchange

    def with_open_turns_closed(self, messages: list[dict]) -> list[dict]:
        """``messages`` with each turn left open before a later question closed as that question closes it live: as it stands where the template writes the question after it, else each call still owed told it was not run, then an empty reply after, else the turn taken out and the two questions joined."""
        turns = open_turns(messages)
        if not turns:
            return messages
        system = system_part(messages)
        closed, done = [], 0
        for asked, nxt, owed in turns:
            exchange = messages[asked:nxt]
            results = [result_message(UNKNOWN_FUNCTION if UNREADABLE_CALL in c
                                      else c["function"]["name"], c["id"],
                                      result_text(NOT_RUN)) for c in owed]
            question = messages[nxt]
            out = ([joined_question(exchange[0], question)] if self._joins_questions()
                   else exchange[:1] + [question])
            forms = ((exchange, exchange + [question]),
                     (exchange + results, exchange + results + [question]),
                     (exchange + results + [{"role": "assistant", "content": ""}],
                      exchange + results + [{"role": "assistant", "content": ""}, question]),
                     (exchange[:1], out))
            chosen = next((form for form, written in forms
                           if self._rendering.render(system + written, True) is not None), exchange)
            if chosen is not exchange:
                self._state.changes.turn_closed = True
                self._state.changes.question_joined |= chosen is forms[-1][0] and self._joins_questions()
            closed += messages[done:asked] + chosen
            done = nxt
        closed += messages[done:]
        return with_questions_joined(closed) if self._joins_questions() else closed

    # ----- nothing blocks a question ------------------------------------------------------
    def make_way_for(self, question: dict, limit: int, generation_prompt: bool) -> dict | None:
        """Close a turn a call left open, and join ``question`` to an unanswered one where the template writes no two questions in a row; the joined question, taken out to be added again, or ``None``.

        Where the two joined would not fit in ``limit`` beside the system
        prompt, the unanswered one goes instead, counted as evicted. See
        docs/decisions/0032-every-question-gets-a-reply.md.
        """
        if self._turn_open():
            self._close_open_turn(question)
        newest = self._state.messages[-1] if self._state.messages else None
        # On a template that writes two in a row, one it refuses is refused as it would be alone.
        if newest is None or newest["role"] != "user" or not self._joins_questions() \
                or self._writes_after(question):
            return None
        self._state.messages.pop()
        joined = joined_question(newest, question)
        if self.length_alone(joined, limit, generation_prompt) > limit:
            self._state.changes.n_evicted += 1
            self._state.changes.evicted_for = "room"
            _log.info("an unanswered question went to make room for the one after it: the "
                      "template writes no two in a row, and the two joined do not fit")
            return None
        self._state.changes.question_joined = True
        _log.info("a question was joined to the one before it, which no reply followed: "
                  "the template writes no two in a row")
        return joined

    def _close_open_turn(self, question: dict) -> None:
        """Close the open turn ahead of ``question``: each call still owed is told it was not run, and a turn waiting for its reply gets an empty one where the template needs it; the turn is taken out where the template writes neither."""
        start = len(self._state.messages)
        owed = unanswered(self._state.messages)
        self._state.messages += [result_message(call.name or UNKNOWN_FUNCTION, call_id,
                                                result_text(NOT_RUN))
                                 for call, call_id in owed]
        self._state.results_cut, self._state.called_in_thought = [], False
        self._state.changes.turn_closed = True
        if self._writes_after(question):
            _log.info("a question closed the open turn, %d call(s) told they were not run",
                      len(owed))
            return
        self._state.messages.append({"role": "assistant", "content": ""})
        if self._writes_after(question):
            _log.info("a question closed the open turn with an empty reply, %d call(s) told "
                      "they were not run", len(owed))
            return
        del self._state.messages[start:]
        _log.info("a question took the open turn out: the template writes it after neither "
                  "its results nor an empty reply")
        # Neither is written: the turn goes, from its first message after the newest question.
        asked = max(i for i, m in enumerate(self._state.messages) if m["role"] == "user")
        del self._state.messages[asked + 1:]

    def _writes_after(self, question: dict) -> bool:
        """Whether the template writes ``question`` after the newest exchange as it stands."""
        return self._rendering.render(self._newest_exchange() + [question], True) is not None

    def cut_to_fit(self, question: dict, room: int, what: str, cut_before: bool = False) -> dict:
        """``question``, or, where it does not fit ``room`` beside the system turn, its end, :data:`QUESTION_CUT_MARK` before it, a warning logged.

        Args:
            cut_before: Whether ``question`` is the rest of one cut already,
                its mark taken off, so that the mark stands before it however
                much of it is kept.

        Raises:
            ContextOverflowError: If not even the mark fits, naming ``what``.
        """
        text = question["content"]
        def kept(start: int) -> dict:
            marked = start or cut_before
            return {**question, "content": QUESTION_CUT_MARK + text[start:] if marked else text}
        def fits(start: int) -> bool:
            return self.length_alone(kept(start), room, generation_prompt=True) <= room
        if fits(0):
            return kept(0)
        if not fits(len(text)):
            # Counted as the shortest it is cut to: its mark alone.
            alone = self.length_alone(kept(len(text)), room, generation_prompt=True)
            raise ContextOverflowError(
                f"{what} needs {alone} tokens with the system prompt but {room} fit under the "
                f"eviction threshold beside the reply's room, and the system prompt leaves no "
                f"room for any of it; shorten the system prompt, or "
                f"{RAISE_THE_ROOM}", alone, room)
        low, high = 1, len(text)        # the least start that fits lies in [low, high]
        search_variant = LoopVariant("cutting a question to fit")
        while low < high:
            search_variant.passes(high - low)
            middle = (low + high) // 2
            if fits(middle):
                high = middle
            else:
                low = middle + 1
        _log.warning("%s is longer than the %d tokens a question may take beside the system "
                     "prompt; its first %d of %d characters were cut", what, room, low, len(text))
        return kept(low)

    def length_alone(self, message: dict, limit: int, generation_prompt: bool) -> int:
        """How many tokens ``message`` takes beside the system turn alone, measured in their render only where its count comes within :data:`NEAR_THE_ROOM_TOKENS` of ``limit``."""
        counted = self._state.checks.with_a_short_question + self._rendering.count_tokens(
            message["content"])
        if counted + NEAR_THE_ROOM_TOKENS <= limit:
            return counted
        return self._rendering.length(system_part(self._state.messages) + [message], generation_prompt)

    def cut_newest_question_again(self) -> bool:
        """Cut the newest message, a question, to the room beside the system turn again, as where a date the template writes grew since it was cut; whether it changed. The mark of an earlier cut is not kept twice."""
        question = self._state.messages[-1]
        text = question["content"]
        cut_before = text.startswith(QUESTION_CUT_MARK)
        rest = text[len(QUESTION_CUT_MARK):] if cut_before else text
        cut = self.cut_to_fit({**question, "content": rest}, self._question_room(),
                               "the newest question", cut_before)
        if cut["content"] == text:
            return False
        self._state.messages[-1] = cut
        return True

    def ensure_writes(self, role: str, text: str, generation_prompt: bool) -> None:
        """Refuse a ``role`` message the template would not write next.

        Raises:
            ValueError: If the template refuses the conversation with it.
        """
        # Refusals turn on the newest messages, such as two user turns in a row, so
        # they are read from the newest exchange, not from a render of the whole conversation.
        # A question after an unanswered one was joined to it where the template writes no two.
        messages = self._newest_exchange() + [{"role": role, "content": text}]
        if self._rendering.render(messages, generation_prompt) is None:
            article = "an" if role == "assistant" else "a"
            raise TemplateRefusedError("text", with_template_reason(
                f"this model's template does not write {article} {role} message here",
                self._rendering.refusal_reason(messages, generation_prompt)))

    def ensure_within(self, limit: int, remedy: str, generation_prompt: bool = False) -> None:
        """Refuse a conversation that still renders longer than ``limit`` once every exchange that can go was dropped, measured with the generation prompt where ``generation_prompt``.

        Raises:
            ContextOverflowError: With ``remedy`` as what to do instead.
        """
        needed = self._targets.length(self._state.messages, generation_prompt=generation_prompt)
        if needed > limit:
            raise no_room_once_dropped(needed, limit, remedy)

    def ensure_fits_once_dropped(self, message: dict, limit: int) -> None:
        """Refuse ``message``, which joins the newest exchange, where the conversation with it renders longer than ``limit`` once every older exchange is dropped, before any is.

        Raises:
            ContextOverflowError: Naming what to do instead.
        """
        messages = self._state.messages
        starts = exchange_starts(messages)
        system = system_part(messages)
        newest = starts[-1] if starts else len(system)
        if newest == len(system):
            return          # nothing can go, so the check after the drops costs no undo
        if (len(self._memory.tokens) + self._rendering.count_tokens(message["content"])
                + NEAR_THE_ROOM_TOKENS <= limit):
            return          # nothing goes for it
        needed = self._rendering.length(system + messages[newest:] + [message], False)
        if needed > limit:
            raise no_room_once_dropped(needed, limit, INJECT_A_QUESTION_FIRST)
