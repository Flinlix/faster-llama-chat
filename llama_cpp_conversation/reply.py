"""What one request produces: its stream of deltas and the summary it ends with."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, get_args

from .errors import LIBRARY_ERRORS, OutOfOrderError, internal_error
from .tools import ToolCall


StopReason = Literal["end_of_turn", "stop_string", "token_limit", "interrupted", "tool_call",
                     "unreadable_tool_call"]
"""Why a generation ended, the same list wherever a reply says it."""
STOP_REASONS: tuple[str, ...] = get_args(StopReason)
"""Every :data:`StopReason`."""

ToolCallPolicy = Literal["withhold", "refuse", "yield"]
"""What becomes of a call the model makes, where a request names it rather than passing a registry."""
TOOL_CALL_POLICIES: tuple[str, ...] = get_args(ToolCallPolicy)
"""Every :data:`ToolCallPolicy`."""


@dataclass
class Reply:
    """What one request produced.

    One generation, or, where the request took several, all of them: the counts
    are then sums, and ``n_prompt_tokens`` and ``stop_reason`` are the last
    one's.

    Attributes:
        text: The assistant's reply, trimmed at any stop string, its thought
            left out.
        n_prefilled_tokens: Tokens decoded for the request: its question and
            what was decoded ahead for it after the reply before, the
            opening of each resumption, and a reply decoded again as a past
            reply. What is decoded after this reply, its closing and the
            start of the next question, counts for the next request.
        n_prompt_tokens: Tokens the cache held when the last generation drew
            its first token: everything it was generated from, its generation
            prompt included.
        n_generated_tokens: Tokens sampled.
        n_evicted_messages: History messages evicted for the request, each
            call's result counting as one, to bring the cache back inside both
            eviction limits; those evicted after the reply before, to make
            room for this request, included.
        stop_reason: One of :data:`~llama_cpp_conversation.reply.STOP_REASONS`: the reply's turn ended,
            a stop string was met, the reply cap or the context ran out, ``cancel`` was
            set, a call was made, or a call was made that could not be read
            and whose failure could not be written for the model to read.
        tool_calls: The calls the model made, in the order it made them.
            Reported whatever became of them, so a call that was refused rather
            than run still says what the model asked for, and one that could
            not be read or was cut short is reported with that
            ``ToolCall.status``.
        tool_limit_reached: Whether ``max_calls`` ran out, so the final
            generation ran without the token that opens a call. Always
            ``False`` where no call was asked for.
        tool_results_cut: The calls whose result was too long for the room
            left even after the oldest exchanges went, and was cut, a note at
            its end telling the model how much was kept; those answered
            before a generation of this request, oldest first.
        n_reasoning_tokens: Of the tokens sampled, about how many were thought:
            the streamed pieces that held any.
        thought: What the model thought before its answer, its tags and the
            whitespace around it left out; empty where it thought nothing.
            Where the request took several generations, each one's thought in
            order, a blank line between two.
        evicted_for: Why history went, as :attr:`Report.evicted_for` says.
        taken_back: Whether the request was taken back, its question with
            it, because the template writes neither its reply nor an empty
            one; the conversation is then as before the request.
        kept_text: The answer as the conversation keeps it, where that
            differs from ``text``: a template that writes only part of an
            answer holding its thought tags keeps that part; ``None`` where
            it keeps ``text``.
    """

    text: str
    n_prefilled_tokens: int
    n_prompt_tokens: int
    n_generated_tokens: int
    n_evicted_messages: int
    stop_reason: StopReason
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_limit_reached: bool = False
    tool_results_cut: tuple[ToolCall, ...] = ()
    n_reasoning_tokens: int = 0
    thought: str = ""
    evicted_for: EvictionCause | None = None
    taken_back: bool = False
    kept_text: str | None = None


EvictionCause = Literal["room", "exchange_limit", "half_context"]
"""Why the oldest history went, because the conversation no longer fit, the exchange limit was reached, or begin kept no more than half the context."""
EVICTION_CAUSES: tuple[str, ...] = get_args(EvictionCause)
"""Every :data:`EvictionCause`."""


@dataclass(frozen=True)
class Report:
    """What an action changed beyond what was asked for, each such change also logged once.

    Attributes:
        n_evicted_messages: History messages that went, each call's result
            counting as one; for the first action after a reply, those that
            went after that reply too.
        evicted_for: One of :data:`~llama_cpp_conversation.reply.EVICTION_CAUSES`, why they went; ``None``
            where none did.
        n_kept_messages: Of a history given to ``begin``, how many messages
            it kept; ``None`` for any other action.
        n_left_out_calls: Calls of the history ``begin`` was given that could
            not be read, the model's or the caller's, which it left out with
            their results, no generated turn holding them.
        n_turns_restored: Turns ``begin`` restored as generated, from the
            generated turns it was given.
        generated_turns_ignored: Whether generated turns ``begin`` was given
            went unused: another model's, or ones changed since
            :meth:`~llama_cpp_conversation.ChatWrapper.conversation` gave them
            out, whose turns are written as the template writes them.
        tool_results_cut: The calls whose result was cut to fit, with a note
            telling the model how much was kept.
        turn_closed: Whether the action closed the open turn: a result that
            did not fit even cut to nothing, or a question that closed it, the
            calls it still owed told they were not run.
        question_joined: Whether a question was joined to one no reply
            followed, the template writing no two in a row.
        taken_back: Whether a reply was taken back with its request, as one
            a result found no room for, with nothing of it the template
            writes, is.
    """

    n_evicted_messages: int = 0
    evicted_for: EvictionCause | None = None
    n_kept_messages: int | None = None
    n_left_out_calls: int = 0
    n_turns_restored: int = 0
    generated_turns_ignored: bool = False
    tool_results_cut: tuple[ToolCall, ...] = ()
    turn_closed: bool = False
    question_joined: bool = False
    taken_back: bool = False


_Delta = TypeVar("_Delta")


class ReplyStream(Generic[_Delta]):
    """One request's deltas, and the :class:`Reply` they add up to.

    Iterate it for what the model produces, then read :attr:`reply` for the
    summary. Closing it early records what was generated, and :attr:`reply`
    then holds the summary of the generation it was closed in, what that
    generation evicted and cut included; a call already handed over stays
    owed, its turn open for the result. Closing is what an abandoned stream
    needs, so it is also a context manager. Like the wrapper's actions, it
    raises only the library's own errors. One thread steps it at a time; a
    stream another thread steps waits for that step.

    Attributes:
        reply: The summary, ``None`` until the stream has ended or was
            closed in a generation.
    """

    def __init__(self, deltas: Generator[_Delta, None, Reply],
                 on_close: Callable[[], Reply | None] | None = None,
                 runs_a_step: Callable[[], bool] = lambda: False) -> None:
        self._deltas = deltas
        self._on_close = on_close
        # Whether the calling thread runs for a step of this stream, as a tool the reply runs does.
        self._runs_a_step = runs_a_step
        self.reply: Reply | None = None
        self._stepping = threading.Lock()
        self._stepper: threading.Thread | None = None
        # When the reader last took a step; None before the first and while one runs.
        self._read_at: float | None = None

    def __iter__(self) -> ReplyStream[_Delta]:
        return self

    def __next__(self) -> _Delta:
        # A second pass over a finished stream yields nothing and leaves the
        # summary standing, rather than replacing it with the None a spent
        # generator returns.
        if self.reply is not None:
            raise StopIteration
        if self._stepper is threading.current_thread() or self._runs_a_step():
            raise OutOfOrderError(None, "a tool the reply runs read that reply's own stream, "
                                        "which would wait for the tool; read nothing of it "
                                        "from a tool")
        with self._stepping:
            self._read_at, self._stepper = None, threading.current_thread()
            try:
                return next(self._deltas)
            except StopIteration as ended:
                if self.reply is None:
                    self.reply = ended.value
                raise
            except LIBRARY_ERRORS:
                raise
            except Exception as error:
                raise internal_error("ReplyStream.__next__", error) from error
            finally:
                self._read_at, self._stepper = time.monotonic(), None

    def close(self) -> None:
        """Stop reading, recording what was generated. Idempotent, as a generator's is.

        Raises:
            OutOfOrderError: Where a step of the stream runs meanwhile, on
                another thread or in a tool the step runs; stop it with the
                reply's ``cancel`` instead.
        """
        if not self._stepping.acquire(blocking=False):
            raise OutOfOrderError(None, "the reply is being read meanwhile, so it cannot be "
                                        "closed now; set the cancel it was given, or close it "
                                        "where it is read")
        try:
            self._close()
        finally:
            self._stepping.release()

    def _unread_s(self) -> float:
        """How long the stream has waited for its reader's next step; ``0`` before the first and while one runs."""
        read_at = self._read_at
        return 0.0 if read_at is None else time.monotonic() - read_at

    def _end_if_unread(self, unread_s: float) -> bool:
        """Close the stream, as :meth:`close` does, where its reader has taken no step for ``unread_s`` seconds; whether it did.

        Its reader then finds it ended, :attr:`reply` as closing it leaves
        it.
        """
        if not self._stepping.acquire(blocking=False):
            return False                # a step runs: it is being read
        try:
            if self._read_at is None or self._unread_s() < unread_s:
                return False
            self._close()
            return True
        finally:
            self._stepping.release()

    def _close(self) -> None:
        try:
            self._deltas.close()
            if self.reply is None and self._on_close is not None:
                self.reply = self._on_close()
        except LIBRARY_ERRORS:
            raise
        except Exception as error:
            raise internal_error("ReplyStream.close", error) from error

    def finish(self) -> Reply | None:
        """Read the stream to its end, and return its :class:`Reply`; ``None`` where it was closed before it started."""
        for _ in self:
            pass
        return self.reply

    def __enter__(self) -> ReplyStream[_Delta]:
        return self

    def __exit__(self, *_) -> None:
        self.close()
