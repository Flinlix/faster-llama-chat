"""What one request produces: its stream of deltas and the summary it ends with."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar, get_args

from .tools import ToolCall


# Why a generation ended, the same list wherever a reply says it.
StopReason = Literal["end_of_turn", "stop_string", "token_limit", "interrupted", "tool_call",
                     "unreadable_tool_call"]
STOP_REASONS: tuple[str, ...] = get_args(StopReason)

# What becomes of a call the model makes, where a request names it rather than passing a registry.
ToolCallPolicy = Literal["withhold", "refuse", "yield"]
TOOL_CALL_POLICIES: tuple[str, ...] = get_args(ToolCallPolicy)


@dataclass
class Reply:
    """What one request produced.

    One generation, or, where the request took several, all of them: the counts
    are then sums, and ``n_prompt_tokens`` and ``stop_reason`` are the last
    one's.

    Attributes:
        text: The assistant's reply, trimmed at any stop string, its thought
            left out.
        n_prefilled_tokens: Tokens decoded for the request, the opening of
            each resumption, a reply decoded again as a past reply, and what
            is decoded after the reply: its closing and the start of the next
            question.
        n_prompt_tokens: Tokens the cache held when the last generation drew
            its first token: everything it was generated from, its generation
            prompt included.
        n_generated_tokens: Tokens sampled.
        n_evicted_messages: History items evicted to bring the cache back
            inside both eviction limits: a message, or a whole tool turn,
            counts as one.
        stop_reason: One of :data:`STOP_REASONS`: the reply's turn ended,
            a stop string was met, the reply cap was reached, ``cancel`` was
            set, a call was made, or a call was made that could not be read
            and whose failure could not be written for the model to read.
        tool_calls: The calls the model made, in the order it made them.
            Reported whatever became of them, so a call that was refused rather
            than run still says what the model asked for, and one that could
            not be read or was cut short is reported with that
            :attr:`~llama_chat.tools.ToolCall.status`.
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
    tool_results_cut: list[ToolCall] = field(default_factory=list)
    n_reasoning_tokens: int = 0
    thought: str = ""
    evicted_for: str | None = None
    taken_back: bool = False
    kept_text: str | None = None


# Why the oldest history went: the conversation no longer fit, the exchange limit was reached,
# or begin kept no more than half the context.
EvictionCause = Literal["room", "exchange_limit", "half_context"]
EVICTION_CAUSES: tuple[str, ...] = get_args(EvictionCause)


@dataclass(frozen=True)
class Report:
    """What an action changed beyond what was asked for, each such change also logged once.

    Attributes:
        n_evicted_messages: History items that went, a tool turn counting as
            one.
        evicted_for: One of :data:`EVICTION_CAUSES`, why they went; ``None``
            where none did.
        n_kept_messages: Of a history given to ``begin``, how many items it
            kept; ``None`` for any other action.
        n_left_out_calls: Calls the model wrote unreadably that ``begin`` left
            out with their error result, no generated turn holding them.
        n_turns_restored: Turns ``begin`` restored as generated, from the
            generated turns it was given.
        generated_turns_ignored: Whether the generated turns ``begin`` was
            given were another model's, and so went unused.
        tool_results_cut: The calls whose result was cut to fit, with a note
            telling the model how much was kept.
        turn_closed: Whether the action closed the open turn: a result that
            did not fit even cut to nothing, or a question that closed it, the
            calls it still owed told they were not run.
        question_joined: Whether a question was joined to one no reply
            followed, the template writing no two in a row.
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


_Delta = TypeVar("_Delta")


class ReplyStream(Generic[_Delta]):
    """One request's deltas, and the :class:`Reply` they add up to.

    Iterate it for what the model produces, then read :attr:`reply` for the
    summary. Closing it early records what was generated, and :attr:`reply`
    then holds the summary of the generation it was closed in, what that
    generation evicted and cut included; a call already handed over stays
    owed, its turn open for the result. Closing is what an abandoned stream
    needs, so it is also a context manager.

    Attributes:
        reply: The summary, ``None`` until the stream has ended or was
            closed in a generation.
    """

    def __init__(self, deltas: Generator[_Delta, None, Reply],
                 on_close: Callable[[], Reply | None] | None = None) -> None:
        self._deltas = deltas
        self._on_close = on_close
        self.reply: Reply | None = None

    def __iter__(self) -> ReplyStream[_Delta]:
        return self

    def __next__(self) -> _Delta:
        # A second pass over a finished stream yields nothing and leaves the
        # summary standing, rather than replacing it with the None a spent
        # generator returns.
        if self.reply is not None:
            raise StopIteration
        try:
            return next(self._deltas)
        except StopIteration as ended:
            self.reply = ended.value
            raise

    def close(self) -> None:
        """Stop reading, recording what was generated. Idempotent, as a generator's is."""
        self._deltas.close()
        if self.reply is None and self._on_close is not None:
            self.reply = self._on_close()

    def finish(self) -> Reply:
        """Read the stream to its end, and return its :class:`Reply`."""
        for _ in self:
            pass
        return self.reply

    def __enter__(self) -> ReplyStream[_Delta]:
        return self

    def __exit__(self, *_) -> None:
        self.close()
