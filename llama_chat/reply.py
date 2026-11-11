"""What one request produces: its stream of deltas, the summary it ends with, and the refusal for want of room."""

from __future__ import annotations

from collections.abc import Generator
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


class ContextOverflowError(ValueError):
    """What was asked for does not fit the context: a message, a history, a declaration, or room for the reply."""


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
            each resumption, and a reply decoded again as a past reply.
        n_prompt_tokens: Tokens the cache held when the last generation drew
            its first token: everything it was generated from, its generation
            prompt included.
        n_generated_tokens: Tokens sampled.
        n_evicted_messages: Messages evicted to bring the cache back inside both
            eviction limits.
        stop_reason: One of :data:`STOP_REASONS`: the reply's turn ended,
            a stop string was met, the token cap was reached, ``cancel`` was
            set, a call was made, or a call was made that could not be read.
        tool_calls: The calls the model made, in the order it made them.
            Reported whatever became of them, so a call that was refused rather
            than run still says what the model asked for, and one that could
            not be read or was cut short is reported with that
            :attr:`~llama_chat.tools.ToolCall.status`.
        tool_limit_reached: Whether ``max_calls`` ran out, so the final
            generation ran with a call it wrote left as text. Always ``False``
            where no call was asked for.
        n_reasoning_tokens: Of the tokens sampled, about how many were thought:
            the streamed pieces that held any.
    """

    text: str
    n_prefilled_tokens: int
    n_prompt_tokens: int
    n_generated_tokens: int
    n_evicted_messages: int
    stop_reason: StopReason
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_limit_reached: bool = False
    n_reasoning_tokens: int = 0


_Delta = TypeVar("_Delta")


class ReplyStream(Generic[_Delta]):
    """One request's deltas, and the :class:`Reply` they add up to.

    Iterate it for what the model produces, then read :attr:`reply` for the
    summary. Closing it early ends the turn cleanly and leaves :attr:`reply`
    unset, there being no summary of a reply nobody read to the end. Closing is
    what an abandoned stream needs, so it is also a context manager.

    Attributes:
        reply: The summary, ``None`` until the stream has ended of its own
            accord.
    """

    def __init__(self, deltas: Generator[_Delta, None, Reply]) -> None:
        self._deltas = deltas
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
        """Stop reading and end the turn. Idempotent, as a generator's is."""
        self._deltas.close()

    def finish(self) -> Reply:
        """Read the stream to its end, and return its :class:`Reply`."""
        for _ in self:
            pass
        return self.reply

    def __enter__(self) -> ReplyStream[_Delta]:
        return self

    def __exit__(self, *_) -> None:
        self.close()
