"""What one request produces: its stream of deltas, the summary it ends with, and the refusal for want of room."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from .tools import ToolCall


class ContextOverflowError(RuntimeError):
    """A request would leave fewer than ``min_reply_tokens`` free for the reply."""


@dataclass
class Reply:
    """What one request produced.

    One generation, or, where the request took several, all of them: the counts
    are then sums and ``stop_reason`` is the last one's.

    Attributes:
        text: The assistant's reply, trimmed at any stop string, its thought
            left out.
        n_prefilled_tokens: Tokens decoded for the request, the opening of
            each resumption, and a reply decoded again as a past reply.
        n_generated_tokens: Tokens sampled.
        n_evicted_messages: Messages evicted to bring the cache back inside both
            eviction limits.
        stop_reason: One of ``"end_of_turn"``, ``"stop_string"``,
            ``"token_limit"``, ``"interrupted"``, ``"tool_call"`` or
            ``"unreadable_tool_call"``.
        tool_calls: The calls the model made, in the order it made them.
            Reported whatever became of them, so a call that was refused rather
            than run still says what the model asked for.
        tool_limit_reached: Whether ``max_calls`` ran out, so the final
            generation ran with the call tokens withheld. Always ``False``
            where no call was asked for.
        n_reasoning_tokens: Of the tokens sampled, about how many were thought:
            the streamed pieces that held any.
    """

    text: str
    n_prefilled_tokens: int
    n_generated_tokens: int
    n_evicted_messages: int
    stop_reason: str
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
