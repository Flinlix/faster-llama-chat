"""Message bookkeeping — the single source of truth for the KV cache.

This module is pure Python with no llama.cpp dependency. The cache for
sequence 0 holds a contiguous run of token positions ``[0, total)``;
each message owns a slice and the table below mirrors that layout exactly.

Invariants (asserted after every structural change):
    * ``messages[0]`` is the system prompt and is never evicted.
    * ``messages[i].pos_end == messages[i + 1].pos_start`` (no gaps/overlaps;
      ``pos_end`` is an exclusive upper bound, so there is no overlap).
    * ``messages[-1].pos_end == total`` (the next decode position).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message occupying a slice of the KV cache.

    Attributes:
        role: ``"system"``, ``"user"`` or ``"assistant"``.
        text: The message text (without template tags).
        token_ids: The token ids actually decoded into the cache, including the
            template tags. ``len(token_ids)`` equals the slice width.
        pos_start: First cache position owned by this message (inclusive).
        pos_end: One past the last cache position (exclusive).
    """

    role: str
    text: str
    token_ids: list[int]
    pos_start: int = 0
    pos_end: int = 0

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class Eviction:
    """Describes a physical cache edit so the backend can apply it.

    The backend should remove ``[remove_start, remove_end)`` then shift every
    surviving token in ``[remove_end, old_total)`` down by ``-shift_delta`` to
    close the gap.

    Attributes:
        remove_start: Inclusive start of the removed token range.
        remove_end: Exclusive end of the removed token range.
        old_total: Total tokens in the cache before removal; marks the right
            boundary of the shift range.
    """

    remove_start: int
    remove_end: int
    old_total: int

    @property
    def shift_delta(self) -> int:
        return self.remove_end - self.remove_start


class MessageTable:
    """Ordered list of messages with prefix-sum position bookkeeping.

    Positions are always *derived* from the ordered token counts, so the table
    cannot drift: any structural change recomputes ``pos_start``/``pos_end`` for
    every message.
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []

    # ----- introspection -------------------------------------------------
    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def total(self) -> int:
        """Total tokens in the cache == next decode position."""
        return self._messages[-1].pos_end if self._messages else 0

    @property
    def has_system(self) -> bool:
        return bool(self._messages) and self._messages[0].role == "system"

    @property
    def n_evictable(self) -> int:
        """Number of messages that may be evicted (everything but the system prompt)."""
        return max(0, len(self._messages) - (1 if self.has_system else 0))

    def __len__(self) -> int:
        return len(self._messages)

    # ----- mutation ------------------------------------------------------
    def reset(self) -> None:
        self._messages.clear()

    def append(self, message: Message) -> Message:
        """Append a message at the end of the cache, assigning its positions."""
        start = self.total
        message.pos_start = start
        message.pos_end = start + message.n_tokens
        self._messages.append(message)
        self._assert_invariants()
        return message

    def evict_oldest(self) -> Eviction:
        """Drop the oldest non-system message and renumber survivors.

        Returns:
            An :class:`Eviction` describing the physical edit to apply to the
            cache (``seq_rm`` then ``seq_add`` shift).

        Raises:
            IndexError: If there is no evictable message.
        """
        idx = 1 if self.has_system else 0
        if idx >= len(self._messages):
            raise IndexError("no evictable message")

        old_total = self.total
        victim = self._messages[idx]
        eviction = Eviction(
            remove_start=victim.pos_start,
            remove_end=victim.pos_end,
            old_total=old_total,
        )
        del self._messages[idx]
        self._renumber()
        self._assert_invariants()
        return eviction

    def evict_oldest_until(self, fits) -> tuple[Eviction | None, int]:
        """Drop oldest non-system messages until ``fits()`` (or none remain).

        The victims are always the contiguous block after the system prompt, so
        the whole batch collapses to one removed range plus one survivor shift.

        Returns:
            The combined :class:`Eviction` (``None`` if nothing was dropped) and
            the number of messages dropped.
        """
        if fits() or self.n_evictable == 0:
            return None, 0
        idx = 1 if self.has_system else 0
        old_total = self.total
        remove_start = self._messages[idx].pos_start
        count = 0
        while not fits() and self.n_evictable > 0:
            del self._messages[idx]
            self._renumber()  # cheap bookkeeping; keeps total/fits() correct each iter
            count += 1
        self._assert_invariants()
        eviction = Eviction(
            remove_start=remove_start,
            remove_end=remove_start + (old_total - self.total),
            old_total=old_total,
        )
        return eviction, count

    # ----- internals -----------------------------------------------------
    def _renumber(self) -> None:
        cursor = 0
        for msg in self._messages:
            msg.pos_start = cursor
            cursor += msg.n_tokens
            msg.pos_end = cursor

    def _assert_invariants(self) -> None:
        cursor = 0
        for i, msg in enumerate(self._messages):
            assert msg.pos_start == cursor, (
                f"message {i} starts at {msg.pos_start}, expected {cursor}"
            )
            assert msg.pos_end == msg.pos_start + msg.n_tokens, (
                f"message {i} width mismatch"
            )
            cursor = msg.pos_end
        if self._messages:
            assert self._messages[-1].pos_end == self.total


def fit_newest_first(
    n_tokens_per_message: list[int], system_tokens: int, budget: int
) -> int:
    """Select how many of the most recent messages fit alongside the system prompt.

    Walks the history newest-first, accumulating token counts until adding the
    next (older) message would exceed ``budget``. The older excess is dropped.

    Args:
        n_tokens_per_message: Token counts of history messages, oldest first.
        system_tokens: Tokens consumed by the system prompt (always kept).
        budget: Maximum total tokens allowed (e.g. ``threshold_tokens``).

    Returns:
        The number of *trailing* (most recent) messages that fit. Messages
        ``n_tokens_per_message[len - k:]`` are kept; the rest are ignored.
    """
    used = system_tokens
    kept = 0
    for n in reversed(n_tokens_per_message):
        if used + n > budget:
            break
        used += n
        kept += 1
    return kept
