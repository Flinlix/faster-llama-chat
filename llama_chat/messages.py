"""Message bookkeeping: the single source of truth for the KV cache layout.

Invariants, asserted after every structural change:
    * ``messages[0]`` is the system prompt and is never evicted.
    * ``messages[i].pos_end == messages[i + 1].pos_start``.
    * ``messages[-1].pos_end == total``, the next decode position.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message occupying a slice of the KV cache.

    Attributes:
        role: ``"system"``, ``"user"`` or ``"assistant"``.
        text: What a reader saw: the message text without the template tags. An
            assistant turn that called a tool leaves out the call and the result
            it was handed as well, both of which are in ``token_ids``.
        token_ids: The token ids actually decoded into the cache, including the
            template tags. ``len(token_ids)`` equals the slice width.
        pos_start: First cache position owned by this message (inclusive).
        pos_end: One past the last cache position (exclusive).
        declares: The tools this message tells the model about, if any. Held
            here so that evicting the message takes the declaration with it.
        holds_call: Whether this assistant turn called a tool, as recorded
            when the call was generated or handed over in a history.
        steps: For a turn that called tools, the turn in order: its text, each
            call and each result, as a history hands one over.
    """

    role: str
    text: str
    token_ids: list[int]
    pos_start: int = 0
    pos_end: int = 0
    declares: tuple[str, ...] = ()
    holds_call: bool = False
    steps: list = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class Eviction:
    """One physical cache edit: remove a token range, then close the gap.

    Remove ``[remove_start, remove_end)``, then shift every surviving token in
    ``[remove_end, old_total)`` down by ``shift_delta``.

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
    """Ordered messages whose positions are derived, never stored independently.

    Any structural change recomputes ``pos_start``/``pos_end`` for every
    message, so the table cannot drift from the cache it mirrors.
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []
        # Reentrant: eviction's ``fits()`` callback re-enters through ``total``.
        self._lock = threading.RLock()

    # ----- introspection -------------------------------------------------
    @property
    def messages(self) -> list[Message]:
        with self._lock:
            return list(self._messages)

    @property
    def total(self) -> int:
        """Total tokens in the cache == next decode position."""
        with self._lock:
            return self._messages[-1].pos_end if self._messages else 0

    def snapshot_rows(self) -> list[dict]:
        """Return the layout as plain dict rows, read atomically.

        Returns:
            One ``{"role", "text", "n_tokens", "pos_start", "pos_end"}`` dict
            per message, mutually consistent even while another thread is
            restructuring the table. ``text`` is what was submitted, which a
            truncated message keeps beside the tokens that survived.
        """
        with self._lock:
            return [
                {
                    "role": m.role,
                    "text": m.text,
                    "n_tokens": m.n_tokens,
                    "pos_start": m.pos_start,
                    "pos_end": m.pos_end,
                }
                for m in self._messages
            ]

    @property
    def has_system(self) -> bool:
        with self._lock:
            return bool(self._messages) and self._messages[0].role == "system"

    @property
    def n_evictable(self) -> int:
        """Number of messages that may be evicted (everything but the system prompt)."""
        with self._lock:
            return max(0, len(self._messages) - (1 if self.has_system else 0))

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    # ----- mutation ------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._messages.clear()

    def append(self, message: Message) -> Message:
        """Append a message at the end of the cache, assigning its positions."""
        with self._lock:
            start = self.total
            message.pos_start = start
            message.pos_end = start + message.n_tokens
            self._messages.append(message)
            self._assert_invariants()
            return message

    def extend_last(self, token_ids: list[int], text: str = "") -> None:
        """Widen the newest message by tokens decoded after it was appended.

        Raises:
            IndexError: If there is no message to widen.
        """
        with self._lock:
            msg = self._messages[-1]
            msg.token_ids.extend(token_ids)
            msg.text += text
            msg.pos_end = msg.pos_start + msg.n_tokens
            self._assert_invariants()

    def replace_last(self, token_ids: list[int], text: str) -> None:
        """Give the newest message other tokens and text, in place.

        Raises:
            IndexError: If there is no message to replace.
        """
        with self._lock:
            msg = self._messages[-1]
            msg.token_ids = list(token_ids)
            msg.text = text
            msg.pos_end = msg.pos_start + msg.n_tokens
            self._assert_invariants()

    def replace_at(self, index: int, token_ids: list[int],
                   declares: tuple[str, ...]) -> None:
        """Give the message at ``index`` other tokens and declarations; the later ones move.

        Raises:
            IndexError: If there is no message at ``index``.
        """
        with self._lock:
            msg = self._messages[index]
            msg.token_ids = list(token_ids)
            msg.declares = declares
            self._renumber()
            self._assert_invariants()

    def remove_last(self) -> None:
        """Drop the newest message.

        Raises:
            IndexError: If there is no message to drop.
        """
        with self._lock:
            self._messages.pop()
            self._assert_invariants()

    def evict_oldest_until(self, fits) -> tuple[Eviction | None, int]:
        """Drop oldest non-system messages until ``fits()``, then up to the next user message.

        The survivors open on a user turn, as every template writes a
        conversation, or none remain. The victims are the contiguous block after
        the system prompt, so a whole batch collapses into one removed range and
        one survivor shift.

        Returns:
            The combined :class:`Eviction` (``None`` if nothing was dropped) and
            the number of messages dropped.
        """
        with self._lock:
            if fits() or self.n_evictable == 0:
                return None, 0
            idx = 1 if self.has_system else 0
            old_total = self.total
            remove_start = self._messages[idx].pos_start
            count = 0
            while self.n_evictable > 0 and (
                    not fits() or (count and self._messages[idx].role != "user")):
                del self._messages[idx]
                self._renumber()  # keeps total and fits() correct each iteration
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


def n_newest_that_fit(
    n_tokens_per_message: list[int], system_tokens: int, budget: int,
    max_messages: int | None = None,
) -> int:
    """Select how many of the most recent messages fit alongside the system prompt.

    Walks the history newest-first, accumulating token counts until the next
    (older) message would cross ``budget`` or ``max_messages`` are already held.
    The older excess is dropped. The two limits are independent, so whichever is
    reached first stops the walk.

    Args:
        n_tokens_per_message: Token counts of history messages, oldest first.
        system_tokens: Tokens consumed by the system prompt (always kept).
        budget: Maximum total tokens allowed (e.g. ``evict_above_tokens``).
        max_messages: Maximum messages to keep. ``None`` leaves ``budget`` as
            the only limit.

    Returns:
        The number of *trailing* (most recent) messages that fit. Messages
        ``n_tokens_per_message[len - k:]`` are kept; the rest are ignored.
    """
    used = system_tokens
    kept = 0
    for n in reversed(n_tokens_per_message):
        if used + n > budget:
            break
        if max_messages is not None and kept >= max_messages:
            break
        used += n
        kept += 1
    return kept
