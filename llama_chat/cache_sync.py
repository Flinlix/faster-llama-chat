"""A cache brought to what its message table holds, in the way that cache allows."""

from __future__ import annotations

from collections.abc import Callable

from .cache_context import CacheContext
from .messages import MessageTable


class CacheSync:
    """Evicts from, cuts back and rebuilds a cache so it holds what the table holds.

    A cache that shifts takes an eviction in place. One that cannot shift is
    cut at its end where it takes that; otherwise it goes back to a state it
    rested in, where one is saved, and is decoded again from the table where
    none is. See the README, "How eviction works".
    """

    def __init__(self, context: CacheContext, table: MessageTable) -> None:
        self._context = context
        self._table = table
        # (tokens held, state): at most the two newest a cache rested in.
        self._rest_states: list[tuple[int, object]] = []

    def evict(self, fits: Callable[[], bool]) -> int:
        """Evict the oldest messages until ``fits()``; returns how many went."""
        eviction, evicted = self._table.evict_oldest_until(fits)
        if eviction is None:
            return evicted
        if self._context.can_shift:
            self._context.apply_eviction(eviction)  # one removal and one shift
        else:
            self.rebuild()
        return evicted

    def cut_to(self, pos: int) -> int | None:
        """Make the cache hold the table's first ``pos`` tokens and nothing after.

        Returns:
            How many tokens were decoded to get there, or ``None`` where the
            whole table was decoded again.
        """
        if self._context.can_shift or self._context.can_cut_tail:
            self._context.remove_from(pos)
            return 0
        decoded = self._restore_to(pos)
        if decoded is None:
            self.rebuild()
        return decoded

    def rebuild(self) -> None:
        """Decode the whole table again into an emptied cache."""
        self._context.reset()
        self._context.prefill(self._table_tokens(), start_pos=0, want_logits=False)
        self._rest_states = []

    def rest(self) -> None:
        """Save the state the cache rests in now, where it needs one and can save it."""
        if self._context.can_shift or self._context.can_cut_tail:
            return                      # a cut needs no saved state
        try:
            state = self._context.save_state()
        except RuntimeError:
            state = None                # a rebuild stands in where none is saved
        if state is None:
            self._rest_states = []
            return
        total = self._table.total
        self._rest_states = [s for s in self._rest_states if s[0] < total][-1:] + [(total, state)]

    def forget(self) -> None:
        """Drop every saved state, as a reset of the conversation does."""
        self._rest_states = []

    @property
    def rested_at(self) -> int | None:
        """How many tokens the newest saved state holds, or ``None`` without one."""
        return self._rest_states[-1][0] if self._rest_states else None

    def _restore_to(self, pos: int) -> int | None:
        """From the newest state saved at or before ``pos``, decoding the table's tokens after it."""
        usable = [s for s in self._rest_states if s[0] <= pos]
        self._rest_states = usable
        if not usable:
            return None
        rested, state = usable[-1]
        try:
            self._context.restore_state(state, rested)
        except RuntimeError:
            self._rest_states = []
            return None
        between = self._table_tokens()[rested:pos]
        if between:
            self._context.prefill(between, start_pos=rested, want_logits=False)
        return len(between)

    def _table_tokens(self) -> list[int]:
        return [token for message in self._table.messages for token in message.token_ids]
