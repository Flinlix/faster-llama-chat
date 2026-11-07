"""A cache kept holding what it should, in the way that cache allows; the one place that decodes into it."""

from __future__ import annotations

from collections.abc import Callable

from .cache_context import CacheContext, GenerationAccumulator
from .messages import Eviction


class CacheSync:
    """Decodes into, evicts from, cuts back and rebuilds a cache so it holds what ``held`` says.

    Every decode goes through here and appends at the cache's end, which is
    tracked here. A cache that shifts takes an eviction in place. One that
    cannot shift is cut at its end where it takes that; otherwise it goes back
    to a state it rested in, where one is saved, and is decoded again from
    ``held`` where none is. A decode that fails part-way leaves the cache
    unknown, and the next :meth:`repair` decodes it again whole. See
    design_history.md, "How eviction works".

    Args:
        context: The cache.
        held: The tokens the cache should hold, from position 0, read at each
            rebuild, restore and rest.
    """

    def __init__(self, context: CacheContext, held: Callable[[], list[int]]) -> None:
        self._context = context
        self._held = held
        self._end = 0
        self._broken = False
        # (tokens held, state): at most the two newest a cache rested in.
        self._rest_states: list[tuple[int, object]] = []

    @property
    def end(self) -> int:
        """How many tokens the cache holds: where the next decode goes."""
        return self._end

    def reset(self) -> None:
        """Empty the cache and forget every saved state, as a new conversation does."""
        self._context.reset()
        self._end = 0
        self._broken = False
        self._rest_states = []

    def decode(self, tokens: list[int], want_logits: bool = False) -> None:
        """Decode ``tokens`` at the cache's end."""
        if not tokens:
            return
        try:
            self._context.prefill(tokens, start_pos=self._end, want_logits=want_logits)
        except BaseException:
            self._broken = True
            raise
        self._end += len(tokens)

    def generate(self, max_generated_tokens: int, out: GenerationAccumulator, **options):
        """Stream a generation from the cache's end; the tokens it decodes count to the end."""
        pieces = self._context.generate(self._end, max_generated_tokens, out=out, **options)
        try:
            yield from pieces
        except GeneratorExit:
            raise                       # closed early: what was decoded is recorded below
        except BaseException:
            self._broken = True
            raise
        finally:
            pieces.close()
            self._end += len(out.token_ids) + out.cached_beyond

    def remove(self, eviction: Eviction | None) -> None:
        """Take out what ``eviction`` names, which ``held`` no longer holds."""
        if eviction is None:
            return
        if not self._context.can_shift:
            self.rebuild()
            return
        try:
            self._context.apply_eviction(eviction)    # one removal and one shift
        except BaseException:
            self._broken = True
            raise
        self._end -= eviction.shift_delta

    def cut_to(self, pos: int) -> int | None:
        """Make the cache hold the first ``pos`` tokens of ``held`` and nothing after.

        Returns:
            How many tokens were decoded to get there, or ``None`` where all of
            ``held`` was decoded again.
        """
        if self._context.can_shift or self._context.can_cut_tail:
            self._context.remove_from(pos)
            self._end = pos
            return 0
        decoded = self._restore_to(pos)
        if decoded is None:
            self.rebuild()
        return decoded

    def rebuild(self) -> None:
        """Decode all of ``held`` again into an emptied cache."""
        self.reset()
        self.decode(self._held())

    def repair(self) -> bool:
        """Decode the cache again whole where a decode failed part-way; whether it did."""
        if not self._broken:
            return False
        self.rebuild()
        return True

    def rest(self) -> None:
        """Save the state the cache rests in now, where it needs one and can save it."""
        assert self._end == len(self._held()), "the cache no longer holds what it should"
        if self._context.can_shift or self._context.can_cut_tail:
            return                      # a cut needs no saved state
        try:
            state = self._context.save_state()
        except RuntimeError:
            state = None                # a rebuild stands in where none is saved
        if state is None:
            self._rest_states = []
            return
        self._rest_states = ([s for s in self._rest_states if s[0] < self._end][-1:]
                             + [(self._end, state)])

    def forget(self) -> None:
        """Drop every saved state."""
        self._rest_states = []

    def _restore_to(self, pos: int) -> int | None:
        """From the newest state saved at or before ``pos``, decoding the held tokens after it."""
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
        self._end = rested
        between = self._held()[rested:pos]
        self.decode(between)
        return len(between)
