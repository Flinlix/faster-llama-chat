"""A cache kept holding a list of tokens, in the way that cache allows; the one place that decodes into it."""

from __future__ import annotations

from .cache_context import CacheContext, Eviction, GenerationAccumulator
from .template import common_prefix_length


class CacheSync:
    """The tokens a cache should hold, and that cache brought to hold them.

    Every change to :attr:`tokens` goes through here. A cache that shifts or
    cuts its end takes each change at once. One that cannot is brought back to
    :attr:`tokens` before the next generation or rest, once however many
    changes came first: from the newest saved state that still holds a
    beginning of them, else by decoding them again whole. Besides the states it
    rested in, such a cache keeps one where :meth:`pin` says, for as long as the
    tokens before it stay. A decode or an edit that fails leaves the cache
    unknown, and it is decoded again whole the same way. See design_history.md,
    "How eviction works".
    """

    def __init__(self, context: CacheContext) -> None:
        self._context = context
        self._tokens: list[int] = []
        self._valid = 0                 # how many of the tokens the cache holds, from position 0
        self._end: int | None = 0       # how many it holds in all; None where unknown
        self._logits = False            # whether the newest decode left logits for its last token
        self.decoded = 0                # every token decoded so far, for what a request cost
        # (tokens held, state): at most the two newest a cache rested in.
        self._rest_states: list[tuple[int, object]] = []
        self._pin: int | None = None    # where a state is kept, besides those
        self._pinned: tuple[int, object] | None = None

    @property
    def tokens(self) -> list[int]:
        """What the cache holds, or is brought to hold before it is next used; not to be changed by the caller."""
        return self._tokens

    @property
    def is_known(self) -> bool:
        """Whether what the cache holds is known: no decode or edit failed since it was last brought up."""
        return self._end is not None

    @property
    def is_current(self) -> bool:
        """Whether the cache holds exactly :attr:`tokens` now."""
        return self._end == self._valid == len(self._tokens)

    def reset(self) -> None:
        """Empty the cache and forget every saved state, as a new conversation does."""
        self._context.reset()
        self._tokens, self._valid, self._end = [], 0, 0
        self._rest_states, self._logits = [], False
        self._pin, self._pinned = None, None

    def pin(self, pos: int | None) -> None:
        """Keep a state saved where the first ``pos`` tokens end, for as long as they stay, where the cache needs saved states; ``None`` keeps none."""
        if pos != self._pin:
            self._pin, self._pinned = pos, None

    def decode(self, tokens: list[int], want_logits: bool = False) -> None:
        """Append ``tokens``, decoded at once where the cache holds everything before them."""
        if not tokens:
            return
        if not self.is_current:
            self._tokens += tokens      # decoded with the rest when the cache is brought up
            return
        self._decode_at_end(tokens, want_logits)

    def generate(self, max_generated_tokens: int, out: GenerationAccumulator, **options):
        """Stream a generation from the cache's end; what it decodes is appended to :attr:`tokens`."""
        self._bring_up(want_logits=True)
        if not self._logits:
            self._decode_again_with_logits()
        start = self._end
        pieces = self._context.generate(start, max_generated_tokens, out=out, **options)
        try:
            yield from pieces
        except GeneratorExit:
            raise                       # closed early: what was decoded is recorded below
        except BaseException:
            self._end = None
            raise
        finally:
            pieces.close()
            if self._end is not None:
                self._tokens += out.token_ids
                self._end = self._valid = len(self._tokens)
                self._end += out.cached_beyond
                self._logits = bool(out.token_ids)
                if out.cached_beyond:
                    self.cut_to(len(self._tokens))
            else:
                self._tokens += out.token_ids

    def remove(self, start: int, end: int) -> None:
        """Take the tokens from ``start`` to ``end`` out, those after them moving down."""
        if start >= end:
            return
        eviction = Eviction(start, end, len(self._tokens))
        del self._tokens[start:end]
        self._forget_states_after(start)
        self._logits = False
        if self._end == self._valid == eviction.old_total and self._context.can_shift:
            try:
                self._context.apply_eviction(eviction)  # one removal and one shift
            except BaseException:
                self._end = None
                raise
            self._end = self._valid = len(self._tokens)
            return
        self._valid = min(self._valid, start)

    def cut_to(self, pos: int) -> None:
        """Keep the first ``pos`` tokens and nothing after."""
        if pos >= len(self._tokens) and self._end == len(self._tokens):
            return
        del self._tokens[pos:]
        self._forget_states_after(pos)
        self._logits = False
        self._valid = min(self._valid, pos)
        if self._end is not None and self._end > pos and self._can_cut:
            try:
                self._context.remove_from(pos)
            except BaseException:
                self._end = None
                raise
            self._end = pos

    def replace(self, tokens: list[int]) -> None:
        """Hold ``tokens`` instead, the cache brought to them before it is next used."""
        if list(tokens) == self._tokens:
            return                      # nothing changed, the logits of the last decode included
        same = common_prefix_length(self._tokens, tokens)
        self._tokens = list(tokens)
        self._valid = min(self._valid, same)
        self._forget_states_after(same)
        self._logits = False

    def repair(self) -> bool:
        """Bring the cache to its tokens where a change was left for later or failed; whether it was needed."""
        if self.is_current:
            return False
        self._bring_up(want_logits=True)
        return True

    def rest(self) -> None:
        """Bring the cache up, then save the state it rests in, where it needs one and can save it."""
        self._bring_up(want_logits=False)
        if self._can_cut:
            return                      # a cut needs no saved state
        try:
            state = self._context.save_state()
        except RuntimeError:
            state = None                # decoding again stands in where none is saved
        if state is None:
            self._rest_states = []
            return
        self._rest_states = ([s for s in self._rest_states if s[0] < self._end][-1:]
                             + [(self._end, state)])

    @property
    def _can_cut(self) -> bool:
        return self._context.can_shift or self._context.can_cut_tail

    def _bring_up(self, want_logits: bool) -> None:
        """Make the cache hold :attr:`tokens`, decoding what it lacks."""
        if self.is_current:
            return
        if self._end is None:
            self._reset_to_nothing()
        elif self._end > self._valid:
            if self._can_cut:
                try:
                    self._context.remove_from(self._valid)
                except BaseException:
                    self._end = None
                    raise
                self._end = self._valid
            elif not self._restore_to(self._valid):
                self._reset_to_nothing()
        self._decode_at_end(self._tokens[self._valid:], want_logits, appended=True)

    def _decode_at_end(self, tokens: list[int], want_logits: bool, appended: bool = False) -> None:
        """Decode ``tokens`` at the cache's end; ``appended`` where they are in :attr:`tokens` already."""
        if not tokens:
            return
        if self._needs_pinned_state() and self._end < self._pin < self._end + len(tokens):
            # Decoded in two, so that the state where the pin is can be saved; all of
            # them held first, so that a failure in either leaves none out.
            if not appended:
                self._tokens += tokens
            split = self._pin - self._end
            self._decode_at_end(tokens[:split], False, appended=True)
            self._decode_at_end(tokens[split:], want_logits, appended=True)
            return
        try:
            self._context.prefill(tokens, start_pos=self._end, want_logits=want_logits)
        except BaseException:
            self._end = None
            if not appended:
                self._tokens += tokens
            raise
        if not appended:
            self._tokens += tokens
        self._end += len(tokens)
        self._valid = self._end
        self._logits = want_logits
        self.decoded += len(tokens)
        if self._needs_pinned_state() and self._end == self._pin:
            try:
                state = self._context.save_state()
            except RuntimeError:
                state = None
            if state is not None:
                self._pinned = (self._end, state)

    def _needs_pinned_state(self) -> bool:
        return self._pin is not None and self._pinned is None and not self._can_cut

    def _decode_again_with_logits(self) -> None:
        """Decode the last token again, so that the next token can be drawn after it."""
        if not self._tokens:
            raise ValueError("the cache holds nothing to generate after")
        last = self._tokens[-1]
        self.cut_to(len(self._tokens) - 1)
        self._bring_up(want_logits=False)
        self._decode_at_end([last], want_logits=True)

    def _reset_to_nothing(self) -> None:
        self._context.reset()
        self._valid, self._end, self._rest_states, self._pinned = 0, 0, [], None

    def _restore_to(self, pos: int) -> bool:
        """Go back to the newest state saved at or before ``pos``; whether one could be."""
        self._forget_states_after(pos)
        saved = self._rest_states + ([self._pinned] if self._pinned else [])
        if not saved:
            return False
        rested, state = max(saved, key=lambda s: s[0])
        try:
            self._context.restore_state(state, rested)
        except RuntimeError:
            self._rest_states, self._pinned = [], None
            return False
        self._valid = self._end = rested
        return True

    def _forget_states_after(self, pos: int) -> None:
        """Drop the saved states that hold more than the first ``pos`` tokens, which changed."""
        self._rest_states = [s for s in self._rest_states if s[0] <= pos]
        if self._pinned is not None and self._pinned[0] > pos:
            self._pinned = None
