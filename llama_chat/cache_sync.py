"""A cache kept holding a list of tokens, in the way that cache allows; the one place that decodes into it."""

from __future__ import annotations

from collections.abc import Sized
from typing import Protocol

import contextlib
import logging

import threading

from .cache_context import CacheContext
from .errors import InternalError
from .generation import GenerationAccumulator
from .generation import generate
from .sequences import common_prefix_length
from .loop_variant import LoopVariant

# Maps a token id to one character, so token lists search as strings.
_TOKEN_OFFSET = 0x10000
# The fewest tokens a part of a long decode watched for a cancel holds; a larger micro-batch is
# a part, so that the parts keep the batch's speed.
_CANCEL_PART_TOKENS = 512
# How many tokens a cache must hold beyond what a begin keeps of it to be kept, and how many more
# of a begin's target a kept cache must hold than the cache does to be switched to.
_WORTH_KEEPING_TOKENS = 256
# How far the cache must go on as the target after a stretch the template wrote, for that
# stretch alone to be shifted out.
_LEFT_OUT_MATCH_TOKENS = 32
# An interrupt that stands set.
_ALWAYS_SET = threading.Event()
_ALWAYS_SET.set()


_log = logging.getLogger(__name__)

class Cancelled(Exception):
    """A decode that :attr:`CacheSync.cancel` stopped between its parts."""



class Interrupt(Protocol):
    """What tells a long decode to stop between its parts: set, it gives way."""

    def is_set(self) -> bool: ...

class CacheSync:
    """The tokens a cache should hold, and that cache brought to hold them.

    Every change to :attr:`tokens` goes through here. A cache that shifts or
    cuts its end takes each change at once; one that cannot is brought back to
    :attr:`tokens` before the next generation or rest. See design_history.md,
    "How eviction works".
    """

    def __init__(self, context: CacheContext, micro_batch_tokens: int | None = None,
                 kept_conversation_bytes: int = 0) -> None:
        self._context = context
        # Where the cache holds only what the template wrote before a reply, until it changes there.
        self._template_written: tuple[int, int] | None = None
        # The caches of conversations switched away from, where any may be kept.
        self._kept = _KeptCaches(kept_conversation_bytes) if kept_conversation_bytes else None
        # A long decode watched for a cancel runs in parts of a micro-batch, at least the floor.
        self._part_tokens = max(_CANCEL_PART_TOKENS, micro_batch_tokens or 0)
        self._tokens: list[int] = []
        self._valid = 0                 # how many of the tokens the cache holds, from position 0
        self._end: int | None = 0       # how many it holds in all; None where unknown
        self._logits = False            # whether the newest decode left logits for its last token
        self.decoded = 0                # every token decoded so far, for what a request cost
        self.version = 0                # changes whenever the tokens do
        # Watched between the parts of a long decode: ``cancel``, also before any decode, then
        # raises Cancelled; ``interrupt`` leaves the rest of the decode to the next bring-up.
        self.cancel: threading.Event | None = None
        self.interrupt: Interrupt | None = None
        # (tokens held, state): at most the two newest a cache rested in.
        self._rest_states: list[tuple[int, object]] = []
        # Where states are kept besides those, by what they are kept for, and the states kept,
        # by how many tokens they hold.
        self._pins: dict[str, frozenset[int]] = {}
        self._pinned: dict[int, object] = {}
        # The states rested in where the cache was cut for a rest the two newest times,
        # besides those: the next change may begin at either.
        self._at_cuts: list[tuple[int, object]] = []

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

    def pin(self, positions, kept_for: str) -> None:
        """Keep a state saved where the first ``pos`` tokens end, for each ``pos`` of ``positions``, for as long as they stay, where the cache needs saved states.

        Args:
            kept_for: What the states are kept for; those kept for something
                else are kept beside them, and these replace the ones it kept.
        """
        self._pins[kept_for] = frozenset(positions)
        kept = frozenset().union(*self._pins.values())
        self._pinned = {pos: state for pos, state in self._pinned.items() if pos in kept}
        if self.is_current:
            self._save_pinned_state()

    def pinned(self, kept_for: str) -> frozenset[int]:
        """Where states are kept for ``kept_for``."""
        return self._pins.get(kept_for, frozenset())

    def decode(self, tokens: list[int], want_logits: bool = False) -> None:
        """Append ``tokens``, decoded at once where the cache holds everything before them."""
        if not tokens:
            return
        self.version += 1
        if not self.is_current:
            self._tokens += tokens      # decoded with the rest when the cache is brought up
            return
        self._decode_at_end(tokens, want_logits)

    def generate(self, max_generated_tokens: int, out: GenerationAccumulator, **options):
        """Stream a generation from the cache's end; what it decodes is appended to :attr:`tokens`.

        A ``cancel`` among ``options`` is watched while what the generation
        follows is decoded too; set there, it ends the generation at once as
        ``"interrupted"``, with nothing generated.
        """
        self.cancel = options.get("cancel")
        try:
            self._bring_up(want_logits=True)
            if not self._logits:
                self._decode_again_with_logits()
        except Cancelled:
            out.stop_reason = "interrupted"
            return
        finally:
            self.cancel = None
        start = self._brought_up_end
        pieces = generate(self._context, start, max_generated_tokens, out=out, **options)
        try:
            yield from pieces
        except GeneratorExit:
            raise                       # closed early: what was decoded is recorded below
        except BaseException:
            self._end = None
            raise
        finally:
            pieces.close()
            if out.token_ids:
                self.version += 1
            if self._end is not None:
                self._tokens += out.token_ids
                self._end = self._valid = len(self._tokens)
                self._end += out.cached_beyond
                self._logits = bool(out.token_ids) and out.ends_with_logits
                if out.cached_beyond:
                    self.cut_to(len(self._tokens))
            else:
                self._tokens += out.token_ids

    def _switch_for(self, target: list[int], same: int) -> int:
        """Keep the cache where ``target`` leaves most of it, and hold a kept one instead where that shares more of ``target``; how much of ``target`` the cache then holds. See design_history.md, "Conversations switched away from are kept"."""
        store = self._kept
        assert store is not None, "a switch is asked only where caches are kept"
        kept = store.longest_start(target)
        switches = kept is not None and kept[1] > same + _WORTH_KEEPING_TOKENS
        # Taken out before the cache is kept beside the rest, which may drop the oldest.
        taken = store.take(kept[0]) if switches and kept is not None else None
        if len(self._tokens) - same > _WORTH_KEEPING_TOKENS and self.is_current \
                and store.may_keep(len(self._tokens)):
            try:
                store.keep(list(self._tokens), self._context.save_whole_state())
            except InternalError:
                raise
            except RuntimeError:
                pass                    # kept or not, the conversation goes on as it would
        if taken is None:
            return same
        tokens, state = taken
        self.version += 1
        self._rest_states, self._pinned, self._at_cuts = [], {}, []
        self._template_written = None
        self._logits = False
        try:
            self._context.restore_whole_state(state)
        except BaseException:
            self._end = None            # decoded whole from nothing at the next bring-up
            raise
        self._tokens = tokens
        self._end = self._valid = len(tokens)
        return common_prefix_length(tokens, target)

    def note_template_written(self, start: int, end: int) -> None:
        """Note that the cache holds from ``start`` to ``end`` only what the template wrote, no message's text, such as the prompt a reply is generated after: the next :meth:`hold` shifts out a stretch within it that its target leaves out, where the cache can shift. It moves with every removal before it and is forgotten with any other change there."""
        self._template_written = (start, end)

    @property
    def is_behind(self) -> bool:
        """Whether the cache holds the start of :attr:`tokens` exactly and only lacks the rest, as a decode an interrupt stopped leaves it."""
        return self._end is not None and self._end == self._valid < len(self._tokens)

    def remove(self, start: int, end: int) -> None:
        """Take the tokens from ``start`` to ``end`` out, those after them moving down."""
        if start >= end:
            return
        if self.is_behind and start < self._valid and self._context.can_shift:
            self._bring_up(want_logits=False)       # what it holds is shifted, not decoded again
        total = len(self._tokens)
        del self._tokens[start:end]
        self.version += 1
        if self._template_written is not None:
            lo, hi = self._template_written
            self._template_written = ((lo - (end - start), hi - (end - start)) if end <= lo
                                      else (lo, hi) if start >= hi else None)
        self._forget_states_after(start)
        self._logits = False
        if self._end == self._valid == total and self._context.can_shift:
            try:
                self._context.apply_eviction(start, end)  # one removal and one shift
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
        if pos < len(self._tokens):
            self.version += 1
        if self._template_written is not None and pos < self._template_written[1]:
            self._template_written = None
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

    def hold(self, target: list[int], *, want_logits: bool = False,
             find_dropped_stretch: bool = False, rest_at_cut: bool = False,
             begins: bool = False) -> int:
        """Make the cache hold ``target``: a stretch it no longer holds removed, everything from the first difference decoded; how many of its tokens it held already, before any stretch was removed.

        Args:
            want_logits: A generation follows the decode at once.
            find_dropped_stretch: The target may go on as the cache does after
                a stretch it no longer holds; that stretch is searched for and
                removed, the rest shifted, instead of decoded again.
            rest_at_cut: Rest where the cache is cut, before the rest is
                decoded, so that a cache that cannot cut can go back there.
            begins: The target is a conversation a begin writes, which may be
                another than the cache holds: the cache is kept where the target
                leaves most of it, and a kept one held where it serves better.
        """
        same = held = common_prefix_length(self._tokens, target)
        if begins and self._kept is not None and self._can_cut:
            # A cache that goes back to saved states keeps none: what it saved is the
            # conversation's, and going back without them would decode it all again.
            same = self._switch_for(target, same)
        changes = same < len(self._tokens)     # more than an append
        found = None
        if find_dropped_stretch and same < len(self._tokens):
            found = _dropped_stretch_end(self._tokens, target, same)
        template_written, self._template_written = self._template_written, None
        if found is None and changes and template_written is not None and self._context.can_shift \
                and template_written[0] <= same < template_written[1]:
            # What the template wrote before a reply and leaves out once the reply is past, such
            # as an empty thought, is shifted out rather than the reply decoded again. See
            # design_history.md, "What the template leaves out of a past reply is shifted out".
            found = _left_out_stretch_end(self._tokens, target, same, template_written[1])
        if found is not None:
            self.remove(same, found)
            same = common_prefix_length(self._tokens, target)
        self.cut_to(same)
        if rest_at_cut:
            self.rest()
            if changes and self._rest_states and self._rest_states[-1][0] == same:
                self._at_cuts = [s for s in self._at_cuts if s[0] < same][-1:] + [
                    self._rest_states[-1]]
        self.decode(target[same:], want_logits=want_logits)
        return held

    def kept_conversations(self) -> tuple | None:
        """The caches kept of conversations switched away from, as they stand now, for :meth:`put_back_kept`; ``None`` where none are kept."""
        return None if self._kept is None else self._kept.entries()

    def put_back_kept(self, kept: tuple | None) -> None:
        """Keep the caches :meth:`kept_conversations` gave as ``kept``, as a failed action leaves them."""
        if self._kept is not None and kept is not None:
            self._kept.put_back(kept)

    def replace(self, tokens: list[int]) -> None:
        """Hold ``tokens`` instead, as :meth:`hold` does without touching the cache, which is brought to them before it is next used."""
        if list(tokens) == self._tokens:
            return                      # nothing changed, the logits of the last decode included
        same = common_prefix_length(self._tokens, tokens)
        self._tokens = list(tokens)
        self.version += 1
        self._template_written = None
        self._valid = min(self._valid, same)
        self._forget_states_after(same)
        self._logits = False

    def forget_undecoded(self) -> None:
        """Forget the tokens the cache was left behind, where it holds those before them exactly, as a decode an interrupt stopped leaves it."""
        if self.is_behind:
            self.cut_to(self._valid)

    @contextlib.contextmanager
    def decoding_one_part(self, applies: bool = True):
        """Where ``applies``, decode at most one part of a long decode meanwhile, cancel or no cancel, leaving the rest to the next bring-up as an interrupt does."""
        if not applies:
            yield
            return
        cancel, interrupt = self.cancel, self.interrupt
        self.cancel, self.interrupt = None, _ALWAYS_SET
        try:
            yield
        finally:
            self.cancel, self.interrupt = cancel, interrupt

    def repair(self) -> bool:
        """Bring the cache to its tokens where a change was left for later or failed; whether it was needed.

        Where bringing it up fails, it is emptied and its tokens decoded
        whole once more; only a failure of that is raised.
        """
        if self.is_current:
            return False
        _log.debug("the cache is decoded again from token %d", self._valid)
        try:
            self._bring_up(want_logits=True)
        except Cancelled:
            raise
        except Exception:
            self._end = None                # decoded whole from nothing on the second try
            self._bring_up(want_logits=True)
        return True

    def rest(self) -> None:
        """Bring the cache up, then save the state it rests in, where it needs one and can save it; nothing where :attr:`interrupt` left the cache behind its tokens."""
        if self._interrupted and not self.is_current:
            return
        self._bring_up(want_logits=False)
        if self._can_cut:
            return                      # a cut needs no saved state
        end = self._brought_up_end
        state = self._pinned.get(end)               # one kept here already serves
        if state is None:
            try:
                state = self._context.save_state()
            except RuntimeError:
                self._rest_states = []  # decoding again stands in where none is saved
                return
        self._rest_states = [s for s in self._rest_states if s[0] < end][-1:] + [(end, state)]

    @property
    def _brought_up_end(self) -> int:
        """Where the cache ends, known once it was brought up, which every decode at its end follows."""
        assert self._end is not None, "the cache's end is unknown until it is brought up"
        return self._end

    @property
    def _interrupted(self) -> bool:
        """Whether :attr:`interrupt` is set, so that a long decode leaves what it has not done."""
        return self.interrupt is not None and self.interrupt.is_set()

    @property
    def keeps_saved_states(self) -> bool:
        """Whether the cache goes back to saved states, as one that can neither shift nor cut its end does."""
        return not self._can_cut

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
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled("cancelled before the decode")
        watched = [event for event in (self.cancel, self.interrupt) if event is not None]
        if watched and len(tokens) > self._part_tokens:
            for at in range(0, len(tokens), self._part_tokens):
                if at and self.cancel is not None and self.cancel.is_set():
                    raise Cancelled("cancelled during the decode")
                if at and self._interrupted:
                    # Held, not decoded: the next bring-up decodes them.
                    if not appended:
                        self._tokens += tokens[at:]
                    self._logits = False
                    return
                part = tokens[at:at + self._part_tokens]
                self._decode_at_end(part, want_logits and at + len(part) == len(tokens),
                                    appended)
            return
        end = self._brought_up_end
        pin = min((p for p in self._pins_to_save() if end < p < end + len(tokens)), default=None)
        if pin is not None:
            # Decoded in two, so that the state where the pin is can be saved; all of
            # them held first, so that a failure in either leaves none out.
            if not appended:
                self._tokens += tokens
            split = pin - end
            self._decode_at_end(tokens[:split], False, appended=True)
            self._decode_at_end(tokens[split:], want_logits, appended=True)
            return
        try:
            self._context.prefill(tokens, start_pos=end, want_logits=want_logits)
        except BaseException:
            self._end = None
            if not appended:
                self._tokens += tokens
            raise
        if not appended:
            self._tokens += tokens
        self._end = end + len(tokens)
        self._valid = self._end
        self._logits = want_logits
        self.decoded += len(tokens)
        self._save_pinned_state()

    def _save_pinned_state(self) -> None:
        """Save the state the cache holds now, where one is to be kept here."""
        if self._end in self._pins_to_save():
            # A state rested in here already serves.
            state = next((s for at, s in self._rest_states if at == self._end), None)
            if state is None:
                try:
                    state = self._context.save_state()
                except RuntimeError:
                    return              # decoding again stands in where none is saved
            self._pinned[self._end] = state

    def _pins_to_save(self) -> set[int]:
        """Where a state is to be kept and none is saved yet, on a cache that needs saved states."""
        if self._can_cut:
            return set()
        return set().union(*self._pins.values()) - set(self._pinned)

    def _decode_again_with_logits(self) -> None:
        """Decode the last token again, so that the next token can be drawn after it."""
        last, written = self._tokens[-1], self._template_written
        self.cut_to(len(self._tokens) - 1)
        self._bring_up(want_logits=False)
        self._decode_at_end([last], want_logits=True)
        self._template_written = written           # the same tokens, decoded again

    def _reset_to_nothing(self) -> None:
        self._context.reset()
        self._valid, self._end, self._rest_states, self._pinned, self._at_cuts = 0, 0, [], {}, []

    def _restore_to(self, pos: int) -> bool:
        """Go back to the newest state saved at or before ``pos``; whether one could be."""
        self._forget_states_after(pos)
        saved = self._rest_states + list(self._pinned.items()) + self._at_cuts
        if not saved:
            return False
        rested, state = max(saved, key=lambda s: s[0])
        try:
            self._context.restore_state(state, rested)
        except RuntimeError:
            self._rest_states, self._pinned, self._at_cuts = [], {}, []
            return False
        self._valid = self._end = rested
        return True

    def _forget_states_after(self, pos: int) -> None:
        """Drop the saved states that hold more than the first ``pos`` tokens, which changed."""
        self._rest_states = [s for s in self._rest_states if s[0] <= pos]
        self._pinned = {kept: state for kept, state in self._pinned.items() if kept <= pos}
        self._at_cuts = [s for s in self._at_cuts if s[0] <= pos]


def _dropped_stretch_end(held: list[int], target: list[int], start: int) -> int | None:
    """Where ``held`` goes on as ``target`` does from ``start``, the stretch before it dropped; ``None`` where nowhere.

    The longest run of at least eight tokens wins.
    """
    memory, wanted = _as_text(held), _as_text(target)
    probe = wanted[start:start + 8]
    best, best_length, at = None, len(probe) - 1, memory.find(probe, start + 1) if probe else -1
    longest = len(wanted) - start
    # Later candidates can only match less than what is left after them.
    candidates_variant = LoopVariant("finding a dropped stretch")
    while at >= 0 and best_length < min(longest, len(memory) - at):
        candidates_variant.passes(len(memory) - at)
        length = common_prefix_length(memory[at:], wanted[start:])
        if length > best_length:
            best, best_length = at, length
        at = memory.find(probe, at + 1)
    return best


class _KeptCaches:
    """The caches of conversations switched away from, each as its tokens and its whole state, oldest first, together within ``bound_bytes``."""

    def __init__(self, bound_bytes: int) -> None:
        self._bound = bound_bytes
        self._kept: list[tuple[list[int], Sized]] = []
        # The fewest tokens a cache held whose state alone took more than the bound.
        self._too_large: int | None = None

    def longest_start(self, target: list[int]) -> tuple[int, int] | None:
        """Which kept cache shares the longest start with ``target``, and how long that start is; ``None`` where none is kept."""
        shared = [(common_prefix_length(tokens, target), at)
                  for at, (tokens, _state) in enumerate(self._kept)]
        if not shared:
            return None
        length, at = max(shared)
        return at, length

    def may_keep(self, n_tokens: int) -> bool:
        """Whether a cache of ``n_tokens`` may fit the bound: none as long as one that held no more did not."""
        return self._too_large is None or n_tokens < self._too_large

    def entries(self) -> tuple:
        """The caches kept, and the fewest tokens one held that alone took more than the bound, as they stand now."""
        return list(self._kept), self._too_large

    def put_back(self, entries: tuple) -> None:
        """Keep what :meth:`entries` gave instead of what is kept now."""
        kept, self._too_large = entries
        self._kept = list(kept)

    def take(self, at: int) -> tuple[list[int], Sized]:
        """The kept cache ``at``, which is kept no longer."""
        return self._kept.pop(at)

    def keep(self, tokens: list[int], state) -> None:
        """Keep ``state``, saved while the cache held ``tokens``, the oldest going while all of them take more than the bound; none that alone takes more."""
        if len(state) > self._bound:
            self._too_large = len(tokens) if self._too_large is None else min(self._too_large,
                                                                                len(tokens))
            return
        self._kept.append((tokens, state))
        total = sum(len(kept) for _tokens, kept in self._kept)
        dropping_variant = LoopVariant("keeping the caches within their bound")
        while total > self._bound:
            dropping_variant.passes(len(self._kept))
            total -= len(self._kept.pop(0)[1])


def _left_out_stretch_end(held: list[int], target: list[int], start: int, last: int) -> int | None:
    """Where ``held`` goes on as ``target`` does from ``start``, a stretch ending at ``last`` at the latest left out; ``None`` where it does nowhere.

    ``held`` must go on as ``target`` for as long as either goes on, up to
    :data:`_LEFT_OUT_MATCH_TOKENS`.
    """
    wanted = target[start:start + _LEFT_OUT_MATCH_TOKENS]
    for end in range(start + 1, min(last, len(held)) + 1):
        after = held[end:end + _LEFT_OUT_MATCH_TOKENS]
        if after and after == wanted[:len(after)] and (len(after) == len(wanted)
                                                       or end + len(after) == len(held)):
            return end
    return None


def _as_text(tokens) -> str:
    return "".join(chr(token + _TOKEN_OFFSET) for token in tokens)
