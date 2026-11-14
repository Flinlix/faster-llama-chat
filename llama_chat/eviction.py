"""Which of a conversation's oldest exchanges go, and when: at a begin, and once it no longer fits or holds too many.

:class:`Eviction` holds no state of its own: it changes the conversation's
state, and records there what went and why, so that an action put back puts
back what an eviction did too. See design_history.md, "How eviction works".
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable

from .config import Config
from .conversation_state import ConversationState
from .messages import THOUGHT_KEY, exchange_count, exchange_starts, history_count, system_part

_log = logging.getLogger(__name__)

# Where the cache cannot shift, the fraction of the room under the eviction
# threshold, after a short question, and of ``max_history_exchanges``, that an
# eviction brings the conversation down to. In (0, 1]. See design_history.md,
# "Settings, add-on files and logging", 266.3.
EVICT_TO_FRACTION_WITHOUT_SHIFT = 0.5


class Eviction:
    """Drops the oldest exchanges of the conversation state it was made with, measured with ``rendering``."""

    def __init__(self, rendering, config: Config, can_shift: bool,
                 state: ConversationState) -> None:
        self._rendering, self._config, self._can_shift = rendering, config, can_shift
        self._state = state

    def with_a_short_question(self) -> int:
        """How many tokens the conversation takes with no history and a short question: what no eviction makes smaller."""
        return self._rendering.length(system_part(self._state.messages)
                                      + [{"role": "user", "content": "?"}], generation_prompt=True)

    def keep_newest_history(self, bound: int,
                            refused: Callable[[list[dict]], Exception]) -> None:
        """Keep of a history to begin with the newest exchanges that render within ``bound``, and at least the newest; what went is the caller's to record.

        Raises:
            Exception: What ``refused`` makes of the messages the template refuses to write.
        """
        first = len(system_part(self._state.messages))
        if len(self._state.messages) == first:
            return
        opens = [first] + [i for i in exchange_starts(self._state.messages) if i > first]

        def length(count: int) -> int:
            kept = self._state.messages[:first] + self._state.messages[opens[-count]:]
            if self._rendering.split(kept, kept[-1]["role"] == "tool") is None:
                raise refused(kept)
            return self._rendering.length(kept, generation_prompt=False)

        def fits(count: int) -> bool:
            return length(count) <= bound

        everything = len(opens)
        # The characters of the newest exchanges, which the lengths measured are scaled by.
        sizes = [len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
                 + len(str(m.get(THOUGHT_KEY) or "")) for m in self._state.messages]
        chars = lambda count: sum(sizes[opens[-count]:]) or 1         # noqa: E731
        kept, more, jumped = 1, min(2, everything), False
        while True:
            measured = length(more)
            if measured <= bound:
                if more == everything:
                    return
                kept = more
                # Where the whole would fit at the rate measured per character, it is tried
                # next, once; where it does not, doubling goes on.
                jumped, more = (True, everything) if not jumped and \
                    measured * chars(everything) <= bound * chars(more) \
                    else (jumped, min(more * 2, everything))
            elif jumped and more == everything and kept * 2 < everything:
                more = kept * 2             # the whole did not fit: doubling goes on from what fits
            else:
                break
        while kept < more - 1:
            middle = (kept + more) // 2
            kept, more = (middle, more) if fits(middle) else (kept, middle)
        self._state.messages[:] = self._state.messages[:first] + self._state.messages[opens[-kept]:]

    def drop_while_over(self, limit: int, generation_prompt: bool, newest_too: bool = False,
                        tokens_only: bool = False, beside_open_turn: bool = False) -> str | None:
        """Drop the oldest exchanges once the conversation renders longer than ``limit`` or holds too many; why they went, ``None`` where none did.

        Where the cache cannot shift, they go until the history is well below
        its room under ``limit`` and its count well below
        ``max_history_exchanges``.

        Args:
            tokens_only: Leave ``max_history_exchanges`` out, which may keep
                nothing on purpose.
            beside_open_turn: Where the open turn alone leaves the history no
                room well below ``limit``, measure that room beside the turn.
        """
        def length() -> int:
            return self._rendering.length(self._state.messages, generation_prompt)

        def too_many(most: int | None = None) -> int:
            return 0 if tokens_only else self._exchanges_over(most)

        if length() <= limit and not too_many():
            return None
        cause = "exchange_limit" if too_many() else "room"
        n_before = history_count(self._state.messages)
        bound, most = self.drop_bound(limit, beside_open_turn), self.exchanges_after_drop()
        self._drop_oldest_exchanges(length, bound, too_many(most), newest_too)
        return cause if self.record(n_before, cause) else None

    def record(self, n_before: int, cause: str) -> int:
        """Record in the state the history messages that went since it held ``n_before``, and ``cause``, one of :data:`~llama_chat.reply.EVICTION_CAUSES`, as why; how many went."""
        n_gone = n_before - history_count(self._state.messages)
        if n_gone:
            self._state.changes.n_evicted += n_gone
            self._state.changes.evicted_for = cause
            _log.info("the %d oldest history message(s) went %s", n_gone, {
                "room": "to make room",
                "exchange_limit": f"to keep max_history_exchanges="
                                  f"{self._config.max_history_exchanges}",
                "half_context": f"so that half of the {self._config.context_size}-token "
                                "context stays free at a begin"}[cause])
        return n_gone

    def _drop_oldest_exchanges(self, length: Callable[[], int],
                               bound: int, at_least: int, newest_too: bool) -> bool:
        """Drop the fewest oldest exchanges, ``at_least`` of them, after which ``length()`` is within ``bound``, or every one that can go; whether any went."""
        messages = list(self._state.messages)

        def dropping(count: int) -> int:
            """Drop ``count`` exchanges from the conversation as it was; how many could go."""
            self._state.messages[:] = messages
            return sum(1 for _ in itertools.takewhile(
                bool, (self.drop_oldest_exchange(newest_too) for _ in range(count))))

        high = dropping(len(messages))
        long_high = length()
        if long_high > bound or high <= at_least:
            return high > 0             # as many as can go, whether or not that is enough
        dropping(0)
        low, long_low = 0, length()     # over after ``low`` drops, within after ``high``
        if at_least:
            low = at_least - 1          # over in count, whatever its length
        halve = False
        while high - low > 1:
            if halve or long_low <= long_high:
                count = (low + high) // 2
            else:
                count = low + math.ceil((high - low) * (long_low - bound) / (long_low - long_high))
            count = min(max(count, low + 1, at_least), high - 1)
            dropping(count)
            measured = length()
            width = high - low
            if measured > bound or count < at_least:
                low, long_low = count, measured
            else:
                high, long_high = count, measured
            halve = not halve and (high - low) * 2 > width
        dropping(high)
        return high > 0

    def drop_bound(self, limit: int, beside_open_turn: bool = False) -> int:
        """How far an eviction brings the conversation down once it is over ``limit``.

        Where the cache cannot shift, what a short question alone takes and that
        fraction of the room after it. With ``beside_open_turn``, where the open
        turn alone goes past that, what the open turn takes with the system turn
        and that fraction of the room after them.
        """
        if self._can_shift:
            return limit
        system = min(self.with_a_short_question(), limit)
        bound = system + int((limit - system) * EVICT_TO_FRACTION_WITHOUT_SHIFT)
        if beside_open_turn:
            starts = exchange_starts(self._state.messages)
            open_turn = self._state.messages[starts[-1] if starts else len(system_part(self._state.messages)):]
            alone = min(self._rendering.length(system_part(self._state.messages) + open_turn,
                                               generation_prompt=True), limit)
            if alone > bound:
                return alone + int((limit - alone) * EVICT_TO_FRACTION_WITHOUT_SHIFT)
        return bound

    def exchanges_after_drop(self) -> int | None:
        """How many exchanges an eviction leaves once there are more than ``max_history_exchanges``.

        Where the cache cannot shift, that fraction of them, at least one.
        """
        limit = self._config.max_history_exchanges
        if limit is None or self._can_shift:
            return limit
        return min(limit, max(1, int(limit * EVICT_TO_FRACTION_WITHOUT_SHIFT)))

    def too_many(self, most: int | None = None) -> bool:
        """Whether the history holds more exchanges than ``most``, by default ``max_history_exchanges``."""
        return self._exchanges_over(most) > 0

    def _exchanges_over(self, most: int | None = None) -> int:
        """How many exchanges the history holds beyond ``most``, by default ``max_history_exchanges``."""
        limit = self._config.max_history_exchanges if most is None else most
        return 0 if limit is None else max(0, exchange_count(self._state.messages) - limit)

    def drop_oldest_exchange(self, newest_too: bool = False) -> bool:
        """Drop the oldest question and everything up to the next; ``False`` where none can go.

        Args:
            newest_too: Let the newest exchange go as well, once its reply is
                complete, as it may after a reply when the next question is to come.
        """
        first = 1 if self._state.messages and self._state.messages[0]["role"] == "system" else 0
        following = next((i for i in exchange_starts(self._state.messages) if i > first),
                         len(self._state.messages))
        if following == len(self._state.messages) and not (
                newest_too and len(self._state.messages) > first
                and self._state.messages[-1]["role"] == "assistant"):
            return False
        del self._state.messages[first:following]
        return True

