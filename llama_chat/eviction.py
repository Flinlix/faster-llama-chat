"""Which of a conversation's oldest exchanges go, and when: at a begin, and once it no longer fits or holds too many.

:class:`Eviction` holds no state of its own: each method is given the
conversation's state and changes it there, so that an action put back puts
back what an eviction did too. See design_history.md, "How eviction works".
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

from .config import Config
from .conversation_state import ConversationState
from .messages import THOUGHT_KEY, exchange_count, exchange_starts, system_part

# Where the cache cannot shift, the fraction of the room under the eviction
# threshold, after a short question, and of ``max_history_exchanges``, that an
# eviction brings the conversation down to. In (0, 1]. See design_history.md,
# "Settings, add-on files and logging", 266.3.
EVICT_TO_FRACTION_WITHOUT_SHIFT = 0.5


class Eviction:
    """Drops the oldest exchanges of the conversation state it is given, measured with ``rendering``."""

    def __init__(self, rendering, config: Config, can_shift: bool) -> None:
        self._rendering, self._config, self._can_shift = rendering, config, can_shift

    def with_a_short_question(self, state: ConversationState) -> int:
        """How many tokens the conversation takes with no history and a short question: what no eviction makes smaller."""
        return self._rendering.length(system_part(state.messages)
                                      + [{"role": "user", "content": "?"}], generation_prompt=True)

    def keep_newest_history(self, state: ConversationState, bound: int,
                            refused: Callable[[list[dict]], Exception]) -> None:
        """Keep of a history to begin with the newest exchanges that render within ``bound``, and at least the newest.

        Raises:
            Exception: What ``refused`` makes of the messages the template refuses to write.
        """
        first = len(system_part(state.messages))
        if len(state.messages) == first:
            return
        opens = [first] + [i for i in exchange_starts(state.messages) if i > first]

        def length(count: int) -> int:
            kept = state.messages[:first] + state.messages[opens[-count]:]
            if self._rendering.split(kept, kept[-1]["role"] == "tool") is None:
                raise refused(kept)
            return self._rendering.length(kept, generation_prompt=False)

        def fits(count: int) -> bool:
            return length(count) <= bound

        everything = len(opens)
        # The characters of the newest exchanges, which the lengths measured are scaled by.
        sizes = [len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or ""))
                 + len(str(m.get(THOUGHT_KEY) or "")) for m in state.messages]
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
        state.messages = state.messages[:first] + state.messages[opens[-kept]:]

    def drop_while_over(self, state: ConversationState, limit: int, generation_prompt: bool, newest_too: bool = False,
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
            return self._rendering.length(state.messages, generation_prompt)

        def too_many(most: int | None = None) -> int:
            return 0 if tokens_only else self._exchanges_over(state, most)

        if length() <= limit and not too_many():
            return None
        cause = "exchange_limit" if too_many() else "room"
        bound, most = self.drop_bound(state, limit, beside_open_turn), self.exchanges_after_drop(state)
        dropped = self._drop_oldest_exchanges(state, length, bound, too_many(most), newest_too)
        return cause if dropped else None

    def _drop_oldest_exchanges(self, state: ConversationState, length: Callable[[], int],
                               bound: int, at_least: int, newest_too: bool) -> bool:
        """Drop the fewest oldest exchanges, ``at_least`` of them, after which ``length()`` is within ``bound``, or every one that can go; whether any went."""
        messages, kept = list(state.messages), state.kept

        def dropping(count: int) -> int:
            """Drop ``count`` exchanges from the conversation as it was; how many could go."""
            state.messages[:], state.kept = messages, kept
            return sum(1 for _ in itertools.takewhile(
                bool, (self.drop_oldest_exchange(state, newest_too) for _ in range(count))))

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

    def drop_bound(self, state: ConversationState, limit: int, beside_open_turn: bool = False) -> int:
        """How far an eviction brings the conversation down once it is over ``limit``.

        Where the cache cannot shift, what a short question alone takes and that
        fraction of the room after it. With ``beside_open_turn``, where the open
        turn alone goes past that, what the open turn takes with the system turn
        and that fraction of the room after them.
        """
        if self._can_shift:
            return limit
        system = min(self.with_a_short_question(state), limit)
        bound = system + int((limit - system) * EVICT_TO_FRACTION_WITHOUT_SHIFT)
        if beside_open_turn:
            starts = exchange_starts(state.messages)
            open_turn = state.messages[starts[-1] if starts else len(system_part(state.messages)):]
            alone = min(self._rendering.length(system_part(state.messages) + open_turn,
                                               generation_prompt=True), limit)
            if alone > bound:
                return alone + int((limit - alone) * EVICT_TO_FRACTION_WITHOUT_SHIFT)
        return bound

    def exchanges_after_drop(self, state: ConversationState) -> int | None:
        """How many exchanges an eviction leaves once there are more than ``max_history_exchanges``.

        Where the cache cannot shift, that fraction of them, at least one.
        """
        limit = self._config.max_history_exchanges
        if limit is None or self._can_shift:
            return limit
        return min(limit, max(1, int(limit * EVICT_TO_FRACTION_WITHOUT_SHIFT)))

    def too_many(self, state: ConversationState, most: int | None = None) -> bool:
        """Whether the history holds more exchanges than ``most``, by default ``max_history_exchanges``."""
        return self._exchanges_over(state, most) > 0

    def _exchanges_over(self, state: ConversationState, most: int | None = None) -> int:
        """How many exchanges the history holds beyond ``most``, by default ``max_history_exchanges``."""
        limit = self._config.max_history_exchanges if most is None else most
        return 0 if limit is None else max(0, exchange_count(state.messages) - limit)

    def drop_oldest_exchange(self, state: ConversationState, newest_too: bool = False) -> bool:
        """Drop the oldest question and everything up to the next; ``False`` where none can go.

        Args:
            newest_too: Let the newest exchange go as well, once its reply is
                complete, as it may after a reply when the next question is to come.
        """
        first = 1 if state.messages and state.messages[0]["role"] == "system" else 0
        following = next((i for i in exchange_starts(state.messages) if i > first),
                         len(state.messages))
        if following == len(state.messages) and not (
                newest_too and len(state.messages) > first
                and state.messages[-1]["role"] == "assistant"):
            return False
        gone = state.messages[first:following]
        del state.messages[first:following]
        state.kept = [k for k in state.kept
                            if not any(k.first is m or k.last is m for m in gone)]
        return True

