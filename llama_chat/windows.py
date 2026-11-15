"""Decoding ahead of the next question from a render of the newest exchanges alone, and dropping the oldest exchanges in such renders, where the template writes them as the whole render does.

:class:`Windows` holds no state of its own: it reads and changes the
conversation's state, and is given its cache. See design_history.md,
"Rendering windows".
"""

from __future__ import annotations

from .cache_sync import CacheSync
from .conversation_state import ConversationState, goes_on_from
from .messages import exchange_starts, history_count, system_part
from .render_checks import STAND_IN
from .sequences import common_prefix_length

# Returned where a window cannot stand for the whole render.
WHOLE = object()


class Windows:
    """Renders windows of the conversation state it was made with, and drops its oldest exchanges in them."""

    def __init__(self, rendering, eviction, config, state: ConversationState) -> None:
        self._rendering, self._eviction, self._config = rendering, eviction, config
        self._state = state

    def ahead(self, memory: CacheSync):
        """What the conversation decodes ahead of the next question, from a render of the system message and the newest two exchanges alone.

        Returns:
            Where in the cache that tag sits, the tokens from it up to where the
            next question goes, how many of the conversation's own follow the
            tag, the window's render with where the next question's run
            begins in it, and where each turn kept as generated after the tag
            begins in those tokens; or ``WHOLE`` where the window cannot stand
            for the whole render.
        """
        messages = self._state.messages
        starts = exchange_starts(messages)
        spliced_at = self._state.spliced_at
        if spliced_at is None or len(starts) < 2 or not goes_on_from(self._state, spliced_at[2]) \
                or messages[starts[-1]] is not spliced_at[2].messages[-1] \
                or messages[starts[-1] - 1]["role"] != "assistant" or messages[-1]["role"] != "assistant" \
                or not (self._state.checks.tool_windows
                        or self._rendering.template_writes(messages[starts[-2]:])):
            return WHOLE
        system = system_part(messages)
        window = system + messages[starts[-2]:] + [{"role": "user", "content": STAND_IN}]
        split = self._rendering.split(window, generation_prompt=True)
        if split is None or split.text.count(STAND_IN) != 1:
            return WHOLE
        older_reply = len(system) + starts[-1] - starts[-2] - 1
        anchor = self._rendering.tag_after_content(split, older_reply)
        run = self._rendering.run_start(split, older_reply + 1, anchor)
        ahead_run = self._rendering.run_start(split, len(window) - 1, anchor)
        if None in (anchor, run, ahead_run):
            return WHOLE
        try:
            to_question = self._rendering.target(window, True, upto=run, start=anchor).tokens
            ahead = self._rendering.target(window, True, upto=ahead_run, start=anchor)
        except ValueError:
            return WHOLE                    # a generated turn the window cannot place
        before = ahead.tokens
        question_at, spliced, _ = spliced_at
        at = question_at - len(to_question)
        if at < 0 or list(memory.tokens[at:question_at]) != to_question \
                or list(memory.tokens[question_at:question_at + len(spliced)]) != spliced \
                or before[:len(to_question)] != to_question:
            return WHOLE
        # What counts against the limits, as :meth:`ConversationRendering.length` counts it: the
        # window without the stand-in.
        plain = self._rendering.split(window[:-1], generation_prompt=False)
        if plain is None or plain.text[:anchor] != split.text[:anchor]:
            return WHOLE
        try:
            length = len(self._rendering.target(window[:-1], False, start=anchor).tokens)
        except ValueError:
            return WHOLE
        return at, before, length, split, ahead_run, ahead.turns_at

    def drop_in(self, memory: CacheSync, kept_until, after: int) -> bool:
        """Drop the oldest exchanges while over the limit, each removed from a render of the oldest two kept.

        It stops before the newest two exchanges, which the window was rendered
        from.

        Args:
            kept_until: Where in the cache the window's tag sits now; nothing
                from there on may be removed.
            after: How many tokens the render has from that tag on.

        Returns:
            Whether the cache is the render up to that tag after it; where it
            is not, the whole conversation is to be rendered.
        """
        limit = self._config.evict_above_tokens
        if not (kept_until() + after > limit or self._eviction.too_many()):
            return True
        cause = "exchange_limit" if self._eviction.too_many() else "room"
        n_before = history_count(self._state.messages)
        try:
            return self._drop_in(memory, kept_until, after)
        finally:
            self._eviction.record(n_before, cause)

    def _drop_in(self, memory: CacheSync, kept_until, after: int) -> bool:
        """:meth:`drop_in`'s work, once the conversation is over a limit."""
        limit = self._config.evict_above_tokens
        bound, most = self._eviction.drop_bound(limit), self._eviction.exchanges_after_drop()
        while kept_until() + after > bound or self._eviction.too_many(most):
            starts = exchange_starts(self._state.messages)
            if len(starts) <= 2:
                break
            # An exchange that leaves a turn open writes what follows it otherwise once it goes.
            oldest_end = self._state.messages[starts[1] - 1]
            if oldest_end["role"] != "assistant" or oldest_end.get("tool_calls"):
                return False
            # The cache up to the tag after the second oldest exchange's reply, before the drop,
            # checked token for token, as no check at load covers it.
            held = self._rendering.head(self._state.messages, through=1,
                                        with_generated_turns=self._state.checks.tool_windows)
            if held is None or list(memory.tokens[:len(held)]) != held:
                return False
            if not self._eviction.drop_oldest_exchange(True):
                break
            if not self._remove_dropped(memory, held, kept_until()):
                return False
        return True

    def _remove_dropped(self, memory: CacheSync, held: list[int],
                        bound: int) -> bool:
        """Remove the oldest exchange, just dropped, from the cache, which begins with ``held``; nothing from ``bound`` on.

        After the drop, the cache up to the tag that ended ``held`` is the
        render of the system message and the oldest two exchanges kept, up to
        the tag after the first one's reply, and everything after that tag is
        as before.
        """
        wanted = self._rendering.head(self._state.messages, through=0,
                                      with_generated_turns=self._state.checks.tool_windows)
        if wanted is None:
            return False
        same = common_prefix_length(held, wanted)
        end = len(held) - (len(wanted) - same)
        if end <= same or end > bound or held[end:] != wanted[same:]:
            return False
        memory.remove(same, end)
        self._state.turns_at = {k: at - (end - same) if at >= end else at
                               for k, at in self._state.turns_at.items()}
        return list(memory.tokens[:len(wanted)]) == wanted
