"""Decoding ahead of the next question from a render of the newest exchanges alone, and dropping the oldest exchanges in such renders, where the template writes them as the whole render does.

:class:`Windows` holds no state of its own: each method is given the
conversation's state and its cache. See design_history.md, "Rendering windows".
"""

from __future__ import annotations

from .cache_sync import CacheSync
from .conversation_state import ConversationState, goes_on_from
from .messages import exchange_starts, system_part
from .render_checks import STAND_IN
from .template import common_prefix_length

# Returned where a window cannot stand for the whole render.
WHOLE = object()


class Windows:
    """Renders windows of the conversation state it is given, and drops its oldest exchanges in them."""

    def __init__(self, rendering, eviction, config) -> None:
        self._rendering, self._eviction, self._config = rendering, eviction, config

    def ahead(self, state: ConversationState, memory: CacheSync):
        """What the conversation decodes ahead of the next question, from a render of the system message and the newest two exchanges alone.

        The window is written as the whole render is from the first tag after
        the older exchange's reply on. That tag is found in the cache from where
        the newest question's run was decoded, that question being the one
        spliced there; the stretch from the tag to the run, and what the splice
        decoded, are checked token for token.

        Returns:
            Where in the cache that tag sits, the tokens from it up to where the
            next question goes, how many of the conversation's own follow the
            tag, and the window's render with where the next question's run
            begins in it; or ``WHOLE`` where the window cannot stand for the
            whole render.
        """
        starts = exchange_starts(state.messages)
        spliced_at = state.spliced_at
        if spliced_at is None or len(starts) < 2 or not goes_on_from(state, spliced_at[2]) \
                or state.messages[starts[-1]] is not spliced_at[2].messages[-1] or [
                m["role"] for m in state.messages[starts[-2]:]] != ["user", "assistant"] * 2 \
                or not self._rendering.template_writes(state.messages[starts[-2]:]):
            return WHOLE
        system = system_part(state.messages)
        window = (system + state.messages[starts[-2]:]
                  + [{"role": "user", "content": STAND_IN}])
        split = self._rendering.split(window, generation_prompt=True)
        if split is None or split.text.count(STAND_IN) != 1:
            return WHOLE
        older_reply = len(system) + 1
        anchor = self._rendering.tag_after_content(split, older_reply)
        run = self._rendering.run_start(split, older_reply + 1, anchor)
        ahead_run = self._rendering.run_start(split, len(window) - 1, anchor)
        if None in (anchor, run, ahead_run):
            return WHOLE
        to_question = self._rendering.tokenize(split.pieces(anchor, run))
        before = self._rendering.tokenize(split.pieces(anchor, ahead_run))
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
        length = len(self._rendering.tokenize(plain.pieces(anchor)))
        return at, before, length, split, ahead_run

    def drop_in(self, state: ConversationState, memory: CacheSync, kept_until, after: int) -> bool:
        """Drop the oldest exchanges while over the limit, each removed from a render of the oldest two kept.

        Args:
            kept_until: Where in the cache the window's tag sits now; nothing
                from there on may be removed.
            after: How many tokens the render has from that tag on.

        It stops before the newest two exchanges, which the window was rendered
        from.

        Returns:
            Whether the cache is the render up to that tag after it; where it
            is not, the whole conversation is to be rendered.
        """
        limit = self._config.evict_above_tokens
        if not (kept_until() + after > limit or self._eviction.too_many(state)):
            return True
        bound, most = self._eviction.drop_bound(state, limit), self._eviction.exchanges_after_drop(state)
        while kept_until() + after > bound or self._eviction.too_many(state, most):
            if len(exchange_starts(state.messages)) <= 2:
                break
            # The cache up to the tag after the second oldest exchange's reply, before the drop,
            # checked token for token, as no check at load covers it.
            held = self._rendering.head(state.messages, through=1)
            if held is None or list(memory.tokens[:len(held)]) != held:
                return False
            if not self._eviction.drop_oldest_exchange(state, True):
                break
            if not self._remove_dropped(state, memory, held, kept_until()):
                return False
        return True

    def _remove_dropped(self, state: ConversationState, memory: CacheSync, held: list[int],
                        bound: int) -> bool:
        """Remove the oldest exchange, just dropped, from the cache, which begins with ``held``; nothing from ``bound`` on.

        After the drop, the cache up to the tag that ended ``held`` is the
        render of the system message and the oldest two exchanges kept, up to
        the tag after the first one's reply, and everything after that tag is
        as before.
        """
        wanted = self._rendering.head(state.messages, through=0)
        if wanted is None:
            return False
        same = common_prefix_length(held, wanted)
        end = len(held) - (len(wanted) - same)
        if end <= same or end > bound or held[end:] != wanted[same:]:
            return False
        memory.remove(same, end)
        state.kept_at = {k: at - (end - same) if at >= end else at
                               for k, at in state.kept_at.items()}
        return list(memory.tokens[:len(wanted)]) == wanted
