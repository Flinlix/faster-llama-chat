"""What a template's renders allow for one opening of a conversation (its system turn, declarations and reasoning), worked out once and kept for the openings worked out last, and the room a short question needs on the date the template writes longest."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable

from .config import Config
from .conversation_state import ConversationState, RenderChecks
from .model_template import clock_forms_written, clock_text, moments_of_this_year, rendered_on
from .messages import system_part
from .render_checks import (ensure_history_stays_written, repeated_question_written_alike,
                            rest_stays_after_a_drop, splice_probe, tool_exchanges,
                            tool_turns_stay_written, windows_are_exact)
from .rendering import ConversationRendering
from .eviction import Eviction
from .errors import RAISE_THE_ROOM, ContextOverflowError, TemplateRefusedError
from .resolved_template import ResolvedTemplate


# How many openings' render checks are kept.
_CHECKS_KEPT = 4


class OpeningChecks:
    """The checks of what renders allow, by opening, and the room a question needs on any date."""

    def __init__(self, context, rendering: ConversationRendering, eviction: Eviction, state: ConversationState, config: Config, template: Callable[[], ResolvedTemplate], max_prompt_tokens: Callable[[], int]) -> None:
        self._context = context
        self._rendering = rendering
        self._eviction = eviction
        self._state = state
        self._config = config
        self._template = template
        self._max_prompt_tokens = max_prompt_tokens
        # The render checks of the openings worked out last, oldest first.
        self._checks_made: dict[tuple[str, int], RenderChecks] = {}
        # What a short question needs on the longest date, by the opening and year it was measured for.
        self._longest_opening: tuple[tuple, int] | None = None

    def remembered_render_checks(self) -> RenderChecks:
        """:meth:`render_checks`, kept for the few openings worked out last, where the template does not read the clock, so that going back to one of them renders nothing."""
        if self._template().properties.reads_clock:
            return self.render_checks()
        key = (json.dumps([system_part(self._state.messages), [spec.as_declaration() for spec in self._state.tools],
                           self._state.reasoning], sort_keys=True, default=str),
               id(self._template()))
        checks = self._checks_made.pop(key, None)
        if checks is None:
            checks = self.render_checks()
        self._checks_made[key] = checks                 # the newest last
        if len(self._checks_made) > _CHECKS_KEPT:      # one more than kept, at most
            del self._checks_made[next(iter(self._checks_made))]
        return checks

    def render_checks(self) -> RenderChecks:
        """What the template's renders allow, as the system turn, the declarations and the clock now leave them.

        Raises:
            ValueError: If the template writes an older exchange otherwise once
                another follows.
        """
        system = system_part(self._state.messages)
        render = self._rendering.render
        ensure_history_stays_written(render, system)
        tools = self._state.tools
        windows = windows_are_exact(render, system, self._rendering.tag_from)
        calls = self._template().properties.can_call_tools
        with clock_forms_written() as forms:
            opening = self._rendering.opening_render(system)
        return RenderChecks(
            opening=opening,
            clock_read=tuple(dict.fromkeys(forms)),
            windows=windows,
            # Only a template that calls tools is asked.
            tool_windows=windows and calls and windows_are_exact(
                render, system, self._rendering.tag_from,
                tool_exchanges(tools[0].name if tools else "probe")),
            splice_probe=splice_probe(render, system),
            repeats_alike=repeated_question_written_alike(render, system),
            with_a_short_question=self._eviction.with_a_short_question(),
            rest_stays=rest_stays_after_a_drop(render, system),
            rest_stays_around_calls=not calls or rest_stays_after_a_drop(
                render, system, tool_exchanges(tools[0].name if tools else "probe")),
            # Without the steps tool calling needs, no turn keeps a call, and the template is
            # not asked.
            tool_turns_stay=self._template().properties.can_call_tools and tool_turns_stay_written(
                render, system, tools[0].name if tools else "probe", thought=bool(self._state.reasoning)))

    def ensure_room_for_a_question(self, remedy: str) -> None:
        """Refuse a system turn and declarations that leave no room for a short question, under ``evict_above_tokens`` and with the room a reply keeps free.

        Raises:
            ContextOverflowError: With ``remedy`` as what to do instead.
        """
        needed = self._with_a_short_question_on_any_date()
        for limit, bound in ((self._config.evict_above_tokens, "the eviction threshold is"),
                             (self._max_prompt_tokens(), "the room beside the reply's is")):
            if needed > limit:
                raise ContextOverflowError(
                    f"the system prompt and its declarations need {needed} tokens with a short "
                    f"question but {bound} {limit}; {remedy}, or {RAISE_THE_ROOM}",
                    needed, limit)

    def clock_reads_as_before(self) -> bool:
        """Whether the clock reads now, in each form the opening wrote it in, as it did when the opening was rendered; a render that wrote none is rendered again to know."""
        read = self._state.checks.clock_read
        return bool(read) and all(clock_text(form) == text for form, text in read)

    def _with_a_short_question_on_any_date(self) -> int:
        """What the conversation takes with no history and the shortest question, on the day of this year whose date the template writes longest."""
        if not self._template().properties.reads_clock:
            return self._eviction.with_a_short_question()
        moments = moments_of_this_year()
        key = (json.dumps([system_part(self._state.messages), self._state.tools, self._state.reasoning],
                          sort_keys=True, default=str), self._template(), moments[0].year)
        if self._longest_opening is not None and self._longest_opening[0] == key:
            return self._longest_opening[1]
        self._rendering.forget_renders()
        with clock_forms_written() as written:
            longest = self._eviction.with_a_short_question()
        # The day whose date and time, in every form the template writes them in, take the
        # most tokens, whatever the language of the clock's names; and each month's last day,
        # for a template that names the months itself.
        forms = sorted({form for form, _text in written})
        day = max(moments, key=lambda moment: sum(
            len(self._context.tokenize(moment.strftime(form), parse_special=False))
            for form in forms))
        month_ends = [moment for moment, after in zip(moments, moments[1:] + [None])
                      if after is None or after.month != moment.month]
        try:
            for moment in dict.fromkeys([day, *month_ends]):
                with rendered_on(moment):
                    self._rendering.forget_renders()
                    with contextlib.suppress(TemplateRefusedError):     # a date it writes nothing on
                        longest = max(longest, self._eviction.with_a_short_question())
        finally:
            self._rendering.forget_renders()
        self._longest_opening = (key, longest)
        return longest
