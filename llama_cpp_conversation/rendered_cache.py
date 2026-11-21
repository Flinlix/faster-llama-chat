"""A conversation as the messages its template is given, and a cache kept holding the template's render of them.

The template renders the conversation, split by a marker render into its own
text and the caller's; a turn that called tools is held in place of the
template's version of it as the tokens it was generated as. The cache is
compared with that target token by token, a stretch the target no longer holds
is removed and the rest shifted, and everything from the first token that
differs is decoded again. The render for the next question is decoded after
each reply, with a stand-in where the question goes, so a question needs only
its own tokens decoded before its reply starts; where the template writes the
newest exchanges and the oldest kept ones the same in a render of them alone,
only those are rendered. See docs/decisions/0001-the-cache-holds-the-templates-render.md, docs/decisions/0005-every-caller-text-is-placed-by-a-marker-render.md and docs/decisions/0026-renders-are-windows-of-the-newest-exchanges.md.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import threading
from collections.abc import Callable, Generator, Sequence
from typing import Any, cast
from dataclasses import replace

from .conversation_types import Conversation, GeneratedTurns
from .generation import AnswerGuard, GenerationAccumulator
from .cache_sync import CacheSync, Cancelled, Interrupt
from .config import Config
from .conversation_state import Changes, ConversationState, Published, Snapshot
from .history_forms import (answer_in_order, completions_messages, read_messages,
                            with_generated_turns, with_questions_joined, without_unreadable)
from .model_template import clock_forms_written
from .messages import (GENERATED_TURN, UNKNOWN_FUNCTION, UNREADABLE_CALL, calling_message,
                       exchange_count, exchange_starts, history_count, next_call_number,
                       result_message, system_part, unanswered)
from .render_checks import STAND_IN
from .rendering import ConversationRendering, GeneratedTurn, Target, generated_turns_in, with_generated_turn
from .eviction import Eviction
from .result_cutting import ResultCutting
from .windows import WHOLE, Windows
from .publishing import conversation_of, generated_turns_of, published
from .errors import (RAISE_THE_ROOM, ContextOverflowError, InternalError, InvalidArgumentError,
                     RefusalError, OutOfOrderError, TemplateRefusedError, UnsupportedError,
                     with_template_reason)
from .resolved_template import ResolvedTemplate, ResolvedTemplates
from .sequences import common_prefix_length
from .reply import EvictionCause, Reply, Report
from .reply_text import ReplyText
from .tools import ToolCall, ToolSpec, result_text
from .loop_variant import LoopVariant
from .cache_targets import CacheTargets
from .opening_checks import OpeningChecks
from .cache_holding import THE_REPLY, CacheHolding
from .kept_turns import KeptTurns
from .message_admission import INJECT_A_QUESTION_FIRST, MessageAdmission

_log = logging.getLogger(__name__)


class RenderedCache:
    """One conversation's messages, as its template is given them, and the cache holding their render.

    Every method expects the caller to hold the conversation's lock, except
    :meth:`conversation`, :meth:`generated_turns`, :meth:`published_pending_calls`,
    :meth:`published_total_tokens`, :meth:`published_turn_open`,
    :meth:`published_awaits_reply` and :meth:`ensure_can_reply`, which read a
    copy published after each change.
    Each action leaves the cache equal to the template's render of the
    messages, a turn that called tools held as it was generated, or, where it
    raises, the conversation and the cache as they were before it.
    """

    def __init__(self, context, templates: ResolvedTemplates, config: Config) -> None:
        self._context = context
        self._templates = templates
        self._config = config
        self._memory = CacheSync(context, config.micro_batch_tokens, config.kept_conversation_bytes)
        # Each template's answer guard, by the template's identity, the template kept alive with it.
        self._guards: dict[int, tuple[ResolvedTemplate, AnswerGuard]] = {}
        # The one state object every part reads and changes; an action put back fills it in place.
        self._state = ConversationState()
        self._rendering = ConversationRendering(context, lambda: self.template, self._state)
        self._targets = CacheTargets(self._rendering, self._state, self._memory)
        self._holding = CacheHolding(self._memory, self._rendering, self._targets, self._state, config,
                                     lambda: self.template)
        self._eviction = Eviction(self._rendering, config, context.can_shift, self._state,
                                  self._targets.length)
        self._cutting = ResultCutting(self._targets.length, self._state, self._text_within)
        self._windows = Windows(self._rendering, self._eviction, config, self._state)
        self._kept = KeptTurns(context, self._rendering, self._targets, self._state,
                               lambda: self.template, lambda: self.turn_open)
        self._admission = MessageAdmission(self._rendering, self._state, self._memory, self._targets,
                                           lambda: self._joins_questions,
                                           lambda: self._question_room, lambda: self.turn_open,
                                           self._newest_exchange)
        self._opening = OpeningChecks(context, self._rendering, self._eviction, self._state, config,
                                      lambda: self.template, lambda: self._max_prompt_tokens)
        # Where the newest generation's turn starts in the cache, and whether it went on
        # with an open turn; left as they are by a failed action.
        self._generation: tuple[int, bool] = (0, False)
        # The conversation and the cache as the open attempt found them, until its reply is
        # recorded, and as the newest request that asked a question found them.
        self._attempt: Snapshot | None = None
        self._request: Snapshot | None = None
        self._published = Published()
        # How many actions, one inside another, run now; the outermost publishes once it ends.
        self._depth = 0
        # Whether a reply left the work after it to do, the history messages that work
        # evicted, with why, which the next action reports, and the tokens it decoded,
        # which the next request counts as its own.
        self._settle_due = False
        self._drop_left = False             # whether the work a reply left waited for a question
        self._evicted_after_reply: tuple[int, EvictionCause | None] = (0, None)
        self._decoded_after_reply = 0
        # What ``check_cache`` found once the work after a reply ended, which the next action
        # raises, as that work may run on a thread no caller reads.
        self._found_fault: InternalError | None = None

    # ----- what it holds ---------------------------------------------------------------
    @property
    def messages(self) -> list[dict]:
        """The conversation as the template is given it; not to be changed by the caller."""
        return self._state.messages

    def conversation(self) -> Conversation:
        """Everything ``begin`` takes to begin the conversation again, by its parameter names, from one publication, the caller's to change; safe to read unlocked."""
        return conversation_of(self._published)

    def held_tokens(self) -> list[int]:
        """The tokens the cache holds, or is brought to hold before it is next used."""
        return list(self._memory.tokens)

    def prompted_tokens(self) -> list[int]:
        """The tokens of the conversation with the generation prompt after it, as a reply now would be generated after."""
        return list(self._holding.prompted().tokens)

    def generated_turns(self) -> GeneratedTurns | None:
        """Each turn kept as generated, from one publication, as :meth:`start` takes them back; ``None`` where none is; safe to read unlocked."""
        return generated_turns_of(self._published)

    def published_pending_calls(self) -> list[ToolCall]:
        """The calls still owed a result, oldest first, from the newest publication, the caller's to change; safe to read unlocked."""
        return copy.deepcopy(list(self._published.pending_calls))

    def published_total_tokens(self) -> int:
        """How many tokens the cache holds, from the newest publication; safe to read unlocked."""
        return self._published.total_tokens

    def published_awaits_reply(self) -> bool:
        """Whether a reply without a question would answer something, from one publication: the newest message is a question, or a turn's results are all in; safe to read unlocked."""
        published = self._published
        if published.turn_open:
            return not published.pending_calls
        return published.begun and published.newest_role == "user"

    def published_turn_open(self) -> bool:
        """Whether the newest turn is open, from the newest publication; safe to read unlocked."""
        return self._published.turn_open

    def ensure_can_reply(self, asks_question: bool) -> None:
        """Refuse a reply the conversation, as the newest publication left it, cannot take; safe to call unlocked.

        Raises:
            ValueError: As :meth:`prompt` and :meth:`generate` refuse, or, with
                ``asks_question``, where no conversation was begun.
        """
        _ensure_can_reply(self._published, asks_question)

    @property
    def template(self) -> ResolvedTemplate:
        """The template the conversation is written with: the model's template for tools where it was begun with tools."""
        return self._templates.for_conversation(self._state.uses_tool_template)

    @property
    def tools(self) -> list[ToolSpec]:
        """What the conversation declares."""
        return list(self._state.tools)

    @property
    def begun(self) -> bool:
        """Whether a conversation was begun and the cache holds it."""
        return self._state.begun

    @property
    def reasoning(self) -> bool | str:
        """Whether the conversation reasons, or the effort level it reasons at, as it was begun."""
        return self._state.reasoning

    @property
    def tools_declared(self) -> bool:
        """Whether the conversation is written as one that declares tools, even none."""
        return self._state.uses_tool_template

    @property
    def has_system_prompt(self) -> bool:
        """Whether the conversation holds a system prompt."""
        return bool(self._system_part())

    @property
    def turn_open(self) -> bool:
        """Whether the newest turn called tools and is not closed: a call is owed a result, or a result waits for the reply after it."""
        return bool(self.pending_calls) or self._ends_on_result()

    @property
    def pending_calls(self) -> list[ToolCall]:
        """The calls still owed a result, oldest first."""
        return [call for call, _ in unanswered(self._state.messages)]

    @property
    def newest_role(self) -> str | None:
        """The role of the newest message, the system message included; ``None`` before any."""
        return self._state.messages[-1]["role"] if self._state.messages else None

    @property
    def total_tokens(self) -> int:
        return len(self._memory.tokens)

    @property
    def decoded(self) -> int:
        """Every token decoded so far, for what a request cost."""
        return self._memory.decoded

    @property
    def turn_policy(self):
        """The policy the newest turn was started with, which a reply going on with it keeps; the wrapper's to set."""
        return self._state.turn_policy

    @turn_policy.setter
    def turn_policy(self, policy) -> None:
        if policy is not self._state.turn_policy:
            self._state.turn_policy = policy
            self._publish()

    def published_turn_policy(self):
        """The policy the newest turn was started with, from the newest publication; safe to read unlocked."""
        return self._published.turn_policy

    @property
    def turn_budget(self):
        """The calls the newest turn may still run, which a reply going on with it keeps; the wrapper's to set and spend."""
        return self._state.turn_budget

    @turn_budget.setter
    def turn_budget(self, budget) -> None:
        self._state.turn_budget = budget

    @property
    def results_cut(self) -> list[ToolCall]:
        """The calls whose result was cut to fit since the newest generation, oldest first."""
        return list(self._state.results_cut)

    @property
    def called_in_thought(self) -> bool:
        """Whether the open turn's newest call was made inside a thought the model had not closed."""
        return self._state.called_in_thought

    @property
    def prompt_text(self) -> str:
        """The template text the next generation follows, which says whether it opens in a thought."""
        return self._state.prompt_text

    # ----- beginning ------------------------------------------------------------------
    def start(self, history: list, tools: list[ToolSpec] | None, reasoning: bool | str,
              generated_turns: dict | None, unreadable_result: dict) -> None:
        """Begin again from ``history``, Chat Completions messages, the newest of them that fit in half the context kept.

        A history that ends on a call keeps it owed a result, and the cache
        holds the render only as far as the results cannot change it.
        ``tools`` other than ``None``, even empty, writes the conversation with
        the model's template for tools. Each turn of ``generated_turns``, as
        :meth:`conversation` gives one, that the history still holds is put
        back as the conversation held it and kept as the tokens it was
        generated as, where they are this model's; a call that could not be
        read that no such turn holds is left out, with its result. A call that
        could not be read and is owed a result next is answered with
        ``unreadable_result``, as every later one is.

        Raises:
            ContextOverflowError: If the system prompt and its declarations leave
                no room under ``evict_above_tokens`` for a short question, or the
                newest exchange beside them leaves no room for a reply, its
                results cut to nothing.
            ValueError: If the history is not in the Chat Completions form this library
                takes, or the template refuses to write the conversation, or
                writes an older exchange otherwise once another follows.
        """
        with self._action(finishes_what_was_left=False):
            # What the cache holds of a conversation this one may go on with.
            cached = ([_without_turn(m) for m in self._state.messages if m["role"] != "system"]
                      if self._state.begun else [])
            before = list(self._state.messages)
            # Written alike, the conversation's own messages may stand for the history's.
            alike = self._state.begun and (self._state.tools, self._state.reasoning,
                                           self._state.uses_tool_template) == (
                list(tools or []), reasoning, tools is not None)
            self._state.tools, self._state.reasoning = list(tools or []), reasoning
            self._state.uses_tool_template = tools is not None
            self._state.unreadable_result = unreadable_result
            self._drop_left = False
            joins = self._joins_questions
            given = read_messages(history)
            if joins:
                # Two questions in a row are written as one, as a running conversation writes them.
                n_read, given = len(given), with_questions_joined(given)
                self._state.changes.question_joined = len(given) < n_read
            messages, _n_found = with_generated_turns(given, self._kept.usable(generated_turns),
                                                      joins)
            held = {i for first, last, _ in generated_turns_in(messages) for i in range(first, last + 1)}
            messages, gone = without_unreadable(messages, held, joins)
            self._state.messages[:] = self._admission.with_open_turns_closed(
                answer_in_order(messages, result_text(unreadable_result)))
            # The messages it opens with that the conversation holds alike are the
            # conversation's own, so that what was placed after them in the cache still stands.
            reused = _reuse_alike(self._state.messages, before) if alike else 0
            prefix = alike and reused == len(self._state.messages)
            # The opening it was begun with, the system turn included, which the checks read.
            opened_alike = alike and len(system_part(before)) == len(self._system_part()) <= reused
            if not alike:
                self._state.anchors = ()        # placed again as the cache is held
            # What the work after the newest reply did is the conversation's before.
            self._evicted_after_reply, self._decoded_after_reply = (0, None), 0
            self._state.left_out_calls = frozenset(gone)
            self._state.changes.n_left_out_calls = len(gone)
            # Counted before the oldest are dropped, and never back, so that no later call
            # takes the id of one dropped, of one left out, or of one this conversation held
            # before.
            self._state.n_calls = max(self._state.n_calls, next_call_number(
                [c["id"] for m in self._state.messages for c in m.get("tool_calls") or ()]
                + sorted(gone)))
            held_turns = {turn for _, _, turn in generated_turns_in(self._state.messages)}
            self._state.turns_at = ({k: at for k, at in self._state.turns_at.items() if k in held_turns}
                                   if alike else {})
            self._state.results_cut = []
            limit, room = self._config.evict_above_tokens, self._max_prompt_tokens
            if not opened_alike:
                self._opening.ensure_room_for_a_question("shorten the prompt or declare fewer tools")
            asks = any(m["role"] == "user" for m in self._state.messages)
            if not prefix:
                # What the conversation held, or the start of it, needs no search: it fit.
                self._keep_newest_history(cached)
            self._kept.drop_unplaceable()
            if not prefix and any(m["role"] != "system" for m in self._state.messages) \
                    and self._rendering.split(self._state.messages, self._ends_on_result()) is None:
                raise self._refused_history(self._state.messages)
            if not opened_alike:
                self._state.checks = self._opening.remembered_render_checks()
            if asks and not any(m["role"] == "user" for m in self._state.messages):
                needed = self._targets.length(self._state.messages, False)
                raise ContextOverflowError(
                    f"the newest exchange of the history, from its user message on, needs "
                    f"{needed} tokens beside the system prompt but {limit} fit; shorten it, "
                    f"{RAISE_THE_ROOM}", needed, limit)
            if asks:
                # The newest exchange may take what a running conversation may, its results
                # cut as they are cut there; a reply or a call that ends it was generated in
                # the room a reply leaves, as the whole context is.
                prompted = self._state.messages[-1]["role"] in ("user", "tool")
                if self._state.messages[-1]["role"] == "assistant":
                    room = self._config.context_size
                # Older exchanges go first, as in a running conversation, then results are cut.
                if self._targets.length(self._state.messages, prompted) > room:
                    self._drop_while_over(room, generation_prompt=prompted)
                if self._targets.length(self._state.messages, prompted) > room:
                    self._cutting.cut_to_fit(room, self._kept.held_indexes())
                needed = self._targets.length(self._state.messages, prompted)
                # A reply stopped by the full context does not fit with its closing; it goes as
                # it goes after the reply.
                if needed > room and not prompted and not self._state.messages[-1].get("tool_calls"):
                    self._drop_while_over(room, generation_prompt=False, newest_too=True)
                    needed = self._targets.length(self._state.messages, prompted)
                if needed > room:
                    raise ContextOverflowError(
                        f"the newest exchange of the history, from its user message on, needs "
                        f"{needed} tokens beside the system prompt but {room} leave room for "
                        f"the reply; shorten it, {RAISE_THE_ROOM}", needed, room)
            self._drop_while_over(limit, generation_prompt=False)
            if not asks:
                self._admission.ensure_within(limit, "shorten the history")
            self._state.changes.n_turns_restored = len(generated_turns_in(self._state.messages))
            # Counted as the caller's messages: those a generated turn holds besides are none of them.
            self._state.changes.n_kept_messages = history_count(
                completions_messages(self._state.messages, joins))
            self._state.called_in_thought = self._kept.open_turn_in_thought()
            self._state.begun = False
            # The records made ahead for the conversation before, freed with it.
            self._state.ahead, self._state.spliced_at = None, None
            self._attempt = self._request = None
            self._holding.pin_opening()
            # Compared with what the cache holds, which is decoded again only from where they
            # differ, a stretch it holds no longer removed.
            if self.pending_calls:
                self._holding.sync(self._settled_while_owed(), logits=False, dropped=True, begins=True)
            elif self._state.messages and self._state.messages[-1]["role"] in ("user", "tool"):
                prompt_target = self._holding.prompted()
                opening = self._holding.prompt_text_start(prompt_target.tokens)
                if opening < len(prompt_target.tokens) and self._memory.keeps_saved_states:
                    # A state saved before the template's text a reply follows, which a cache
                    # that cannot be cut goes back to where what follows is written otherwise.
                    self._holding.sync(Target(prompt_target.tokens[:opening], prompt_target.turns_at),
                               logits=False, dropped=True, begins=True)
                    self._memory.rest()
                self._holding.sync(prompt_target, dropped=True, begins=True)
            else:
                self._holding.render_ahead(dropped=True, begins=True)
            self._memory.rest()
            self._state.begun = True
            self._log_begin(sum(m["role"] != "system" for m in history), generated_turns)

    def _keep_newest_history(self, cached: list[dict]) -> None:
        """Keep the newest of the history being begun that fits under the eviction threshold where the cache holds the oldest of it already, as a conversation this one goes on with, else within half the context; the rest is recorded as evicted."""
        given, limit = list(self._state.messages), self._config.evict_above_tokens
        # Each search starts from a count near its answer: what the cache holds, then what fits
        # under the threshold, which a smaller bound never exceeds.
        start_from = exchange_count(cached) if cached else None
        if cached:
            start_from = self._eviction.keep_newest_history(limit, self._refused_history,
                                                            start_from)
            history = [_without_turn(m) for m in self._state.messages if m["role"] != "system"]
            if _goes_on_from(history, cached):
                self._eviction.record(history_count(given), "room")
                return
            self._state.messages[:] = given
        self._eviction.keep_newest_history(min(limit, self._config.context_size // 2),
                                           self._refused_history, start_from)
        self._eviction.record(history_count(given), "half_context")

    def _log_begin(self, n_given: int, generated_turns: dict | None) -> None:
        """Say once what ``begin`` left out of a history of ``n_given`` messages beyond the oldest, which the eviction says, and of the generated turns it was given; nothing where it kept them all as given. See docs/decisions/0036-one-report-of-what-an-action-changed.md."""
        changes, said = self._state.changes, []
        if changes.n_left_out_calls:
            said.append(f"left out {changes.n_left_out_calls} call(s) that could not be read, "
                        "with their results")
        if changes.generated_turns_ignored:
            said.append("the generated turns given are another model's, so every turn is "
                        "written this model's way")
        elif generated_turns:
            given = sum(len(record["kept"]) for record in generated_turns["turns"])
            if changes.n_turns_restored < given:
                said.append(f"restored {changes.n_turns_restored} of {given} generated turns "
                            "exactly, the others being no longer among the messages as they were")
        if said:
            _log.info("begin kept %d of %d history messages; %s", changes.n_kept_messages,
                      n_given, "; ".join(said))

    def _settled_while_owed(self) -> Target:
        """The render of the conversation as far as the results still owed cannot change it: what the cache holds until they are in."""
        standing = self._targets.target_of(self._state.messages, generation_prompt=False)
        answered = self._targets.target_of(
            self._state.messages + [result_message(call.name, call_id, STAND_IN)
                                    for call, call_id in unanswered(self._state.messages)],
            generation_prompt=False)
        return Target(standing.tokens[:common_prefix_length(standing.tokens, answered.tokens)], {})

    def _refused_history(self, messages: list[dict]) -> TemplateRefusedError:
        """The refusal of ``messages``, a history this model's template will not write, naming the first message it refuses where it stands and the template's own cause."""
        def written(i: int) -> bool:
            # With the message after it too, for a template that judges a message by its next.
            part = messages[:i + 1]
            return (self._writes(part) or self._rendering.render(part, True) is not None
                    or self._writes(messages[:i + 2]))
        # Halved, as the parts a template writes run from the start up to the one it refuses:
        # the first message whose part is refused, after one whose part is written.
        low = 1 if messages and messages[0]["role"] == "system" else 0
        high = len(messages) - 1
        refusing_variant = LoopVariant("finding the refused message of a history")
        while low < high:
            refusing_variant.passes(high - low)
            middle = (low + high) // 2
            if written(middle):
                low = middle + 1
            else:
                high = middle
        at = low
        refused, part = messages[at], messages[:at + 1]
        # Two questions in a row stand here only where the template writes them: a begin joins them.
        refusal = (f"this model's template does not write the {refused['role']} message "
                   f"{_shown(refused)} where it stands")
        return TemplateRefusedError("messages", with_template_reason(
            refusal, self._rendering.refusal_reason(part, part[-1]["role"] == "tool")))

    def _writes(self, messages: list[dict]) -> bool:
        """Whether the template writes ``messages``, as they are or with a question after them."""
        return (self._rendering.render(messages, False) is not None
                or self._rendering.render(messages + [{"role": "user", "content": STAND_IN}],
                                          True) is not None)

    def _newest_exchange(self) -> list[dict]:
        """The system message and the newest exchange alone, which the template's refusals turn on."""
        starts = exchange_starts(self._state.messages)
        return self._system_part() + (self._state.messages[starts[-1]:] if starts else [])

    def _ends_on_result(self) -> bool:
        return bool(self._state.messages) and self._state.messages[-1]["role"] == "tool"

    # ----- adding messages --------------------------------------------------------------
    def ask(self, text: str, cancel: threading.Event | None = None) -> None:
        """Decode ``text`` as the next question, with the generation prompt after it, the oldest exchanges dropped for it where needed.

        Raises:
            ValueError: If no conversation was begun, or the template does not
                write the question here; a turn a call left open is closed first.
            ContextOverflowError: If the room the system prompt leaves is too
                small for any question; one too long for it keeps its end, its
                beginning cut, and a warning is logged.
            Cancelled: If ``cancel`` was set before the question was decoded
                whole, which leaves everything as it was.
        """
        self._memory.cancel = cancel
        try:
            with _refused_as_the_text():
                self._ask(text)
        finally:
            self._memory.cancel = None

    def _ask(self, text: str) -> None:
        """:meth:`ask`'s work."""
        with self._action():
            _ensure_begun(self, "generate into")
            room = self._max_prompt_tokens
            question = self._admission.cut_to_fit({"role": "user", "content": text}, self._question_room,
                                        "the request")
            text = question["content"]
            joined = self._admission.make_way_for(question, room, generation_prompt=True)
            if joined is None:
                self._admission.ensure_writes("user", text, generation_prompt=True)
            if joined is not None or not self._holding.splice(question, room):
                self._state.messages.append(joined or question)
                # As a spliced one, the question may open one exchange more until its reply settles.
                dropped = self._drop_while_over(room, generation_prompt=True, tokens_only=True)
                self._holding.sync(self._fitting_prompt(), dropped=dropped)
            free = self._config.context_size - len(self._memory.tokens)
            if free < self._config.min_reply_tokens:
                raise ContextOverflowError(
                    f"the reply needs {self._config.min_reply_tokens} tokens but {free} are "
                    f"free; shorten the request, or {RAISE_THE_ROOM}",
                    self._config.min_reply_tokens, free)
            if self._memory.cancel is not None:
                # Decoded now, in the parts a cancel is read between, not deferred to the
                # generation; one set by the end takes the whole question back.
                self._memory.repair()
                if self._memory.cancel.is_set():
                    raise Cancelled("cancelled while the question was decoded")

    # ----- what an action changed beyond what was asked ------------------------------------
    def report(self) -> Report:
        """The changes since the caller last asked, the evictions after the newest reply included, and a fresh start for the next."""
        self._take_evicted_after_reply()
        changes, self._state.changes = self._state.changes, Changes()
        return Report(n_evicted_messages=changes.n_evicted,
                      evicted_for=changes.evicted_for if changes.n_evicted else None,
                      n_kept_messages=changes.n_kept_messages,
                      n_left_out_calls=changes.n_left_out_calls,
                      n_turns_restored=changes.n_turns_restored,
                      generated_turns_ignored=changes.generated_turns_ignored,
                      tool_results_cut=tuple(changes.tool_results_cut),
                      turn_closed=changes.turn_closed,
                      question_joined=changes.question_joined,
                      taken_back=changes.taken_back)

    def forget_told(self) -> None:
        """Start the changes afresh, those of earlier actions told already; the evictions after the newest reply are the next report's."""
        self._state.changes = Changes()

    def _take_evicted_after_reply(self) -> None:
        """Count the evictions of the work after the newest reply among the changes."""
        went, cause = self._evicted_after_reply
        if went:
            self._state.changes.n_evicted += went
            self._state.changes.evicted_for = cause
        self._evicted_after_reply = (0, None)

    # ----- the work after a reply ---------------------------------------------------------
    @property
    def settle_due(self) -> bool:
        """Whether a reply left the work after it to do: the evictions and the decode ahead of the next question."""
        return self._settle_due

    def settle(self, interrupt: Interrupt | None = None, question: str | None = None,
               reply_tokens: int = 0) -> None:
        """Do the work a reply left, where it left any: drop what is over the limits, then render and decode ahead for the next question.

        Args:
            interrupt: Set by an action that waits for this one, it stops a
                long decode between its parts; what the work did stands, and
                the cache is left behind its tokens, which the next action
                decodes before its own work.
            question: The text of the question the reply waiting for this
                one answers. Where a drop would decode the history again, no
                tools are declared, no exchange limit is passed, and the
                question and a reply of ``reply_tokens`` fit beside the cache
                as it is, the work is left to the work after that reply, once:
                the question after that waits for it.
            reply_tokens: The cap on that reply's tokens.

        Raises:
            Exception: What a failed decode raises, which leaves the
                conversation as it was and the cache to be decoded again.
        """
        if not self._settle_due:
            return
        self._settle_due = False
        if question is not None and not self._drop_left \
                and self._answers_first(question, reply_tokens):
            # The reply makes the work due again; one taken back leaves it to the next.
            self._drop_left = True
            return
        self._drop_left = False
        changes = replace(self._state.changes,
                          tool_results_cut=list(self._state.changes.tool_results_cut))
        decoded = self.decoded
        self._memory.interrupt = interrupt
        try:
            self._settle()
            if self._config.check_cache:
                try:
                    if interrupt is None or not interrupt.is_set():
                        self._check_cache(prompted=False)
                except InternalError as fault:
                    _log.error("check_cache: %s", fault)
                    self._found_fault = fault       # raised by the next action, on any thread
        except Cancelled:
            _log.debug("the work after a reply gave way to the next action")
            return
        finally:
            self._memory.interrupt = None
            self._decoded_after_reply += self.decoded - decoded
        went = self._state.changes.n_evicted - changes.n_evicted
        if went:
            # Kept apart, so that the next action reports them though it starts afresh.
            self._evicted_after_reply = (self._evicted_after_reply[0] + went,
                                         self._state.changes.evicted_for)
            self._state.changes = changes

    def _answers_first(self, question: str, reply_tokens: int) -> bool:
        """Whether ``question`` is answered before the drop the newest reply left: on a cache that cannot shift, which decodes the history it keeps again, without tools, whose results need room no cap bounds, where the cache holds everything it is to hold and the question and a reply of ``reply_tokens`` fit beside it."""
        if self._context.can_shift or self._state.tools or not self._memory.is_current \
                or self._eviction.too_many() \
                or self._targets.length(self._state.messages, generation_prompt=False) \
                <= self._config.evict_above_tokens:
            return False
        asked = self._state.messages + [{"role": "user", "content": question}]
        return self._targets.length(asked, generation_prompt=True) + reply_tokens \
            <= self._config.context_size

    def changes_so_far(self) -> Changes:
        """What the actions changed since the caller last asked for a :meth:`report`, left to be asked for still."""
        return replace(self._state.changes, tool_results_cut=list(self._state.changes.tool_results_cut))

    # ----- one attempt per generation ---------------------------------------------------
    def open_attempt(self, starts_request: bool, asks: bool) -> None:
        """Note the conversation and the cache as they are, for :meth:`roll_back` until the attempt's reply is recorded.

        Args:
            starts_request: Whether the attempt is a request's first generation.
            asks: Whether that request asks a question of its own, so that a
                reply taken back with it puts back what the question evicted.
        """
        self._attempt = Snapshot(self._state.copied(), list(self._memory.tokens))
        if starts_request:
            self._request = self._attempt if asks else None

    def roll_back(self, brings_up: bool = True) -> None:
        """Put the conversation and the cache back as the open attempt found them.

        The cache is brought back at once where it is known; without
        ``brings_up``, only as far as one part of a long decode, as an
        interrupt leaves it. One a failure left unknown, and the rest of such a
        decode, is brought back by the next action.
        """
        if self._attempt is not None:
            self._go_back_to(self._attempt, brings_up)

    def _go_back_to(self, snapshot: Snapshot, brings_up: bool = True) -> None:
        """Hold the conversation and the cache's tokens of ``snapshot``, ending the attempt."""
        self._state.put_back(snapshot.state)
        self._memory.replace(snapshot.tokens)
        self._attempt = None
        if self._memory.is_known:
            with contextlib.suppress(Exception), self._memory.decoding_one_part(not brings_up):
                self._memory.repair()       # where it fails, the next action decodes again
        self._publish()

    def add(self, role: str, text: str, answers_first: Sequence[tuple[str, object]] = ()) -> None:
        """Decode a message without a reply after it, the oldest exchanges dropped for it where needed.

        A question is put in the stand-in's place as :meth:`ask` puts one, the
        generation prompt after it: within the room left for a reply, the
        conversation over ``evict_above_tokens`` until its reply settles, unless
        ``max_history_exchanges`` is reached first. A question writes
        ``answers_first`` before it, as :meth:`generation` does, within the
        same change.

        Raises:
            ValueError: If no conversation was begun, an assistant message is
                given while a reply is open, or the template does not write it
                after this conversation.
            ContextOverflowError: If it cannot fit beside the system prompt
                alone, or an assistant message beside the system prompt and the
                newest exchange, which no eviction drops; checked before
                anything is dropped.
        """
        with _refused_as_the_text(), self._action():
            if role == "user":
                _ensure_begun(self, "inject into")
                if answers_first:
                    self._write_answers_first(answers_first)
            else:
                _ensure_ready(self, "inject", "inject into")
            message = {"role": role, "content": text}
            limit = self._config.evict_above_tokens
            # A question is answered: it needs the room its reply leaves too.
            asks = role == "user"
            bound = self._question_room if asks else limit
            if asks:
                message = self._admission.cut_to_fit(message, bound, "the injected question")
                text = message["content"]
            alone = self._admission.length_alone(message, bound, generation_prompt=asks)
            if alone > bound:
                raise ContextOverflowError(
                    f"the injected message needs {alone} tokens but {bound} fit beside the "
                    f"system prompt; shorten the message or the system prompt, or {RAISE_THE_ROOM}",
                    alone, bound)
            if not asks:
                # Refused as the text first, so that its cause does not turn on how full it is.
                self._admission.ensure_writes(role, text, generation_prompt=False)
                self._admission.ensure_fits_once_dropped(message, bound)
            joined = self._admission.make_way_for(message, bound, generation_prompt=True) if asks else None
            if asks and joined is None:
                self._admission.ensure_writes(role, text, generation_prompt=False)
            room = self._max_prompt_tokens
            fits = (self._config.max_history_exchanges is None
                    or exchange_count(self._state.messages) < self._config.max_history_exchanges)
            if role != "user" or joined is not None or not fits or not self._holding.splice(message, room):
                self._state.messages.append(joined or message)
                # A question keeps the room its reply needs, as a begin of it does.
                dropped = self._drop_while_over(bound, generation_prompt=asks)
                self._admission.ensure_within(bound, INJECT_A_QUESTION_FIRST,
                                    generation_prompt=asks)
                self._holding.sync(self._targets.target_of(self._state.messages, generation_prompt=False),
                           logits=False, dropped=dropped)
            self._memory.rest()

    def prompt(self) -> None:
        """Decode the generation prompt after the newest message, a user message, for a reply to it, the oldest exchanges dropped to leave room for the reply where needed.

        Raises:
            OutOfOrderError: If no conversation was begun, or the newest
                message is not a user message; it is not called while a reply
                is open.
            ContextOverflowError: If the conversation leaves too little room for
                a reply once every exchange that can go is dropped and the
                question is cut to fit beside the system prompt.
        """
        with self._action():
            _ensure_can_reply(self, asks_question=False)
            spliced_at = self._state.spliced_at
            if spliced_at is not None and self._holding.is_current(spliced_at[2]):
                return          # spliced with the generation prompt after it, within the room
            dropped = self._drop_while_over(self._max_prompt_tokens, generation_prompt=True,
                                            tokens_only=True)
            self._holding.sync(self._fitting_prompt(), dropped=dropped)

    def _fitting_prompt(self) -> Target:
        """The conversation with the generation prompt after it, a newest question cut again where it no longer fits.

        Raises:
            ContextOverflowError: If it leaves too little room for a reply.
        """
        target = self._holding.prompted()
        room = self._max_prompt_tokens
        if len(target.tokens) > room and self._state.messages[-1]["role"] == "user" \
                and self._admission.cut_newest_question_again():
            target = self._holding.prompted()
        if len(target.tokens) > room:
            raise ContextOverflowError(
                f"the conversation needs {len(target.tokens)} tokens but {room} leave room "
                f"for the reply; {RAISE_THE_ROOM}", len(target.tokens), room)
        if self._memory.keeps_saved_states:
            # One state, where the reply's turn begins: a reply written again or shortened decodes
            # the few tokens of the generation prompt again rather than save the cache twice.
            end = len(target.tokens)
            start = self._state.prompt_start
            self._memory.pin([start if 0 < start <= end else end] if self._holding.reply_kept_from_its_start
                             else [], THE_REPLY)
        return target

    # ----- generating -------------------------------------------------------------------
    def generate(self, max_generated_tokens: int, generated: GenerationAccumulator, **options):
        """Stream one generation of at most ``max_generated_tokens`` from the cache's end; what it decodes is held as the cache's.

        It stops short of ``max_generated_tokens`` where the context leaves less room
        beside the closing after the reply.

        Raises:
            ValueError: If no conversation was begun, the open turn still owes
                a result, or no turn is open and the newest message is not a
                user message.
        """
        _ensure_can_reply(self, asks_question=False)
        self._state.results_cut = []
        self._generation = (min(self._state.prompt_start, len(self._memory.tokens)),
                            self.turn_open)
        if self._holding.reply_kept_from_its_start and self._context.can_shift:
            # The prompt the template wrote before the reply, which holds no message's text,
            # where the template may leave part of it out of the past reply.
            self._memory.note_template_written(self._generation[0], len(self._memory.tokens))
        return self._memory.generate(self._generation_room(max_generated_tokens), generated,
                                     **options)

    def _generation_room(self, max_generated_tokens: int) -> int:
        """The most tokens a generation from the cache's end may take: at most ``max_generated_tokens``, and the closing after the reply left room."""
        room = (self._config.context_size - len(self._memory.tokens)
                - self.template.properties.n_closing_tokens)
        return max(0, min(max_generated_tokens, room))

    def _shortest_call_tokens(self) -> int:
        """Tokens the shortest call to a declared tool takes at least: the marks around it, the shortest name and one token of arguments."""
        return self.template.n_call_mark_tokens + 1 + min(
            len(self._context.tokenize(spec.name, add_special=False, parse_special=False))
            for spec in self._state.tools)

    def generation(self, text: str | None, *, recognizes_calls: bool, withhold: bool,
                   max_generated_tokens: int, stop: tuple[str, ...],
                   cancel: threading.Event | None, sampling, unreadable_result: dict,
                   on_started: Callable[[], None] | None,
                   on_reply: Callable[[Reply], None],
                   answers_first: Sequence[tuple[str, object]] = (),
                   max_thought_tokens: int | None = None) -> Generator[str, None, Reply]:
        """Generate one reply from the cache's end, stream what it shows, and record what it left, as a Reply.

        Without ``text``, it goes on with the open turn, adding no message of
        its own, or else answers the newest message, which must be a user
        message already written. A call it stops on leaves the turn open for a
        result.

        Every exit before the generation's first token, by a cancel or a
        failure, leaves the conversation and the cache as this call found
        them, the history its question evicted included; so does a failure
        while the reply is recorded. See docs/decisions/0033-one-undo-path.md.

        Args:
            text, stop, cancel: As :meth:`~llama_cpp_conversation.wrapper.ChatWrapper.reply`.
            max_generated_tokens: The cap on this generation's tokens.
            max_thought_tokens: The cap on a thought's tokens, ``None`` for none.
            sampling: Every value the generation draws with.
            recognizes_calls: ``False`` leaves a call the model writes as text,
                so this generation cannot end on a call.
            withhold: Also keep a call's opening token from ever being drawn.
            unreadable_result: What a call this generation writes that cannot
                be read is answered with.
            on_started: Called once the request is written and the generation
                starts, before anything it produces is recorded; given to a
                request's first generation alone.
            on_reply: Given the Reply once the generation ends, early or not.
            answers_first: ``(call_id, answer)`` pairs written first, each while
                its call is the oldest still owed, taken back with the rest
                where the request does not start.
        """
        decoded_from = self.decoded
        evicted_before = self._state.changes.n_evicted
        # Only a template that calls tools pays for telling the calls of this generation.
        called_before = (_call_ids_since_question(self._state.messages)
                         if self.template.properties.can_write_results else None)
        self.open_attempt(starts_request=on_started is not None, asks=text is not None)
        try:
            if cancel is not None and cancel.is_set():
                raise Cancelled("cancelled before the request was written")
            if answers_first:
                # Within the attempt, so that a request refused or cancelled takes them back too.
                self._write_answers_first(answers_first)
            if text is not None:
                self.ask(text, cancel)
            elif not self.turn_open:
                self.prompt()
            if self._config.check_cache:
                self._check_cache(prompted=True)
        except Cancelled:
            self.roll_back(brings_up=False)     # the next action decodes what it needs
            return Reply(text="", n_prefilled_tokens=self.decoded - decoded_from,
                         n_prompt_tokens=self.total_tokens, n_generated_tokens=0,
                         n_evicted_messages=0, stop_reason="interrupted")
        except BaseException:
            self.roll_back()
            raise
        n_prompt_tokens = self.total_tokens
        # The request is written: what the work after the newest reply did was done for it.
        decoded_from, self._decoded_after_reply = decoded_from - self._decoded_after_reply, 0
        self._take_evicted_after_reply()
        generated = GenerationAccumulator()
        results_cut = self.results_cut
        template = self.template
        if recognizes_calls and self._state.tools and template.can_withhold_calls \
                and self._generation_room(max_generated_tokens) < self._shortest_call_tokens():
            # No call could close in the room left: the model answers in text instead.
            recognizes_calls, withhold = False, True
        try:
            goes_on = text is None and self.turn_open
            reply_text = ReplyText(template.properties.reasoning,
                                   template.answer_opening(self.reasoning),
                                   template.reply_header(self.reasoning),
                                   self.prompt_text, list(stop), generated.tag_spans,
                                   inside=goes_on and self.called_in_thought,
                                   stops_after_text=self._config.min_answer_tokens is not None)
            pieces = self.generate(
                max_generated_tokens, generated, cancel=cancel, sampling=sampling,
                ends_turn=self._turn_end_test(reply_text, generated),
                boundary=replace(template.call_boundary(recognizes_calls, withhold),
                                 starts_in_thought=reply_text.opens_in_thought),
                guard=replace(self._answer_guard(template), max_thought_tokens=max_thought_tokens),
                answer_so_far=lambda: reply_text.answer)
            if on_started is not None:
                on_started()
        except BaseException:
            self.roll_back()
            raise
        failed = False
        try:
            for piece in pieces:
                shown = reply_text.feed(piece)
                if shown:
                    yield shown
                if reply_text.met_stop_string:
                    pieces.close()
                    generated.stop_reason = "stop_string"
                    break
            else:
                if shown := reply_text.finish(generated.stop_reason):
                    yield shown
        except BaseException:
            failed = True
            raise
        finally:
            pieces.close()
            read: list[ToolCall] = []
            if not generated.token_ids and (failed or generated.stop_reason == "interrupted"):
                # Nothing was produced, so nothing changes; a cancelled request leaves the
                # decode back to the next action.
                self.roll_back(brings_up=failed or generated.stop_reason != "interrupted")
            else:
                try:
                    # Runs on normal completion and on early close alike.
                    read = self._record_generated(generated, reply_text, unreadable_result)
                except BaseException:
                    self.roll_back()
                    raise
            changes = self._state.changes
            # A generation put back puts back what it evicted too.
            n_evicted_messages = changes.n_evicted - evicted_before
            reply = Reply(
                evicted_for=changes.evicted_for if n_evicted_messages else None,
                taken_back=changes.taken_back, kept_text=changes.kept_text,
                text=reply_text.answer,
                n_prefilled_tokens=self.decoded - decoded_from,
                n_prompt_tokens=n_prompt_tokens,
                n_generated_tokens=len(generated.token_ids),
                n_evicted_messages=n_evicted_messages,
                stop_reason=generated.stop_reason,
                # A reply taken back made no call; what is owed then is the turn it went back to.
                tool_calls=_reported_calls(generated, [] if changes.taken_back
                                           else self.pending_calls if called_before is None
                                           else self._calls_to_report(called_before)),
                tool_results_cut=tuple(results_cut),
                n_reasoning_tokens=reply_text.thought_pieces,
                thought=reply_text.thought,
            )
            if read and not reply.tool_calls and generated.stop_reason == "tool_call" \
                    and not changes.taken_back:
                # The turn closed on what the model wrote, for want of room to answer its calls.
                reply = replace(reply, stop_reason="token_limit", tool_calls=[
                    call if call.status == "unreadable" else replace(
                        call, status="unfinished") for call in read])
            on_reply(reply)
        return reply

    def _answer_guard(self, template: ResolvedTemplate) -> AnswerGuard:
        """What keeps a generation on ``template`` from ending without an answer, worked out once per template."""
        known = self._guards.get(id(template))
        if known is not None:
            return known[1]
        context = self._context
        ends = [token for token in context.special_tokens() if context.ends_generation(token)]
        turn_end = template.properties.turn_end_token
        if turn_end is not None and turn_end not in ends:
            ends.append(turn_end)
        reasoning = template.properties.reasoning
        guard = AnswerGuard(
            end_tokens=tuple(ends), min_answer_tokens=self._config.min_answer_tokens,
            thought_tags=(reasoning.thought_open.strip(), reasoning.thought_close.strip())
            if reasoning is not None else (),
            thought_closing=tuple(context.tokenize(reasoning.thought_close, add_special=False,
                                                   parse_special=True))
            if reasoning is not None else (),
            answer_room_tokens=self._config.answer_room_tokens)
        self._guards[id(template)] = (template, guard)
        return guard

    def _turn_end_test(self, reply_text: ReplyText,
                       generated: GenerationAccumulator) -> Callable[[int], bool] | None:
        """Whether a sampled token ends the reply ``generated`` collects, besides the end-of-generation tokens."""
        turn_end = self.template.properties.turn_end_token
        if turn_end is None:
            return None
        return reply_text.turn_end_test(turn_end, self._context.token_text(turn_end),
                                        generated.token_ids, self._context.token_bytes,
                                        self._context.is_special)

    def _record_generated(self, generated: GenerationAccumulator, reply_text: ReplyText,
                          unreadable_result: dict) -> list[ToolCall]:
        """Record what one generation left and settle the cache after it; the calls it read.

        A call that could not be read ends the generation as
        ``"unreadable_tool_call"``, and is answered with ``unreadable_result``.
        """
        calls = self._read_calls(generated)
        called = generated.stop_reason == "tool_call"
        if called and not calls:
            generated.stop_reason = "unreadable_tool_call"
        tags = self.template.properties.reasoning
        # Given to a template that writes a past reply's thought; another is written without it.
        kept_thought = reply_text.thought if tags and tags.past_keeps_thought else ""
        self.record_reply(reply_text.answer, kept_thought, calls, called, unreadable_result,
                          stopped=reply_text.met_stop_string, in_thought=reply_text.in_thought)
        return calls

    def _read_calls(self, generated: GenerationAccumulator) -> list[ToolCall]:
        """The calls ``generated`` stopped on, empty where it produced no readable call.

        A call the parser cannot read counts as no call; any other fault is raised.
        """
        if generated.stop_reason != "tool_call":
            return []
        try:
            calls = self.template.read_calls(
                generated.tool_call, {spec.name: spec.parameters for spec in self._state.tools})
        except ValueError:
            return []
        return self._with_usable_ids(calls) if any(call.id for call in calls) else calls

    def _with_usable_ids(self, calls: list[ToolCall]) -> list[ToolCall]:
        """``calls``, each keeping the id the model wrote for it where that id is new to the conversation and the template writes the call under it; any other is given an id of its own."""
        taken = {c["id"] for m in self._state.messages for c in m.get("tool_calls") or ()}
        taken |= self._state.left_out_calls
        usable = []
        for call in calls:
            if call.id and (call.id in taken or self._rendering.render(
                    self._newest_exchange() + [calling_message("", [call])], False) is None):
                _log.info("the id %r the model wrote for a call to %s is not used, as it is "
                          "%s; the call is given one of its own", call.id, call.name,
                          "taken" if call.id in taken else "one the template does not write")
                call = replace(call, id="")
            taken.add(call.id)
            usable.append(call)
        return usable

    def record_reply(self, answer: str, thought: str, calls: list[ToolCall],
                     called: bool, unreadable_result: dict, stopped: bool = False,
                     in_thought: bool = False) -> None:
        """Record what the newest generation produced, then settle the cache after it.

        A turn that called tools keeps the tokens it was generated as, and stays
        open while a call is owed a result; any other reply closes the turn. A
        reply the template refuses to write, or that produced nothing it writes, is
        taken back with its request. Where the generation failed, the cache is left
        to be decoded again by the next action.

        Recording the reply ends the attempt :meth:`open_attempt` opened; where
        that raises, the caller puts the attempt back, and a fault after it keeps
        the reply.

        Args:
            thought: The reply's thought, given to the template as its own to
                write or leave out.
            calls: What it called, each owed a result.
            called: Whether it stopped on a call, read or not.
            unreadable_result: What a call that could not be read is answered with.
            stopped: Whether a stop string ended the answer, so that what was
                generated from it on is no reply's.
            in_thought: Whether the generation ended inside a thought, so that
                the turn its call leaves open goes on inside it.
        """
        if not self._memory.is_known and not answer and not called:
            # It failed before it said anything the template writes: nothing changes.
            self.roll_back()
            return
        with self._publishing():
            recorded, unplaced = self._append_reply(answer, thought, calls, called, stopped)
            self._state.called_in_thought = called and in_thought
            if recorded != answer:
                self._state.changes.kept_text = recorded
                _log.info("a reply's answer holds the template's thought tags; the part the "
                          "template writes is kept: %r", recorded)
            try:
                self._after_reply(recorded, called, called and not calls, unplaced,
                                  recorded != answer, unreadable_result)
            finally:
                self._attempt = None        # recorded: a fault after it keeps the reply

    def _append_reply(self, answer: str, thought: str, calls: list[ToolCall], called: bool,
                      stopped: bool) -> tuple[str, bool]:
        """Add the newest generation's reply to the messages, its turn kept where it called tools.

        Returns:
            The answer as recorded, and whether a generated turn could no longer be
            placed.
        """
        start, resumed = self._generation
        unreadable = called and not calls
        if answer and not calls and self._holds_thought_tags(answer):
            # Where the template writes none of it as text, it is recorded as an answer with no text.
            tags = self.template.properties.reasoning
            assert tags is not None, "thought tags are held only where the template reasons"
            answer = self._rendering.written_answer(
                self._state.messages, answer, thought,
                (tags.thought_close.strip(), tags.thought_open.strip())) or ""
        asked = calls or ([ToolCall("", {}, status="unreadable")]
                          if unreadable else [])
        message: dict[str, Any] = ({"role": "assistant", "content": answer} if not asked
                                   else calling_message(answer, asked, self._state.n_calls))
        if thought:
            message["reasoning_content"] = thought
        if asked:
            # Past each id the model wrote too, so that no id given later repeats one.
            self._state.n_calls = max(self._state.n_calls + len(asked), next_call_number(
                [c["id"] for c in message["tool_calls"]]))
        # A reply with neither text nor call has nothing to place its end by, and one a stop
        # string ended holds what was generated after it: a turn kept so far stays kept up to
        # its call, and the rest is the template's.
        if (called or resumed) and (answer or asked) and not stopped:
            messages, span = self._state.messages, 1
            open_turn = self._kept.open_generated_turn() if resumed else None
            if open_turn is not None:
                # The whole turn is kept, its results and their prompt included.
                at, turn = open_turn
                messages[at] = with_generated_turn(messages[at], None)
                start = min(start, self._state.turns_at.pop(turn))
                span = len(messages) - (at - turn.span + 1) + 1
            message[GENERATED_TURN] = GeneratedTurn(tuple(self._memory.tokens[start:]), span, bool(asked))
            self._state.turns_at[message[GENERATED_TURN]] = start
        self._state.messages.append(message)
        # A reply that leaves every turn that called tools as it was cannot move one.
        moves_kept = called or resumed or not self._state.checks.tool_turns_stay
        return answer, moves_kept and self._kept.drop_unplaceable()

    def _after_reply(self, answer: str, called: bool, unreadable: bool, unplaced: bool,
                     rewritten: bool, unreadable_result: dict) -> None:
        """Answer a call that could not be read, or close the turn and settle the cache.

        Args:
            rewritten: Whether ``answer`` was recorded otherwise than generated.
        """
        _start, resumed = self._generation
        if not self._memory.is_known:
            # The generation failed: the next action decodes the cache again. One that left
            # no answer it writes changes nothing, as one that said nothing; one recorded
            # otherwise than generated is decoded again as it was recorded.
            if not answer and not called:
                self.roll_back()
            elif rewritten:
                self._memory.replace(
                    self._rendering.target(self._state.messages, generation_prompt=False).tokens)
            return
        self._state.unreadable_result = unreadable_result
        if unreadable:
            try:
                written = self._write_unreadable(unreadable_result)
            except BaseException:
                self._kept.forget_calls()        # the cache is decoded again by the next action
                # Without its call, the reply may be one the template does not write.
                self._take_back_unwritten(answer)
                raise
            if not written:
                self._kept.forget_calls()
        elif self.turn_open and self._answer_unreadable_owed():
            return                          # an entry of a batch that could not be read
        if self.turn_open:
            if unplaced:
                self._holding.sync(self._targets.target_of(self._state.messages, generation_prompt=False),
                           logits=False)
            return
        # A reply whose call was withdrawn, for want of room for its answer, is a reply too.
        replied = not self._state.messages[-1].get("tool_calls")
        if replied and self._take_back_unwritten(answer):
            return
        if not answer and not called and not resumed \
                and self.template.properties.writes_two_user_turns:
            # A reply all thought or cut before its answer leaves the question
            # unanswered, as the next question then follows it.
            self._remove_from(len(self._state.messages) - 1)
            return
        self._settle_due = True             # done by :meth:`settle`, once the reply has ended

    def _holds_thought_tags(self, text: str) -> bool:
        """Whether ``text`` holds the template's thought opening or closing, which a template may cut a past reply at."""
        tags = self.template.properties.reasoning
        return tags is not None and any(tag.strip() and tag.strip() in text
                                        for tag in (tags.thought_open, tags.thought_close))

    def _write_unreadable(self, result: dict) -> bool:
        """Answer a call that could not be read with ``result``, where the template writes a result and the context has room; whether it was."""
        call = self.pending_calls[0]
        if not self._renders_result(result_message(call.name or UNKNOWN_FUNCTION, call.id,
                                                   result_text(result))):
            return False
        try:
            self._add_result(call.id, result)
            return True
        except ContextOverflowError:
            return False

    def _write_answers_first(self, answers: Sequence[tuple[str, object]]) -> None:
        """Write each ``(call_id, answer)`` while its call is the oldest still owed; where no room is left for one, the turn closes on what the model wrote."""
        for call_id, answer in answers:
            owed = self.pending_calls
            if not owed or owed[0].id != call_id:
                return
            try:
                self.add_result(call_id, answer)
            except ContextOverflowError:
                self.close_unanswered()
                return

    def _calls_to_report(self, called_before: set[str]) -> list[ToolCall]:
        """The calls this generation made, in the order the model wrote them: those still owed, and the unreadable ones answered at once; ``called_before`` holds the ids of the calls made before it."""
        owed = {call.id: call for call in self.pending_calls}
        made = []
        for message in _since_question(self._state.messages):
            for call in message.get("tool_calls") or ():
                if call["id"] in owed:
                    made.append(owed.pop(call["id"]))
                elif UNREADABLE_CALL in call and call["id"] not in called_before:
                    made.append(ToolCall("", {}, call["id"], "unreadable"))
        return made + list(owed.values())

    def _answer_unreadable_owed(self) -> bool:
        """Answer each call still owed that could not be read, while one is next in order, with the failure the model is told; whether the turn closed instead, for want of room for it."""
        answering_variant = LoopVariant("answering unreadable calls")
        while self.pending_calls and self.pending_calls[0].status == "unreadable" \
                and self._state.unreadable_result is not None:
            answering_variant.passes(len(self.pending_calls))
            if not self._write_unreadable(self._state.unreadable_result):
                # Left owed, it would hold the turn open with no call the caller can answer.
                self.close_unanswered()
                return True
        return False

    def _take_back_unwritten(self, answer: str) -> bool:
        """Take the newest reply back with its request where the template writes neither it nor an empty one in its place; whether it did."""
        if self._writes(self._targets.newest_part(self._state.messages) if not answer
                        else self._newest_exchange()):
            return False
        _log.info("a reply the template does not write, nor an empty one in its place, "
                  "was taken back with its request")
        if self._request is not None:
            # Taken back with the question this request asked, and what it evicted.
            self._go_back_to(self._request)
            self._state.changes.taken_back = True
            self._publish()
            return True
        self._state.changes.taken_back = True
        starts = exchange_starts(self._state.messages)
        self._remove_from(starts[-1] if starts else len(self._system_part()))
        return True

    def _renders_result(self, message: dict) -> bool:
        """Whether the template writes ``message``, a result, after the conversation."""
        # A history with no question has no exchange to stand for it: it is read whole.
        newest = (self._newest_exchange() if exchange_starts(self._state.messages)
                  else self._state.messages)
        return self._rendering.render(newest + [message], False) not in (
            None, self._rendering.render(newest, False))

    def close_unanswered(self) -> None:
        """Close the open turn where no room is left to answer its newest calls, as the reply cap closes one.

        The newest reply is kept without those calls, its tokens as generated,
        and the results already written for them are taken out again; where the
        template writes nothing of what is left of it, it is taken back with its
        request, as an empty reply is.
        """
        with self._action():
            calling = max(i for i, m in enumerate(self._state.messages) if m.get("tool_calls"))
            del self._state.messages[calling + 1:]
            self._state.results_cut = []
            self._kept.forget_calls()
            self._state.changes.turn_closed = True
            if not self._state.messages[-1]["content"] and not self._writes(
                    self._targets.newest_part(self._state.messages)):
                # Nothing the template writes is left of the reply: taken back with its request.
                self._state.changes.taken_back = True
                _log.info("no room was left to answer the newest calls, and nothing of the "
                          "reply is left to write: taken back with its request")
                starts = exchange_starts(self._state.messages)
                self._remove_from(starts[-1] if starts else len(self._system_part()))
                return
            _log.info("no room was left to answer the newest calls: the turn closed on what "
                      "the model wrote")
            self._settle()

    def add_results(self, results: Sequence[tuple[str, object]]) -> None:
        """Record each ``(call_id, result)`` in order, as :meth:`add_result` does, all of them or, where one is refused or fails, none.

        Raises:
            ValueError: As :meth:`add_result`, for the first that is refused.
            ContextOverflowError: As :meth:`add_result`.
        """
        with self._action():
            for call_id, result in results:
                self.add_result(call_id, result)

    def add_result(self, call_id: str, result) -> None:
        """Record what the oldest owed call returned, then, after the last one, decode the prompt for the reply.

        A result too long for the room left drops the oldest exchanges first,
        as a question does, the turn it answers kept whole. Where it still
        cannot fit, it is cut, :data:`RESULT_CUT_NOTE` after it telling the
        model how much was kept, and its call is noted in :attr:`results_cut`.
        It is written whole where each result still owed after it still fits
        empty; otherwise the room left keeps space for each of them as that
        note with nothing kept, unless not even this one would fit then.

        Raises:
            ValueError: If no call is waiting, this is not the one owed, the
                result cannot be written as text, or the template writes no
                result.
            ContextOverflowError: If not even a result cut to nothing leaves
                room for the reply.
        """
        self._add_result(call_id, result)
        self._answer_unreadable_owed()

    def _add_result(self, call_id: str, result) -> None:
        """:meth:`add_result`'s work for one call."""
        with self._action():
            if not self._state.begun:
                raise OutOfOrderError(None, "no conversation to answer a call in; call begin() first")
            if call_id in self._state.left_out_calls:
                raise OutOfOrderError(
                    "call_id", f"the call {call_id!r} could not be read, as it named no function "
                    "or its arguments were no JSON object, so begin left it out; no result is "
                    "owed for it")
            waiting = unanswered(self._state.messages)
            if not waiting:
                raise OutOfOrderError("call_id", "no tool call is waiting for a result")
            owed, owed_id = waiting[0]
            if call_id != owed_id:
                later = call_id in {later_id for _, later_id in waiting[1:]}
                raise OutOfOrderError(
                    "call_id", f"the waiting call is {owed_id!r}, not {call_id!r}; results are "
                    "owed in the order the model asked for them" if later else
                    f"no call {call_id!r} is waiting for a result; the waiting one is "
                    f"{owed_id!r}")
            message = result_message(owed.name or UNKNOWN_FUNCTION, owed_id, result_text(result))
            if not self._renders_result(message):
                raise UnsupportedError("result", "this model's template renders no tool result, "
                                                 "so a call's result cannot be written")
            dropped = self._cutting.write(
                message, waiting[1:], self._max_prompt_tokens,
                lambda: self._drop_while_over(self._max_prompt_tokens, generation_prompt=True,
                                              tokens_only=True, beside_open_turn=True))
            if not self.pending_calls:
                self._holding.sync(self._fitting_prompt(), dropped=dropped)
            elif dropped:
                self._holding.sync(self._targets.target_of(self._state.messages, generation_prompt=False),
                           logits=False, dropped=True)
            if dropped:
                self._memory.repair()           # with the logits the reply needs, then saved
                self._memory.rest()

    # ----- changing what it holds -----------------------------------------------------
    def truncate_newest(self, keep_text: str) -> None:
        """Shorten the newest reply's text to ``keep_text``.

        A reply that ends a turn that called tools keeps the tokens it was
        generated as up to ``keep_text``.

        Raises:
            ValueError: If no conversation was begun, a reply is open, which
                a newest reply that holds a call leaves, the newest message is
                not a reply, ``keep_text`` is not a prefix of its text, or the
                template does not write the reply shortened to it, or the
                tokens of a reply kept as generated hold more than its text.
        """
        with self._action():
            _ensure_ready(self, "truncate a reply", "truncate in")
            newest = self._state.messages[-1] if self._state.messages else None
            if newest is None or newest["role"] != "assistant":
                raise OutOfOrderError(None, "the newest message is not an assistant reply")
            if not newest["content"].startswith(keep_text):
                raise InvalidArgumentError("keep_text",
                                           "keep_text is not a prefix of the newest reply")
            if keep_text == newest["content"]:
                return
            shortened = {**newest, "content": keep_text}
            turn = newest.get(GENERATED_TURN)
            messages = self._state.messages
            if turn is not None and not keep_text:
                # A reply without text has nothing to place its end by: the turn stays kept up
                # to its newest call, as one whose reply said nothing is, the rest the template's.
                del shortened[GENERATED_TURN]
                first = len(messages) - turn.span
                calling = max((i for i in range(first, len(messages))
                               if messages[i].get("tool_calls")), default=None)
                # Without a call left in it, nothing of the turn is kept as generated.
                if calling is not None:
                    cut = self._kept.generated_to_its_call(turn, newest["content"],
                                                      calling - first + 1)
                    if cut is None:
                        raise TemplateRefusedError(
                            "keep_text", "the newest reply's tokens as generated cannot be cut "
                            "after its turn's newest call; keep some of its text")
                    messages[calling] = with_generated_turn(messages[calling], cut)
            elif turn is not None:
                cut = self._kept.generated_up_to(turn, newest["content"], keep_text)
                if cut is None:
                    raise TemplateRefusedError(
                        "keep_text", "the newest reply's tokens as generated hold more than "
                        "its text, such as a call that went, so they cannot be cut at keep_text")
                shortened[GENERATED_TURN] = cut
            messages[-1] = shortened
            if not self._writes(self._targets.newest_part(self._state.messages)):
                raise TemplateRefusedError(
                    "keep_text", f"this model's template does not write the newest reply "
                    f"shortened to {keep_text!r}; keep more of its text")
            self._holding.render_ahead()
            self._memory.rest()

    def _remove_from(self, first: int) -> None:
        """Take the messages from ``first`` on out again, then drop what is over the limits."""
        with self._action():
            del self._state.messages[first:]
            dropped = self._drop_while_over(self._config.evict_above_tokens,
                                            generation_prompt=False)
            # As after any reply, so that what the template writes ahead of the next
            # question, a default system prompt say, stays in the cache.
            self._holding.render_ahead(dropped=dropped)
            self._memory.rest()

    # ----- after a reply ----------------------------------------------------------
    def _settle(self) -> None:
        """Drop what is over the limits, then render and decode ahead for the next question."""
        with self._action():
            limit = self._config.evict_above_tokens
            plan = self._windows.ahead(self._memory) if self._state.checks.windows else WHOLE
            if plan is not WHOLE:
                # Dropped first, so the cache shifts before the decode ahead, not before the next question's.
                at, before, length, split, run, window_turns_at = cast(tuple, plan)
                held = len(self._memory.tokens)
                kept_until = lambda: at - (held - len(self._memory.tokens))   # noqa: E731
                done = self._windows.drop_in(self._memory, kept_until, length)
                # Where the window's own older exchange must go too, or what is decoded ahead
                # would not fit, the rest is rendered whole.
                done = done and not (kept_until() + length > limit or self._eviction.too_many()
                                     or kept_until() + len(before) > self._config.context_size)
                if done:
                    # The tag the next question's exchange opens with, after the newest reply.
                    tag = self._rendering.tag_after_content(split, len(split.contents) - 2)
                    # The tags that open the newest exchange, where the window began, and the
                    # next question's, after the newest reply.
                    newest = exchange_starts(self._state.messages)[-1]
                    anchors = ((newest - 1, kept_until()),) + (() if tag is None or tag > run else ((
                        len(self._state.messages) - 1, kept_until() + len(before)
                        - len(self._rendering.tokenize(split.pieces(tag, run)))),))
                    base = kept_until()
                    turns_at = {**{k: p for k, p in self._state.turns_at.items() if p < base},
                               **{k: base + p for k, p in window_turns_at.items()}}
                    self._holding.sync(Target(list(self._memory.tokens[:base]) + before, turns_at, anchors),
                               logits=False)
                    self._holding.note_ahead(split, run)
                else:
                    self._drop_over_after_a_reply(limit)
                    self._holding.render_ahead(dropped=True)
            else:
                self._holding.render_ahead(dropped=self._drop_over_after_a_reply(limit))
            self._memory.rest()

    def _drop_over_after_a_reply(self, limit: int) -> bool:
        """Drop the oldest exchanges while the conversation is over ``limit`` or holds too many, the newest kept until the next question comes, which a reader may still cut back, unless it does not fit the context; whether any went."""
        dropped = self._drop_while_over(limit, generation_prompt=False)
        if self._targets.length(self._state.messages, generation_prompt=False) > self._config.context_size:
            dropped = self._drop_while_over(self._config.context_size, generation_prompt=False,
                                            newest_too=True) or dropped
        return dropped

    def _system_part(self) -> list[dict]:
        return system_part(self._state.messages)

    def _text_within(self, text: str, n_tokens: int) -> int:
        """How many characters of ``text`` the first ``n_tokens`` of its own tokens spell, as a caller's text."""
        tokens = self._rendering.tokenize([(text, True)])
        if n_tokens >= len(tokens):
            return len(text)
        return len(b"".join(self._context.token_bytes(token)
                            for token in tokens[:n_tokens]).decode(errors="ignore"))

    # ----- limits -------------------------------------------------------------------
    @property
    def _max_prompt_tokens(self) -> int:
        """The most tokens the cache may hold when a reply starts: ``min_reply_tokens`` left free, and never less than the shortest reply that holds an answer."""
        return self._config.context_size - max(self._config.min_reply_tokens,
                                               self.template.n_shortest_answer_tokens)

    @property
    def _question_room(self) -> int:
        """The most tokens the system turn and a question alone may take: under the eviction threshold, with the reply's room left."""
        return min(self._config.evict_above_tokens, self._max_prompt_tokens)

    def _drop_while_over(self, limit: int, generation_prompt: bool, **options) -> bool:
        """Drop the oldest exchanges once the conversation is over ``limit`` or holds too many, as :meth:`Eviction.drop_while_over` does, in windows of the oldest exchanges first where they stand for the render; whether any went."""
        went = False
        if not options.get("newest_too"):
            went = self._drop_in_windows(limit, generation_prompt,
                                         not options.get("tokens_only"),
                                         options.get("beside_open_turn", False))
        return self._eviction.drop_while_over(limit, generation_prompt, **options) is not None \
            or went

    def _drop_in_windows(self, limit: int, generation_prompt: bool, counts_exchanges: bool,
                         beside_open_turn: bool) -> bool:
        """Drop the oldest exchanges as :meth:`Windows.drop_in` does, each checked against a render of the oldest two and removed from the cache, where a window of the newest stands for the rest; whether any went."""
        # A cache that cannot shift decodes the rest again after a drop, which costs more
        # than the render; it drops as far as the whole render says, once.
        if not self._eviction.keeps_the_rest:
            return False
        view = self._targets.view(self._state.messages, generation_prompt)
        if view is None or not view.base:
            return False
        anchor_at, held = len(view.base), len(self._memory.tokens)
        after = len(self._targets.view_target(view).tokens) - anchor_at
        n_before = len(self._state.messages)
        self._windows.drop_in(self._memory, lambda: anchor_at - (held - len(self._memory.tokens)),
                              after, limit, counts_exchanges, beside_open_turn)
        return len(self._state.messages) < n_before

    # ----- one action at a time --------------------------------------------------------
    @contextlib.contextmanager
    def _action(self, finishes_what_was_left: bool = True):
        """Run one action: from a cache brought back where a decode failed, following a clock the template reads, and put back as it was where the action raises.

        Args:
            finishes_what_was_left: Whether what a decode an interrupt stopped
                left undecoded is decoded first; a begin, which may hold
                another conversation, decodes only what it holds.
        """
        remedied = None
        if self._found_fault is not None:
            # What the check found after the newest reply, on a thread no caller reads; a
            # begin, which writes the cache anew, is its remedy, so it is kept only where that fails.
            fault, self._found_fault = self._found_fault, None
            if finishes_what_was_left:
                raise fault
            remedied = fault
        try:
            with self._acting(finishes_what_was_left):
                yield
        except BaseException:
            if remedied is not None and self._found_fault is None:
                self._found_fault = remedied
            raise

    @contextlib.contextmanager
    def _acting(self, finishes_what_was_left: bool):
        """:meth:`_action`'s work once a fault the check found is dealt with."""
        if not self._depth:
            # An action inside another keeps its renders; one whose template reads the clock
            # renders anew, as the clock may read otherwise now.
            self._rendering.next_action(keeps_renders=not self.template.properties.reads_clock)
        # Put back whole where the action fails, the part an interrupt left undecoded included.
        whole = list(self._memory.tokens)
        if not finishes_what_was_left:
            self._memory.forget_undecoded()
        if not (finishes_what_was_left and self._memory.is_behind
                and (self._context.can_shift or self._context.can_cut_tail)):
            # What an interrupt left undecoded is otherwise decoded with what the action adds,
            # in one batch; a cache that goes back to saved states is brought up first, so that
            # the states it keeps are saved where they are pinned.
            self._memory.repair()
        if self._state.begun and self.template.properties.reads_clock \
                and not self._opening.clock_reads_as_before():
            with clock_forms_written() as written:
                opening = self._rendering.opening_render(self._system_part())
            if opening == self._state.checks.opening:
                # Written alike, the reading it was written at is noted, so the next action
                # compares with it.
                self._state.checks = replace(self._state.checks,
                                             clock_read=tuple(dict.fromkeys(written)))
            else:
                # What was rendered ahead holds the old date, say; it is rendered whole now.
                self._state.checks = self._opening.render_checks()
                self._state.anchors = ()
                self._holding.pin_opening()
        saved, kept = self._state.copied(), self._memory.kept_conversations()
        # The copy made above serves unless the bring-up changed what the cache holds.
        held = whole if len(whole) > len(self._memory.tokens) or whole == self._memory.tokens \
            else list(self._memory.tokens)
        # Whether the cache held everything it is to hold as the action began.
        was_current = self._memory.is_current and held == self._memory.tokens
        with self._publishing():
            try:
                yield
            except BaseException as error:
                self._state.put_back(saved)
                self._memory.replace(held)
                self._memory.put_back_kept(kept)
                if isinstance(error, (RefusalError, InternalError)) and was_current \
                        and self._memory.is_known:
                    # An edit a refused action made, such as a shift for an eviction, is undone
                    # now (rule 6), as is one a fault of the library stopped partway; one the
                    # engine failed at leaves the cache to the next action.
                    with contextlib.suppress(Exception):
                        self._memory.repair()
                raise

    @contextlib.contextmanager
    def _publishing(self):
        """Run a change, published once the outermost change running ends, whether or not it raised."""
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            self._publish()

    def _publish(self) -> None:
        """Publish the conversation for readers that hold no lock, unless a change running publishes it once it ends."""
        if self._depth:
            return
        self._published = published(self._state, self.pending_calls, self.turn_open,
                                    len(self._memory.tokens), self._joins_questions,
                                    self._context.model_fingerprint)

    def _check_cache(self, prompted: bool) -> None:
        """Raise :class:`InternalError` where the cache disagrees with a fresh begin of the conversation, ``prompted`` where a reply is about to be generated."""
        from .cache_check import check_cache       # only where the setting asks for it
        check_cache(list(self._memory.tokens),
                    conversation_of(published(self._state, self.pending_calls, self.turn_open,
                                              len(self._memory.tokens), self._joins_questions,
                                              self._context.model_fingerprint)),
                    self._state.unreadable_result, self._context, self._templates, self._config,
                    RenderedCache, prompted=prompted)

    @property
    def _joins_questions(self) -> bool:
        """Whether two questions in a row are written as one, as the template writes no two."""
        return not self.template.properties.writes_two_user_turns


@contextlib.contextmanager
def _refused_as_the_text():
    """Name the text a question or a message was given as the field of a refusal of the conversation the template will not write with it."""
    try:
        yield
    except TemplateRefusedError as refused:
        if refused.field != "messages":
            raise
        raise TemplateRefusedError("text", str(refused)) from None


# ----- what an action may start from -------------------------------------------------
def _ensure_begun(conversation, into: str) -> None:
    """Refuse an action before a conversation was begun, which alone holds the sequence opener and the system prompt."""
    if not conversation.begun:
        raise OutOfOrderError(None, f"no conversation to {into}; call begin() first")


def _ensure_ready(conversation, action: str, into: str | None = None) -> None:
    """Refuse an action before a conversation was begun, or inside a turn a call left open.

    ``conversation`` is a :class:`RenderedCache` or what it published.
    """
    if not conversation.begun:
        raise OutOfOrderError(None, f"no conversation to {into or action}; call begin() first")
    if conversation.turn_open:
        raise OutOfOrderError(
            None,
            f"cannot {action} while a reply is open: it stopped on a tool "
            "call and its turn is unclosed. Answer each owed call with "
            "inject_tool_result() and go on with reply(), ask a new question, "
            "which closes the turn, or begin again.")


def _ensure_can_reply(conversation, asks_question: bool) -> None:
    """Refuse a reply ``conversation`` cannot take: a question before a conversation was begun, or, without one, a result still owed or nothing to answer."""
    if asks_question:
        _ensure_begun(conversation, "generate into")    # a question closes an open turn itself
    elif not conversation.turn_open:
        _ensure_ready(conversation, "start a request", "generate into")
        if conversation.newest_role != "user":
            raise OutOfOrderError("text", "there is nothing to answer: the newest message "
                                          "is not a user message; give the question as text")
    elif conversation.pending_calls:
        raise OutOfOrderError(None, f"{len(conversation.pending_calls)} call(s) still owe a "
                                    "result; answer each with inject_tool_result()")



def _shown(message: dict) -> str:
    """``message`` as an error names it: the first words of its text, else its calls, else that it has no text."""
    text, calls = message.get("content") or "", message.get("tool_calls") or ()
    if text:
        return repr(text[:40] + ("..." if len(text) > 40 else ""))
    return f"with {len(calls)} call(s)" if calls else "without text"





# ----- helpers ---------------------------------------------------------------------
def _reported_calls(generated: GenerationAccumulator, pending: list[ToolCall]) -> list[ToolCall]:
    """The calls one generation reports: those it left owed a result, or the one it could not read or finish."""
    if generated.stop_reason == "unreadable_tool_call":
        return [ToolCall("", {}, status="unreadable", text=generated.tool_call)]
    if generated.cut_inside_call:
        return [ToolCall("", {}, status="unfinished")]
    return pending



def _reuse_alike(messages: list[dict], held: list[dict]) -> int:
    """Put each message of ``held`` in the place of the one of ``messages`` there, from the first on, while the two say the same and keep the same turn as generated; how many were put."""
    count = 0
    for given, own in zip(messages, held):
        given_turn, own_turn = given.get(GENERATED_TURN), own.get(GENERATED_TURN)
        if _without_turn(given) != _without_turn(own) or (given_turn is None) != (own_turn is None) \
                or given_turn is not None and own_turn is not None \
                and (given_turn.tokens, given_turn.span, given_turn.on_call) \
                != (own_turn.tokens, own_turn.span, own_turn.on_call):
            break
        messages[count] = own
        count += 1
    return count


def _without_turn(message: dict) -> dict:
    """``message`` without the generated turn it may end, for comparing what it says."""
    return {key: value for key, value in message.items() if key != GENERATED_TURN}


def _goes_on_from(history: list[dict], cached: list[dict]) -> bool:
    """Whether ``history`` opens with what ``cached`` ends with, so that a cache holding ``cached`` holds its oldest part."""
    return bool(history) and any(history[:len(cached) - i] == cached[i:]
                                 for i in range(len(cached)))


def _since_question(messages: list[dict]) -> list[dict]:
    """The messages after the newest question, oldest first."""
    at = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), -1)
    return messages[at + 1:]


def _call_ids_since_question(messages: list[dict]) -> set[str]:
    """The ids of every call made since the newest question."""
    return {c["id"] for m in _since_question(messages) for c in m.get("tool_calls") or ()}
