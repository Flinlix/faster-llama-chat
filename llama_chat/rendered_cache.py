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
only those are rendered. See design_history.md, "The render is the
conversation's encoding", "Every text placed by a marker render" and
"Rendering windows".
"""

from __future__ import annotations

import contextlib
import copy
import logging
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace

from .generation import GenerationAccumulator
from .cache_sync import CacheSync, Cancelled
from .config import Config
from .conversation_state import (Ahead, Anchor, Changes, ConversationState, MadeFor, Published,
                                 RenderChecks, Snapshot, goes_on_from)
from .history_forms import (answer_in_order, completions_messages, joined_question,
                            read_messages, with_saved_turns, without_unreadable)
from .messages import (KEPT_TURN, UNKNOWN_FUNCTION, UNREADABLE_CALL, calling_message,
                       exchange_count, exchange_starts, history_count, next_call_number,
                       result_message, system_part)
from .render_checks import (STAND_IN, ensure_history_stays_written, question_splices,
                            repeated_question_written_alike, rest_stays_after_a_drop, splice_probe,
                            tool_exchanges, tool_turns_stay_written, windows_are_exact)
from .rendering import ConversationRendering, KeptTurn, Target, kept_turns, with_kept
from .eviction import Eviction
from .result_cutting import ResultCutting
from .windows import WHOLE, Windows
from .publishing import conversation_of, generated_turns_of, published
from .errors import (RAISE_THE_ROOM, ContextOverflowError, InvalidArgumentError,
                     OutOfOrderError, TemplateRefusedError, UnsupportedError,
                     with_template_reason)
from .resolved_template import ResolvedTemplate, ResolvedTemplates
from .split_render import SplitRender
from .sequences import common_prefix_length
from .reply import Reply, Report
from .reply_text import ReplyText
from .tools import ToolCall, ToolSpec, result_text

_log = logging.getLogger(__name__)

# How far the count of a message alone may fall short of its length in the system turn's render.
_NEAR_THE_ROOM_TOKENS = 64



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
        self._memory = CacheSync(context)
        # The one state object every part reads and changes; an action put back fills it in place.
        self._state = ConversationState()
        self._rendering = ConversationRendering(context, lambda: self.template, self._state)
        self._eviction = Eviction(self._rendering, config, context.can_shift, self._state,
                                  self._length)
        self._cutting = ResultCutting(self._length, self._state)
        self._windows = Windows(self._rendering, self._eviction, config, self._state)
        # Where the newest generation's turn starts in the cache, and whether it went on
        # with an open turn; left as they are by a failed action.
        self._generation: tuple[int, bool] = (0, False)
        # The conversation and the cache as the open attempt found them, until its reply is
        # recorded, and as the newest request that asked a question found them.
        self._attempt: Snapshot | None = None
        self._request: Snapshot | None = None
        self._rolled_back = False
        self._published = Published()
        # How many actions, one inside another, run now; the outermost publishes once it ends.
        self._depth = 0

    # ----- what it holds ---------------------------------------------------------------
    @property
    def messages(self) -> list[dict]:
        """The conversation as the template is given it; not to be changed by the caller."""
        return self._state.messages

    def conversation(self) -> dict:
        """Everything ``begin`` takes to begin the conversation again, by its parameter names, from one publication, the caller's to change; safe to read unlocked."""
        return conversation_of(self._published)

    def generated_turns(self) -> dict | None:
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
                ``asks_question``, as :meth:`ask` does where a reply is open.
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
    def reasoning(self) -> bool:
        """Whether the conversation reasons, as it was begun."""
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
        return [call for call, _ in _unanswered(self._state.messages)]

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
        self._state.turn_policy = policy

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
    def start(self, history: list, tools: list[ToolSpec] | None, reasoning: bool,
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
        with self._action():
            # What the cache holds of a conversation this one may go on with.
            cached = ([_without_turn(m) for m in self._state.messages if m["role"] != "system"]
                      if self._state.begun else [])
            self._state.tools, self._state.reasoning = list(tools or []), reasoning
            self._state.anchors = ()            # placed again as the cache is held
            self._state.uses_tool_template = tools is not None
            self._state.unreadable_result = unreadable_result
            joins = self._joins_questions
            messages, _n_found = with_saved_turns(read_messages(history),
                                                  self._usable_turns(generated_turns), joins)
            held = {i for first, last, _ in kept_turns(messages) for i in range(first, last + 1)}
            messages, gone = without_unreadable(messages, held, joins)
            self._state.messages[:] = answer_in_order(messages, result_text(unreadable_result))
            self._state.left_out_calls = frozenset(gone)
            self._state.changes.n_left_out_calls = len(gone)
            # Counted before the oldest are dropped, and never back, so that no later call
            # takes the id of one dropped, of one left out, or of one this conversation held
            # before.
            self._state.n_calls = max(self._state.n_calls, next_call_number(
                [c["id"] for m in self._state.messages for c in m.get("tool_calls") or ()]
                + sorted(gone)))
            self._state.kept_at = {}
            self._state.results_cut = []
            limit, room = self._config.evict_above_tokens, self._max_prompt_tokens
            self._ensure_room_for_a_question("shorten the prompt or declare fewer tools")
            asks = any(m["role"] == "user" for m in self._state.messages)
            self._keep_newest_history(cached)
            self._drop_unplaceable()
            if any(m["role"] != "system" for m in self._state.messages) \
                    and self._rendering.split(self._state.messages, self._ends_on_result()) is None:
                raise self._refused_history(self._state.messages)
            self._state.checks = self._render_checks()
            if asks and not any(m["role"] == "user" for m in self._state.messages):
                needed = self._rendering.length(self._state.messages, False)
                raise ContextOverflowError(
                    f"the newest exchange of the history, from its user message on, needs "
                    f"{needed} tokens beside the system prompt but {limit} fit; shorten it, "
                    f"{RAISE_THE_ROOM}", needed, limit)
            if asks:
                # The newest exchange may take what a running conversation may, its results
                # cut as they are cut there; a call no result follows yet was generated in
                # the room a reply leaves, as the whole context is.
                prompted = self._state.messages[-1]["role"] in ("user", "tool")
                if self._state.messages[-1].get("tool_calls"):
                    room = self._config.context_size
                if self._rendering.length(self._state.messages, prompted) > room:
                    self._cutting.cut_to_fit(room, self._held_indexes())
                needed = self._rendering.length(self._state.messages, prompted)
                if needed > room:
                    raise ContextOverflowError(
                        f"the newest exchange of the history, from its user message on, needs "
                        f"{needed} tokens beside the system prompt but {room} leave room for "
                        f"the reply; shorten it, {RAISE_THE_ROOM}", needed, room)
            self._drop_while_over(limit, generation_prompt=False)
            if not asks:
                self._ensure_within(limit, "shorten the history")
            self._state.changes.n_turns_restored = len(kept_turns(self._state.messages))
            # Counted as the caller's messages: those a kept turn holds besides are none of them.
            self._state.changes.n_kept_messages = history_count(
                completions_messages(self._state.messages, joins))
            self._state.called_in_thought = self._open_turn_in_thought()
            self._state.begun = False
            # The records made ahead for the conversation before, freed with it.
            self._state.ahead, self._state.spliced_at = None, None
            self._attempt = self._request = None
            self._pin_opening()
            # Compared with what the cache holds, which is decoded again only from where they
            # differ, a stretch it holds no longer removed.
            if self.pending_calls:
                self._sync(self._settled_while_owed(), logits=False, dropped=True)
            elif self._state.messages and self._state.messages[-1]["role"] in ("user", "tool"):
                prompted = self._prompted()
                opening = self._prompt_text_start(prompted.tokens)
                if opening < len(prompted.tokens) and self._memory.keeps_saved_states:
                    # A state saved before the template's text a reply follows, which a cache
                    # that cannot be cut goes back to where what follows is written otherwise.
                    self._sync(Target(prompted.tokens[:opening], prompted.kept_at), logits=False,
                               dropped=True)
                    self._memory.rest()
                self._sync(prompted, dropped=True)
            else:
                self._render_ahead(dropped=True)
            self._memory.rest()
            self._state.begun = True
            self._log_begin(sum(m["role"] != "system" for m in history), generated_turns)

    def _keep_newest_history(self, cached: list[dict]) -> None:
        """Keep the newest of the history being begun that fits under the eviction threshold where the cache holds the oldest of it already, as a conversation this one goes on with, else within half the context; the rest is recorded as evicted."""
        given, limit = list(self._state.messages), self._config.evict_above_tokens
        if cached:
            self._eviction.keep_newest_history(limit, self._refused_history)
            history = [_without_turn(m) for m in self._state.messages if m["role"] != "system"]
            if _goes_on_from(history, cached):
                self._eviction.record(history_count(given), "room")
                return
            self._state.messages[:] = given
        self._eviction.keep_newest_history(min(limit, self._config.context_size // 2),
                                           self._refused_history)
        self._eviction.record(history_count(given), "half_context")

    def _usable_turns(self, generated_turns: dict | None) -> list[dict]:
        """The saved turns of ``generated_turns`` whose tokens are this model's, every one of them; none where they are another model's."""
        if not generated_turns:
            return []
        if generated_turns["model"] != self._context.model_fingerprint():
            self._state.changes.generated_turns_ignored = True
            return []
        return generated_turns["turns"]

    def _held_indexes(self) -> set[int]:
        """Where the messages a kept turn's tokens stand for are."""
        return {i for first, last, _ in kept_turns(self._state.messages)
                for i in range(first, last + 1)}

    def _open_turn_in_thought(self) -> bool:
        """Whether the open turn's newest call, kept as generated, was made inside a thought the model had not closed, as its tokens read."""
        tags, messages = self.template.properties.reasoning, self._state.messages
        calling = next((m for m in reversed(messages) if m.get("tool_calls")), None)
        turn = calling.get(KEPT_TURN) if calling is not None else None
        if tags is None or turn is None or not turn.on_call or not self.turn_open \
                or not tags.thought_open.strip():
            return False
        text = "".join(self._context.token_text(token) for token in turn.tokens)
        return text.rfind(tags.thought_open.strip()) > text.rfind(tags.thought_close.strip())

    def _log_begin(self, n_given: int, generated_turns: dict | None) -> None:
        """Say once what ``begin`` left out of a history of ``n_given`` messages beyond the oldest, which the eviction says, and of the generated turns it was given; nothing where it kept them all as given. See design_history.md, "One report of what an action changed"."""
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
        standing = self._target_of(self._state.messages, generation_prompt=False)
        answered = self._target_of(
            self._state.messages + [result_message(call.name, call_id, STAND_IN)
                                    for call, call_id in _unanswered(self._state.messages)],
            generation_prompt=False)
        return Target(standing.tokens[:common_prefix_length(standing.tokens, answered.tokens)], {})

    def _refused_history(self, messages: list[dict]) -> TemplateRefusedError:
        """The refusal of ``messages``, a history this model's template will not write."""
        roles = [m["role"] for m in messages]
        refusal = self._refusal("user") if ["user", "user"] in [
            roles[i:i + 2] for i in range(len(roles) - 1)] \
            else "this model's template refuses to write this conversation"
        return TemplateRefusedError("messages", with_template_reason(
            refusal, self._rendering.refusal_reason(messages, messages[-1]["role"] == "tool")))

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
            ValueError: If no conversation was begun, a reply is open, or the
                template does not write the question here.
            ContextOverflowError: If the question cannot fit beside room for
                its reply, checked against the system prompt alone before
                anything is dropped.
            Cancelled: If ``cancel`` was set before the question was decoded
                whole, which leaves everything as it was.
        """
        self._memory.cancel = cancel
        try:
            self._ask(text)
        finally:
            self._memory.cancel = None

    def _ask(self, text: str) -> None:
        """:meth:`ask`'s work."""
        with self._action():
            _ensure_begun(self, "generate into")
            room = self._max_prompt_tokens
            question = {"role": "user", "content": text}
            alone = self._length_alone(question, room, generation_prompt=True)
            if alone > room:
                raise ContextOverflowError(
                    f"the request needs {alone} tokens but {room} fit beside room for the "
                    f"reply; shorten the request or the system prompt, or {RAISE_THE_ROOM}",
                    alone, room)
            joined = self._make_way_for(question, room, generation_prompt=True)
            if joined is None:
                self._ensure_writes("user", text, generation_prompt=True)
            if joined is not None or not self._splice(question, room):
                self._state.messages.append(joined or question)
                # As a spliced one, the question may open one exchange more until its reply settles.
                dropped = self._drop_while_over(room, generation_prompt=True, tokens_only=True)
                self._sync(self._fitting_prompt(), dropped=dropped)
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
        """The changes since the caller last asked, and a fresh start for the next."""
        changes, self._state.changes = self._state.changes, Changes()
        return Report(n_evicted_messages=changes.n_evicted,
                      evicted_for=changes.evicted_for if changes.n_evicted else None,
                      n_kept_messages=changes.n_kept_messages,
                      n_left_out_calls=changes.n_left_out_calls,
                      n_turns_restored=changes.n_turns_restored,
                      generated_turns_ignored=changes.generated_turns_ignored,
                      tool_results_cut=tuple(changes.tool_results_cut),
                      turn_closed=changes.turn_closed,
                      question_joined=changes.question_joined)

    def changes_so_far(self) -> Changes:
        """What the actions changed since the caller last asked for a :meth:`report`, left to be asked for still."""
        return replace(self._state.changes, tool_results_cut=list(self._state.changes.tool_results_cut))

    # ----- nothing blocks a question ------------------------------------------------------
    def _make_way_for(self, question: dict, limit: int, generation_prompt: bool) -> dict | None:
        """Close a turn a call left open, and join ``question`` to an unanswered one the template writes no second question after; the joined question, taken out to be added again, or ``None``.

        Where the two joined would not fit in ``limit`` beside the system
        prompt, the unanswered one goes instead, counted as evicted. See
        design_history.md, "Nothing blocks a question".
        """
        if self.turn_open:
            self._close_open_turn(question)
        newest = self._state.messages[-1] if self._state.messages else None
        if newest is None or newest["role"] != "user" or self._writes_after(question):
            return None
        self._state.messages.pop()
        joined = joined_question(newest, question)
        if self._length_alone(joined, limit, generation_prompt) > limit:
            self._state.changes.n_evicted += 1
            self._state.changes.evicted_for = "room"
            _log.info("an unanswered question went to make room for the one after it: the "
                      "template writes no two in a row, and the two joined do not fit")
            return None
        self._state.changes.question_joined = True
        _log.info("a question was joined to the one before it, which no reply followed: "
                  "the template writes no two in a row")
        return joined

    def _close_open_turn(self, question: dict) -> None:
        """Close the open turn ahead of ``question``: each call still owed is told it was not run, and a turn waiting for its reply gets an empty one where the template needs it; the turn is taken out where the template writes neither."""
        start = len(self._state.messages)
        owed = _unanswered(self._state.messages)
        self._state.messages += [result_message(call.name or UNKNOWN_FUNCTION, call_id,
                                                result_text(_NOT_RUN))
                                 for call, call_id in owed]
        self._state.results_cut, self._state.called_in_thought = [], False
        self._state.changes.turn_closed = True
        if self._writes_after(question):
            _log.info("a question closed the open turn, %d call(s) told they were not run",
                      len(owed))
            return
        self._state.messages.append({"role": "assistant", "content": ""})
        if self._writes_after(question):
            _log.info("a question closed the open turn with an empty reply, %d call(s) told "
                      "they were not run", len(owed))
            return
        del self._state.messages[start:]
        _log.info("a question took the open turn out: the template writes it after neither "
                  "its results nor an empty reply")
        # Neither is written: the turn goes, from its first message after the newest question.
        asked = max(i for i, m in enumerate(self._state.messages) if m["role"] == "user")
        del self._state.messages[asked + 1:]

    def _writes_after(self, question: dict) -> bool:
        """Whether the template writes ``question`` after the newest exchange as it stands."""
        return self._rendering.render(self._newest_exchange() + [question], True) is not None

    # ----- one attempt per generation ---------------------------------------------------
    def open_attempt(self, starts_request: bool, asks: bool) -> None:
        """Note the conversation and the cache as they are, for :meth:`roll_back` until the attempt's reply is recorded.

        Args:
            starts_request: Whether the attempt is a request's first generation.
            asks: Whether that request asks a question of its own, so that a
                reply taken back with it puts back what the question evicted.
        """
        self._attempt = Snapshot(self._state.copied(), list(self._memory.tokens))
        self._rolled_back = False
        if starts_request:
            self._request = self._attempt if asks else None

    def roll_back(self) -> None:
        """Put the conversation and the cache back as the open attempt found them.

        The cache is brought back at once where it is known; one a failure
        left unknown is brought back by the next action.
        """
        if self._attempt is not None:
            self._go_back_to(self._attempt)

    @property
    def rolled_back(self) -> bool:
        """Whether the newest attempt was put back, by :meth:`roll_back` or as its reply was taken back."""
        return self._rolled_back

    def _go_back_to(self, snapshot: Snapshot) -> None:
        """Hold the conversation and the cache's tokens of ``snapshot``, ending the attempt."""
        self._state.put_back(snapshot.state)
        self._memory.replace(snapshot.tokens)
        self._attempt, self._rolled_back = None, True
        if self._memory.is_known:
            with contextlib.suppress(Exception):
                self._memory.repair()       # where it fails, the next action decodes again
        self._publish()

    def _splice(self, question: dict, room: int) -> bool:
        """Add ``question``, decoded in the stand-in's place in the render made ahead, where that is exact; whether it was."""
        ahead = self._state.ahead
        text = question["content"]
        # A template may write a message otherwise where a later one repeats its text, which
        # only a render of the whole conversation shows.
        if (ahead is None or not self._is_current(ahead.made_for)
                or not self._rendering.is_plain(text)
                or (not self._state.checks.repeats_alike
                    and any(m["content"] == text for m in self._state.messages))
                or not question_splices(self._rendering.render, self._system_part(), text,
                                        self._state.checks.splice_probe)):
            return False
        rest = [(text if own and piece == STAND_IN else piece, own) for piece, own in ahead.rest]
        tokens = self._rendering.tokenize(rest)
        if len(self._memory.tokens) + len(tokens) > room:
            return False
        if self._memory.keeps_saved_states:
            newest = max(self._memory.pinned(_NEWEST_EXCHANGES), default=None)
            self._memory.pin([at for at in (newest, len(self._memory.tokens)) if at is not None],
                             _NEWEST_EXCHANGES)
        self._state.messages.append(question)
        self._memory.decode(tokens, want_logits=True)
        self._state.spliced_at = (len(self._memory.tokens) - len(tokens), tokens,
                                  self._made_for_now())
        self._note_prompt(ahead.after_question, ahead.reply_opening, self._memory.tokens)
        return True

    def add(self, role: str, text: str) -> None:
        """Decode a message without a reply after it, the oldest exchanges dropped for it where needed.

        A question is put in the stand-in's place as :meth:`ask` puts one, the
        generation prompt after it: within the room left for a reply, the
        conversation over ``evict_above_tokens`` until its reply settles, unless
        ``max_history_exchanges`` is reached first.

        Raises:
            ValueError: If no conversation was begun, a reply is open, or the
                template does not write it after this conversation.
            ContextOverflowError: If it cannot fit beside the system prompt
                alone, checked before anything is dropped.
        """
        with self._action():
            if role == "user":
                _ensure_begun(self, "inject into")
            else:
                _ensure_ready(self, "inject", "inject into")
            message = {"role": role, "content": text}
            limit = self._config.evict_above_tokens
            alone = self._length_alone(message, limit, generation_prompt=False)
            if alone > limit:
                raise ContextOverflowError(
                    f"the injected message needs {alone} tokens but {limit} fit beside the "
                    f"system prompt; shorten the message or the system prompt, or {RAISE_THE_ROOM}",
                    alone, limit)
            joined = (self._make_way_for(message, limit, generation_prompt=False)
                      if role == "user" else None)
            if joined is None:
                self._ensure_writes(role, text, generation_prompt=False)
            room = self._max_prompt_tokens
            fits = (self._config.max_history_exchanges is None
                    or exchange_count(self._state.messages) < self._config.max_history_exchanges)
            if role != "user" or joined is not None or not fits or not self._splice(message, room):
                self._state.messages.append(joined or message)
                dropped = self._drop_while_over(limit, generation_prompt=False)
                self._ensure_within(limit, "shorten the message or inject a user message first")
                self._sync(self._target_of(self._state.messages, generation_prompt=False),
                           logits=False, dropped=dropped)
            self._memory.rest()

    def _length_alone(self, message: dict, limit: int, generation_prompt: bool) -> int:
        """How many tokens ``message`` takes beside the system turn alone, measured in their render only where its count comes within :data:`_NEAR_THE_ROOM_TOKENS` of ``limit``."""
        counted = self._state.checks.with_a_short_question + self._rendering.count_tokens(
            message["content"])
        if counted + _NEAR_THE_ROOM_TOKENS <= limit:
            return counted
        return self._rendering.length(self._system_part() + [message], generation_prompt)

    def prompt(self) -> None:
        """Decode the generation prompt after the newest message, a user message, for a reply to it, the oldest exchanges dropped to leave room for the reply where needed.

        Raises:
            OutOfOrderError: If no conversation was begun, a reply is open, or
                the newest message is not a user message.
            ContextOverflowError: If the conversation leaves too little room for
                a reply once every exchange that can go is dropped.
        """
        with self._action():
            _ensure_ready(self, "start a request", "generate into")
            _ensure_can_reply(self, asks_question=False)
            spliced_at = self._state.spliced_at
            if spliced_at is not None and self._is_current(spliced_at[2]):
                return          # spliced with the generation prompt after it, within the room
            dropped = self._drop_while_over(self._max_prompt_tokens, generation_prompt=True,
                                            tokens_only=True)
            self._sync(self._fitting_prompt(), dropped=dropped)

    def _fitting_prompt(self) -> Target:
        """The conversation with the generation prompt after it.

        Raises:
            ContextOverflowError: If it leaves too little room for a reply.
        """
        target = self._prompted()
        room = self._max_prompt_tokens
        if len(target.tokens) > room:
            raise ContextOverflowError(
                f"the conversation needs {len(target.tokens)} tokens but {room} leave room "
                f"for the reply; {RAISE_THE_ROOM}", len(target.tokens), room)
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
        room = (self._config.context_size - len(self._memory.tokens)
                - self.template.properties.n_closing_tokens)
        return self._memory.generate(max(0, min(max_generated_tokens, room)), generated, **options)

    def generation(self, text: str | None, *, recognizes_calls: bool, withhold: bool,
                   max_generated_tokens: int, stop: tuple[str, ...],
                   cancel: threading.Event | None, sampling, unreadable_result: dict,
                   on_started: Callable[[], None] | None,
                   on_reply: Callable[[Reply], None]) -> Generator[str, None, Reply]:
        """Generate one reply from the cache's end, stream what it shows, and record what it left, as a Reply.

        Without ``text``, it goes on with the open turn, adding no message of
        its own, or else answers the newest message, which must be a user
        message already written. A call it stops on leaves the turn open for a
        result.

        Every exit before the generation's first token, by a cancel or a
        failure, leaves the conversation and the cache as this call found
        them, the history its question evicted included; so does a failure
        while the reply is recorded. See design_history.md, "One undo path".

        Args:
            text, stop, cancel: As :meth:`~llama_chat.wrapper.ChatWrapper.reply`.
            max_generated_tokens: The cap on this generation's tokens.
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
        """
        decoded_from = self.decoded
        evicted_before = self._state.changes.n_evicted
        self.open_attempt(starts_request=on_started is not None, asks=text is not None)
        try:
            if cancel is not None and cancel.is_set():
                raise Cancelled("cancelled before the request was written")
            if text is not None:
                self.ask(text, cancel)
            elif not self.turn_open:
                self.prompt()
        except Cancelled:
            self.roll_back()
            return Reply(text="", n_prefilled_tokens=self.decoded - decoded_from,
                         n_prompt_tokens=self.total_tokens, n_generated_tokens=0,
                         n_evicted_messages=0, stop_reason="interrupted")
        except BaseException:
            self.roll_back()
            raise
        n_prompt_tokens = self.total_tokens
        generated = GenerationAccumulator()
        results_cut = self.results_cut
        template = self.template
        try:
            goes_on = text is None and self.turn_open
            reply_text = ReplyText(template.properties.reasoning,
                                   template.answer_opening(self.reasoning),
                                   template.reply_header(self.reasoning),
                                   self.prompt_text, list(stop), generated.tag_spans,
                                   inside=goes_on and self.called_in_thought)
            pieces = self.generate(
                max_generated_tokens, generated, cancel=cancel, sampling=sampling,
                ends_turn=self._turn_end_test(reply_text, generated),
                boundary=replace(template.call_boundary(recognizes_calls, withhold),
                                 starts_in_thought=reply_text.opens_in_thought))
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
            if not generated.token_ids and (failed or generated.stop_reason == "interrupted"):
                self.roll_back()            # nothing was produced, so nothing changes
            else:
                try:
                    # Runs on normal completion and on early close alike.
                    self._record_generated(generated, reply_text, unreadable_result)
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
                tool_calls=_reported_calls(generated, self.pending_calls),
                tool_results_cut=results_cut,
                n_reasoning_tokens=reply_text.thought_pieces,
                thought=reply_text.thought,
            )
            on_reply(reply)
        return reply

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
                          unreadable_result: dict) -> None:
        """Record what one generation left and settle the cache after it.

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
            The answer as recorded, and whether a kept turn could no longer be
            placed.
        """
        start, resumed = self._generation
        unreadable = called and not calls
        if answer and not calls and self._holds_thought_tags(answer):
            # Where the template writes none of it as text, it is recorded as an answer with no text.
            tags = self.template.properties.reasoning
            answer = self._rendering.written_answer(
                self._state.messages, answer, thought,
                (tags.thought_close.strip(), tags.thought_open.strip())) or ""
        asked = calls or ([ToolCall("", {}, status="unreadable")]
                          if unreadable else [])
        message = ({"role": "assistant", "content": answer} if not asked
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
            open_turn = self._open_kept() if resumed else None
            if open_turn is not None:
                # The whole turn is kept, its results and their prompt included.
                at, turn = open_turn
                messages[at] = with_kept(messages[at], None)
                start = min(start, self._state.kept_at.pop(turn))
                span = len(messages) - (at - turn.span + 1) + 1
            message[KEPT_TURN] = KeptTurn(tuple(self._memory.tokens[start:]), span, bool(asked))
            self._state.kept_at[message[KEPT_TURN]] = start
        self._state.messages.append(message)
        # A reply that leaves every turn that called tools as it was cannot move one.
        moves_kept = called or resumed or not self._state.checks.tool_turns_stay
        return answer, moves_kept and self._drop_unplaceable()

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
                self._forget_calls()        # the cache is decoded again by the next action
                raise
            if not written:
                self._forget_calls()
        elif self.turn_open and self._answer_unreadable_owed():
            return                          # an entry of a batch that could not be read
        if self.turn_open:
            if unplaced:
                self._sync(self._target_of(self._state.messages, generation_prompt=False),
                           logits=False)
            return
        # A reply whose call was withdrawn, for want of room for its answer, is a reply too.
        replied = not self._state.messages[-1].get("tool_calls")
        if replied and not self._writes(self._newest_part(self._state.messages) if not answer
                                        else self._newest_exchange()):
            _log.info("a reply the template does not write, nor an empty one in its place, "
                      "was taken back with its request")
            if self._request is not None:
                # Taken back with the question this request asked, and what it evicted.
                self._go_back_to(self._request)
                self._state.changes.taken_back = True
                self._publish()
                return
            self._state.changes.taken_back = True
            starts = exchange_starts(self._state.messages)
            self._remove_from(starts[-1] if starts else len(self._system_part()))
            return
        if not answer and not called and not resumed \
                and self.template.properties.writes_two_user_turns:
            # A reply all thought or cut before its answer leaves the question
            # unanswered, as the next question then follows it.
            self._remove_from(len(self._state.messages) - 1)
            return
        self._settle()

    def _holds_thought_tags(self, text: str) -> bool:
        """Whether ``text`` holds the template's thought opening or closing, which a template may cut a past reply at."""
        tags = self.template.properties.reasoning
        return tags is not None and any(tag.strip() and tag.strip() in text
                                        for tag in (tags.thought_open, tags.thought_close))

    def _open_kept(self) -> tuple[int, KeptTurn] | None:
        """The kept turn the open reply goes on with, and where its last message stands: the one ending on its newest call, placed in the cache."""
        messages = self._state.messages
        at = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("tool_calls")),
                  None)
        turn = messages[at].get(KEPT_TURN) if at is not None else None
        if turn is None or not turn.on_call or turn not in self._state.kept_at:
            return None
        return at, turn

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

    def _answer_unreadable_owed(self) -> bool:
        """Answer each call still owed that could not be read, while one is next in order, with the failure the model is told; whether the turn closed instead, for want of room for it."""
        while self.pending_calls and self.pending_calls[0].status == "unreadable" \
                and self._state.unreadable_result is not None:
            if not self._write_unreadable(self._state.unreadable_result):
                # Left owed, it would hold the turn open with no call the caller can answer.
                self.close_unanswered()
                return True
        return False

    def _renders_result(self, message: dict) -> bool:
        """Whether the template writes ``message``, a result, after the conversation."""
        # A history with no question has no exchange to stand for it: it is read whole.
        newest = (self._newest_exchange() if exchange_starts(self._state.messages)
                  else self._state.messages)
        return self._rendering.render(newest + [message], False) not in (
            None, self._rendering.render(newest, False))

    def _forget_calls(self) -> None:
        """Record the newest reply without the call it made, which could be told nothing; its tokens stay as generated."""
        newest = {k: v for k, v in self._state.messages[-1].items() if k != "tool_calls"}
        turn = newest.get(KEPT_TURN)
        if turn is not None:
            newest[KEPT_TURN] = KeptTurn(turn.tokens, turn.span, False)
        self._state.messages[-1] = newest
        self._state.kept_at = {}
        self._drop_unplaceable()

    def _drop_unplaceable(self) -> bool:
        """Hold each kept turn the template writes where it cannot be placed as the template writes it instead; whether any was.

        Where a window of the newest exchanges stands for the render, only
        the turns it holds are placed: those before it stay as they were.
        """
        messages = self._state.messages
        if not kept_turns(messages):
            return False
        found = self._window_of(messages)
        system = system_part(messages)
        rendered = messages if found is None else system + messages[found[0]:]
        offset = 0 if found is None else found[0] - len(system)
        unplaced = set()
        for generation_prompt in (False, True):
            split = self._rendering.split(rendered, generation_prompt)
            if split is not None:
                unplaced |= {last + offset for first, last, turn in kept_turns(rendered)
                             if self._rendering.place(first, last, turn.on_call, split) is None}
        for last in unplaced:
            messages[last] = with_kept(messages[last], None)
        held = {turn for _, _, turn in kept_turns(messages)}
        self._state.kept_at = {k: at for k, at in self._state.kept_at.items() if k in held}
        return bool(unplaced)

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
            self._forget_calls()
            self._state.changes.turn_closed = True
            if not self._state.messages[-1]["content"] and not self._writes(
                    self._newest_part(self._state.messages)):
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
            waiting = _unanswered(self._state.messages)
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
                self._sync(self._fitting_prompt(), dropped=dropped)
            elif dropped:
                self._sync(self._target_of(self._state.messages, generation_prompt=False),
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
            ValueError: If no conversation was begun, a reply is open, the
                newest message is not a reply or holds a tool call, one that
                could not be read included, ``keep_text`` is not a prefix of
                its text, or the template does not write the reply shortened
                to it, or the tokens of a reply kept as generated hold more
                than its text.
        """
        with self._action():
            _ensure_ready(self, "truncate a reply", "truncate in")
            newest = self._state.messages[-1] if self._state.messages else None
            if newest is None or newest["role"] != "assistant":
                raise OutOfOrderError(None, "the newest message is not an assistant reply")
            if newest.get("tool_calls"):
                raise OutOfOrderError(None, "the newest reply holds a tool call, which would "
                                            "be separated from its result")
            if not newest["content"].startswith(keep_text):
                raise InvalidArgumentError("keep_text",
                                           "keep_text is not a prefix of the newest reply")
            if keep_text == newest["content"]:
                return
            shortened = {**newest, "content": keep_text}
            turn = newest.get(KEPT_TURN)
            if turn is not None:
                cut = self._kept_up_to(turn, newest["content"], keep_text)
                if cut is None:
                    raise TemplateRefusedError(
                        "keep_text", "the newest reply's tokens as generated hold more than "
                        "its text, such as a call that went, so they cannot be cut at keep_text")
                shortened[KEPT_TURN] = cut
            self._state.messages[-1] = shortened
            if not self._writes(self._newest_part(self._state.messages)):
                raise TemplateRefusedError(
                    "keep_text", f"this model's template does not write the newest reply "
                    f"shortened to {keep_text!r}; keep more of its text")
            self._render_ahead()
            self._memory.rest()

    def _kept_up_to(self, turn: KeptTurn, text: str, keep_text: str) -> KeptTurn | None:
        """``turn``, ending on a reply's ``text``, as its tokens hold the reply cut to ``keep_text``: its generated tokens up to the last whole one before the cut, the rest of ``keep_text`` tokenized; ``None`` where its tokens do not end on ``text``."""
        pieces = [self._context.token_bytes(token) for token in turn.tokens]
        whole, said = b"".join(pieces), text.encode()
        if not whole.endswith(said):
            return None
        end = len(whole) - len(said) + len(keep_text.encode())
        n, at = len(pieces), len(whole)
        while n and (at > end or not _decodes(whole[at:end])):
            n -= 1
            at -= len(pieces[n])
        rest = whole[at:end].decode()
        tokens = turn.tokens[:n] + tuple(self._rendering.tokenize([(rest, True)]) if rest else ())
        return KeptTurn(tokens, turn.span, turn.on_call)

    def _pin_opening(self) -> None:
        """Keep a state saved where the conversation's opening ends, where the cache goes back to saved states."""
        self._memory.pin([self._rendering.opening_length(self._system_part())]
                         if self._memory.keeps_saved_states else [], "opening")

    def _render_checks(self) -> RenderChecks:
        """What the template's renders allow, as the system turn, the declarations and the clock now leave them.

        Raises:
            ValueError: If the template writes an older exchange otherwise once
                another follows.
        """
        system = self._system_part()
        render = self._rendering.render
        ensure_history_stays_written(render, system)
        tools = self._state.tools
        # Declarations that move with the newest question leave no window as the whole
        # render writes it.
        windows = (not self._moves_declarations()
                   and windows_are_exact(render, system, self._rendering.tag_from))
        calls = self.template.properties.can_call_tools
        return RenderChecks(
            opening=self._rendering.opening_render(system),
            windows=windows,
            # Only a template that calls tools is asked.
            tool_windows=windows and calls and windows_are_exact(
                render, system, self._rendering.tag_from,
                tool_exchanges(tools[0].name if tools else "probe")),
            splice_probe=splice_probe(render, system),
            repeats_alike=repeated_question_written_alike(render, system),
            with_a_short_question=self._eviction.with_a_short_question(),
            rest_stays=rest_stays_after_a_drop(render, system),
            # Without the steps tool calling needs, no turn keeps a call, and the template is
            # not asked.
            tool_turns_stay=self.template.properties.can_call_tools and tool_turns_stay_written(
                render, system, tools[0].name if tools else "probe", thought=self._state.reasoning))

    def _ensure_writes(self, role: str, text: str, generation_prompt: bool) -> None:
        """Refuse a ``role`` message the template would not write next.

        Raises:
            ValueError: If the template refuses the conversation with it.
        """
        # Refusals turn on the newest messages, such as two user turns in a row, so
        # they are read from the newest exchange, not from a render of the whole conversation.
        messages = self._newest_exchange() + [{"role": role, "content": text}]
        if self._rendering.render(messages, generation_prompt) is None:
            raise TemplateRefusedError("text", with_template_reason(
                self._refusal(role), self._rendering.refusal_reason(messages, generation_prompt)))

    def _refusal(self, role: str) -> str:
        """Why the template will not write a ``role`` message next."""
        newest = self._state.messages[-1]["role"] if self._state.messages else None
        if role == "user" and newest == "user":
            return ("this model's template does not write two user turns in a row; write a "
                    "reply between them")
        return f"this model's template does not write a {role} message here"

    def _ensure_room_for_a_question(self, remedy: str) -> None:
        """Refuse a system turn and declarations that leave no room under ``evict_above_tokens`` for a short question.

        Raises:
            ContextOverflowError: With ``remedy`` as what to do instead.
        """
        limit, needed = self._config.evict_above_tokens, self._eviction.with_a_short_question()
        if needed > limit:
            raise ContextOverflowError(
                f"the system prompt and its declarations need {needed} tokens with a short "
                f"question but the eviction threshold is {limit}; {remedy}, or {RAISE_THE_ROOM}",
                needed, limit)

    def _ensure_within(self, limit: int, remedy: str) -> None:
        """Refuse a conversation that still renders longer than ``limit`` once every exchange that can go was dropped.

        Raises:
            ContextOverflowError: With ``remedy`` as what to do instead.
        """
        needed = self._length(self._state.messages, generation_prompt=False)
        if needed > limit:
            raise ContextOverflowError(
                f"the conversation needs {needed} tokens but the eviction threshold is "
                f"{limit}, and the messages no user message follows cannot be evicted; "
                f"{remedy}, or {RAISE_THE_ROOM}", needed, limit)

    def _remove_from(self, first: int) -> None:
        """Take the messages from ``first`` on out again, then drop what is over the limits."""
        with self._action():
            del self._state.messages[first:]
            dropped = self._drop_while_over(self._config.evict_above_tokens,
                                            generation_prompt=False)
            # As after any reply, so that what the template writes ahead of the next
            # question, a default system prompt say, stays in the cache.
            self._render_ahead(dropped=dropped)
            self._memory.rest()

    # ----- after a reply ----------------------------------------------------------
    def _settle(self) -> None:
        """Drop what is over the limits, then render and decode ahead for the next question."""
        with self._action():
            limit = self._config.evict_above_tokens
            plan = self._windows.ahead(self._memory) if self._state.checks.windows else WHOLE
            if plan is not WHOLE:
                # Dropped first, so the cache shifts before the decode ahead, not before the next question's.
                at, before, length, split, run = plan
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
                    anchors = () if tag is None or tag > run else ((
                        len(self._state.messages) - 1, kept_until() + len(before)
                        - len(self._rendering.tokenize(split.pieces(tag, run)))),)
                    self._sync(Target(list(self._memory.tokens[:kept_until()]) + before,
                                      self._state.kept_at, anchors), logits=False)
                    self._note_ahead(split, run)
                else:
                    self._drop_while_over(limit, generation_prompt=False, newest_too=True)
                    self._render_ahead(dropped=True)
            else:
                dropped = self._drop_while_over(limit, generation_prompt=False, newest_too=True)
                self._render_ahead(dropped=dropped)
            self._memory.rest()

    def _render_ahead(self, dropped: bool = False) -> None:
        """Decode the render for the next question up to where the question goes."""
        messages = self._state.messages + [{"role": "user", "content": STAND_IN}]
        view = self._view(messages, generation_prompt=True)
        split = view.split if view is not None else None
        run = (self._rendering.run_start(split, len(split.contents) - 1, view.start)
               if split is not None and split.text.count(STAND_IN) == 1 else None)
        if run is not None:
            before = self._view_target(view, upto=run)
            if len(before.tokens) <= self._config.context_size:
                self._sync(before, logits=False, dropped=dropped)
                self._note_ahead(split, run)
                return
        # The next question will be rendered when it comes.
        if self._state.messages and self._rendering.split(self._state.messages, False) is None:
            return
        self._sync(self._target_of(self._state.messages, generation_prompt=False),
                   logits=False, dropped=dropped)

    def _system_part(self) -> list[dict]:
        return system_part(self._state.messages)

    # ----- renders of the newest exchanges --------------------------------------------
    def _window_of(self, messages: list[dict]) -> tuple[int, Anchor] | None:
        """Where a window of ``messages`` begins that stands for their render after one of the cache's anchors, the newest that still stands, and that anchor; ``None`` where no window does.

        The window opens with the exchange the anchor's message ends, and
        holds at most two more.
        """
        if not self._state.checks.windows:
            return None
        for anchor in reversed(self._state.anchors):
            found = self._window_after(messages, anchor)
            if found is not None:
                return found, anchor
        return None

    def _window_after(self, messages: list[dict], anchor: Anchor) -> int | None:
        """Where a window of ``messages`` after ``anchor`` begins, where one stands for their render."""
        if anchor.index >= len(messages) or messages[anchor.index] is not anchor.message \
                or len(self._memory.tokens) <= anchor.at:
            return None
        starts = exchange_starts(messages)
        opening = [i for i, at in enumerate(starts) if at <= anchor.index]
        if not opening or len(starts) - opening[-1] > 3:
            return None
        start = starts[opening[-1]]
        if not self._state.checks.tool_windows \
                and not self._rendering.template_writes(messages[start:]):
            return None
        kept = tuple(turn for _, _, turn in kept_turns(messages[:anchor.index + 1]))
        if len(kept) != len(anchor.kept) or any(a is not b for a, b in zip(kept, anchor.kept)):
            return None
        return start

    def _view(self, messages: list[dict], generation_prompt: bool) -> _View | None:
        """``messages`` rendered as the cache holds them: a window of the newest of them after the cache's anchor where one stands for the whole, else all of them; ``None`` where the template refuses them."""
        found = self._window_of(messages)
        if found is not None:
            start, anchor = found
            system = system_part(messages)
            window = system + messages[start:]
            split = self._rendering.split(window, generation_prompt)
            tag = (None if split is None
                   else self._rendering.tag_after_content(split, len(system) + anchor.index - start))
            if tag is not None:
                view = _View(split, window, start - len(system), tag,
                             list(self._memory.tokens[:anchor.at]),
                             {k: at for k, at in self._state.kept_at.items() if at < anchor.at},
                             generation_prompt)
                opening = self._rendering.target(window, generation_prompt, start=tag).tokens[:1]
                if opening == list(self._memory.tokens[anchor.at:anchor.at + 1]):
                    return view
        split = self._rendering.split(messages, generation_prompt)
        return None if split is None else _View(split, messages, 0, 0, [], {}, generation_prompt)

    def _view_target(self, view: _View, upto: int | None = None) -> Target:
        """What the cache should hold for the messages ``view`` stands for, up to ``upto`` in its render, with where the newest exchange of what it renders opens."""
        rendering, base = self._rendering, len(view.base)
        found = rendering.target(view.messages, view.generation_prompt, upto=upto, start=view.start)

        def at(tag: int) -> int:
            return base + len(rendering.target(view.messages, view.generation_prompt, upto=tag,
                                               start=view.start).tokens)

        starts = [s for s in exchange_starts(view.messages) if s > len(system_part(view.messages))]
        end = len(view.split.text) if upto is None else upto
        tags = [(s - 1, rendering.tag_after_content(view.split, s - 1)) for s in starts[-2:]]
        known = [(i, tag) for i, tag in tags if tag is not None and view.start <= tag <= end]
        anchors = tuple((i + view.offset, at(tag)) for i, tag in known)
        # The exchanges of the conversation, not one a stand-in after it opens.
        conversation = [(i, tag) for i, tag in known
                        if i + 1 + view.offset < len(self._state.messages)]
        return Target(view.base + found.tokens,
                      {**view.base_kept, **{k: base + p for k, p in found.kept_at.items()}},
                      anchors, tuple(at(tag) for _, tag in conversation[-2:]))

    def _target_of(self, messages: list[dict], generation_prompt: bool) -> Target:
        """What the cache should hold for ``messages``, from a window of the newest of them where one stands for the whole.

        Raises:
            ValueError: As :meth:`ConversationRendering.target` does.
        """
        view = self._view(messages, generation_prompt)
        if view is None:
            return self._rendering.target(messages, generation_prompt)
        return self._view_target(view)

    def _length(self, messages: list[dict], generation_prompt: bool) -> int:
        """How many tokens the cache holds for ``messages``, as :meth:`ConversationRendering.length` counts them, from a window of the newest of them where one stands for the whole."""
        if self._window_of(messages) is not None:
            view = self._view(messages, generation_prompt)
            if view is not None and view.base:
                return len(self._view_target(view).tokens)
        return self._rendering.length(messages, generation_prompt)

    def _newest_part(self, messages: list[dict]) -> list[dict]:
        """The system message and the window of the newest of ``messages`` that stands for their render, or all of them where none does; what a refusal of the newest is read from."""
        found = self._window_of(messages)
        return messages if found is None else system_part(messages) + messages[found[0]:]

    # ----- the cache against the target -----------------------------------------
    def _sync(self, target: Target, *, logits: bool = True, dropped: bool = False) -> None:
        """Make the cache hold ``target``, and note where its kept turns begin there.

        Args:
            logits: A generation follows the decode at once.
            dropped: Exchanges were dropped, so that the target may go on as
                the cache does after a stretch.
        """
        if self._memory.keeps_saved_states:
            # States there let a change to the newest exchanges decode only those exchanges.
            places = (self._newest_exchanges_at() if target.exchanges_at is None
                      else self._with_older_pinned(target.exchanges_at))
            if places is not None:
                self._memory.pin(places, _NEWEST_EXCHANGES)
        # Declarations that move with the newest question leave a stretch behind, and the
        # next question's move back to where this cut is made, or past it.
        moves = self._moves_declarations()
        # The anchors before the first token the target changes still stand.
        same = common_prefix_length(self._memory.tokens, target.tokens)
        self._memory.hold(target.tokens, want_logits=logits,
                          find_dropped_stretch=dropped or moves, rest_at_cut=moves)
        self._state.kept_at = target.kept_at
        messages = self._state.messages
        new = {index: Anchor(messages[index], index, at, tuple(
                   turn for _, _, turn in kept_turns(messages[:index + 1])))
               for index, at in target.anchors if index < len(messages)}
        standing = {a.index: a for a in self._state.anchors
                    if a.at < same and a.index < min(new, default=len(messages))}
        self._state.anchors = tuple(sorted({**standing, **new}.values(),
                                           key=lambda a: a.index)[-2:])

    def _with_older_pinned(self, places: tuple[int, ...]) -> list[int]:
        """``places``, where a window placed the tags of the newest exchanges it holds, with the newest pinned before them where it holds only one."""
        if len(places) >= 2:
            return list(places)
        older = [at for at in self._memory.pinned(_NEWEST_EXCHANGES) if not places or at < places[0]]
        return ([max(older)] if older else []) + list(places)

    def _newest_exchanges_at(self) -> list[int] | None:
        """Where in the cache each of the two newest exchanges begins, the tag that opens its question, those that cannot be told left out; ``None`` where this action rendered the conversation whole in neither form, which it is not rendered again for."""
        messages = self._state.messages
        starts = [at for at in exchange_starts(messages)[-2:] if at]
        for prompted in (False, True):
            split = self._rendering.split_made(messages, prompted)
            if split is not None:
                break
        else:
            return None
        tags = [self._rendering.tag_after_content(split, at - 1) for at in starts]
        return [len(self._rendering.target(messages, prompted, upto=tag).tokens)
                for tag in tags if tag is not None]

    def _moves_declarations(self) -> bool:
        """Whether the declarations move with the newest question, leaving a stretch behind."""
        return bool(self._state.tools) and self.template.properties.declares_before_newest_turn

    def _prompted(self) -> Target:
        """The conversation with the generation prompt after it, noting the template text a reply follows."""
        view = self._view(self._state.messages, generation_prompt=True)
        if view is None:
            self._rendering.target(self._state.messages, generation_prompt=True)   # its refusal
        target, split = self._view_target(view), view.split
        last = max((end for _, end in split.own), default=0)
        self._note_prompt(split.text[last:], self._reply_opening(split), target.tokens)
        return target

    def _prompt_text_start(self, target: list[int]) -> int:
        """Where in ``target`` the template text the next reply follows begins, the target's end where it does not end on it."""
        tokens = self._rendering.tokenize([(self._state.prompt_text, False)])
        ends = tokens and list(target[-len(tokens):]) == tokens
        return len(target) - (len(tokens) if ends else 0)

    def _reply_opening(self, split: SplitRender) -> list[tuple[str, bool]]:
        """The render ``split`` from where a reply after its newest message begins as a kept turn."""
        start = self._rendering.tag_after_content(split, len(split.contents) - 1)
        return split.pieces(start) if start is not None else []

    def _note_prompt(self, prompt_text: str, reply_opening: list[tuple[str, bool]],
                     target) -> None:
        """Note the template text a reply follows, and where in ``target`` the reply's kept turn begins: at ``reply_opening``, the render's end."""
        self._state.prompt_text = prompt_text
        tokens = self._rendering.tokenize(reply_opening)
        ends = tokens and list(target[-len(tokens):]) == tokens
        self._state.prompt_start = len(target) - (len(tokens) if ends else 0)

    # ----- limits -------------------------------------------------------------------
    @property
    def _max_prompt_tokens(self) -> int:
        """The most tokens the cache may hold when a reply starts, ``min_reply_tokens`` left free."""
        return self._config.context_size - self._config.min_reply_tokens

    def _drop_while_over(self, limit: int, generation_prompt: bool, **options) -> bool:
        """Drop the oldest exchanges once the conversation is over ``limit`` or holds too many, as :meth:`Eviction.drop_while_over` does; whether any went."""
        return self._eviction.drop_while_over(limit, generation_prompt, **options) is not None

    # ----- what a record made ahead was made for ----------------------------------------
    def _note_ahead(self, split: SplitRender, run: int) -> None:
        """Note ``split``, decoded up to ``run``, as the render made ahead for the conversation and the cache as they are now."""
        self._state.ahead = Ahead(rest=split.pieces(run),
                                   after_question=split.text[split.contents[-1][1]:],
                                   reply_opening=self._reply_opening(split),
                                   made_for=self._made_for_now())

    def _made_for_now(self) -> MadeFor:
        """The conversation and the cache as they are now, for a record made of them."""
        state = self._state
        return MadeFor(tuple(state.messages), tuple(state.tools), state.reasoning,
                        state.checks.opening, self._memory.version)

    def _is_current(self, made_for: MadeFor) -> bool:
        """Whether the conversation and the cache are still those ``made_for`` was made for."""
        return (made_for.cache_version == self._memory.version
                and len(made_for.messages) == len(self._state.messages)
                and goes_on_from(self._state, made_for))

    # ----- one action at a time --------------------------------------------------------
    @contextlib.contextmanager
    def _action(self):
        """Run one action: from a cache brought back where a decode failed, following a clock the template reads, and put back as it was where the action raises."""
        self._rendering.forget_renders()
        self._memory.repair()
        if self._state.begun and self.template.properties.reads_clock:
            opening = self._rendering.opening_render(self._system_part())
            if opening != self._state.checks.opening:
                # What was rendered ahead holds the old date, say; it is rendered whole now.
                self._state.checks = self._render_checks()
                self._state.anchors = ()
                self._pin_opening()
        saved = self._state.copied()
        held = list(self._memory.tokens)
        with self._publishing():
            try:
                yield
            except BaseException:
                self._state.put_back(saved)
                self._memory.replace(held)
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

    @property
    def _joins_questions(self) -> bool:
        """Whether two questions in a row are written as one, as the template writes no two."""
        return not self.template.properties.writes_two_user_turns


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
            "call and its turn is unclosed. Finish it with "
            "reply(), or discard it with begin().")


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


# What the states a cache that cannot be cut keeps where the newest exchanges begin are kept for.
_NEWEST_EXCHANGES = "newest exchanges"

# What a call still owed is told when a question closes its turn.
_NOT_RUN = {"error": "not_run", "message": "the call was not run: the user asked something new"}



# ----- helpers ---------------------------------------------------------------------
@dataclass(frozen=True)
class _View:
    """Messages as the cache holds them: rendered whole, or as a window of the newest after the cache's anchor.

    The cache holds ``base`` and then the tokens of ``split`` from ``start``
    on; ``messages`` is what was rendered, and the conversation's message
    ``i`` is its ``i - offset``-th, the system message aside.
    """

    split: SplitRender
    messages: list[dict]
    offset: int
    start: int
    base: list[int]
    base_kept: dict
    generation_prompt: bool


def _reported_calls(generated: GenerationAccumulator, pending: list[ToolCall]) -> list[ToolCall]:
    """The calls one generation reports: those it left owed a result, or the one it could not read or finish."""
    if generated.stop_reason == "unreadable_tool_call":
        return [ToolCall("", {}, status="unreadable", text=generated.tool_call)]
    if generated.cut_inside_call:
        return [ToolCall("", {}, status="unfinished")]
    return pending


def _decodes(data: bytes) -> bool:
    """Whether ``data`` is whole UTF-8 text."""
    try:
        data.decode()
    except UnicodeDecodeError:
        return False
    return True


def _without_turn(message: dict) -> dict:
    """``message`` without the kept turn it may end, for comparing what it says."""
    return {key: value for key, value in message.items() if key != KEPT_TURN}


def _goes_on_from(history: list[dict], cached: list[dict]) -> bool:
    """Whether ``history`` opens with what ``cached`` ends with, so that a cache holding ``cached`` holds its oldest part."""
    return bool(history) and any(history[:len(cached) - i] == cached[i:]
                                 for i in range(len(cached)))


def _unanswered(messages: list[dict]) -> list[tuple[ToolCall, str]]:
    """The calls of the newest turn that no result after them answers yet."""
    calling = None
    for i in range(len(messages) - 1, -1, -1):          # back to the newest call, at most
        if messages[i].get("tool_calls"):
            calling = i
            break
        if messages[i]["role"] != "tool":
            return []                   # a message after the newest call: nothing is owed
    if calling is None:
        return []
    answered = len(messages) - calling - 1
    return [(ToolCall("", {}, c["id"], "unreadable") if UNREADABLE_CALL in c else
             ToolCall(c["function"]["name"], copy.deepcopy(c["function"]["arguments"]), c["id"]),
             c["id"]) for c in messages[calling]["tool_calls"][answered:]]
