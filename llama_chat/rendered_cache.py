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
import functools
import logging
import threading
from dataclasses import replace

from .cache_context import GenerationAccumulator
from .cache_sync import CacheSync, Cancelled
from .config import Config
from .conversation_state import (Ahead, Changes, ConversationState, MadeFor, Published, Snapshot,
                                 goes_on_from)
from .messages import (UNKNOWN_FUNCTION, UNREADABLE_CALL, as_messages, calling_message,
                       exchange_count, exchange_starts, history_count,
                       next_call_number, portable_digest, result_message, system_part,
                       with_result_text)
from .render_checks import (STAND_IN, ensure_history_stays_written, question_splices,
                            repeated_question_written_alike, splice_probe, tool_turns_stay_written,
                            windows_are_exact)
from .rendering import ConversationRendering, KeptTurn, Target
from .eviction import Eviction
from .result_cutting import ResultCutting, cut_result
from .windows import WHOLE, Windows
from .publishing import published
from .errors import (ContextOverflowError, InvalidArgumentError, OutOfOrderError,
                     TemplateRefusedError, UnsupportedError)
from .resolved_template import ResolvedTemplate, ResolvedTemplates
from .split_render import SplitRender
from .template import common_prefix_length
from .tokenizing import split_at_tags
from .reply import Report
from .tools import ToolCall, ToolSpec, result_text

_log = logging.getLogger(__name__)

# How far the count of a message alone may fall short of its length in the system turn's render.
_NEAR_THE_ROOM_TOKENS = 64



class RenderedCache:
    """One conversation's messages, as its template is given them, and the cache holding their render.

    Every method expects the caller to hold the conversation's lock, except
    :meth:`conversation`, :meth:`published_pending_calls`,
    :meth:`published_total_tokens`, :meth:`published_turn_open` and
    :meth:`ensure_can_reply`, which read a
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
        self._state = ConversationState()
        self._rendering = ConversationRendering(context, lambda: self.template,
                                                lambda: self._state)
        self._eviction = Eviction(self._rendering, config, context)
        self._cutting = ResultCutting(self._rendering)
        self._windows = Windows(self._rendering, self._eviction, config)
        # Where the newest generation's turn starts in the cache, and whether it went on
        # with an open turn; left as they are by a failed action.
        self._generation: tuple[int, bool] = (0, False)
        # The conversation and the cache as the open attempt found them, until its reply is
        # recorded, and as the newest request that asked a question found them.
        self._attempt: Snapshot | None = None
        self._request: Snapshot | None = None
        self._rolled_back = False
        self._changes = Changes()
        # What a call that could not be read is answered with, as the newest reply was told.
        self._unreadable_result: dict | None = None
        self._published = Published()

    # ----- what it holds ---------------------------------------------------------------
    @property
    def messages(self) -> list[dict]:
        """The conversation as the template is given it; not to be changed by the caller."""
        return self._state.messages

    def conversation(self) -> dict:
        """Everything ``begin`` takes to begin the conversation again, by its parameter names, from one publication, the caller's to change; safe to read unlocked."""
        published = self._published
        return {"system_prompt": published.system_prompt,
                "messages": copy.deepcopy(list(published.history)),
                "tools": None if published.tools is None else copy.deepcopy(list(published.tools)),
                "reasoning": published.reasoning, "generated_turns": self.generated_turns()}

    def generated_turns(self) -> dict | None:
        """Each turn kept as generated, from one publication, as :meth:`start` takes them back; ``None`` where none is; safe to read unlocked."""
        kept = self._published.kept
        if not kept:
            return None
        return {"model": self._context.model_fingerprint(), "turns": [
            {"tokens": list(tokens), "message_count": count, "digest": digest}
            for tokens, count, digest in kept]}

    def published_pending_calls(self) -> list[ToolCall]:
        """The calls still owed a result, oldest first, from the newest publication, the caller's to change; safe to read unlocked."""
        return copy.deepcopy(list(self._published.pending_calls))

    def published_total_tokens(self) -> int:
        """How many tokens the cache holds, from the newest publication; safe to read unlocked."""
        return self._published.total_tokens

    def published_has_pending_calls(self) -> bool:
        """Whether a call is owed a result, from the newest publication; safe to read unlocked."""
        return bool(self._published.pending_calls)

    def published_awaits_answer(self) -> bool:
        """Whether a begun conversation's newest message is a user message, from the newest publication; safe to read unlocked."""
        return self._published.begun and self._published.newest_role == "user"

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
    def start(self, system_prompt: str, history, tools: list[ToolSpec] | None,
              reasoning: bool, generated_turns: dict | None = None) -> int:
        """Begin again from ``system_prompt`` and the newest of ``history`` that fits in half the context; how many items were kept.

        A history that ends on a call keeps it owed a result, and the cache
        holds the render only as far as the results cannot change it.
        ``tools`` other than ``None``, even empty, writes the conversation with
        the model's template for tools. Each turn of ``generated_turns``, as
        :meth:`conversation` gives one, whose messages are kept unchanged is
        held as the tokens it was generated as, where they are this
        model's; a call the model wrote unreadably that no such turn covers
        is left out, with its error result.

        Raises:
            ContextOverflowError: If the system prompt and its declarations leave
                no room under ``evict_above_tokens`` for a short question, or the
                newest exchange beside them exceeds it.
            ValueError: If the template refuses to write the conversation, or
                writes an older exchange otherwise once another follows.
        """
        with self._action():
            self._state.tools, self._state.reasoning = list(tools or []), reasoning
            self._state.uses_tool_template = tools is not None
            self._rendering.forget_renders()
            system = [{"role": "system", "content": system_prompt}] if system_prompt else []
            self._state.messages = system + as_messages(history, self._state.n_calls)
            # Counted before the oldest are dropped, and never back, so that no later call
            # takes the id of one dropped, or of one this conversation held before.
            self._state.n_calls = max(self._state.n_calls, next_call_number(
                c["id"] for m in self._state.messages for c in m.get("tool_calls") or ()))
            self._state.kept, self._state.kept_at = [], {}
            self._state.called_in_thought = False
            self._state.results_cut = []
            limit = self._config.evict_above_tokens
            self._ensure_room_for_a_question("shorten the prompt or declare fewer tools")
            asks = any(m["role"] == "user" for m in self._state.messages)
            given = len(self._state.messages)
            self._eviction.keep_newest_history(self._state, min(limit, self._config.context_size // 2),
                                               self._refused_history)
            if len(self._state.messages) < given:
                self._changes.evicted_for = "half_context"
            self._restore_generated(generated_turns)
            if any(m["role"] != "system" for m in self._state.messages) \
                    and self._rendering.split(self._state.messages, self._ends_on_result()) is None:
                raise self._refused_history(self._state.messages)
            self._check_renders()
            needed = self._rendering.length(self._state.messages, False)
            if asks and (not any(m["role"] == "user" for m in self._state.messages)
                         or needed > limit):
                raise ContextOverflowError(
                    f"the newest exchange of the history, from its user message on, needs "
                    f"{needed} tokens beside the system prompt but {limit} fit; shorten it, "
                    f"{_RAISE_THE_ROOM}", needed, limit)
            self._drop_while_over(limit, generation_prompt=False)
            self._ensure_within(limit, "shorten the history")
            n_kept = len(self._state.messages) - len(system)
            self._leave_out_unreadable()
            self._changes.n_kept_messages = n_kept
            self._log_begin(len(history), generated_turns)
            self._state.begun = False
            self._memory.reset()
            # The records made ahead for the conversation before, freed with it.
            self._state.ahead, self._state.spliced_at = None, None
            self._attempt = self._request = None
            self._state.opening = self._rendering.opening_render(self._system_part())
            self._pin_opening()
            if self.pending_calls:
                self._sync(self._settled_while_owed(), logits=False)
            elif self._state.messages and self._state.messages[-1]["role"] in ("user", "tool"):
                prompted = self._prompted()
                opening = self._prompt_text_start(prompted.tokens)
                if opening < len(prompted.tokens) and self._memory.keeps_saved_states:
                    # A state saved before the template's text a reply follows, which a cache
                    # that cannot be cut goes back to where what follows is written otherwise.
                    self._sync(Target(prompted.tokens[:opening], prompted.kept_at), logits=False)
                    self._memory.rest()
                self._sync(prompted)
            else:
                self._render_ahead()
            self._memory.rest()
            self._state.begun = True
            return self._changes.n_kept_messages

    def _restore_generated(self, generated_turns: dict | None) -> None:
        """Hold each of ``generated_turns`` whose messages stand unchanged as the tokens it was generated as, where they are this model's."""
        if not generated_turns:
            return
        if generated_turns.get("model") != self._context.model_fingerprint():
            self._changes.generated_turns_ignored = True
            return
        messages = self._state.messages
        # A kept turn opens on a reply that calls, so only spans that start on one are tried.
        starts = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")]
        taken = set()
        for turn in generated_turns.get("turns", ()):
            count = turn["message_count"]
            first = next((i for i in reversed(starts) if i not in taken and i + count <= len(messages)
                          and portable_digest(messages[i:i + count]) == turn["digest"]), None)
            if first is None:
                continue                # dropped with the oldest, or changed: written anew
            taken.add(first)
            last = first + count - 1
            self._state.kept.append(KeptTurn(tuple(turn["tokens"]), messages[first],
                                             messages[last], bool(messages[last].get("tool_calls"))))
        self._changes.n_turns_restored = len(self._state.kept)
        self._state.kept.sort(key=lambda k: next(i for i, m in enumerate(messages) if m is k.first))
        self._drop_unplaceable()

    def _log_begin(self, n_given: int, generated_turns: dict | None) -> None:
        """Say once what ``begin`` left out of a history of ``n_given`` items, or of the generated turns it was given; nothing where it kept all of them as given. See design_history.md, "One report of what an action changed"."""
        changes, said = self._changes, []
        if changes.n_kept_messages < n_given:
            why = {"half_context": f"so that half of the {self._config.context_size}-token "
                                   "context stays free",
                   "exchange_limit": "to keep max_history_exchanges="
                                     f"{self._config.max_history_exchanges}",
                   "room": "to fit beside the system prompt"}[changes.evicted_for or "room"]
            said.append(f"left out the {n_given - changes.n_kept_messages} oldest {why}")
        if changes.n_left_out_calls:
            said.append(f"left out {changes.n_left_out_calls} call(s) the model wrote "
                        "unreadably, with their error results")
        if changes.generated_turns_ignored:
            said.append("the generated turns given are another model's, so every turn is "
                        "written this model's way")
        elif generated_turns and changes.n_turns_restored < len(generated_turns.get("turns", ())):
            said.append(f"restored {changes.n_turns_restored} of "
                        f"{len(generated_turns['turns'])} generated turns exactly, the others "
                        "being no longer among the messages as they were")
        if said:
            _log.info("begin kept %d of %d history messages; %s", changes.n_kept_messages,
                      n_given, "; ".join(said))

    def _leave_out_unreadable(self) -> None:
        """Leave out each call the model wrote unreadably that no kept turn holds as generated, with its error result, joining the replies on either side of it."""
        held = {id(m) for k in self._state.kept for m in (k.first, k.last)}
        kept, gone, joins = [], set(), False
        for message in self._state.messages:
            calls = message.get("tool_calls") or ()
            if id(message) not in held and any(UNREADABLE_CALL in c for c in calls):
                self._changes.n_left_out_calls += sum(UNREADABLE_CALL in c for c in calls)
                gone |= {c["id"] for c in calls if UNREADABLE_CALL in c}
                left = [c for c in calls if UNREADABLE_CALL not in c]
                message = ({**message, "tool_calls": left} if left
                           else {k: v for k, v in message.items() if k != "tool_calls"})
                joins = not left            # the reply after the call goes on with this one
            elif message["role"] == "tool" and message["tool_call_id"] in gone:
                continue
            elif joins and message["role"] == "assistant" and not calls:
                before = kept.pop()
                message = {**message, "content": "\n\n".join(
                    text for text in (before["content"], message["content"]) if text)}
                joins = False
            else:
                joins = False
            kept.append(message)
        self._state.messages = kept

    def _settled_while_owed(self) -> Target:
        """The render of the conversation as far as the results still owed cannot change it: what the cache holds until they are in."""
        standing = self._rendering.target(self._state.messages, generation_prompt=False)
        answered = self._rendering.target(
            self._state.messages + [result_message(call.name, call_id, STAND_IN)
                                    for call, call_id in _unanswered(self._state.messages)],
            generation_prompt=False)
        return Target(standing.tokens[:common_prefix_length(standing.tokens, answered.tokens)], {})

    def _refused_history(self, messages: list[dict]) -> TemplateRefusedError:
        """The refusal of ``messages``, a history this model's template will not write."""
        roles = [m["role"] for m in messages]
        return TemplateRefusedError("messages", self._refusal("user") if ["user", "user"] in [
            roles[i:i + 2] for i in range(len(roles) - 1)]
            else "this model's template refuses to write this conversation")

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
    def ask(self, text: str, cancel: threading.Event | None = None) -> int:
        """Decode ``text`` as the next question, with the generation prompt after it; how many history messages were dropped for it.

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
            return self._ask(text)
        finally:
            self._memory.cancel = None

    def _ask(self, text: str) -> int:
        """:meth:`ask`'s work."""
        with self._action():
            _ensure_begun(self, "generate into")
            room = self._max_prompt_tokens
            question = {"role": "user", "content": text}
            alone = self._length_alone(question, room, generation_prompt=True)
            if alone > room:
                raise ContextOverflowError(
                    f"the request needs {alone} tokens but {room} fit beside room for the "
                    f"reply; shorten the request or the system prompt, or {_RAISE_THE_ROOM}",
                    alone, room)
            joined = self._make_way_for(question)
            n_before = history_count(self._state.messages) + (joined is not None)
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
                    f"free; shorten the request, or {_RAISE_THE_ROOM}",
                    self._config.min_reply_tokens, free)
            if self._memory.cancel is not None:
                # Decoded now, in the parts a cancel is read between, not deferred to the
                # generation; one set by the end takes the whole question back.
                self._memory.repair()
                if self._memory.cancel.is_set():
                    raise Cancelled("cancelled while the question was decoded")
            return n_before + 1 - history_count(self._state.messages)

    # ----- what an action changed beyond what was asked ------------------------------------
    def report(self, n_evicted_messages: int) -> Report:
        """The changes since the caller last asked, the actions having evicted ``n_evicted_messages`` history messages, and a fresh start for the next."""
        changes, self._changes = self._changes, Changes()
        return Report(n_evicted_messages=n_evicted_messages,
                      evicted_for=changes.evicted_for if n_evicted_messages else None,
                      n_kept_messages=changes.n_kept_messages,
                      n_left_out_calls=changes.n_left_out_calls,
                      n_turns_restored=changes.n_turns_restored,
                      generated_turns_ignored=changes.generated_turns_ignored,
                      tool_results_cut=tuple(changes.tool_results_cut),
                      turn_closed=changes.turn_closed,
                      question_joined=changes.question_joined)

    def changes_so_far(self) -> Changes:
        """What the actions changed since the caller last asked for a :meth:`report`, left to be asked for still."""
        return replace(self._changes, tool_results_cut=list(self._changes.tool_results_cut))

    # ----- nothing blocks a question ------------------------------------------------------
    def _make_way_for(self, question: dict) -> dict | None:
        """Close a turn a call left open, and join ``question`` to an unanswered one the template writes no second question after; the joined question, taken out to be added again, or ``None``.

        See design_history.md, "Nothing blocks a question".
        """
        if self.turn_open:
            self._close_open_turn(question)
        newest = self._state.messages[-1] if self._state.messages else None
        if newest is None or newest["role"] != "user" or self._writes_after(question):
            return None
        self._state.messages.pop()
        self._changes.question_joined = True
        _log.info("a question was joined to the one before it, which no reply followed: "
                  "the template writes no two in a row")
        return {**newest, "content": f"{newest['content']}\n\n{question['content']}"}

    def _close_open_turn(self, question: dict) -> None:
        """Close the open turn ahead of ``question``: each call still owed is told it was not run, and a turn waiting for its reply gets an empty one where the template needs it; the turn is taken out where the template writes neither."""
        start = len(self._state.messages)
        owed = _unanswered(self._state.messages)
        self._state.messages += [result_message(call.name, call_id, result_text(_NOT_RUN), _NOT_RUN)
                                 for call, call_id in owed]
        self._state.results_cut, self._state.called_in_thought = [], False
        self._changes.turn_closed = True
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
        gone = self._state.messages[asked + 1:]
        del self._state.messages[asked + 1:]
        self._state.kept = [k for k in self._state.kept
                            if not any(k.last is m or k.first is m for m in gone)]
        self._state.kept_at = {k: at for k, at in self._state.kept_at.items()
                               if k in self._state.kept}

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
        self._state = snapshot.state.copied()
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
                or not _is_plain(text, self._context)
                or (not self._state.repeats_alike
                    and any(m["content"] == text for m in self._state.messages))
                or not question_splices(self._rendering.render, self._system_part(), text,
                                        self._state.splice_probe)):
            return False
        rest = [(text if own and piece == STAND_IN else piece, own) for piece, own in ahead.rest]
        tokens = self._rendering.tokenize(rest)
        if len(self._memory.tokens) + len(tokens) > room:
            return False
        self._state.messages.append(question)
        self._memory.decode(tokens, want_logits=True)
        self._state.spliced_at = (len(self._memory.tokens) - len(tokens), tokens,
                                  self._made_for_now())
        self._note_prompt(ahead.after_question, ahead.reply_opening, self._memory.tokens)
        return True

    def add(self, role: str, text: str) -> int:
        """Decode a message without a reply after it; returns how many history messages were dropped for it.

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
                    f"system prompt; shorten the message or the system prompt, or {_RAISE_THE_ROOM}",
                    alone, limit)
            joined = self._make_way_for(message) if role == "user" else None
            n_before = history_count(self._state.messages) + (joined is not None)
            if joined is None:
                self._ensure_writes(role, text, generation_prompt=False)
            room = self._max_prompt_tokens
            fits = (self._config.max_history_exchanges is None
                    or exchange_count(self._state.messages) < self._config.max_history_exchanges)
            if role != "user" or joined is not None or not fits or not self._splice(message, room):
                self._state.messages.append(joined or message)
                dropped = self._drop_while_over(limit, generation_prompt=False)
                self._ensure_within(limit, "shorten the message or inject a user message first")
                self._sync(self._rendering.target(self._state.messages, generation_prompt=False),
                           logits=False, dropped=dropped)
            self._memory.rest()
            return n_before + 1 - history_count(self._state.messages)

    def _length_alone(self, message: dict, limit: int, generation_prompt: bool) -> int:
        """How many tokens ``message`` takes beside the system turn alone, measured in their render only where its count comes within :data:`_NEAR_THE_ROOM_TOKENS` of ``limit``."""
        counted = self._state.with_a_short_question + len(
            self._context.tokenize(message["content"], add_special=False, parse_special=False))
        if counted + _NEAR_THE_ROOM_TOKENS <= limit:
            return counted
        return self._rendering.length(self._system_part() + [message], generation_prompt)

    def prompt(self) -> int:
        """Decode the generation prompt after the newest message, a user message, for a reply to it; how many history messages were dropped to leave room for the reply.

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
                return 0        # spliced with the generation prompt after it, within the room
            n_before = history_count(self._state.messages)
            dropped = self._drop_while_over(self._max_prompt_tokens, generation_prompt=True,
                                            tokens_only=True)
            self._sync(self._fitting_prompt(), dropped=dropped)
            return n_before - history_count(self._state.messages)

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
                f"for the reply; {_RAISE_THE_ROOM}", len(target.tokens), room)
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
                - self.template.n_closing_tokens)
        return self._memory.generate(max(0, min(max_generated_tokens, room)), generated, **options)

    def record_reply(self, answer: str, thought: str, calls: list[ToolCall],
                     called: bool, unreadable_result: dict, stopped: bool = False,
                     in_thought: bool = False) -> int:
        """Record what the newest generation produced, then settle the cache after it.

        A turn that called tools keeps the tokens it was generated as, and stays
        open while a call is owed a result. A call that could not be read is
        answered with an error result, so the turn goes on to say so, or, where
        that result has no room, the turn closes as a reply. Any other reply
        closes the turn: one the template refuses to write, or that produced
        nothing it writes, is taken back with its request, the newest question
        and everything after it;
        one with no answer, after a question, is taken out again where the
        template writes two questions in a row; and the cache is then brought
        to the template's render and decoded ahead for the next question.

        An answer holding the template's own thought tags is kept as a part
        of it the template writes as text, and as an answer with no text where
        it writes none, which the rules above then take.

        Where the generation failed, what it said is recorded, or, where it
        said nothing the template writes, the attempt is put back; the cache is
        left to be decoded again by the next action. A reply taken back with
        the question its request asked puts back what that question evicted
        too. Recording the reply ends the attempt :meth:`open_attempt` opened;
        where that raises, the caller puts the attempt back, and a fault after
        it keeps the reply.

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

        Returns:
            How many history messages were dropped.
        """
        if not self._memory.is_known and not answer and not called:
            # It failed before it said anything the template writes: nothing changes.
            self.roll_back()
            return 0
        try:
            recorded, unplaced = self._append_reply(answer, thought, calls, called, stopped)
            self._state.called_in_thought = called and in_thought
            if recorded != answer:
                self._changes.kept_text = recorded
                _log.info("a reply's answer holds the template's thought tags; the part the "
                          "template writes is kept: %r", recorded)
            try:
                return self._after_reply(recorded, called, called and not calls, unplaced,
                                         recorded != answer, unreadable_result)
            finally:
                self._attempt = None        # recorded: a fault after it keeps the reply
        finally:
            self._publish()

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
            tags = self.template.reasoning
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
            self._state.n_calls += len(asked)
        # A reply with neither text nor call has nothing to place its end by, and one a stop
        # string ended holds what was generated after it: a turn kept so far stays kept up to
        # its call, and the rest is the template's.
        if (called or resumed) and (answer or asked) and not stopped:
            first, open_turn = message, self._open_kept() if resumed else None
            if open_turn is not None:
                # The whole turn is kept, its results and their prompt included.
                self._state.kept.remove(open_turn)
                start, first = min(start, self._state.kept_at.pop(open_turn)), open_turn.first
            self._state.kept.append(KeptTurn(tuple(self._memory.tokens[start:]), first, message,
                                             bool(asked)))
            self._state.kept_at[self._state.kept[-1]] = start
        self._state.messages.append(message)
        # A reply that leaves every turn that called tools as it was cannot move one.
        moves_kept = called or resumed or not self._state.tool_turns_stay
        return answer, moves_kept and self._drop_unplaceable()

    def _after_reply(self, answer: str, called: bool, unreadable: bool, unplaced: bool,
                     rewritten: bool, unreadable_result: dict) -> int:
        """Answer a call that could not be read, or close the turn and settle the cache; how many history messages were dropped.

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
            return 0
        self._unreadable_result = unreadable_result
        if unreadable:
            try:
                written = self._write_unreadable(unreadable_result)
            except BaseException:
                self._forget_calls()        # the cache is decoded again by the next action
                raise
            if not written:
                self._forget_calls()
        elif self.turn_open:
            self._answer_unreadable_owed()  # an entry of a batch that could not be read
        if self.turn_open:
            if unplaced:
                self._sync(self._rendering.target(self._state.messages, generation_prompt=False),
                           logits=False)
            return 0
        # A reply whose call was withdrawn, for want of room for its answer, is a reply too.
        replied = not self._state.messages[-1].get("tool_calls")
        if replied and not self._writes(self._state.messages if not answer
                                        else self._newest_exchange()):
            self._changes.taken_back = True
            _log.info("a reply the template does not write, nor an empty one in its place, "
                      "was taken back with its request")
            if self._request is not None:
                # Taken back with the question this request asked, and what it evicted.
                self._go_back_to(self._request)
                return 0
            starts = exchange_starts(self._state.messages)
            return self._remove_from(starts[-1] if starts else len(self._system_part()))
        if not answer and not called and not resumed \
                and self.template.properties.writes_two_user_turns:
            # A reply all thought or cut before its answer leaves the question
            # unanswered, as the next question then follows it.
            return self._remove_from(len(self._state.messages) - 1)
        return self._settle()

    def _holds_thought_tags(self, text: str) -> bool:
        """Whether ``text`` holds the template's thought opening or closing, which a template may cut a past reply at."""
        tags = self.template.reasoning
        return tags is not None and any(tag.strip() and tag.strip() in text
                                        for tag in (tags.thought_open, tags.thought_close))

    def _open_kept(self) -> KeptTurn | None:
        """The kept turn the open reply goes on with: the one ending on its newest call, placed in the cache."""
        calling = next((m for m in reversed(self._state.messages) if m.get("tool_calls")), None)
        return next((k for k in self._state.kept if k.last is calling and k.on_call
                     and k in self._state.kept_at), None)

    def _write_unreadable(self, result: dict) -> bool:
        """Answer a call that could not be read with ``result``, where the template writes a result and the context has room; whether it was."""
        call = self.pending_calls[0]
        if not self._renders_result(result_message(call.name or UNKNOWN_FUNCTION, call.id,
                                                   result_text(result))):
            return False
        try:
            self._add_result(call, result)
            return True
        except ContextOverflowError:
            return False

    def _answer_unreadable_owed(self) -> int:
        """Answer each call still owed that could not be read, while one is next in order, with the failure the model is told; how many history messages went for room."""
        dropped = 0
        while self.pending_calls and self.pending_calls[0].status == "unreadable" \
                and self._unreadable_result is not None:
            n_before = history_count(self._state.messages)
            if not self._write_unreadable(self._unreadable_result):
                break
            dropped += max(0, n_before - history_count(self._state.messages))
        return dropped

    def _renders_result(self, message: dict) -> bool:
        """Whether the template writes ``message``, a result, after the conversation.

        Read from the newest exchange, the turn the result goes in, as a
        question's refusal is, not from a render of the whole conversation.
        """
        # A history with no question has no exchange to stand for it: it is read whole.
        newest = (self._newest_exchange() if exchange_starts(self._state.messages)
                  else self._state.messages)
        return self._rendering.render(newest + [message], False) not in (
            None, self._rendering.render(newest, False))

    def _forget_calls(self) -> None:
        """Record the newest reply without the call it made, which could be told nothing; its tokens stay as generated."""
        newest = {k: v for k, v in self._state.messages[-1].items() if k != "tool_calls"}
        old = self._state.messages[-1]
        self._state.messages[-1] = newest
        self._state.kept = [KeptTurn(k.tokens, newest if k.first is old else k.first, newest, False)
                            if k.last is old else k for k in self._state.kept]
        self._state.kept_at = {}
        self._drop_unplaceable()

    def _drop_unplaceable(self) -> bool:
        """Hold each kept turn the template writes where it cannot be placed as the template writes it instead; whether any was."""
        messages, kept = self._state.messages, self._state.kept
        if not kept:
            return False
        for generation_prompt in (False, True):
            split = self._rendering.split(messages, generation_prompt)
            if split is not None:
                kept = [k for k in kept if self._rendering.place(k, messages, split) is not None]
        dropped = len(kept) < len(self._state.kept)
        self._state.kept = kept
        self._state.kept_at = {k: at for k, at in self._state.kept_at.items() if k in kept}
        return dropped

    def close_unanswered(self) -> int:
        """Close the open turn where no room is left to answer its newest calls, as the reply cap closes one; how many history messages were dropped.

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
            self._changes.turn_closed = True
            if not self._state.messages[-1]["content"] and not self._writes(self._state.messages):
                # Nothing the template writes is left of the reply: taken back with its request.
                self._changes.taken_back = True
                _log.info("no room was left to answer the newest calls, and nothing of the "
                          "reply is left to write: taken back with its request")
                starts = exchange_starts(self._state.messages)
                return self._remove_from(starts[-1] if starts else len(self._system_part()))
            _log.info("no room was left to answer the newest calls: the turn closed on what "
                      "the model wrote")
            return self._settle()

    def add_result(self, call: ToolCall, result) -> int:
        """Record what the oldest owed call returned, then, after the last one, decode the prompt for the reply; how many history messages were dropped for it.

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
        dropped = self._add_result(call, result)
        return dropped + self._answer_unreadable_owed()

    def _add_result(self, call: ToolCall, result) -> int:
        """:meth:`add_result`'s work for one call."""
        with self._action():
            if not self._state.begun:
                raise OutOfOrderError(None, "no conversation to answer a call in; call begin() first")
            waiting = _unanswered(self._state.messages)
            if not waiting:
                raise OutOfOrderError("call", "no tool call is waiting for a result")
            owed, owed_id = waiting[0]
            if replace(call, id=call.id or owed.id) != owed:
                raise OutOfOrderError(
                    "call", f"the waiting call is {owed!r}, not {call!r}; results "
                    "are owed in the order the model asked for them")
            text = result_text(result)
            message = result_message(owed.name or UNKNOWN_FUNCTION, owed_id, text, result)
            if not self._renders_result(message):
                raise UnsupportedError("result", "this model's template renders no tool result, "
                                                 "so a call's result cannot be written")
            room = self._max_prompt_tokens
            n_before = history_count(self._state.messages)
            # Each result still owed after this one is held empty, so that this one is
            # written whole only where those still fit after it.
            self._state.messages += [message] + [result_message(later.name, later_id, "")
                                                 for later, later_id in waiting[1:]]
            at = len(self._state.messages) - len(waiting)
            dropped = False
            fits = functools.partial(self._cutting.fits, self._state)
            if not fits(at, text, room):
                # Held cut to nothing instead, so that what this one is cut to leaves room
                # for each of them, its note included.
                del self._state.messages[at + 1:]
                self._state.messages += [result_message(later.name, later_id, cut_result(text, 0))
                                         for later, later_id in waiting[1:]]
                dropped = self._drop_while_over(room, generation_prompt=True, tokens_only=True,
                                                beside_open_turn=True)
                at = len(self._state.messages) - len(waiting)
                if not fits(at, text, room):
                    if text and not fits(at, cut_result(text, 0), room):
                        del self._state.messages[at + 1:]       # this one first, if only it fits
                    if text and not fits(at, cut_result(text, 0), room):
                        earlier = self._cutting.cut_an_earlier_result(
                            self._state, at, cut_result(text, 0), room)
                        if earlier is not None:
                            self._note_cut(*earlier, "an earlier result", "to make room")
                    if not text or not fits(at, cut_result(text, 0), room):
                        raise ContextOverflowError(
                            "not even the result cut to nothing leaves room for the reply in "
                            f"what is left of the context; {_RAISE_THE_ROOM}")
                    kept = self._cutting.longest_fitting(self._state, at, text, room)
                    self._state.messages[at] = with_result_text(message, cut_result(text, kept))
                    self._note_cut(owed, kept, len(text), "the result", "to fit")
            del self._state.messages[at + 1:]
            if not self.pending_calls:
                self._sync(self._fitting_prompt(), dropped=dropped)
            elif dropped:
                self._sync(self._rendering.target(self._state.messages, generation_prompt=False),
                           logits=False, dropped=True)
            if dropped:
                self._memory.repair()           # with the logits the reply needs, then saved
                self._memory.rest()
            return n_before - history_count(self._state.messages)

    def _note_cut(self, call: ToolCall, kept: int, total: int, what: str, why: str) -> None:
        """Note that ``call``'s result was cut to ``kept`` of its ``total`` characters."""
        self._state.results_cut.append(call)
        self._changes.tool_results_cut.append(call)
        _log.info("%s of %s was cut to %d of %d characters %s", what, call.name, kept, total, why)

    # ----- changing what it holds -----------------------------------------------------
    def truncate_newest(self, keep_text: str) -> None:
        """Shorten the newest reply's text to ``keep_text``.

        Raises:
            ValueError: If no conversation was begun, a reply is open, the
                newest message is not a reply or holds a tool call, one that
                could not be read included, ``keep_text`` is not a prefix of
                its text, or the template does not write the reply shortened
                to it.
        """
        with self._action():
            _ensure_ready(self, "truncate a reply", "truncate in")
            newest = self._state.messages[-1] if self._state.messages else None
            if newest is None or newest["role"] != "assistant":
                raise OutOfOrderError(None, "the newest message is not an assistant reply")
            if any(m["role"] == "tool" or m.get("tool_calls") for m in self._state.messages[-2:]) \
                    or any(k.last is newest for k in self._state.kept):
                raise OutOfOrderError(None, "the newest reply holds a tool call, which would "
                                            "be separated from its result")
            if not newest["content"].startswith(keep_text):
                raise InvalidArgumentError("keep_text",
                                           "keep_text is not a prefix of the newest reply")
            if keep_text == newest["content"]:
                return
            self._state.messages[-1] = {**newest, "content": keep_text}
            if not self._writes(self._state.messages):
                raise TemplateRefusedError(
                    "keep_text", f"this model's template does not write the newest reply "
                    f"shortened to {keep_text!r}; keep more of its text")
            self._render_ahead()
            self._memory.rest()

    def change_settings(self, tools: list[ToolSpec] | None, reasoning: bool, accept=None) -> int:
        """Run the conversation under ``tools`` and ``reasoning`` from now on, every message kept; how many history messages were dropped.

        ``tools`` other than ``None``, even empty, writes it with the model's
        template for tools. The cache is decoded again from where the new
        render first differs. The oldest exchanges go only where the
        conversation no longer fits.

        Args:
            accept: Called once the conversation is ready and before anything
                changes; one that raises refuses the settings.

        Raises:
            OutOfOrderError: If no conversation was begun or a reply is open.
            TemplateRefusedError: If the template under them writes an older
                exchange otherwise once another follows, or a turn kept as
                generated where it cannot be told apart.
            ContextOverflowError: If the system prompt with the declarations
                leaves no room under ``evict_above_tokens`` for a short question.
        """
        with self._action():
            _ensure_ready(self, "change the settings", "change the settings of")
            if accept is not None:
                accept()
            self._state.tools, self._state.reasoning = list(tools or []), reasoning
            self._state.uses_tool_template = tools is not None
            self._rendering.forget_renders()
            self._ensure_room_for_a_question("declare fewer tools")
            limit = self._config.evict_above_tokens
            self._check_renders()
            self._state.opening = self._rendering.opening_render(self._system_part())
            self._pin_opening()
            n_before = history_count(self._state.messages)
            dropped = self._drop_while_over(limit, generation_prompt=False, newest_too=True)
            self._render_ahead(dropped=dropped)
            self._memory.rest()
            return n_before - history_count(self._state.messages)

    def _pin_opening(self) -> None:
        """Keep a state saved where the conversation's opening ends, where the cache goes back to saved states."""
        self._memory.pin(self._rendering.opening_length(self._system_part())
                         if self._memory.keeps_saved_states else None)

    def _check_renders(self) -> None:
        """Check again what the template's renders allow, as the system turn, the declarations and the clock now leave them.

        Raises:
            ValueError: If the template writes an older exchange otherwise once
                another follows.
        """
        system = self._system_part()
        ensure_history_stays_written(self._rendering.render, system)
        # Declarations that move with the newest question leave no window as the whole render writes it.
        self._state.windows = (not self._moves_declarations()
                               and windows_are_exact(self._rendering.render, system,
                                                     self._rendering.tag_from))
        self._state.splice_probe = splice_probe(self._rendering.render, system)
        self._state.repeats_alike = repeated_question_written_alike(self._rendering.render, system)
        self._state.with_a_short_question = self._eviction.with_a_short_question(self._state)
        # Without the steps tool calling needs, no turn keeps a call, and the template is not asked.
        self._state.tool_turns_stay = self.template.can_call_tools and tool_turns_stay_written(
            self._rendering.render, system, self._state.tools[0].name if self._state.tools else "probe",
            thought=self._state.reasoning)

    def _ensure_writes(self, role: str, text: str, generation_prompt: bool) -> None:
        """Refuse a ``role`` message the template would not write next.

        Raises:
            ValueError: If the template refuses the conversation with it.
        """
        # Refusals turn on the newest messages, such as two user turns in a row, so
        # they are read from the newest exchange, not from a render of the whole conversation.
        if self._rendering.render(self._newest_exchange() + [{"role": role, "content": text}],
                                  generation_prompt) is None:
            raise TemplateRefusedError("text", self._refusal(role))

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
        limit, needed = self._config.evict_above_tokens, self._eviction.with_a_short_question(self._state)
        if needed > limit:
            raise ContextOverflowError(
                f"the system prompt and its declarations need {needed} tokens with a short "
                f"question but the eviction threshold is {limit}; {remedy}, or {_RAISE_THE_ROOM}",
                needed, limit)

    def _ensure_within(self, limit: int, remedy: str) -> None:
        """Refuse a conversation that still renders longer than ``limit`` once every exchange that can go was dropped.

        What is left over then has no user message after it, and an eviction
        drops only up to the next one.

        Raises:
            ContextOverflowError: With ``remedy`` as what to do instead.
        """
        needed = self._rendering.length(self._state.messages, generation_prompt=False)
        if needed > limit:
            raise ContextOverflowError(
                f"the conversation needs {needed} tokens but the eviction threshold is "
                f"{limit}, and the messages no user message follows cannot be evicted; "
                f"{remedy}, or {_RAISE_THE_ROOM}", needed, limit)

    def _remove_from(self, first: int) -> int:
        """Take the messages from ``first`` on out again, then drop what is over the limits; how many history messages were dropped."""
        with self._action():
            gone = self._state.messages[first:]
            del self._state.messages[first:]
            self._state.kept = [k for k in self._state.kept
                                if not any(k.first is m or k.last is m for m in gone)]
            n_before = history_count(self._state.messages)
            dropped = self._drop_while_over(self._config.evict_above_tokens,
                                            generation_prompt=False)
            # As after any reply, so that what the template writes ahead of the next
            # question, a default system prompt say, stays in the cache.
            self._render_ahead(dropped=dropped)
            self._memory.rest()
            return n_before - history_count(self._state.messages)

    # ----- after a reply ----------------------------------------------------------
    def _settle(self) -> int:
        """Drop what is over the limits, then render and decode ahead for the next question; how many history messages were dropped."""
        with self._action():
            n_before = history_count(self._state.messages)
            limit = self._config.evict_above_tokens
            plan = self._windows.ahead(self._state, self._memory) if self._state.windows else WHOLE
            if plan is not WHOLE:
                # Dropped first, so the cache shifts before the decode ahead, not before the next question's.
                at, before, length, split, run = plan
                held = len(self._memory.tokens)
                kept_until = lambda: at - (held - len(self._memory.tokens))   # noqa: E731
                done = self._windows.drop_in(self._state, self._memory, kept_until, length)
                # Where the window's own older exchange must go too, or what is decoded ahead
                # would not fit, the rest is rendered whole.
                done = done and not (kept_until() + length > limit or self._eviction.too_many(self._state)
                                     or kept_until() + len(before) > self._config.context_size)
                if done:
                    self._sync(Target(list(self._memory.tokens[:kept_until()]) + before,
                                      self._state.kept_at), logits=False)
                    self._note_ahead(split, run)
                else:
                    self._drop_while_over(limit, generation_prompt=False, newest_too=True)
                    self._render_ahead(dropped=True)
            else:
                dropped = self._drop_while_over(limit, generation_prompt=False, newest_too=True)
                self._render_ahead(dropped=dropped)
            self._memory.rest()
            return n_before - history_count(self._state.messages)

    def _render_ahead(self, dropped: bool = False) -> None:
        """Decode the render for the next question up to where the question goes."""
        messages = self._state.messages + [{"role": "user", "content": STAND_IN}]
        split = self._rendering.split(messages, generation_prompt=True)
        run = (self._rendering.run_start(split, len(messages) - 1, 0)
               if split is not None and split.text.count(STAND_IN) == 1 else None)
        if run is not None:
            before = self._rendering.target(messages, generation_prompt=True, upto=run)
            if len(before.tokens) <= self._config.context_size:
                self._sync(before, logits=False, dropped=dropped)
                self._note_ahead(split, run)
                return
        # The next question will be rendered when it comes.
        if self._state.messages and self._rendering.split(self._state.messages, False) is None:
            return
        self._sync(self._rendering.target(self._state.messages, generation_prompt=False),
                   logits=False, dropped=dropped)

    def _system_part(self) -> list[dict]:
        return system_part(self._state.messages)

    # ----- the cache against the target -----------------------------------------
    def _sync(self, target: Target, *, logits: bool = True, dropped: bool = False) -> None:
        """Make the cache hold ``target``, and note where its kept turns begin there.

        Args:
            logits: A generation follows the decode at once.
            dropped: Exchanges were dropped, so that the target may go on as
                the cache does after a stretch.
        """
        # Declarations that move with the newest question leave a stretch behind, and the
        # next question's move back to where this cut is made, or past it.
        moves = self._moves_declarations()
        self._memory.hold(target.tokens, want_logits=logits,
                          find_dropped_stretch=dropped or moves, rest_at_cut=moves)
        self._state.kept_at = target.kept_at

    def _moves_declarations(self) -> bool:
        """Whether the declarations move with the newest question, leaving a stretch behind."""
        return bool(self._state.tools) and self.template.declares_before_newest_turn

    def _prompted(self) -> Target:
        """The conversation with the generation prompt after it, noting the template text a reply follows."""
        target = self._rendering.target(self._state.messages, generation_prompt=True)
        split = self._rendering.split(self._state.messages, generation_prompt=True)
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
        start = self._rendering.turn_start(split, len(split.contents) - 1)
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
        cause = self._eviction.drop_while_over(self._state, limit, generation_prompt, **options)
        if cause is not None:
            self._changes.evicted_for = cause
            _log.debug("the oldest history went, for %s", cause.replace("_", " "))
        return cause is not None

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
                        state.opening, self._memory.version)

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
        if self._state.begun and self.template.reads_clock:
            opening = self._rendering.opening_render(self._system_part())
            if opening != self._state.opening:
                # What was rendered ahead holds the old date, say; it is rendered whole now.
                self._check_renders()
                self._state.opening = opening
                self._pin_opening()
        saved = self._state.copied()
        held = list(self._memory.tokens)
        try:
            yield
        except BaseException:
            self._state = saved
            self._memory.replace(held)
            raise
        finally:
            self._publish()

    def _publish(self) -> None:
        """Publish the conversation for readers that hold no lock."""
        self._published = published(self._state, self.pending_calls, self.turn_open,
                                    len(self._memory.tokens))


# ----- what an action may start from -------------------------------------------------
def _ensure_begun(conversation, into: str) -> None:
    """Refuse an action before a conversation was begun, which alone holds the sequence opener and the system prompt."""
    if not conversation.begun:
        raise OutOfOrderError(None, f"no conversation to {into}; call begin() first")


def _ensure_ready(conversation, action: str, into: str | None = None) -> None:
    """Refuse an action before a conversation was begun, or inside a turn a call left open.

    ``conversation`` is a :class:`RenderedCache` or what it published. Only a
    begun conversation holds the sequence opener and the system prompt, and a
    message inside a model turn that never ended is a shape no template renders
    and no model was trained on.
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


# What a call still owed is told when a question closes its turn.
_NOT_RUN = {"error": "not_run", "message": "the call was not run: the user asked something new"}

# The one wording of what makes more room, at the end of every ContextOverflowError.
_RAISE_THE_ROOM = "raise context_size or evict_above_fraction, or lower min_reply_tokens"


# ----- helpers ---------------------------------------------------------------------
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


def _is_plain(text: str, reader) -> bool:
    """A question the stand-in may stand for: words without tags or surrounding space."""
    if not text or text != text.strip() or STAND_IN in text:
        return False
    split = split_at_tags(reader, text)
    return split is not None and not any(is_tag for _, is_tag in split)
