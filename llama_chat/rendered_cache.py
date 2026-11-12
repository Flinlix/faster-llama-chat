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
from dataclasses import dataclass, field, fields, replace

from .cache_context import GenerationAccumulator
from .cache_sync import CacheSync
from .config import Config
from .messages import (as_history, as_messages, calling_message, exchange_count, exchange_starts,
                       kept_items, next_call_number, result_message, system_part)
from .render_checks import (STAND_IN, ensure_history_stays_written, question_splices,
                            windows_are_exact, writes_two_user_turns)
from .rendering import ConversationRendering, KeptTurn, Target
from .reply import ContextOverflowError
from .resolved_template import ResolvedTemplate, ResolvedTemplates
from .split_render import SplitRender
from .template import common_prefix_length
from .tokenizing import split_at_tags
from .tools import ToolCall, ToolSpec, result_text

# The name a result carries where no real one is to be had: answering a call
# that could not be read that far. Which name is used does not change what a
# template renders around it.
UNKNOWN_FUNCTION = "unknown"
# Returned where a window cannot stand for the whole render.
_WHOLE = object()
# Where the cache cannot shift, the fraction of the room under the eviction
# threshold, after a short question, that an eviction brings the conversation
# down to. In (0, 1]: low enough that a cache decoded again after an eviction
# takes several turns to fill again, high enough to keep history. See
# design_history.md, "Settings, add-on files and logging", 266.3.
EVICT_TO_FRACTION_WITHOUT_SHIFT = 0.5


@dataclass(frozen=True, eq=False)
class _MadeFor:
    """The conversation and the cache a record made ahead was made for, which it is used with only while they stay."""

    messages: tuple[dict, ...]
    tools: tuple[ToolSpec, ...]
    reasoning: bool
    opening: str | None
    cache_version: int


@dataclass
class _Ahead:
    """The render for the next question, decoded up to the run of text the question joins."""

    rest: list[tuple[str, bool]]          # from that run on, the stand-in as the caller's own
    after_question: str                   # the template's text after the question
    reply_opening: list[tuple[str, bool]]  # the render from where the reply's kept turn begins
    made_for: _MadeFor


@dataclass
class _ConversationState:
    """What a failed action puts back as it was, the cache's tokens besides."""

    messages: list[dict] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    # Whether it is written with the model's template for tools: begun with tools, or
    # declaring some. True wherever ``tools`` holds any.
    uses_tool_template: bool = False
    reasoning: bool = False
    kept: list[KeptTurn] = field(default_factory=list)
    # Where each placed kept turn starts in the cache.
    kept_at: dict[KeptTurn, int] = field(default_factory=dict)
    n_calls: int = 0
    windows: bool = False                 # whether windows of the conversation stand for its render
    ahead: _Ahead | None = None
    prompt_text: str = ""                 # the template text the next generation follows
    prompt_start: int = 0                 # where in the cache the generation prompt begins
    # Where a spliced question's run begins, the tokens decoded there, and what they were
    # decoded for: the conversation with the question newest, and the cache after them.
    spliced_at: tuple[int, list[int], _MadeFor] | None = None
    # The tokens before the question the newest ask added, and the conversation and the
    # cache that ask left, until a generation follows.
    asked: tuple[list[int], _MadeFor] | None = None
    begun: bool = False
    opening: str | None = None            # how the conversation opens, to see a clock move on

    def copied(self) -> _ConversationState:
        """This state as it is now, each list and dict copied so that later changes leave it alone."""
        return _ConversationState(**{f.name: _copied(getattr(self, f.name)) for f in fields(self)})


@dataclass(frozen=True)
class _Published:
    """The conversation as the newest finished change left it, for readers that hold no lock."""

    system_prompt: str = ""
    history: tuple = ()
    tools: tuple | None = None            # None where it is not written with the template for tools
    reasoning: bool = False
    pending_calls: tuple = ()
    begun: bool = False
    turn_open: bool = False
    newest_role: str | None = None
    total_tokens: int = 0


class RenderedCache:
    """One conversation's messages, as its template is given them, and the cache holding their render.

    Every method expects the caller to hold the conversation's lock, except
    :meth:`conversation`, :meth:`published_pending_calls`,
    :meth:`published_total_tokens`, :meth:`published_turn_open` and
    :meth:`ensure_can_reply`, which read a copy published after each change.
    Each action leaves the cache equal to the template's render of the
    messages, a turn that called tools held as it was generated, or, where it
    raises, the conversation and the cache as they were before it.
    """

    def __init__(self, context, templates: ResolvedTemplates, config: Config) -> None:
        self._context = context
        self._templates = templates
        self._config = config
        self._memory = CacheSync(context)
        self._state = _ConversationState()
        self._rendering = ConversationRendering(context, lambda: self.template,
                                                lambda: self._state)
        # Left as they are by a failed action. The newest generation: where it began,
        # whether it resumed a turn, and the tokens before the question it answers where
        # its request asked one.
        self._generation: tuple[int, bool, list[int] | None] = (0, False, None)
        self._published = _Published()

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
                "reasoning": published.reasoning}

    def published_pending_calls(self) -> list[ToolCall]:
        """The calls still owed a result, oldest first, from the newest publication, the caller's to change; safe to read unlocked."""
        return copy.deepcopy(list(self._published.pending_calls))

    def published_total_tokens(self) -> int:
        """How many tokens the cache holds, from the newest publication; safe to read unlocked."""
        return self._published.total_tokens

    def published_turn_open(self) -> bool:
        """Whether the newest turn is open, from the newest publication; safe to read unlocked."""
        return self._published.turn_open

    def published_template(self) -> ResolvedTemplate:
        """The template the conversation is written with, from the newest publication; safe to read unlocked."""
        return self._templates.for_conversation(self._published.tools is not None)

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
    def prompt_text(self) -> str:
        """The template text the next generation follows, which says whether it opens in a thought."""
        return self._state.prompt_text

    # ----- beginning ------------------------------------------------------------------
    def start(self, system_prompt: str, history, tools: list[ToolSpec] | None,
              reasoning: bool) -> int:
        """Begin again from ``system_prompt`` and the newest of ``history`` that fits in half the context; how many items were kept.

        A history that ends on a call keeps it owed a result. ``tools`` other
        than ``None``, even empty, writes the conversation with the model's
        template for tools.

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
            limit = self._config.evict_above_tokens
            self._ensure_room_for_a_question("shorten the prompt or declare fewer tools")
            asks = any(m["role"] == "user" for m in self._state.messages)
            self._keep_newest_history(min(limit, self._config.context_size // 2))
            if any(m["role"] != "system" for m in self._state.messages) \
                    and self._rendering.split(self._state.messages, self._ends_on_result()) is None:
                raise self._refused_history(self._state.messages)
            self._check_renders()
            if asks and (not any(m["role"] == "user" for m in self._state.messages)
                         or self._rendering.length(self._state.messages, False) > limit):
                raise ContextOverflowError(
                    "the newest exchange of the history, from its user message on, does not "
                    "fit beside the system prompt; shorten it or raise "
                    "context_size/evict_above_fraction")
            self._drop_while_over(limit, generation_prompt=False)
            self._ensure_within(limit, "shorten the history")
            self._state.begun = False
            self._memory.reset()
            # The records made ahead for the conversation before, freed with it.
            self._state.ahead, self._state.spliced_at, self._state.asked = None, None, None
            self._state.opening = self._rendering.opening_render(self._system_part())
            self._memory.pin(self._rendering.opening_length(self._system_part()))
            if self._state.messages and self._state.messages[-1]["role"] in ("user", "tool"):
                self._sync(self._prompted())
            elif self.pending_calls:
                self._sync(self._rendering.target(self._state.messages, generation_prompt=False))
            else:
                self._render_ahead()
            self._memory.rest()
            self._state.begun = True
            return kept_items(history, len(self._state.messages) - len(self._system_part()))

    def _keep_newest_history(self, bound: int) -> None:
        """Keep of a history to begin with the newest exchanges that render within ``bound``, and at least the newest.

        Only the newest messages are rendered: twice as many exchanges each
        time until they no longer fit, then the count between is halved, so a
        history far longer than the context costs a few renders of about
        ``bound`` each.

        Raises:
            ValueError: If the template refuses to write what would be kept.
        """
        first = len(self._system_part())
        if len(self._state.messages) == first:
            return
        opens = [first] + [i for i in exchange_starts(self._state.messages) if i > first]

        def fits(count: int) -> bool:
            kept = self._state.messages[:first] + self._state.messages[opens[-count]:]
            if self._rendering.split(kept, kept[-1]["role"] == "tool") is None:
                raise self._refused_history(kept)
            return self._rendering.length(kept, generation_prompt=False) <= bound

        kept, more = 1, 2
        while more < len(opens) and fits(more):
            kept, more = more, more * 2
        if more >= len(opens):
            if fits(len(opens)):
                return
            more = len(opens)
        while kept < more - 1:
            middle = (kept + more) // 2
            kept, more = (middle, more) if fits(middle) else (kept, middle)
        self._state.messages = self._state.messages[:first] + self._state.messages[opens[-kept]:]

    def _refused_history(self, messages: list[dict]) -> ValueError:
        """The refusal of ``messages``, a history this model's template will not write."""
        roles = [m["role"] for m in messages]
        return ValueError(self._refusal("user") if ["user", "user"] in [
            roles[i:i + 2] for i in range(len(roles) - 1)]
            else "this model's template refuses to write this conversation")

    def _writes(self, messages: list[dict]) -> bool:
        """Whether the template writes ``messages``, as they are or with a question after them."""
        return (self._rendering.render(messages, False) is not None
                or self._rendering.render(messages + [{"role": "user", "content": STAND_IN}],
                                          True) is not None)

    def _ends_on_result(self) -> bool:
        return bool(self._state.messages) and self._state.messages[-1]["role"] == "tool"

    # ----- adding messages --------------------------------------------------------------
    def ask(self, text: str) -> int:
        """Decode ``text`` as the next question, with the generation prompt after it; how many history items were dropped for it.

        Raises:
            ValueError: If no conversation was begun, a reply is open, or the
                template does not write the question here.
            ContextOverflowError: If the question cannot fit beside room for
                its reply, checked against the system prompt alone before
                anything is dropped.
        """
        with self._action():
            _ensure_ready(self, "start a request", "generate into")
            room = self._max_prompt_tokens
            question = {"role": "user", "content": text}
            alone = self._rendering.length(self._system_part() + [question], generation_prompt=True)
            if alone > room:
                raise ContextOverflowError(
                    f"request needs {alone} tokens but only {room} fit beside room for the reply; "
                    "shorten the request or the system prompt, or raise context_size or lower "
                    "min_reply_tokens")
            self._ensure_writes("user", text, generation_prompt=True)
            before = list(self._memory.tokens)
            n_before = len(as_history(self._state.messages))
            if not self._splice(question, room):
                self._state.messages.append(question)
                # As a spliced one, the question may open one exchange more until its reply settles.
                dropped = self._drop_while_over(room, generation_prompt=True, tokens_only=True)
                self._sync(self._fitting_prompt(), dropped=dropped)
            free = self._config.context_size - len(self._memory.tokens)
            if free < self._config.min_reply_tokens:
                raise ContextOverflowError(
                    f"only {free} tokens free for the reply "
                    f"(min_reply_tokens={self._config.min_reply_tokens}); shorten "
                    "the request or lower evict_above_fraction")
            self._state.asked = before, self._made_for_now()
            return n_before + 1 - len(as_history(self._state.messages))

    def _splice(self, question: dict, room: int) -> bool:
        """Add ``question``, decoded in the stand-in's place in the render made ahead, where that is exact; whether it was."""
        ahead = self._state.ahead
        text = question["content"]
        if (ahead is None or not self._is_current(ahead.made_for)
                or not _is_plain(text, self._context)
                or not question_splices(self._rendering.render, self._system_part(), text)):
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
        """Decode a message without a reply after it; returns how many history items were dropped for it.

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
            _ensure_ready(self, "inject", "inject into")
            message = {"role": role, "content": text}
            limit = self._config.evict_above_tokens
            alone = self._rendering.length(self._system_part() + [message], generation_prompt=False)
            if alone > limit:
                raise ContextOverflowError(
                    f"injected message needs {alone} tokens but only {limit} fit beside the "
                    "system prompt; shorten the message or the system prompt, or raise "
                    "context_size/evict_above_fraction")
            self._ensure_writes(role, text, generation_prompt=False)
            n_before = len(as_history(self._state.messages))
            room = self._max_prompt_tokens
            fits = (self._config.max_history_exchanges is None
                    or exchange_count(self._state.messages) < self._config.max_history_exchanges)
            if role != "user" or not fits or not self._splice(message, room):
                self._state.messages.append(message)
                dropped = self._drop_while_over(limit, generation_prompt=False)
                self._ensure_within(limit, "shorten the message or inject a user message first")
                self._sync(self._rendering.target(self._state.messages, generation_prompt=False),
                           logits=False, dropped=dropped)
            self._memory.rest()
            return n_before + 1 - len(as_history(self._state.messages))

    def prompt(self) -> None:
        """Decode the generation prompt after the newest message, a user message, for a reply to it.

        Raises:
            ValueError: If no conversation was begun, a reply is open, or the
                newest message is not a user message.
            ContextOverflowError: If the conversation with it leaves too little
                room for a reply.
        """
        with self._action():
            _ensure_ready(self, "start a request", "generate into")
            _ensure_can_reply(self, asks_question=False)
            spliced_at = self._state.spliced_at
            if spliced_at is not None and self._is_current(spliced_at[2]):
                return          # spliced with the generation prompt after it, within the room
            self._sync(self._fitting_prompt())

    def _fitting_prompt(self) -> Target:
        """The conversation with the generation prompt after it.

        Raises:
            ContextOverflowError: If it leaves too little room for a reply.
        """
        target = self._prompted()
        room = self._max_prompt_tokens
        if len(target.tokens) > room:
            raise ContextOverflowError(
                f"the conversation needs {len(target.tokens)} tokens but only {room} leave "
                "room for the reply; lower evict_above_fraction or min_reply_tokens, so that "
                "an eviction leaves that room, or raise context_size")
        return target

    # ----- generating -------------------------------------------------------------------
    def generate(self, max_tokens: int, generated: GenerationAccumulator, **options):
        """Stream one generation of at most ``max_tokens`` from the cache's end; what it decodes is held as the cache's.

        It stops short of ``max_tokens`` where the context leaves less room
        beside the closing after the reply.

        Raises:
            ValueError: If no conversation was begun, the open turn still owes
                a result, or no turn is open and the newest message is not a
                user message.
        """
        _ensure_can_reply(self, asks_question=False)
        asked, self._state.asked = self._state.asked, None      # frees its copy of the cache
        before_question = asked[0] if asked and self._is_current(asked[1]) else None
        self._generation = (min(self._state.prompt_start, len(self._memory.tokens)),
                            self.turn_open, before_question)
        room = (self._config.context_size - len(self._memory.tokens)
                - self.template.n_closing_tokens)
        return self._memory.generate(max(0, min(max_tokens, room)), generated, **options)

    def record_reply(self, answer: str, thought: str, calls: list[ToolCall],
                     called: bool, unreadable_result: dict, stopped: bool = False) -> int:
        """Record what the newest generation produced, then settle the cache after it.

        A turn that called tools keeps the tokens it was generated as, and stays
        open while a call is owed a result. A call that could not be read is
        answered with an error result, so the turn goes on to say so, or, where
        that result has no room, the turn closes as a reply. Any other reply
        closes the turn: one that produced nothing the template writes is taken
        back with its request, the newest question and everything after it;
        one with no answer, after a question, is taken out again where the
        template writes two questions in a row; and the cache is then brought
        to the template's render and decoded ahead for the next question.

        Where the generation failed, what it said is recorded, or, where it
        said nothing, the question its request asked is taken out again; the
        cache is left to be decoded again by the next action.

        Args:
            thought: The reply's thought, given to the template as its own to
                write or leave out.
            calls: What it called, each owed a result.
            called: Whether it stopped on a call, read or not.
            unreadable_result: What a call that could not be read is answered with.
            stopped: Whether a stop string ended the answer, so that what was
                generated from it on is no reply's.

        Returns:
            How many history items were dropped.
        """
        start, resumed, before_question = self._generation
        if not self._memory.is_known and not answer and not called:
            # It failed before it said anything: the request is taken back whole.
            if before_question is not None:
                self._state.messages.pop()
                self._memory.replace(before_question)
                self._publish()
            return 0
        unreadable = called and not calls
        asked = calls or ([ToolCall(UNKNOWN_FUNCTION, {})] if unreadable else [])
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
        unplaced = self._drop_unplaceable()
        self._publish()
        if not self._memory.is_known:
            return 0                # the generation failed: the next action decodes the cache again
        if unreadable:
            try:
                written = self._write_unreadable(unreadable_result)
            except BaseException:
                self._forget_calls()        # the cache is decoded again by the next action
                raise
            if not written:
                self._forget_calls()
        if self.turn_open:
            if unplaced:
                self._sync(self._rendering.target(self._state.messages, generation_prompt=False),
                           logits=False)
            return 0
        if not answer and not called and not self._writes(self._state.messages):
            starts = exchange_starts(self._state.messages)
            return self._remove_from(starts[-1] if starts else len(self._system_part()))
        if not answer and not called and not resumed \
                and writes_two_user_turns(self._rendering.render):
            # A reply all thought or cut before its answer leaves the question
            # unanswered, as the next question then follows it.
            return self._remove_from(len(self._state.messages) - 1)
        return self._settle()

    def _open_kept(self) -> KeptTurn | None:
        """The kept turn the open reply goes on with: the one ending on its newest call, placed in the cache."""
        calling = next((m for m in reversed(self._state.messages) if m.get("tool_calls")), None)
        return next((k for k in self._state.kept if k.last is calling and k.on_call
                     and k in self._state.kept_at), None)

    def _write_unreadable(self, result: dict) -> bool:
        """Answer a call that could not be read with ``result``, where the template writes a result and the context has room; whether it was."""
        call = self.pending_calls[0]
        if not self._renders_result(result_message(call.name, call.id, result_text(result))):
            return False
        try:
            self.add_result(call, result)
            return True
        except ContextOverflowError:
            return False

    def _renders_result(self, message: dict) -> bool:
        """Whether the template writes ``message``, a result, after the conversation."""
        return self._rendering.render(self._state.messages + [message], False) not in (
            None, self._rendering.render(self._state.messages, False))

    def _forget_calls(self) -> None:
        """Record the newest reply without the call it made, which could be told nothing; its tokens stay as generated."""
        newest = {k: v for k, v in self._state.messages[-1].items() if k != "tool_calls"}
        old = self._state.messages[-1]
        self._state.messages[-1] = newest
        self._state.kept = [KeptTurn(k.tokens, newest if k.first is old else k.first, newest, False)
                            if k.last is old else k for k in self._state.kept]
        self._state.kept_at = {}
        self._drop_unplaceable()
        self._publish()

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
        """Close the open turn where no room is left to answer its newest calls, as the reply cap closes one; how many history items were dropped.

        The newest reply is kept without those calls, its tokens as generated,
        and the results already written for them are taken out again; where the
        template writes nothing of what is left of it, it is taken back with its
        request, as an empty reply is.
        """
        calling = max(i for i, m in enumerate(self._state.messages) if m.get("tool_calls"))
        del self._state.messages[calling + 1:]
        self._forget_calls()
        if not self._state.messages[-1]["content"] and not self._writes(self._state.messages):
            # Nothing the template writes is left of the reply: taken back with its request.
            starts = exchange_starts(self._state.messages)
            return self._remove_from(starts[-1] if starts else len(self._system_part()))
        return self._settle()

    def add_result(self, call: ToolCall, result) -> None:
        """Record what the oldest owed call returned, clipped where it cannot fit; after the last one, decode the prompt for the reply.

        Raises:
            ValueError: If no call is waiting, this is not the one owed, the
                result cannot be written as text, or the template writes no
                result.
            ContextOverflowError: If not even an empty result leaves room for
                the reply.
        """
        with self._action():
            if not self._state.begun:
                raise ValueError("no conversation to answer a call in; call begin() first")
            waiting = _unanswered(self._state.messages)
            if not waiting:
                raise ValueError("no tool call is waiting for a result")
            owed, owed_id = waiting[0]
            if not isinstance(call, ToolCall) or replace(call, id=call.id or owed.id) != owed:
                raise ValueError(
                    f"the waiting call is {owed!r}, not {call!r}; results "
                    "are owed in the order the model asked for them")
            text = result_text(result)
            message = result_message(owed.name, owed_id, text)
            if not self._renders_result(message):
                raise ValueError("this model's template renders no tool result, so a call's "
                                 "result cannot be written")
            room = self._max_prompt_tokens
            self._state.messages.append({**message, "content": ""})
            if len(self._prompted().tokens) > room:
                raise ContextOverflowError(
                    "the result wrapping alone leaves no room for the reply in what is left "
                    "of the context; this turn cannot be continued, start a new conversation "
                    "with begin()")
            self._state.messages[-1] = {**message, "content": self._clipped(message, text, room)}
            if not self.pending_calls:
                self._sync(self._fitting_prompt())

    def _clipped(self, message: dict, text: str, room: int) -> str:
        """The longest start of ``text`` with which the conversation, ``message`` newest, still fits in ``room``."""
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            self._state.messages[-1] = {**message, "content": text[:middle]}
            if len(self._prompted().tokens) <= room:
                low = middle
            else:
                high = middle - 1
        return text[:low]

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
                raise ValueError("the newest message is not an assistant reply")
            if any(m["role"] == "tool" or m.get("tool_calls") for m in self._state.messages[-2:]) \
                    or any(k.last is newest for k in self._state.kept):
                raise ValueError("the newest reply holds a tool call, which would "
                                 "be separated from its result")
            if not newest["content"].startswith(keep_text):
                raise ValueError("keep_text is not a prefix of the newest reply")
            if keep_text == newest["content"]:
                return
            self._state.messages[-1] = {**newest, "content": keep_text}
            if not self._writes(self._state.messages):
                raise ValueError(f"this model's template does not write the newest reply "
                                 f"shortened to {keep_text!r}; keep more of its text")
            self._render_ahead()
            self._memory.rest()

    def declare(self, specs: list[ToolSpec], accept=None) -> int:
        """Add ``specs`` where the template declares tools; how many history items were dropped.

        Args:
            accept: Called once the conversation is ready and before anything
                is written; one that raises refuses the declarations.

        Raises:
            ValueError: If no conversation was begun or a reply is open, if the
                model writes a conversation with tools in a template of its
                own and this one was begun without tools, or if the template
                with them writes an older exchange otherwise once another
                follows.
            ContextOverflowError: If the system prompt with them leaves no room
                under ``evict_above_tokens`` for a short question.
        """
        with self._action():
            _ensure_ready(self, "declare tools", "declare into")
            if not self._state.uses_tool_template and self._templates.has_template_for_tools:
                raise ValueError(
                    "this model writes a conversation with tools in a template of its own, "
                    "and this conversation was begun without tools; begin again with the "
                    "tools, begin(..., tools=...), to declare them")
            if accept is not None:
                accept()
            self._state.tools = self._state.tools + list(specs)
            self._state.uses_tool_template = True
            self._rendering.forget_renders()
            self._ensure_room_for_a_question("declare fewer tools")
            limit = self._config.evict_above_tokens
            self._check_renders()
            self._state.opening = self._rendering.opening_render(self._system_part())
            self._memory.pin(self._rendering.opening_length(self._system_part()))
            n_before = len(as_history(self._state.messages))
            dropped = self._drop_while_over(limit, generation_prompt=False, newest_too=True)
            self._render_ahead(dropped=dropped)
            self._memory.rest()
            return n_before - len(as_history(self._state.messages))

    def _check_renders(self) -> None:
        """Check again what the template's renders allow, as the system turn, the declarations and the clock now leave them.

        Raises:
            ValueError: If the template writes an older exchange otherwise once
                another follows.
        """
        system = self._system_part()
        ensure_history_stays_written(self._rendering.render, system)
        self._state.windows = windows_are_exact(self._rendering.render, system,
                                                self._rendering.tag_from)

    def _ensure_writes(self, role: str, text: str, generation_prompt: bool) -> None:
        """Refuse a ``role`` message the template would not write next.

        Raises:
            ValueError: If the template refuses the conversation with it.
        """
        # Refusals turn on the newest messages, such as two user turns in a row, so
        # they are read from the newest exchange, not from a render of the whole conversation.
        starts = exchange_starts(self._state.messages)
        newest = self._system_part() + (self._state.messages[starts[-1]:] if starts else [])
        if self._rendering.render(newest + [{"role": role, "content": text}],
                                  generation_prompt) is None:
            raise ValueError(self._refusal(role))

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
        limit, needed = self._config.evict_above_tokens, self._with_a_short_question()
        if needed > limit:
            raise ContextOverflowError(
                f"the system prompt and its declarations need {needed} tokens with a short "
                f"question but the eviction threshold is {limit}; {remedy}, or raise "
                "context_size/evict_above_fraction")

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
                f"{remedy}, or raise context_size/evict_above_fraction")

    def _with_a_short_question(self) -> int:
        """How many tokens the conversation takes with no history and a short question: what no eviction makes smaller."""
        return self._rendering.length(self._system_part() + [{"role": "user", "content": "?"}],
                                      generation_prompt=True)

    def _remove_from(self, first: int) -> int:
        """Take the messages from ``first`` on out again, then drop what is over the limits; how many history items were dropped."""
        with self._action():
            gone = self._state.messages[first:]
            del self._state.messages[first:]
            self._state.kept = [k for k in self._state.kept
                                if not any(k.first is m or k.last is m for m in gone)]
            n_before = len(as_history(self._state.messages))
            dropped = self._drop_while_over(self._config.evict_above_tokens,
                                            generation_prompt=False)
            self._sync(self._rendering.target(self._state.messages, generation_prompt=False),
                       logits=False, dropped=dropped)
            self._memory.rest()
            return n_before - len(as_history(self._state.messages))

    # ----- after a reply ----------------------------------------------------------
    def _settle(self) -> int:
        """Drop what is over the limits, then render and decode ahead for the next question; how many history items were dropped."""
        with self._action():
            n_before = len(as_history(self._state.messages))
            limit = self._config.evict_above_tokens
            plan = self._window_ahead() if self._state.windows else _WHOLE
            if plan is not _WHOLE:
                # Dropped first, so the cache shifts before the decode ahead, not before the next question's.
                at, before, length, split, run = plan
                held = len(self._memory.tokens)
                kept_until = lambda: at - (held - len(self._memory.tokens))   # noqa: E731
                done = self._drop_in_windows(kept_until, length)
                # Where the window's own older exchange must go too, or what is decoded ahead
                # would not fit, the rest is rendered whole.
                done = done and not (kept_until() + length > limit or self._too_many()
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
            return n_before - len(as_history(self._state.messages))

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

    # ----- windows --------------------------------------------------------------------
    def _window_ahead(self):
        """What :meth:`_render_ahead` decodes, from a render of the system message and the newest two exchanges alone.

        The window is written as the whole render is from the first tag after
        the older exchange's reply on. That tag is found in the cache from where
        the newest question's run was decoded, that question being the one
        spliced there; the stretch from the tag to the run, and what the splice
        decoded, are checked token for token.

        Returns:
            Where in the cache that tag sits, the tokens from it up to where the
            next question goes, how many of the conversation's own follow the
            tag, and the window's render with where the next question's run
            begins in it; or ``_WHOLE`` where the window cannot stand for the
            whole render.
        """
        starts = exchange_starts(self._state.messages)
        spliced_at = self._state.spliced_at
        if spliced_at is None or len(starts) < 2 or not self._goes_on_from(spliced_at[2]) \
                or self._state.messages[starts[-1]] is not spliced_at[2].messages[-1] or [
                m["role"] for m in self._state.messages[starts[-2]:]] != ["user", "assistant"] * 2 \
                or not self._rendering.template_writes(self._state.messages[starts[-2]:]):
            return _WHOLE
        system = self._system_part()
        window = (system + self._state.messages[starts[-2]:]
                  + [{"role": "user", "content": STAND_IN}])
        split = self._rendering.split(window, generation_prompt=True)
        if split is None or split.text.count(STAND_IN) != 1:
            return _WHOLE
        older_reply = len(system) + 1
        anchor = self._rendering.tag_after_content(split, older_reply)
        run = self._rendering.run_start(split, older_reply + 1, anchor)
        ahead_run = self._rendering.run_start(split, len(window) - 1, anchor)
        if None in (anchor, run, ahead_run):
            return _WHOLE
        to_question = self._rendering.tokenize(split.pieces(anchor, run))
        before = self._rendering.tokenize(split.pieces(anchor, ahead_run))
        question_at, spliced, _ = spliced_at
        at = question_at - len(to_question)
        if at < 0 or list(self._memory.tokens[at:question_at]) != to_question \
                or list(self._memory.tokens[question_at:question_at + len(spliced)]) != spliced \
                or before[:len(to_question)] != to_question:
            return _WHOLE
        # What counts against the limits, as :meth:`ConversationRendering.length` counts it: the
        # window without the stand-in.
        plain = self._rendering.split(window[:-1], generation_prompt=False)
        if plain is None or plain.text[:anchor] != split.text[:anchor]:
            return _WHOLE
        length = len(self._rendering.tokenize(plain.pieces(anchor)))
        return at, before, length, split, ahead_run

    def _drop_in_windows(self, kept_until, after: int) -> bool:
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
        if not (kept_until() + after > limit or self._too_many()):
            return True
        bound = self._drop_bound(limit)
        while kept_until() + after > bound or self._too_many():
            if len(exchange_starts(self._state.messages)) <= 2:
                break
            # The cache up to the tag after the second oldest exchange's reply, before the drop,
            # checked token for token, as no check at load covers it.
            held = self._rendering.head(self._state.messages, through=1)
            if held is None or list(self._memory.tokens[:len(held)]) != held:
                return False
            if not self._drop_oldest_exchange(True):
                break
            if not self._remove_dropped_in_window(held, kept_until()):
                return False
        return True

    def _remove_dropped_in_window(self, held: list[int], bound: int) -> bool:
        """Remove the oldest exchange, just dropped, from the cache, which begins with ``held``; nothing from ``bound`` on.

        After the drop, the cache up to the tag that ended ``held`` is the
        render of the system message and the oldest two exchanges kept, up to
        the tag after the first one's reply, and everything after that tag is
        as before.
        """
        wanted = self._rendering.head(self._state.messages, through=0)
        if wanted is None:
            return False
        same = common_prefix_length(held, wanted)
        end = len(held) - (len(wanted) - same)
        if end <= same or end > bound or held[end:] != wanted[same:]:
            return False
        self._memory.remove(same, end)
        self._state.kept_at = {k: at - (end - same) if at >= end else at
                               for k, at in self._state.kept_at.items()}
        return list(self._memory.tokens[:len(wanted)]) == wanted

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

    def _drop_while_over(self, limit: int, generation_prompt: bool, newest_too: bool = False,
                         tokens_only: bool = False) -> bool:
        """Drop the oldest exchanges once the conversation renders longer than ``limit`` or holds too many; whether any went.

        Where the cache cannot shift, they go until the history is well below
        its room under ``limit``, so that the cache is decoded again only every
        few turns.

        Args:
            tokens_only: Leave ``max_history_exchanges`` out, which may keep
                nothing on purpose.
        """
        def over(bound):
            return (self._rendering.length(self._state.messages, generation_prompt) > bound
                    or not tokens_only and self._too_many())

        if not over(limit):
            return False
        bound, dropped = self._drop_bound(limit), False
        while over(bound) and self._drop_oldest_exchange(newest_too):
            dropped = True
        return dropped

    def _drop_bound(self, limit: int) -> int:
        """How far an eviction brings the conversation down once it is over ``limit``.

        Where the cache cannot shift, what a short question alone takes and
        that fraction of the room after it, so that history is kept however
        long the system turn and wherever the template writes declarations.
        """
        if self._context.can_shift:
            return limit
        system = min(self._with_a_short_question(), limit)
        return system + int((limit - system) * EVICT_TO_FRACTION_WITHOUT_SHIFT)

    def _too_many(self) -> bool:
        """Whether the history holds more exchanges than ``max_history_exchanges``."""
        limit = self._config.max_history_exchanges
        return limit is not None and exchange_count(self._state.messages) > limit

    def _drop_oldest_exchange(self, newest_too: bool = False) -> bool:
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
        gone = self._state.messages[first:following]
        del self._state.messages[first:following]
        self._state.kept = [k for k in self._state.kept
                            if not any(k.first is m or k.last is m for m in gone)]
        return True

    # ----- what a record made ahead was made for ----------------------------------------
    def _note_ahead(self, split: SplitRender, run: int) -> None:
        """Note ``split``, decoded up to ``run``, as the render made ahead for the conversation and the cache as they are now."""
        self._state.ahead = _Ahead(rest=split.pieces(run),
                                   after_question=split.text[split.contents[-1][1]:],
                                   reply_opening=self._reply_opening(split),
                                   made_for=self._made_for_now())

    def _made_for_now(self) -> _MadeFor:
        """The conversation and the cache as they are now, for a record made of them."""
        state = self._state
        return _MadeFor(tuple(state.messages), tuple(state.tools), state.reasoning,
                        state.opening, self._memory.version)

    def _is_current(self, made_for: _MadeFor) -> bool:
        """Whether the conversation and the cache are still those ``made_for`` was made for."""
        return (made_for.cache_version == self._memory.version
                and len(made_for.messages) == len(self._state.messages)
                and self._goes_on_from(made_for))

    def _goes_on_from(self, made_for: _MadeFor) -> bool:
        """Whether the conversation goes on from the one ``made_for`` was made for: its messages first, the same declarations, reasoning and opening."""
        state = self._state
        return (len(made_for.messages) <= len(state.messages)
                and all(a is b for a, b in zip(made_for.messages, state.messages))
                and made_for.tools == tuple(state.tools)
                and (made_for.reasoning, made_for.opening) == (state.reasoning, state.opening))

    # ----- one action at a time --------------------------------------------------------
    @contextlib.contextmanager
    def _action(self):
        """Run one action: from a cache brought back where a decode failed, following a clock the template reads, and put back as it was where the action raises."""
        self._rendering.forget_renders()
        self._memory.repair()
        if self._state.begun:
            opening = self._rendering.opening_render(self._system_part())
            if opening != self._state.opening:
                # What was rendered ahead holds the old date, say; it is rendered whole now.
                self._check_renders()
                self._state.opening = opening
                self._memory.pin(self._rendering.opening_length(self._system_part()))
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
        system = self._system_part()
        self._published = _Published(
            system_prompt=system[0]["content"] if system else "",
            history=tuple(as_history(self._state.messages)),
            tools=tuple(self._state.tools) if self._state.uses_tool_template else None,
            reasoning=self._state.reasoning, pending_calls=tuple(self.pending_calls),
            begun=self._state.begun, turn_open=self.turn_open, newest_role=self.newest_role,
            total_tokens=len(self._memory.tokens))


# ----- what an action may start from -------------------------------------------------
def _ensure_ready(conversation, action: str, into: str | None = None) -> None:
    """Refuse an action before a conversation was begun, or inside a turn a call left open.

    ``conversation`` is a :class:`RenderedCache` or what it published. Only a
    begun conversation holds the sequence opener and the system prompt, and a
    message inside a model turn that never ended is a shape no template renders
    and no model was trained on.
    """
    if not conversation.begun:
        raise ValueError(f"no conversation to {into or action}; call begin() first")
    if conversation.turn_open:
        raise ValueError(
            f"cannot {action} while a reply is open: it stopped on a tool "
            "call and its turn is unclosed. Finish it with "
            "reply(), or discard it with begin().")


def _ensure_can_reply(conversation, asks_question: bool) -> None:
    """Refuse a reply ``conversation`` cannot take: a question while a reply is open, or, without one, a result still owed or nothing to answer."""
    if asks_question or not conversation.turn_open:
        _ensure_ready(conversation, "start a request", "generate into")
        if not asks_question and conversation.newest_role != "user":
            raise ValueError("there is nothing to answer: the newest message "
                             "is not a user message")
    elif conversation.pending_calls:
        raise ValueError(f"{len(conversation.pending_calls)} call(s) still owe a result")


# ----- helpers ---------------------------------------------------------------------
def _copied(value):
    """``value`` as it is now, a list or dict copied so that later changes leave it alone."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _unanswered(messages: list[dict]) -> list[tuple[ToolCall, str]]:
    """The calls of the newest turn that no result after them answers yet."""
    calling = next((i for i in range(len(messages) - 1, -1, -1)
                    if messages[i].get("tool_calls")), None)
    if calling is None or any(m["role"] not in ("tool",) for m in messages[calling + 1:]):
        return []
    answered = len(messages) - calling - 1
    return [(ToolCall(c["function"]["name"], copy.deepcopy(c["function"]["arguments"]), c["id"]),
             c["id"]) for c in messages[calling]["tool_calls"][answered:]]


def _is_plain(text: str, reader) -> bool:
    """A question the stand-in may stand for: words without tags or surrounding space."""
    if not text or text != text.strip() or STAND_IN in text:
        return False
    split = split_at_tags(reader, text)
    return split is not None and not any(is_tag for _, is_tag in split)
