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
import json
from dataclasses import dataclass

from .cache_context import GenerationAccumulator
from .cache_sync import CacheSync
from .config import Config
from .messages import as_history, as_messages, calling_message, exchange_starts
from .render_checks import (STAND_IN, ensure_history_stays_written, question_splices, splices,
                            windows_are_exact, writes_two_user_turns)
from .reply import ContextOverflowError
from .resolved_template import ResolvedTemplate
from .split_render import SplitRender, split_render
from .tokenizing import MemoTokenizer, tokenize_pieces
from .tools import ToolCall, ToolSpec, result_text

# The name a result carries where no real one is to be had: answering a call
# that could not be read that far. Which name is used does not change what a
# template renders around it.
UNKNOWN_FUNCTION = "unknown"
_UNREADABLE_RESULT = {"error": "unreadable_call",
                      "message": "the tool call failed: the call could not be read"}
# Returned where a window cannot stand for the whole render.
_WHOLE = object()
# Maps a token id to one character, so token lists compare and search as strings.
_TOKEN_OFFSET = 0x10000
# What a failed action puts back as it was, the cache's tokens besides.
_ROLLED_BACK = ("_messages", "_tools", "_reasoning", "_kept", "_owed", "_n_calls", "_splices",
                "_windows", "_ahead", "_prompt_text", "_prompt_start", "_spliced_at", "_begun",
                "_opening", "_kept_at", "_asked")


@dataclass
class _Ahead:
    """The render for the next question, decoded up to the run of text the question joins."""

    rest: list[tuple[str, bool]]          # from that run on, the stand-in as the caller's own
    after_question: str                   # the template's text after the question
    n_messages: int                       # the conversation it was rendered for


@dataclass(frozen=True, eq=False)
class _Kept:
    """A turn held as the tokens it was generated as, in place of the template's version of it."""

    tokens: tuple[int, ...]
    first: dict                           # its first message
    last: dict                            # the message it ends with
    on_call: bool                         # whether it ends on a call, else on a reply's text


class RenderedCache:
    """One conversation's messages, as its template is given them, and the cache holding their render.

    Every method expects the caller to hold the conversation's lock, except
    :meth:`history` and :attr:`system_prompt`, which read a copy published
    after each change. Each action leaves the cache equal to the template's
    render of the messages, a turn that called tools held as it was generated,
    or, where it raises, the conversation and the cache as they were before it.
    """

    def __init__(self, context, template: ResolvedTemplate, config: Config) -> None:
        self._context = context
        self._template = template
        self._config = config
        self._reader = MemoTokenizer(context)
        self._memory = CacheSync(context)
        self._messages: list[dict] = []
        self._tools: list[ToolSpec] = []
        self._reasoning = False
        self._kept: list[_Kept] = []
        self._kept_at: dict[_Kept, int] = {}    # where each placed kept turn starts in the cache
        self._owed: list[tuple[ToolCall, str]] = []     # calls still to be answered, with their ids
        self._n_calls = 0
        self._splices = False
        self._windows = False             # whether windows of the conversation stand for its render
        self._ahead: _Ahead | None = None
        self._prompt_text = ""            # the template text the next generation follows
        self._prompt_start = 0            # where in the cache the generation prompt begins
        # Where a spliced question's run begins, the question, and the tokens decoded there.
        self._spliced_at: tuple[int, dict, list[int]] | None = None
        # The question the newest ask added and the tokens before it, until a generation follows.
        self._asked: tuple[dict, list[int]] | None = None
        self._begun = False
        self._opening: str | None = None  # how the conversation opens, to see a clock move on
        # Where the newest one began, whether it resumed a turn, and the tokens before
        # the question it answers where its request asked one.
        self._generation: tuple[int, bool, list[int] | None] = (0, False, None)
        self._renders: dict = {}           # this action's renders and splits, by what was rendered
        self._published: tuple[tuple, str] = ((), "")
        self._tag_tokens = {text: token for token in (template.tool_tokens or ())
                            if (text := context.token_text(token))}

    # ----- what it holds ---------------------------------------------------------------
    @property
    def messages(self) -> list[dict]:
        """The conversation as the template is given it; not to be changed by the caller."""
        return self._messages

    @property
    def system_prompt(self) -> str:
        """The system prompt, as last published; safe to read unlocked."""
        return self._published[1]

    def history(self) -> list:
        """The conversation as ``begin`` takes it, as last published, the caller's to change; safe to read unlocked."""
        return copy.deepcopy(list(self._published[0]))

    @property
    def tools(self) -> list[ToolSpec]:
        """What the conversation declares."""
        return list(self._tools)

    @property
    def begun(self) -> bool:
        """Whether a conversation was begun and the cache holds it."""
        return self._begun

    @property
    def reply_open(self) -> bool:
        """Whether the newest turn called tools and is not closed: a call is owed a result, or a result waits for the reply after it."""
        return bool(self._owed) or bool(self._messages) and self._messages[-1]["role"] == "tool"

    @property
    def owed_calls(self) -> list[ToolCall]:
        """The calls still owed a result, oldest first."""
        return [call for call, _ in self._owed]

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
        return self._prompt_text

    # ----- beginning ------------------------------------------------------------------
    def start(self, system_prompt: str, history, tools: list[ToolSpec], reasoning: bool) -> int:
        """Begin again from ``system_prompt`` and as much of ``history`` as fits; how many items were kept.

        A history that ends on a call keeps it owed a result.

        Raises:
            ContextOverflowError: If the system prompt alone, or the newest
                exchange beside it, exceeds ``evict_above_tokens``.
            ValueError: If the template refuses to write the conversation, or
                writes an older exchange otherwise once another follows.
        """
        with self._action():
            self._tools, self._reasoning = list(tools), reasoning
            self._renders.clear()
            system = [{"role": "system", "content": system_prompt}] if system_prompt else []
            self._messages = system + as_messages(history)
            self._kept, self._kept_at, self._owed = [], {}, []
            if any(m["role"] != "system" for m in self._messages) \
                    and self._split(self._messages, self._ends_on_result()) is None:
                roles = [m["role"] for m in self._messages]
                raise ValueError(self._refusal("user") if ["user", "user"] in [
                    roles[i:i + 2] for i in range(len(roles) - 1)]
                    else "this model's template refuses to write this conversation")
            limit = self._config.evict_above_tokens
            alone = self._system_length()
            if alone > limit:
                raise ContextOverflowError(
                    f"system prompt needs {alone} tokens but the eviction threshold is "
                    f"{limit}; shorten the prompt or raise context_size/evict_above_fraction")
            asks = any(m["role"] == "user" for m in self._messages)
            self._splices = splices(self._render, system)
            ensure_history_stays_written(self._render, system)
            self._windows = self._splices and windows_are_exact(self._render, system,
                                                                self._tag_from)
            self._drop_while_over(limit, generation_prompt=False, tokens_only=True)
            if asks and (not any(m["role"] == "user" for m in self._messages)
                         or self._length(self._messages, False) > limit):
                raise ContextOverflowError(
                    "the newest exchange of the history, from its user message on, does not "
                    "fit beside the system prompt; shorten it or raise "
                    "context_size/evict_above_fraction")
            self._drop_while_over(limit, generation_prompt=False)
            self._n_calls = sum(len(m.get("tool_calls") or ()) for m in self._messages)
            self._owed = _unanswered(self._messages)
            self._begun = False
            self._memory.reset()
            self._ahead, self._spliced_at, self._asked = None, None, None
            self._opening = self._opening_render()
            if self._messages and self._messages[-1]["role"] in ("user", "tool"):
                self._sync(self._prompted())
            elif self._owed:
                self._sync(self._tokens(self._messages, generation_prompt=False))
            else:
                self._render_ahead()
            self._memory.rest()
            self._begun = True
            return len(as_history(self._messages))

    def _ends_on_result(self) -> bool:
        return bool(self._messages) and self._messages[-1]["role"] == "tool"

    # ----- adding messages --------------------------------------------------------------
    def ask(self, text: str, accept=None) -> int:
        """Decode ``text`` as the next question, with the generation prompt after it; how many history items were dropped for it.

        Args:
            accept: Called once every check has passed and before anything is
                written; one that raises refuses the question.

        Raises:
            ValueError: If no conversation was begun, a reply is open, or the
                template does not write the question here.
            ContextOverflowError: If the question cannot fit beside room for
                its reply, checked against the system prompt alone before
                anything is dropped.
        """
        with self._action():
            self._ensure_ready("start a request", "generate into")
            room = self._config.context_size - self._config.min_reply_tokens
            question = {"role": "user", "content": text}
            alone = self._length(self._system_part() + [question], generation_prompt=True)
            if alone > room:
                raise ContextOverflowError(
                    f"request needs {alone} tokens but only {room} fit beside room for the reply")
            self.ensure_writes("user", text, generation_prompt=True)
            if accept is not None:
                accept()
            before = list(self._memory.tokens)
            n_before = len(as_history(self._messages))
            if not self._splice(question, room):
                self._messages.append(question)
                self._drop_while_over(room, generation_prompt=True)
                target = self._prompted()
                if len(target) > room:
                    raise ContextOverflowError("the question does not fit beside room for its reply")
                self._sync(target)
            free = self._config.context_size - len(self._memory.tokens)
            if free < self._config.min_reply_tokens:
                raise ContextOverflowError(
                    f"only {free} tokens free for the reply "
                    f"(min_reply_tokens={self._config.min_reply_tokens}); shorten "
                    "the request or lower evict_above_fraction")
            self._asked = question, before
            return n_before + 1 - len(as_history(self._messages))

    def _splice(self, question: dict, room: int) -> bool:
        """Add ``question``, decoded in the stand-in's place in the render made ahead, where that is exact; whether it was."""
        ahead, self._ahead, self._spliced_at = self._ahead, None, None
        text = question["content"]
        if (ahead is None or not self._splices or ahead.n_messages != len(self._messages)
                or not _is_plain(text, self._context)
                or not question_splices(self._render, self._system_part(), text)):
            return False
        rest = [(text if own and piece == STAND_IN else piece, own) for piece, own in ahead.rest]
        tokens = self._tokenize(rest)
        if len(self._memory.tokens) + len(tokens) > room:
            return False
        self._messages.append(question)
        self._memory.decode(tokens, want_logits=True)
        self._spliced_at = (len(self._memory.tokens) - len(tokens), question, tokens)
        self._note_prompt(ahead.after_question, self._memory.tokens)
        return True

    def add(self, role: str, text: str, accept=None) -> int:
        """Decode a message without a reply after it; returns how many history items were dropped for it.

        A question is put in the stand-in's place as :meth:`ask` puts one, the
        generation prompt after it.

        Args:
            accept: As :meth:`ask`.

        Raises:
            ValueError: If no conversation was begun, a reply is open, or the
                template does not write it after this conversation.
            ContextOverflowError: If it cannot fit beside the system prompt
                alone, checked before anything is dropped.
        """
        with self._action():
            self._ensure_ready("inject", "inject into")
            message = {"role": role, "content": text}
            limit = self._config.evict_above_tokens
            alone = self._length(self._system_part() + [message], generation_prompt=False)
            if alone > limit:
                raise ContextOverflowError(
                    f"injected message needs {alone} tokens but only {limit} fit beside the "
                    "system prompt")
            self.ensure_writes(role, text, generation_prompt=False)
            if accept is not None:
                accept()
            n_before = len(as_history(self._messages))
            room = self._config.context_size - self._config.min_reply_tokens
            fits = (self._config.max_history_messages is None
                    or n_before < self._config.max_history_messages)
            if role != "user" or not fits or not self._splice(message, min(room, limit)):
                self._messages.append(message)
                dropped = self._drop_while_over(limit, generation_prompt=False)
                self._ahead, self._spliced_at = None, None
                self._sync(self._tokens(self._messages, generation_prompt=False), logits=False,
                           dropped=dropped)
            self._memory.rest()
            return n_before + 1 - len(as_history(self._messages))

    def prompt(self) -> None:
        """Decode the generation prompt after the newest message, a user message, for a reply to it.

        Raises:
            ValueError: If no conversation was begun, a reply is open, or the
                newest message is not a user message.
            ContextOverflowError: If the conversation with it leaves too little
                room for a reply.
        """
        with self._action():
            self._ensure_ready("start a request", "generate into")
            if not self._messages or self._messages[-1]["role"] != "user":
                raise ValueError("there is nothing to answer: the newest message "
                                 "is not a user message")
            self._sync(self._fitting_prompt())

    def _fitting_prompt(self) -> list[int]:
        """The conversation with the generation prompt after it.

        Raises:
            ContextOverflowError: If it leaves too little room for a reply.
        """
        target = self._prompted()
        room = self._config.context_size - self._config.min_reply_tokens
        if len(target) > room:
            raise ContextOverflowError(
                f"the conversation needs {len(target)} tokens but only {room} leave room "
                "for the reply")
        return target

    # ----- generating -------------------------------------------------------------------
    def generate(self, max_tokens: int, generated: GenerationAccumulator, **options):
        """Stream one generation from the cache's end; what it decodes is held as the cache's.

        Raises:
            ValueError: If no conversation was begun, or the open turn still
                owes a result.
        """
        if not self._begun:
            raise ValueError("no conversation to generate into; call begin() first")
        if self._owed:
            raise ValueError(f"{len(self._owed)} call(s) still owe a result")
        asked = self._asked if self._asked and self._messages[-1] is self._asked[0] else None
        self._asked = None
        self._generation = (min(self._prompt_start, len(self._memory.tokens)), self.reply_open,
                            asked[1] if asked else None)
        return self._memory.generate(max_tokens, generated, **options)

    def record_reply(self, answer: str, thought: str, calls: list[ToolCall],
                     called: bool, recorded=None) -> int:
        """Record what the newest generation produced, then settle the cache after it.

        A turn that called tools keeps the tokens it was generated as, and stays
        open while a call is owed a result. A call that could not be read is
        answered with an error result, so the turn goes on to say so, or, where
        that result has no room, the turn closes as a reply. Any other reply
        closes the turn: one with no answer, after a question, is taken out
        again where the template writes two questions in a row, and the cache
        is then brought to the template's render and decoded ahead for the next
        question.

        Where the generation failed, what it said is recorded, or, where it
        said nothing, the question its request asked is taken out again; the
        cache is left to be decoded again by the next action.

        Args:
            thought: The reply's thought, given to the template as its own to
                write or leave out.
            calls: What it called, each owed a result.
            called: Whether it stopped on a call, read or not.
            recorded: Called once a reply that closes its turn is recorded,
                before the cache is settled after it.

        Returns:
            How many history items were dropped.
        """
        start, resumed, before_question = self._generation
        if not self._memory.is_known:
            self._ahead, self._spliced_at = None, None
        if not self._memory.is_known and not answer and not called:
            # It failed before it said anything: the request is taken back whole.
            if before_question is not None:
                self._messages.pop()
                self._memory.replace(before_question)
                self._publish()
            return 0
        unreadable = called and not calls
        asked = calls or ([ToolCall(UNKNOWN_FUNCTION, {})] if unreadable else [])
        message = ({"role": "assistant", "content": answer} if not asked
                   else calling_message(answer, asked, self._n_calls))
        if thought:
            message["reasoning_content"] = thought
        if asked:
            ids = [c["id"] for c in message["tool_calls"]]
            self._n_calls += len(asked)
            self._owed = list(zip(asked, ids))
        # A reply with neither text nor call has nothing to place its end by: a turn
        # kept so far stays kept up to its call, and the rest is the template's.
        if (called or resumed) and (answer or asked):
            first, open_turn = message, self._open_kept() if resumed else None
            if open_turn is not None:
                # The whole turn is kept, its results and their prompt included.
                self._kept.remove(open_turn)
                start, first = min(start, self._kept_at.pop(open_turn)), open_turn.first
            self._kept.append(_Kept(tuple(self._memory.tokens[start:]), first, message,
                                    bool(asked)))
            self._kept_at[self._kept[-1]] = start
        self._messages.append(message)
        self._publish()

        def report_closed():
            if recorded is not None and not self.reply_open:
                recorded()

        if not self._memory.is_known:
            report_closed()
            return 0                # the generation failed: the next action decodes the cache again
        if unreadable:
            try:
                written = self._write_unreadable()
            except BaseException:
                self._forget_calls()        # the cache is decoded again by the next action
                report_closed()
                raise
            if not written:
                self._forget_calls()
        if self.reply_open:
            return 0
        report_closed()
        if not answer and not called and not resumed \
                and writes_two_user_turns(self._render):
            # A reply all thought or cut before its answer leaves the question
            # unanswered, as the next question then follows it.
            return self._remove_newest()
        return self._settle()

    def _open_kept(self) -> _Kept | None:
        """The kept turn the open reply goes on with: the one ending on its newest call, placed in the cache."""
        calling = next((m for m in reversed(self._messages) if m.get("tool_calls")), None)
        return next((k for k in self._kept if k.last is calling and k.on_call
                     and k in self._kept_at), None)

    def _write_unreadable(self) -> bool:
        """Answer a call that could not be read, where the context has room; whether it was."""
        try:
            self.add_result(UNKNOWN_FUNCTION, _UNREADABLE_RESULT)
            return True
        except ValueError:          # no room for it, or the template writes no result
            return False

    def _forget_calls(self) -> None:
        """Record the newest reply without the call it made, which could be told nothing; its tokens stay as generated."""
        newest = {k: v for k, v in self._messages[-1].items() if k != "tool_calls"}
        old = self._messages[-1]
        self._messages[-1] = newest
        self._owed = []
        self._kept = [_Kept(k.tokens, newest if k.first is old else k.first, newest, False)
                      if k.last is old else k for k in self._kept]
        self._kept_at = {}
        self._publish()

    def add_result(self, call: ToolCall | str, result) -> None:
        """Record what the oldest owed call returned, clipped where it cannot fit; after the last one, decode the prompt for the reply.

        Raises:
            ValueError: If no call is waiting, this is not the one owed, or the
                template writes no result.
            ContextOverflowError: If not even an empty result leaves room for
                the reply.
        """
        with self._action():
            if not self._begun:
                raise ValueError("no conversation to answer a call in; call begin() first")
            name = call.name if isinstance(call, ToolCall) else call
            if not self._owed:
                raise ValueError("no tool call is waiting for a result")
            owed, owed_id = self._owed[0]
            if name != owed.name or (isinstance(call, ToolCall) and call != owed):
                raise ValueError(
                    f"the waiting call is {owed.name!r}, not {name!r}; results "
                    "are owed in the order the model asked for them")
            text = result_text(result)
            message = {"role": "tool", "name": name, "tool_call_id": owed_id, "content": text}
            if self._render(self._messages + [message], False) in (
                    None, self._render(self._messages, False)):
                raise ValueError("this model's template renders no tool result, so a call's "
                                 "result cannot be written")
            self._owed.pop(0)
            room = self._config.context_size - self._config.min_reply_tokens
            self._messages.append({**message, "content": ""})
            if len(self._prompted()) > room:
                raise ContextOverflowError(
                    "the result wrapping alone leaves no room for the reply in what is left "
                    "of the context; this turn cannot be continued, start a new conversation "
                    "with begin()")
            self._messages[-1] = {**message, "content": self._clipped(message, text, room)}
            if not self._owed:
                self._sync(self._fitting_prompt())

    def _clipped(self, message: dict, text: str, room: int) -> str:
        """The longest start of ``text`` with which the conversation, ``message`` newest, still fits in ``room``."""
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            self._messages[-1] = {**message, "content": text[:middle]}
            if len(self._prompted()) <= room:
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
                could not be read included, or ``keep_text`` is not a prefix
                of its text.
        """
        with self._action():
            self._ensure_ready("truncate a reply", "truncate in")
            newest = self._messages[-1] if self._messages else None
            if newest is None or newest["role"] != "assistant":
                raise ValueError("the newest message is not an assistant reply")
            if any(m["role"] == "tool" or m.get("tool_calls") for m in self._messages[-2:]) \
                    or any(k.last is newest for k in self._kept):
                raise ValueError("the newest reply holds a tool call, which would "
                                 "be separated from its result")
            if not newest["content"].startswith(keep_text):
                raise ValueError("keep_text is not a prefix of the newest reply")
            if keep_text == newest["content"]:
                return
            self._messages[-1] = {**newest, "content": keep_text}
            self._spliced_at = None
            self._render_ahead()
            self._memory.rest()

    def declare(self, specs: list[ToolSpec], accept=None) -> int:
        """Add ``specs`` where the template declares tools; how many history items were dropped.

        Args:
            accept: As :meth:`ask`.

        Raises:
            ValueError: If no conversation was begun or a reply is open.
            ContextOverflowError: If the system prompt with them exceeds ``evict_above_tokens``.
        """
        with self._action():
            self._ensure_ready("declare tools", "declare into")
            if accept is not None:
                accept()
            self._tools = self._tools + list(specs)
            self._renders.clear()
            limit = self._config.evict_above_tokens
            alone = self._system_length()
            if alone > limit:
                raise ContextOverflowError(
                    f"the system turn with these declarations needs {alone} tokens but the "
                    f"eviction threshold is {limit}; declare fewer tools or raise "
                    "context_size/evict_above_fraction")
            self._windows = self._splices and windows_are_exact(
                self._render, self._system_part(), self._tag_from)
            self._opening = self._opening_render()
            n_before = len(as_history(self._messages))
            dropped = self._drop_while_over(limit, generation_prompt=False, newest_too=True)
            self._spliced_at = None
            self._render_ahead(dropped=dropped)
            self._memory.rest()
            return n_before - len(as_history(self._messages))

    def ensure_writes(self, role: str, text: str, generation_prompt: bool) -> None:
        """Refuse a ``role`` message the template would not write next.

        Raises:
            ValueError: If the template refuses the conversation with it.
        """
        # Refusals turn on the newest messages, such as two user turns in a row, so
        # they are read from the newest exchange, not from a render of the whole conversation.
        starts = exchange_starts(self._messages)
        newest = self._system_part() + (self._messages[starts[-1]:] if starts else [])
        if self._render(newest + [{"role": role, "content": text}], generation_prompt) is None:
            raise ValueError(self._refusal(role))

    def _refusal(self, role: str) -> str:
        """Why the template will not write a ``role`` message next."""
        newest = self._messages[-1]["role"] if self._messages else None
        if role == "user" and newest == "user":
            return ("this model's template does not write two user turns in a row; write a "
                    "reply between them")
        if role == "system":
            return "this model's template writes no system message after the conversation has begun"
        return f"this model's template does not write a {role} message here"

    def _ensure_ready(self, action: str, into: str | None = None) -> None:
        """Refuse an action before a conversation was begun, or inside a turn a call left open.

        Only a begun conversation holds the sequence opener and the system
        prompt, and a message inside a model turn that never ended is a shape
        no template renders and no model was trained on.
        """
        if not self._begun:
            raise ValueError(f"no conversation to {into or action}; call begin() first")
        if self.reply_open:
            raise ValueError(
                f"cannot {action} while a reply is open: it stopped on a tool "
                "call and its turn is unclosed. Finish it with "
                "reply(), or discard it with begin().")

    def _system_length(self) -> int:
        """How many tokens the system turn takes, its declarations and the sequence opener included.

        Where the template writes no system message alone, it is what the
        system message adds to a render of a question.
        """
        system = self._system_part()
        if self._render(system, False) is not None:
            return len(self._tokens(system, generation_prompt=False))
        question = [{"role": "user", "content": STAND_IN}]
        return (len(self._tokens(system + question, generation_prompt=True))
                - len(self._tokens(question, generation_prompt=True)) + 1)

    def _remove_newest(self) -> int:
        """Take the newest message out again, then drop what is over the limits; how many history items were dropped."""
        with self._action():
            self._messages.pop()
            n_before = len(as_history(self._messages))
            dropped = self._drop_while_over(self._config.evict_above_tokens,
                                            generation_prompt=False)
            self._ahead, self._spliced_at = None, None
            self._sync(self._tokens(self._messages, generation_prompt=False), logits=False,
                       dropped=dropped)
            self._memory.rest()
            return n_before - len(as_history(self._messages))

    # ----- after a reply ----------------------------------------------------------
    def _settle(self) -> int:
        """Drop what is over the limits, then render and decode ahead for the next question; how many history items were dropped."""
        with self._action():
            n_before = len(as_history(self._messages))
            limit = self._config.evict_above_tokens
            plan = self._window_ahead() if self._windows else _WHOLE
            if plan is not _WHOLE:
                # Dropped first, so the cache shifts before the decode ahead, not before the next question's.
                at, before, length, ahead = plan
                held = len(self._memory.tokens)
                kept_until = lambda: at - (held - len(self._memory.tokens))   # noqa: E731
                done = self._drop_in_windows(kept_until, length)
                # Where the window's own older exchange must go too, or what is decoded ahead
                # would not fit, the rest is rendered whole.
                done = done and not (kept_until() + length > limit or self._too_many()
                                     or kept_until() + len(before) > self._config.context_size)
                if done:
                    self._sync(list(self._memory.tokens[:kept_until()]) + before, logits=False)
                    ahead.n_messages = len(self._messages)
                    self._ahead = ahead
                else:
                    self._drop_while_over(limit, generation_prompt=False, newest_too=True)
                    self._render_ahead(dropped=True)
            else:
                dropped = self._drop_while_over(limit, generation_prompt=False, newest_too=True)
                self._render_ahead(dropped=dropped)
            self._spliced_at = None
            self._memory.rest()
            return n_before - len(as_history(self._messages))

    def _render_ahead(self, dropped: bool = False) -> None:
        """Decode the render for the next question up to where the question goes."""
        messages = self._messages + [{"role": "user", "content": STAND_IN}]
        split = self._split(messages, generation_prompt=True)
        self._ahead = None
        run = (self._run_start(split, len(messages) - 1, 0)
               if split is not None and split.text.count(STAND_IN) == 1 else None)
        if run is not None:
            before = self._tokens(messages, generation_prompt=True, upto=run)
            if len(before) <= self._config.context_size:
                self._sync(before, logits=False, dropped=dropped)
                self._ahead = _Ahead(rest=split.pieces(run),
                                     after_question=split.text[split.contents[-1][1]:],
                                     n_messages=len(self._messages))
                return
        # The next question will be rendered when it comes.
        if self._messages and self._split(self._messages, False) is None:
            return
        self._sync(self._tokens(self._messages, generation_prompt=False), logits=False,
                   dropped=dropped)

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
            tag, and the render ahead from there; or ``_WHOLE`` where the
            window cannot stand for the whole render.
        """
        starts = exchange_starts(self._messages)
        if self._spliced_at is None or len(starts) < 2 \
                or self._messages[starts[-1]] is not self._spliced_at[1] or [
                m["role"] for m in self._messages[starts[-2]:]] != ["user", "assistant"] * 2 \
                or not self._template_writes(self._messages[starts[-2]:]):
            return _WHOLE
        system = self._system_part()
        window = system + self._messages[starts[-2]:] + [{"role": "user", "content": STAND_IN}]
        split = self._split(window, generation_prompt=True)
        if split is None or split.text.count(STAND_IN) != 1:
            return _WHOLE
        older_reply = len(system) + 1
        anchor = self._tag_after_content(split, older_reply)
        run = self._run_start(split, older_reply + 1, anchor)
        ahead_run = self._run_start(split, len(window) - 1, anchor)
        if None in (anchor, run, ahead_run):
            return _WHOLE
        to_question = self._tokenize(split.pieces(anchor, run))
        before = self._tokenize(split.pieces(anchor, ahead_run))
        question_at, _, spliced = self._spliced_at
        at = question_at - len(to_question)
        if at < 0 or list(self._memory.tokens[at:question_at]) != to_question \
                or list(self._memory.tokens[question_at:question_at + len(spliced)]) != spliced \
                or before[:len(to_question)] != to_question:
            return _WHOLE
        # What counts against the limits, as :meth:`_length` counts it: the window without the stand-in.
        plain = self._split(window[:-1], generation_prompt=False)
        if plain is None or plain.text[:anchor] != split.text[:anchor]:
            return _WHOLE
        length = len(self._tokenize(plain.pieces(anchor)))
        return at, before, length, _Ahead(
            rest=split.pieces(ahead_run), after_question=split.text[split.contents[-1][1]:],
            n_messages=len(self._messages))

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
            if len(exchange_starts(self._messages)) <= 2:
                break
            # The cache up to the tag after the second oldest exchange's reply, before the drop,
            # checked token for token, as no check at load covers it.
            held = self._head(through=1)
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
        wanted = self._head(through=0)
        if wanted is None:
            return False
        same = _common_prefix(_as_text(held), _as_text(wanted))
        end = len(held) - (len(wanted) - same)
        if end <= same or end > bound or held[end:] != wanted[same:]:
            return False
        self._memory.remove(same, end)
        self._kept_at = {k: at - (end - same) if at >= end else at
                         for k, at in self._kept_at.items()}
        return list(self._memory.tokens[:len(wanted)]) == wanted

    def _head(self, through: int) -> list[int] | None:
        """The render up to the first tag after the reply of oldest exchange ``through``, 0 or 1, from the system message and the oldest two exchanges; ``None`` where it cannot be told."""
        starts = exchange_starts(self._messages)
        if len(starts) <= through:
            return None
        bounds = starts[1:] + [len(self._messages)]
        kept = self._messages[starts[0]:bounds[min(1, len(bounds) - 1)]]
        if not self._template_writes(kept):
            return None
        system = self._system_part()
        split = self._split(system + kept + [{"role": "user", "content": STAND_IN}],
                            generation_prompt=True)
        if split is None:
            return None
        anchor = self._tag_after_content(split, len(system) + bounds[through] - starts[0] - 1)
        if anchor is None:
            return None
        return self._tokenize(split.pieces(0, anchor), add_special=True)

    def _template_writes(self, messages: list[dict]) -> bool:
        """Whether the template's render alone stands for ``messages`` in the cache: no call, result or turn kept as generated among them."""
        kept = {id(m) for k in self._kept for m in (k.first, k.last)}
        return not any(m["role"] == "tool" or m.get("tool_calls") or id(m) in kept
                       for m in messages)

    def _system_part(self) -> list[dict]:
        return [m for m in self._messages[:1] if m["role"] == "system"]

    def _tag_after_content(self, split: SplitRender, index: int) -> int | None:
        """Where the first tag after message ``index``'s content begins, before the next text of the caller's own."""
        if split.contents[index] is None:
            return None
        return self._tag_from(split.text, split.contents[index][1], split)

    def _tag_from(self, text: str, at: int, split: SplitRender | None = None) -> int | None:
        """Where the first tag at or after ``at`` begins, before the next own text of ``split`` where given."""
        end = next((s for s, _ in split.own if s >= at), len(text)) if split else len(text)
        pieces = self._context.split_at_tags(text[at:end])
        if pieces is None:
            return None
        for piece, is_tag in pieces:
            if is_tag:
                return at
            at += len(piece)
        return None

    def _run_start(self, split: SplitRender | None, index: int, start: int | None) -> int | None:
        """Where the run of text holding message ``index``'s content begins: after the last tag from ``start`` on."""
        if split is None or start is None or split.contents[index] is None:
            return None
        run = None
        for piece, own in split.pieces(start, split.contents[index][0]):
            if own:
                start += len(piece)
                continue
            pieces = self._context.split_at_tags(piece)
            if pieces is None:
                return None
            for part, is_tag in pieces:
                start += len(part)
                if is_tag:
                    run = start
        return run

    # ----- the cache against the target -----------------------------------------
    def _sync(self, target: list[int], *, logits: bool = True, dropped: bool = False) -> None:
        """Make the cache hold ``target``: a stretch it no longer holds removed, everything from the first difference decoded.

        Args:
            logits: A generation follows the decode at once.
            dropped: Exchanges were dropped, so that the target may go on as
                the cache does after a stretch.
        """
        memory, wanted = _as_text(self._memory.tokens), _as_text(target)
        same = _common_prefix(memory, wanted)
        if same < len(memory) and (dropped or self._moves_declarations()):
            found = _dropped_stretch_end(memory, wanted, same)
            if found is not None:
                self._memory.remove(same, found)
                same = _common_prefix(_as_text(self._memory.tokens), wanted)
        self._memory.cut_to(same)
        if self._moves_declarations():
            # The next question's declarations move back to here or past it: a cache
            # that cannot cut goes back to a state saved here, not to its start.
            self._memory.rest()
        self._memory.decode(target[same:], want_logits=logits)

    def _moves_declarations(self) -> bool:
        """Whether the declarations move with the newest question, leaving a stretch behind."""
        return bool(self._tools) and self._template.declares_before_newest_turn

    def _prompted(self) -> list[int]:
        """The conversation with the generation prompt after it, noting the template text a reply follows."""
        target = self._tokens(self._messages, generation_prompt=True)
        split = self._split(self._messages, generation_prompt=True)
        last = max((end for _, end in split.own), default=0)
        self._note_prompt(split.text[last:], target)
        return target

    def _note_prompt(self, prompt_text: str, target) -> None:
        """Note the template text a reply follows, and where it begins in ``target`` from its first tag on."""
        self._prompt_text = prompt_text
        first = self._tag_from(prompt_text, 0)
        tokens = self._tokenize([(prompt_text[first:], False)]) if first is not None else []
        ends = tokens and list(target[-len(tokens):]) == tokens
        self._prompt_start = len(target) - (len(tokens) if ends else 0)

    # ----- rendering ----------------------------------------------------------------
    def _render(self, messages, generation_prompt: bool) -> str | None:
        tools = [spec.as_declaration() for spec in self._tools] or None
        key = ("render", json.dumps([messages, generation_prompt], sort_keys=True, default=str))
        if key not in self._renders:
            self._renders[key] = self._template.render(
                messages, tools, generation_prompt=generation_prompt, reasoning=self._reasoning)
        return self._renders[key]

    def _split(self, messages, generation_prompt: bool) -> SplitRender | None:
        """``messages`` rendered and split into the template's text and the caller's.

        Raises:
            ValueError: If the template writes a text otherwise than as text in its place.
        """
        key = ("split", json.dumps([messages, generation_prompt], sort_keys=True, default=str))
        if key not in self._renders:
            self._renders[key] = split_render(lambda m: self._render(m, generation_prompt),
                                              messages, self._template.fragments.content)
        return self._renders[key]

    def _tokens(self, messages, generation_prompt: bool, upto: int | None = None) -> list[int]:
        """``messages`` as the cache holds them: their render tokenized, each kept turn as it was generated.

        Args:
            upto: Where in the render to stop.

        Raises:
            ValueError: If the template refuses to write them.
        """
        if not messages:
            return self._tokenize([], add_special=True)   # the opener alone
        split = self._split(messages, generation_prompt)
        if split is None:
            raise ValueError("the template refuses to write this conversation")
        end = len(split.text) if upto is None else upto
        placed = sorted(found for kept in self._kept
                        if (found := self._place(kept, messages, split)) is not None
                        and found[1] <= end)
        tokens, at, kept_at = [], 0, {}
        for start, stop, n in placed:
            if start < at:
                continue                # within a turn placed already
            tokens += self._tokenize(split.pieces(at, start), add_special=not tokens and at == 0)
            kept_at[self._kept[n]] = len(tokens)
            tokens += self._kept[n].tokens
            at = stop
        tokens += self._tokenize(split.pieces(at, end), add_special=not tokens and at == 0)
        if messages is self._messages or messages[:len(self._messages)] == self._messages:
            self._kept_at = kept_at
        return tokens

    def _place(self, kept: _Kept, messages: list[dict], split: SplitRender):
        """Where the template writes ``kept``'s turn in ``split``: from the first tag after the question's text to the end of its call or reply; ``None`` where it cannot be placed."""
        first = next((i for i, m in enumerate(messages) if m is kept.first), None)
        last = next((i for i, m in enumerate(messages) if m is kept.last), None)
        if not first or last is None or split.contents[first - 1] is None:
            return None
        start = self._tag_after_content(split, first - 1)
        if kept.on_call:
            called = split.call_ends[last]
            closing = self._template.call_close_text
            found = split.text.find(closing, called) if called is not None and closing else -1
            end = found + len(closing) if found >= 0 else None
        elif split.contents[last] is not None:
            end = split.contents[last][1]
        else:
            # A reply without text: it ends where the template's closing after it begins,
            # the last before the next text of the caller's own, as the question has one too.
            closing = self._template.fragments.assistant_suffix.strip()
            following = next((s for s, _ in split.own if start is not None and s > start),
                             len(split.text))
            found = (split.text.rfind(closing, start, following)
                     if closing and start is not None else -1)
            end = found if found >= 0 else None
        if start is None or end is None or end < start:
            return None
        return start, end, self._kept.index(kept)

    def _tokenize(self, pieces: list[tuple[str, bool]], add_special: bool = False) -> list[int]:
        """``pieces`` as a render tokenizes them, a call's tags as the single tokens a generated call holds."""
        return tokenize_pieces(self._reader, pieces, add_special=add_special,
                               tag_tokens=self._tag_tokens)

    def _opening_render(self) -> str | None:
        """How the template opens this conversation, which changes where it writes the date, say."""
        return self._render(self._system_part() + [{"role": "user", "content": STAND_IN}], True)

    # ----- limits -------------------------------------------------------------------
    def _drop_while_over(self, limit: int, generation_prompt: bool, newest_too: bool = False,
                         tokens_only: bool = False) -> bool:
        """Drop the oldest exchanges once the conversation renders longer than ``limit`` or holds too many; whether any went.

        Where the cache cannot shift, they go until it is well below ``limit``,
        so that the cache is decoded again only every few turns.

        Args:
            tokens_only: Leave ``max_history_messages`` out, which may keep
                nothing on purpose.
        """
        def over(bound):
            return (self._length(self._messages, generation_prompt) > bound
                    or not tokens_only and self._too_many())

        if not over(limit):
            return False
        bound, dropped = self._drop_bound(limit), False
        while over(bound) and self._drop_oldest_exchange(newest_too):
            dropped = True
        return dropped

    def _drop_bound(self, limit: int) -> int:
        """How far an eviction brings the conversation down once it is over ``limit``."""
        if self._context.can_shift:
            return limit
        return int(limit * self._config.evict_to_fraction_without_shift)

    def _too_many(self) -> bool:
        """Whether the history holds more items than ``max_history_messages``."""
        limit = self._config.max_history_messages
        return limit is not None and len(as_history(self._messages)) > limit

    def _length(self, messages: list[dict], generation_prompt: bool) -> int:
        """How many tokens the cache holds for ``messages``, each kept turn as it was generated.

        Where the template refuses a conversation that ends on a reply, up to
        where a question after it would go.
        """
        if not generation_prompt and messages and messages[-1]["role"] != "user" \
                and self._render(messages, False) is None:
            asked = messages + [{"role": "user", "content": STAND_IN}]
            split = self._split(asked, True)
            if split is not None and split.contents[-1] is not None:
                return len(self._tokens(asked, True, upto=split.contents[-1][0]))
        return len(self._tokens(messages, generation_prompt))

    def _drop_oldest_exchange(self, newest_too: bool = False) -> bool:
        """Drop the oldest question and everything up to the next; ``False`` where none can go.

        Args:
            newest_too: Let the newest exchange go as well, once its reply is
                complete, as it may after a reply when the next question is to come.
        """
        first = 1 if self._messages and self._messages[0]["role"] == "system" else 0
        following = next((i for i in exchange_starts(self._messages) if i > first),
                         len(self._messages))
        if following == len(self._messages) and not (
                newest_too and len(self._messages) > first
                and self._messages[-1]["role"] == "assistant"):
            return False
        gone = self._messages[first:following]
        del self._messages[first:following]
        self._kept = [k for k in self._kept
                      if not any(k.first is m or k.last is m for m in gone)]
        return True

    # ----- one action at a time --------------------------------------------------------
    @contextlib.contextmanager
    def _action(self):
        """Run one action: from a cache brought back where a decode failed, following a clock the template reads, and put back as it was where the action raises."""
        self._renders.clear()
        self._memory.repair()
        if self._begun:
            opening = self._opening_render()
            if opening != self._opening:
                # What was rendered ahead holds the old date, say; it is rendered whole now.
                self._opening, self._ahead, self._spliced_at = opening, None, None
        saved = {name: _copied(getattr(self, name)) for name in _ROLLED_BACK}
        held = list(self._memory.tokens)
        try:
            yield
        except BaseException:
            for name, value in saved.items():
                setattr(self, name, value)
            self._memory.replace(held)
            raise
        finally:
            self._publish()

    def _publish(self) -> None:
        """Publish the history and the system prompt for readers that hold no lock."""
        system = self._system_part()
        self._published = (tuple(as_history(self._messages)),
                           system[0]["content"] if system else "")


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
    return [(ToolCall(c["function"]["name"], copy.deepcopy(c["function"]["arguments"])), c["id"])
            for c in messages[calling]["tool_calls"][answered:]]


def _is_plain(text: str, reader) -> bool:
    """A question the stand-in may stand for: words without tags or surrounding space."""
    if not text or text != text.strip() or STAND_IN in text:
        return False
    split = reader.split_at_tags(text)
    return split is not None and not any(is_tag for _, is_tag in split)


def _dropped_stretch_end(memory: str, wanted: str, start: int) -> int | None:
    """Where the cache goes on as ``wanted`` does from ``start``, the stretch before it dropped; ``None`` where nowhere.

    The longest run of at least eight tokens wins. A wrong one costs decoding
    only: what differs after it is decoded again.
    """
    probe = wanted[start:start + 8]
    best, best_length, at = None, len(probe) - 1, memory.find(probe, start + 1) if probe else -1
    while at >= 0:
        length = _common_prefix(memory[at:], wanted[start:])
        if length > best_length:
            best, best_length = at, length
        at = memory.find(probe, at + 1)
    return best


def _as_text(tokens) -> str:
    return "".join(chr(token + _TOKEN_OFFSET) for token in tokens)


def _common_prefix(one: str, other: str) -> int:
    """How long a beginning the two share, found by halving."""
    low, high = 0, min(len(one), len(other))
    while low < high:
        middle = (low + high + 1) // 2
        if one[:middle] == other[:middle]:
            low = middle
        else:
            high = middle - 1
    return low
