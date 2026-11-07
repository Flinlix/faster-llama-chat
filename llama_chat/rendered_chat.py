"""A conversation the model's own template writes, kept in the cache by decoding only what changed.

A prototype beside :class:`~llama_chat.wrapper.ChatWrapper`. The template renders
the whole conversation; the cache is compared with that render token by token,
an exchange that was dropped is removed and the rest shifted, and everything
from the first token that differs is decoded again. The render for the next
question is made after each reply, with a stand-in where the question goes, so
a question needs only its own tokens decoded before the reply starts. Where the
template writes the newest exchanges and the oldest kept ones the same in a
render of them alone, only those are rendered, so the work after a reply does
not grow with the conversation. See design_history.md, "The template as the
program" and "Rendering windows".
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field

from .call_parsing import parse_calls
from .cache_context import CACHE_CONTEXT, ensure_answers
from .config import Config, Sampling
from .cache_context import GenerationAccumulator
from .cache_sync import CacheSync
from .context import KVContext
from .messages import Eviction, next_turn_start
from .reply_text import AnswerOpening, AnswerStops, ThoughtFilter, begins_in_thought, reply_header, \
    turn_end_test
from .resolved_template import ResolvedTemplate
from .template_extract import call_id
from .tokenizing import tokenize_pieces
from .tools import ToolCall, ToolRegistry, ToolResult, ToolSpec, ToolTurn, result_text
from .reply import ContextOverflowError, Reply, ReplyStream

# Stands where the next question will go in the render made ahead of it.
_STAND_IN = "QQSTANDINQQ"
# Questions a template must write exactly as the stand-in for questions to be put in its place.
_SPLICE_PROBES = ("Plain words here.", "Wie spät ist es?", "One more, please")
# The exchanges a template must write the same in a window as in the whole render for windows to be used.
_WINDOW_PROBES = tuple(
    [{"role": "user", "content": f"Question {n}, asked plainly."},
     {"role": "assistant", "content": f"Reply {n}, given plainly."}] for n in range(5))
# Returned where a window cannot stand for the whole render.
_WHOLE = object()
# Maps a token id to one character, so token lists compare and search as strings.
_TOKEN_OFFSET = 0x10000


@dataclass
class SyncCounts:
    """What keeping the cache equal to the render cost, summed over the conversation.

    Attributes:
        decoded_waiting: Tokens decoded after a question arrived, before its reply.
        decoded_ahead: Tokens decoded after a reply, while it is read or spoken.
        removed: Tokens removed from the middle of the cache and shifted past.
        rebuilt: Times the whole cache was decoded again.
        renders_waiting: Renders made after a question arrived.
        waiting_s: Time from a question to the start of its generation.
        ahead_s: Time spent after replies, rendering and decoding ahead.
        splice_misses: Questions put in the stand-in's place that the template
            then wrote otherwise.
        earliest_cut: The lowest position the cache was ever cut back to.
        settled_whole: Replies after which the whole conversation was rendered,
            a window not standing for it.
    """

    decoded_waiting: int = 0
    decoded_ahead: int = 0
    removed: int = 0
    rebuilt: int = 0
    renders_waiting: int = 0
    waiting_s: float = 0.0
    ahead_s: float = 0.0
    splice_misses: int = 0
    earliest_cut: int | None = None
    settled_whole: int = 0


@dataclass
class _Ahead:
    """The render for the next question, decoded up to the run of text the question joins."""

    rest: list[tuple[str, bool]]          # from that run on, the stand-in as the caller's own
    after_question: str                   # the template's text after the question
    n_messages: int                       # the conversation it was rendered for


@dataclass
class _Memo:
    """A model reader whose tokenizer remembers what it tokenized."""

    reader: object
    cache: dict = field(default_factory=dict)

    def tokenize(self, text, add_special=False, parse_special=True):
        key = (text, add_special, parse_special)
        if key not in self.cache:
            if len(self.cache) > 20000:
                self.cache.clear()
            self.cache[key] = self.reader.tokenize(text, add_special=add_special,
                                                   parse_special=parse_special)
        return self.cache[key]

    def split_at_tags(self, text):
        return self.reader.split_at_tags(text)


class RenderedChat:
    """One conversation over a cache context, written by the model's own template."""

    def __init__(self, config: Config | None = None, context=None) -> None:
        config = config or Config()
        owns_context = context is None
        self._context = KVContext(config) if owns_context else context
        try:
            ensure_answers(self._context, CACHE_CONTEXT)
            if config.context_size > self._context.context_size:
                raise ValueError(f"context_size is {config.context_size} but the cache context "
                                 f"holds {self._context.context_size} tokens")
            # Read only for what the model generates: thoughts, calls, the end of a turn.
            self._template = ResolvedTemplate(self._context)
        except BaseException:
            if owns_context:
                self._context.close()
            raise
        self._config = config
        self._reader = _Memo(self._context)
        self._messages: list[dict] = []
        self._tools: list[ToolSpec] = []
        self._registry: ToolRegistry | None = None
        self._reasoning = False
        self._cache: list[int] = []       # what the cache holds, token by token
        self._memory = CacheSync(self._context, lambda: self._cache)
        self._pinned: list[list[int]] = []  # each tool turn as it was generated
        self._ahead: _Ahead | None = None
        self._pending: list[ToolCall] = []
        self._reply_open = False
        self._n_calls = 0
        self._warmed = False
        self._splices = False
        self._windows = False             # whether windows of the conversation stand for its render
        self._prompt_text = ""            # the template text the next generation follows
        self._spliced_at: int | None = None
        self._lock = threading.Lock()
        self._renders: dict = {}           # the newest few renders of this call, by what was rendered
        self.counts = SyncCounts()

    # ----- the conversation ------------------------------------------------------
    def begin(self, system_prompt: str, history=(), *, tools=None,
              reasoning: bool | None = None) -> int:
        """Start a conversation from ``system_prompt`` and ``history``; returns how many messages were kept."""
        with self._lock:
            self._renders.clear()
            if not self._warmed:
                self._context.warmup()
                self._warmed = True
            self._registry = tools if isinstance(tools, ToolRegistry) else None
            self._tools = (list(tools.specs()) if isinstance(tools, ToolRegistry)
                           else list(tools or []))
            self._reasoning = bool(reasoning)
            self._messages = ([{"role": "system", "content": system_prompt}]
                              if system_prompt else []) + _as_messages(history)
            # Calls this conversation makes go on from the ids its history used.
            self._n_calls = sum(len(m.get("tool_calls") or ()) for m in self._messages)
            self._pending, self._reply_open = [], False
            self._spliced_at = None
            self._memory.reset()
            self._cache, self._pinned = [], []
            self._splices = self._splice_is_exact()
            self._ensure_history_stays_written()
            self._windows = self._splices and self._windows_are_exact()
            dropped = self._drop_while_over(self._evict_limit(), generation_prompt=False)
            if self._messages and self._messages[-1]["role"] == "user":
                self._ahead = None
                self._sync(self._prompted_tokens(), dropped=dropped)
            else:
                self._render_ahead(dropped=dropped)
            self._memory.rest()
            return sum(1 for m in self._messages if m["role"] != "system")

    def reply(self, text: str | None = None, *, calls="refuse", max_calls: int | None = None,
              max_reply_tokens: int | None = None, stop: list[str] | None = None,
              cancel: threading.Event | None = None,
              sampling: Sampling | None = None) -> ReplyStream:
        """Answer ``text``, or the newest user message, or go on with an open turn, as ChatWrapper.reply."""
        if calls not in ("withhold", "refuse", "yield", "run") and not isinstance(calls, ToolRegistry):
            raise ValueError('calls must be "withhold", "refuse", "yield", "run" or a ToolRegistry')
        return ReplyStream(self._exchange(text, calls, max_calls, max_reply_tokens, stop,
                                          cancel, sampling))

    def inject_tool_result(self, call: ToolCall | str, result) -> None:
        """Hand the model what the oldest owed call returned."""
        name = call.name if isinstance(call, ToolCall) else call
        with self._lock:
            self._renders.clear()
            if not self._pending or self._pending[0].name != name:
                raise ValueError("no such call is waiting for a result")
            self._pending.pop(0)
            self._messages.append({"role": "tool", "name": name,
                                   "tool_call_id": call_id(self._n_calls - len(self._pending) - 1),
                                   "content": result_text(result)})
            if not self._pending:
                start = time.perf_counter()
                self.counts.decoded_waiting += self._sync(self._prompted_tokens())
                self.counts.waiting_s += time.perf_counter() - start

    def declare_unknown_tools(self, tools) -> None:
        """Tell the model about more tools; the template writes them where it writes declarations."""
        specs = list(tools.specs()) if isinstance(tools, ToolRegistry) else list(tools)
        with self._lock:
            self._renders.clear()
            known = {spec.name for spec in self._tools}
            self._tools += [spec for spec in specs if spec.name not in known]
            if isinstance(tools, ToolRegistry):
                self._registry = tools
            self._windows = self._splices and self._windows_are_exact()
            self._spliced_at = None
            self._render_ahead()

    def truncate_last_reply(self, keep_text: str) -> None:
        """Shorten the newest reply to ``keep_text``, a prefix of it."""
        with self._lock:
            self._renders.clear()
            newest = self._messages[-1] if self._messages else {}
            if newest.get("role") != "assistant" or newest.get("tool_calls"):
                raise ValueError("the newest message is not a reply that may change")
            if not newest["content"].startswith(keep_text):
                raise ValueError("keep_text is not a prefix of the newest reply")
            newest["content"] = keep_text
            self._spliced_at = None
            self._render_ahead()

    def history(self) -> list:
        """The conversation as ``begin`` takes it: ``(role, text)`` pairs and tool turns."""
        return _as_history(self._messages)

    @property
    def messages(self) -> list[dict]:
        """The conversation as the template is given it."""
        return [dict(m) for m in self._messages]

    @property
    def total_tokens(self) -> int:
        return len(self._cache)

    def close(self) -> None:
        self._context.close()

    # ----- one request ------------------------------------------------------------
    def _exchange(self, text, calls, max_calls, max_reply_tokens, stop, cancel, sampling):
        registry = calls if isinstance(calls, ToolRegistry) else (
            self._registry if calls == "run" else None)
        calls_left = math.inf if max_calls is None else max_calls
        replies: list[Reply] = []
        with self._lock:
            self._renders.clear()
            start, decoded = time.perf_counter(), self.counts.decoded_waiting
            if text is not None:
                if self._reply_open:
                    raise ValueError("a reply is open: hand it its results first")
                self._ask(text)
            elif self._pending:
                raise ValueError("calls are waiting for their results")
            elif not self._reply_open:
                if not self._messages or self._messages[-1]["role"] != "user":
                    raise ValueError("there is nothing to answer")
                self.counts.renders_waiting += 1
                self._spliced_at = None
                self.counts.decoded_waiting += self._sync(self._prompted_tokens())
            self.counts.waiting_s += time.perf_counter() - start
            while True:
                allow = bool(self._tools) and calls not in ("withhold", "refuse") and calls_left > 0
                generation = yield from self._generate(allow, max_reply_tokens, stop, cancel,
                                                       sampling)
                replies.append(generation)
                if not (generation.tool_calls and registry is not None):
                    break
                for call in generation.tool_calls:
                    calls_left -= 1
                    self._lock.release()
                    try:
                        self.inject_tool_result(call, registry.call(call))
                    finally:
                        self._lock.acquire()
                if self._pending:
                    break
            if not self._reply_open:
                self._settle()
        return Reply(text="".join(r.text for r in replies),
                     n_prefilled_tokens=self.counts.decoded_waiting - decoded,
                     n_generated_tokens=sum(r.n_generated_tokens for r in replies),
                     n_evicted_messages=0, stop_reason=replies[-1].stop_reason,
                     tool_calls=[c for r in replies for c in r.tool_calls])

    def _ask(self, text: str) -> None:
        """Decode ``text`` as the next question, put in the stand-in's place where that is exact."""
        self._messages.append({"role": "user", "content": text})
        room = self._config.context_size - self._config.min_reply_tokens
        ahead = self._ahead
        if (ahead is not None and self._splices and ahead.n_messages == len(self._messages) - 1
                and _is_plain(text, self._context)):
            rest = [(piece.replace(_STAND_IN, text) if own else piece, own)
                    for piece, own in ahead.rest]
            tokens = tokenize_pieces(self._reader, rest)
            if len(self._cache) + len(tokens) <= room:
                self._prefill(tokens)
                self.counts.decoded_waiting += len(tokens)
                self._ahead = None
                self._spliced_at = len(self._cache) - len(tokens)
                self._prompt_text = ahead.after_question
                return
        self._ahead = None
        self._spliced_at = None
        self.counts.renders_waiting += 1
        dropped = self._drop_while_over(room, generation_prompt=True)
        target = self._prompted_tokens()
        if len(target) > room:
            self._messages.pop()
            raise ContextOverflowError("the question does not fit beside room for its reply")
        self.counts.decoded_waiting += self._sync(target, dropped=dropped)

    def _generate(self, allow_tools, max_reply_tokens, stop, cancel, sampling):
        """One generation from the cache's end, recorded as the newest assistant message."""
        reasoning = self._template.reasoning
        thought = (ThoughtFilter(reasoning, begins_in_thought(reasoning, self._prompt_text))
                   if reasoning else None)
        formatter = self._template.formatter(self._reasoning)
        opening = AnswerOpening(formatter.answer_opening(),
                                reply_header(formatter.answer_opening(), reasoning))
        answer = AnswerStops(list(self._config.stop_strings) + list(stop or []))
        generated = GenerationAccumulator()
        cap = max_reply_tokens if max_reply_tokens is not None else self._config.max_reply_tokens
        budget = self._config.context_size - len(self._cache)
        turn_end = self._template.turn_end_token
        ends_turn = (turn_end_test(turn_end, self._context.token_text(turn_end), reasoning, thought)
                     if turn_end is not None else None)
        start = len(self._cache)
        pieces = self._memory.generate(max(0, min(cap, budget)), generated,
                                       cancel=cancel, sampling=sampling, ends_turn=ends_turn,
                                       boundary=self._template.call_boundary(allow_tools))
        try:
            for piece in pieces:
                shown = answer.feed(opening.feed(thought.feed(piece) if thought else piece))
                if shown:
                    yield shown
                if answer.stopped:
                    pieces.close()
                    generated.stop_reason = "stop_string"
                    break
            else:
                complete = generated.stop_reason not in ("interrupted", "token_limit")
                last = opening.feed(thought.flush(complete) if thought else "")
                header_done = complete and generated.stop_reason != "tool_call"
                if shown := answer.flush(last + opening.flush(header_done)):
                    yield shown
        finally:
            pieces.close()
            self._cache += generated.token_ids + [0] * generated.cached_beyond
            if generated.cached_beyond:
                self._cut_to(len(self._cache) - generated.cached_beyond)
            calls = self._read_calls(generated)
            if generated.stop_reason == "tool_call":
                # A turn that called tools keeps the tokens it was generated as.
                self._pinned.append(self._cache[start:])
            message = {"role": "assistant", "content": answer.text}
            if calls:
                message["tool_calls"] = [
                    {"id": call_id(self._n_calls + n), "type": "function",
                     "function": {"name": call.name, "arguments": call.arguments}}
                    for n, call in enumerate(calls)]
                self._n_calls += len(calls)
            self._messages.append(message)
            self._pending = list(calls)
            self._reply_open = bool(calls)
        return Reply(text=answer.text, n_prefilled_tokens=0,
                     n_generated_tokens=len(generated.token_ids), n_evicted_messages=0,
                     stop_reason=generated.stop_reason, tool_calls=list(calls),
                     n_reasoning_tokens=thought.thought_pieces if thought else 0)

    def _read_calls(self, generated: GenerationAccumulator) -> list[ToolCall]:
        if generated.stop_reason != "tool_call":
            return []
        body = generated.tool_call
        if self._template.call_texts is not None:
            body = self._template.call_lead_in + body
        try:
            return parse_calls(body, self._template.call_body, self._template.call_lead_in)
        except Exception:
            return []

    def _prompted_tokens(self) -> list[int]:
        """The conversation with the generation prompt after it, noting the template text a reply follows."""
        rendered = self._render(self._messages, generation_prompt=True)
        if rendered is None:
            raise ValueError("the template refuses to write this conversation")
        pieces = _own_pieces(rendered, self._messages)
        self._prompt_text = "".join(text for text, own in pieces[-1:] if not own)
        return tokenize_pieces(self._reader, pieces, add_special=True)

    # ----- after a reply ----------------------------------------------------------
    def _settle(self) -> None:
        """Evict what is over the limit, then render and decode ahead for the next question."""
        start = time.perf_counter()
        spliced_at = self._spliced_at
        plan = self._window_ahead() if self._windows else _WHOLE
        cut = None
        if plan is not _WHOLE:
            # Dropped first, so the cache shifts before the decode ahead, not before the next question's.
            at, before, ahead = plan
            held, rest = len(self._cache), len(tokenize_pieces(self._reader, ahead.rest))
            dropped, done = self._drop_in_windows(
                lambda: at - (held - len(self._cache)), len(before) + rest, spare_window=True)
            if done:
                cut = self._sync(self._cache[:at - (held - len(self._cache))] + before,
                                 ahead=True, report_cut=True)
                # Where the window's own older exchange must go too, it goes after the decode.
                more, done = self._drop_in_windows(lambda: len(self._cache), rest)
                dropped = dropped or more
                ahead.n_messages = len(self._messages)
                self._ahead = ahead
            if not done:
                # The cache is brought to the whole render from wherever the windows left it.
                self.counts.settled_whole += 1
                self._drop_while_over(self._evict_limit(), generation_prompt=False, newest_too=True)
                self._render_ahead(dropped=True)
        else:
            self.counts.settled_whole += 1
            dropped = self._drop_while_over(self._evict_limit(), generation_prompt=False,
                                            newest_too=True)
            cut = self._render_ahead(dropped=dropped)
        if spliced_at is not None and cut is not None and cut < spliced_at and not dropped:
            self.counts.splice_misses += 1
        self._spliced_at = None
        self._memory.rest()
        self.counts.ahead_s += time.perf_counter() - start

    def _render_ahead(self, dropped: bool = False) -> int | None:
        """Decode the render for the next question up to where the question goes; returns the first token decoded again."""
        messages = self._messages + [{"role": "user", "content": _STAND_IN}]
        rendered = self._render(messages, generation_prompt=True)
        self._ahead = None
        split = (_split_before_run(_own_pieces(rendered, messages), self._context)
                 if rendered is not None and rendered.count(_STAND_IN) == 1 else None)
        if split is None:
            # The next question will be rendered when it comes.
            if self._messages and self._render(self._messages, False) is None:
                return None
            return self._sync(self._tokens(self._messages, generation_prompt=False),
                              ahead=True, report_cut=True, dropped=dropped)
        before, rest = split
        cut = self._sync(tokenize_pieces(self._reader, before, add_special=True),
                         ahead=True, report_cut=True, dropped=dropped)
        self._ahead = _Ahead(rest=rest, after_question=rendered.split(_STAND_IN, 1)[1],
                             n_messages=len(self._messages))
        return cut

    # ----- windows --------------------------------------------------------------------
    def _window_ahead(self):
        """What :meth:`_render_ahead` decodes, from a render of the system message and the newest two exchanges alone.

        The window is written as the whole render is from the first tag after
        the older exchange's reply on. That tag is found in the cache from where
        the newest question's run was decoded, the stretch up to there checked
        token for token.

        Returns:
            Where in the cache that tag sits, the tokens from it up to where the
            next question goes, and the render ahead from there; or ``_WHOLE``
            where the window cannot stand for the whole render.
        """
        question_at = self._spliced_at
        starts = self._exchange_starts()
        if question_at is None or len(starts) < 2 or [
                m["role"] for m in self._messages[starts[-2]:]] != ["user", "assistant"] * 2 \
                or any(m.get("tool_calls") for m in self._messages[starts[-2]:]):
            return _WHOLE
        system = self._system_part()
        window = system + self._messages[starts[-2]:] + [{"role": "user", "content": _STAND_IN}]
        rendered = self._render(window, generation_prompt=True)
        if rendered is None or rendered.count(_STAND_IN) != 1:
            return _WHOLE
        spans = _content_spans(rendered, window)
        older_reply = len(system) + 1
        anchor = self._tag_after(rendered, spans, older_reply)
        run = self._run_start(rendered, spans, older_reply + 1, anchor)
        if anchor is None or run is None:
            return _WHOLE
        split = _split_before_run(_pieces_between(rendered, spans, anchor, len(rendered)),
                                  self._context)
        if split is None:
            return _WHOLE
        before, rest = split
        to_question = tokenize_pieces(self._reader, _pieces_between(rendered, spans, anchor, run))
        before_tokens = tokenize_pieces(self._reader, before)
        at = question_at - len(to_question)
        if at < 0 or self._cache[at:question_at] != to_question \
                or before_tokens[:len(to_question)] != to_question:
            return _WHOLE
        return at, before_tokens, _Ahead(rest=rest, after_question=rendered.split(_STAND_IN, 1)[1],
                                         n_messages=len(self._messages))

    def _drop_in_windows(self, kept_until, after: int,
                         spare_window: bool = False) -> tuple[bool, bool]:
        """Drop the oldest exchanges while over the limit, each removed from a render of the oldest two kept.

        Args:
            kept_until: Where in the cache the window's tag sits now; nothing
                from there on may be removed.
            after: How many tokens the render has from that tag on.
            spare_window: Stop before the newest two exchanges, which the
                window was rendered from.

        Returns:
            Whether any exchange was dropped, and whether the cache is the
            render up to that tag after it; where it is not, the whole
            conversation is to be rendered.
        """
        dropped = False
        while kept_until() + after > self._evict_limit():
            if spare_window and len(self._exchange_starts()) <= 2:
                break
            if not self._drop_oldest_exchange(newest_too=True):
                break
            dropped = True
            if not self._remove_dropped_in_window(kept_until()):
                return dropped, False
        return dropped, True

    def _remove_dropped_in_window(self, bound: int) -> bool:
        """Remove what the oldest exchanges held from the cache before ``bound``, from a render of the system message and the oldest two kept."""
        starts = self._exchange_starts()
        if not starts:
            return False
        head_end = starts[2] if len(starts) > 2 else len(self._messages)
        kept = self._messages[starts[0]:head_end]
        if any(m["role"] == "tool" or m.get("tool_calls") for m in kept):
            return False
        system = self._system_part()
        head = system + kept + [{"role": "user", "content": _STAND_IN}]
        rendered = self._render(head, generation_prompt=True)
        if rendered is None:
            return False
        spans = _content_spans(rendered, head)
        first_end = (starts[1] if len(starts) > 1 else len(self._messages)) - starts[0]
        anchor = self._tag_after(rendered, spans, len(system) + first_end - 1)
        if anchor is None:
            return False
        wanted = tokenize_pieces(self._reader, _pieces_between(rendered, spans, 0, anchor),
                                 add_special=True)
        memory, wanted_text = _as_text(self._cache[:bound]), _as_text(wanted)
        same = _common_prefix(memory, wanted_text)
        if same < len(wanted):
            found = _dropped_stretch_end(memory, wanted_text, same)
            if found is None or found - same + len(wanted) > bound:
                return False
            self._remove_stretch(same, found)
        return self._cache[:len(wanted)] == wanted

    def _windows_are_exact(self) -> bool:
        """Whether this template writes the newest two exchanges, and the oldest two kept after a drop, as the whole render does.

        Checked on five exchanges with the conversation's system message,
        tools and reasoning: the window of the newest two from the first tag
        after the older one's reply, the history before that tag as the prompt
        for the newest reply wrote it, and after dropping the oldest exchange,
        the oldest two kept up to the first tag after the first one's reply and
        everything after that tag as before.
        """
        system = self._system_part()
        stand_in = [{"role": "user", "content": _STAND_IN}]
        ex = [list(e) for e in _WINDOW_PROBES]
        whole = self._render(system + sum(ex, []) + stand_in, True)
        window = self._render(system + ex[3] + ex[4] + stand_in, True)
        prompt = self._render(system + sum(ex[:4], []) + ex[4][:1], True)
        dropped = self._render(system + sum(ex[1:], []) + stand_in, True)
        head = self._render(system + ex[1] + ex[2] + stand_in, True)
        if None in (whole, window, prompt, dropped, head):
            return False
        older, first = ex[3][1]["content"], ex[1][1]["content"]
        at = [self._tag_after_text(text, content) for text, content in
              ((whole, older), (window, older), (dropped, first), (head, first), (whole, first))]
        if None in at:
            return False
        tail, in_window, drop, in_head, kept = at
        return (whole[tail:] == window[in_window:] and prompt[:tail] == whole[:tail]
                and dropped[:drop] == head[:in_head] and dropped[drop:] == whole[kept:])

    def _exchange_starts(self) -> list[int]:
        return [i for i, m in enumerate(self._messages) if m["role"] == "user"]

    def _system_part(self) -> list[dict]:
        return [m for m in self._messages[:1] if m["role"] == "system"]

    def _tag_after(self, rendered: str, spans, index: int) -> int | None:
        """Where the first tag after message ``index``'s content begins, before the next content."""
        if spans[index] is None:
            return None
        end = spans[index][1]
        following = next((s[0] for s in spans[index + 1:] if s is not None), len(rendered))
        found = self._first_tag(rendered[end:following])
        return None if found is None else end + found

    def _tag_after_text(self, rendered: str, content: str) -> int | None:
        found = rendered.find(content)
        if found < 0:
            return None
        end = found + len(content)
        first = self._first_tag(rendered[end:])
        return None if first is None else end + first

    def _run_start(self, rendered: str, spans, index: int, start: int | None) -> int | None:
        """Where the run of text holding message ``index``'s content begins: after the last tag from ``start`` on."""
        if start is None or spans[index] is None:
            return None
        split = self._context.split_at_tags(rendered[start:spans[index][0]])
        if split is None:
            return None
        at, run = start, None
        for piece, is_tag in split:
            at += len(piece)
            if is_tag:
                run = at
        return run

    def _first_tag(self, text: str) -> int | None:
        split = self._context.split_at_tags(text)
        if split is None:
            return None
        at = 0
        for piece, is_tag in split:
            if is_tag:
                return at
            at += len(piece)
        return None

    # ----- the cache against the render ----------------------------------------
    def _sync(self, target: list[int], *, ahead: bool = False, report_cut: bool = False,
              dropped: bool = False):
        """Make the cache hold ``target``, removing a dropped stretch and decoding from the first difference.

        Args:
            dropped: Whether exchanges were dropped from the conversation, so
                that the target may go on as the cache does after a stretch.

        Returns:
            How many tokens were decoded, or with ``report_cut`` where the
            first token decoded again lies (``None`` where nothing was).
        """
        target = self._keeping_pinned(target)
        memory, wanted = _as_text(self._cache), _as_text(target)
        same = _common_prefix(memory, wanted)
        if dropped and same < len(memory):
            found = _dropped_stretch_end(memory, wanted, same)
            if found is not None:
                self._remove_stretch(same, found)
                memory = _as_text(self._cache)
                same = _common_prefix(memory, wanted)
        cut = same if same < len(self._cache) else None
        if same < len(self._cache):
            self._cut_to(same)
        tokens = target[len(self._cache):]
        if tokens:
            self._prefill(tokens)
        if ahead:
            self.counts.decoded_ahead += len(tokens)
        if report_cut:
            return cut if cut is not None else (len(self._cache) - len(tokens) if tokens else None)
        return len(tokens)

    def _remove_stretch(self, start: int, end: int) -> None:
        """Take the cache's tokens from ``start`` to ``end`` out, the rest shifted or decoded again whole."""
        eviction = Eviction(start, end, len(self._cache))
        del self._cache[start:end]
        self.counts.rebuilt += not self._context.can_shift
        self._memory.remove(eviction)
        self.counts.removed += end - start

    def _cut_to(self, pos: int) -> None:
        """Keep the cache's first ``pos`` tokens: cut, back to a saved state, or decoded again."""
        if self.counts.earliest_cut is None or pos < self.counts.earliest_cut:
            self.counts.earliest_cut = pos
        del self._cache[pos:]
        if self._memory.cut_to(pos) is None:
            self.counts.rebuilt += 1

    def _prefill(self, tokens: list[int]) -> None:
        self._memory.decode(tokens, want_logits=True)
        self._cache += tokens

    def _keeping_pinned(self, target: list[int]) -> list[int]:
        """``target`` with each tool turn the cache holds as it was generated, where the template writes it otherwise.

        The turn is found in the cache; the target goes on after it where it
        holds what the cache holds after the turn, or, for a turn the cache
        ends with, after the call's closing as the template writes it.
        """
        memory = _as_text(self._cache)
        for turn in self._pinned:
            wanted = _as_text(target)
            same = _common_prefix(memory, wanted)
            at = memory.find(_as_text(turn))
            end = at + len(turn)
            if at < 0 or not at <= same < end:
                continue                # gone, or written as generated already
            if end < len(memory):
                follows = memory[end:end + 8]
                resume = wanted.find(follows, same)
            else:
                resume = -1
                for closing in self._call_closings(turn):
                    found = wanted.find(closing, same)
                    if found >= 0:
                        resume = found + len(closing)
                        break
            if resume >= 0:
                target = self._cache[:end] + target[resume:]
        return target

    def _call_closings(self, turn: list[int]) -> list[str]:
        """How a call's closing may read in a render: as the turn generated it, and as its text tokenizes."""
        closings = [_as_text(turn[-1:])]
        boundary = self._template.call_boundary(True)
        text = (boundary.call_texts[1] if boundary.call_texts
                else self._context.token_text(boundary.tool_tokens[1]) if boundary.tool_tokens
                else "")
        if text:
            closings.append(_as_text(self._reader.tokenize(text, parse_special=True)))
        return closings

    # ----- rendering ----------------------------------------------------------------
    def _render(self, messages, generation_prompt: bool) -> str | None:
        tools = [spec.as_declaration() for spec in self._tools] or None
        key = json.dumps([messages, tools, generation_prompt, self._reasoning], sort_keys=True,
                         default=str)
        if key not in self._renders:
            if len(self._renders) >= 4:
                self._renders.pop(next(iter(self._renders)))
            self._renders[key] = self._context.render_with_model_template(
                messages, tools=tools, generation_prompt=generation_prompt,
                reasoning=self._reasoning, **self._template.template_variables)
        return self._renders[key]

    def _tokens(self, messages, generation_prompt: bool) -> list[int]:
        if not messages:
            return tokenize_pieces(self._reader, [], add_special=True)   # the opener alone
        rendered = self._render(messages, generation_prompt)
        if rendered is None:
            raise ValueError("the template refuses to write this conversation")
        return tokenize_pieces(self._reader, _own_pieces(rendered, messages), add_special=True)

    def _evict_limit(self) -> int:
        return int(self._config.evict_above_fraction * self._config.context_size)

    def _drop_while_over(self, limit: int, generation_prompt: bool, newest_too: bool = False) -> bool:
        """Drop the oldest exchanges while the conversation renders longer than ``limit``; whether any went."""
        dropped = False
        while self._length(generation_prompt) > limit and self._drop_oldest_exchange(newest_too):
            dropped = True
        return dropped

    def _length(self, generation_prompt: bool) -> int:
        """How many tokens the conversation renders to, with the stand-in after it where it ends on a reply."""
        if not generation_prompt and (not self._messages or self._messages[-1]["role"] != "user"):
            # A template may refuse a conversation without a question; the next one is coming.
            messages = self._messages + [{"role": "user", "content": _STAND_IN}]
            if self._render(messages, True) is not None:
                return len(self._tokens(messages, True))
        return len(self._tokens(self._messages, generation_prompt))

    def _drop_oldest_exchange(self, newest_too: bool = False) -> bool:
        """Drop the oldest question and everything up to the next; ``False`` where none can go.

        Args:
            newest_too: Let the newest exchange go as well, once its reply is
                complete, as it may after a reply when the next question is to come.
        """
        first = 1 if self._messages and self._messages[0]["role"] == "system" else 0
        following = next_turn_start([m["role"] for m in self._messages], first + 1)
        if following == len(self._messages) and not (
                newest_too and len(self._messages) > first
                and self._messages[-1]["role"] == "assistant" and not self._reply_open):
            return False
        del self._messages[first:following]
        return True

    def _ensure_history_stays_written(self) -> None:
        """Refuse a template that writes an older exchange otherwise once another follows.

        Such a conversation could only be kept by decoding it again each turn,
        which the rolling memory does not do. Changes within the newest
        exchange, such as a past reply written without its thought or a system
        prompt moving to the newest question, are what the cache follows.

        Raises:
            ValueError: If the template rewrites anything before the newest exchange.
        """
        system = [m for m in self._messages[:1] if m["role"] == "system"]
        exchanges = [[{"role": "user", "content": f"QQASK{n}QQ"},
                      {"role": "assistant", "content": f"QQREPLY{n}QQ"}] for n in range(3)]
        two = self._render(system + exchanges[0] + exchanges[1], False)
        three = self._render(system + exchanges[0] + exchanges[1] + exchanges[2], False)
        if two is None or three is None or "QQREPLY0QQ" not in two:
            return
        kept = two.index("QQREPLY0QQ") + len("QQREPLY0QQ")
        if _common_prefix(two, three) < kept:
            raise ValueError("this model's template writes an older exchange otherwise once "
                             "another follows, which a rolling memory cannot keep")

    def _splice_is_exact(self) -> bool:
        """Whether this template writes a plain question exactly where it writes the stand-in."""
        system = [m for m in self._messages[:1] if m["role"] == "system"]
        base = system + [{"role": "user", "content": "Hello."},
                         {"role": "assistant", "content": "Hi."}]
        ahead = self._render(base + [{"role": "user", "content": _STAND_IN}], True)
        if ahead is None or ahead.count(_STAND_IN) != 1:
            return False
        return all(self._render(base + [{"role": "user", "content": probe}], True)
                   == ahead.replace(_STAND_IN, probe) for probe in _SPLICE_PROBES)


# ----- conversions -------------------------------------------------------------------
def _as_messages(history) -> list[dict]:
    """``begin``'s history as the messages a template is given."""
    messages, n = [], 0
    for item in history:
        if not isinstance(item, ToolTurn):
            role, text = item
            messages.append({"role": role, "content": text})
            continue
        text, calls, owed = "", [], []          # owed: the ids of the calls still to be answered
        for step in item.steps:
            if isinstance(step, str):
                text += step
            elif isinstance(step, ToolCall):
                calls.append(step)
            elif isinstance(step, ToolResult):
                if calls:
                    ids = [call_id(n + k) for k in range(len(calls))]
                    messages.append({"role": "assistant", "content": text, "tool_calls": [
                        {"id": ids[k], "type": "function",
                         "function": {"name": c.name, "arguments": c.arguments}}
                        for k, c in enumerate(calls)]})
                    owed += ids
                    n, text, calls = n + len(calls), "", []
                messages.append({"role": "tool", "name": step.name,
                                 "tool_call_id": owed.pop(0) if owed else call_id(n - 1),
                                 "content": result_text(step.result)})
        messages.append({"role": "assistant", "content": text})
    return messages


def _as_history(messages: list[dict]) -> list:
    """The messages back as ``begin`` takes them."""
    history, steps = [], []
    for message in messages:
        role = message["role"]
        if role == "system":
            continue
        if role == "assistant" and message.get("tool_calls"):
            if message["content"]:
                steps.append(message["content"])
            steps += [ToolCall(c["function"]["name"], c["function"]["arguments"])
                      for c in message["tool_calls"]]
        elif role == "tool":
            steps.append(ToolResult(message["name"], message["content"]))
        elif role == "assistant" and steps:
            history.append(ToolTurn(tuple(steps + ([message["content"]] if message["content"] else []))))
            steps = []
        else:
            history.append((role, message["content"]))
    return history


def _own_pieces(rendered: str, messages: list[dict]) -> list[tuple[str, bool]]:
    """``rendered`` in pieces, each message's content marked as the caller's own."""
    return _pieces_between(rendered, _content_spans(rendered, messages), 0, len(rendered))


def _content_spans(rendered: str, messages: list[dict]) -> list[tuple[int, int] | None]:
    """Where each message's content sits in ``rendered``, ``None`` where it is not found.

    Raises:
        ValueError: Where a content holding a tag cannot be found in the render,
            which would then read it as the template's own tags.
    """
    spans, at = [], 0
    for message in messages:
        content = message.get("content") or ""
        span = None
        for candidate in (content, content.strip()):
            if candidate:
                found = rendered.find(candidate, at)
                if found >= 0:
                    span = (found, found + len(candidate))
                    at = span[1]
                    break
        if span is None and content and any(ch in content for ch in "<[|"):
            raise ValueError("a message's text is not where the template writes it")
        spans.append(span)
    return spans


def _pieces_between(rendered: str, spans, start: int, end: int) -> list[tuple[str, bool]]:
    """``rendered`` from ``start`` to ``end`` in pieces, the contents in ``spans`` marked as the caller's own."""
    pieces, at = [], start
    for span in spans:
        if span is None or span[1] <= start or span[0] >= end:
            continue
        own_start, own_end = max(span[0], start), min(span[1], end)
        pieces += [(rendered[at:own_start], False), (rendered[own_start:own_end], True)]
        at = own_end
    pieces.append((rendered[at:end], False))
    return [(text, own) for text, own in pieces if text]


def _split_before_run(pieces, reader):
    """The pieces before the run of text the stand-in sits in, and that run on."""
    index = next(i for i, (text, own) in enumerate(pieces) if own and text == _STAND_IN)
    before, run = list(pieces[:index]), []
    while before:
        text, own = before[-1]
        if own:
            run.insert(0, before.pop())
            continue
        split = reader.split_at_tags(text)
        if split is None:
            return None
        tags = [k for k, (_, is_tag) in enumerate(split) if is_tag]
        if not tags:
            run.insert(0, before.pop())
            continue
        last = tags[-1]
        head = "".join(piece for piece, _ in split[:last + 1])
        before[-1] = (head, False)
        tail = text[len(head):]
        if tail:
            run.insert(0, (tail, False))
        break
    return before, run + list(pieces[index:])


def _is_plain(text: str, reader) -> bool:
    """A question the stand-in may stand for: words without tags or surrounding space."""
    if not text or text != text.strip() or _STAND_IN in text:
        return False
    split = reader.split_at_tags(text)
    return split is not None and not any(is_tag for _, is_tag in split)


def _dropped_stretch_end(memory: str, wanted: str, start: int) -> int | None:
    """Where the cache goes on as ``wanted`` does from ``start``, the stretch before it dropped; ``None`` where nowhere."""
    probe = wanted[start:start + 8]
    best, best_length, at = None, len(probe) - 1, memory.find(probe, start + 1) if probe else -1
    while at >= 0:
        length = _common_prefix(memory[at:], wanted[start:])
        if length > best_length:
            best, best_length = at, length
        at = memory.find(probe, at + 1)
    return best


def _as_text(tokens: list[int]) -> str:
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
