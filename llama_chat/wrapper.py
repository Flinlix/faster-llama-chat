"""ChatWrapper: the public API, prefilling as little as possible.

``begin`` resets and prefills the system prompt plus what recent history fits;
``inject`` prefills one message without generating; ``turn`` and ``stream_turn``
prefill only the new request and then generate. Crossing either eviction limit
removes the oldest non-system messages and shifts the survivors down, reusing
their KV.

Two invariants hold after every action, and the tests enforce both: the physical
cache equals the concatenation of the message table, and every prefill appends
at the cache's end.

The turn-closing tag is decoded after the post-reply eviction rather than before
it. See the README, "How eviction works".
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from .config import Config
from .context import GenerationAccumulator, KVContext
from .messages import Message, MessageTable, n_newest_that_fit
from .template import CallBody, TemplateFragments, TemplateFormatter
from .template_extract import extract_call_body
from .tools import ToolCall, ToolRegistry, ToolSpec, parse_calls


# Where the prompt, a tool result and the reply after it go in a render. No one
# of them contains another, so a split on any lands where it means to.
_PROMPT = "PROMPTMSG"
_RESULT = "RESULTMSG"
_TAIL = "TAILMSG"


class ContextOverflowError(RuntimeError):
    """A request would leave fewer than ``min_reply_tokens`` free for the reply."""


@dataclass
class Turn:
    """What one request produced.

    One generation, or, where tools ran, every generation of the exchange:
    the counts are then sums and ``stop_reason`` is the last one's.

    Attributes:
        text: The assistant's reply, trimmed at any stop string.
        n_prefilled_tokens: Tokens decoded for the request and each resumption.
        n_generated_tokens: Tokens sampled.
        n_evicted_messages: Messages evicted to bring the cache back inside both
            eviction limits.
        stop_reason: One of ``"end_of_turn"``, ``"stop_string"``,
            ``"token_limit"``, ``"interrupted"``, ``"tool_call"`` or
            ``"unreadable_tool_call"``.
        tool_calls: The calls the model made, in the order it made them. A reply
            that stopped on one leaves its turn open until every one of them has
            a result.
        tool_limit_reached: Whether ``max_calls`` ran out, so the final
            generation ran with the call tokens withheld. Always ``False``
            where no call was asked for.
    """

    text: str
    n_prefilled_tokens: int
    n_generated_tokens: int
    n_evicted_messages: int
    stop_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_limit_reached: bool = False


@dataclass(frozen=True)
class _SystemTurnSplit:
    """The system turn this model renders, with tools declared and without.

    Split where the prompt goes, so the declarations are what the two renders
    differ by. Which side of the prompt they land on is the template's business.

    Attributes:
        declared_head, declared_tail: What the render with tools puts before
            and after the prompt.
        plain_head, plain_tail: The same for the render without them.
    """

    declared_head: str
    declared_tail: str
    plain_head: str
    plain_tail: str


class ChatWrapper:
    """Stateful single-conversation chat over llama.cpp with KV reuse."""

    # ----- setting up ----------------------------------------------------------
    def __init__(
        self,
        config: Config | None = None,
        context: KVContext | None = None,
        *,
        fragments: TemplateFragments | None = None,
        call_body: CallBody | None = None,
        formatter: Callable[[TemplateFragments], TemplateFormatter] | None = None,
        on_message: Callable[[str, str], None] | None = None,
        on_load_progress: Callable[[float], None] | None = None,
        **kwargs,
    ) -> None:
        """Load the model, resolve the chat template, and validate it.

        Args:
            config: Full configuration. Mutually exclusive with ``**kwargs``.
            context: A cache context of your own, in place of the one built
                from ``config``. :meth:`close` closes it too; only a
                construction that fails part-way leaves it alone.
            fragments: Explicit chat-template fragments, for a model that ships
                no embedded template. Recovered from the model otherwise.
            call_body: Explicit argument spelling, for a model whose call shape
                the probes cannot recover. Without one, such a model still
                detects its calls and reports them unreadable.
            formatter: Factory building the :class:`TemplateFormatter` from the
                resolved fragments, for a generation prompt that carries more
                than the turn tag.
            on_message: Invoked with ``(role, text)`` for every genuine new
                message, so not for the system prompt or a ``begin`` replay.
                Runs under the lock, so it must not call back into the wrapper.
            on_load_progress: Invoked during model loading with a fraction in
                ``[0.0, 1.0]``. Must not raise. Ignored when ``context`` is
                supplied.
            **kwargs: :class:`~llama_chat.config.Config` field overrides, used
                to build the config when ``config`` is not given.

        Raises:
            TypeError: If both ``config`` and config ``**kwargs`` are passed.
            ValueError: If the chat template cannot be recovered or fails
                validation against the model.
        """
        if config is None:
            config = Config(**kwargs)
        elif kwargs:
            raise TypeError(
                "pass either a Config or Config field overrides, not both "
                f"(got config and {sorted(kwargs)})"
            )
        self._on_message = on_message
        owns_context = context is None
        self._context = (
            KVContext(config, on_load_progress=on_load_progress)
            if owns_context else context
        )
        self._config = config
        try:
            self._fragments = fragments if fragments is not None else self._resolve_fragments()
            self._validate_template(self._fragments)
            self._formatter = (formatter or TemplateFormatter)(self._fragments)
            self._check_fragments_match_model()
        except BaseException:
            if owns_context:
                self._context.close()
            raise
        self._table = MessageTable()
        # None where this model cannot call tools at all.
        self._tool_tokens, self._call_lead_in = self._resolve_tool_tokens()
        self._call_body = (call_body if call_body is not None
                           else self._resolve_call_body())
        self._registry = None
        # An open turn and the calls owing a result are not the same thing: the
        # last result empties the list while the turn is still open.
        self._reply_open = False
        self._pending_calls = []
        # Not read off the table: `declare_tools` appends a system-role message,
        # so one stray call would make an unbegun table look begun.
        self._begun = False
        self._warmed = False
        self._closed = False
        # One KV sequence is shared across turns, so concurrent decodes would
        # corrupt it. `stream_turn` holds this for the whole turn.
        self._lock = threading.Lock()

    def begin(self, system_prompt: str, messages: list[tuple[str, str]] | None = None,
              tools: ToolRegistry | list[ToolSpec] | None = None) -> None:
        """Reset the conversation: prefill the system prompt and recent history.

        The first call also warms the generation graph, so the first ``turn``
        streams without a cold start.

        Args:
            system_prompt: Always kept, never evicted.
            messages: Optional ``(role, text)`` history, oldest first. Only the
                most recent that fit alongside the system prompt are prefilled,
                under both eviction limits; older excess is dropped.
            tools: Tools to declare, rendered into the system turn where this
                model's template puts them, which is the one placement that
                costs no extra prefill. A whole
                :class:`~llama_chat.tools.ToolRegistry` is kept, so
                :meth:`answer` needs no registry of its own; a list of specs is
                declared and not kept.

        Raises:
            ValueError: If the system prompt alone exceeds
                ``evict_above_tokens``. The previous conversation is left intact.
        """
        messages = messages or []
        with self._lock:
            self._ensure_open()
            # Validate before resetting: a failed begin must not destroy the
            # conversation it was about to replace.
            registry = tools if isinstance(tools, ToolRegistry) else None
            specs = registry.specs_to_declare() if registry is not None else tools
            sys_tokens = self._system_tokens(system_prompt, specs)
            if len(sys_tokens) > self._config.evict_above_tokens:
                raise ValueError(
                    f"system prompt needs {len(sys_tokens)} tokens but the "
                    f"eviction threshold is {self._config.evict_above_tokens}; "
                    "shorten the prompt or raise context_size/evict_above_fraction"
                )
            hist_tokens = [
                self._fragment_tokens(role, text)
                for role, text in messages
            ]

            kept = n_newest_that_fit(
                [len(t) for t in hist_tokens], len(sys_tokens),
                self._config.evict_above_tokens, self._config.max_history_messages,
            )
            keep_from = len(messages) - kept
            kept_messages = messages[keep_from:]
            kept_tokens = hist_tokens[keep_from:]

            self._context.reset()
            self._table.reset()
            self._begun = False     # true again only once the prefill lands
            # The turn a call was holding open went with the table.
            self._reply_open = False
            self._pending_calls = []
            self._registry = registry
            if not self._warmed:
                self._context.warmup()
                self._warmed = True

            # One decode for the whole conversation.
            all_tokens = list(sys_tokens)
            for toks in kept_tokens:
                all_tokens.extend(toks)
            self._context.prefill(all_tokens, start_pos=0, want_logits=False)

            self._table.append(Message("system", system_prompt, sys_tokens))
            for (role, text), toks in zip(kept_messages, kept_tokens):
                self._table.append(Message(role, text, toks))
            self._begun = True

    # ----- adding context ------------------------------------------------------
    def inject(self, text: str, role: str = "user") -> int:
        """Prefill one message as context without generating.

        Evicts the oldest messages until the new one fits inside both eviction
        limits, then prefills it.

        Returns:
            The number of messages evicted to make room.

        Raises:
            ValueError: If a tool call has left a reply open, or if the message
                cannot fit even after evicting every non-system message. Raised
                before anything is evicted.
        """
        with self._lock:
            self._ensure_open()
            self._ensure_no_open_reply("inject")
            tokens = self._fragment_tokens(role, text)
            # Checked against the best case before anything is evicted, so a
            # rejected message cannot destroy history it would never fit beside.
            floor = self._table.messages[0].n_tokens if self._table.has_system else 0
            tokens = self._fit_or_raise(
                tokens, budget=self._config.evict_above_tokens - floor,
                description="injected message", role=role,
            )
            evicted = self._evict_until(
                lambda: self._fits(len(tokens), arriving_messages=1))

            self._context.prefill(tokens, start_pos=self._table.total, want_logits=False)
            self._table.append(Message(role, text, tokens))
            if self._on_message:
                self._on_message(role, text)
            return evicted

    # ----- asking: one generation ----------------------------------------------
    def stream_turn(
        self, text: str, *, max_reply_tokens: int | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
    ) -> Generator[str, None, Turn]:
        """Add a user request and stream the reply token by token.

        Lazy: nothing happens, the headroom check included, until the first
        iteration. Closing the stream early still terminates the turn and
        records exactly the tokens that reached the cache, so the next turn
        stays valid; a reply that stopped on a tool call is the exception, its
        turn being left open on purpose.

        The generator must be driven from one thread at a time. To interrupt
        from another, pass ``cancel``. An abandoned generator holds the
        wrapper's lock until it is collected, so always exhaust or close it.

        Args:
            text: The user request text, tokenized with special-token parsing
                off. See the README, "Special tokens in message content".
            max_reply_tokens: Override for the per-turn generation cap.
            stop: Extra stop strings for this request, added to the config's.
            cancel: Setting it, from any thread, stops generation within one
                token and closes the turn with ``stop_reason == "interrupted"``.

        Yields:
            Reply text deltas.

        Returns:
            A :class:`Turn` describing the reply and the work performed.

        Raises:
            ValueError: If a tool call has left a reply open.
            ContextOverflowError: If too little room remains for the reply.
        """
        with self._lock:
            self._ensure_open()
            self._ensure_no_open_reply("start a request")
            user_tokens = self._fragment_tokens("user", text)
            open_tokens = self._context.tokenize(self._formatter.assistant_open(), add_special=False)
            close_tokens = self._context.tokenize(self._formatter.assistant_close(), add_special=False)

            # The prompt plus its turn closer must fit in the context.
            user_tokens = self._fit_or_raise(
                user_tokens,
                budget=self._config.context_size - len(open_tokens) - len(close_tokens) - self._table.total,
                description="request", role="user",
            )
            n_prompt = len(user_tokens) + len(open_tokens)

            # Refused before the cache is mutated.
            free = self._config.context_size - self._table.total - n_prompt - len(close_tokens)
            if free < self._config.min_reply_tokens:
                raise ContextOverflowError(
                    f"only {free} tokens free for the reply "
                    f"(min_reply_tokens={self._config.min_reply_tokens}); shorten "
                    f"the request or lower evict_above_fraction"
                )

            # One decode for the request text and the assistant-open tag: the
            # backend flags logits on the final token only either way.
            self._context.prefill(user_tokens + open_tokens,
                              start_pos=self._table.total, want_logits=True)
            self._table.append(Message("user", text, user_tokens))
            if self._on_message:
                self._on_message("user", text)

            return (yield from self._generate_into_turn(
                open_tokens=open_tokens, close_tokens=close_tokens,
                n_prefilled_tokens=n_prompt, max_reply_tokens=max_reply_tokens, stop=stop,
                cancel=cancel, allow_tools=True))

    def turn(
        self, text: str, *, max_reply_tokens: int | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
    ) -> Turn:
        """Add a user request and generate a reply. :meth:`stream_turn`, drained."""
        stream = self.stream_turn(text, max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel)
        try:
            while True:
                next(stream)
        except StopIteration as done:
            return done.value

    # ----- asking: until it answers --------------------------------------------
    def stream_answer(self, text: str, registry: ToolRegistry | None = None, *,
                      max_calls: int = 4,
                      **kwargs) -> Generator[str, None, Turn]:
        """Answer ``text``, running whatever tools the model calls on the way.

        Yields the reply's text deltas and nothing else, and returns the
        :class:`Turn` summing every generation. The last permitted round withholds
        the call tokens, so the turn always closes.

        Args:
            text: The request.
            registry: What to run calls against. Defaults to the one
                :meth:`begin` was given. Without either, a call the model
                invented is answered saying so and the reply goes on.
            max_calls: How many tools one turn may use. A call past the ceiling
                is answered with an error result rather than run.
            **kwargs: Passed to every generation, so ``max_reply_tokens``
                bounds each one rather than the whole exchange.

        Raises:
            ContextOverflowError: If a result no longer fits in what is left of
                the context. The turn is then open and cannot be continued.
        """
        registry = registry if registry is not None else self._registry
        events = self.stream_events(text, **kwargs)
        calls_left = max_calls
        any_call_asked = False
        generations: list[Turn] = []
        try:
            while True:
                asked = []
                while True:
                    # Driven by hand rather than with `for`, which would discard
                    # the Turn each generation returns.
                    try:
                        event = next(events)
                    except StopIteration as ended:
                        generations.append(ended.value)
                        break
                    if isinstance(event, ToolCall):
                        asked.append(event)
                    else:
                        yield event
                if not asked:
                    # The ceiling binds only once a call has been asked for:
                    # without one the last generation kept its call tokens.
                    return _sum_generations(generations,
                                            any_call_asked and calls_left == 0)
                any_call_asked = True
                # In the order asked, which is the order the template renders.
                for call in asked:
                    if registry is None:
                        # Raising would leave the turn open with no way to end it.
                        result = {"error": "no_tools",
                                  "message": "this conversation has no tools"}
                    elif calls_left > 0:
                        result = registry.call(call)
                        calls_left -= 1
                    else:
                        result = {"error": "tool_limit_reached",
                                  "message": "no tool calls left in this turn; "
                                             f"the limit is {max_calls}"}
                    self.inject_tool_result(call.name, result)
                events = self.continue_stream(
                    allow_tools=registry is not None and calls_left > 0,
                    **kwargs)
        finally:
            events.close()   # the current generation holds the wrapper's lock

    def answer(self, text: str, registry: ToolRegistry | None = None, *,
               max_calls: int = 4, **kwargs) -> Turn:
        """Answer ``text``, running whatever tools the model calls.

        :meth:`stream_answer`, drained. Arguments and failures are its own.

        Returns:
            A :class:`Turn` summing every generation; the reply is its ``text``.
        """
        stream = self.stream_answer(text, registry, max_calls=max_calls, **kwargs)
        while True:
            try:
                next(stream)
            except StopIteration as ended:
                return ended.value

    # ----- tools: driving the loop yourself ------------------------------------
    def stream_events(self, text: str,
                      **kwargs) -> Generator[str | ToolCall, None, Turn]:
        """As :meth:`stream_turn`, but yielding tool calls as well as text.

        Yields ``str`` deltas, then one
        :class:`~llama_chat.tools.ToolCall` per call the model made. Answer each
        with :meth:`inject_tool_result`, in order, then :meth:`continue_stream`.
        """
        return (yield from _as_events(self.stream_turn(text, **kwargs)))

    def continue_stream(self, **kwargs) -> Generator[str | ToolCall, None, Turn]:
        """Resume the reply a tool call interrupted.

        Yields events as :meth:`stream_events` does. Every result must be in
        first. ``allow_tools=False`` withholds the call tokens, so this generation
        cannot end on another call and the turn is certain to close.
        """
        return (yield from _as_events(self._resume(**kwargs)))

    def inject_tool_result(self, name: str, result) -> None:
        """Hand the model what the tool it called returned.

        Prefilled inside the turn the call left open, wrapped the way this
        model's own template wraps a result, and tokenized with special-token
        parsing off. Nothing is evicted while a turn is open, so such a turn
        grows towards ``context_size``. See the README, "Tool calling".

        Args:
            name: The function that ran. It must be the one the oldest
                unanswered call asked for.
            result: What it returned. Anything but a string is sent as JSON.

        Raises:
            ValueError: If no call is waiting, or ``name`` is not the one owed.
            ContextOverflowError: If not even the wrapping fits in what is left
                of the context.
        """
        with self._lock:
            self._ensure_open()
            if not self._pending_calls:
                raise ValueError("no tool call is waiting for a result")
            owed = self._pending_calls[0]
            if name != owed.name:
                raise ValueError(
                    f"the waiting call is {owed.name!r}, not {name!r}; results "
                    "are owed in the order the model asked for them")
            # Everything that can fail happens before the first mutation.
            prefix, suffix = self._result_wrapping(name)
            tags = self._context.tokenize(prefix, add_special=False, parse_special=True)
            closing = self._context.tokenize(suffix, add_special=False, parse_special=True)
            body = self._context.tokenize(_result_text(result), add_special=False,
                                      parse_special=False)
            # The body is clipped to fit; the tags never are. See the README,
            # "Tool calling".
            free = self._config.context_size - self._table.total
            if len(tags) + len(closing) > free:
                raise ContextOverflowError(
                    f"the result wrapping alone needs "
                    f"{len(tags) + len(closing)} tokens but only {free} are "
                    f"left in the context; this turn cannot be continued, "
                    f"start a new conversation with begin()")
            room = free - len(tags) - len(closing) - self._config.min_reply_tokens
            tokens = tags + body[:max(0, room)] + closing
            self._pending_calls.pop(0)      # results are owed in call order
            self._context.prefill(tokens, start_pos=self._table.total, want_logits=True)
            self._table.extend_last(tokens)

    def declare_tools(self, specs: list[ToolSpec]) -> int:
        """Tell the model about tools it was not told about at :meth:`begin`.

        Costs one prefill, and writes an ordinary message, so this declaration
        can be evicted later. One that has to survive belongs in ``begin``.

        Args:
            specs: What to declare. An empty list does nothing.

        Returns:
            How many messages were evicted to make room.

        Raises:
            ValueError: If no conversation has been begun, if a tool call has
                left a reply open, if this model's template declares no tools,
                or if the declarations cannot fit beside the system prompt.
        """
        if not specs:
            return 0
        with self._lock:
            self._ensure_open()
            self._ensure_no_open_reply("declare tools")
            # `begin` resets the table, so a declaration made before one would
            # be thrown away without the caller ever learning of it.
            if not self._begun:
                raise ValueError("no conversation to declare into; call begin() "
                                 "first, or pass tools= to it")
            block = self._declaration_block(specs)
            prefix, _, suffix = self._formatter.parts("system", "")
            # The model's own tags, so it joins the suffix and tokenizes
            # special-on. There is no caller text in this message.
            tokens = self._context.tokenize_fragment(prefix, "", block + suffix)
            floor = self._table.messages[0].n_tokens if self._table.has_system else 0
            # Never truncated: half a declaration is a tag block no template
            # produces, and it would sit in the cache for the whole conversation.
            room = self._config.evict_above_tokens - floor
            if len(tokens) > room:
                raise ValueError(
                    f"these declarations need {len(tokens)} tokens but only "
                    f"{room} fit beside the system prompt; declare fewer tools "
                    f"or raise context_size/evict_above_fraction")
            evicted = self._evict_until(
                lambda: self._fits(len(tokens), arriving_messages=1))
            self._context.prefill(tokens, start_pos=self._table.total, want_logits=False)
            self._table.append(Message("system", "", tokens))
            return evicted

    # ----- introspection -------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        """How many tokens the cache currently holds."""
        return self._table.total

    @property
    def context_size(self) -> int:
        """The hard limit in tokens, which :attr:`total_tokens` counts against."""
        return self._config.context_size

    @property
    def evict_above_tokens(self) -> int:
        """The total above which the oldest messages are evicted."""
        return self._config.evict_above_tokens

    @property
    def max_history_messages(self) -> int | None:
        """The most non-system messages kept, or ``None`` for no message limit."""
        return self._config.max_history_messages

    @property
    def can_shift(self) -> bool:
        """Whether eviction can shift surviving KV in place.

        ``False`` means eviction re-prefills the survivors instead. See the
        README, "How eviction works".
        """
        return self._context.can_shift

    @property
    def busy(self) -> bool:
        """Whether a turn currently holds the wrapper. Advisory: it can be stale."""
        if not self._lock.acquire(blocking=False):
            return True
        self._lock.release()
        return False

    def wait_idle(self, timeout_s: float | None = None) -> bool:
        """Block until no turn holds the wrapper. Advisory, as :attr:`busy` is.

        Args:
            timeout_s: Seconds to wait; ``None`` waits indefinitely.

        Returns:
            ``True`` if the wrapper went idle, ``False`` on timeout.
        """
        if not self._lock.acquire(timeout=-1 if timeout_s is None else timeout_s):
            return False
        self._lock.release()
        return True

    def snapshot(self) -> list[dict]:
        """Return the conversation and its cache layout, as plain dict rows.

        Safe from any thread, including while another is mid-turn.
        """
        return self._table.snapshot_rows()

    @property
    def can_call_tools(self) -> bool:
        """Whether this model's template describes tool calls at all."""
        return self._tool_tokens is not None

    @property
    def awaiting_tool_result(self) -> bool:
        """Whether any call is still holding the current turn open."""
        return bool(self._pending_calls)

    @property
    def reply_open(self) -> bool:
        """Whether a reply is unfinished, so no new message may be added.

        Outlasts :attr:`awaiting_tool_result`: the last result empties the calls
        while the turn stays open until :meth:`continue_stream` closes it. While
        this is ``True``, :meth:`turn`, :meth:`inject` and :meth:`declare_tools`
        refuse.
        """
        return self._reply_open

    # ----- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        """Free the underlying context and model. Idempotent and thread-safe.

        Waits for any in-flight turn. Closes a context passed to the
        constructor as well. Every action after close raises ``RuntimeError``.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._context.close()

    def __enter__(self) -> ChatWrapper:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----- internals -----------------------------------------------------------
    def _resume(self, *, max_reply_tokens: int | None = None,
                stop: list[str] | None = None,
                cancel: threading.Event | None = None,
                allow_tools: bool = True):
        """Generate on into the open reply, adding no message of its own."""
        with self._lock:
            self._ensure_open()
            if self._pending_calls:
                raise ValueError(f"{len(self._pending_calls)} call(s) still owe "
                                 f"a result")
            if not self._reply_open:
                raise ValueError("no reply is open; start one with stream_turn()")
            close_tokens = self._context.tokenize(self._formatter.assistant_close(),
                                              add_special=False)
            # See the README, "A resumed turn needs its generation prompt back".
            open_tokens = self._context.tokenize(self._formatter.resume_open(),
                                             add_special=False)
            if open_tokens:
                self._context.prefill(open_tokens, start_pos=self._table.total,
                                  want_logits=True)
                self._table.extend_last(open_tokens)
            return (yield from self._generate_into_turn(
                open_tokens=[], close_tokens=close_tokens, n_prefilled_tokens=0,
                max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
                allow_tools=allow_tools))

    def _resolve_tool_tokens(self) -> tuple[tuple[int, int] | None, str]:
        """This model's call tags as tokens, and the text it writes before a name."""
        fragments = self._fragments.tools
        if fragments is None:
            return None, ""
        opener = self._context.first_special_token(fragments.call_open)
        closer = self._context.first_special_token(fragments.call_close)
        if opener is None or closer is None:
            return None, ""
        tag = self._context.token_text(opener)
        lead_in = fragments.call_open[len(tag):] \
            if fragments.call_open.startswith(tag) else ""
        return (opener, closer), lead_in

    def _resolve_call_body(self) -> CallBody | None:
        """This model's argument spelling, or ``None`` if it cannot be read.

        A call stays detectable without it; only its contents do not.
        """
        if self._tool_tokens is None or self._fragments.tools is None:
            return None
        return extract_call_body(self._context.render_with_model_template,
                                 self._fragments.tools.call_open,
                                 self._context.token_text(self._tool_tokens[1]))

    def _system_tokens(self, system_prompt: str,
                       specs: list[ToolSpec] | None) -> list[int]:
        """The system turn, with any declarations where the template puts them.

        The declarations join the suffix rather than the content: they are the
        model's own tags and tokenize special-on, the prompt must not.
        """
        prefix, content, suffix = self._formatter.parts("system", system_prompt)
        if not specs:
            return self._context.tokenize_fragment(prefix, content, suffix,
                                               add_special=True)
        # Rendered whole: where declarations go is the template's business.
        split = self._declaration_turn(specs)
        return self._context.tokenize_fragment(split.declared_head, content,
                                           split.declared_tail, add_special=True)

    def _plain_system_turn(self) -> str | None:
        """The system turn this model renders with no tools, prompt sentinelled."""
        return self._context.render_with_model_template(
            [{"role": "system", "content": _PROMPT}])

    def _declaration_turn(self, specs: list[ToolSpec]) -> _SystemTurnSplit:
        """The system turn this model renders for ``tools``, and without them.

        Rendered by the template, so no declaration shape is written down here.

        Raises:
            ValueError: If the template renders no declarations.
        """
        rendered = self._context.render_with_model_template(
            [{"role": "system", "content": _PROMPT}],
            tools=[spec.as_declaration() for spec in specs])
        plain = self._plain_system_turn()
        if rendered is None or plain is None or _PROMPT not in rendered \
                or _PROMPT not in plain or rendered == plain:
            raise ValueError("this model's template declares no tools")
        declared_head, _, declared_tail = rendered.partition(_PROMPT)
        plain_head, _, plain_tail = plain.partition(_PROMPT)
        return _SystemTurnSplit(declared_head, declared_tail,
                                plain_head, plain_tail)

    def _declaration_block(self, specs: list[ToolSpec]) -> str:
        """Only the declarations, for adding to a conversation already begun.

        What the two renders differ by, from whichever side of the prompt the
        template put them on.

        Raises:
            ValueError: If the template renders no declarations.
        """
        split = self._declaration_turn(specs)
        block = ""
        if split.declared_head.startswith(split.plain_head):
            block += split.declared_head[len(split.plain_head):]
        if split.declared_tail.endswith(split.plain_tail):
            block += split.declared_tail[:len(split.declared_tail)
                                         - len(split.plain_tail)]
        if not block:
            raise ValueError("this model's template declares no tools")
        return block

    def _result_wrapping(self, name: str) -> tuple[str, str]:
        """What this model's template wraps ``name``'s result in.

        Recovered per result, because the wrapping names the function.

        Raises:
            ValueError: If the template renders no result block.
        """
        rendered = self._context.render_with_model_template([
            {"role": "user", "content": _PROMPT},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": name, "arguments": {}}}]},
            {"role": "tool", "name": name, "content": _RESULT},
            {"role": "assistant", "content": _TAIL},
        ])
        closer = self._context.token_text(self._tool_tokens[1])
        if rendered is None or _RESULT not in rendered \
                or _TAIL not in rendered or closer not in rendered:
            raise ValueError("this model's template renders no tool result")
        head, _, tail = rendered.partition(_RESULT)
        return head.rsplit(closer, 1)[1], tail.partition(_TAIL)[0]

    def _generate_into_turn(self, *, open_tokens: list[int],
                            close_tokens: list[int], n_prefilled_tokens: int,
                            max_reply_tokens: int | None, stop: list[str] | None,
                            cancel: threading.Event | None,
                            allow_tools: bool) -> Generator[str, None, Turn]:
        """Generate from the cache's end and book what it left, as a Turn.

        Shared by a fresh turn, whose generation prompt is prefilled but not yet
        in the table, and a resumed one, which brings none of its own.

        Args:
            open_tokens: The generation prompt, empty when resuming.
            close_tokens: The turn terminator, decoded once the turn closes.
            n_prefilled_tokens: What the caller decoded to get here, for the Turn.
            max_reply_tokens, stop, cancel: As :meth:`stream_turn`.
            allow_tools: ``False`` withholds the call tokens, so this generation
                cannot end on a call and the turn is certain to close.
        """
        gen_start = self._table.total + len(open_tokens)
        budget = self._config.context_size - gen_start - len(close_tokens)
        cap = max_reply_tokens if max_reply_tokens is not None else self._config.max_reply_tokens
        stops = list(self._config.stop_strings) + list(stop or [])

        generated = GenerationAccumulator()
        try:
            yield from self._context.generate(
                gen_start, max(0, min(cap, budget)), stops, out=generated,
                cancel=cancel,
                tool_tokens=self._tool_tokens if allow_tools else None,
            )
        finally:
            # Runs on normal completion and on early close alike.
            n_evicted_messages = self._book_reply(generated, open_tokens,
                                                 close_tokens)

        return Turn(
            text=generated.text,
            n_prefilled_tokens=n_prefilled_tokens,
            n_generated_tokens=len(generated.token_ids),
            n_evicted_messages=n_evicted_messages,
            stop_reason=generated.stop_reason,
            tool_calls=list(self._pending_calls),
        )

    def _book_reply(self, generated: GenerationAccumulator,
                    open_tokens: list[int], close_tokens: list[int]) -> int:
        """Record what one generation left.

        The turn is closed unless a call is waiting for its result. The
        closing tag joins the message only in :meth:`_close_turn`, after the
        post-reply eviction, so the table always describes what the cache
        holds.

        Returns:
            How many messages were evicted.
        """
        tokens = open_tokens + generated.token_ids
        if self._reply_open:
            self._table.extend_last(tokens, generated.text)
        else:
            self._table.append(Message("assistant", generated.text, tokens))
        msg = self._table.messages[-1]
        self._pending_calls = self._read_calls(generated)
        if generated.stop_reason == "tool_call" and not self._pending_calls:
            # An unreadable call must not hold the turn open: nothing would ever
            # close it. Closing costs the call and keeps the conversation usable.
            generated.stop_reason = "unreadable_tool_call"
        self._reply_open = bool(self._pending_calls)
        if self._reply_open:
            return 0
        if self._on_message:
            self._on_message("assistant", msg.text)
        # Trim back after the reply, so the cache rests inside both eviction
        # limits for the next turn (and on barge-in too). The closer counts
        # against the threshold even though it is decoded after the trim.
        n_evicted_messages = self._evict_until(lambda: self._fits(len(close_tokens)))
        self._close_turn(msg, close_tokens)
        return n_evicted_messages

    def _read_calls(self, generated: GenerationAccumulator) -> list[ToolCall]:
        """The calls ``generated`` stopped on, empty where it produced no readable call.

        Every failure counts as "not a call": this runs with the turn still
        open, so one that got out would leave nothing to close it.
        """
        if generated.stop_reason != "tool_call":
            return []
        try:
            return parse_calls(generated.tool_call, self._call_body,
                               self._call_lead_in)
        except Exception:
            return []

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ChatWrapper is closed")

    def _ensure_no_open_reply(self, action: str) -> None:
        """Refuse to append a message inside a turn a tool call left open.

        Such a message would sit inside a model turn that never ended, which no
        template renders and no model was trained on.
        """
        if self._reply_open:
            raise ValueError(
                f"cannot {action} while a reply is open: it stopped on a tool "
                "call and its turn is unclosed. Finish it with "
                "continue_stream(), or discard it with begin()."
            )

    def _resolve_fragments(self) -> TemplateFragments:
        """Recover the template fragments from the model's own chat template."""
        if not hasattr(self._context, "extract_fragments"):
            raise ValueError("context cannot supply template fragments")
        fragments = self._context.extract_fragments()
        if fragments is None:
            raise ValueError(
                "the model ships no chat template, so the fragments cannot be "
                "derived; pass them explicitly with ChatWrapper(fragments=...)"
            )
        return fragments

    def _validate_template(self, fragments: TemplateFragments) -> None:
        """Ensure the turn-terminator is a real special token of this model.

        A terminator that splits into plain text never lets the model stop, so
        it would run to the token cap every turn.
        """
        if not hasattr(self._context, "tokenizes_to_special"):
            return
        terminator = fragments.assistant_suffix.strip()
        if terminator and not self._context.tokenizes_to_special(terminator):
            raise ValueError(
                "chat template is not valid for this model: its turn-terminator "
                f"{terminator!r} does not tokenize to a special token, so "
                "generation would never stop. The model's embedded chat template "
                "is malformed; supply correct tags with ChatWrapper(fragments=...)."
            )

    def _check_fragments_match_model(self) -> None:
        """Assert the user and assistant fragments match the model's own template.

        Tags that do not reproduce what the model saw in training make
        incremental prefill write tokens it never learned. The system fragment
        is not probed, there being no portable ground truth for it.
        """
        if not hasattr(self._context, "render_with_model_template"):
            return
        # The surrounding whitespace catches a wrong `trim_content` too.
        probe = [
            {"role": "user", "content": " ping "},
            {"role": "assistant", "content": " pong "},
        ]
        rendered = self._context.render_with_model_template(probe)
        if rendered is None:
            return
        model_tokens = self._context.tokenize(rendered, add_special=False)
        # Built exactly as real prefill builds it, tags special-on and content
        # special-off, so any mismatch is a tag or trim divergence.
        fragment_tokens: list[int] = []
        for m in probe:
            fragment_tokens.extend(self._fragment_tokens(m["role"], m["content"]))
        if model_tokens != fragment_tokens:
            fragment_text = "".join(self._formatter.fragment(m["role"], m["content"]) for m in probe)
            raise ValueError(
                "chat template fragments do not match the model's trained "
                "template:\n"
                f"  model renders:    {rendered!r}\n"
                f"  fragments render: {fragment_text!r}\n"
                "The recovered fragments do not round-trip through the model's "
                "template; supply correct tags with ChatWrapper(fragments=...)."
            )

    def _fits(self, arriving_tokens: int, arriving_messages: int = 0) -> bool:
        """Whether the cache would rest inside both eviction limits.

        The two limits are OR'd as triggers, so resting inside means clearing
        both. The arriving counts describe what is about to be appended but is
        not in the table yet, which is how room is made for a message before it
        exists; a reply already in the table brings neither.
        """
        if self._table.total + arriving_tokens > self._config.evict_above_tokens:
            return False
        limit = self._config.max_history_messages
        return limit is None or self._table.n_evictable + arriving_messages <= limit

    def _evict_until(self, fits) -> int:
        """Evict the oldest non-system messages until ``fits()`` or none remain.

        Shifts the survivors in place where the cache allows it, and re-prefills
        the whole surviving conversation where it does not. See the README,
        "How eviction works".
        """
        if self._context.can_shift:
            eviction, evicted = self._table.evict_oldest_until(fits)
            if eviction is not None:
                self._context.apply_eviction(eviction)  # one seq_rm + one seq_add
            return evicted

        # Survivors are decided first, then re-prefilled.
        _, evicted = self._table.evict_oldest_until(fits)
        if evicted:
            self._rebuild_cache()
        return evicted

    def _close_turn(self, msg: Message, close_tokens: list[int]) -> None:
        """Decode the assistant turn's closing tag.

        Skipped where the eviction dropped this very message.
        """
        messages = self._table.messages
        if not messages or messages[-1] is not msg:
            return
        self._context.prefill(close_tokens, start_pos=self._table.total, want_logits=False)
        self._table.extend_last(close_tokens)

    def _rebuild_cache(self) -> None:
        """Re-prefill the cache from the surviving messages (no in-place shift)."""
        self._context.reset()
        all_tokens: list[int] = []
        for m in self._table.messages:
            all_tokens.extend(m.token_ids)
        self._context.prefill(all_tokens, start_pos=0, want_logits=False)

    def _fit_or_raise(self, tokens: list[int], budget: int, description: str,
                      role: str) -> list[int]:
        """Apply ``oversize_policy`` to a message that does not fit.

        Truncation drops content from the end and re-appends the role's
        turn-terminator, so a clipped message still closes its turn.
        """
        if len(tokens) <= budget:
            return tokens
        if self._config.oversize_policy == "truncate":
            suffix = self._context.tokenize(self._formatter.suffix(role), add_special=False)
            if budget > len(suffix):
                return tokens[: budget - len(suffix)] + suffix
        raise ValueError(
            f"{description} needs {len(tokens)} tokens but only {budget} fit "
            f"(oversize_policy='{self._config.oversize_policy}')"
        )

    def _fragment_tokens(self, role: str, text: str, add_special: bool = False) -> list[int]:
        """Tokenize one message: tags special-on, content special-off."""
        prefix, content, suffix = self._formatter.parts(role, text)
        return self._context.tokenize_fragment(prefix, content, suffix, add_special=add_special)


def _as_events(stream):
    """Re-shape a delta stream into events, ending on the calls it stopped at.

    The wrapped generator is closed whatever happens, so an early close still
    ends the turn and releases the wrapper's lock.
    """
    turn = None
    try:
        while True:
            try:
                delta = next(stream)
            except StopIteration as done:
                turn = done.value
                break
            yield delta
    finally:
        stream.close()
    for call in turn.tool_calls:
        yield call
    return turn


def _sum_generations(generations: list[Turn], tool_limit_reached: bool) -> Turn:
    """Sum the generations of one exchange into the Turn that describes it."""
    return Turn(
        text="".join(t.text for t in generations),
        n_prefilled_tokens=sum(t.n_prefilled_tokens for t in generations),
        n_generated_tokens=sum(t.n_generated_tokens for t in generations),
        n_evicted_messages=sum(t.n_evicted_messages for t in generations),
        stop_reason=generations[-1].stop_reason,
        tool_calls=[call for t in generations for call in t.tool_calls],
        tool_limit_reached=tool_limit_reached,
    )


def _result_text(result) -> str:
    """One tool's return value as the text the model is handed."""
    return result if isinstance(result, str) else json.dumps(result)
