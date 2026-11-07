"""ChatWrapper: the public API, prefilling as little as possible.

``begin`` resets and prefills the system prompt plus what recent history fits;
``inject`` prefills one message without generating; ``reply_without_tools``
and ``reply_without_tools_stream`` prefill only the new request and then
generate, decoding the reply again only where the template writes a past reply
otherwise. Crossing either eviction limit removes the oldest non-system
messages and shifts the survivors down, reusing their KV.

Two invariants hold after every action, and the tests enforce both: the physical
cache equals the concatenation of the message table, and every prefill appends
at the cache's end.

The turn-closing tag is decoded after the post-reply eviction rather than before
it. See the README, "How eviction works".
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

from .call_parsing import parse_calls
from .config import Config, Sampling
from .context import GenerationAccumulator, KVContext, _stop_hit
from .messages import Message, MessageTable, n_newest_that_fit
from .reasoning import AnswerOpening, ThoughtFilter, begins_in_thought, turn_end_test
from .resolved_template import ResolvedTemplate, ensure_context_answers
from .template import TemplateFragments, ToolCallFragments
from .tools import ToolCall, ToolRegistry, ToolResult, ToolSpec, ToolTurn, result_text


# The name a result block carries where no real one is to be had: answering a
# call that could not be read that far. Which name is used does not change what
# a template renders around it.
_UNKNOWN_FUNCTION = "unknown"

# What the wrapper calls on its cache context, checked where one is adopted
# rather than at the first call that reaches what is missing. Resolving the
# template checks what it calls for itself.
_CACHE_CONTEXT_SURFACE = ("tokenize", "tokenize_fragment", "prefill", "reset",
                          "warmup", "generate", "apply_eviction", "remove_from",
                          "can_shift", "close")


class ContextOverflowError(RuntimeError):
    """A request would leave fewer than ``min_reply_tokens`` free for the reply."""


@dataclass
class Reply:
    """What one request produced.

    One generation, or, where the request took several, all of them: the counts
    are then sums and ``stop_reason`` is the last one's.

    Attributes:
        text: The assistant's reply, trimmed at any stop string, its thought
            left out.
        n_prefilled_tokens: Tokens decoded for the request, the opening of
            each resumption, and a reply decoded again as a past reply.
        n_generated_tokens: Tokens sampled.
        n_evicted_messages: Messages evicted to bring the cache back inside both
            eviction limits.
        stop_reason: One of ``"end_of_turn"``, ``"stop_string"``,
            ``"token_limit"``, ``"interrupted"``, ``"tool_call"`` or
            ``"unreadable_tool_call"``.
        tool_calls: The calls the model made, in the order it made them.
            Reported whatever became of them, so a call that was refused rather
            than run still says what the model asked for.
        tool_limit_reached: Whether ``max_calls`` ran out, so the final
            generation ran with the call tokens withheld. Always ``False``
            where no call was asked for.
        n_reasoning_tokens: Of the tokens sampled, about how many were thought:
            the streamed pieces that held any.
    """

    text: str
    n_prefilled_tokens: int
    n_generated_tokens: int
    n_evicted_messages: int
    stop_reason: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_limit_reached: bool = False
    n_reasoning_tokens: int = 0


_Delta = TypeVar("_Delta")


class ReplyStream(Generic[_Delta]):
    """One request's deltas, and the :class:`Reply` they add up to.

    Iterate it for what the model produces, then read :attr:`reply` for the
    summary. Closing it early ends the turn cleanly and leaves :attr:`reply`
    unset, there being no summary of a reply nobody read to the end. Closing is
    what an abandoned stream needs, so it is also a context manager.

    Attributes:
        reply: The summary, ``None`` until the stream has ended of its own
            accord.
    """

    def __init__(self, deltas: Generator[_Delta, None, Reply]) -> None:
        self._deltas = deltas
        self.reply: Reply | None = None

    def __iter__(self) -> ReplyStream[_Delta]:
        return self

    def __next__(self) -> _Delta:
        # A second pass over a finished stream yields nothing and leaves the
        # summary standing, rather than replacing it with the None a spent
        # generator returns.
        if self.reply is not None:
            raise StopIteration
        try:
            return next(self._deltas)
        except StopIteration as ended:
            self.reply = ended.value
            raise

    def close(self) -> None:
        """Stop reading and end the turn. Idempotent, as a generator's is."""
        self._deltas.close()

    def __enter__(self) -> ReplyStream[_Delta]:
        return self

    def __exit__(self, *_) -> None:
        self.close()


class ChatWrapper:
    """Stateful single-conversation chat over a cache context, with KV reuse."""

    # ----- setting up ----------------------------------------------------------
    def __init__(
        self,
        config: Config | None = None,
        context=None,
        *,
        fragments: TemplateFragments | None = None,
        tool_calls: ToolCallFragments | None = None,
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
            tool_calls: Explicit call tags, and optionally the argument
                spelling inside them, for a model whose template describes
                neither. Without the spelling, such a model still detects its
                calls and reports them unreadable.
            on_message: Invoked with ``(role, text)`` for every genuine new
                message, so not for the system prompt or a ``begin`` replay.
                Runs under the lock, so it must not call back into the wrapper.
            on_load_progress: Invoked during model loading with a fraction in
                ``[0.0, 1.0]``. Must not raise. Ignored when ``context`` is
                supplied.
            **kwargs: :class:`~llama_chat.config.Config` field overrides, used
                to build the config when ``config`` is not given.

        Raises:
            TypeError: If both ``config`` and config ``**kwargs`` are passed,
                or if ``context`` does not answer everything the wrapper calls
                on it.
            ValueError: If the chat template cannot be recovered, or if it
                fails validation against the model.
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
        # Everything that can be settled about this model is settled here, so
        # that what fails, fails while the wrapper is being built.
        try:
            ensure_context_answers(self._context, _CACHE_CONTEXT_SURFACE)
            if hasattr(self._context, "save_state"):
                ensure_context_answers(self._context, ("restore_state",))
            self._template = ResolvedTemplate(
                self._context, fragments=fragments, tool_calls=tool_calls)
            # The conversation's own, set again by every begin.
            self._formatter = self._template.formatter()
        except BaseException:
            if owns_context:
                self._context.close()
            raise
        self._table = MessageTable()
        self._registry = None
        # An open turn and the calls owing a result are not the same thing: the
        # last result empties the list while the turn is still open.
        self._reply_open = False
        self._pending_calls = []
        self._batch = []                # the names of the calls made together
        self._reasoning = False
        # The template text an open turn ends with, which a resumption opens on.
        self._turn_ends_with = ""
        # For a cache that cannot shift: the last two states it rested in, each
        # with how many tokens it held then, oldest first, so the newest reply
        # can be changed without rebuilding the conversation: one from before
        # it and one from after. See the README, "How eviction works".
        self._rest_states = []
        # Not read off the table: `declare_unknown_tools` appends a system-role message,
        # so one stray call would make an unbegun table look begun.
        self._begun = False
        self._warmed = False
        self._closed = False
        # One KV sequence is shared across generations, so concurrent decodes
        # would corrupt it. It is held for a whole generation, not a decode.
        self._lock = threading.Lock()

    def begin(self, system_prompt: str,
              messages: list[tuple[str, str] | ToolTurn] | None = None,
              tools: ToolRegistry | list[ToolSpec] | None = None, *,
              reasoning: bool = False) -> None:
        """Reset the conversation: prefill the system prompt and recent history.

        The first call also warms the generation graph, so the first request
        streams without a cold start. Every other action refuses until this has
        run, because only this prefills the sequence opener and the system
        prompt.

        Args:
            system_prompt: Always kept, never evicted. Empty, and with no
                tools, it is no system message at all, and the conversation
                opens as the template opens one without.
            messages: Optional history, oldest first: ``(role, text)`` pairs,
                and a :class:`~llama_chat.tools.ToolTurn` for an assistant turn
                that called tools, written as this model's template writes one.
                A history that ends on a tool turn's result leaves that turn
                open, as a result written into a live turn does, for
                :meth:`resume_until_tool` to go on with. Only the most recent
                that fit alongside the system prompt are prefilled, under both
                eviction limits; older excess is dropped.
            tools: Tools to declare, rendered into the system turn where this
                model's template puts them, which is the one placement that
                costs no extra prefill and that no eviction can take back. From
                a whole :class:`~llama_chat.tools.ToolRegistry`, what it marks
                for the system prompt; the rest waits for
                :meth:`declare_unknown_tools`. A registry is also kept, so
                :meth:`reply_with_tools` needs none of its own; a list of specs
                is declared and not kept. Refused up front where this model
                could not use them, rather than at the first call.
            reasoning: Whether the model reasons before each answer, as its
                template turns that on. It belongs to the conversation, so it
                changes only here. Every reply's thought is kept out of what
                it yields, either way.

        Raises:
            ValueError: If the system prompt alone exceeds
                ``evict_above_tokens``, or if ``tools`` names anything or the
                history holds a tool turn and this model cannot call tools: no
                call in its template, no declarations, no recoverable argument
                spelling, or no result block, or if ``reasoning`` is asked of a
                model that cannot reason. The previous conversation is left
                intact either way.
        """
        messages = messages or []
        with self._lock:
            self._ensure_open()
            # Validate before resetting: a failed begin must not destroy the
            # conversation it was about to replace.
            registry = tools if isinstance(tools, ToolRegistry) else None
            specs = _specs_for_system_prompt(tools)
            if specs:
                self._template.ensure_tools_usable()
            formatter = self._template.formatter(reasoning)
            sys_tokens = self._system_tokens(formatter, system_prompt, specs, reasoning)
            if len(sys_tokens) > self._config.evict_above_tokens:
                raise ValueError(
                    f"system prompt needs {len(sys_tokens)} tokens but the "
                    f"eviction threshold is {self._config.evict_above_tokens}; "
                    "shorten the prompt or raise context_size/evict_above_fraction"
                )
            ends_open = (bool(messages) and isinstance(messages[-1], ToolTurn)
                         and bool(messages[-1].steps)
                         and isinstance(messages[-1].steps[-1], ToolResult))
            last = len(messages) - 1
            hist = [self._history_tokens(formatter, item, ends_open and i == last,
                                         reasoning)
                    for i, item in enumerate(messages)]
            hist_tokens = [tokens for tokens, _ in hist]

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
            self._formatter, self._reasoning = formatter, reasoning
            if not self._warmed:
                self._context.warmup()
                self._warmed = True

            # One decode for the whole conversation. A turn left open is
            # generated into next, so its last token's logits are kept.
            ends_open = ends_open and bool(kept_messages)
            all_tokens = list(sys_tokens)
            for toks in kept_tokens:
                all_tokens.extend(toks)
            self._context.prefill(all_tokens, start_pos=0, want_logits=ends_open)

            self._table.append(Message(
                "system", system_prompt, sys_tokens,
                declares=tuple(spec.name for spec in specs or [])))
            for item, toks in zip(kept_messages, kept_tokens):
                role, text = (("assistant", item.text) if isinstance(item, ToolTurn)
                              else item)
                self._table.append(Message(role, text, toks))
            self._reply_open = ends_open
            self._turn_ends_with = hist[-1][1] if ends_open else ""
            self._begun = True
            self._rest_states = []
            if not ends_open:
                self._save_rest()

    # ----- adding context ------------------------------------------------------
    def inject(self, text: str, role: str = "user") -> int:
        """Prefill one message as context without generating.

        Evicts the oldest messages until the new one fits inside both eviction
        limits, then prefills it.

        Returns:
            The number of messages evicted to make room.

        Raises:
            ValueError: If no conversation has been begun, if a tool call has
                left a reply open, or if the message cannot fit even after
                evicting every non-system message. Raised before anything is
                evicted.
        """
        with self._lock:
            self._ensure_open()
            self._ensure_begun("inject into")
            self._ensure_no_open_reply("inject")
            if role == "system":
                prefix, suffix = self._later_system_tags()
                tokens = self._context.tokenize_fragment(
                    prefix, self._template.fragments.content(role, text), suffix)
            else:
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
            self._save_rest()
            return evicted

    # ----- asking: without tools -----------------------------------------------
    def reply_without_tools_stream(
        self, text: str, *, max_reply_tokens: int | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
        sampling: Sampling | None = None,
    ) -> ReplyStream[str]:
        """Add a user request and stream the reply token by token.

        Lazy: nothing happens, the headroom check included, until the first
        iteration. Closing the stream early still terminates the turn and
        records what was generated, as any reply is recorded, so the next
        generation stays valid.

        Runs no tool, ever, not even one :meth:`begin` was given. A call the
        model makes anyway is kept out of the reply text, told that the call
        failed and that no tool can be run, and the reply goes on from there, so
        the model can say in words that it could not do the thing, a blank line
        parting the text before the call from the text after it where the two
        would join without whitespace. That costs one further generation and is
        the only case where this is not a single one.

        The stream must be driven from one thread at a time. To interrupt from
        another, pass ``cancel``. An abandoned stream holds the wrapper's lock
        until it is collected, so exhaust it, close it, or use it in a ``with``.

        Args:
            text: The user request text, tokenized with special-token parsing
                off. See the README, "Special tokens in message content".
            max_reply_tokens: Override for the per-generation cap.
            stop: Extra stop strings for this request, added to the config's.
            cancel: Setting it, from any thread, stops generation within one
                token and closes the turn with ``stop_reason == "interrupted"``.
            sampling: This request's own sampling settings, in place of the
                config's for the fields it sets.

        Returns:
            A :class:`ReplyStream` of the text deltas, whose ``reply`` is the
            :class:`Reply` once it has ended.

        Raises:
            ValueError: If no conversation has been begun, or if a tool call
                has left a reply open.
            ContextOverflowError: If too little room remains for the reply, or
                if the error answering a call no longer fits. The turn is then
                open and cannot be continued.
        """
        return ReplyStream(self._generate_until_answered(
            text, registry=None, max_reply_tokens=max_reply_tokens,
            stop=stop, cancel=cancel, sampling=sampling))

    def reply_without_tools(
        self, text: str, *, max_reply_tokens: int | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
        sampling: Sampling | None = None,
    ) -> Reply:
        """Add a user request and generate a reply, no tool of any kind run.

        :meth:`reply_without_tools_stream`, drained.
        """
        stream = self.reply_without_tools_stream(
            text, max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
            sampling=sampling)
        for _ in stream:
            pass
        return stream.reply     # set, the stream having ended of its own accord

    # ----- asking: with tools --------------------------------------------------
    def reply_with_tools_stream(
        self, text: str, registry: ToolRegistry | None = None, *,
        max_calls: int | None = None, max_reply_tokens: int | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
        sampling: Sampling | None = None) -> ReplyStream[str]:
        """Reply to ``text``, running whatever tools the model calls on the way.

        Yields the reply's text deltas and nothing else; the stream's ``reply``
        sums every generation. Where the text before a call and the text after
        it would join without whitespace, a blank line goes between them. The
        last permitted round withholds the call tokens, so the turn always
        closes.

        Args:
            text: The request.
            registry: What to run calls against. Defaults to the one
                :meth:`begin` was given. Without either, a call the model
                invented is answered saying so and the reply goes on.
            max_calls: How many tools one turn may use. ``None``, the
                default, sets no ceiling: the exchange runs until the model
                stops calling, bounded only by the context. A call past a
                ceiling is answered with an error result rather than run.
            max_reply_tokens, stop, cancel, sampling: As
                :meth:`reply_without_tools_stream`, and applied to every
                generation of the exchange, so ``max_reply_tokens`` bounds each
                one rather than the whole.

        Raises:
            ValueError: If no conversation has been begun, or if a registry
                holding tools is given to a model that could never call one.
                Driving the loop yourself with :meth:`reply_until_tool` is not
                refused, because nothing there promises a tool will run.
            ContextOverflowError: If a result no longer fits in what is left of
                the context. The turn is then open and cannot be continued.
        """
        return ReplyStream(self._generate_until_answered(
            text,
            registry=registry if registry is not None else self._registry,
            max_calls=max_calls, max_reply_tokens=max_reply_tokens, stop=stop,
            cancel=cancel, sampling=sampling))

    def reply_with_tools(self, text: str, registry: ToolRegistry | None = None,
                         *, max_calls: int | None = None,
                         max_reply_tokens: int | None = None,
                         stop: list[str] | None = None,
                         cancel: threading.Event | None = None,
                         sampling: Sampling | None = None) -> Reply:
        """Reply to ``text``, running whatever tools the model calls.

        :meth:`reply_with_tools_stream`, drained. Arguments and failures are
        its own.

        Returns:
            A :class:`Reply` summing every generation; the reply is its ``text``.
        """
        stream = self.reply_with_tools_stream(
            text, registry, max_calls=max_calls,
            max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
            sampling=sampling)
        for _ in stream:
            pass
        return stream.reply     # set, the stream having ended of its own accord

    # ----- tools: driving the loop yourself ------------------------------------
    def reply_until_tool(self, text: str, *,
                         max_reply_tokens: int | None = None,
                         stop: list[str] | None = None,
                         cancel: threading.Event | None = None,
                         sampling: Sampling | None = None,
                         ) -> ReplyStream[str | ToolCall]:
        """Stream the reply, handing you each tool call as the model makes it.

        Yields ``str`` deltas, then one
        :class:`~llama_chat.tools.ToolCall` per call the model made. Answer each
        with :meth:`inject_tool_result`, in order, then :meth:`resume_until_tool`.
        Asking to see the calls is what holds the turn open for them, so this
        and :meth:`resume_until_tool` are the only entry points that leave one
        unfinished.

        Raises:
            ValueError: If no conversation has been begun, or if a tool call
                has left a reply open.
            ContextOverflowError: If too little room remains for the reply.
        """
        return ReplyStream(_deltas_then_calls(self._request_and_generate(
            text, max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
            sampling=sampling)))

    def resume_until_tool(self, *, max_reply_tokens: int | None = None,
                          stop: list[str] | None = None,
                          cancel: threading.Event | None = None,
                          sampling: Sampling | None = None,
                          ) -> ReplyStream[str | ToolCall]:
        """Resume the reply a tool call interrupted.

        Yields what :meth:`reply_until_tool` yields. Every result must be in
        first, and this generation may itself end on another call, leaving the
        turn open again for the same treatment.

        Raises:
            ValueError: If no reply is open, or if a call is still owed a
                result.
        """
        return self._resume_until_tool(
            allow_tools=True, max_reply_tokens=max_reply_tokens, stop=stop,
            cancel=cancel, sampling=sampling)

    def _resume_until_tool(self, *, allow_tools: bool,
                           **kwargs) -> ReplyStream[str | ToolCall]:
        """Resume, with the call tokens withheld where ``allow_tools`` is False.

        Withholding is the exchange loop's business, not a caller's: it is what
        makes the last permitted round certain to close the turn.
        """
        return ReplyStream(_deltas_then_calls(
            self._resume(allow_tools=allow_tools, **kwargs)))

    def inject_tool_result(self, call: ToolCall | str, result) -> int:
        """Hand the model what the tool it called returned.

        Prefilled inside the turn the call left open, wrapped the way this
        model's own template wraps a result, and tokenized with special-token
        parsing off. Nothing is evicted while a turn is open, so such a turn
        grows towards ``context_size``. See the README, "Tool calling".

        Args:
            call: The call being answered, as it was handed to you, or its name
                alone. Either way it must be the oldest unanswered one.
            result: What it returned. Anything but a string is sent as JSON.

        Returns:
            How many messages were evicted to make room, reported as
            :meth:`inject` reports it. Nothing is evicted while a turn is open,
            the result going inside it, so this is ``0``.

        Raises:
            ValueError: If no call is waiting, or this is not the one owed.
            ContextOverflowError: If not even the wrapping fits in what is left
                of the context.
        """
        name = call.name if isinstance(call, ToolCall) else call
        with self._lock:
            self._ensure_open()
            if not self._pending_calls:
                raise ValueError("no tool call is waiting for a result")
            owed = self._pending_calls[0]
            if name != owed.name or (isinstance(call, ToolCall) and call != owed):
                raise ValueError(
                    f"the waiting call is {owed.name!r}, not {name!r}; results "
                    "are owed in the order the model asked for them")
            # Popped only once the write has succeeded, so a refused result
            # leaves the call still owed.
            self._write_tool_result(name, result)
            self._pending_calls.pop(0)      # results are owed in call order
            return 0

    def _write_tool_result(self, name: str, result) -> None:
        """Prefill one result block into the open turn. The lock must be held.

        Raises:
            ValueError: If this model's template renders no result block.
            ContextOverflowError: If not even the wrapping fits in what is left
                of the context.
        """
        # Everything that can fail happens before the first mutation.
        # A result answers the oldest call owed, so its place in the batch is
        # how many of the batch are no longer owed.
        batch = self._batch if name in self._batch else [name]
        index = len(batch) - len(self._pending_calls) if batch is self._batch else 0
        prefix, suffix = self._template.result_wrapping(
            name, self._reasoning, batch=batch, index=index)
        tags = self._context.tokenize(prefix, add_special=False, parse_special=True)
        closing = self._context.tokenize(suffix, add_special=False, parse_special=True)
        body = self._context.tokenize(result_text(result), add_special=False,
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
        self._context.prefill(tokens, start_pos=self._table.total, want_logits=True)
        self._table.extend_last(tokens)
        self._turn_ends_with = suffix

    # ----- shortening the newest reply -----------------------------------------
    def truncate_last_reply(self, keep_text: str) -> None:
        """Shorten the newest assistant message to ``keep_text``.

        For a reply that was cut short where it was being read or heard: the
        model is left holding only what the reader got. The cut is at the
        cache's end, so the tokens the shortened message shares with the old
        one are not decoded again; the rest of it, its closing tag at least, is.
        A cache that cannot shift is rebuilt from the whole conversation
        instead, as its eviction is.

        Args:
            keep_text: What stays, a prefix of the message's text. The whole
                text changes nothing.

        Raises:
            ValueError: If no conversation has been begun, if a reply is open,
                if the newest message is not the assistant's or holds a tool
                call, or if ``keep_text`` is not a prefix of its text.
        """
        with self._lock:
            msg = self._newest_reply("truncate")
            if not msg.text.startswith(keep_text):
                raise ValueError("keep_text is not a prefix of the newest reply")
            if keep_text == msg.text:
                return
            self._rewrite_newest(msg, self._fragment_tokens("assistant", keep_text),
                                 keep_text)
            self._save_rest()

    def _newest_reply(self, action: str):
        """The newest message, refused unless it is an assistant reply that may change."""
        self._ensure_open()
        self._ensure_begun(f"{action} in")
        self._ensure_no_open_reply(f"{action} a reply")
        msg = self._table.messages[-1]
        if msg.role != "assistant":
            raise ValueError("the newest message is not an assistant reply")
        if self._holds_call(msg):
            raise ValueError("the newest reply holds a tool call, which would "
                             "be separated from its result")
        return msg

    def _holds_call(self, msg) -> bool:
        tool_tokens = self._template.tool_tokens
        return tool_tokens is not None and tool_tokens[0] in msg.token_ids

    def _later_system_tags(self) -> tuple[str, str]:
        """The tags of a system message written after the conversation has begun.

        Raises:
            ValueError: If the model's template writes no such turn.
        """
        tags = self._template.later_system_tags
        if tags is not None:
            return tags
        if self._template.preamble() is None:       # no template: the fragments'
            return self._template.fragments.system_prefix, \
                self._template.fragments.system_suffix
        raise ValueError("this model's template writes no system message after "
                         "the conversation has begun")

    def _remove_newest(self, msg) -> None:
        """Drop the newest message from the table and the cache."""
        self._table.remove_last()
        self._cut_back_to(msg.pos_start)

    def _rewrite_newest(self, msg, tokens: list[int], text: str) -> int:
        """Give the newest message ``tokens``, decoding only where they differ.

        Returns:
            How many tokens were decoded.
        """
        shared = _shared_prefix_length(msg.token_ids, tokens)
        self._table.replace_last(tokens, text)
        decoded = self._cut_back_to(msg.pos_start + shared)
        if decoded is None:             # rebuilt, the new tokens with the rest
            return self._table.total
        self._context.prefill(tokens[shared:], start_pos=msg.pos_start + shared,
                              want_logits=False)
        return decoded + len(tokens) - shared

    def _cut_back_to(self, pos: int) -> int | None:
        """Make the cache hold the table's first ``pos`` tokens and nothing after.

        A cut where the cache takes one; else from a state it rested in, where
        one at or before ``pos`` is saved; else a rebuild of the whole table.
        See the README, "How eviction works".

        Returns:
            How many tokens were decoded to get there, or ``None`` where the
            whole table was rebuilt.
        """
        if self._context.can_shift or getattr(self._context, "can_cut_tail", False):
            self._context.remove_from(pos)
            return 0
        decoded = self._back_to(pos)
        if decoded is None:
            self._rebuild_cache()
        return decoded

    def declare_unknown_tools(self, tools: ToolRegistry | list[ToolSpec]) -> int:
        """Tell the model about the tools it has not been told about yet.

        Costs one prefill, and writes an ordinary message, so this declaration
        can be evicted later. One that has to survive belongs in ``begin``.

        Args:
            tools: What to declare: a whole
                :class:`~llama_chat.tools.ToolRegistry`, of which everything it
                holds is taken, or the specs themselves. Nothing to
                declare does nothing, and neither does a tool the conversation
                already declares, wherever that declaration sits: the system
                turn, which cannot be evicted, or a message this method wrote.
                One whose message has since been evicted counts as unknown
                again and is written back. The registry is not
                kept, unlike ``begin``'s: which one a turn runs against is that
                method's decision, and a declaration adds names rather than
                replacing what can run.

        Returns:
            How many messages were evicted to make room.

        Raises:
            ValueError: If no conversation has been begun, if a tool call has
                left a reply open, if this model cannot call tools at all (see
                :meth:`begin`), if its template declares no tools, or if the
                declarations cannot fit beside the system prompt.
        """
        with self._lock:
            self._ensure_open()
            known = self._declared_names()
            specs = [spec for spec in _all_specs(tools) or []
                     if spec.name not in known]
            if not specs:
                return 0
            self._ensure_begun("declare into")
            self._ensure_no_open_reply("declare tools")
            self._template.ensure_tools_usable()
            block = self._template.declaration_block(specs)
            prefix, suffix = self._later_system_tags()
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
            self._table.append(Message("system", "", tokens,
                                       declares=tuple(spec.name for spec in specs)))
            self._save_rest()
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
        """Whether a generation holds the wrapper. Advisory: it can be stale."""
        if not self._lock.acquire(blocking=False):
            return True
        self._lock.release()
        return False

    def wait_idle(self, timeout_s: float | None = None) -> bool:
        """Block until no generation holds the wrapper. Advisory, as :attr:`busy` is.

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

        Safe from any thread, including while another is mid-generation.
        """
        return self._table.snapshot_rows()

    @property
    def can_call_tools(self) -> bool:
        """Whether tools can be used with this model, every step of the way.

        Four things have to hold, and all are settled at construction: the
        template describes a call, it renders the declarations that tell the
        model a tool exists, the argument spelling was recovered, and it renders
        a block to answer a call in. True here is exactly what :meth:`begin`
        accepts ``tools`` for. Where only the spelling is missing a call is
        still detected, so :attr:`Reply.stop_reason` can still be
        ``"unreadable_tool_call"``, but nothing runnable comes out of it. Supply
        the spelling as the body of ``ChatWrapper(tool_calls=...)`` to make
        this ``True``.
        """
        return self._template.can_call_tools

    @property
    def awaiting_tool_result(self) -> bool:
        """Whether any call is still holding the current turn open."""
        return bool(self._pending_calls)

    @property
    def can_reason(self) -> bool:
        """Whether :meth:`begin` may be asked to have the model reason."""
        return self._template.can_reason

    @property
    def reply_open(self) -> bool:
        """Whether a reply is unfinished, so no new message may be added.

        Outlasts :attr:`awaiting_tool_result`: the last result empties the calls
        while the turn stays open until :meth:`resume_until_tool` closes it. While
        this is ``True``, :meth:`reply_without_tools`, :meth:`inject` and
        :meth:`declare_unknown_tools` refuse.
        """
        return self._reply_open

    # ----- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        """Free the underlying context and model. Idempotent and thread-safe.

        Waits for any in-flight generation. Closes a context passed to the
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
    def _generate_until_answered(
        self, text: str, *, registry: ToolRegistry | None,
        max_calls: int | None = None,
        **kwargs) -> Generator[str, None, Reply]:
        """Generate, settle whatever the model calls, and go on until it answers.

        Args:
            registry: What to run calls against, already resolved. ``None``
                refuses every call with an error result rather than falling back
                to anything, which is what makes a tool-free entry point
                tool-free.
            max_calls: How many tools one turn may use, ``None`` for no
                ceiling. Nothing is run where ``registry`` is ``None``, so it
                does not bind there.
            text, **kwargs: As :meth:`reply_without_tools_stream`.

        Raises:
            ValueError: If ``registry`` holds tools this model could never
                call, which is what
                :meth:`~llama_chat.resolved_template.ResolvedTemplate.ensure_tools_usable`
                decides.
        """
        # A registry is the same promise a declaration is, so it is checked the
        # same way: an exchange that cannot run a tool is refused before it
        # starts, not after the model has asked for one.
        if registry is not None and len(registry) > 0:
            self._template.ensure_tools_usable()
        stream = self.reply_until_tool(text, **kwargs)
        # An absent ceiling counts down without ever reaching zero, so the
        # three places that read it need no case of their own.
        calls_left = math.inf if max_calls is None else max_calls
        any_call_asked = False
        generations: list[Reply] = []
        try:
            while True:
                asked, generation = yield from self._read_generation(
                    stream, after="".join(g.text for g in generations))
                generations.append(generation)
                if not asked:
                    if self.reply_open:
                        # An unreadable call: its result is already written, and
                        # the turn wants the reply that follows it. Nothing is
                        # owed, since nothing could be read, so the resume is
                        # free to run. It spends budget like any other failed
                        # call, so a model that cannot write a readable one
                        # retries at most `max_calls` times.
                        any_call_asked = True
                        if registry is not None and calls_left > 0:
                            calls_left -= 1
                        stream = self._resume_until_tool(
                            allow_tools=registry is not None and calls_left > 0,
                            **kwargs)
                        continue
                    # The ceiling binds only once a call has been asked for:
                    # without one the last generation kept its call tokens.
                    return _sum_generations(generations,
                                            any_call_asked and calls_left == 0)
                any_call_asked = True
                settled, calls_left = self._settle_calls(
                    asked, registry, calls_left, max_calls)
                may_call_again = registry is not None and calls_left > 0
                for call, result in settled:
                    self.inject_tool_result(
                        call, _explain_failure(result, may_call_again))
                stream = self._resume_until_tool(allow_tools=may_call_again, **kwargs)
        finally:
            stream.close()   # the current generation holds the wrapper's lock

    def _read_generation(self, stream, after: str):
        """Yield one generation's text, and return what it left behind.

        Driven by hand rather than with ``for``, which would discard the
        :class:`Reply` the generation returns. Where ``after``, the text shown
        before it, and this text would join without whitespace, a blank line
        goes between them.

        Returns:
            The calls the model asked for, in the order it asked, and the
            :class:`Reply` describing the generation, its text starting with
            the blank line where one was put.
        """
        asked, separator = [], None if after and not after[-1].isspace() else ""
        while True:
            try:
                event = next(stream)
            except StopIteration as ended:
                return asked, replace(ended.value, text=(separator or "") + ended.value.text)
            if isinstance(event, ToolCall):
                asked.append(event)
                continue
            if separator is None and event:
                separator = "" if event[0].isspace() else "\n\n"
                event = separator + event
            yield event

    def _settle_calls(self, asked: list[ToolCall], registry: ToolRegistry | None,
                      calls_left, max_calls: int | None):
        """Decide what every call of one round gets back, running what may run.

        Settled before any of it is written, because what a failure tells the
        model to do next depends on the budget the last call leaves. In the
        order asked, which the template renders in.

        Returns:
            The ``(call, result)`` pairs, and what is left of the budget.
        """
        settled = []
        for call in asked:
            if registry is None:
                # Not "no tools exist": `begin` may have declared some. Raising
                # would leave the turn open with no way to end it.
                result = {"error": "tools_not_run",
                          "message": "no tool can be run for this request"}
            elif calls_left > 0:
                result = registry.call(call)
                calls_left -= 1
            else:
                result = {"error": "tool_limit_reached",
                          "message": "no tool calls left in this turn; "
                                     f"the limit is {max_calls}"}
            settled.append((call, result))
        return settled, calls_left

    def _request_and_generate(
        self, text: str, *, max_reply_tokens: int | None = None,
        stop: list[str] | None = None, cancel: threading.Event | None = None,
        sampling: Sampling | None = None,
    ) -> Generator[str, None, Reply]:
        """Add the user request, then generate one reply from the cache's end.

        One generation, so a call it stops on leaves the turn open for a result.
        Every public entry point reaches generation through here.
        """
        with self._lock:
            self._ensure_open()
            self._ensure_begun("generate into")
            self._ensure_no_open_reply("start a request")
            user_tokens = self._fragment_tokens("user", text)
            open_tokens = self._context.tokenize(self._formatter.generation_prompt(), add_special=False)
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

            return (yield from self._generate_into_reply(
                open_tokens=open_tokens, close_tokens=close_tokens,
                n_prefilled_tokens=n_prompt, max_reply_tokens=max_reply_tokens, stop=stop,
                cancel=cancel, allow_tools=True, sampling=sampling,
                decoded=self._formatter.generation_prompt()))

    def _resume(self, *, max_reply_tokens: int | None = None,
                stop: list[str] | None = None,
                cancel: threading.Event | None = None,
                allow_tools: bool = True,
                sampling: Sampling | None = None):
        """Generate on into the open reply, adding no message of its own."""
        with self._lock:
            self._ensure_open()
            if self._pending_calls:
                raise ValueError(f"{len(self._pending_calls)} call(s) still owe "
                                 f"a result")
            if not self._reply_open:
                raise ValueError("no reply is open; start one with reply_until_tool()")
            close_tokens = self._context.tokenize(self._formatter.assistant_close(),
                                              add_special=False)
            # See the README, "A resumed turn opens as a new one does".
            opening = self._formatter.prompt_beyond_tag()
            n_opened = 0
            if not self._turn_ends_with.endswith(opening):
                open_tokens = self._context.tokenize(opening, add_special=False)
                self._context.prefill(open_tokens, start_pos=self._table.total,
                                      want_logits=True)
                self._table.extend_last(open_tokens)
                self._turn_ends_with += opening
                n_opened = len(open_tokens)
            return (yield from self._generate_into_reply(
                open_tokens=[], close_tokens=close_tokens, n_prefilled_tokens=n_opened,
                max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
                allow_tools=allow_tools, sampling=sampling,
                decoded=self._turn_ends_with))

    def _history_tokens(self, formatter, item: tuple[str, str] | ToolTurn,
                        open_turn: bool, reasoning: bool) -> tuple[list[int], str]:
        """One message of a ``begin`` history, as it is prefilled.

        Returns:
            The tokens, and the template text an open turn ends with, empty for
            any other.
        """
        if not isinstance(item, ToolTurn):
            prefix, content, suffix = formatter.parts(*item)
            return self._context.tokenize_fragment(prefix, content, suffix), ""
        # Refuses a model that cannot call tools.
        pieces = self._template.past_tool_turn(item, open_turn, reasoning)
        open_id, close_id = self._template.tool_tokens
        tags = {text: token for token in (open_id, close_id)
                if (text := self._context.token_text(token))}
        between_tags = re.compile("(" + "|".join(map(re.escape, tags)) + ")")
        tokens: list[int] = []
        for text, own in pieces:
            if own:
                tokens += self._context.tokenize(text, add_special=False,
                                                 parse_special=False)
                continue
            # The call tags as the single tokens a generated call holds.
            for part in between_tags.split(text):
                if part in tags:
                    tokens.append(tags[part])
                elif part:
                    tokens += self._context.tokenize(part, add_special=False,
                                                     parse_special=True)
        ending = pieces[-1][0] if open_turn and pieces and not pieces[-1][1] else ""
        return tokens, ending

    def _system_tokens(self, formatter, system_prompt: str,
                       specs: list[ToolSpec] | None, reasoning: bool) -> list[int]:
        """The system turn, with any declarations where the template puts them.

        The declarations join the suffix rather than the content: they are the
        model's own tags and tokenize special-on, the prompt must not. An empty
        prompt with nothing declared is no system message, and opens the
        conversation as the template opens one without.
        """
        preamble = self._template.preamble(reasoning)
        if not system_prompt and preamble is not None:
            if specs:
                preamble = self._template.preamble_declaring(specs, reasoning)
            return self._context.tokenize_fragment(preamble, "", "", add_special=True)
        if preamble is not None and not self._template.has_system_turn:
            raise ValueError(
                "this model's template has no system turn of its own: it writes a "
                "system prompt into the first user turn, which a system prompt "
                "kept for the whole conversation cannot be")
        prefix, content, suffix = formatter.parts("system", system_prompt)
        if not specs:
            return self._context.tokenize_fragment(prefix, content, suffix,
                                                   add_special=True)
        # Rendered whole: where declarations go is the template's business.
        head, tail = self._template.system_turn_declaring(specs, reasoning)
        return self._context.tokenize_fragment(head, content, tail,
                                               add_special=True)

    def _generate_into_reply(self, *, open_tokens: list[int],
                             close_tokens: list[int], n_prefilled_tokens: int,
                             max_reply_tokens: int | None,
                             stop: list[str] | None,
                             cancel: threading.Event | None,
                             allow_tools: bool,
                             sampling: Sampling | None = None,
                             decoded: str = "",
                             ) -> Generator[str, None, Reply]:
        """Generate from the cache's end and record what it left, as a Reply.

        Shared by a fresh turn, whose generation prompt is prefilled but not yet
        in the table, and a resumed one, which brings none of its own.

        Args:
            open_tokens: The generation prompt, empty when resuming.
            close_tokens: The turn terminator, decoded once the turn closes.
            n_prefilled_tokens: What the caller decoded to get here, for the Reply.
            max_reply_tokens, stop, cancel, sampling: As
                :meth:`reply_without_tools_stream`.
            allow_tools: ``False`` withholds the call tokens, so this generation
                cannot end on a call and the turn is certain to close.
            decoded: The template text generation follows, which says whether
                it starts inside a thought.
        """
        gen_start = self._table.total + len(open_tokens)
        budget = self._config.context_size - gen_start - len(close_tokens)
        cap = max_reply_tokens if max_reply_tokens is not None else self._config.max_reply_tokens
        stops = list(self._config.stop_strings) + list(stop or [])

        generated = GenerationAccumulator()
        tags = self._template.reasoning
        thought = ThoughtFilter(tags, begins_in_thought(tags, decoded)) if tags else None
        # Where a thought is filtered out, stop strings are matched against the
        # answer here rather than against the whole text by the context.
        opening = AnswerOpening(self._formatter.answer_opening())
        answer_stops = stops if thought else []
        answer = _AnswerStops(answer_stops)
        pieces = self._context.generate(
            gen_start, max(0, min(cap, budget)), [] if thought else stops,
            out=generated, cancel=cancel,
            tool_tokens=self._template.tool_tokens if allow_tools else None,
            sampling=sampling, ends_turn=self._turn_end_test(thought))
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
                # A cut reply may end on the start of a thought's tag.
                complete = generated.stop_reason not in ("interrupted", "token_limit")
                last = opening.feed(thought.flush(complete) if thought else "")
                if shown := answer.flush(last + opening.flush(complete)):
                    yield shown
        finally:
            pieces.close()
            # Runs on normal completion and on early close alike.
            had_thought = bool(thought and thought.thought_pieces)
            kept_as_generated = had_thought and tags.past_keeps_thought
            n_evicted_messages, n_decoded = self._record_reply(
                generated, answer.text, open_tokens, close_tokens,
                as_past=not kept_as_generated)
            if not self._reply_open:
                self._save_rest()

        return Reply(
            text=answer.text,
            n_prefilled_tokens=n_prefilled_tokens + n_decoded,
            n_generated_tokens=len(generated.token_ids),
            n_evicted_messages=n_evicted_messages,
            stop_reason=generated.stop_reason,
            tool_calls=list(self._pending_calls),
            n_reasoning_tokens=thought.thought_pieces if thought else 0,
        )

    def _turn_end_test(self, thought: ThoughtFilter | None) -> Callable[[int], bool] | None:
        """Whether a sampled token ends the reply, besides the end-of-generation tokens."""
        turn_end = self._template.turn_end_token
        if turn_end is None:
            return None
        # Read at each sample: the filter has been fed every piece yielded before it.
        return turn_end_test(turn_end, self._context.token_text(turn_end),
                             self._template.reasoning, thought)

    def _record_reply(self, generated: GenerationAccumulator, answer: str,
                      open_tokens: list[int], close_tokens: list[int],
                      as_past: bool) -> tuple[int, int]:
        """Record what one generation left.

        The turn is closed unless a call is waiting for its result. Where
        ``as_past`` is set and a closed reply's tokens differ from the
        template's past reply of its answer, the reply is decoded again in that
        form, from the first token that differs, after an eviction sized by
        that form, so the decode carries the eviction's shift; a turn that
        called tools keeps the tokens it was generated as. The message's text
        is the answer either way, and a closed reply without one, all thought
        or cut before its answer, is removed.

        Returns:
            How many messages were evicted, and how many tokens were decoded
            after the generation.
        """
        tokens = open_tokens + generated.token_ids
        if self._reply_open:
            self._table.extend_last(tokens, answer)
        else:
            self._table.append(Message("assistant", answer, tokens))
        msg = self._table.messages[-1]
        self._pending_calls = self._read_calls(generated)
        self._batch = [call.name for call in self._pending_calls]
        unreadable = (generated.stop_reason == "tool_call"
                      and not self._pending_calls)
        if unreadable:
            # Told, not dropped: a call left with no result is a shape no
            # template produces, and the model would read it back every turn.
            # The block needs a function name and there is none to read, so it
            # carries `_UNKNOWN_FUNCTION` and the message says what happened.
            generated.stop_reason = "unreadable_tool_call"
            unreadable = self._try_write_tool_result(_UNKNOWN_FUNCTION, {
                "error": "unreadable_call",
                "message": "the tool call failed: the call could not be read",
            })
        self._reply_open = bool(self._pending_calls) or unreadable
        if self._reply_open:
            return 0, 0
        holds_call = self._holds_call(msg)
        if not msg.text and not holds_call:
            self._remove_newest(msg)
            return self._evict_until(lambda: self._fits(0)), 0
        if self._on_message:
            self._on_message("assistant", msg.text)
        past = self._fragment_tokens("assistant", msg.text)
        fits = self._table.total - msg.n_tokens + len(past) <= self._config.context_size
        # A thought, what a generation prompt adds after the tag, the whitespace
        # a template strips, and a word generated in other pieces than its
        # text tokenizes into all make the two differ.
        if as_past and fits and not holds_call and past != msg.token_ids + close_tokens:
            return self._record_as_past(msg, past)
        # Trim back after the reply, so the cache rests inside both eviction
        # limits for the next generation, an early close included. The closer
        # counts against the threshold even though it is decoded after the trim.
        n_evicted_messages = self._evict_until(lambda: self._fits(len(close_tokens)))
        self._close_turn(msg, close_tokens)
        return n_evicted_messages, 0

    def _record_as_past(self, msg, past: list[int]) -> tuple[int, int]:
        """Evict as ``past`` needs, then give the newest reply ``past``.

        Returns:
            How many messages were evicted, and how many tokens were decoded.
        """
        fits = lambda: self._fits(len(past) - msg.n_tokens)    # noqa: E731
        if not self._context.can_shift and not fits():
            # Evicting rebuilds such a cache anyway, so the rewrite joins it.
            # One rebuild for the eviction and the rewrite together.
            _, evicted = self._table.evict_oldest_until(fits)
            if self._table.messages and self._table.messages[-1] is msg:
                self._table.replace_last(past, msg.text)
            self._rebuild_cache()
            return evicted, self._table.total
        evicted = self._evict_until(fits)
        if not (self._table.messages and self._table.messages[-1] is msg):
            return evicted, 0
        return evicted, self._rewrite_newest(msg, past, msg.text)

    def _try_write_tool_result(self, name: str, result) -> bool:
        """Write a result block if this model and this context have room for one.

        Recording a reply runs in a ``finally`` and is what leaves the cache and
        the table agreeing, so it must not raise. Where the template renders no
        result block, or what is left of the context cannot hold one, the model
        cannot be told and the turn closes instead.

        Returns:
            Whether the result was written, and so whether the turn is still
            open for the reply that answers it.
        """
        try:
            self._write_tool_result(name, result)
            return True
        except (ValueError, ContextOverflowError):
            return False

    def _read_calls(self, generated: GenerationAccumulator) -> list[ToolCall]:
        """The calls ``generated`` stopped on, empty where it produced no readable call.

        Every failure counts as "not a call": this runs with the turn still
        open, so one that got out would leave nothing to close it.
        """
        if generated.stop_reason != "tool_call":
            return []
        try:
            return parse_calls(generated.tool_call, self._template.call_body,
                               self._template.call_lead_in)
        except Exception:
            return []

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ChatWrapper is closed")

    def _declared_names(self) -> set[str]:
        """Every tool the conversation currently tells the model about.

        Read off the table, so a declaration that was evicted stops counting on
        its own and the next one writes it again.
        """
        return {name for message in self._table.messages
                for name in message.declares}

    def _ensure_begun(self, action: str) -> None:
        """Refuse to work on a conversation that was never begun.

        Only :meth:`begin` prefills the sequence-opening token and the system
        prompt, so what is added before one reaches the model as a fragment of
        a conversation it was never trained to see.

        Raises:
            ValueError: If :meth:`begin` has not run.
        """
        if not self._begun:
            raise ValueError(f"no conversation to {action}; call begin() first")

    def _ensure_no_open_reply(self, action: str) -> None:
        """Refuse to append a message inside a turn a tool call left open.

        Such a message would sit inside a model turn that never ended, which no
        template renders and no model was trained on.
        """
        if self._reply_open:
            raise ValueError(
                f"cannot {action} while a reply is open: it stopped on a tool "
                "call and its turn is unclosed. Finish it with "
                "resume_until_tool(), or discard it with begin()."
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
        self._rest_states = []

    def _save_rest(self) -> None:
        """Save the state a cache that cannot shift rests in now, where it can be saved."""
        if self._context.can_shift or not hasattr(self._context, "save_state"):
            return
        if getattr(self._context, "can_cut_tail", False):
            return                      # a cut needs no saved state
        try:
            state = self._context.save_state()
        except RuntimeError:
            state = None                # a rebuild stands in where none is saved
        if state is None:
            self._rest_states = []
            return
        self._rest_states = [s for s in self._rest_states
                             if s[0] < self._table.total][-1:] + [(self._table.total, state)]

    def _back_to(self, pos: int) -> int | None:
        """Put a cache that cannot shift back to holding the table's first ``pos`` tokens.

        From the latest state it rested in at or before ``pos``, decoding what
        the table holds after it.

        Returns:
            How many tokens were decoded, or ``None`` where no state could be
            restored, the cache then left for the caller to rebuild.
        """
        usable = [s for s in self._rest_states if s[0] <= pos]
        self._rest_states = usable
        if not usable:
            return None
        rested, state = usable[-1]
        try:
            self._context.restore_state(state, rested)
        except RuntimeError:
            self._rest_states = []
            return None
        between = [t for m in self._table.messages for t in m.token_ids][rested:pos]
        if between:
            self._context.prefill(between, start_pos=rested, want_logits=False)
        return len(between)

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


class _AnswerStops:
    """Collects an answer as it streams, ending it at the earliest stop string.

    Text that could still begin a stop string is held back until it is known
    not to. Without stop strings everything passes at once.
    """

    def __init__(self, stops: list[str]) -> None:
        self._stops = [s for s in stops if s]
        self._hold = max((len(s) for s in self._stops), default=1) - 1
        self._held = ""
        self.text = ""
        self.stopped = False

    def feed(self, piece: str) -> str:
        """The part of ``piece`` that is certainly answer, possibly empty."""
        if not self._stops:
            self.text += piece
            return piece
        self._held += piece
        hit = _stop_hit(self._held, self._stops)
        if hit is not None:
            self.stopped = True
            return self._pass(self._held[:hit], "")
        cut = max(0, len(self._held) - self._hold)
        return self._pass(self._held[:cut], self._held[cut:])

    def flush(self, last: str = "") -> str:
        """Everything still held, with ``last`` appended, once the answer has ended."""
        if not self._stops:
            return self.feed(last)
        shown = self.feed(last)
        if self.stopped:
            return shown
        return shown + self._pass(self._held, "")

    def _pass(self, shown: str, held: str) -> str:
        self._held = held
        self.text += shown
        return shown


def _shared_prefix_length(a: list[int], b: list[int]) -> int:
    """How many leading tokens ``a`` and ``b`` have in common."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _specs_for_system_prompt(tools) -> list[ToolSpec] | None:
    """What ``tools`` puts in the system turn, registry or plain list."""
    if isinstance(tools, ToolRegistry):
        return tools.specs_for_system_prompt()
    return tools


def _all_specs(tools) -> list[ToolSpec] | None:
    """Every declaration ``tools`` holds, registry or plain list."""
    if isinstance(tools, ToolRegistry):
        return tools.specs()
    return tools


def _deltas_then_calls(stream):
    """Yield a generation's text deltas, then the calls it stopped at.

    The wrapped generator is closed whatever happens, so an early close still
    ends the turn and releases the wrapper's lock.
    """
    reply = None
    try:
        while True:
            try:
                delta = next(stream)
            except StopIteration as done:
                reply = done.value
                break
            yield delta
    finally:
        stream.close()
    for call in reply.tool_calls:
        yield call
    return reply


def _sum_generations(generations: list[Reply], tool_limit_reached: bool) -> Reply:
    """Sum the generations of one exchange into the Reply that describes it."""
    return Reply(
        text="".join(t.text for t in generations),
        n_prefilled_tokens=sum(t.n_prefilled_tokens for t in generations),
        n_generated_tokens=sum(t.n_generated_tokens for t in generations),
        n_evicted_messages=sum(t.n_evicted_messages for t in generations),
        stop_reason=generations[-1].stop_reason,
        tool_calls=[call for t in generations for call in t.tool_calls],
        tool_limit_reached=tool_limit_reached,
        n_reasoning_tokens=sum(t.n_reasoning_tokens for t in generations),
    )


def _explain_failure(result, may_call_again: bool):
    """Say that the call failed and what to do next, leaving a real result alone.

    A dict carrying ``error`` is a failure, whether the registry built it or a
    tool returned one itself. Its ``message`` becomes the reason inside a
    sentence that names the failure and ends in the one instruction that is not
    already implied: whether another call is worth making. What to do instead is
    left unsaid, there being nothing to do but answer.
    """
    if not isinstance(result, dict) or "error" not in result:
        return result
    reason = result.get("message") or result["error"]
    return {**result,
            "message": f"the tool call failed: {reason}. "
                       + ("try again" if may_call_again else "do not call again")}


