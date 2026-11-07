"""ChatWrapper: the public API, prefilling as little as possible.

``begin`` resets and prefills the system prompt plus what recent history fits;
``inject`` prefills one message without generating; ``reply`` prefills only
the new request and then
generate, decoding the reply again only where the template writes a past reply
otherwise. Crossing either eviction limit removes the oldest non-system
messages and shifts the survivors down, reusing their KV.

Two invariants hold after every action, and the tests enforce both: the physical
cache equals the concatenation of the message table, and every prefill appends
at the cache's end.

The turn-closing tag is decoded after the post-reply eviction rather than before
it. See design_history.md, "How eviction works".
"""

from __future__ import annotations

import contextlib
import math
import threading
from collections.abc import Callable, Generator
from dataclasses import replace

from .call_parsing import parse_calls
from .config import Config, Sampling
from .cache_context import GenerationAccumulator
from .context import KVContext
from .messages import Message, MessageTable, n_newest_that_fit
from .reply_text import (AnswerOpening, AnswerStops, ThoughtFilter, begins_in_thought,
                         reply_header, turn_end_test)
from .cache_context import CACHE_CONTEXT, ensure_answers
from .cache_sync import CacheSync
from .tokenizing import tokenize_message, tokenize_pieces
from .reply import ContextOverflowError, Reply, ReplyStream
from .resolved_template import ResolvedTemplate, written_between
from .template import TemplateFragments, ToolCallFragments
from .tools import ToolCall, ToolRegistry, ToolResult, ToolSpec, ToolTurn, result_text


# The name a result block carries where no real one is to be had: answering a
# call that could not be read that far. Which name is used does not change what
# a template renders around it.
_UNKNOWN_FUNCTION = "unknown"

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
    ) -> None:
        """Load the model, resolve the chat template, and validate it.

        Args:
            config: The configuration; :class:`~llama_chat.config.Config`'s
                defaults where it is not given.
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

        Raises:
            TypeError: If ``context`` does not answer everything the wrapper
                calls on it.
            ValueError: If the chat template cannot be recovered, if it
                fails validation against the model, or if ``config`` asks for a
                larger context than ``context`` holds.
        """
        if config is None:
            config = Config()
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
            ensure_answers(self._context, CACHE_CONTEXT)
            if config.context_size > self._context.context_size:
                raise ValueError(
                    f"context_size is {config.context_size} but the cache context holds "
                    f"{self._context.context_size} tokens; give both the same size")
            self._template_given = {"fragments": fragments, "tool_calls": tool_calls}
            self._template = ResolvedTemplate(self._context, **self._template_given)
            # The conversation's own, set again by every begin.
            self._formatter = self._template.formatter()
        except BaseException:
            if owns_context:
                self._context.close()
            raise
        self._table = MessageTable()
        self._registry = None
        self._moving_specs: list[ToolSpec] = []   # declared before the newest user turn
        self._system_specs: list[ToolSpec] = []   # declared in the system turn
        # An open turn and the calls owing a result are not the same thing: the
        # last result empties the list while the turn is still open.
        self._reply_open = False
        self._pending_calls = []
        self._batch = []                # the names of the calls made together
        self._reasoning = False
        # The template text an open turn ends with, which a resumption opens on.
        self._turn_ends_with = ""
        self._cache = CacheSync(self._context, self._held)
        # Not read off the table: a stray system message would make an unbegun
        # table look begun.
        self._begun = False
        self._warmed = False
        self._closed = False
        # One KV sequence is shared across generations, so concurrent decodes
        # would corrupt it. It is held for a whole generation, not a decode.
        self._lock = threading.Lock()

    def begin(self, system_prompt: str,
              messages: list[tuple[str, str] | ToolTurn] | None = None,
              tools: ToolRegistry | list[ToolSpec] | None = None, *,
              reasoning: bool = False) -> int:
        """Reset the conversation: prefill the system prompt and recent history.

        The first call also warms the model for every size of batch it will
        decode, so the first request streams without a cold start. Every other
        action refuses until this has run, because only this prefills the
        sequence opener and the system prompt.

        Args:
            system_prompt: Always kept, never evicted. Empty, and with no
                tools, it is no system message at all, and the conversation
                opens as the template opens one without.
            messages: Optional history, oldest first: ``(role, text)`` pairs,
                and a :class:`~llama_chat.tools.ToolTurn` for an assistant turn
                that called tools, written as this model's template writes one.
                A history that ends on a tool turn's result leaves that turn
                open, as a result written into a live turn does, for
                :meth:`reply` to go on with. Only the most recent
                that fit alongside the system prompt are prefilled, under both
                eviction limits, from a user turn on; older excess is dropped.
            tools: Tools to declare, rendered into the system turn where this
                model's template puts them, which is the one placement that
                costs no extra prefill and that no eviction can take back. From
                a whole :class:`~llama_chat.tools.ToolRegistry`, what it marks
                for the system prompt; the rest waits for
                :meth:`declare_unknown_tools`. A registry is also kept, so
                :meth:`reply` running calls needs none of its own; a list of specs
                is declared and not kept. Refused up front where this model
                could not use them, rather than at the first call.
            reasoning: Whether the model reasons before each answer, as its
                template turns that on. It belongs to the conversation, so it
                changes only here. Every reply's thought is kept out of what
                it yields, either way.

        Returns:
            How many items of ``messages`` were kept.

        Raises:
            ContextOverflowError: If the system prompt alone exceeds
                ``evict_above_tokens``, or the newest exchange of ``messages``,
                from its user message on, cannot fit beside it.
            ValueError: If ``tools`` names anything or the
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
            if not self._template.is_current():     # it writes the date, say, and that moved on
                self._template = ResolvedTemplate(self._context, **self._template_given)
            registry = tools if isinstance(tools, ToolRegistry) else None
            specs = _specs_for_system_prompt(tools)
            if specs:
                self._template.ensure_tools_usable()
            formatter = self._template.formatter(reasoning)
            moving = bool(specs) and self._template.declares_in_newest_turn
            sys_tokens = self._system_tokens(formatter, system_prompt,
                                             None if moving else specs, reasoning)
            if len(sys_tokens) > self._config.evict_above_tokens:
                raise ContextOverflowError(
                    f"system prompt needs {len(sys_tokens)} tokens but the "
                    f"eviction threshold is {self._config.evict_above_tokens}; "
                    "shorten the prompt or raise context_size/evict_above_fraction"
                )
            ends_open = (bool(messages) and isinstance(messages[-1], ToolTurn)
                         and bool(messages[-1].steps)
                         and isinstance(messages[-1].steps[-1], ToolResult))
            last = len(messages) - 1
            hist = [self._history_tokens(formatter, item, ends_open and i == last,
                                         reasoning, specs)
                    for i, item in enumerate(messages)]
            hist_tokens = [tokens for tokens, _ in hist]
            for i in range(len(messages) - 1):
                if _is_user(messages[i]) and _is_user(messages[i + 1]):
                    hist_tokens[i] = hist_tokens[i] + self._join_tokens()
            carrier = next((i for i in range(len(messages) - 1, -1, -1)
                            if _is_user(messages[i])), None) if moving else None
            if carrier is not None:
                held, _ = self._moving_declaration(specs, reasoning)
                hist_tokens[carrier] = held + hist_tokens[carrier]

            roles = ["user" if _is_user(item) else "assistant" for item in messages]
            lengths = [len(t) for t in hist_tokens]
            # Under the token limit alone: the message limit may keep nothing on purpose.
            if "user" in roles and not n_newest_that_fit(
                    lengths, len(sys_tokens), self._config.evict_above_tokens, roles=roles):
                newest = sum(lengths[len(roles) - 1 - roles[::-1].index("user"):])
                raise ContextOverflowError(
                    f"the newest exchange of the history, from its user message on, needs "
                    f"{newest} tokens but only "
                    f"{self._config.evict_above_tokens - len(sys_tokens)} fit beside the "
                    "system prompt; shorten it or raise context_size/evict_above_fraction")
            kept = n_newest_that_fit(lengths, len(sys_tokens), self._config.evict_above_tokens,
                                     self._config.max_history_messages, roles=roles)
            keep_from = len(messages) - kept
            kept_messages = messages[keep_from:]
            kept_tokens = hist_tokens[keep_from:]

            self._cache.reset()
            self._begun = False     # true again only once the prefill lands
            # The turn a call was holding open went with the table.
            self._reply_open = False
            self._pending_calls = []
            self._registry = registry
            self._moving_specs = list(specs) if moving else []
            self._system_specs = [] if moving else list(specs or [])
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
            self._cache.decode(all_tokens, want_logits=ends_open)

            self._table.reset(Message(
                "system", system_prompt, sys_tokens,
                declares=tuple(spec.name for spec in specs or [])))
            for i, (item, toks) in enumerate(zip(kept_messages, kept_tokens), keep_from):
                role, text = (("assistant", item.text) if isinstance(item, ToolTurn)
                              else item)
                declares = (tuple(spec.name for spec in specs) if i == carrier else ())
                self._table.append(Message(
                    role, text, toks, declares=declares, holds_call=isinstance(item, ToolTurn),
                    steps=list(item.steps) if isinstance(item, ToolTurn) else []))
            self._reply_open = ends_open
            self._turn_ends_with = hist[-1][1] if ends_open else ""
            self._begun = True
            if not ends_open:
                self._cache.rest()
            return len(kept_messages)

    # ----- adding context ------------------------------------------------------
    def inject(self, text: str, role: str = "user") -> int:
        """Prefill one message as context without generating.

        Evicts the oldest messages until the new one fits inside both eviction
        limits, then prefills it.

        Returns:
            The number of messages evicted to make room.

        Raises:
            ValueError: If no conversation has been begun, or if a tool call
                has left a reply open.
            ContextOverflowError: If the message cannot fit even after
                evicting every non-system message. Raised before anything is
                evicted.
        """
        with self._lock:
            self._ensure_open()
            self._ensure_begun("inject into")
            self._ensure_no_open_reply("inject")
            if role == "system":
                prefix, suffix = self._later_system_tags()
                tokens = tokenize_message(self._context, 
                    prefix, self._template.fragments.content(role, text), suffix)
            else:
                tokens = self._fragment_tokens(role, text)
            held, declares = (self._moving_declaration(self._moving_specs, self._reasoning)
                              if role == "user" else ([], ()))
            join_tokens = self._user_turn_join_tokens(role)
            # Checked against the best case before anything is evicted, so a
            # rejected message cannot destroy history it would never fit beside.
            floor = self._table.messages[0].n_tokens
            tokens = self._fit_or_raise(
                tokens, budget=self._config.evict_above_tokens - floor - len(held),
                description="injected message", role=role,
            )
            tokens = held + tokens
            evicted = self._evict(
                lambda: self._fits(len(join_tokens) + len(tokens), arriving_messages=1))
            if evicted:             # the turn it would have closed may be gone
                join_tokens = self._user_turn_join_tokens(role)
            if self._on_message:    # before the message is written; see the request's
                self._on_message(role, text)
            if held:
                self._move_declaration_on()

            self._cache.decode(join_tokens + tokens)
            if join_tokens:
                self._table.extend_last(join_tokens)
            self._table.append(Message(role, text, tokens, declares=declares))
            self._cache.rest()
            return evicted

    # ----- asking ---------------------------------------------------------------
    def reply(self, text: str | None = None, *,
              calls: str | ToolRegistry = "refuse",
              max_calls: int | None = None,
              max_reply_tokens: int | None = None,
              stop: list[str] | None = None,
              cancel: threading.Event | None = None,
              sampling: Sampling | None = None) -> ReplyStream:
        """Generate the reply, streaming it as it is generated.

        Lazy: nothing happens, the headroom check included, until the first
        iteration. Closing the stream early still ends the turn and records what
        was generated, so the next generation stays valid. The stream must be
        driven from one thread at a time; to interrupt from another, pass
        ``cancel``. An abandoned stream holds the wrapper's lock until it is
        collected, so exhaust it, close it, use it in a ``with``, or call
        :meth:`ReplyStream.finish`.

        Args:
            text: The user request, tokenized with special-token parsing off.
                See the README, "Special tokens in message content". ``None``
                answers the newest message, a user message already written, or
                goes on with a turn a call left open once every result is in.
            calls: What becomes of a tool call the model makes.

                * ``"refuse"``: nothing is run, not even a tool :meth:`begin`
                  was given. The call is kept out of the text, the model is
                  told no tool can be run, and the reply goes on, a blank line
                  parting the text before the call from the text after it
                  where they would join without whitespace.
                * ``"yield"``: each call is handed over as a
                  :class:`~llama_chat.tools.ToolCall` after the text, and the
                  turn stays open. Answer each with :meth:`inject_tool_result`,
                  in order, then call this again without ``text``.
                * ``"run"``, or a :class:`~llama_chat.tools.ToolRegistry`:
                  each call is run, against the registry :meth:`begin` was
                  given or the one passed, its result written, and the reply
                  goes on until the model answers. The last permitted round
                  withholds the call tokens, so the turn always closes.
            max_calls: Where calls are run, how many one turn may use. ``None``
                sets no ceiling; a call past one is answered with an error
                result rather than run.
            max_reply_tokens: Override for the per-generation cap.
            stop: Stop strings for this request, added to the config's,
                matched against the answer only.
            cancel: Setting it, from any thread, stops generation within one
                token and ends the turn with ``stop_reason == "interrupted"``.
            sampling: This request's own sampling settings, in place of the
                config's for the fields it sets.

        Returns:
            A :class:`ReplyStream` of the text deltas, and with ``"yield"`` the
            calls after them; its ``reply`` is the :class:`Reply` once it has
            ended, summing every generation where calls were run.

        Raises:
            ValueError: If no conversation has been begun; if a tool call has
                left a reply open and ``text`` is given, or a result is still
                owed; if ``text`` is ``None`` with nothing to answer; or if a
                registry holding tools is given to a model that could never
                call one.
            ContextOverflowError: If too little room remains for the reply, or
                a result or the error answering a call no longer fits. The
                turn is then open and cannot be continued.
        """
        options = dict(max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
                       sampling=sampling)
        if calls == "yield":
            return self._until_call(text, allow_tools=True, **options)
        if calls == "refuse":
            registry = None
        elif calls == "run":
            registry = self._registry
        elif isinstance(calls, ToolRegistry):
            registry = calls
        else:
            raise ValueError(f"calls must be 'refuse', 'yield', 'run' or a ToolRegistry, "
                             f"not {calls!r}")
        return ReplyStream(self._generate_until_answered(
            text, registry=registry, max_calls=max_calls, **options))

    def _until_call(self, text: str | None, *, allow_tools: bool, held: bool = False,
                    **options) -> ReplyStream[str | ToolCall]:
        """One generation, from ``text`` or on from the open turn, then the calls it stopped at.

        ``allow_tools`` False withholds the call tokens, which is what makes
        the last permitted round of an exchange certain to close the turn.
        ``held`` says the caller holds the lock already, for a whole exchange.
        """
        return ReplyStream(_deltas_then_calls(
            self._one_generation(text, allow_tools, held, **options)))

    def _one_generation(self, text, allow_tools: bool, held: bool, **options):
        """Resume the open turn or start a request, decided under the lock."""
        with contextlib.nullcontext() if held else self._lock:
            self._ensure_open()
            self._cache.repair()
            if text is None and self._reply_open:
                return (yield from self._resume(allow_tools=allow_tools, **options))
            return (yield from self._request_and_generate(text, **options))

    def inject_tool_result(self, call: ToolCall | str, result) -> None:
        """Hand the model what the tool it called returned.

        Prefilled inside the turn the call left open, wrapped the way this
        model's own template wraps a result, and tokenized with special-token
        parsing off. Nothing is evicted while a turn is open, so such a turn
        grows towards ``context_size``. See the README, "Tool calling".

        Args:
            call: The call being answered, as it was handed to you, or its name
                alone. Either way it must be the oldest unanswered one.
            result: What it returned. Anything but a string is sent as JSON.

        Raises:
            ValueError: If no call is waiting, or this is not the one owed.
            ContextOverflowError: If not even the wrapping fits in what is left
                of the context.
        """
        with self._lock:
            self._ensure_open()
            self._cache.repair()
            self._inject_result(call, result)

    def _inject_result(self, call: ToolCall | str, result) -> None:
        """As :meth:`inject_tool_result`, with the lock held."""
        name = call.name if isinstance(call, ToolCall) else call
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
        self._table.add_steps(ToolResult(name, result))

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
        text = written_between(result_text(result), prefix, suffix)
        # The body is clipped to fit; the tags never are. See
        # "Tool calling".
        free = self._config.context_size - self._table.total
        if len(tags) + len(closing) > free:
            raise ContextOverflowError(
                f"the result wrapping alone needs "
                f"{len(tags) + len(closing)} tokens but only {free} are "
                f"left in the context; this turn cannot be continued, "
                f"start a new conversation with begin()")
        room = free - len(tags) - len(closing) - self._config.min_reply_tokens
        # Whole, as a render of the conversation tokenizes it; clipped where it must be.
        tokens = tokenize_message(self._context, prefix, text, suffix)
        if len(tokens) > room + len(tags) + len(closing):
            body = self._context.tokenize(text, add_special=False, parse_special=False)
            tokens = tags + body[:max(0, room)] + closing
        self._cache.decode(tokens, want_logits=True)
        self._table.extend_last(tokens)
        self._turn_ends_with = suffix

    # ----- shortening the newest reply -----------------------------------------
    def truncate_last_reply(self, keep_text: str) -> None:
        """Shorten the newest assistant message to ``keep_text``.

        For a reply that was cut short where it was being read or heard: the
        model is left holding only what the reader got. The cut is at the
        cache's end, so the tokens the shortened message shares with the old
        one are not decoded again; the rest of it, its closing tag at least, is.
        A cache that cannot cut there goes back to a state it rested in, and is
        decoded again from the whole conversation only where none is saved.

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
            self._cache.rest()

    def _newest_reply(self, action: str):
        """The newest message, refused unless it is an assistant reply that may change."""
        self._ensure_open()
        self._ensure_begun(f"{action} in")
        self._ensure_no_open_reply(f"{action} a reply")
        msg = self._table.messages[-1]
        if msg.role != "assistant":
            raise ValueError("the newest message is not an assistant reply")
        if msg.holds_call:
            raise ValueError("the newest reply holds a tool call, which would "
                             "be separated from its result")
        return msg

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
        self._cache.cut_to(msg.pos_start)

    def _rewrite_newest(self, msg, tokens: list[int], text: str) -> int:
        """Give the newest message ``tokens``, decoding only where they differ.

        Returns:
            How many tokens were decoded.
        """
        shared = _shared_prefix_length(msg.token_ids, tokens)
        self._table.replace_last(tokens, text)
        decoded = self._cache.cut_to(msg.pos_start + shared)
        if decoded is None:             # rebuilt, the new tokens with the rest
            return self._table.total
        self._cache.decode(tokens[shared:])
        return decoded + len(tokens) - shared

    def declare_unknown_tools(self, tools: ToolRegistry | list[ToolSpec]) -> int:
        """Tell the model about the tools it has not been told about yet.

        Written where the template declares tools, never as a message of its
        own: into the system turn, with the conversation after it decoded again
        once, or, where the template declares tools right before the newest
        user turn, into what that turn carries, decoded again from there.

        Args:
            tools: What to declare: a whole
                :class:`~llama_chat.tools.ToolRegistry`, of which everything it
                holds is taken, or the specs themselves. Nothing to declare does
                nothing, and neither does a tool the conversation already
                declares. The registry is not kept, unlike ``begin``'s: which
                one a turn runs against is that method's decision, and a
                declaration adds names rather than replacing what can run.

        Returns:
            How many messages were evicted to make room.

        Raises:
            ValueError: If no conversation has been begun, if a tool call has
                left a reply open, if this model cannot call tools at all (see
                :meth:`begin`), if its template declares no tools, or if the
                system turn with the declarations would exceed
                ``evict_above_tokens``.
        """
        with self._lock:
            self._ensure_open()
            known = self._declared_names() | {spec.name for spec in self._moving_specs}
            specs = [spec for spec in _all_specs(tools) or []
                     if spec.name not in known]
            if not specs:
                return 0
            self._ensure_begun("declare into")
            self._ensure_no_open_reply("declare tools")
            self._template.ensure_tools_usable()
            if self._template.declares_in_newest_turn:
                return self._declare_in_newest_turn(specs)
            declared = self._system_specs + specs
            system = self._table.messages[0]
            tokens = self._system_tokens(self._formatter, system.text, declared,
                                         self._reasoning)
            if len(tokens) > self._config.evict_above_tokens:
                raise ContextOverflowError(
                    f"the system turn with these declarations needs {len(tokens)} tokens "
                    f"but the eviction threshold is {self._config.evict_above_tokens}; "
                    "declare fewer tools or raise context_size/evict_above_fraction")
            evicted = self._evict(lambda: self._fits(len(tokens) - system.n_tokens))
            self._rewrite_from(0, tokens, tuple(spec.name for spec in declared))
            self._system_specs = declared
            self._cache.rest()
            return evicted

    # ----- introspection -------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        """How many tokens the cache currently holds."""
        return self._table.total

    @property
    def config(self) -> Config:
        """The configuration this wrapper runs with, its limits included."""
        return self._config

    @property
    def can_shift(self) -> bool:
        """Whether eviction can shift surviving KV in place.

        ``False`` means a cut, a saved state or decoding the survivors again
        instead. See design_history.md, "How eviction works".
        """
        return self._context.can_shift

    def wait_idle(self, timeout_s: float | None = None) -> bool:
        """Block until no generation holds the wrapper; advisory, since one may start right after.

        Args:
            timeout_s: Seconds to wait; ``None`` waits indefinitely.

        Returns:
            ``True`` if the wrapper went idle, ``False`` on timeout.
        """
        if not self._lock.acquire(timeout=-1 if timeout_s is None else timeout_s):
            return False
        self._lock.release()
        return True

    @property
    def system_prompt(self) -> str:
        """The system prompt the conversation was begun with, for :meth:`begin` beside :meth:`history`."""
        messages = self._table.messages
        return messages[0].text if messages else ""

    def history(self) -> list[tuple[str, str] | ToolTurn]:
        """The conversation the cache holds, oldest first, as :meth:`begin` takes it.

        Every message after the system prompt: ``(role, text)`` pairs, and a
        :class:`~llama_chat.tools.ToolTurn` for an assistant turn that called
        tools, with its calls and results. Begun again from it, with the same
        system prompt and tools, the conversation reads in as the template
        writes it.
        """
        return [ToolTurn(tuple(m.steps)) if m.steps else (m.role, m.text)
                for m in self._table.messages[1:]]

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
        while the turn stays open until :meth:`reply` closes it. While
        this is ``True``, :meth:`reply` with a request, :meth:`inject` and
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
            text, **kwargs: As :meth:`reply`.

        The lock is held for the whole exchange, tools running included, so
        no other action comes between its rounds; a tool must not call back
        into the wrapper.

        Raises:
            ValueError: If ``registry`` holds tools this model could never
                call, which is what
                :meth:`~llama_chat.resolved_template.ResolvedTemplate.ensure_tools_usable`
                decides.
        """
        with self._lock:
            # A registry is the same promise a declaration is, so it is checked the
            # same way: an exchange that cannot run a tool is refused before it
            # starts, not after the model has asked for one.
            if registry is not None and len(registry) > 0:
                self._template.ensure_tools_usable()
            stream = self._until_call(text, allow_tools=True, held=True, **kwargs)
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
                            stream = self._until_call(
                                None, allow_tools=registry is not None and calls_left > 0,
                                held=True, **kwargs)
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
                        self._inject_result(call, _explain_failure(result, may_call_again))
                    stream = self._until_call(None, allow_tools=may_call_again, held=True,
                                              **kwargs)
            finally:
                stream.close()

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
        self, text: str | None, *, max_reply_tokens: int | None = None,
        stop: list[str] | None = None, cancel: threading.Event | None = None,
        sampling: Sampling | None = None,
    ) -> Generator[str, None, Reply]:
        """Add the user request, then generate one reply from the cache's end.

        Without ``text``, the reply answers the newest message, which must be
        a user message already written. One generation, so a call it stops on
        leaves the turn open for a result. The lock must be held.
        """
        self._ensure_open()
        self._ensure_begun("generate into")
        self._ensure_no_open_reply("start a request")
        if text is None:
            newest = self._table.messages[-1]
            if newest.role != "user":
                raise ValueError("there is nothing to answer: the newest message "
                                 "is not a user message")
            join_tokens, held, declares, user_tokens = [], [], (), []
        else:
            join_tokens = self._user_turn_join_tokens("user")
            held, declares = self._moving_declaration(self._moving_specs, self._reasoning)
            user_tokens = self._fragment_tokens("user", text)
        open_tokens = self._context.tokenize(self._formatter.generation_prompt(), add_special=False)
        close_tokens = self._context.tokenize(self._formatter.assistant_close(), add_special=False)

        # The prompt plus its turn closer must fit in the context once
        # everything but the system prompt is evicted; checked before
        # anything is, so a refused request destroys no history.
        floor = self._table.messages[0].n_tokens
        if text is not None:
            user_tokens = held + self._fit_or_raise(
                user_tokens,
                budget=(self._config.context_size - len(open_tokens) - len(close_tokens)
                        - len(join_tokens) - len(held) - floor),
                description="request", role="user",
            )
        needed = (len(join_tokens) + len(user_tokens) + len(open_tokens)
                  + len(close_tokens) + self._config.min_reply_tokens)
        if self._evict(
                lambda: self._table.total + needed <= self._config.context_size):
            if text is None and self._table.messages[-1].role != "user":
                raise ContextOverflowError("the message to answer does not fit "
                                           "beside room for its reply")
            if text is not None:    # the turn it joined may be gone
                join_tokens = self._user_turn_join_tokens("user")
        n_prompt = len(join_tokens) + len(user_tokens) + len(open_tokens)

        # Refused before the cache is mutated.
        free = self._config.context_size - self._table.total - n_prompt - len(close_tokens)
        if free < self._config.min_reply_tokens:
            raise ContextOverflowError(
                f"only {free} tokens free for the reply "
                f"(min_reply_tokens={self._config.min_reply_tokens}); shorten "
                f"the request or lower evict_above_fraction"
            )

        # Reported after every check and before the cache changes, so a
        # callback that raises refuses the request with nothing written.
        if self._on_message and text is not None:
            self._on_message("user", text)
        # One decode for the request text and the assistant-open tag: the
        # backend flags logits on the final token only either way.
        if text is not None:
            self._move_declaration_on()
        self._cache.decode(join_tokens + user_tokens + open_tokens, want_logits=True)
        if join_tokens:
            self._table.extend_last(join_tokens)
        if text is not None:
            self._table.append(Message("user", text, user_tokens, declares=declares))

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
        """Generate on into the open reply, adding no message of its own. The lock must be held."""
        self._ensure_open()
        if self._pending_calls:
            raise ValueError(f"{len(self._pending_calls)} call(s) still owe "
                             f"a result")
        if not self._reply_open:
            raise ValueError("no reply is open; start one with reply(text, calls='yield')")
        close_tokens = self._context.tokenize(self._formatter.assistant_close(),
                                          add_special=False)
        # The turn goes on from where the template's result wrapping ends it.
        return (yield from self._generate_into_reply(
            open_tokens=[], close_tokens=close_tokens, n_prefilled_tokens=0,
            max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
            allow_tools=allow_tools, sampling=sampling,
            decoded=self._turn_ends_with))

    def _history_tokens(self, formatter, item: tuple[str, str] | ToolTurn,
                        open_turn: bool, reasoning: bool,
                        specs: list[ToolSpec] | None) -> tuple[list[int], str]:
        """One message of a ``begin`` history, as it is prefilled.

        Returns:
            The tokens, and the template text an open turn ends with, empty for
            any other.
        """
        if not isinstance(item, ToolTurn):
            prefix, content, suffix = formatter.parts(*item)
            return tokenize_message(self._context, prefix, content, suffix), ""
        # Refuses a model that cannot call tools.
        pieces = self._template.past_tool_turn(item, open_turn, reasoning, specs)
        # The call tags as the single tokens a generated call holds.
        tags = {text: token for token in (self._template.tool_tokens or ())
                if (text := self._context.token_text(token))}
        tokens = tokenize_pieces(self._context, pieces, tag_tokens=tags)
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
            return tokenize_message(self._context, preamble, "", "", add_special=True)
        if preamble is not None and not self._template.has_system_turn:
            raise ValueError(
                "this model's template has no system turn of its own: it writes a "
                "system prompt into the first user turn, which a system prompt "
                "kept for the whole conversation cannot be")
        prefix, content, suffix = formatter.parts("system", system_prompt)
        if not specs:
            return tokenize_message(self._context, prefix, content, suffix,
                                                   add_special=True)
        # Rendered whole: where declarations go is the template's business.
        head, tail = self._template.system_turn_declaring(specs, reasoning)
        return tokenize_message(self._context, head, content, tail,
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
                :meth:`reply`.
            allow_tools: ``False`` withholds the call tokens, so this generation
                cannot end on a call and the turn is certain to close.
            decoded: The template text generation follows, which says whether
                it starts inside a thought.
        """
        budget = self._config.context_size - self._cache.end - len(close_tokens)
        cap = max_reply_tokens if max_reply_tokens is not None else self._config.max_reply_tokens
        stops = list(self._config.stop_strings) + list(stop or [])

        generated = GenerationAccumulator()
        tags = self._template.reasoning
        thought = ThoughtFilter(tags, begins_in_thought(tags, decoded)) if tags else None
        # Stop strings are matched against the answer only, after the thought
        # and any header for another reader are taken out.
        opening = AnswerOpening(self._formatter.answer_opening(),
                                reply_header(self._formatter.answer_opening(), tags))
        answer = AnswerStops(stops)
        pieces = self._cache.generate(
            max(0, min(cap, budget)), generated, cancel=cancel,
            sampling=sampling, ends_turn=self._turn_end_test(thought),
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
                # A cut reply may end on the start of a thought's tag.
                complete = generated.stop_reason not in ("interrupted", "token_limit")
                last = opening.feed(thought.flush(complete) if thought else "")
                # A header held back at a call is the call's, not the answer's.
                header_done = complete and generated.stop_reason != "tool_call"
                if shown := answer.flush(last + opening.flush(header_done)):
                    yield shown
        finally:
            pieces.close()
            # Runs on normal completion and on early close alike.
            had_thought = bool(thought and thought.thought_pieces)
            kept_as_generated = had_thought and tags.past_keeps_thought
            n_evicted_messages, n_decoded, answered = self._record_reply(
                generated, answer.text, open_tokens, close_tokens,
                as_past=not kept_as_generated)
            if not self._reply_open:
                self._cache.rest()
            if answered is not None and self._on_message:
                self._on_message("assistant", answered)

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
                      as_past: bool) -> tuple[int, int, str | None]:
        """Record what one generation left.

        The turn is closed unless a call is waiting for its result. Where
        ``as_past`` is set and a closed reply's tokens differ from the
        template's past reply of its answer, the reply is decoded again in that
        form, from the first token that differs, after an eviction sized by
        that form, so the decode carries the eviction's shift. A reply kept as
        generated, its thought included, is decoded again where its text
        tokenizes into other tokens than were generated; a turn that called
        tools keeps the tokens it was generated as. The message's text
        is the answer either way. A closed reply without one, all thought or
        cut before its answer, is removed where the template writes two user
        turns in a row, and kept as an empty reply where it does not.

        Returns:
            How many messages were evicted, how many tokens were decoded after
            the generation, and the answer of a closed reply that stays, which
            the caller reports to ``on_message`` once all of this has landed.
        """
        tokens = open_tokens + generated.token_ids
        if self._reply_open:
            self._table.extend_last(tokens, answer)
        else:
            self._table.append(Message("assistant", answer, tokens))
        msg = self._table.messages[-1]
        if generated.cached_beyond:
            self._cache.cut_to(self._table.total)
        self._pending_calls = self._read_calls(generated)
        self._batch = [call.name for call in self._pending_calls]
        if generated.stop_reason == "tool_call":
            self._table.mark_holds_call()
        if msg.holds_call:
            # The turn as a history hands it over: what the model wrote, then its calls.
            self._table.add_steps(*([answer] if answer else []), *self._pending_calls)
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
            return 0, 0, None
        holds_call = msg.holds_call
        if not msg.text and not holds_call and self._template.user_turn_join is not None:
            self._remove_newest(msg)
            return self._evict(lambda: self._fits(0)), 0, None
        past = self._fragment_tokens("assistant", msg.text)
        fits = self._table.total - msg.n_tokens + len(past) <= self._config.context_size
        # A thought, what a generation prompt adds after the tag, the whitespace
        # a template strips, and a word generated in other pieces than its
        # text tokenizes into all make the two differ.
        if as_past and fits and not holds_call and past != msg.token_ids + close_tokens:
            return *self._record_as_past(msg, past), msg.text
        if not as_past and not holds_call:
            # Kept as generated, in the tokens its text tokenizes into.
            n_open = len(open_tokens)
            kept = (msg.token_ids[:n_open] + self._context.retokenized(msg.token_ids[n_open:])
                    + close_tokens)
            if kept != msg.token_ids + close_tokens:
                return *self._record_as_past(msg, kept), msg.text
        # Trim back after the reply, so the cache rests inside both eviction
        # limits for the next generation, an early close included. The closer
        # counts against the threshold even though it is decoded after the trim.
        n_evicted_messages = self._evict(lambda: self._fits(len(close_tokens)))
        self._close_turn(msg, close_tokens)
        return n_evicted_messages, 0, msg.text

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
            self._cache.rebuild()
            return evicted, self._table.total
        evicted = self._evict(fits)
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
        body = generated.tool_call
        if self._template.call_texts is not None:
            body = self._template.call_lead_in + body   # the opening holds its start
        try:
            return parse_calls(body, self._template.call_body, self._template.call_lead_in)
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
        self._cache.repair()

    def _ensure_no_open_reply(self, action: str) -> None:
        """Refuse to append a message inside a turn a tool call left open.

        Such a message would sit inside a model turn that never ended, which no
        template renders and no model was trained on.
        """
        if self._reply_open:
            raise ValueError(
                f"cannot {action} while a reply is open: it stopped on a tool "
                "call and its turn is unclosed. Finish it with "
                "reply(), or discard it with begin()."
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

    def _held(self) -> list[int]:
        """The tokens the cache should hold: every message's, in order."""
        return [token for message in self._table.messages for token in message.token_ids]

    def _evict(self, fits) -> int:
        """Evict the oldest exchanges until ``fits()``, from the table and the cache; how many messages went."""
        eviction, evicted = self._table.evict_oldest_until(fits)
        self._cache.remove(eviction)
        return evicted

    def _close_turn(self, msg: Message, close_tokens: list[int]) -> None:
        """Decode the assistant turn's closing tag.

        Skipped where the eviction dropped this very message.
        """
        messages = self._table.messages
        if not messages or messages[-1] is not msg:
            return
        self._cache.decode(close_tokens)
        self._table.extend_last(close_tokens)

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
        raise ContextOverflowError(
            f"{description} needs {len(tokens)} tokens but only {budget} fit "
            f"(oversize_policy='{self._config.oversize_policy}')"
        )

    def _moving_declaration(self, specs: list[ToolSpec],
                            reasoning: bool) -> tuple[list[int], tuple[str, ...]]:
        """The declarations of ``specs`` a user turn opens with, where they move with the newest."""
        if not specs:
            return [], ()
        text = self._template.newest_turn_declaration(specs, reasoning)
        return (self._context.tokenize(text, add_special=False),
                tuple(spec.name for spec in specs))

    def _declaration_carrier(self) -> int | None:
        """Where the user turn that carries moving declarations sits in the table."""
        return next((i for i, msg in enumerate(self._table.messages)
                     if msg.role == "user" and msg.declares), None)

    def _move_declaration_on(self) -> None:
        """Take the declarations off the turn that carries them, for a newer turn to carry."""
        index = self._declaration_carrier()
        if index is None:
            return
        msg = self._table.messages[index]
        held, _ = self._moving_declaration(
            [spec for spec in self._moving_specs if spec.name in msg.declares], self._reasoning)
        assert msg.token_ids[:len(held)] == held, "a carrier opens with its declarations"
        self._rewrite_from(index, msg.token_ids[len(held):], ())

    def _declare_in_newest_turn(self, specs: list[ToolSpec]) -> int:
        """Add ``specs`` to the declarations the newest user turn carries.

        Returns:
            How many messages were evicted, which is none.
        """
        self._moving_specs += specs
        index = self._declaration_carrier()
        if index is not None:
            msg = self._table.messages[index]
            old, _ = self._moving_declaration(
                [spec for spec in self._moving_specs if spec.name in msg.declares],
                self._reasoning)
            new, declares = self._moving_declaration(self._moving_specs, self._reasoning)
            self._rewrite_from(index, new + msg.token_ids[len(old):], declares)
        self._cache.rest()
        return 0

    def _rewrite_from(self, index: int, tokens: list[int], declares: tuple[str, ...]) -> None:
        """Give the message at ``index`` ``tokens``, decoding it and every later one again."""
        start = self._table.messages[index].pos_start
        self._table.replace_at(index, tokens, declares)
        if self._cache.cut_to(start) is None:
            return                  # rebuilt from the table, the new tokens with it
        rest = [t for msg in self._table.messages[index:] for t in msg.token_ids]
        self._cache.decode(rest)

    def _user_turn_join_tokens(self, role: str) -> list[int]:
        """What the newest message gains before a ``role`` message is written after it.

        Raises:
            ValueError: If both are user turns and the template writes no such pair.
        """
        newest = self._table.messages[-1] if self._table.messages else None
        if role != "user" or newest is None or newest.role != "user":
            return []
        return self._join_tokens()

    def _join_tokens(self) -> list[int]:
        """What a user turn ends with where another user turn follows it."""
        join = self._template.user_turn_join
        if join is None:
            raise ValueError("this model's template does not write two user turns "
                             "in a row; write a reply between them")
        return self._context.tokenize(join, add_special=False) if join else []

    def _fragment_tokens(self, role: str, text: str, add_special: bool = False) -> list[int]:
        """Tokenize one message: tags special-on, content special-off."""
        prefix, content, suffix = self._formatter.parts(role, text)
        return tokenize_message(self._context, prefix, content, suffix, add_special=add_special)


def _is_user(item) -> bool:
    """Whether a ``begin`` history item is a user turn."""
    return not isinstance(item, ToolTurn) and item[0] == "user"


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


