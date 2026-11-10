"""ChatWrapper: the public API over one conversation, its cache holding what the model's template writes.

The conversation is kept as the messages a chat template is given, and the
cache as the template's render of them, decoded only where it differs from
what the cache holds. See :mod:`llama_chat.rendered_cache` and
design_history.md, "The render is the conversation's encoding".
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Generator
from dataclasses import replace

from .cache_context import CACHE_CONTEXT, GenerationAccumulator, ensure_answers
from .config import Config, Sampling
from .context import KVContext
from .rendered_cache import RenderedCache
from .reply import CALL_POLICIES, Reply, ReplyStream
from .reply_text import ReplyText
from .resolved_template import ResolvedTemplate
from .tools import ToolCall, ToolRegistry, ToolSpec, ToolTurn


class ChatWrapper:
    """Stateful single-conversation chat over a cache context, with KV reuse."""

    # ----- setting up ----------------------------------------------------------
    def __init__(
        self,
        config: Config | None = None,
        context=None,
        *,
        on_load_progress: Callable[[float], None] | None = None,
    ) -> None:
        """Load the model, resolve the chat template, and validate it.

        Args:
            config: The configuration; :class:`~llama_chat.config.Config`'s
                defaults where it is not given.
            context: A cache context of your own, in place of the one built
                from ``config``. :meth:`close` closes it too; only a
                construction that fails part-way leaves it alone.
            on_load_progress: Invoked during model loading with a fraction in
                ``[0.0, 1.0]``. Must not raise. Ignored when ``context`` is
                supplied.

        Raises:
            TypeError: If ``context`` does not answer everything the wrapper
                calls on it.
            ValueError: If the model ships no chat template, if its template
                fails validation against the model, or if ``config`` asks for
                a larger context than ``context`` holds.
        """
        if config is None:
            config = Config()
        owns_context = context is None
        self._context = (
            KVContext(config, on_load_progress=on_load_progress)
            if owns_context else context
        )
        self._config = self._context.config if owns_context else config
        # Everything that can be settled about this model is settled here, so
        # that what fails, fails while the wrapper is being built.
        try:
            ensure_answers(self._context, CACHE_CONTEXT)
            if config.context_size > self._context.context_size:
                raise ValueError(
                    f"context_size is {config.context_size} but the cache context holds "
                    f"{self._context.context_size} tokens; give both the same size")
            self._template = ResolvedTemplate(self._context)
        except BaseException:
            if owns_context:
                self._context.close()
            raise
        self._conversation = RenderedCache(self._context, self._template, config)
        self._closed = False
        # One KV sequence is shared across generations, so concurrent decodes
        # would corrupt it. It is held for a whole generation, not a decode.
        self._lock = threading.Lock()

    def begin(self, system_prompt: str,
              messages: list[tuple[str, str] | ToolTurn] | None = None,
              tools: list[ToolSpec] | None = None, *,
              reasoning: bool = False) -> int:
        """Reset the conversation: prefill the system prompt and recent history.

        Every other action refuses until this has run, because only this
        prefills the sequence opener and the system prompt.

        Args:
            system_prompt: Always kept, never evicted. Empty, it is no system
                message at all, and the conversation opens as the template
                opens one without.
            messages: Optional history, oldest first: ``(role, text)`` pairs,
                and a :class:`~llama_chat.tools.ToolTurn` for an assistant turn
                that called tools, written as this model's template writes one.
                A history that ends on a tool turn's result leaves that turn
                open, as a result written into a live turn does, for
                :meth:`reply` to go on with. Only the most recent are kept,
                from a user turn on: those that fit with the system prompt in
                half the context, at least the newest exchange, and no more
                than ``max_history_messages``. Older ones are dropped
                unrendered.
            tools: Tools to declare, written where this model's template
                declares them; nothing else of them is kept. Refused up front
                where this model could not use them, rather than at the first
                call.
            reasoning: Whether the model reasons before each answer, as its
                template turns that on. It belongs to the conversation, so it
                changes only here. Every reply's thought is kept out of what
                it yields, either way.

        Returns:
            How many items of ``messages`` were kept.

        Raises:
            ContextOverflowError: If the system prompt and its declarations
                leave no room under ``evict_above_tokens`` for a short question,
                or the newest exchange of ``messages``, from its user message
                on, cannot fit beside them.
            ValueError: If ``tools`` names anything or the history holds a
                tool turn and this model cannot call tools; if ``reasoning`` is
                asked of a model that cannot reason; if the template has no
                system turn and a system prompt is given; or if the template
                refuses to write the conversation. The previous conversation is
                left intact either way.
        """
        messages = messages or []
        with self._lock:
            self._ensure_open()
            # Validate before resetting: a failed begin must not destroy the
            # conversation it was about to replace.
            specs = list(tools or [])
            if specs or any(isinstance(item, ToolTurn) for item in messages):
                self._template.ensure_tools_usable()
            if reasoning:
                self._template.ensure_can_reason()
            if system_prompt and not self._template.has_system_turn:
                raise ValueError(
                    "this model's template does not keep a system prompt in place: it "
                    "writes one where the questions after it change it, so a system "
                    "prompt kept for the whole conversation cannot be")
            return self._conversation.start(system_prompt, messages, specs, reasoning)

    # ----- adding context ------------------------------------------------------
    def inject(self, text: str, role: str = "user") -> int:
        """Prefill one message as context without generating.

        Evicts the oldest exchanges until the new message fits inside both
        eviction limits, then prefills it. A question goes where the next one
        was prepared for, as :meth:`reply` puts one, and is evicted for after
        its reply, unless ``max_history_messages`` is reached first.

        Returns:
            The number of messages evicted to make room.

        Raises:
            ValueError: If no conversation has been begun, if a tool call
                has left a reply open, or if the template does not write such a
                message here.
            ContextOverflowError: If the message cannot fit even after
                evicting every non-system message. Raised before anything is
                evicted.
        """
        with self._lock:
            self._ensure_open()
            return self._conversation.add(role, text)

    # ----- asking ---------------------------------------------------------------
    def reply(self, text: str | None = None, *,
              tool_calls: str | ToolRegistry = "refuse",
              max_calls: int | None = None,
              max_reply_tokens: int | None = None,
              stop: list[str] | None = None,
              cancel: threading.Event | None = None,
              sampling: Sampling | None = None) -> ReplyStream:
        """Generate the reply, streaming it as it is generated.

        A conversation that cannot take the reply is refused at once, before a
        stream is returned; everything else waits for the first iteration, the
        headroom check included, and the conversation is checked again there,
        since another thread may change it in between. Closing the stream
        early still ends the turn and records what was generated, so the next
        generation stays valid. The stream must be driven from one thread at a
        time; to interrupt from another, pass ``cancel``. An abandoned stream
        holds the wrapper's lock until it is collected, so exhaust it, close
        it, use it in a ``with``, or call :meth:`ReplyStream.finish`.

        Args:
            text: The user request, tokenized with special-token parsing off.
                See the README, "Special tokens in message content". ``None``
                answers the newest message, a user message already written, or
                goes on with a turn a call left open once every result is in.
            tool_calls: What becomes of a tool call the model makes.

                * ``"withhold"``: the model cannot write a call at all; the
                  token that opens one is never drawn, so the reply is text. A
                  model that wanted to call may then write something else
                  entirely; see design_history.md, "Second structure review".
                * ``"refuse"``: nothing is run, not even a tool that was
                  declared. The call is kept out of the text, the model is
                  told no tool can be run, and the reply goes on, a blank line
                  parting the text before the call from the text after it
                  where they would join without whitespace.
                * ``"yield"``: each call is handed over as a
                  :class:`~llama_chat.tools.ToolCall` after the text, and the
                  turn stays open. Answer each with :meth:`inject_tool_result`,
                  in order, then call this again without ``text``.
                * A :class:`~llama_chat.tools.ToolRegistry`: each call is run
                  against it, its result written, and the reply goes on until
                  the model answers. In the last permitted round a call the
                  model writes is not recognized and stays text, so the turn
                  always closes.
            max_calls: Where a registry is given, how many calls one turn may
                use. ``None`` sets no ceiling; a call past one is answered with
                an error result rather than run. Only with a registry.
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
            ValueError: At once, if no conversation has been begun; if a tool
                call has left a reply open and ``text`` is given, or a result
                is still owed; if ``text`` is ``None`` with nothing to answer;
                or if ``max_calls`` is given without a registry. At the first
                iteration, if a registry holding tools is given to a model
                that could never call one.
            RuntimeError: At once, if the wrapper is closed.
            ContextOverflowError: If too little room remains for the reply, or
                a result or the error answering a call no longer fits. The
                turn is then open and cannot be continued.
        """
        options = dict(max_reply_tokens=max_reply_tokens, stop=stop, cancel=cancel,
                       sampling=sampling)
        runs = isinstance(tool_calls, ToolRegistry)
        if not runs and tool_calls not in CALL_POLICIES:
            raise ValueError(f"tool_calls must be {', '.join(map(repr, CALL_POLICIES))} or a "
                             f"ToolRegistry, not {tool_calls!r}")
        if max_calls is not None and not runs:
            raise ValueError(f"max_calls limits calls that are run, and "
                             f"tool_calls={tool_calls!r} runs none")
        self._ensure_open()
        # Read from what the newest change published, so it never waits for the lock.
        self._conversation.ensure_can_reply(asks_question=text is not None)
        if tool_calls == "yield":
            return ReplyStream(self._until_call(text, recognizes_calls=True, **options))
        if tool_calls == "withhold":
            return ReplyStream(self._until_call(text, recognizes_calls=False, withhold=True,
                                                **options))
        return ReplyStream(self._generate_until_answered(
            text, registry=tool_calls if runs else None, max_calls=max_calls, **options))

    def _until_call(self, text: str | None,
                    **options) -> Generator[str | ToolCall, None, Reply]:
        """One generation, from ``text`` or on from the open turn, then the calls it stopped at."""
        with self._lock:
            generation = yield from self._generate(text, **options)
        # The lock is released before the calls: a caller answers each while reading the stream.
        yield from generation.tool_calls
        return generation

    def inject_tool_result(self, call: ToolCall, result) -> None:
        """Hand the model what the tool it called returned.

        Written where this model's own template writes a result, inside the
        turn the call left open, with special-token parsing off. Nothing is
        evicted while a turn is open, so such a turn grows towards
        ``context_size``; a result too long for what is left is clipped. See
        the README, "Tool calling".

        Args:
            call: The call being answered, as it was handed to you. It must
                be the oldest unanswered one.
            result: What it returned. Anything but a string is sent as JSON.

        Raises:
            ValueError: If no call is waiting, this is not the one owed, or the
                result is neither text nor anything JSON can write. The call
                stays owed.
            ContextOverflowError: If not even an empty result leaves room for
                the reply in what is left of the context.
        """
        with self._lock:
            self._ensure_open()
            self._conversation.add_result(call, result)

    # ----- shortening the newest reply -----------------------------------------
    def truncate_last_reply(self, keep_text: str) -> None:
        """Shorten the newest assistant message to ``keep_text``.

        For a reply that was cut short where it was being read or heard: the
        model is left holding only what the reader got. The cache is cut back
        to where the shortened reply differs and decoded again from there.

        Args:
            keep_text: What stays, a prefix of the message's text. The whole
                text changes nothing.

        Raises:
            ValueError: If no conversation has been begun, if a reply is open,
                if the newest message is not the assistant's or holds a tool
                call, one that could not be read included, or if
                ``keep_text`` is not a prefix of its text.
        """
        with self._lock:
            self._ensure_open()
            self._conversation.truncate_newest(keep_text)

    def declare_tools(self, tools: list[ToolSpec]) -> int:
        """Tell the model about the tools it has not been told about yet.

        Written where the template declares tools, never as a message of its
        own; the cache is decoded again from where the declarations change it.

        Args:
            tools: What to declare. Nothing to declare does nothing, and
                neither does a tool the conversation already declares.

        Returns:
            How many messages were evicted to make room.

        Raises:
            ValueError: If no conversation has been begun, if a tool call has
                left a reply open, or if this model cannot call tools at all
                (see :meth:`begin`).
            ContextOverflowError: If the system turn with the declarations
                would leave no room under ``evict_above_tokens`` for a short
                question.
        """
        with self._lock:
            self._ensure_open()
            known = {spec.name for spec in self._conversation.tools}
            specs = [spec for spec in tools if spec.name not in known]
            if not specs:
                return 0
            return self._conversation.declare(specs, accept=self._template.ensure_tools_usable)

    # ----- introspection -------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        """How many tokens the cache currently holds."""
        return self._conversation.total_tokens

    @property
    def config(self) -> Config:
        """The configuration this wrapper runs with, its limits included.

        Where the wrapper loaded the model itself, a sampling setting left
        unset holds the value the model supplies; with a cache context of your
        own, it is the configuration as given.
        """
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

    def conversation(self) -> tuple[str, list[tuple[str, str] | ToolTurn]]:
        """The system prompt and the history the cache holds, as :meth:`begin` takes them.

        The history is every message after the system prompt, oldest first:
        ``(role, text)`` pairs, and a :class:`~llama_chat.tools.ToolTurn` for
        an assistant turn that called tools, with its calls and results.
        ``begin(*chat.conversation())``, with the same tools, begins again as
        the template writes it, a call still owed a result owed again. Safe
        from any thread, including while another is mid-generation: both are
        read as the newest finished change left them.
        """
        return self._conversation.conversation()

    @property
    def can_call_tools(self) -> bool:
        """Whether tools can be used with this model, every step of the way.

        Four things have to hold, and all are settled at construction: the
        template describes a call, it renders the declarations that tell the
        model a tool exists, the argument spelling was recovered, and it renders
        a block to answer a call in. True here is exactly what :meth:`begin`
        accepts ``tools`` for. Where only the spelling is missing a call is
        still detected, so :attr:`Reply.stop_reason` can still be
        ``"unreadable_tool_call"``, but nothing runnable comes out of it.
        """
        return self._template.can_call_tools

    @property
    def can_reason(self) -> bool:
        """Whether :meth:`begin` may be asked to have the model reason."""
        return self._template.can_reason

    @property
    def can_take_system_prompt(self) -> bool:
        """Whether :meth:`begin` accepts a system prompt: the template writes one where no later question moves it."""
        return self._template.has_system_turn

    @property
    def reply_open(self) -> bool:
        """Whether a reply is unfinished, so no new message may be added.

        True from a call until :meth:`reply` closes the turn, the results in
        between included. While it is, :meth:`reply` with a request,
        :meth:`inject` and :meth:`declare_tools` refuse.
        """
        return self._conversation.reply_open

    @property
    def owed_calls(self) -> list[ToolCall]:
        """The calls the model made that still wait for a result, oldest first.

        Each is answered with :meth:`inject_tool_result`, in this order, and
        :meth:`reply` goes on with the turn only once none is left. Safe from
        any thread, as :meth:`conversation` is: the calls as the newest
        finished change left them, each a copy that is the caller's to change.
        """
        return self._conversation.published_owed_calls()

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
            registry: What to run calls against. ``None`` refuses every call
                with an error result.
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
            deltas = self._generate(text, recognizes_calls=True, **kwargs)
            # An absent ceiling counts down without ever reaching zero, so the
            # three places that read it need no case of their own.
            calls_left = math.inf if max_calls is None else max_calls
            any_call_asked = False
            generations: list[Reply] = []
            try:
                while True:
                    generation = yield from self._read_generation(
                        deltas, after="".join(g.text for g in generations))
                    generations.append(generation)
                    if not generation.tool_calls:
                        if self._conversation.reply_open:
                            # An unreadable call: its result is already written, and
                            # the turn wants the reply that follows it. Nothing is
                            # owed, since nothing could be read, so the resume is
                            # free to run. It spends budget like any other failed
                            # call, so a model that cannot write a readable one
                            # retries at most `max_calls` times.
                            any_call_asked = True
                            if registry is not None and calls_left > 0:
                                calls_left -= 1
                            deltas = self._generate(
                                None, recognizes_calls=registry is not None and calls_left > 0,
                                **kwargs)
                            continue
                        # The ceiling binds only once a call has been asked for:
                        # without one the last generation kept its call tokens.
                        return _sum_generations(generations,
                                                any_call_asked and calls_left == 0)
                    any_call_asked = True
                    settled, calls_left = self._settle_calls(
                        generation.tool_calls, registry, calls_left, max_calls)
                    may_call_again = registry is not None and calls_left > 0
                    for call, result in settled:
                        self._conversation.add_result(call, _explain_failure(result, may_call_again))
                    deltas = self._generate(None, recognizes_calls=may_call_again, **kwargs)
            finally:
                deltas.close()

    def _read_generation(self, deltas: Generator[str, None, Reply], after: str):
        """Yield one generation's text, and return the :class:`Reply` describing it.

        Driven by hand rather than with ``for``, which would discard that
        :class:`Reply`. Where ``after``, the text shown before it, and this
        text would join without whitespace, a blank line goes between them,
        and the Reply's text starts with it.
        """
        separator = None if after and not after[-1].isspace() else ""
        while True:
            try:
                delta = next(deltas)
            except StopIteration as ended:
                return replace(ended.value, text=(separator or "") + ended.value.text)
            if separator is None and delta:
                separator = "" if delta[0].isspace() else "\n\n"
                delta = separator + delta
            yield delta

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

    def _generate(self, text: str | None, *, recognizes_calls: bool,
                  max_reply_tokens: int | None, stop: list[str] | None,
                  cancel: threading.Event | None, sampling: Sampling | None,
                  withhold: bool = False) -> Generator[str, None, Reply]:
        """Generate one reply from the cache's end and record what it left, as a Reply.

        Without ``text``, it goes on with the open turn, adding no message of
        its own, or else answers the newest message, which must be a user
        message already written. One generation, so a call it stops on leaves
        the turn open for a result. The lock must be held.

        Args:
            text, max_reply_tokens, stop, cancel, sampling: As :meth:`reply`.
            recognizes_calls: ``False`` leaves a call the model writes as text,
                so this generation cannot end on a call and the turn is certain
                to close.
            withhold: Also keep a call's opening token from ever being drawn.
        """
        self._ensure_open()
        conversation = self._conversation
        decoded_from = conversation.decoded
        n_evicted_messages = 0
        if text is not None:
            n_evicted_messages = conversation.ask(text)
        elif not conversation.reply_open:
            conversation.prompt()
        cap = max_reply_tokens if max_reply_tokens is not None else self._config.max_reply_tokens
        stops = list(self._config.stop_strings) + list(stop or [])
        n_prompt_tokens = conversation.total_tokens

        generated = GenerationAccumulator()
        reply_text = ReplyText(self._template.reasoning,
                               self._template.answer_opening(conversation.reasoning),
                               conversation.prompt_text, stops)
        pieces = conversation.generate(
            cap, generated, cancel=cancel,
            sampling=sampling, ends_turn=self._turn_end_test(reply_text),
            boundary=self._template.call_boundary(recognizes_calls, withhold))
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
        finally:
            pieces.close()
            # Runs on normal completion and on early close alike.
            n_evicted_messages += self._record_reply(generated, reply_text)
        return Reply(
            text=reply_text.answer,
            n_prefilled_tokens=conversation.decoded - decoded_from,
            n_prompt_tokens=n_prompt_tokens,
            n_generated_tokens=len(generated.token_ids),
            n_evicted_messages=n_evicted_messages,
            stop_reason=generated.stop_reason,
            tool_calls=conversation.owed_calls,
            n_reasoning_tokens=reply_text.thought_pieces,
        )

    def _turn_end_test(self, reply_text: ReplyText) -> Callable[[int], bool] | None:
        """Whether a sampled token ends the reply, besides the end-of-generation tokens."""
        turn_end = self._template.turn_end_token
        if turn_end is None:
            return None
        # Read at each sample: every piece yielded before it has been fed.
        return reply_text.turn_end_test(turn_end, self._context.token_text(turn_end))

    def _record_reply(self, generated: GenerationAccumulator, reply_text: ReplyText) -> int:
        """Record what one generation left and settle the cache after it; how many history items were evicted.

        A call that could not be read ends the generation as
        ``"unreadable_tool_call"``.
        """
        calls = self._read_calls(generated)
        called = generated.stop_reason == "tool_call"
        if called and not calls:
            generated.stop_reason = "unreadable_tool_call"
        tags, answer = self._template.reasoning, reply_text.answer
        # Given to a template that writes a past reply's thought; another is written without it.
        kept_thought = reply_text.thought if tags and tags.past_keeps_thought else ""
        return self._conversation.record_reply(answer, kept_thought, calls, called)

    def _read_calls(self, generated: GenerationAccumulator) -> list[ToolCall]:
        """The calls ``generated`` stopped on, empty where it produced no readable call.

        A call the parser cannot read counts as no call; any other fault is raised.
        """
        if generated.stop_reason != "tool_call":
            return []
        try:
            return self._template.read_calls(generated.tool_call)
        except (ValueError, RecursionError):
            return []

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ChatWrapper is closed")


def _sum_generations(generations: list[Reply], tool_limit_reached: bool) -> Reply:
    """Sum the generations of one exchange into the Reply that describes it."""
    return Reply(
        text="".join(t.text for t in generations),
        n_prefilled_tokens=sum(t.n_prefilled_tokens for t in generations),
        n_prompt_tokens=generations[-1].n_prompt_tokens,
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
