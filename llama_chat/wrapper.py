"""ChatWrapper: the public API over one conversation, its cache holding what the model's template writes.

The conversation is kept as the messages a chat template is given, and the
cache as the template's render of them, decoded only where it differs from
what the cache holds. See :mod:`llama_chat.rendered_cache` and
design_history.md, "The render is the conversation's encoding".
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import weakref
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field, replace

from .addon_files import addon_file, with_model_sampling
from .cache_context import (CACHE_CONTEXT, GenerationAccumulator, ensure_answers,
                            ensure_can_go_back)
from .config import Config, Sampling, ensure_reply_cap, generation_sampling
from .context import KVContext
from .messages import ensure_role, ensure_text
from .rendered_cache import RenderedCache
from .reply import TOOL_CALL_POLICIES, ContextOverflowError, Reply, ReplyStream, ToolCallPolicy
from .reply_text import ReplyText
from .resolved_template import ResolvedTemplate, ResolvedTemplates
from .tools import (ToolCall, ToolRegistry, ToolSpec, ToolTurn, ensure_names_unique,
                    unwritable_failure)

_log = logging.getLogger(__name__)


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
        """Load the model, read its add-on file, resolve the chat template, and validate it.

        The add-on file and the sampling the model recommends are read the
        same way for a context of your own as for one this loads.

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
            InvalidSettingError: If ``model_path`` is unset or names no
                file, where the wrapper loads the model itself.
            RuntimeError: Where the wrapper loads the model itself, if the
                model or its context cannot be created, or the cache cannot
                perform one of the edits eviction rests on.
            ValueError: If the model ships no chat template, if its template
                fails validation against the model, if ``config`` asks for
                a larger context than ``context`` holds, or if ``context`` can
                neither shift its cache, cut its end nor save a state. Also if
                the model's add-on file names its template but is malformed,
                if its ``reasoning_on`` writes the prompt as reasoning off
                does, or if a sampling value the model's GGUF recommends, which
                neither ``config`` nor the add-on file replaces, is refused.
        """
        if config is None:
            config = Config()
        owns_context = context is None
        self._context = (
            KVContext(config, on_load_progress=on_load_progress)
            if owns_context else context
        )
        # Everything that can be settled about this model is settled here.
        try:
            ensure_answers(self._context, CACHE_CONTEXT)
            if config.context_size > self._context.context_size:
                raise ValueError(
                    f"context_size is {config.context_size} but the cache context holds "
                    f"{self._context.context_size} tokens; give both the same size")
            if not owns_context:
                ensure_can_go_back(self._context)    # a loaded one is tried at load
            addon = addon_file(config.model_path, self._context.chat_template_text())
            self._config = with_model_sampling(
                config, self._context.recommended_sampling(), addon)
            self._templates = ResolvedTemplates(self._context, addon["call"], addon["reasoning_on"])
            self._conversation = RenderedCache(self._context, self._templates, self._config)
        except BaseException:
            if owns_context:
                self._context.close()
            raise
        self._closed = False
        # Held for a whole action, a generation included, not for one decode.
        self._lock = threading.Lock()
        # The thread that holds it while one does: for a reply, the one that read it last.
        self._holder: threading.Thread | None = None
        # Each reply's steps not yet collected, which close() ends first on the thread one holds it for.
        self._replies: weakref.WeakSet[Generator] = weakref.WeakSet()
        # The policy the newest turn was started with, or for a turn the history opened the
        # first one a reply gave it, and the calls it may still run, which a reply going on
        # with it keeps.
        self._turn_policy: ToolCallPolicy | ToolRegistry | None = None
        self._turn_budget = _CallBudget(None)

    def begin(self, system_prompt: str,
              messages: list[tuple[str, str] | dict | ToolTurn] | None = None,
              tools: Iterable[ToolSpec] | None = None, *,
              reasoning: bool = False) -> int:
        """Reset the conversation: prefill the system prompt and recent history.

        Every other action refuses until this has run, because only this
        prefills the sequence opener and the system prompt.

        Args:
            system_prompt: Always kept, never evicted. Empty, it is no system
                message at all, and the conversation opens as the template
                opens one without.
            messages: Optional history, oldest first: ``(role, text)`` pairs
                or ``{"role": role, "content": text}`` dictionaries, the role
                ``"user"`` or ``"assistant"``, a reply's dictionary with its
                thought under ``"reasoning_content"`` where it has one, and a
                :class:`~llama_chat.tools.ToolTurn` for an assistant turn that
                called tools, written as this model's template writes one.
                A history that ends on a tool turn's result leaves that turn
                open, as a result written into a live turn does, for
                :meth:`reply` to go on with. Only the most recent are kept,
                from a user turn on: those that fit with the system prompt in
                half the context, at least the newest exchange, and no more
                than ``max_history_exchanges``. Older ones are dropped
                unrendered.
            tools: Tools to declare, a list of specs or a registry, written
                where this model's template declares them; nothing else of
                them is kept. Refused up front
                where this model could not use them, rather than at the first
                call. Given at all, even empty, it writes the conversation
                with the model's template for tools, where it ships one, so
                that :meth:`declare_tools` can add tools later; ``None``
                writes it with the default template.
            reasoning: Whether the model reasons before each answer, as its
                template turns that on or off; :attr:`can_turn_off_reasoning`
                says whether it has an off. It belongs to the conversation, so it
                changes only here. Every reply's thought is kept out of what
                it yields, either way.

        Returns:
            How many items of ``messages`` were kept.

        Raises:
            ContextOverflowError: If the system prompt and its declarations
                leave no room under ``evict_above_tokens`` for a short question,
                or the newest exchange of ``messages``, from its user message
                on, cannot fit beside them, or ``messages`` hold no user
                message and cannot fit beside them either.
            TypeError: If the system prompt or a history message's text is
                not a ``str``, or a history item is of another form.
            ValueError: If a history message has another role, or a
                dictionary other keys; if ``tools`` names one tool twice; if
                ``tools`` names anything or the history holds a tool turn and
                this model cannot call tools; if ``reasoning`` is asked of a
                model that cannot reason; if the template has no system turn
                and a system prompt is given; if two calls of the history carry
                one id; or if the template refuses to write the conversation.
                The previous conversation is left intact either way.
        """
        ensure_text(system_prompt, "the system prompt")
        messages = messages or []
        with self._held():
            self._ensure_open()
            # Validate before resetting: a failed begin must not destroy the
            # conversation it was about to replace.
            specs = None if tools is None else list(tools)
            ensure_names_unique(specs or [])
            template = self._templates.for_conversation(specs is not None)
            if specs or any(isinstance(item, ToolTurn) for item in messages):
                template.ensure_tools_usable()
            if reasoning:
                template.ensure_can_reason()
            if system_prompt and not template.has_system_turn:
                raise ValueError(
                    "this model's template does not keep a system prompt in place: it "
                    "writes one where the questions after it change it, so a system "
                    "prompt kept for the whole conversation cannot be")
            kept = self._conversation.start(system_prompt, messages, specs, reasoning)
            # A turn the history leaves open was started by no reply.
            self._turn_policy, self._turn_budget = None, _CallBudget(None)
            return kept

    # ----- adding context ------------------------------------------------------
    def inject(self, text: str, role: str = "user") -> int:
        """Prefill one message as context without generating.

        A question goes where the next one was prepared for, as :meth:`reply`
        puts one, and the oldest exchanges are evicted for it after its reply.
        An assistant message, or a question once ``max_history_exchanges`` is
        reached, evicts them first, until it fits inside both eviction limits.

        Args:
            text: The message's text.
            role: ``"user"`` or ``"assistant"``. The system prompt is
                :meth:`begin`'s, and a tool result :meth:`inject_tool_result`'s.

        Returns:
            The number of messages evicted to make room.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            ValueError: If ``role`` is another one, if no conversation has
                been begun, if a tool call has left a reply open, or if the
                template does not write such a message here.
            ContextOverflowError: If the message cannot fit even after
                evicting every non-system message, raised before anything is
                evicted; or if the conversation with it cannot fit once every
                exchange that can go is evicted, since no user message follows
                the rest.
        """
        ensure_role(role)
        ensure_text(text, "an injected message's text")
        with self._held():
            self._ensure_open()
            return self._conversation.add(role, text)

    # ----- asking ---------------------------------------------------------------
    def reply(self, text: str | None = None, *,
              tool_call_policy: ToolCallPolicy | ToolRegistry | None = None,
              max_calls: int | None = None,
              max_reply_tokens: int | None = None,
              stop: str | list[str] | None = None,
              cancel: threading.Event | None = None,
              sampling: Sampling | None = None) -> ReplyStream:
        """Generate the reply, streaming it as it is generated.

        A conversation that cannot take the reply is refused at once, before a
        stream is returned; everything else waits for the first iteration, the
        headroom check included, and the conversation is checked again there,
        since another thread may change it in between. Closing the stream
        early records what was generated, so the next generation stays valid;
        a call already handed over stays owed, its turn open for the result.
        The stream must be driven from one thread at a time; to interrupt from
        another, pass ``cancel``. An abandoned stream holds the wrapper's lock
        until it is collected, so exhaust it, close it, use it in a ``with``,
        or call :meth:`ReplyStream.finish`.

        Args:
            text: The user request, tokenized with special-token parsing off.
                See the README, "Special tokens in message content". ``None``
                answers the newest message, a user message already written, or
                goes on with a turn a call left open once every result is in.
            tool_call_policy: What becomes of a tool call the model makes.
                ``None``, the default, keeps the policy of the turn this goes
                on with when it runs, where the reply that started it was
                given one, a registry with what is left of its ``max_calls``,
                and is ``"refuse"`` otherwise.

                * ``"withhold"``: the model cannot write a call at all; the
                  token that opens one is never drawn, so the reply is text. A
                  model that wanted to call may then write something else
                  entirely; see design_history.md, "Second structure review".
                * ``"refuse"``: nothing is run, not even a tool that was
                  declared. The call is kept out of the text, the model is
                  told no tool can be run, and the reply goes on without
                  drawing the token that opens a call, as under
                  ``"withhold"``, a blank line parting the text before the
                  call from the text after it where they would join without
                  whitespace.
                * ``"yield"``: each call is handed over as a
                  :class:`~llama_chat.tools.ToolCall` after the text, and the
                  turn stays open. Answer each with :meth:`inject_tool_result`,
                  in order, then call this again without ``text``; that reply
                  is parted from the text before the call as under
                  ``"refuse"``. A call that cannot be read is answered with a
                  failure and the reply goes on; the model may write another
                  call as long as there is room, and after
                  ``MAX_UNREADABLE_CALLS_IN_A_ROW`` such calls in a row it is
                  told not to call again.
                * A :class:`~llama_chat.tools.ToolRegistry`: each call is run
                  against it, its result written, and the reply goes on until
                  the model answers. Each call that runs is handed over in the
                  stream just before it runs. In the last permitted round the
                  token that opens a call is never drawn, as under
                  ``"withhold"``, so the turn always closes.
            max_calls: Where a registry is given, how many calls one turn may
                use, each call handed over counted. ``None`` sets no ceiling, or,
                going on with a turn, keeps what is left of the turn's; a call
                past one is answered with an error result rather than run. Only
                with a registry.
            max_reply_tokens: This request's cap on the tokens of the reply,
                every generation of it together, a thought included, in place
                of the config's.
            stop: Texts that end the answer, matched against the answer
                only; a single ``str`` is one such text. Read when this is
                called, so a later change to the list changes nothing.
            cancel: Setting it, from any thread, stops generation within one
                token and ends the turn with ``stop_reason == "interrupted"``.
            sampling: This request's own sampling settings, in place of the
                config's for the fields it sets.

        Returns:
            A :class:`ReplyStream` of the text deltas, with ``"yield"`` the
            calls after them, and with a registry each call before it runs;
            its ``reply`` is the :class:`Reply` once it has ended, summing
            every generation where calls were run.

        Raises:
            ValueError: At once, if no conversation has been begun; if a tool
                call has left a reply open and ``text`` is given, or a result
                is still owed; if ``text`` is ``None`` with nothing to answer;
                or if ``max_calls`` is given without a registry or is not a
                whole number of at least 0. At the first
                iteration, if a registry holding tools is given to a model
                that could never call one, or if ``max_calls`` is given and
                the turn the reply goes on with by then runs no call.
            TypeError: At once, if ``text`` is neither ``None`` nor a ``str``,
                ``stop`` is neither a ``str`` nor a list of ``str``, ``cancel``
                has no ``is_set``, or ``sampling`` is neither ``None`` nor a
                :class:`~llama_chat.config.Sampling`.
            InvalidSettingError: At once, if ``max_reply_tokens`` allows no token.
            RuntimeError: At once, if the wrapper is closed; at the first
                iteration, if this thread is still reading another reply.
            ContextOverflowError: If too little room remains for the reply. A
                result or the error answering a call that no longer fits
                raises nothing: the turn closes on what the model wrote, and
                the reply ends with ``"token_limit"``.
        """
        if text is not None:
            ensure_text(text, "the request's text")
        ensure_reply_cap(max_reply_tokens)
        stop_texts = _stop_texts(stop)
        _ensure_cancel_and_sampling(cancel, sampling)
        options = dict(max_reply_tokens=max_reply_tokens, stop=stop_texts, cancel=cancel,
                       sampling=generation_sampling(self._config, sampling))
        if tool_call_policy is not None and not isinstance(tool_call_policy, ToolRegistry) \
                and tool_call_policy not in TOOL_CALL_POLICIES:
            raise ValueError(f"tool_call_policy must be "
                             f"{', '.join(map(repr, TOOL_CALL_POLICIES))} or a "
                             f"ToolRegistry, not {tool_call_policy!r}")
        if max_calls is not None and (not isinstance(max_calls, int) or isinstance(max_calls, bool)
                                      or max_calls < 0):
            raise ValueError(f"max_calls must be None or a whole number of at least 0, "
                             f"not {max_calls!r}")
        # Read from what the newest change published, so it never waits for the lock; the
        # policy is chosen again under the lock, from the turn open when the reply runs.
        _ensure_max_calls_runs(self._policy_for(
            tool_call_policy, text is None and self._conversation.published_turn_open()), max_calls)
        self._ensure_open()
        self._conversation.ensure_can_reply(asks_question=text is not None)
        steps = self._noting_each_reader(self._replying(text, tool_call_policy, max_calls, **options))
        self._replies.add(steps)
        return ReplyStream(steps)

    def _noting_each_reader(self, steps: Generator[str | ToolCall, None, Reply]
                            ) -> Generator[str | ToolCall, None, Reply]:
        """``steps``, the thread that takes each step noted as the one holding the wrapper, while the reply holds it."""
        holds = False
        try:
            while True:
                if holds:
                    self._holder = threading.current_thread()
                try:
                    step = next(steps)
                except StopIteration as ended:
                    return ended.value
                holds = self._holder is threading.current_thread()
                yield step
        except GeneratorExit:
            # Closed at a step: ``steps`` still waits there, and is closed on this thread.
            if holds:
                self._holder = threading.current_thread()
            steps.close()
            raise

    def _replying(self, text: str | None, given: ToolCallPolicy | ToolRegistry | None,
                  max_calls: int | None, **options) -> Generator[str | ToolCall, None, Reply]:
        """The reply :meth:`reply` describes, its policy chosen under the lock: ``given``, else the open turn's."""
        with self._held():
            goes_on = text is None and self._conversation.turn_open
            policy = self._policy_for(given, goes_on)
            _ensure_max_calls_runs(policy, max_calls)
            budget = self._turn_budget if goes_on and max_calls is None else _CallBudget(max_calls)
            options["on_started"] = lambda: self._keep_turn(given, policy, budget)
            if policy not in ("yield", "withhold"):
                return (yield from self._generate_until_answered(
                    text, policy=policy, budget=budget, **options))
            if policy == "yield":
                generation = yield from self._generate_until_called(text, **options)
            else:
                generation = yield from self._read_generation(
                    self._generate(text, recognizes_calls=False, may_call_again=False,
                                   withhold=True, **options),
                    after=self._open_turn_text())
        # The lock is released before the calls: a caller answers each while reading the stream.
        yield from _answerable(generation.tool_calls)
        return generation

    def _policy_for(self, given: ToolCallPolicy | ToolRegistry | None,
                    goes_on: bool) -> ToolCallPolicy | ToolRegistry:
        """The policy a reply runs with: ``given``, else the turn's where it goes on with one, else ``"refuse"``."""
        # Compared with None: an empty registry is false.
        if given is not None:
            return given
        return self._turn_policy if goes_on and self._turn_policy is not None else "refuse"

    def _keep_turn(self, given: ToolCallPolicy | ToolRegistry | None,
                   policy: ToolCallPolicy | ToolRegistry, budget: _CallBudget) -> None:
        """Note what a reply whose generation has started leaves its turn; the lock must be held.

        A reply that opens a turn starts it with ``policy``; one going on with
        a turn the history opened gives it ``given``, where it gives one. The
        turn keeps ``budget``, the calls it may still run.
        """
        if not self._conversation.turn_open or (self._turn_policy is None and given is not None):
            self._turn_policy = policy
        self._turn_budget = budget

    def inject_tool_result(self, call: ToolCall, result) -> int:
        """Hand the model what the tool it called returned; how many older history items were evicted for it.

        Written where this model's own template writes a result, inside the
        turn the call left open, with special-token parsing off. A result too
        long for what is left of the context evicts the oldest exchanges
        first, the open turn kept whole; one still too long is cut, a note at
        its end telling the model how much was kept, and the next reply lists
        its call in ``Reply.tool_results_cut``. Where not even a result cut to
        nothing fits, the turn closes on what the model wrote, as a reply that
        answers its own calls closes one, and ``turn_open`` is then false. See
        the README, "Tool calling".

        Args:
            call: The call being answered, as it was handed to you, or one
                built without an id that names the same function with the
                same arguments. It must be the oldest unanswered one.
            result: What it returned. Anything but a string is sent as JSON.
                One JSON cannot write, ``NaN`` and infinity included, is
                answered with the ``unwritable_result`` failure a registered
                tool's would be, and a warning is logged.

        Raises:
            ValueError: If no call is waiting, this is not the one owed, or
                this model's template writes no result. The call stays owed.
        """
        failure = unwritable_failure(result)
        with self._held():
            self._ensure_open()
            written = result if failure is None else _explain_failure(
                failure, may_call_again=self._turn_policy == "yield")
            try:
                evicted = self._conversation.add_result(call, written)
            except ContextOverflowError:
                evicted = self._conversation.close_unanswered()
        if failure is not None:
            _log.warning("the result handed over for %s cannot be written, so the model is told "
                         "the call failed (unwritable_result): %s", call.name, failure["message"])
        return evicted

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
                call, one that could not be read included, if ``keep_text``
                is not a prefix of its text, or if the template does not write
                the reply shortened to it, as one that writes no empty reply
                does not write it shortened to ``""``.
        """
        with self._held():
            self._ensure_open()
            self._conversation.truncate_newest(keep_text)

    def declare_tools(self, tools: Iterable[ToolSpec]) -> int:
        """Tell the model about the tools it has not been told about yet.

        Written where the template declares tools, never as a message of its
        own; the cache is decoded again from where the declarations change it.
        A model that ships a template for tools takes declarations only into a
        conversation begun with ``tools``, which is written with that template.

        Args:
            tools: What to declare, a list of specs or a registry. Nothing to
                declare does nothing, and
                neither does a tool the conversation already declares.

        Returns:
            How many messages were evicted to make room.

        Raises:
            ValueError: If ``tools`` names one tool twice, if no conversation
                has been begun, if a tool call has left a reply open, if this
                model cannot call tools at all
                (see :meth:`begin`), if the model ships a template for tools
                and the conversation was begun without ``tools``, or if the template with the declarations
                writes an older exchange otherwise once another follows.
            ContextOverflowError: If the system turn with the declarations
                would leave no room under ``evict_above_tokens`` for a short
                question.
        """
        with self._held():
            self._ensure_open()
            tools = list(tools)
            ensure_names_unique(tools)
            known = {spec.name for spec in self._conversation.tools}
            specs = [spec for spec in tools if spec.name not in known]
            if not specs:
                return 0
            return self._conversation.declare(
                specs, accept=self._templates.for_conversation(True).ensure_tools_usable)

    # ----- introspection -------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        """How many tokens the cache holds, as the newest finished change left it; safe from any thread."""
        return self._conversation.published_total_tokens()

    @property
    def config(self) -> Config:
        """The configuration this wrapper runs with, its limits included.

        A sampling setting left unset holds the value the model supplies,
        through its add-on file or its own recommendation, whether the wrapper
        loaded the model itself or was given a cache context of your own.
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
            timeout_s: Seconds to wait, at least 0; ``None`` waits indefinitely.

        Returns:
            ``True`` if the wrapper went idle, ``False`` on timeout.

        Raises:
            ValueError: If ``timeout_s`` is negative.
            RuntimeError: If this thread is reading a reply, which would never
                end while it waits.
        """
        if timeout_s is not None and timeout_s < 0:
            raise ValueError(f"timeout_s must be None or at least 0, not {timeout_s!r}")
        self._ensure_not_holding()
        if not self._lock.acquire(timeout=-1 if timeout_s is None else timeout_s):
            return False
        self._lock.release()
        return True

    def conversation(self) -> dict:
        """Everything :meth:`begin` takes to begin the conversation the cache holds again, by its parameter names.

        ``system_prompt``, ``messages``, ``tools`` and ``reasoning``, so that
        ``begin(**chat.conversation())`` begins again as it was, a call still
        owed a result owed again. ``messages`` is every message after the
        system prompt, oldest first: each message a ``{"role", "content"}``
        dictionary, with ``"reasoning_content"`` where the template keeps a
        reply's thought, and a :class:`~llama_chat.tools.ToolTurn` for
        an assistant turn that called tools, with its thoughts, calls and
        results. ``tools`` is every declaration, those of
        :meth:`declare_tools` included, and ``None`` where the conversation
        was begun without tools and declares none. Safe from any thread, including while
        another is mid-generation: all of it is read as the newest finished
        change left it.
        """
        return self._conversation.conversation()

    @property
    def can_call_tools(self) -> bool:
        """Whether tools can be used with this model, every step of the way.

        Four things have to hold of the template a conversation that declares
        tools is written with, its template for tools where the model ships
        one, and all are settled at construction: the template describes a call, it renders the declarations that tell the
        model a tool exists, the argument spelling was recovered, and it renders
        a block to answer a call in. True here is exactly what :meth:`begin`
        accepts ``tools`` for. Where only the spelling is missing a call is
        still detected, so :attr:`Reply.stop_reason` can still be
        ``"unreadable_tool_call"``, but nothing runnable comes out of it.
        """
        return self._templates.for_conversation(True).can_call_tools

    @property
    def can_reason(self) -> bool:
        """Whether :meth:`begin` may be asked to have the model reason; safe from any thread, as the newest finished change left it."""
        return self._conversation.published_template().can_reason

    @property
    def can_turn_off_reasoning(self) -> bool:
        """Whether :meth:`begin` without reasoning has the model answer without it.

        Reasoning off is the template's own off. Where the template has none,
        a model that reasons does so in every reply, ``reasoning`` or not,
        and its thought is kept out of the answer either way. True where the
        template marks no thought. Safe from any thread, as the newest
        finished change left it.
        """
        return self._conversation.published_template().can_turn_off_reasoning

    @property
    def can_take_system_prompt(self) -> bool:
        """Whether :meth:`begin` accepts a system prompt: the template writes one where no later question moves it; safe from any thread, as the newest finished change left it."""
        return self._conversation.published_template().has_system_turn

    @property
    def turn_open(self) -> bool:
        """Whether a reply is unfinished, so no new message may be added.

        True from a call until :meth:`reply` closes the turn, the results in
        between included. While it is, :meth:`reply` with a request,
        :meth:`inject` and :meth:`declare_tools` refuse. Safe from any thread,
        as :meth:`conversation` is: as the newest finished change left it.
        """
        return self._conversation.published_turn_open()

    @property
    def pending_calls(self) -> list[ToolCall]:
        """The calls the model made that still wait for a result, oldest first.

        Each is answered with :meth:`inject_tool_result`, in this order, and
        :meth:`reply` goes on with the turn only once none is left. Safe from
        any thread, as :meth:`conversation` is: the calls as the newest
        finished change left them, each a copy that is the caller's to change.
        """
        return self._conversation.published_pending_calls()

    # ----- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        """Free the underlying context and model. Idempotent and thread-safe.

        Waits for any in-flight generation; on the thread that reads an
        unfinished reply, ends that reply first, as closing its stream does,
        and from a tool that reply runs, is refused as any action is.
        Closes a context passed to the constructor as well. Every action after
        close raises ``RuntimeError`` once its arguments are checked;
        :meth:`conversation` and the properties that say where it stands still
        answer.
        """
        try:
            if self._holder is threading.current_thread():
                # From a tool the reply runs, that reply is running, and close() is refused below.
                for steps in [s for s in self._replies if not s.gi_running]:
                    steps.close()
        finally:
            with self._held():
                if not self._closed:
                    self._closed = True
                    self._context.close()

    def __enter__(self) -> ChatWrapper:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----- internals -----------------------------------------------------------
    def _generate_until_answered(
        self, text: str | None, *, policy: ToolRegistry | ToolCallPolicy, budget: _CallBudget,
        on_started: Callable[[], None], max_reply_tokens: int | None,
        **kwargs) -> Generator[str | ToolCall, None, Reply]:
        """Generate, settle whatever the model calls, and go on until it answers.

        Each call that runs is handed over in the stream before it runs. The
        lock must be held, and is for the whole exchange, tools running
        included, so no other action comes between its rounds; a tool that
        calls back into the wrapper is refused.

        Args:
            policy: The registry to run calls against, or ``"refuse"``, which
                answers every call with an error result.
            budget: The calls the turn may still run. Nothing is run under
                ``"refuse"``, so it does not bind there.
            on_started: As :meth:`_generate`, for the first generation.
            text, max_reply_tokens, **kwargs: As :meth:`reply`.

        Raises:
            ValueError: If ``policy`` is a registry holding tools this model
                could never call, which is what
                :meth:`~llama_chat.resolved_template.ResolvedTemplate.ensure_tools_usable`
                decides.
        """
        registry = policy if isinstance(policy, ToolRegistry) else None
        if registry is not None and len(registry) > 0:
            self._template.ensure_tools_usable()
        any_call_asked = self._conversation.turn_open      # a turn going on has called
        open_turn_text = self._open_turn_text()
        reply_cap = self._reply_cap(max_reply_tokens)
        generations: list[Reply] = []

        unreadable_in_a_row = 0

        def may_call_after_unreadable() -> bool:
            """Whether a call is read again after an unreadable one, which spends one."""
            return (registry is not None and budget.calls_left > 1
                    and unreadable_in_a_row + 1 < MAX_UNREADABLE_CALLS_IN_A_ROW)

        def tokens_left() -> int:
            """What the reply cap leaves the next generation."""
            return max(0, reply_cap - sum(g.n_generated_tokens for g in generations))

        deltas = self._generate(text, recognizes_calls=True,
                                may_call_again=may_call_after_unreadable(),
                                max_reply_tokens=reply_cap, on_started=on_started, **kwargs)
        try:
            while True:
                generation = yield from self._read_generation(
                    deltas, after=open_turn_text + "".join(g.text for g in generations))
                generations.append(generation)
                if not _answerable(generation.tool_calls):
                    if self._conversation.turn_open:
                        # An unreadable call: its result is already written, and
                        # the turn wants the reply that follows it. Nothing is
                        # owed, since nothing could be read, so the resume is
                        # free to run. It spends budget like any other failed
                        # call, so a model that cannot write a readable one
                        # retries at most `max_calls` times, and never more than
                        # MAX_UNREADABLE_CALLS_IN_A_ROW times in a row.
                        any_call_asked = True
                        unreadable_in_a_row += 1
                        if registry is not None and budget.calls_left > 0:
                            budget.calls_left -= 1
                        deltas = self._generate_round(
                            registry is not None and budget.calls_left > 0
                            and unreadable_in_a_row < MAX_UNREADABLE_CALLS_IN_A_ROW,
                            may_call_after_unreadable(), max_reply_tokens=tokens_left(), **kwargs)
                        continue
                    # The ceiling binds only once a call has been asked for:
                    # without one the last generation kept its call tokens.
                    return _sum_generations(generations,
                                            any_call_asked and budget.calls_left == 0)
                any_call_asked = True
                unreadable_in_a_row = 0
                if tokens_left() == 0:
                    # The call spent the reply cap, so nothing would read its result: it is
                    # not run, and the turn closes as one the cap cuts does.
                    dropped = self._conversation.close_unanswered()
                    generations[-1] = replace(
                        generations[-1], stop_reason="token_limit",
                        tool_calls=[call if call.status != "complete"
                                    else ToolCall("", {}, status="unfinished")
                                    for call in generations[-1].tool_calls],
                        n_evicted_messages=generations[-1].n_evicted_messages + dropped)
                    return _sum_generations(generations, budget.calls_left == 0)
                settled = yield from self._settle_calls(
                    _answerable(generation.tool_calls), registry, budget)
                may_call_again = registry is not None and budget.calls_left > 0
                try:
                    dropped = self._write_results(settled, may_call_again)
                    generations[-1] = replace(
                        generations[-1],
                        n_evicted_messages=generations[-1].n_evicted_messages + dropped)
                except ContextOverflowError:
                    # No room is left to answer: the turn closes on what the model
                    # wrote, as one the reply cap cuts does.
                    dropped = self._conversation.close_unanswered()
                    generations[-1] = replace(
                        generations[-1], stop_reason="token_limit",
                        n_evicted_messages=generations[-1].n_evicted_messages + dropped)
                    return _sum_generations(generations, budget.calls_left == 0)
                deltas = self._generate_round(may_call_again, may_call_after_unreadable(),
                                              max_reply_tokens=tokens_left(), **kwargs)
        finally:
            deltas.close()

    def _generate_round(self, recognizes_calls: bool, may_call_again: bool,
                        **kwargs) -> Generator[str, None, Reply]:
        """Go on with the open turn after a round of results; one that recognizes no call cannot open one with a tag."""
        return self._generate(None, recognizes_calls=recognizes_calls,
                              may_call_again=may_call_again, withhold=not recognizes_calls,
                              **kwargs)

    def _generate_until_called(self, text: str | None, *, max_reply_tokens: int | None,
                               **kwargs) -> Generator[str, None, Reply]:
        """Generate until the model makes a call that can be handed over, or answers.

        A call that cannot be read is answered with a failure and the reply
        goes on, every generation of it within one reply cap; once
        ``MAX_UNREADABLE_CALLS_IN_A_ROW`` such calls came in a row, it goes on
        reading no call. The lock must be held.
        """
        reply_cap = self._reply_cap(max_reply_tokens)
        open_turn_text = self._open_turn_text()
        generations: list[Reply] = []
        while True:
            used = sum(g.n_generated_tokens for g in generations)
            unreadable_in_a_row = len(generations)
            recognizes_calls = unreadable_in_a_row < MAX_UNREADABLE_CALLS_IN_A_ROW
            generation = yield from self._read_generation(
                self._generate(text, recognizes_calls=recognizes_calls,
                               may_call_again=unreadable_in_a_row + 1 < MAX_UNREADABLE_CALLS_IN_A_ROW,
                               withhold=not recognizes_calls,
                               max_reply_tokens=max(0, reply_cap - used), **kwargs),
                after=open_turn_text + "".join(g.text for g in generations))
            generations.append(generation)
            if _answerable(generation.tool_calls) or not self._conversation.turn_open:
                return generation if len(generations) == 1 else _sum_generations(generations, False)
            text = None
            kwargs.pop("on_started", None)

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

    def _open_turn_text(self) -> str:
        """What the model has written so far in the turn a call left open; empty where none is open."""
        if not self._conversation.turn_open:
            return ""
        written = []
        for message in reversed(self._conversation.messages):
            if message["role"] not in ("assistant", "tool"):
                break
            if message["role"] == "assistant":
                written.append(message["content"])
        return "".join(reversed(written))

    def _settle_calls(self, asked: list[ToolCall], registry: ToolRegistry | None,
                      budget: _CallBudget):
        """Decide what every call of one round gets back, running what may run, and hand over each call that runs before it runs.

        Settled before any of it is written, because what a failure tells the
        model to do next depends on the budget the last call leaves. In the
        order asked, which the template renders in. A call handed over counts
        against ``budget`` from then on. Where the stream is closed at a call
        handed over, the results of the calls that ran before it are written,
        and it and the rest of the round stay owed.

        Returns:
            The ``(call, result)`` pairs.
        """
        settled = []
        for call in asked:
            if registry is None:
                # Not "no tools exist": `begin` may have declared some. Raising
                # would leave the turn open with no way to end it.
                result = {"error": "tools_not_run",
                          "message": "no tool can be run for this request"}
            elif budget.calls_left > 0:
                budget.calls_left -= 1
                try:
                    yield call
                except GeneratorExit:
                    # Where no room is left, the calls stay owed, as the caller's own
                    # result would find no room either.
                    with contextlib.suppress(ContextOverflowError):
                        self._write_results(settled, budget.calls_left > 0)
                    raise
                result = registry.call(call)
            else:
                result = {"error": "tool_limit_reached",
                          "message": "no tool calls left in this turn; "
                                     f"the limit is {budget.max_calls}"}
            settled.append((call, result))
        return settled

    def _write_results(self, settled: list[tuple[ToolCall, object]], may_call_again: bool) -> int:
        """Write the result of each settled call, in order, a failure saying whether another call is worth making; how many history items were evicted for them."""
        return sum(self._conversation.add_result(call, _explain_failure(result, may_call_again))
                   for call, result in settled)

    def _generate(self, text: str | None, *, recognizes_calls: bool, may_call_again: bool,
                  max_reply_tokens: int | None, stop: tuple[str, ...],
                  cancel: threading.Event | None, sampling: Sampling,
                  withhold: bool = False,
                  on_started: Callable[[], None] = lambda: None) -> Generator[str, None, Reply]:
        """Generate one reply from the cache's end and record what it left, as a Reply.

        Without ``text``, it goes on with the open turn, adding no message of
        its own, or else answers the newest message, which must be a user
        message already written. One generation, so a call it stops on leaves
        the turn open for a result. The lock must be held.

        Args:
            text, stop, cancel: As :meth:`reply`.
            max_reply_tokens: The cap on this generation's tokens, ``None``
                for the config's.
            sampling: Every value the generation draws with.
            recognizes_calls: ``False`` leaves a call the model writes as text,
                so this generation cannot end on a call and the turn is certain
                to close.
            may_call_again: Whether a call is read again after a call this
                generation writes that cannot be read, which the answer to
                that call tells the model.
            withhold: Also keep a call's opening token from ever being drawn.
            on_started: Called once the request is written and the generation
                starts, before anything it produces is recorded.
        """
        self._ensure_open()
        conversation = self._conversation
        decoded_from = conversation.decoded
        n_evicted_messages = 0
        if text is not None:
            n_evicted_messages = conversation.ask(text)
        elif not conversation.turn_open:
            conversation.prompt()
        max_generated_tokens = self._reply_cap(max_reply_tokens)
        n_prompt_tokens = conversation.total_tokens

        generated = GenerationAccumulator()
        results_cut = conversation.results_cut
        reply_text = ReplyText(self._template.reasoning,
                               self._template.answer_opening(conversation.reasoning),
                               self._template.reply_header(conversation.reasoning),
                               conversation.prompt_text, list(stop), generated.tag_spans)
        pieces = conversation.generate(
            max_generated_tokens, generated, cancel=cancel,
            sampling=sampling,
            ends_turn=self._turn_end_test(reply_text, generated),
            boundary=replace(self._template.call_boundary(recognizes_calls, withhold),
                             starts_in_thought=reply_text.opens_in_thought))
        on_started()
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
            n_evicted_messages += self._record_reply(generated, reply_text, may_call_again)
        return Reply(
            text=reply_text.answer,
            n_prefilled_tokens=conversation.decoded - decoded_from,
            n_prompt_tokens=n_prompt_tokens,
            n_generated_tokens=len(generated.token_ids),
            n_evicted_messages=n_evicted_messages,
            stop_reason=generated.stop_reason,
            tool_calls=_reported_calls(generated, conversation.pending_calls),
            tool_results_cut=results_cut,
            n_reasoning_tokens=reply_text.thought_pieces,
            thought=reply_text.thought,
        )

    def _reply_cap(self, max_reply_tokens: int | None) -> int:
        """The cap on a reply's tokens: ``max_reply_tokens``, else the config's, else the context's size."""
        return next((c for c in (max_reply_tokens, self._config.max_reply_tokens) if c is not None),
                    self._config.context_size)

    def _turn_end_test(self, reply_text: ReplyText,
                       generated: GenerationAccumulator) -> Callable[[int], bool] | None:
        """Whether a sampled token ends the reply ``generated`` collects, besides the end-of-generation tokens."""
        turn_end = self._template.turn_end_token
        if turn_end is None:
            return None
        return reply_text.turn_end_test(turn_end, self._context.token_text(turn_end),
                                        generated.token_ids, self._context.token_bytes,
                                        self._context.is_special)

    def _record_reply(self, generated: GenerationAccumulator, reply_text: ReplyText,
                      may_call_again: bool) -> int:
        """Record what one generation left and settle the cache after it; how many history items were evicted.

        A call that could not be read ends the generation as
        ``"unreadable_tool_call"``, and is answered with a failure that says
        whether to try again.
        """
        calls = self._read_calls(generated)
        called = generated.stop_reason == "tool_call"
        if called and not calls:
            generated.stop_reason = "unreadable_tool_call"
        tags, answer = self._template.reasoning, reply_text.answer
        # Given to a template that writes a past reply's thought; another is written without it.
        kept_thought = reply_text.thought if tags and tags.past_keeps_thought else ""
        return self._conversation.record_reply(
            answer, kept_thought, calls, called, _explain_failure(_UNREADABLE, may_call_again),
            stopped=reply_text.met_stop_string)

    def _read_calls(self, generated: GenerationAccumulator) -> list[ToolCall]:
        """The calls ``generated`` stopped on, empty where it produced no readable call.

        A call the parser cannot read counts as no call; any other fault is raised.
        """
        if generated.stop_reason != "tool_call":
            return []
        try:
            return self._template.read_calls(
                generated.tool_call,
                {spec.name: spec.parameters for spec in self._conversation.tools})
        except ValueError:
            return []

    @property
    def _template(self) -> ResolvedTemplate:
        """The template the conversation is written with now."""
        return self._conversation.template

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ChatWrapper is closed")

    @contextlib.contextmanager
    def _held(self):
        """Hold the wrapper for one action, refusing one the thread that already holds it asks for."""
        self._ensure_not_holding()
        with self._lock:
            self._holder = threading.current_thread()
            try:
                yield
            finally:
                self._holder = None

    def _ensure_not_holding(self) -> None:
        """Refuse an action on the thread whose reply still holds the wrapper, which would wait for itself.

        Raises:
            RuntimeError: Naming the reply this thread is reading.
        """
        if self._holder is threading.current_thread():
            raise RuntimeError(
                "the reply this thread is reading still holds this conversation, and an "
                "action now would wait for that reply for ever; read the reply to its end "
                "or close it first, and call nothing on the wrapper from a tool it runs")


@dataclass
class _CallBudget:
    """How many calls a turn run against a registry may still use, and the ceiling it was given, ``None`` for none."""

    max_calls: int | None
    # An absent ceiling counts down without ever reaching zero, so what reads it
    # needs no case of its own.
    calls_left: float = field(init=False)

    def __post_init__(self) -> None:
        self.calls_left = math.inf if self.max_calls is None else self.max_calls


def _reported_calls(generated: GenerationAccumulator, pending: list[ToolCall]) -> list[ToolCall]:
    """The calls one generation reports: those it left owed a result, or the one it could not read or finish."""
    if generated.stop_reason == "unreadable_tool_call":
        return [ToolCall("", {}, status="unreadable", text=generated.tool_call)]
    if generated.cut_inside_call:
        return [ToolCall("", {}, status="unfinished")]
    return pending


def _stop_texts(stop) -> tuple[str, ...]:
    """A request's stop texts, read once; a single ``str`` is one stop text.

    Raises:
        TypeError: If ``stop`` is neither a ``str`` nor a list of ``str``.
    """
    texts = (stop,) if isinstance(stop, str) else tuple(stop) if isinstance(stop, Iterable) else stop
    if texts is not None and (not isinstance(texts, tuple)
                              or not all(isinstance(text, str) for text in texts)):
        raise TypeError(f"stop must be a str or a list of str, not {stop!r}")
    return texts or ()


def _ensure_cancel_and_sampling(cancel, sampling) -> None:
    """Refuse a request's ``cancel`` or ``sampling`` of a kind :meth:`ChatWrapper.reply` cannot use.

    Raises:
        TypeError: Naming the option and what it must be.
    """
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise TypeError(f"cancel must be a threading.Event, not {cancel!r}")
    if sampling is not None and not isinstance(sampling, Sampling):
        raise TypeError(f"sampling must be a Sampling, not {sampling!r}")


def _ensure_max_calls_runs(policy: ToolCallPolicy | ToolRegistry, max_calls: int | None) -> None:
    """Refuse a ceiling on calls under a policy that runs none.

    Raises:
        ValueError: If ``max_calls`` is given and ``policy`` is no registry.
    """
    if max_calls is not None and not isinstance(policy, ToolRegistry):
        raise ValueError(f"max_calls limits calls that are run, and "
                         f"tool_call_policy={policy!r} runs none")


def _answerable(calls: list[ToolCall]) -> list[ToolCall]:
    """Of the calls a generation reports, those that were read, which a result answers."""
    return [call for call in calls if call.status == "complete"]


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
        tool_results_cut=[call for t in generations for call in t.tool_results_cut],
        n_reasoning_tokens=sum(t.n_reasoning_tokens for t in generations),
        thought="\n\n".join(t.thought for t in generations if t.thought),
    )


# How many calls that could not be read one reply answers before it reads no more calls.
MAX_UNREADABLE_CALLS_IN_A_ROW = 3
# What a call that could not be read is answered with, before what to do next.
_UNREADABLE = {"error": "unreadable_call", "message": "the call could not be read"}


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
