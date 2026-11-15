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
from .cache_context import CACHE_CONTEXT, ensure_answers, ensure_can_go_back
from .config import Config, Sampling, ensure_reply_cap, generation_sampling
from .context import KVContext
from .messages import ensure_role, ensure_text
from .history_forms import ensure_saved_turns
from .rendered_cache import RenderedCache
from .errors import (ContextOverflowError, InvalidArgumentError, InvalidSettingError,
                     UnsupportedError)
from .reply import TOOL_CALL_POLICIES, Reply, ReplyStream, Report, ToolCallPolicy
from .resolved_template import Capabilities, ResolvedTemplate, ResolvedTemplates
from .tool_registry import ToolRegistry
from .tools import ToolCall, ToolSpec, ensure_declarable, unwritable_failure

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
                if the file beside the model is not JSON, if its
                ``reasoning_on`` writes the prompt as reasoning off
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
                raise InvalidSettingError(
                    "context_size",
                    f"context_size is {config.context_size} but the cache context holds "
                    f"{self._context.context_size} tokens; give both the same size")
            if not owns_context:
                ensure_can_go_back(self._context)    # a loaded one is tried at load
            addon = addon_file(config.model_path, self._context.chat_template_text())
            self._config = with_model_sampling(
                config, self._context.recommended_sampling(), addon)
            self._templates = ResolvedTemplates(self._context, addon["call"], addon["reasoning_on"])
            if any(self._templates.capabilities(tools).can_call_tools for tools in (False, True)):
                # Worked out now, so that the first turn kept as generated does not wait for it.
                self._context.model_fingerprint()
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

    def begin(self, messages: list[dict] | None = None,
              tools: Iterable[ToolSpec] | None = None, *,
              reasoning: bool = False, generated_turns: dict | None = None) -> Report:
        """Begin the conversation, or begin it again: prefill the system prompt and recent history, decoding only where they differ from what the cache holds.

        Every other action refuses until this has run, because only this
        prefills the sequence opener and the system prompt.

        Args:
            messages: The conversation so far, oldest first, as Chat Completions chat
                messages, or ``None`` for none:

                * ``{"role": "system", "content": text}``, only first: the
                  system prompt, always kept, never evicted. An empty one is
                  a system turn with no text; without one, the conversation
                  opens as the template opens one without.
                * ``{"role": "user", "content": text}``: a question.
                * ``{"role": "assistant", "content": text}``: a reply, its
                  thought under ``"reasoning_content"`` where it has one, and
                  its calls under ``"tool_calls"``, each ``{"id": id, "type":
                  "function", "function": {"name": name, "arguments": json}}``,
                  the arguments a JSON object as a string, ``""`` for none;
                  ``content`` may be ``None`` where it makes calls.
                * ``{"role": "tool", "tool_call_id": id, "content": text}``:
                  a call's result, in the order of the calls.

                Every call needs an id of its own. A call whose arguments are
                no JSON object could not be read: it is left out with its
                result, and a later result for it is refused. A history that
                ends on a call or a result leaves that turn open, as a live
                turn is left, for :meth:`reply` to go on with. Only the most
                recent are kept, from a user turn on: those that fit with the
                system prompt in half the context, or under the eviction
                threshold where the cache holds the oldest of them already,
                at least the newest exchange, and no more than
                ``max_history_exchanges``; the
                newest exchange may take the room a reply leaves, its results
                cut to fit as :meth:`inject_tool_result` cuts one. Older ones
                are dropped unrendered.
            tools: Tools to declare, a list of :class:`~llama_chat.tools.ToolSpec`
                such as ``registry.specs()`` gives, written
                where this model's template declares them; nothing else of
                them is kept. Refused up front
                where this model could not use them, rather than at the first
                call. Given at all, even empty, it writes the conversation
                with the model's template for tools, where it ships one;
                ``None`` writes it with the default template. Beginning
                again with the conversation and other tools changes them,
                the cache decoded again only from where it differs.
            reasoning: Whether the model reasons before each answer, as its
                template turns that on or off; :meth:`capabilities` says
                whether it has an off. It belongs to the conversation, and
                changes as the tools do. Every reply's thought is kept out of
                what it yields, either way.
            generated_turns: What :meth:`conversation` gave beside the
                messages: each turn the model generated with a call, as its
                tokens and its messages as the conversation held them. On
                the model it came from, each such turn the messages still
                hold, found by its messages and the question before them, is
                put back as it was and restored as generated, token for
                token; on another model, or without it, every turn is written
                as this model's template writes it. See design_history.md,
                "Two history forms".

        Returns:
            What it changed beyond what was asked: how many of the messages
            after the system message it kept and why it left the others out,
            the calls it left out, the results it cut, and the generated
            turns it restored. Each such change is also logged once, at
            ``INFO``, under the logger ``llama_chat``.

        Raises:
            ContextOverflowError: If the system prompt and its declarations
                leave no room under ``evict_above_tokens`` for a short question,
                or the newest exchange of ``messages``, from its user message
                on, leaves no room for a reply beside them even with its
                results cut to nothing (one that ends on a call no result
                follows yet may fill the context, as the call did when it was
                made), or ``messages`` hold no user message and cannot fit
                beside them either.
            TypeError: If ``messages`` is no list, a message or a call is no
                dictionary, a text, a function's name or a call's arguments is
                of another type than stated above, or ``tools`` holds anything
                but tool specs.
            UnsupportedError: If ``tools`` names anything, or the history
                holds a call or a result, and the template this conversation is
                written with cannot call tools; if ``reasoning`` is asked and
                that template cannot reason; or if a system message is given
                and it has no system turn. Its ``field`` names the parameter:
                ``tools``, ``messages`` or ``reasoning``.
                :meth:`capabilities` says beforehand what is accepted.
            InvalidArgumentError: If a message holds a key its role does not
                take here, such as ``name``, a list of content parts or a role
                such as ``developer``, each named; if ``tools`` names one tool
                twice; if two calls of the history carry one id, or a call has
                none; if a result has no ``tool_call_id``, is no text, answers
                no call owed one, or not the oldest; or if ``generated_turns``
                is not what :meth:`conversation` gives, naming
                ``generated_turns``.
            TemplateRefusedError: If the template refuses to write the
                conversation.
            RuntimeError: If the wrapper is closed, or this thread is still
                reading a reply.
            The previous conversation is left intact on every refusal here.
        """
        ensure_saved_turns(generated_turns)
        if messages is not None and (isinstance(messages, (str, bytes, dict))
                                     or not isinstance(messages, Iterable)):
            raise TypeError(f"messages must be a list of chat message dictionaries, the system "
                            f"message first where there is one, not {messages!r}")
        if tools is not None and not isinstance(tools, Iterable):
            raise TypeError(f"tools must be a list of ToolSpec, such as ToolRegistry.specs() "
                            f"gives, not {tools!r}")
        messages = list(messages or [])         # read once, a generator included
        with self._held():
            self._ensure_open()
            # Validate before resetting: a failed begin must not destroy the
            # conversation it was about to replace.
            specs = None if tools is None else list(tools)
            ensure_declarable(specs or [])
            uses_tools = specs is not None
            calls = any(_calls_tools(item) for item in messages)
            if calls and not uses_tools and not self._templates.capabilities(False).can_call_tools \
                    and self._templates.capabilities(True).can_call_tools:
                raise UnsupportedError(
                    "messages", "this model reads a turn that called tools only in a "
                    "conversation begun with tools; pass tools, an empty list included")
            opens_on_system = bool(messages) and isinstance(messages[0], dict) \
                and messages[0].get("role") == "system"
            self._templates.ensure_accepts(
                uses_tools, tools_field="tools" if specs else "messages" if calls else None,
                reasoning=reasoning, system_prompt_field="messages" if opens_on_system else None)
            self._conversation.report()         # changes of earlier actions are told already
            # A begin leaves no turn a reply started, so a call it could not read is answered
            # as one in a turn that runs no call.
            self._conversation.start(messages, specs, reasoning, generated_turns,
                                     _explain_failure(_UNREADABLE, may_call_again=False))
            # A turn the history leaves open was started by no reply.
            self._conversation.turn_policy = None
            self._conversation.turn_budget = _CallBudget(None)
            return self._conversation.report()

    # ----- adding context ------------------------------------------------------
    def inject(self, text: str, role: str = "user") -> Report:
        """Prefill one message as context without generating.

        A question goes where the next one was prepared for, as :meth:`reply`
        puts one, and the oldest exchanges are evicted for it after its reply.
        An assistant message, or a question once ``max_history_exchanges`` is
        reached, evicts them first, until it fits inside both eviction limits.
        A question closes a turn a call left open first, as :meth:`reply`
        with ``text`` does, and joins a question no reply follows where the
        template writes no two in a row, or, where the two joined would not
        fit beside the system prompt, takes its place, the unanswered one
        counted as evicted.

        Args:
            text: The message's text.
            role: ``"user"`` or ``"assistant"``. The system prompt is
                :meth:`begin`'s, and a tool result :meth:`inject_tool_result`'s.

        Returns:
            What it changed beyond what was asked: the history messages evicted
            to make room, and why; a turn it
            closed or a question it joined.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            InvalidArgumentError: If ``role`` is another one.
            OutOfOrderError: If no conversation has been begun, or an
                assistant message is given while a tool call has left a reply
                open.
            TemplateRefusedError: If the template does not write an assistant
                message here.
            RuntimeError: If the wrapper is closed, or this thread is still
                reading a reply.
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
            self._conversation.report()
            self._conversation.add(role, text)
            return self._conversation.report()

    # ----- asking ---------------------------------------------------------------
    def reply(self, text: str | None = None, *,
              tool_call_policy: ToolCallPolicy | ToolRegistry | None = None,
              max_calls: int | None = None,
              max_reply_tokens: int | None = None,
              stop: str | Iterable[str] | None = None,
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
                * A :class:`~llama_chat.tool_registry.ToolRegistry`: each call is run
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
                only; a single ``str`` is one such text, any other iterable
                holds them. Read when this is called, so a later change to
                it changes nothing.
            cancel: Setting it, from any thread, stops generation within one
                token and ends the turn with ``stop_reason == "interrupted"``.
            sampling: This request's own sampling settings, in place of the
                config's for the fields it sets.

        Returns:
            A :class:`ReplyStream` of the text deltas, with ``"yield"`` the
            calls after them, and with a registry each call before it runs;
            its ``reply`` is the :class:`Reply` once it has ended, summing
            every generation where calls were run.

        A question, ``text`` given, is never refused for the state the
        conversation is in: a turn a call left open is closed first, each call
        still owed told it was not run and an empty reply added where the
        template needs one (or, where the template writes neither, the turn
        taken out), and a question no reply follows is joined to it where the
        template writes no two in a row, or, where the two joined would not
        fit beside the system prompt, replaced by it, the unanswered one
        counted as evicted. It is refused only where it cannot
        fit beside the system prompt alone, or where the template does not
        write that question as its text, such as one asked twice or an empty
        one on some templates; either refusal leaves the next question free.
        See design_history.md, "Nothing blocks a question".

        Raises:
            OutOfOrderError: At once, if no conversation has been begun; or,
                without ``text``, if a result is still owed or there is
                nothing to answer.
            InvalidArgumentError: At once, if ``tool_call_policy`` is a text
                that is none of :data:`~llama_chat.reply.TOOL_CALL_POLICIES`,
                or ``max_calls`` is below 0 or given without a registry; at
                the first iteration, if ``max_calls`` is given and the turn the
                reply goes on with by then runs no call.
            UnsupportedError: At the first iteration, if a registry holding
                tools is given to a model that could never call one, naming
                ``tool_call_policy``.
            TypeError: At once, if ``text`` is neither ``None`` nor a ``str``,
                ``tool_call_policy`` is neither a text nor a registry,
                ``max_calls`` is no whole number, ``stop`` is neither a
                ``str`` nor an iterable of ``str``, ``cancel`` has no
                ``is_set``, or ``sampling`` is neither ``None`` nor a
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
        # This stream's newest generation's Reply, which the stream reads where it is closed early.
        newest: dict[str, Reply] = {}
        options = dict(max_reply_tokens=max_reply_tokens, stop=stop_texts, cancel=cancel,
                       sampling=generation_sampling(self._config, sampling),
                       on_reply=lambda reply: newest.update(reply=reply))
        if tool_call_policy is not None and not isinstance(tool_call_policy, (str, ToolRegistry)):
            raise TypeError(f"tool_call_policy must be one of "
                            f"{', '.join(map(repr, TOOL_CALL_POLICIES))} or a ToolRegistry, "
                            f"not {tool_call_policy!r}")
        if isinstance(tool_call_policy, str) and tool_call_policy not in TOOL_CALL_POLICIES:
            raise InvalidArgumentError(
                "tool_call_policy", f"tool_call_policy must be one of "
                f"{', '.join(map(repr, TOOL_CALL_POLICIES))}, not {tool_call_policy!r}")
        if max_calls is not None and (not isinstance(max_calls, int) or isinstance(max_calls, bool)):
            raise TypeError(f"max_calls must be None or a whole number, not {max_calls!r}")
        if max_calls is not None and max_calls < 0:
            raise InvalidArgumentError("max_calls", f"max_calls must be at least 0, not {max_calls}")
        # Read from what the newest change published, so it never waits for the lock; the
        # policy is chosen again under the lock, from the turn open when the reply runs.
        _ensure_max_calls_runs(self._policy_for(
            tool_call_policy, text is None and self._conversation.published_turn_open()), max_calls)
        self._ensure_open()
        self._conversation.ensure_can_reply(asks_question=text is not None)
        steps = self._noting_each_reader(self._replying(text, tool_call_policy, max_calls, **options))
        self._replies.add(steps)
        return ReplyStream(steps, on_close=lambda: newest.get("reply"))

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
            self._conversation.report()         # changes of earlier actions are told already
            goes_on = text is None and self._conversation.turn_open
            policy = self._policy_for(given, goes_on)
            _ensure_max_calls_runs(policy, max_calls)
            budget = (self._conversation.turn_budget if goes_on and max_calls is None
                      else _CallBudget(max_calls))
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
                    after=self._open_turn_text() if text is None else "")
        # The lock is released before the calls: a caller answers each while reading the stream.
        yield from _answerable(generation.tool_calls)
        return generation

    def _policy_for(self, given: ToolCallPolicy | ToolRegistry | None,
                    goes_on: bool) -> ToolCallPolicy | ToolRegistry:
        """The policy a reply runs with: ``given``, else the turn's where it goes on with one, else ``"refuse"``."""
        # Compared with None: an empty registry is false.
        if given is not None:
            return given
        policy = self._conversation.turn_policy
        return policy if goes_on and policy is not None else "refuse"

    def _keep_turn(self, given: ToolCallPolicy | ToolRegistry | None,
                   policy: ToolCallPolicy | ToolRegistry, budget: _CallBudget) -> None:
        """Note what a reply whose generation has started leaves its turn; the lock must be held.

        A reply that opens a turn starts it with ``policy``; one going on with
        a turn the history opened gives it ``given``, where it gives one. The
        turn keeps ``budget``, the calls it may still run.
        """
        conversation = self._conversation
        if not conversation.turn_open or (conversation.turn_policy is None and given is not None):
            self._conversation.turn_policy = policy
        self._conversation.turn_budget = budget

    def inject_tool_result(self, call_id: str, result) -> Report:
        """Hand the model what the tool it called returned; what that changed beyond what was asked.

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
            call_id: The id of the call being answered, as it was handed to
                you; the oldest one still owed a result.
            result: What it returned. Anything but a string is written as
                JSON at once, and :meth:`conversation` gives that text back.
                One JSON cannot write, ``NaN`` and infinity included, is
                answered with the ``unwritable_result`` failure a registered
                tool's would be, and a warning is logged.

        Returns:
            What it changed beyond what was asked: the history messages evicted
            for it, the results cut to fit, and whether the turn closed for
            want of room instead, its result dropped.

        Raises:
            TypeError: If ``call_id`` is no ``str``.
            OutOfOrderError: If no conversation has been begun; or, its
                ``field`` then ``call_id``, if no call is waiting, ``call_id``
                is not the one owed, or it names a call :meth:`begin` left out
                as one that could not be read. The call stays owed.
            UnsupportedError: If this model's template writes no result,
                naming ``result``.
            RuntimeError: If the wrapper is closed, or this thread is still
                reading a reply.
        """
        if not isinstance(call_id, str):
            raise TypeError(f"call_id must be the id of the call being answered, a str, "
                            f"not {call_id!r}")
        failure = unwritable_failure(result)
        with self._held():
            self._ensure_open()
            # A failure is told as a failure a registered tool's is, whoever ran the tool.
            policy = self._conversation.turn_policy
            may_call_again = policy == "yield" or (isinstance(policy, ToolRegistry)
                                                   and self._conversation.turn_budget.calls_left > 0)
            written = _explain_failure(result if failure is None else failure, may_call_again)
            self._conversation.report()
            try:
                self._conversation.add_result(call_id, written)
            except ContextOverflowError:
                self._conversation.close_unanswered()
            report = self._conversation.report()
        if failure is not None:
            told = ("the turn closed for want of room instead" if report.turn_closed
                    else "so the model is told the call failed (unwritable_result)")
            _log.warning("the result handed over for %s cannot be written, %s: %s",
                         call_id, told, failure["message"])
        return report

    # ----- shortening the newest reply -----------------------------------------
    def truncate_last_reply(self, keep_text: str) -> None:
        """Shorten the newest assistant message to ``keep_text``.

        For a reply that was cut short where it was being read or heard: the
        model is left holding only what the reader got. The cache is cut back
        to where the shortened reply differs and decoded again from there.

        Args:
            keep_text: What stays, a prefix of the message's text, or of the
                text its reply was streamed as, which after a tool's result
                may open with a blank line. The whole text changes nothing.

        Raises:
            TypeError: If ``keep_text`` is not a ``str``.
            OutOfOrderError: If no conversation has been begun, if a reply is
                open, or if the newest message is not the assistant's or holds
                a tool call, one that could not be read included.
            InvalidArgumentError: If ``keep_text`` is not a prefix of its text.
            TemplateRefusedError: If the template does not write the reply
                shortened to it, as one that writes no empty reply does not
                write it shortened to ``""``; or, for the reply that ends a
                turn that called tools, if its tokens as generated hold more
                than its text, such as a call that went for want of room.
            RuntimeError: If the wrapper is closed, or this thread is still
                reading a reply.
        """
        ensure_text(keep_text, "keep_text")
        with self._held():
            self._ensure_open()
            messages = self._conversation.messages
            if messages and messages[-1]["role"] == "assistant":
                parting = _parting(_shown_before(messages), messages[-1]["content"])
                if parting and (keep_text.startswith(parting) or parting.startswith(keep_text)):
                    keep_text = keep_text[len(parting):]
            self._conversation.truncate_newest(keep_text)

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
            timeout_s: Seconds to wait, at least 0; ``None``, infinity or more
                than the platform's lock can wait waits indefinitely.

        Returns:
            ``True`` if the wrapper went idle, ``False`` on timeout.

        Raises:
            ValueError: If ``timeout_s`` is negative or not a number.
            RuntimeError: If this thread is reading a reply, which would never
                end while it waits.
        """
        if timeout_s is not None and (isinstance(timeout_s, bool)
                                      or not isinstance(timeout_s, (int, float))):
            raise TypeError(f"timeout_s must be None or a number of seconds, not {timeout_s!r}")
        if timeout_s is not None and (timeout_s != timeout_s or timeout_s < 0):    # NaN is unequal
            raise InvalidArgumentError(
                "timeout_s", f"timeout_s must be a number of seconds of at least 0, not {timeout_s}")
        self._ensure_not_holding()
        # Longer than the lock can wait is waiting indefinitely, infinity included.
        waits = timeout_s is None or timeout_s > threading.TIMEOUT_MAX
        if not self._lock.acquire(timeout=-1 if waits else timeout_s):
            return False
        self._lock.release()
        return True

    def conversation(self) -> dict:
        """Everything :meth:`begin` takes to begin the conversation the cache holds again, by its parameter names.

        ``messages``, ``tools``, ``reasoning`` and ``generated_turns``, so
        that ``begin(**chat.conversation())`` begins again as it was, a call
        still owed a result owed again and, on the same model, each turn that
        called tools as the tokens it was generated as.

        ``messages`` is every message, the system message first where there
        is one, in the Chat Completions form :meth:`begin` takes: a reply's text
        ``None`` where it makes calls and says nothing, ``""`` where it does
        neither, its thought under ``"reasoning_content"`` where the template
        keeps it, and each call's arguments as compact JSON in their own
        order; each result as the text the model read, cut with its note
        where it was cut. A call the model wrote unreadably is left out with
        its result, as :meth:`begin` leaves one out. Whatever :meth:`begin`
        takes without leaving anything out, this gives back unchanged, apart
        from the spelling of arguments and an empty text written as above.

        ``generated_turns`` is ``None`` where no turn is kept as generated;
        it is a JSON-ready dictionary, only meaningful to this model, holding
        each such turn's tokens and its messages as the conversation holds
        them, a call written unreadably and a thought included, and the
        messages alone begin any model again. ``tools`` is every declaration,
        and ``None`` where the conversation was begun without tools. Safe from any
        thread, including while another is mid-generation: all of it is read
        as the newest finished change left it.
        """
        return self._conversation.conversation()

    def generated_turns(self) -> dict | None:
        """What :meth:`conversation` gives as ``generated_turns``, without copying the messages; ``None`` where no turn is kept as generated.

        Each turn is found again by its messages wherever they stand, so a
        reading taken now still serves messages a caller adds to or drops from
        later. Safe from any thread, as :meth:`conversation` is.
        """
        return self._conversation.generated_turns()

    def capabilities(self, *, tools: bool) -> Capabilities:
        """What :meth:`begin` accepts for a conversation begun with ``tools`` given, even empty, or without.

        A model may ship a template of its own for a conversation that
        declares tools; ``tools=True`` answers for that one and
        ``tools=False`` for the default, each settled when the model loaded
        and the same before and after any :meth:`begin`. Safe from any thread.
        """
        return self._templates.capabilities(tools)

    @property
    def turn_open(self) -> bool:
        """Whether a reply is unfinished, so no new message may be added.

        True from a call until :meth:`reply` closes the turn, the results in
        between included. While it is, :meth:`reply` with a request,
        and :meth:`inject` with an assistant message refuse. Safe from any thread,
        as :meth:`conversation` is: as the newest finished change left it.
        """
        return self._conversation.published_turn_open()

    @property
    def awaits_reply(self) -> bool:
        """Whether :meth:`reply` without text would answer something: the newest message is a question, or a turn's results are all in.

        Safe from any thread, as :meth:`conversation` is, and copies nothing.
        """
        return self._conversation.published_awaits_reply()

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
        lock must be held for the whole exchange, tools running included; a
        tool that calls back into the wrapper is refused.

        Args:
            policy: The registry to run calls against, or ``"refuse"``, which
                answers every call with an error result.
            budget: The calls the turn may still run. Nothing is run under
                ``"refuse"``, so it does not bind there.
            on_started: As :meth:`_generate`, for the first generation.
            text, max_reply_tokens, **kwargs: As :meth:`reply`.

        Raises:
            UnsupportedError: If ``policy`` is a registry holding tools this
                model could never call.
        """
        registry = policy if isinstance(policy, ToolRegistry) else None
        if registry is not None and len(registry) > 0:
            self._templates.ensure_accepts(self._conversation.tools_declared,
                                           tools_field="tool_call_policy")
        # A turn going on has called; a question closes it first.
        any_call_asked = text is None and self._conversation.turn_open
        open_turn_text = self._open_turn_text() if text is None else ""
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

        # A call no result could answer is not read, so it cannot leave its turn open for good.
        deltas = self._generate(text, recognizes_calls=self._template.properties.can_write_results,
                                may_call_again=may_call_after_unreadable(),
                                max_reply_tokens=reply_cap, on_started=on_started, **kwargs)
        try:
            while True:
                generation = yield from self._read_generation(
                    deltas, after=open_turn_text + "".join(g.text for g in generations))
                generations.append(generation)
                if not _answerable(generation.tool_calls):
                    # A request cancelled before it started leaves an open turn open.
                    if self._conversation.turn_open and generation.stop_reason != "interrupted":
                        # An unreadable call, already answered: the turn goes on, spending
                        # budget. See design_history.md, "An unreadable call spends the budget".
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
                    dropped = self._evicted_by(self._conversation.close_unanswered)
                    generations[-1] = replace(
                        generations[-1], stop_reason="token_limit",
                        tool_calls=[call if call.status != "complete"
                                    else replace(call, status="unfinished")
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
                    dropped = self._evicted_by(self._conversation.close_unanswered)
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
            if (_answerable(generation.tool_calls) or not self._conversation.turn_open
                    or generation.stop_reason == "interrupted"):
                return generation if len(generations) == 1 else _sum_generations(generations, False)
            text = None
            kwargs.pop("on_started", None)

    def _read_generation(self, deltas: Generator[str, None, Reply], after: str):
        """Yield one generation's text, and return the :class:`Reply` describing it.

        Where ``after``, the text shown before it, and this text would join
        without whitespace, a blank line goes between them, and the Reply's
        text starts with it.
        """
        separator = None
        try:
            while True:
                try:
                    delta = next(deltas)
                except StopIteration as ended:
                    return replace(ended.value, text=(separator or "") + ended.value.text)
                if separator is None and delta:
                    separator = _parting(after, delta)
                    delta = separator + delta
                yield delta
        finally:
            # Closed here, a failure while the generation records itself reaches the caller.
            deltas.close()

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

        The calls are settled in the order asked, before any result is written.
        A call handed over counts against ``budget`` from then on. Where the
        stream is closed at a call handed over, or anything is raised while it
        runs, a ``KeyboardInterrupt`` included, the results of the calls that ran
        before it are written, and it and the rest of the round stay owed.

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
                    result = registry.call(call)
                except BaseException:
                    # Closed at the call, or stopped while it ran (Ctrl+C): the calls that ran
                    # keep their results. Where no room is left, they stay owed, as the
                    # caller's own result would find no room either.
                    with contextlib.suppress(ContextOverflowError):
                        self._write_results(settled, budget.calls_left > 0)
                    raise
            else:
                result = {"error": "tool_limit_reached",
                          "message": "no tool calls left in this turn; "
                                     f"the limit is {budget.max_calls}"}
            settled.append((call, result))
        return settled

    def _write_results(self, settled: list[tuple[ToolCall, object]], may_call_again: bool) -> int:
        """Write the result of each settled call, in order, a failure saying whether another call is worth making; how many history messages were evicted for them."""
        def write() -> None:
            for call, result in settled:
                self._conversation.add_result(call.id, _explain_failure(result, may_call_again))
        return self._evicted_by(write)

    def _evicted_by(self, action: Callable[[], None]) -> int:
        """Run ``action``; how many history messages it evicted."""
        before = self._conversation.changes_so_far().n_evicted
        action()
        return self._conversation.changes_so_far().n_evicted - before

    def _generate(self, text: str | None, *, recognizes_calls: bool, may_call_again: bool,
                  max_reply_tokens: int | None, stop: tuple[str, ...],
                  cancel: threading.Event | None, sampling: Sampling,
                  on_reply: Callable[[Reply], None], withhold: bool = False,
                  on_started: Callable[[], None] | None = None) -> Generator[str, None, Reply]:
        """One generation, as :meth:`~llama_chat.rendered_cache.RenderedCache.generation` runs it; the lock must be held.

        Args:
            text, stop, cancel: As :meth:`reply`.
            max_reply_tokens: The cap on this generation's tokens, ``None``
                for the config's.
            may_call_again: Whether a call is read again after a call this
                generation writes that cannot be read, which the answer to
                that call tells the model.
            on_reply: Given the generation's Reply once it ends, early or
                not, for a stream closed before it returns.
        """
        self._ensure_open()
        generation = self._conversation.generation(
            text, recognizes_calls=recognizes_calls, withhold=withhold,
            max_generated_tokens=self._reply_cap(max_reply_tokens), stop=stop, cancel=cancel,
            sampling=sampling, unreadable_result=_explain_failure(_UNREADABLE, may_call_again),
            on_started=on_started, on_reply=on_reply)
        return (yield from generation)

    def _reply_cap(self, max_reply_tokens: int | None) -> int:
        """The cap on a reply's tokens: ``max_reply_tokens``, else the config's, else the context's size."""
        return next((c for c in (max_reply_tokens, self._config.max_reply_tokens) if c is not None),
                    self._config.context_size)

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


def _stop_texts(stop) -> tuple[str, ...]:
    """A request's stop texts, read once; a single ``str`` is one stop text.

    Raises:
        TypeError: If ``stop`` is neither a ``str`` nor an iterable of ``str``.
    """
    texts = (stop,) if isinstance(stop, str) else tuple(stop) if isinstance(stop, Iterable) else stop
    if texts is not None and (not isinstance(texts, tuple)
                              or not all(isinstance(text, str) for text in texts)):
        raise TypeError(f"stop must be a str or an iterable of str, not {stop!r}")
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
        raise InvalidArgumentError("max_calls", f"max_calls limits calls that are run, and "
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
        evicted_for=next((t.evicted_for for t in reversed(generations) if t.evicted_for), None),
        taken_back=generations[-1].taken_back,
        kept_text=generations[-1].kept_text,
    )


# How many calls that could not be read one reply answers before it reads no more calls.
MAX_UNREADABLE_CALLS_IN_A_ROW = 3
# What a call that could not be read is answered with, before what to do next.
_UNREADABLE = {"error": "unreadable_call", "message": "the call could not be read"}


def _explain_failure(result, may_call_again: bool):
    """Say that the call failed and whether to call again, leaving a real result alone.

    A dict carrying ``error`` is a failure, whether the registry built it or a
    tool returned one itself; its ``message`` becomes the reason.
    """
    if not isinstance(result, dict) or "error" not in result:
        return result
    reason = result.get("message") or result["error"]
    return {**result,
            "message": f"the tool call failed: {reason}. "
                       + ("try again" if may_call_again else "do not call again")}


def _parting(before: str, text: str) -> str:
    """What a stream shows between ``before``, the text shown before a generation in its turn, and ``text``, that generation's: a blank line where they would join without whitespace."""
    return "\n\n" if before and text and not before[-1].isspace() and not text[0].isspace() else ""


def _shown_before(messages: list[dict]) -> str:
    """The text of the replies of the newest turn before its newest message, which that message was streamed after."""
    asked = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    return "".join(m["content"] or "" for m in messages[asked + 1:-1] if m["role"] == "assistant")


def _calls_tools(item) -> bool:
    """Whether the history item ``item`` holds a call or a call's result."""
    return isinstance(item, dict) and (item.get("role") == "tool" or bool(item.get("tool_calls")))
