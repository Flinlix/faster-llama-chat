"""ChatWrapper: the public API over one conversation, its cache holding what the model's template writes.

The conversation is kept as the messages a chat template is given, and the
cache as the template's render of them, decoded only where it differs from
what the cache holds. See :mod:`llama_cpp_conversation.rendered_cache` and
docs/decisions/0001-the-cache-holds-the-templates-render.md.
"""

from __future__ import annotations

import contextlib
import logging
import inspect
import threading
import time
import weakref
from typing import Any
from collections.abc import Callable, Generator, Iterable

from .addon_files import addon_file, with_model_sampling
from .cache_context import CACHE_CONTEXT, ensure_answers, ensure_can_go_back
from .config import Config, Sampling, ensure_token_cap, generation_sampling, model_file
from .conversation_state import reasoning_switch
from .messages import ensure_role, ensure_text
from .history_forms import ensure_generated_turns, read_messages
from .rendered_cache import RenderedCache
from .errors import (ArgumentTypeError, ContextOverflowError, InvalidArgumentError,
                     InvalidSettingError, OutOfOrderError, UnsupportedError,
                     raising_library_errors_only)
from .reply import TOOL_CALL_POLICIES, Reply, ReplyStream, Report, ToolCallPolicy
from .resolved_template import Capabilities, ResolvedTemplates
from .tool_registry import ToolRegistry
from .tools import ToolCall, ToolSpec, ensure_declarable, ensure_no_tags, unwritable_failure
from .loop_variant import LoopVariant
from .reply_rounds import UNREADABLE, CallBudget, ReplyRounds, answerable, explain_failure, parting

_log = logging.getLogger(__name__)


@raising_library_errors_only
class ChatWrapper:
    """Stateful single-conversation chat over a cache context, with KV reuse.

    Once built, every public action raises only the library's own errors
    (:data:`~llama_cpp_conversation.errors.LIBRARY_ERRORS`), each naming the argument and
    the cause; a refused action changes nothing, and a failed reply keeps only
    what it had produced.
    """

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
            config: The configuration; :class:`~llama_cpp_conversation.config.Config`'s
                defaults where it is not given.
            context (CacheContext | None): A cache context of your own, in place of the one built
                from ``config``. :meth:`close` closes it too; only a
                construction that fails part-way leaves it alone.
            on_load_progress: Invoked during model loading with a fraction in
                ``[0.0, 1.0]``. Must not raise. Ignored when ``context`` is
                supplied.

        Raises:
            ArgumentTypeError: If ``context`` does not answer everything the wrapper
                calls on it.
            ImportError: Where the wrapper loads the model itself and the
                engine's bindings are not installed.
            InvalidSettingError: If ``model_path`` is unset or names no
                file, where the wrapper loads the model itself, with the
                engine's bindings installed or not.
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
        if owns_context:
            # The engine is loaded with the first model file, so that a context of one's own needs none.
            model_file(config)
            from .context import KVContext
            self._context = KVContext(config, on_load_progress=on_load_progress)
        else:
            self._context = context
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
            self._templates = ResolvedTemplates(self._context, addon["call"], addon["reasoning_on"],
                                                addon["reasoning_levels"])
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
        self._replies: weakref.WeakSet[Generator[Any, Any, Any]] = weakref.WeakSet()
        # Held while that set is added to or copied, which any thread may do.
        self._replies_lock = threading.Lock()
        # Held while the model is read outside the wrapper's lock, and while it is freed.
        self._model_lock = threading.Lock()
        # The actions that wait for the wrapper, which the work after a reply gives way to.
        self._waiting = _Waiting()
        # Every wait for the wrapper, a wait for it to go idle included, which an unread reply
        # is ended for.
        self._blocked = _Waiting()
        # How many begins have run: a reply belongs to the conversation it was asked of.
        self._begins = 0
        # The reply that holds the wrapper while it waits for its reader to take the next step.
        self._reply_waiting_for_reader: Callable[[], ReplyStream | None] | None = None
        # The timer that ends a reply nobody reads, while an action waits for it.
        self._unread_watch: threading.Timer | None = None
        self._unread_watch_lock = threading.Lock()
        # The threads tools run on, which may not act on the wrapper the reply that runs them holds.
        self._tool_threads: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
        # Those of them past ``tool_timeout_s``, whose calls were answered as failed meanwhile.
        self._late_tool_threads: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
        self._rounds = ReplyRounds(self._conversation, self._templates, self._config,
                                   self._ensure_open, self._tool_threads, self._late_tool_threads)

    def begin(self, messages: list[dict] | None = None,
              tools: Iterable[ToolSpec] | None = None, *,
              reasoning: bool = False, reasoning_level: str | None = None,
              generated_turns: dict | None = None) -> Report:
        """Begin the conversation, or begin it again: prefill the system prompt and recent history, decoding only where they differ from what the cache holds.

        Every other action refuses until this has run, because only this
        prefills the sequence opener and the system prompt. The previous
        conversation is left intact on every refusal here.

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

                Two questions in a row, no reply between them, are joined
                into one where the template writes no two in a row, as
                :meth:`inject` joins them. Every call needs an id of its own.
                A call whose arguments are
                no JSON object, or that names no function, could not be read:
                it is left out with its result, and a later result for it is
                refused. A history that
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
            tools: Tools to declare, a list of :class:`~llama_cpp_conversation.tools.ToolSpec`
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
            reasoning_level: With ``reasoning``, the effort level the model
                reasons at, one of the ``reasoning_levels``
                :meth:`capabilities` lists; ``None`` for the template's own.
            generated_turns: What :meth:`conversation` gave beside the
                messages: each turn the model generated with a call, as its
                tokens and its messages as the conversation held them. On
                the model it came from, each such turn the messages still
                hold, found by its messages and the question before them, is
                put back as it was and restored as generated, token for
                token; on another model, or without it, every turn is written
                as this model's template writes it. Trusted input: its tokens
                are restored as they stand, tags included, so give back only
                what :meth:`conversation` gave. See docs/decisions/0029-two-history-forms.md.

        Returns:
            What it changed beyond what was asked: how many of the messages
            after the system message it kept and why it left the others out,
            the calls it left out, the results it cut, and the generated
            turns it restored. Each such change is also logged once, at
            ``INFO``, under the logger ``llama_cpp_conversation``.

        Raises:
            ContextOverflowError: If the system prompt and its declarations
                leave no room under the eviction threshold
                (``evict_above_fraction`` of ``context_size``) for a short question,
                or the newest exchange of ``messages``, from its user message
                on, leaves no room for a reply beside them even with its
                results cut to nothing (one that ends on a reply or a call
                may fill the context, as it did when it was generated), or ``messages`` hold no user message and cannot fit
                beside them either.
            ArgumentTypeError: If ``messages`` is no list, a message or a call is no
                dictionary, a text, a function's name or a call's arguments is
                of another type than stated above, or ``tools`` holds anything
                but tool specs.
            UnsupportedError: If ``tools`` names anything, or the history
                holds a call or a result, and the template this conversation is
                written with cannot call tools; if ``reasoning`` is asked and
                that template cannot reason, or ``reasoning_level`` names a
                level it does not know; or if a system message is given
                and the template moves a system prompt with later questions. Its ``field`` names the parameter:
                ``tools``, ``messages``, ``reasoning`` or ``reasoning_level``.
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
            OutOfOrderError: If the wrapper is closed, or a tool the reply
                that holds the wrapper runs asks for it.
        """
        ensure_generated_turns(generated_turns)
        _ensure_bool(reasoning, "reasoning")
        if reasoning_level is not None and not isinstance(reasoning_level, str):
            raise ArgumentTypeError(f"reasoning_level must be None or one of the reasoning_levels "
                                    f"capabilities() lists, not {reasoning_level!r}")
        if reasoning_level is not None and not reasoning:
            raise InvalidArgumentError("reasoning_level", "a reasoning level is given only with "
                                       "reasoning=True; without it the model does not reason")
        switch = reasoning_switch(reasoning, reasoning_level)
        if messages is not None and (isinstance(messages, (str, bytes, dict))
                                     or not isinstance(messages, Iterable)):
            raise ArgumentTypeError(f"messages must be a list of chat message dictionaries, the system "
                            f"message first where there is one, not {messages!r}")
        if tools is not None and not isinstance(tools, Iterable):
            raise ArgumentTypeError(f"tools must be a list of ToolSpec, such as ToolRegistry.specs() "
                            f"gives, not {tools!r}")
        messages = list(messages or [])         # read once, a generator included
        # Checked before the wrapper is taken, so that a begin refused for its own arguments
        # ends no reply its thread reads, and destroys no conversation it was to replace.
        specs = None if tools is None else list(tools)
        ensure_declarable(specs or [])
        with self._model_lock:              # the model is read: close() waits, and none is closed
            self._ensure_open()
            ensure_no_tags(specs or [], self._holds_tag)
            if specs:
                self._templates.for_conversation(True).ensure_names_read_back(
                    [spec.name for spec in specs], "tools")
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
            reasoning=switch, system_prompt_field="messages" if opens_on_system else None)
        read_messages(messages)
        with self._held():
            self._ensure_open()
            self._conversation.forget_told()    # changes of earlier actions are told already
            # A begin leaves no turn a reply started, so a call it could not read is answered
            # as one in a turn that runs no call.
            self._conversation.start(messages, specs, switch, generated_turns,
                                     explain_failure(UNREADABLE, may_call_again=False))
            # A turn the history leaves open was started by no reply.
            self._conversation.turn_policy = None
            self._conversation.turn_budget = CallBudget(self._config.max_calls)
            self._rounds.answers_waiting = []
            self._begins += 1
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
            ArgumentTypeError: If ``text`` is not a ``str``.
            InvalidArgumentError: If ``role`` is another one, or ``text``
                holds a lone surrogate, which no model can read.
            OutOfOrderError: If no conversation has been begun, an
                assistant message is given while a tool call has left a reply
                open, the wrapper is closed, or a tool the reply that holds the
                wrapper runs asks for it.
            TemplateRefusedError: If the template does not write the message
                here as its text, a question asked twice or an empty one on
                some templates included, naming ``text``.
            ContextOverflowError: If an assistant message cannot fit even
                after evicting every non-system message, raised before
                anything is evicted (a question is cut to fit as
                :meth:`reply` cuts one); or if the conversation with it cannot fit once every
                exchange that can go is evicted, since no user message follows
                the rest, which is also raised before anything is evicted.
        """
        ensure_role(role)
        ensure_text(text, "an injected message's text")
        with self._held():
            self._ensure_open()
            self._conversation.forget_told()
            # A question closes an open turn: the answers the engine failed at are written first.
            answers = ([(call.id, answer) for call, answer in self._rounds.answers_waiting]
                       if role == "user" else [])
            self._conversation.add(role, text, answers)
            if role == "user":
                self._rounds.answers_waiting = []
            return self._conversation.report()

    # ----- asking ---------------------------------------------------------------
    def reply(self, text: str | None = None, *,
              tool_call_policy: ToolCallPolicy | ToolRegistry | None = None,
              max_calls: int | None = None,
              max_reply_tokens: int | None = None,
              max_thought_tokens: int | None = None,
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
                See docs/guide/without-a-template.md, "Special tokens in message content". ``None``
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
                  entirely; see docs/decisions/0039-tool-call-policies.md. Where a call opens with plain text
                  rather than a tag, that text ends the reply, and the call is
                  reported as ``"unfinished"``.
                * ``"refuse"``: nothing is run, not even a tool that was
                  declared. The call is kept out of the text, the model is
                  told no tool can be run, and the reply goes on without
                  drawing the token that opens a call, as under
                  ``"withhold"``, a blank line parting the text before the
                  call from the text after it where they would join without
                  whitespace.
                * ``"yield"``: each call is handed over as a
                  :class:`~llama_cpp_conversation.tools.ToolCall` after the text, and the
                  turn stays open. Answer each with :meth:`inject_tool_result`,
                  in order, then call this again without ``text``; that reply
                  is parted from the text before the call as under
                  ``"refuse"``. A call that cannot be read is answered with a
                  failure and the reply goes on; the model may write another
                  call as long as there is room, and after three such calls
                  in a row it is told not to call again.
                * A :class:`~llama_cpp_conversation.tool_registry.ToolRegistry`: each call is run
                  against it, its result written, and the reply goes on until
                  the model answers. Each call that runs is handed over in the
                  stream just before it runs. In the last permitted round the
                  token that opens a call is never drawn, as under
                  ``"withhold"``, so the turn always closes.
            max_calls: Where a registry is given, how many calls one turn may
                use, each call handed over counted. ``None`` takes the
                config's ``max_calls``, or, going on with a turn, keeps what
                is left of the turn's; a call
                past one is answered with an error result rather than run. Only
                with a registry.
            max_reply_tokens: This request's cap on the tokens of the reply,
                every generation of it together, a thought included, in place
                of the config's.
            max_thought_tokens: This request's cap on the tokens of each
                thought, in place of the config's.
            stop: Texts that end the answer, matched against the answer
                only; a single ``str`` is one such text, any other iterable
                holds them. Read when this is called, so a later change to
                it changes nothing.
            cancel: Setting it, from any thread, stops generation within one
                token and ends the reply with ``stop_reason == "interrupted"``.
                Set while registered tools run, it runs no later call of the
                round, those not run staying owed; set before the reply's first
                word, it also stops the wait for a reply another thread holds
                and the decode the reply before left, within 512 tokens, or
                ``micro_batch_tokens`` where that is more.
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
        counted as evicted. A question too long for the room the system prompt
        leaves under the eviction threshold keeps as much of its end as fits,
        possibly none of it, :data:`~llama_cpp_conversation.messages.QUESTION_CUT_MARK`
        in place of its beginning, and a warning is logged. It is refused only
        where the template does not write that question as its text, such as
        an empty one on some templates, which leaves the next question free.
        The reply holds at least ``min_answer_tokens`` of answer, and a
        thought is closed in time to leave the answer ``answer_room_tokens``.
        See docs/decisions/0032-every-question-gets-a-reply.md.

        Raises:
            OutOfOrderError: At once, if no conversation has been begun or
                the wrapper is closed, or, without ``text``, if a result is
                still owed or there is nothing to answer; when it is read, if a
                begin has replaced the conversation it was asked of since, or a
                tool the reply that holds the wrapper runs asked for it.
            InvalidArgumentError: At once, if ``tool_call_policy`` is a text
                that is none of :data:`~llama_cpp_conversation.reply.TOOL_CALL_POLICIES`,
                or ``max_calls`` is given without a registry; at the first
                iteration, if ``max_calls`` is given and the turn the reply
                goes on with by then runs no call.
            UnsupportedError: At the first iteration, if a registry holding
                tools is given to a model that could never call one, naming
                ``tool_call_policy``.
            TemplateRefusedError: At the first iteration, if the template does
                not write the question as its text, such as one asked twice or
                an empty one on some templates, naming ``text``.
            ArgumentTypeError: At once, if ``text`` is neither ``None`` nor a ``str``,
                ``tool_call_policy`` is neither a text nor a registry,
                ``stop`` is neither a ``str`` nor an iterable of ``str``,
                ``cancel`` has no ``is_set``, or ``sampling`` is neither
                ``None`` nor a :class:`~llama_cpp_conversation.config.Sampling`.
            InvalidSettingError: At once, if ``max_reply_tokens`` or
                ``max_thought_tokens`` is no whole number of at least 1, or
                ``max_calls`` no whole number of at least 0, as their config
                fields are refused.
            ContextOverflowError: Only where too little room remains for the
                reply to go on with an open turn; a question is cut to fit
                instead, one injected earlier cut again where the system
                prompt has grown since. A result or the error answering a call that no longer fits
                raises nothing: the turn closes on what the model wrote, and
                the reply ends with ``"token_limit"``.
        """
        if text is not None:
            ensure_text(text, "the request's text")
        ensure_token_cap(max_reply_tokens, "max_reply_tokens")
        ensure_token_cap(max_thought_tokens, "max_thought_tokens")
        stop_texts = _stop_texts(stop)
        _ensure_cancel_and_sampling(cancel, sampling)
        # This stream's newest generation's Reply, which the stream reads where it is closed early.
        newest: dict[str, Reply] = {}
        options = dict(max_reply_tokens=max_reply_tokens, stop=stop_texts, cancel=cancel,
                       max_thought_tokens=(self._config.max_thought_tokens
                                           if max_thought_tokens is None else max_thought_tokens),
                       sampling=generation_sampling(self._config, sampling),
                       on_reply=lambda reply: newest.update(reply=reply))
        if tool_call_policy is not None and not isinstance(tool_call_policy, (str, ToolRegistry)):
            raise ArgumentTypeError(f"tool_call_policy must be one of "
                            f"{', '.join(map(repr, TOOL_CALL_POLICIES))} or a ToolRegistry, "
                            f"not {tool_call_policy!r}")
        if isinstance(tool_call_policy, str) and tool_call_policy not in TOOL_CALL_POLICIES:
            raise InvalidArgumentError(
                "tool_call_policy", f"tool_call_policy must be one of "
                f"{', '.join(map(repr, TOOL_CALL_POLICIES))}, not {tool_call_policy!r}")
        # A request's own setting, refused as its config field is.
        if max_calls is not None and (not isinstance(max_calls, int) or isinstance(max_calls, bool)
                                      or max_calls < 0):
            raise InvalidSettingError("max_calls", f"max_calls must be None or a whole number of "
                                                   f"at least 0, not {max_calls!r}")
        # Read from what the newest change published, so it never waits for the lock; the
        # policy is chosen again under the lock, from the turn open when the reply runs.
        published_turn = self._conversation.published_turn_policy()
        _ensure_max_calls_runs(_policy_of(
            tool_call_policy, published_turn if text is None and self._conversation.published_turn_open()
            else None), max_calls)
        self._ensure_open()
        if not self._rounds.answers_waiting:
            # Answers the engine failed at are written first, under the lock.
            self._conversation.ensure_can_reply(asks_question=text is not None)
        # Weak, so that an abandoned stream is collected, and its hold on the wrapper ends.
        stream_ref: list[Callable[[], ReplyStream | None]] = []
        steps = self._noting_each_reader(
            self._replying(text, tool_call_policy, max_calls, begun=self._begins, **options),
            stream_ref)
        with self._replies_lock:
            self._replies.add(steps)
        stream = ReplyStream(steps, on_close=lambda: newest.get("reply"),
                             runs_a_step=lambda: threading.current_thread() in self._tool_threads)
        stream_ref.append(weakref.ref(stream))
        return stream

    def _noting_each_reader(self, steps: Generator[str | ToolCall, None, Reply],
                            stream_ref: list[Callable[[], ReplyStream | None]]
                            ) -> Generator[str | ToolCall, None, Reply]:
        """``steps``, the thread that takes each step noted as the one holding the wrapper, while the reply holds it; the stream ``stream_ref`` names noted as waiting for its reader while it does."""
        holds = False
        try:
            while True:
                # Held first, so that the thread that read before never finds itself holding
                # with no reply waiting for it.
                if holds:
                    self._holder = threading.current_thread()
                if self._reply_waiting_for_reader is stream_ref[0]:
                    self._reply_waiting_for_reader = None
                try:
                    step = next(steps)
                except StopIteration as ended:
                    self._start_settling()
                    return ended.value
                holds = self._holder is threading.current_thread()
                if holds:
                    self._reply_waiting_for_reader = stream_ref[0]
                yield step
        except GeneratorExit:
            # Closed at a step: ``steps`` still waits there, and is closed on this thread.
            if holds:
                self._holder = threading.current_thread()
            if self._reply_waiting_for_reader is stream_ref[0]:
                self._reply_waiting_for_reader = None
            try:
                steps.close()
            finally:
                self._start_settling()
            raise

    def _start_settling(self) -> None:
        """Have the work a reply left done on a thread of its own, once the reply has ended, so that its stream ends at once; the next action does it first where that thread has not yet."""
        if self._conversation.settle_due:
            # Not a daemon: a process that ends right after a reply waits for it, rather than
            # tearing the model down under its decode.
            settling = threading.Thread(target=self._settle_when_free, name="llama_cpp_conversation after a reply")
            try:
                settling.start()
            except RuntimeError:
                pass                # none can start, as at shutdown: the next action does the work

    def _settle_when_free(self) -> None:
        """Do the work a reply left, holding the wrapper, leaving a long decode to an action that waits for it; where the wrapper stays held longer than :data:`_AFTER_REPLY_WAIT_S`, the next action does that work."""
        time.sleep(_AFTER_REPLY_PAUSE_S)    # the reader first takes in the stream's end
        # An action that took the wrapper meanwhile did the work first; it is not waited for.
        if not self._conversation.settle_due or not self._lock.acquire(timeout=_AFTER_REPLY_WAIT_S):
            return
        self._holder = threading.current_thread()
        try:
            if not self._closed and self._conversation.settle_due:
                self._conversation.settle(self._waiting)
        except Exception:
            _log.warning("the work after a reply failed; the next action decodes the cache "
                         "again", exc_info=True)
        finally:
            self._holder = None
            self._lock.release()

    def _replying(self, text: str | None, given: ToolCallPolicy | ToolRegistry | None,
                  max_calls: int | None, begun: int, **options
                  ) -> Generator[str | ToolCall, None, Reply]:
        """The reply :meth:`reply` describes, its policy chosen under the lock: ``given``, else the open turn's; refused where a begin has replaced the conversation it was asked of, ``begun`` begins in."""
        with self._held(cancel=options["cancel"], question=text,
                        reply_tokens=self._rounds.reply_cap(options["max_reply_tokens"])) as held:
            if not held:
                # Cancelled while it waited for another action: nothing changed.
                return Reply(text="", n_prefilled_tokens=0, n_prompt_tokens=self.total_tokens,
                             n_generated_tokens=0, n_evicted_messages=0, stop_reason="interrupted")
            if self._begins != begun:
                raise OutOfOrderError(
                    None, "this reply was asked of a conversation that begin() has replaced "
                    "since; ask it again of the conversation begun now")
            self._conversation.forget_told()    # changes of earlier actions are told already
            goes_on = text is None and self._conversation.turn_open
            policy = self._policy_for(given, goes_on)
            _ensure_max_calls_runs(policy, max_calls)
            budget = (self._conversation.turn_budget if goes_on and max_calls is None
                      else CallBudget(self._config.max_calls if max_calls is None else max_calls))
            options["on_started"] = lambda: self._keep_turn(given, policy, budget)
            if policy not in ("yield", "withhold"):
                return (yield from self._rounds.until_answered(
                    text, policy=policy, budget=budget, **options))
            if policy == "yield":
                generation = yield from self._rounds.until_called(text, **options)
            else:
                generation = yield from self._rounds.read_generation(
                    self._rounds.generate(text, recognizes_calls=False, may_call_again=False,
                                   withhold=True, **options),
                    after=self._rounds.open_turn_text() if text is None else "")
        # The lock is released before the calls: a caller answers each while reading the stream.
        yield from answerable(generation.tool_calls)
        return generation

    def _policy_for(self, given: ToolCallPolicy | ToolRegistry | None,
                    goes_on: bool) -> ToolCallPolicy | ToolRegistry:
        """The policy a reply runs with: ``given``, else the turn's where it goes on with one, else ``"refuse"``; the lock must be held."""
        return _policy_of(given, self._conversation.turn_policy if goes_on else None)

    def _keep_turn(self, given: ToolCallPolicy | ToolRegistry | None,
                   policy: ToolCallPolicy | ToolRegistry, budget: CallBudget) -> None:
        """Note what a reply whose generation has started leaves its turn; the lock must be held.

        A reply that opens a turn starts it with ``policy``; one going on with
        a turn the history opened gives it ``given``, where it gives one. The
        turn keeps ``budget``, the calls it may still run.
        """
        conversation = self._conversation
        if not conversation.turn_open or (conversation.turn_policy is None and given is not None):
            self._conversation.turn_policy = policy
        self._conversation.turn_budget = budget

    def inject_tool_result(self, call_id: str, result: object) -> Report:
        """Hand the model what the tool it called returned; what that changed beyond what was asked.

        Written where this model's own template writes a result, inside the
        turn the call left open, with special-token parsing off. A result too
        long for what is left of the context evicts the oldest exchanges
        first, the open turn kept whole; one still too long is cut, a note at
        its end telling the model how much was kept, and the next reply lists
        its call in ``Reply.tool_results_cut``. Where not even a result cut to
        nothing fits, the turn closes on what the model wrote, as a reply that
        answers its own calls closes one, and ``turn_open`` is then false. See
        docs/guide/tool-calling.md.

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
            ArgumentTypeError: If ``call_id`` is no ``str``.
            OutOfOrderError: If no conversation has been begun, the wrapper
                is closed, or a tool the reply that holds the wrapper runs asks
                for it; or, its ``field`` then ``call_id``, if no call is
                waiting, ``call_id`` is not the one owed, or it names a call
                :meth:`begin` left out as one that could not be read. The call
                stays owed.
            UnsupportedError: If this model's template writes no result,
                naming ``result``.
        """
        if not isinstance(call_id, str):
            raise ArgumentTypeError(f"call_id must be the id of the call being answered, a str, "
                            f"not {call_id!r}")
        failure = unwritable_failure(result)
        with self._held():
            self._ensure_open()
            # A failure is told as a failure a registered tool's is, whoever ran the tool.
            policy = self._conversation.turn_policy
            may_call_again = policy == "yield" or (isinstance(policy, ToolRegistry)
                                                   and self._conversation.turn_budget.calls_left > 0)
            written = explain_failure(result if failure is None else failure, may_call_again)
            self._conversation.forget_told()
            # The answers kept for older calls first, the caller's own in place of one kept for
            # its call, all of them or, where one is refused, none.
            owed = [call.id for call in self._conversation.pending_calls]
            kept = {call.id: answer for call, answer in self._rounds.answers_waiting}
            before = owed[:owed.index(call_id)] if call_id in owed else []
            results = [(owed_id, kept[owed_id]) for owed_id in before if owed_id in kept]
            try:
                self._conversation.add_results(results + [(call_id, written)])
            except ContextOverflowError:
                self._conversation.close_unanswered()
            self._rounds.answers_waiting = [(call, answer) for call, answer in self._rounds.answers_waiting
                                     if call.id not in {call_id, *before}]
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
                may open with a blank line, or the whole turn's where one
                stream showed the text before its call too. A cut inside the
                text before the turn's call cuts the newest reply to nothing;
                the text a call followed stays. A cut that reads as either is
                read as one of the newest reply. The whole text changes
                nothing.

        Raises:
            ArgumentTypeError: If ``keep_text`` is not a ``str``.
            OutOfOrderError: If no conversation has been begun, a reply is
                open, the newest message is not the assistant's or holds a
                tool call, one that could not be read included, the wrapper is
                closed, or a tool the reply that holds the wrapper runs asks
                for it.
            InvalidArgumentError: If ``keep_text`` is not a prefix of its text.
            TemplateRefusedError: If the template does not write the reply
                shortened to it, as one that writes no empty reply does not
                write it shortened to ``""``; or, for the reply that ends a
                turn that called tools, if its tokens as generated hold more
                than its text, such as a call that went for want of room.
        """
        ensure_text(keep_text, "keep_text", "keep_text")
        with self._held():
            self._ensure_open()
            messages = self._conversation.messages
            if messages and messages[-1]["role"] == "assistant":
                keep_text = _kept_of_the_newest(_shown_before(messages), messages[-1]["content"],
                                                keep_text)
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
        instead. See docs/decisions/0022-eviction-drops-whole-oldest-exchanges.md.
        """
        return self._context.can_shift

    def wait_idle(self, timeout_s: float | None = None) -> bool:
        """Block until no generation holds the wrapper, nor the work after a reply; advisory, since one may start right after.

        Args:
            timeout_s: Seconds to wait, at least 0; ``None``, infinity or more
                than the platform's lock can wait waits indefinitely.

        Returns:
            ``True`` if the wrapper went idle, ``False`` on timeout.

        Raises:
            ArgumentTypeError: If ``timeout_s`` is neither ``None`` nor a number.
            InvalidArgumentError: If it is negative or not a number at all
                (``NaN``).
            OutOfOrderError: If a tool the reply that holds the wrapper runs
                asks for it.
        """
        if timeout_s is not None and (isinstance(timeout_s, bool)
                                      or not isinstance(timeout_s, (int, float))):
            raise ArgumentTypeError(f"timeout_s must be None or a number of seconds, not {timeout_s!r}")
        if timeout_s is not None and (timeout_s != timeout_s or timeout_s < 0):    # NaN is unequal
            raise InvalidArgumentError(
                "timeout_s", f"timeout_s must be a number of seconds of at least 0, not {timeout_s}")
        self._ensure_not_holding()
        # Longer than the lock can wait is waiting indefinitely, infinity included.
        waits = timeout_s is None or timeout_s > threading.TIMEOUT_MAX
        # The work after a reply goes on, as waiting for it to end asks.
        if not self._acquire(-1 if waits or timeout_s is None else timeout_s, gives_way=False):
            return False
        try:
            self._settle_due_work()
        finally:
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
        from the spelling of arguments, an empty text written as above, and two
        questions in a row joined where the template writes no two.

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
        _ensure_bool(tools, "tools")
        return self._templates.capabilities(tools)

    @property
    def turn_open(self) -> bool:
        """Whether a reply is unfinished: a call is owed a result, or the results wait for the reply after them.

        True from a call until :meth:`reply` closes the turn, or a question
        closes it, or a result finds no room. While it is, only an assistant
        message and a shortened reply are refused; a question closes the turn
        first. Safe from any thread, as :meth:`conversation` is: as the newest
        finished change left it.
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
        and from a tool that reply runs, is refused as any action is, changing
        nothing. A reply asked for and not yet read raises once it is read.
        Closes a context passed to the constructor as well. Every action after
        close raises :class:`~llama_cpp_conversation.errors.OutOfOrderError` once its
        arguments are checked;
        :meth:`conversation` and the properties that say where it stands still
        answer.
        """
        try:
            # Through its stream, as the next action ends it, so that the stream keeps its
            # summary; one another thread has begun to step is waited for below instead.
            # From a tool a reply runs, that reply is running, and close() is refused below
            # before it changes anything.
            with self._replies_lock:
                replies = list(self._replies)
            running = any(inspect.getgeneratorstate(s) == inspect.GEN_RUNNING for s in replies)
            if self._holder is threading.current_thread() and not running \
                    and self._end_reply_waiting_for_reader() is not False:
                # A reply begun is ended as closing its stream ends it; one not begun
                # raises, as after any close, once it is read.
                for steps in [s for s in replies
                              if inspect.getgeneratorstate(s) == inspect.GEN_SUSPENDED]:
                    steps.close()
        finally:
            with self._held(settles=False), self._model_lock:
                if not self._closed:
                    self._closed = True
                    self._context.close()

    def __enter__(self) -> ChatWrapper:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _holds_tag(self, text: str) -> bool:
        """Whether ``text`` holds one of the model's tags."""
        return any(self._context.is_special(token) for token in self._context.tokenize(text))

    def _ensure_open(self) -> None:
        if self._closed:
            raise OutOfOrderError(None, "this ChatWrapper is closed, and takes no action; "
                                        "build a new one")

    @contextlib.contextmanager
    def _held(self, settles: bool = True, cancel=None, question: str | None = None,
              reply_tokens: int = 0):
        """Hold the wrapper for one action, refusing one a tool the holding reply runs asks for; the work a reply left is done first, where ``settles``, or after the reply to ``question``, of at most ``reply_tokens``, where that waits less.

        Where ``cancel`` is set, the work a reply left stops, and so does the
        wait for the wrapper, which then yields ``False`` without holding it;
        it yields ``True`` once the wrapper is held.
        """
        self._ensure_not_holding()
        if not self._acquire(cancel=cancel):
            yield False
            return
        try:
            self._holder = threading.current_thread()
            if settles:
                self._settle_due_work(cancel, question, reply_tokens)
            yield True
        finally:
            self._holder = None
            self._lock.release()

    def _settle_due_work(self, cancel=None, question: str | None = None,
                         reply_tokens: int = 0) -> None:
        """Do the work a reply left, where the thread for it has not yet, stopped where ``cancel`` is set, as :meth:`RenderedCache.settle` does before ``question`` and its ``reply_tokens``; the lock must be held. A failure is logged: it left the conversation as it was, and the cache to be decoded again."""
        if self._closed or not self._conversation.settle_due:
            return
        try:
            self._conversation.settle(cancel, question, reply_tokens)
        except Exception:
            _log.warning("the work after a reply failed; the cache is decoded again",
                         exc_info=True)

    def _acquire(self, timeout_s: float = -1, gives_way: bool = True, cancel=None) -> bool:
        """Take the wrapper's lock, waiting at most ``timeout_s`` (``-1`` for no limit) and while ``cancel`` is unset; a reply that holds it unread for ``unread_reply_timeout_s`` is ended meanwhile, and the work after a reply gives way where ``gives_way``."""
        if self._lock.acquire(blocking=False):
            return True
        with self._blocked, self._waiting if gives_way else contextlib.nullcontext():
            self._watch_unread_reply()
            if cancel is None:
                return self._lock.acquire(timeout=timeout_s)
            # Waited for in slices, the time left until the longest wait the platform allows
            # its measure, so that a cancel is read between them.
            deadline_s = time.monotonic() + threading.TIMEOUT_MAX
            waits_variant = LoopVariant("waiting for the wrapper until a cancel")
            while not cancel.is_set():
                waits_variant.passes(int((deadline_s - time.monotonic()) * 1000))
                if self._lock.acquire(timeout=_CANCEL_LOOK_S):
                    return True
            return False

    def _watch_unread_reply(self, after_s: float | None = None) -> None:
        """Have a reply that holds the wrapper unread ended once ``unread_reply_timeout_s`` has passed, while an action waits for it."""
        limit = self._config.unread_reply_timeout_s
        if limit is None:
            return
        with self._unread_watch_lock:
            if self._unread_watch is not None:
                return
            self._unread_watch = threading.Timer(
                min(limit if after_s is None else after_s, threading.TIMEOUT_MAX),
                self._end_unread_reply)
            self._unread_watch.daemon = True
            self._unread_watch.start()

    def _end_unread_reply(self) -> None:
        """End the reply that holds the wrapper unread for ``unread_reply_timeout_s``, while an action waits; look again later where none has yet. A failure is logged, on the timer's thread no caller reads."""
        try:
            self._end_unread_reply_now()
        except Exception:
            _log.error("ending a reply nobody read failed; the action waiting for it waits on",
                       exc_info=True)

    def _end_unread_reply_now(self) -> None:
        """:meth:`_end_unread_reply`, its failures raised."""
        with self._unread_watch_lock:
            self._unread_watch = None
        if not self._blocked.is_set():
            return
        limit = self._config.unread_reply_timeout_s
        assert limit is not None, "only a limit arms the watch"
        named = self._reply_waiting_for_reader
        stream = named() if named is not None else None
        if stream is not None and stream._end_if_unread(limit):
            _log.warning("a reply that nobody read for %s s was ended, so that the action "
                         "waiting for the wrapper goes on", limit)
            return
        unread_s = stream._unread_s() if stream is not None else 0.0
        self._watch_unread_reply(max(limit - unread_s, _UNREAD_WATCH_MIN_S))

    def _ensure_not_holding(self) -> None:
        """End this thread's own reply where it holds the wrapper waiting for this thread to read on, and refuse an action a tool the holding reply runs asks for, which would wait for itself.

        Raises:
            OutOfOrderError: Where a tool of the reply that holds the wrapper
                asks for an action, or a tool past ``tool_timeout_s`` does.
        """
        current = threading.current_thread()
        if current in self._late_tool_threads:
            raise OutOfOrderError(
                None, "a tool that did not return within tool_timeout_s asked for an action on "
                "the wrapper; its call was answered as failed and what it returns is dropped, "
                "so call nothing on the wrapper from a tool")
        if current in self._tool_threads or self._holder is current \
                and self._reply_waiting_for_reader is None:
            raise OutOfOrderError(
                None, "a tool a reply runs asked for an action on the wrapper, which would "
                "wait for that reply while the reply waits for the tool; call nothing on the "
                "wrapper from a tool it runs")
        if self._holder is current:
            # A stream another thread has begun to step is that thread's from then on: the
            # action waits for it instead.
            if self._end_reply_waiting_for_reader():
                _log.warning("a reply this thread had not read to its end was ended by its "
                             "next action")

    def _end_reply_waiting_for_reader(self) -> bool | None:
        """End the reply that holds the wrapper waiting for its reader, through its stream, unless a step of it runs; whether it did, ``None`` where no stream waits."""
        named = self._reply_waiting_for_reader
        stream = named() if named is not None else None
        return None if stream is None else stream._end_if_unread(0.0)


class _Waiting:
    """How many actions wait for the wrapper, counted while each waits; set while any does, as an event is."""

    def __init__(self) -> None:
        self._count = 0
        self._changed = threading.Condition()

    def __enter__(self) -> None:
        with self._changed:
            self._count += 1
            self._changed.notify_all()

    def __exit__(self, *_) -> None:
        with self._changed:
            self._count -= 1

    def is_set(self) -> bool:
        return self._count > 0

    def wait(self, timeout: float | None = None) -> bool:
        """Block until an action waits; whether one did within ``timeout``."""
        with self._changed:
            return self._changed.wait_for(self.is_set, timeout)


def _policy_of(given: ToolCallPolicy | ToolRegistry | None,
               turn: ToolCallPolicy | ToolRegistry | None) -> ToolCallPolicy | ToolRegistry:
    """The policy a reply runs with: ``given``, else ``turn``, that of the turn it goes on with, else ``"refuse"``."""
    # Compared with None: an empty registry is false.
    if given is not None:
        return given
    return turn if turn is not None else "refuse"


def _stop_texts(stop) -> tuple[str, ...]:
    """A request's stop texts, read once; a single ``str`` is one stop text.

    Raises:
        TypeError: If ``stop`` is neither a ``str`` nor an iterable of ``str``.
    """
    texts = (stop,) if isinstance(stop, str) else tuple(stop) if isinstance(stop, Iterable) else stop
    if texts is not None and (not isinstance(texts, tuple)
                              or not all(isinstance(text, str) for text in texts)):
        raise ArgumentTypeError(f"stop must be a str or an iterable of str, not {stop!r}")
    return texts or ()


def _ensure_cancel_and_sampling(cancel, sampling) -> None:
    """Refuse a request's ``cancel`` or ``sampling`` of a kind :meth:`ChatWrapper.reply` cannot use.

    Raises:
        TypeError: Naming the option and what it must be.
    """
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise ArgumentTypeError(f"cancel must be a threading.Event, or anything with an is_set() "
                                f"method, not {cancel!r}")
    if sampling is not None and not isinstance(sampling, Sampling):
        raise ArgumentTypeError(f"sampling must be a Sampling, not {sampling!r}")


def _ensure_max_calls_runs(policy: ToolCallPolicy | ToolRegistry, max_calls: int | None) -> None:
    """Refuse a ceiling on calls under a policy that runs none.

    Raises:
        InvalidArgumentError: If ``max_calls`` is given and ``policy`` is no registry.
    """
    if max_calls is not None and not isinstance(policy, ToolRegistry):
        raise InvalidArgumentError("max_calls", f"max_calls limits calls that are run, and "
                                                f"tool_call_policy={policy!r} runs none")


# How long the work after a reply waits for the wrapper before it leaves that work to the
# next action, which does it first; at most this long a process waits for it at its end.
_AFTER_REPLY_WAIT_S = 1.0
# How long the work after a reply waits before it starts, so that the reader of the stream
# takes in its end without sharing the interpreter with it; at least 0.
_AFTER_REPLY_PAUSE_S = 0.005
# The least time between two looks for a reply that nobody reads.
_UNREAD_WATCH_MIN_S = 0.05
# How often a reply waiting for the wrapper looks at its cancel.
_CANCEL_LOOK_S = 0.05


def _ensure_bool(value, name: str) -> None:
    """Refuse a switch that is not ``True`` or ``False``.

    Raises:
        ArgumentTypeError: Naming ``name``.
    """
    if not isinstance(value, bool):
        raise ArgumentTypeError(f"{name} must be True or False, not {value!r}")


def _kept_of_the_newest(before: str, newest: str, keep_text: str) -> str:
    """What of ``newest``, the newest reply's text, a reader kept who got ``keep_text`` of a stream that showed ``before`` first, the turn's earlier replies, or of one that showed only that reply."""
    between = parting(before, newest)
    turn = before + between
    if before and keep_text.startswith(turn):
        return keep_text[len(turn):]
    if before and keep_text and turn.startswith(keep_text) and not newest.startswith(keep_text):
        return ""                       # it ended before the newest reply's text
    if between and (keep_text.startswith(between) or between.startswith(keep_text)):
        return keep_text[len(between):]
    return keep_text


def _shown_before(messages: list[dict]) -> str:
    """The text of the replies of the newest turn before its newest message, which that message was streamed after."""
    asked = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    return "".join(m["content"] or "" for m in messages[asked + 1:-1] if m["role"] == "assistant")


def _calls_tools(item) -> bool:
    """Whether the history item ``item`` holds a call or a call's result."""
    return isinstance(item, dict) and (item.get("role") == "tool" or bool(item.get("tool_calls")))
