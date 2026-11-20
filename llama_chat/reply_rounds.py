"""A reply's generations: one, or rounds of calls run against a registry or refused, until the model answers or the reply cap ends it."""

from __future__ import annotations

import logging
import contextlib
import copy
import math
import threading
import weakref
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field, replace

from .config import Config, DrawnSampling
from .rendered_cache import RenderedCache
from .errors import ContextOverflowError
from .reply import Reply, ToolCallPolicy
from .resolved_template import ResolvedTemplate, ResolvedTemplates
from .tool_registry import ToolRegistry
from .tools import ToolCall
from .loop_variant import LoopVariant

_log = logging.getLogger(__name__)


@dataclass
class CallBudget:
    """How many calls a turn run against a registry may still use, and the ceiling it was given, ``None`` for none."""

    max_calls: int | None
    # An absent ceiling counts down without ever reaching zero, so what reads it
    # needs no case of its own.
    calls_left: float = field(init=False)

    def __post_init__(self) -> None:
        self.calls_left = math.inf if self.max_calls is None else self.max_calls


def answerable(calls: list[ToolCall]) -> list[ToolCall]:
    """Of the calls a generation reports, those that were read, which a result answers."""
    return [call for call in calls if call.status == "complete"]


def _unreadable(calls: list[ToolCall]) -> bool:
    """Whether a generation reports a call that could not be read, which its turn goes on after."""
    return any(call.status == "unreadable" for call in calls)


def _closed_for_room(generation: Reply, unanswered: list[str], ran: set[str]) -> Reply:
    """``generation`` once its turn closed on what the model wrote, for want of room to answer the calls ``unanswered`` names: each reported ``"result_dropped"`` where its tool ran, ``"unfinished"`` where none did."""
    return replace(generation, stop_reason="token_limit", tool_calls=[
        replace(call, status="result_dropped" if call.id in ran else "unfinished")
        if call.id in unanswered else call for call in generation.tool_calls])


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
        tool_results_cut=tuple(call for t in generations for call in t.tool_results_cut),
        n_reasoning_tokens=sum(t.n_reasoning_tokens for t in generations),
        thought="\n\n".join(t.thought for t in generations if t.thought),
        evicted_for=next((t.evicted_for for t in reversed(generations) if t.evicted_for), None),
        taken_back=generations[-1].taken_back,
        kept_text=generations[-1].kept_text,
    )


# How many calls that could not be read one reply answers before it reads no more calls.
MAX_UNREADABLE_CALLS_IN_A_ROW = 3

# What a call that could not be read is answered with, before what to do next.
UNREADABLE = {"error": "unreadable_call", "message": "the call could not be read"}


@dataclass(frozen=True)
class _Raised:
    """What a tool's thread raised, to be raised again on the reply's thread."""

    error: BaseException


def explain_failure(result, may_call_again: bool):
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


def parting(before: str, text: str) -> str:
    """What a stream shows between ``before``, the text shown before a generation in its turn, and ``text``, that generation's: a blank line where they would join without whitespace."""
    return "\n\n" if before and text and not before[-1].isspace() and not text[0].isspace() else ""


class ReplyRounds:
    """Generates a reply in as many rounds as its calls need, and writes their results."""

    def __init__(self, conversation: RenderedCache, templates: ResolvedTemplates, config: Config, ensure_open: Callable[[], None], tool_threads: weakref.WeakSet[threading.Thread], late_tool_threads: weakref.WeakSet[threading.Thread]) -> None:
        self._conversation = conversation
        self._templates = templates
        self._config = config
        self._ensure_open = ensure_open
        self._tool_threads = tool_threads
        self._late_tool_threads = late_tool_threads
        # The answers to calls whose writing the engine failed at, which the next reply or
        # result writes first, each while its call is still owed.
        self.answers_waiting: list[tuple[ToolCall, object]] = []

    # ----- internals -----------------------------------------------------------
    def until_answered(
        self, text: str | None, *, policy: ToolRegistry | ToolCallPolicy, budget: CallBudget,
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
            on_started: As :meth:`generate`, for the first generation.
            text, max_reply_tokens, **kwargs: As :meth:`~llama_chat.wrapper.ChatWrapper.reply`.

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
        open_turn_text = self.open_turn_text() if text is None else ""
        reply_cap = self.reply_cap(max_reply_tokens)
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
        deltas = self.generate(text, recognizes_calls=self._template.properties.can_write_results,
                                may_call_again=may_call_after_unreadable(),
                                max_reply_tokens=reply_cap, on_started=on_started, **kwargs)
        try:
            rounds_variant = LoopVariant("a reply's rounds of calls")
            while True:
                rounds_variant.passes(tokens_left())
                generation = yield from self.read_generation(
                    deltas, after=open_turn_text + "".join(g.text for g in generations))
                generations.append(generation)
                if not answerable(generation.tool_calls):
                    # A request cancelled before it started, or cut by the reply cap, leaves an
                    # open turn open.
                    if self._conversation.turn_open and _unreadable(generation.tool_calls) \
                            and generation.stop_reason != "interrupted":
                        # An unreadable call, already answered: the turn goes on, spending
                        # budget. See docs/decisions/0040-every-call-is-answered-and-reported.md.
                        any_call_asked = True
                        unreadable_in_a_row += 1
                        if registry is not None and budget.calls_left > 0:
                            budget.calls_left -= 1
                        deltas = self._round(
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
                cancel = kwargs["cancel"]
                settled, ran = yield from self._settle_calls(
                    answerable(generation.tool_calls), registry, budget, cancel)
                may_call_again = registry is not None and budget.calls_left > 0
                # Cancelled between calls, the calls that ran keep their results and the rest
                # stay owed, as where the stream is closed at a call.
                cancelled = len(settled) < len(answerable(generation.tool_calls))
                dropped, unanswered = self._write_or_close(settled, may_call_again)
                generations[-1] = replace(
                    generations[-1],
                    n_evicted_messages=generations[-1].n_evicted_messages + dropped)
                if unanswered:
                    generations[-1] = _closed_for_room(generations[-1], unanswered, ran)
                if cancelled:
                    generations[-1] = replace(generations[-1], stop_reason="interrupted")
                if cancelled or unanswered:
                    return _sum_generations(generations, not cancelled and budget.calls_left == 0)
                deltas = self._round(may_call_again, may_call_after_unreadable(),
                                              max_reply_tokens=tokens_left(), **kwargs)
        finally:
            deltas.close()

    def _round(self, recognizes_calls: bool, may_call_again: bool,
                        **kwargs) -> Generator[str, None, Reply]:
        """Go on with the open turn after a round of results; one that recognizes no call cannot open one with a tag."""
        return self.generate(None, recognizes_calls=recognizes_calls,
                              may_call_again=may_call_again, withhold=not recognizes_calls,
                              **kwargs)

    def until_called(self, text: str | None, *, max_reply_tokens: int | None,
                               **kwargs) -> Generator[str, None, Reply]:
        """Generate until the model makes a call that can be handed over, or answers.

        A call that cannot be read is answered with a failure and the reply
        goes on, every generation of it within one reply cap; once
        ``MAX_UNREADABLE_CALLS_IN_A_ROW`` such calls came in a row, it goes on
        reading no call. The lock must be held.
        """
        reply_cap = self.reply_cap(max_reply_tokens)
        open_turn_text = self.open_turn_text()
        generations: list[Reply] = []
        tries_variant = LoopVariant("a reply's tries after unreadable calls")
        while True:
            tries_variant.passes(reply_cap - sum(g.n_generated_tokens for g in generations))
            used = sum(g.n_generated_tokens for g in generations)
            unreadable_in_a_row = len(generations)
            recognizes_calls = unreadable_in_a_row < MAX_UNREADABLE_CALLS_IN_A_ROW
            generation = yield from self.read_generation(
                self.generate(text, recognizes_calls=recognizes_calls,
                               may_call_again=unreadable_in_a_row + 1 < MAX_UNREADABLE_CALLS_IN_A_ROW,
                               withhold=not recognizes_calls,
                               max_reply_tokens=max(0, reply_cap - used), **kwargs),
                after=open_turn_text + "".join(g.text for g in generations))
            generations.append(generation)
            if (answerable(generation.tool_calls) or not self._conversation.turn_open
                    or not _unreadable(generation.tool_calls)
                    or generation.stop_reason == "interrupted"):
                return generation if len(generations) == 1 else _sum_generations(generations, False)
            text = None
            kwargs.pop("on_started", None)

    def read_generation(self, deltas: Generator[str, None, Reply], after: str):
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
                    separator = parting(after, delta)
                    delta = separator + delta
                yield delta
        finally:
            # Closed here, a failure while the generation records itself reaches the caller.
            deltas.close()

    def open_turn_text(self) -> str:
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
                      budget: CallBudget, cancel):
        """Decide what every call of one round gets back, running what may run, and hand over each call that runs before it runs.

        The calls are settled in the order asked, before any result is written.
        A call handed over counts against ``budget`` from then on. Where the
        stream is closed at a call handed over, or anything is raised while it
        runs, a ``KeyboardInterrupt`` included, the results of the calls that ran
        before it are written, and it and the rest of the round stay owed.
        Where ``cancel`` is set before a call would run, the calls from it on
        are left out.

        Returns:
            The ``(call, result)`` pairs, and the ids of the calls whose tool ran.
        """
        settled: list[tuple[ToolCall, object]] = []
        ran: set[str] = set()
        for call in asked:
            if registry is not None and cancel is not None and cancel.is_set():
                break
            if registry is None:
                # Not "no tools exist": `begin` may have declared some. Raising
                # would leave the turn open with no way to end it.
                result = {"error": "tools_not_run",
                          "message": "no tool can be run for this request"}
            elif budget.calls_left > 0:
                budget.calls_left -= 1
                try:
                    # A copy: what the reader does with it changes nothing the tool is given.
                    yield replace(call, arguments=copy.deepcopy(call.arguments))
                    ran.add(call.id)
                    result = self._run_tool(registry, call)
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
        return settled, ran

    def _run_tool(self, registry: ToolRegistry, call: ToolCall):
        """What ``registry`` returns for ``call``, run on a thread of its own; a failure where it does not return within ``tool_timeout_s``.

        A tool past its time keeps running, and what it returns is dropped.
        Without a time limit, it runs on the reply's own thread.
        """
        if self._config.tool_timeout_s is None:
            return registry.call(call)
        returned: list = []

        def run() -> None:
            try:
                returned.append(registry.call(call))
            except BaseException as error:      # raised again on the reply's thread
                returned.append(_Raised(error))
        runner = threading.Thread(target=run, name=f"llama_chat tool {call.name}", daemon=True)
        self._tool_threads.add(runner)
        runner.start()
        runner.join(min(self._config.tool_timeout_s, threading.TIMEOUT_MAX))
        if not returned:
            self._late_tool_threads.add(runner)
            _log.warning("the tool %s did not return within %s s; the model is told it failed",
                         call.name, self._config.tool_timeout_s)
            return {"error": "tool_timeout",
                    "message": f"the tool did not return within {self._config.tool_timeout_s} s"}
        if isinstance(returned[0], _Raised):
            raise returned[0].error
        return returned[0]

    def _write_results(self, settled: Sequence[tuple[ToolCall, object]],
                       may_call_again: bool) -> tuple[int, list[str]]:
        """Write the result of each settled call, in order, a failure saying whether another call is worth making, until the turn closes for want of room; how many history messages were evicted for them, and the ids of the calls left unanswered.

        Where anything but a want of room stops the writing, the answers not
        yet written wait for the next reply or result, which writes them first.
        """
        answers = [(call, explain_failure(result, may_call_again)) for call, result in settled]
        unanswered: list[str] = []

        def write() -> None:
            for at, (call, answer) in enumerate(answers):
                if call.id not in {owed.id for owed in self._conversation.pending_calls}:
                    unanswered.extend(later.id for later, _ in answers[at:])
                    return
                self._conversation.add_result(call.id, answer)
        try:
            return self._evicted_by(write), unanswered
        except ContextOverflowError:
            raise
        except BaseException:
            owed = {call.id for call in self._conversation.pending_calls}
            self.answers_waiting = [(call, answer) for call, answer in answers if call.id in owed]
            raise

    def _write_or_close(self, settled: Sequence[tuple[ToolCall, object]],
                        may_call_again: bool) -> tuple[int, list[str]]:
        """Write the results of ``settled``, as :meth:`_write_results` does, the turn closed on what the model wrote where no room is left for one; how many history messages went, and the ids of the calls left unanswered."""
        try:
            return self._write_results(settled, may_call_again)
        except ContextOverflowError:
            owed = {call.id for call in self._conversation.pending_calls}
            dropped = self._evicted_by(self._conversation.close_unanswered)
            return dropped, [call.id for call, _ in settled if call.id in owed]

    def _evicted_by(self, action: Callable[[], None]) -> int:
        """Run ``action``; how many history messages it evicted."""
        before = self._conversation.changes_so_far().n_evicted
        action()
        return self._conversation.changes_so_far().n_evicted - before

    def generate(self, text: str | None, *, recognizes_calls: bool, may_call_again: bool,
                  max_reply_tokens: int | None, stop: tuple[str, ...],
                  cancel: threading.Event | None, sampling: DrawnSampling,
                  max_thought_tokens: int | None,
                  on_reply: Callable[[Reply], None], withhold: bool = False,
                  on_started: Callable[[], None] | None = None) -> Generator[str, None, Reply]:
        """One generation, as :meth:`~llama_chat.rendered_cache.RenderedCache.generation` runs it; the lock must be held.

        Args:
            text, stop, cancel: As :meth:`~llama_chat.wrapper.ChatWrapper.reply`.
            max_reply_tokens: The cap on this generation's tokens, ``None``
                for the config's.
            max_thought_tokens: The cap on a thought's tokens, ``None`` for none.
            may_call_again: Whether a call is read again after a call this
                generation writes that cannot be read, which the answer to
                that call tells the model.
            on_reply: Given the generation's Reply once it ends, early or
                not, for a stream closed before it returns.
        """
        self._ensure_open()
        answers_first: list[tuple[str, object]] = []
        if on_started is not None and self.answers_waiting:
            # The answers the engine failed at, written by the request's first generation; once
            # it starts they are written, and wait no more.
            answers_first = [(call.id, answer) for call, answer in self.answers_waiting]
            first_started = on_started

            def on_started() -> None:
                self.answers_waiting = []
                first_started()
        generation = self._conversation.generation(
            text, recognizes_calls=recognizes_calls, withhold=withhold,
            max_generated_tokens=self.reply_cap(max_reply_tokens), stop=stop, cancel=cancel,
            sampling=sampling, max_thought_tokens=max_thought_tokens, unreadable_result=explain_failure(UNREADABLE, may_call_again),
            on_started=on_started, on_reply=on_reply, answers_first=answers_first)
        return (yield from generation)

    def reply_cap(self, max_reply_tokens: int | None) -> int:
        """The cap on a reply's tokens: ``max_reply_tokens``, else the config's, else the context's size."""
        return next((c for c in (max_reply_tokens, self._config.max_reply_tokens) if c is not None),
                    self._config.context_size)

    @property
    def _template(self) -> ResolvedTemplate:
        """The template the conversation is written with now."""
        return self._conversation.template
