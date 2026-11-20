"""The state a conversation keeps between actions, and the copies made of it: what an action puts back, what readers without the lock see, and what an action changed unasked."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass, replace

from .rendering import GeneratedTurn
from .reply import EvictionCause
from .tools import ToolCall, ToolSpec


def reasoning_switch(reasoning: bool, level: str | None) -> bool | str:
    """The switch a conversation's renders take for ``reasoning`` and ``reasoning_level`` as ``begin`` takes them: off, on as the template's own, or on at the effort level named."""
    return level if reasoning and level is not None else reasoning


def reasoning_asked(switch: bool | str) -> tuple[bool, str | None]:
    """``reasoning`` and ``reasoning_level`` as ``begin`` took them, for the ``switch`` :func:`reasoning_switch` made of them."""
    return bool(switch), switch if isinstance(switch, str) else None


@dataclass(frozen=True, eq=False)
class MadeFor:
    """The conversation and the cache a record made ahead was made for, which it is used with only while they stay."""

    messages: tuple[dict, ...]
    tools: tuple[ToolSpec, ...]
    reasoning: bool | str
    opening: str | None
    cache_version: int


@dataclass(frozen=True, eq=False)
class Anchor:
    """Where the cache holds the tag that opens the exchange after ``message``, the ``index``-th message, which windows of the newest exchanges are placed after.

    It stands while that message is still the ``index``-th and the generated turns
    up to it are still ``kept``, by identity.
    """

    message: dict
    index: int
    at: int
    kept: tuple[GeneratedTurn, ...]


@dataclass
class Ahead:
    """The render for the next question, decoded up to the run of text the question joins."""

    rest: list[tuple[str, bool]]          # from that run on, the stand-in as the caller's own
    after_question: str                   # the template's text after the question
    reply_opening: list[tuple[str, bool]]  # the render from where the reply's generated turn begins
    made_for: MadeFor


@dataclass
class Changes:
    """What the actions since the caller last asked changed beyond what was asked, as :class:`~llama_chat.reply.Report` gives it."""

    n_evicted: int = 0                    # history messages evicted, each result counting as one
    evicted_for: EvictionCause | None = None
    n_kept_messages: int | None = None
    n_left_out_calls: int = 0
    n_turns_restored: int = 0
    generated_turns_ignored: bool = False
    tool_results_cut: list[ToolCall] = field(default_factory=list)
    turn_closed: bool = False
    question_joined: bool = False
    taken_back: bool = False
    kept_text: str | None = None          # a reply's answer as kept, where not as shown


@dataclass(frozen=True)
class RenderChecks:
    """What the template's renders allow, as the system turn, the declarations and the clock leave them; worked out again whenever one of those changes."""

    opening: str | None = None            # how the conversation opens, to see a clock move on
    # Each form the opening wrote the clock in, with what it wrote; empty where it wrote none.
    clock_read: tuple[tuple[str, str], ...] = ()
    windows: bool = False                 # whether windows of the conversation stand for its render
    splice_probe: str | None = None       # what a question's render after one exchange is compared with
    repeats_alike: bool = True            # whether a repeated question is written as any other
    # Whether a past turn that called tools is written as before once another exchange follows.
    tool_turns_stay: bool = False
    # Whether windows of the conversation stand for its render where they hold turns that
    # called tools too.
    tool_windows: bool = False
    # What the system turn and a one-character question take, with the generation prompt.
    with_a_short_question: int = 0
    # Whether the exchanges after the oldest are written as before once it goes, so that
    # a cache that shifts keeps them; and whether they are where turns that called tools are
    # among them.
    rest_stays: bool = True
    rest_stays_around_calls: bool = True


@dataclass
class ConversationState:
    """What a failed action puts back as it was, the cache's tokens besides."""

    messages: list[dict] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    # Whether it is written with the model's template for tools: begun with tools, or
    # declaring some. True wherever ``tools`` holds any.
    uses_tool_template: bool = False
    reasoning: bool | str = False
    # Where each placed generated turn starts in the cache.
    turns_at: dict[GeneratedTurn, int] = field(default_factory=dict)
    n_calls: int = 0
    results_cut: list[ToolCall] = field(default_factory=list)   # since the newest generation
    checks: RenderChecks = RenderChecks()
    # Whether the open turn's newest call was made inside a thought the model had not closed.
    called_in_thought: bool = False
    ahead: Ahead | None = None
    prompt_text: str = ""                 # the template text the next generation follows
    prompt_start: int = 0                 # where in the cache the generation prompt begins
    # Where a spliced question's run begins, the tokens decoded there, and what they were
    # decoded for: the conversation with the question newest, and the cache after them.
    spliced_at: tuple[int, list[int], MadeFor] | None = None
    # Where the cache holds the tags that open the newest exchanges, or the next one, at
    # most two, the newest last.
    anchors: tuple[Anchor, ...] = ()
    begun: bool = False
    # What a call that could not be read is answered with, as the newest reply was told.
    unreadable_result: dict | None = None
    # The ids of the calls a begin left out as unreadable, which no result answers.
    left_out_calls: frozenset[str] = frozenset()
    # The policy the newest turn was started with, and the calls it may still run, which a
    # reply going on with it keeps; the wrapper's own, kept here to be put back with the rest.
    turn_policy: object = None
    turn_budget: object = None
    # What the actions since the caller last asked changed beyond what was asked.
    changes: Changes = field(default_factory=Changes)

    def copied(self) -> ConversationState:
        """This state as it is now, each list and dict copied so that later changes leave it alone."""
        return ConversationState(**{f.name: copied_value(getattr(self, f.name)) for f in fields(self)})

    def put_back(self, saved: ConversationState) -> None:
        """Hold what ``saved`` holds again, in this same object, copied so that later changes leave ``saved`` alone."""
        for f in fields(self):
            setattr(self, f.name, copied_value(getattr(saved, f.name)))


@dataclass(frozen=True)
class Snapshot:
    """The conversation and the cache's tokens at one moment, for an attempt to go back to."""

    state: ConversationState
    tokens: list[int]


@dataclass(frozen=True)
class Published:
    """The conversation as the newest finished change left it, for readers that hold no lock."""

    messages: tuple = ()                  # as the template is given them; never changed in place
    joins_questions: bool = False         # whether two questions in a row are written as one
    tools: tuple | None = None            # None where it is not written with the template for tools
    reasoning: bool | str = False
    pending_calls: tuple = ()
    begun: bool = False
    turn_open: bool = False
    newest_role: str | None = None
    total_tokens: int = 0
    # The model's fingerprint, where a turn is kept: what the generated turns' tokens are valid for.
    model: str | None = None
    turn_policy: object = None            # the policy the newest turn was started with


def copied_value(value):
    """``value`` as it is now, a list or dict copied so that later changes leave it alone."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, Changes):
        return replace(value, tool_results_cut=list(value.tool_results_cut))
    if is_dataclass(value) and not isinstance(value, type) and not value.__dataclass_params__.frozen:
        return copy.copy(value)
    return value


def goes_on_from(state: ConversationState, made_for: MadeFor) -> bool:
    """Whether the conversation goes on from the one ``made_for`` was made for: its messages first, the same declarations, reasoning and opening."""
    return (len(made_for.messages) <= len(state.messages)
            and all(a is b for a, b in zip(made_for.messages, state.messages))
            and made_for.tools == tuple(state.tools)
            and (made_for.reasoning, made_for.opening) == (state.reasoning, state.checks.opening))
