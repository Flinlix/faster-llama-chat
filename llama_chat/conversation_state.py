"""The state a conversation keeps between actions, and the copies made of it: what an action puts back, what readers without the lock see, and what an action changed unasked."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from .rendering import KeptTurn
from .tools import ToolCall, ToolSpec


@dataclass(frozen=True, eq=False)
class MadeFor:
    """The conversation and the cache a record made ahead was made for, which it is used with only while they stay."""

    messages: tuple[dict, ...]
    tools: tuple[ToolSpec, ...]
    reasoning: bool
    opening: str | None
    cache_version: int


@dataclass
class Ahead:
    """The render for the next question, decoded up to the run of text the question joins."""

    rest: list[tuple[str, bool]]          # from that run on, the stand-in as the caller's own
    after_question: str                   # the template's text after the question
    reply_opening: list[tuple[str, bool]]  # the render from where the reply's kept turn begins
    made_for: MadeFor


@dataclass
class ConversationState:
    """What a failed action puts back as it was, the cache's tokens besides."""

    messages: list[dict] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)
    # Whether it is written with the model's template for tools: begun with tools, or
    # declaring some. True wherever ``tools`` holds any.
    uses_tool_template: bool = False
    reasoning: bool = False
    kept: list[KeptTurn] = field(default_factory=list)
    # Where each placed kept turn starts in the cache.
    kept_at: dict[KeptTurn, int] = field(default_factory=dict)
    n_calls: int = 0
    results_cut: list[ToolCall] = field(default_factory=list)   # since the newest generation
    windows: bool = False                 # whether windows of the conversation stand for its render
    splice_probe: str | None = None       # what a question's render after one exchange is compared with
    repeats_alike: bool = True            # whether a repeated question is written as any other
    # Whether a past turn that called tools is written as before once another exchange follows.
    tool_turns_stay: bool = False
    # What the system turn and a one-character question take, with the generation prompt.
    with_a_short_question: int = 0
    # Whether the open turn's newest call was made inside a thought the model had not closed.
    called_in_thought: bool = False
    ahead: Ahead | None = None
    prompt_text: str = ""                 # the template text the next generation follows
    prompt_start: int = 0                 # where in the cache the generation prompt begins
    # Where a spliced question's run begins, the tokens decoded there, and what they were
    # decoded for: the conversation with the question newest, and the cache after them.
    spliced_at: tuple[int, list[int], MadeFor] | None = None
    # The tokens before the question the newest ask added, and the conversation and the
    # cache that ask left, until a generation follows.
    asked: tuple[list[int], MadeFor] | None = None
    begun: bool = False
    opening: str | None = None            # how the conversation opens, to see a clock move on

    def copied(self) -> ConversationState:
        """This state as it is now, each list and dict copied so that later changes leave it alone."""
        return ConversationState(**{f.name: copied_value(getattr(self, f.name)) for f in fields(self)})


@dataclass
class Changes:
    """What the actions since the caller last asked changed beyond what was asked, as :class:`~llama_chat.reply.Report` gives it."""

    evicted_for: str | None = None
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
class Snapshot:
    """The conversation and the cache's tokens at one moment, for an attempt to go back to."""

    state: ConversationState
    tokens: list[int]


@dataclass(frozen=True)
class Published:
    """The conversation as the newest finished change left it, for readers that hold no lock."""

    system_prompt: str = ""
    history: tuple = ()
    tools: tuple | None = None            # None where it is not written with the template for tools
    reasoning: bool = False
    pending_calls: tuple = ()
    begun: bool = False
    turn_open: bool = False
    newest_role: str | None = None
    total_tokens: int = 0
    # Each kept turn's tokens, how many messages it covers, and their portable digest.
    kept: tuple = ()
    # The model's fingerprint, where a turn is kept: what the kept turns' tokens are valid for.
    model: str | None = None


def copied_value(value):
    """``value`` as it is now, a list or dict copied so that later changes leave it alone."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def goes_on_from(state: ConversationState, made_for: MadeFor) -> bool:
    """Whether the conversation goes on from the one ``made_for`` was made for: its messages first, the same declarations, reasoning and opening."""
    return (len(made_for.messages) <= len(state.messages)
            and all(a is b for a, b in zip(made_for.messages, state.messages))
            and made_for.tools == tuple(state.tools)
            and (made_for.reasoning, made_for.opening) == (state.reasoning, state.opening))
