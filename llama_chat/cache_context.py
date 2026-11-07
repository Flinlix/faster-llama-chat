"""What resolving a template and running a conversation call on a cache context.

Two protocols: the model as it is read (its tokenizer, tags, chat template and
add-on file), and that plus the cache a conversation is decoded into. Each is
checked once, where a context is adopted, rather than at the first call that
finds a member missing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Iterator, Protocol

from .config import Sampling
from .messages import Eviction
from .template import CallBoundary, TemplateFragments


@dataclass
class GenerationAccumulator:
    """What one generation has produced so far, filled as tokens stream in.

    Every token is decoded into the cache before its text is yielded, so a
    caller that abandons the stream still reads exactly what reached the cache.

    Attributes:
        token_ids: Every token decoded into the cache.
        text: Visible reply text so far.
        stop_reason: ``"end_of_turn"``, ``"token_limit"``, ``"interrupted"`` or
            ``"tool_call"``.
        tool_call: What the model generated between its call tags, tags
            excluded, when ``stop_reason`` is ``"tool_call"``. Never part of
            ``text``.
        cached_beyond: Tokens decoded into the cache after ``token_ids`` that
            belong to no reply, such as the first tokens of a closing written
            whole later; the caller cuts them away again.
    """

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    stop_reason: str = "token_limit"
    tool_call: str = ""
    cached_beyond: int = 0


class ModelReader(Protocol):
    """The model's tokenizer, its own tags, its chat template and its add-on file."""

    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]: ...

    def retokenized(self, tokens: list[int]) -> list[int]: ...

    def tags(self) -> tuple[str, ...]: ...

    def split_at_tags(self, text: str) -> list[tuple[str, bool]] | None: ...

    def tokenizes_to_tags(self, text: str) -> bool: ...

    def first_special_token(self, text: str) -> int | None: ...

    def token_text(self, token: int) -> str: ...

    def render_with_model_template(self, messages: list[dict], tools: list[dict] | None = None,
                                   *, generation_prompt: bool = False,
                                   reasoning: bool | None = None,
                                   **variables) -> str | None:
        """``None`` where the model ships no template or it refuses the shape."""

    def chat_template_text(self) -> str | None: ...

    def extract_fragments(self) -> TemplateFragments | None: ...

    def addon_call(self) -> dict: ...


class CacheContext(ModelReader, Protocol):
    """The same, with the cache a conversation is decoded into."""

    @property
    def context_size(self) -> int:
        """How many tokens the cache holds at most."""

    @property
    def can_shift(self) -> bool: ...

    @property
    def can_cut_tail(self) -> bool: ...

    def reset(self) -> None: ...

    def warmup(self) -> None: ...

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None: ...

    def generate(self, start_pos: int, max_generated_tokens: int,
                 out: GenerationAccumulator | None = None,
                 cancel: threading.Event | None = None, sampling: Sampling | None = None,
                 ends_turn: Callable[[int], bool] | None = None,
                 boundary: CallBoundary = CallBoundary()) -> Iterator[str]: ...

    def apply_eviction(self, ev: Eviction) -> None: ...

    def remove_from(self, pos: int) -> None: ...

    def save_state(self):
        """``None`` where the cache cannot save one."""

    def restore_state(self, state, pos: int) -> None: ...

    def close(self) -> None: ...


def _members(protocol) -> tuple[str, ...]:
    return tuple(name for name in vars(protocol)
                 if not name.startswith("_"))


MODEL_READER = _members(ModelReader)
CACHE_CONTEXT = MODEL_READER + _members(CacheContext)


def ensure_answers(context, members: tuple[str, ...]) -> None:
    """Refuse a context missing any of ``members``.

    Raises:
        TypeError: Naming every member that is missing.
    """
    missing = [name for name in members if not hasattr(context, name)]
    if missing:
        raise TypeError(f"{type(context).__name__} does not answer {missing}, "
                        "which this library calls on a cache context")
