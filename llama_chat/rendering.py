"""A conversation's renders: any messages as its template writes them, split into the template's text and the caller's, and tokenized as the cache holds them.

A turn kept as the tokens it was generated as stands in the tokens for the
template's version of it. See design_history.md, "The target, built without
the cache".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .messages import exchange_starts, system_part
from .render_checks import STAND_IN
from .resolved_template import ResolvedTemplate
from .split_render import SplitRender, split_render
from .template import common_prefix_length
from .tokenizing import MemoTokenizer, split_at_tags, tokenize_pieces
from .tools import ToolSpec


@dataclass(frozen=True, eq=False)
class KeptTurn:
    """A turn held as the tokens it was generated as, in place of the template's version of it."""

    tokens: tuple[int, ...]
    first: dict                           # its first message
    last: dict                            # the message it ends with
    on_call: bool                         # whether it ends on a call, else on a reply's text


@dataclass(frozen=True)
class Target:
    """What the cache should hold for some messages, and where each kept turn placed in it begins."""

    tokens: list[int]
    kept_at: dict[KeptTurn, int]


class RenderInputs(Protocol):
    """What a conversation's renders are made of besides its messages."""

    tools: list[ToolSpec]
    reasoning: bool
    kept: list[KeptTurn]


class ConversationRendering:
    """One conversation's renders, made with its declarations, its reasoning switch and its kept turns.

    ``inputs`` returns those as they are, and is read at every use. Renders
    and splits are reused until :meth:`forget_renders`, which is due wherever
    the declarations, the reasoning switch or the clock may have changed.
    """

    def __init__(self, context, template: Callable[[], ResolvedTemplate],
                 inputs: Callable[[], RenderInputs]) -> None:
        self._context = context
        self._template_of = template
        self._inputs = inputs
        self._reader = MemoTokenizer(context)
        self._memo: dict = {}             # renders and splits, by what was rendered
        self._tag_tokens: dict = {}       # each template's call tags, by their texts

    @property
    def _template(self) -> ResolvedTemplate:
        """The template the conversation is written with now."""
        return self._template_of()

    def forget_renders(self) -> None:
        """Render everything again from now on."""
        self._memo.clear()

    # ----- rendering ----------------------------------------------------------------
    def render(self, messages, generation_prompt: bool) -> str | None:
        """``messages`` as the template writes them, ``None`` where it refuses that shape."""
        inputs = self._inputs()
        tools = [spec.as_declaration() for spec in inputs.tools] or None
        key = ("render", json.dumps([messages, generation_prompt], sort_keys=True, default=str))
        if key not in self._memo:
            self._memo[key] = self._template.render(
                messages, tools, generation_prompt=generation_prompt,
                reasoning=inputs.reasoning)
        return self._memo[key]

    def split(self, messages, generation_prompt: bool) -> SplitRender | None:
        """``messages`` rendered and split into the template's text and the caller's.

        Raises:
            ValueError: If the template writes a text otherwise than as text in its place.
        """
        key = ("split", json.dumps([messages, generation_prompt], sort_keys=True, default=str))
        if key not in self._memo:
            self._memo[key] = split_render(lambda m: self.render(m, generation_prompt),
                                           messages, self._template.fragments.content)
        return self._memo[key]

    def opening_render(self, system: list[dict]) -> str | None:
        """How the template opens a conversation on ``system``, which changes where it writes the date, say."""
        return self.render(system + [{"role": "user", "content": STAND_IN}], True)

    # ----- the target ---------------------------------------------------------------
    def target(self, messages, generation_prompt: bool, upto: int | None = None) -> Target:
        """``messages`` as the cache holds them: their render tokenized, each kept turn as it was generated.

        Args:
            upto: Where in the render to stop.

        Raises:
            ValueError: If the template refuses to write them, or writes a kept
                turn among them where it cannot be placed.
        """
        if not messages:
            return Target(self.tokenize([], add_special=True), {})     # the opener alone
        split = self.split(messages, generation_prompt)
        if split is None:
            raise ValueError("the template refuses to write this conversation")
        end = len(split.text) if upto is None else upto
        turns = self._inputs().kept
        placed = []
        for kept in turns:
            if not all(any(m is end for m in messages) for end in (kept.first, kept.last)):
                continue            # a turn these messages do not hold
            found = self.place(kept, messages, split)
            if found is None:
                raise ValueError("this model's template writes a turn that called tools where "
                                 "it cannot be told apart, so the turn could not keep the tokens "
                                 "it was generated as")
            if found[1] <= end:
                placed.append(found)
        placed.sort()
        tokens, at, kept_at = [], 0, {}
        for start, stop, n in placed:
            assert start >= at, "two kept turns overlap"
            tokens += self.tokenize(split.pieces(at, start), add_special=not tokens and at == 0)
            kept_at[turns[n]] = len(tokens)
            tokens += turns[n].tokens
            at = stop
        tokens += self.tokenize(split.pieces(at, end), add_special=not tokens and at == 0)
        return Target(tokens, kept_at)

    def length(self, messages: list[dict], generation_prompt: bool) -> int:
        """How many tokens the cache holds for ``messages``, each kept turn as it was generated.

        Where the template refuses a conversation that ends on a reply, up to
        where a question after it would go.
        """
        if not generation_prompt and messages and messages[-1]["role"] != "user" \
                and self.render(messages, False) is None:
            asked = messages + [{"role": "user", "content": STAND_IN}]
            split = self.split(asked, True)
            if split is not None and split.contents[-1] is not None:
                return len(self.target(asked, True, upto=split.contents[-1][0]).tokens)
        return len(self.target(messages, generation_prompt).tokens)

    def opening_length(self, system: list[dict]) -> int:
        """How many tokens open a conversation on ``system`` whatever its oldest exchange: the system turn, its declarations and the sequence opener.

        The tokens two renders share that differ only in the oldest question,
        so that what a template writes there only once a question follows is
        counted, and nothing that moves with the newest question is.
        """
        first, second = (self.target(system + [
            {"role": "user", "content": question}, {"role": "assistant", "content": "A reply."},
            {"role": "user", "content": "Another question?"}], generation_prompt=True).tokens
            for question in ("Alpha one?", "Zulu two?"))
        return common_prefix_length(first, second)

    def head(self, messages: list[dict], through: int) -> list[int] | None:
        """The render up to the first tag after the reply of oldest exchange ``through``, 0 or 1, from the system message and the oldest two exchanges of ``messages``; ``None`` where it cannot be told."""
        starts = exchange_starts(messages)
        if len(starts) <= through:
            return None
        bounds = starts[1:] + [len(messages)]
        kept = messages[starts[0]:bounds[min(1, len(bounds) - 1)]]
        if not self.template_writes(kept):
            return None
        system = system_part(messages)
        split = self.split(system + kept + [{"role": "user", "content": STAND_IN}],
                           generation_prompt=True)
        if split is None:
            return None
        anchor = self.tag_after_content(split, len(system) + bounds[through] - starts[0] - 1)
        if anchor is None:
            return None
        return self.tokenize(split.pieces(0, anchor), add_special=True)

    def tokenize(self, pieces: list[tuple[str, bool]], add_special: bool = False) -> list[int]:
        """``pieces`` as a render tokenizes them, a call's tags as the single tokens a generated call holds."""
        return tokenize_pieces(self._reader, pieces, add_special=add_special,
                               tag_tokens=self._call_tags())

    def _call_tags(self) -> dict[str, int]:
        """The tokens the template's call tags are, by their texts."""
        template = self._template
        if template not in self._tag_tokens:
            self._tag_tokens[template] = {
                text: token for token in (template.call_boundary().tool_tokens or ())
                if (text := self._context.token_text(token))}
        return self._tag_tokens[template]

    # ----- kept turns -----------------------------------------------------------------
    def place(self, kept: KeptTurn, messages: list[dict], split: SplitRender):
        """Where the template writes ``kept``'s turn in ``split``: from the first tag after the question's text to the end of its call or reply; ``None`` where it cannot be placed."""
        first = next((i for i, m in enumerate(messages) if m is kept.first), None)
        last = next((i for i, m in enumerate(messages) if m is kept.last), None)
        if not first or last is None:
            return None
        start = self.turn_start(split, first - 1)
        if kept.on_call:
            called = split.call_ends[last]
            closing = self._template.call_close_text
            found = split.text.find(closing, called) if called is not None and closing else -1
            end = found + len(closing) if found >= 0 else None
        elif split.contents[last] is not None:
            end = split.contents[last][1]
        else:
            # A reply without text: it ends where the template's closing after it begins,
            # the last before the next text of the caller's own, as the question has one too.
            closing = self._template.fragments.assistant_suffix.strip()
            following = next((s for s, _ in split.own if start is not None and s > start),
                             len(split.text))
            found = (split.text.rfind(closing, start, following)
                     if closing and start is not None else -1)
            end = found if found >= 0 else None
        if start is None or end is None or end < start:
            return None
        return start, end, self._inputs().kept.index(kept)

    def turn_start(self, split: SplitRender, before: int) -> int | None:
        """Where a turn kept as generated begins in ``split`` after message ``before``: at the first tag after its content."""
        return self.tag_after_content(split, before)

    def template_writes(self, messages: list[dict]) -> bool:
        """Whether the template's render alone stands for ``messages`` in the cache: no call, result or turn kept as generated among them."""
        kept = {id(m) for k in self._inputs().kept for m in (k.first, k.last)}
        return not any(m["role"] == "tool" or m.get("tool_calls") or id(m) in kept
                       for m in messages)

    # ----- tags ---------------------------------------------------------------------
    def tag_after_content(self, split: SplitRender, index: int) -> int | None:
        """Where the first tag after message ``index``'s content begins, before the next text of the caller's own."""
        if split.contents[index] is None:
            return None
        return self.tag_from(split.text, split.contents[index][1], split)

    def tag_from(self, text: str, at: int, split: SplitRender | None = None) -> int | None:
        """Where the first tag at or after ``at`` begins, before the next own text of ``split`` where given."""
        end = next((s for s, _ in split.own if s >= at), len(text)) if split else len(text)
        pieces = split_at_tags(self._context, text[at:end])
        if pieces is None:
            return None
        for piece, is_tag in pieces:
            if is_tag:
                return at
            at += len(piece)
        return None

    def run_start(self, split: SplitRender | None, index: int, start: int | None) -> int | None:
        """Where the run of text holding message ``index``'s content begins: after the last tag from ``start`` on."""
        if split is None or start is None or split.contents[index] is None:
            return None
        run = None
        for piece, own in split.pieces(start, split.contents[index][0]):
            if own:
                start += len(piece)
                continue
            pieces = split_at_tags(self._context, piece)
            if pieces is None:
                return None
            for part, is_tag in pieces:
                start += len(part)
                if is_tag:
                    run = start
        return run
