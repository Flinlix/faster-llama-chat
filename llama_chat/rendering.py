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

from .errors import TemplateRefusedError, with_template_reason
from .messages import (GENERATED_TURN, THOUGHT_KEY, UNREADABLE_CALL,
                       exchange_starts, system_part)
from .render_checks import STAND_IN
from .resolved_template import ResolvedTemplate
from .split_render import SplitRender, split_render
from .sequences import common_prefix_length
from .tokenizing import MemoTokenizer, split_at_tags, tokenize_pieces
from .tools import ToolSpec

# Stands in for a reply's text: letters only, so that no template trims or escapes it.
_ANSWER_MARK = "QQANSWERMARKQQ"

@dataclass(frozen=True, eq=False)
class GeneratedTurn:
    """A turn held as the tokens it was generated as, in place of the template's version of it; the message it ends with holds it under :data:`~llama_chat.messages.GENERATED_TURN`."""

    tokens: tuple[int, ...]
    span: int                             # how many messages it covers, the one holding it the last
    on_call: bool                         # whether it ends on a call, else on a reply's text


def generated_turns_in(messages: list[dict]) -> list[tuple[int, int, GeneratedTurn]]:
    """Each turn ``messages`` keep as generated, oldest first: the indexes of its first and last message, and the turn."""
    return [(i - message[GENERATED_TURN].span + 1, i, message[GENERATED_TURN])
            for i, message in enumerate(messages) if GENERATED_TURN in message]


def with_generated_turn(message: dict, turn: GeneratedTurn | None) -> dict:
    """``message``, as a copy, ending ``turn``, or ending none where it is ``None``."""
    rest = {key: value for key, value in message.items() if key != GENERATED_TURN}
    return rest if turn is None else {**rest, GENERATED_TURN: turn}


@dataclass(frozen=True)
class Target:
    """What the cache should hold for some messages, and where each generated turn placed in it begins.

    ``anchors`` holds, for the two newest exchanges of the messages rendered
    where known, the index of the message each follows and where in
    ``tokens`` the tag that opens it begins; ``exchanges_at`` is where the
    tags that open the conversation's two newest exchanges begin, those known.
    """

    tokens: list[int]
    turns_at: dict[GeneratedTurn, int]
    anchors: tuple[tuple[int, int], ...] = ()
    exchanges_at: tuple[int, ...] | None = None


class RenderInputs(Protocol):
    """What a conversation's renders are made of besides its messages."""

    tools: list[ToolSpec]
    reasoning: bool


class ConversationRendering:
    """One conversation's renders, made with its declarations, its reasoning switch and its generated turns.

    ``inputs`` holds those, and is read at every use as it is then. Renders
    and splits are reused while the template, the declarations and the
    reasoning switch stay as they were, and until :meth:`forget_renders`,
    which each outermost action calls, so that the memo holds one action's
    renders.
    """

    def __init__(self, context, template: Callable[[], ResolvedTemplate],
                 inputs: RenderInputs) -> None:
        self._context = context
        self._template_of = template
        self._inputs = inputs
        self._reader = MemoTokenizer(context)
        self._memo: dict = {}             # renders and splits, by what was rendered
        self._memo_made_for: tuple | None = None    # the template and inputs the memo holds
        self._tag_tokens: dict = {}       # each template's call tags, by their texts

    @property
    def _template(self) -> ResolvedTemplate:
        """The template the conversation is written with now."""
        return self._template_of()

    def forget_renders(self) -> None:
        """Render everything again from now on."""
        self._memo.clear()

    def _current_memo(self) -> dict:
        """The memo of renders and splits, emptied first where the template, the declarations or the reasoning switch changed since it was filled."""
        inputs = self._inputs
        made_for = (self._template, tuple(inputs.tools), inputs.reasoning)
        if made_for != self._memo_made_for:
            self._memo.clear()
            self._memo_made_for = made_for
        return self._memo

    # ----- rendering ----------------------------------------------------------------
    def render(self, messages, generation_prompt: bool) -> str | None:
        """``messages`` as the template writes them, ``None`` where it refuses that shape."""
        return self._render_given(self._as_given(messages), generation_prompt)

    def _render_given(self, messages, generation_prompt: bool,
                      declarations: list[dict] | None = None) -> str | None:
        """``messages``, as the template is given them, rendered with ``declarations``, by default the conversation's; ``None`` where it refuses that shape."""
        inputs = self._inputs
        tools = self._declarations() if declarations is None else declarations
        key = ("render", json.dumps([messages, declarations, generation_prompt], sort_keys=True,
                                    default=str))
        memo = self._current_memo()
        if key not in memo:
            memo[key] = self._template.render(
                messages, tools, generation_prompt=generation_prompt,
                reasoning=inputs.reasoning)
        return memo[key]

    def refusal_reason(self, messages, generation_prompt: bool) -> str | None:
        """What the template itself says where :meth:`render` refuses ``messages``; ``None`` where it says nothing."""
        return self._template.refusal_reason(self._as_given(messages), self._declarations(),
                                             generation_prompt=generation_prompt,
                                             reasoning=self._inputs.reasoning)

    def _declarations(self) -> list[dict] | None:
        """What the conversation declares, as a template is handed it; ``None`` for nothing."""
        return [spec.as_declaration() for spec in self._inputs.tools] or None

    def _as_given(self, messages: list[dict]) -> list[dict]:
        """``messages`` as the template is given them: each calling message of a generated turn without its text, which the turn's tokens hold."""
        inside = {i for first, last, _ in generated_turns_in(messages) for i in range(first, last + 1)}
        if inside or any(UNREADABLE_CALL in c for m in messages for c in m.get("tool_calls") or ()):
            # What a caller gave is the conversation's to give back: not the template's to write.
            messages = [_without_kept_texts(m) for m in messages]
        if not any(messages[i].get("tool_calls") and messages[i]["content"] for i in inside):
            return messages
        return [{**m, "content": ""} if i in inside and m.get("tool_calls") else m
                for i, m in enumerate(messages)]

    def split(self, messages, generation_prompt: bool) -> SplitRender | None:
        """``messages`` rendered and split into the template's text and the caller's.

        Raises:
            ValueError: If the template writes a text otherwise than as text in its place.
        """
        given = self._as_given(messages)
        key = ("split", json.dumps([given, generation_prompt], sort_keys=True, default=str))
        memo = self._current_memo()
        if key not in memo:
            declarations = self._declarations()
            # The render as it is is the one :meth:`render` makes, and shares its memo.
            split = split_render(
                lambda m, declared: self._render_given(
                    m, generation_prompt, None if declared is declarations else declared),
                given, self._template.properties.fragments.content, declarations,
                reads_clock=self._template.properties.reads_clock)
            memo[key] = split
        return memo[key]

    def split_made(self, messages, generation_prompt: bool) -> SplitRender | None:
        """:meth:`split` of ``messages`` where this action made it already, ``None`` where it did not."""
        given = self._as_given(messages)
        key = ("split", json.dumps([given, generation_prompt], sort_keys=True, default=str))
        return self._current_memo().get(key)

    def opening_render(self, system: list[dict]) -> str | None:
        """How the template opens a conversation on ``system``, which changes where it writes the date, say."""
        return self.render(system + [{"role": "user", "content": STAND_IN}], True)

    # ----- the target ---------------------------------------------------------------
    def target(self, messages, generation_prompt: bool, upto: int | None = None,
               start: int = 0) -> Target:
        """``messages`` as the cache holds them: their render tokenized, each generated turn as it was generated.

        Args:
            upto: Where in the render to stop.
            start: Where in the render to begin, the start of a tag; each
                generated turn there or after it is placed, and the places in
                ``turns_at`` count from it.

        Raises:
            ValueError: If the template refuses to write them, or writes a kept
                turn among them where it cannot be placed, or across ``start``.
        """
        if not messages:
            return Target(self.tokenize([], add_special=True), {})     # the opener alone
        split = self.split(messages, generation_prompt)
        if split is None:
            raise TemplateRefusedError("messages", with_template_reason(
                "the template refuses to write this conversation",
                self.refusal_reason(messages, generation_prompt)))
        end = len(split.text) if upto is None else upto
        placed = []
        for first, last, kept in generated_turns_in(messages):
            found = self.place(first, last, kept.on_call, split)
            if found is None or found[0] < start < found[1]:
                raise TemplateRefusedError("messages", "this model's template writes a turn that called tools where "
                                 "it cannot be told apart, so the turn could not keep the tokens "
                                 "it was generated as")
            if start <= found[0] and found[1] <= end:
                placed.append((*found, kept))
        tokens: list[int] = []
        at, turns_at = start, {}
        for begins, stop, kept in placed:
            assert begins >= at, "two generated turns overlap"
            tokens += self.tokenize(split.pieces(at, begins), add_special=not tokens and at == 0)
            turns_at[kept] = len(tokens)
            tokens += kept.tokens
            at = stop
        tokens += self.tokenize(split.pieces(at, end), add_special=not tokens and at == 0)
        return Target(tokens, turns_at)

    def written_answer(self, messages: list[dict], answer: str, thought: str = "",
                       cuts: tuple[str, ...] = ()) -> str | None:
        """What of ``answer``, a reply after ``messages`` with ``thought``, a conversation can hold; ``None`` where nothing.

        The answer itself where the template writes it as its text. Else a
        part of it the template does: the part it writes of the whole answer
        as a past reply, or what follows the last of ``cuts``, texts a
        template may cut a reply at, each trimmed or not; the form that
        renders as the whole answer does is preferred.
        """
        after = [{"role": "user", "content": STAND_IN}]
        reply = {"role": "assistant", "content": answer, **({THOUGHT_KEY: thought} if thought else {})}

        def with_content(content: str) -> list[dict]:
            return messages + [{**reply, "content": content}] + after

        def written_as_text(content: str) -> bool:
            """Whether the template writes ``content`` as its text, trimmed at most, as the newest reply and as a past one."""
            forms = {content, content.strip(), content.lstrip(), content.rstrip()}
            for conversation, prompt in ((with_content(content)[:-1], False), (with_content(content), True)):
                try:
                    split = self.split(conversation, prompt)
                except ValueError:
                    return False            # the template writes it otherwise than as its text
                where = split.contents[len(messages)] if split is not None else None
                if split is None or where is None or split.text[where[0]:where[1]] not in forms:
                    return False
            return True

        if written_as_text(answer):
            return answer
        as_is, marked = self.render(with_content(answer), True), self.render(with_content(_ANSWER_MARK), True)
        parts = [answer.rsplit(tag, 1)[1] for tag in cuts if tag and tag in answer]
        if as_is is not None and marked is not None and marked.count(_ANSWER_MARK) == 1:
            head, tail = marked.split(_ANSWER_MARK)
            if len(head) + len(tail) <= len(as_is) and as_is.startswith(head) and as_is.endswith(tail):
                parts.insert(0, as_is[len(head):len(as_is) - len(tail)])
        forms = [form for form in dict.fromkeys(variant for part in parts for variant in (
                     part, part.strip(), part.lstrip(), part.rstrip()))
                 if form.strip() and form in answer and written_as_text(form)]
        return next((form for form in forms if self.render(with_content(form), True) == as_is),
                    forms[0] if forms else None)

    def length(self, messages: list[dict], generation_prompt: bool) -> int:
        """How many tokens the cache holds for ``messages``, each generated turn as it was generated.

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
        """How many tokens open a conversation on ``system`` whatever its oldest exchange: the system turn, its declarations and the sequence opener."""
        first, second = (self.target(system + [
            {"role": "user", "content": question}, {"role": "assistant", "content": "A reply."},
            {"role": "user", "content": "Another question?"}], generation_prompt=True).tokens
            for question in ("Alpha one?", "Zulu two?"))
        return common_prefix_length(first, second)

    def head(self, messages: list[dict], through: int,
             with_generated_turns: bool = False) -> list[int] | None:
        """The cache's tokens up to the first tag after the reply of oldest exchange ``through``, 0 or 1, from the system message and the oldest two exchanges of ``messages``; ``None`` where they cannot be told.

        Args:
            with_generated_turns: Whether exchanges that called tools may be among
                them, each turn kept as generated placed in their render.
        """
        starts = exchange_starts(messages)
        if len(starts) <= through:
            return None
        bounds = starts[1:] + [len(messages)]
        oldest = messages[starts[0]:bounds[min(1, len(bounds) - 1)]]
        if not with_generated_turns and not self.template_writes(oldest):
            return None
        system = system_part(messages)
        written = system + oldest + [{"role": "user", "content": STAND_IN}]
        split = self.split(written, generation_prompt=True)
        if split is None:
            return None
        anchor = self.tag_after_content(split, len(system) + bounds[through] - starts[0] - 1)
        if anchor is None:
            return None
        try:
            return self.target(written, True, upto=anchor).tokens
        except ValueError:
            return None                 # a generated turn its render cannot place

    def tokenize(self, pieces: list[tuple[str, bool]], add_special: bool = False) -> list[int]:
        """``pieces`` as a render tokenizes them, a call's tags as the single tokens a generated call holds."""
        return tokenize_pieces(self._reader, pieces, add_special=add_special,
                               tag_tokens=self._call_tags())

    def count_tokens(self, text: str) -> int:
        """How many tokens ``text`` takes as a caller's text, a tag in it read as its letters."""
        return len(self._reader.tokenize(text, add_special=False, parse_special=False))

    def is_plain(self, text: str) -> bool:
        """Whether ``text`` is a question the stand-in may stand for: words without tags or surrounding space."""
        if not text or text != text.strip() or STAND_IN in text:
            return False
        split = split_at_tags(self._reader, text)
        return split is not None and not any(is_tag for _, is_tag in split)

    def _call_tags(self) -> dict[str, int]:
        """The tokens the template's call tags are, by their texts."""
        template = self._template
        if template not in self._tag_tokens:
            self._tag_tokens[template] = {
                text: token for token in (template.call_boundary().tool_tokens or ())
                if (text := self._context.token_text(token))}
        return self._tag_tokens[template]

    # ----- generated turns -----------------------------------------------------------------
    def place(self, first: int, last: int, on_call: bool,
              split: SplitRender) -> tuple[int, int] | None:
        """Where the template writes the generated turn of messages ``first`` to ``last`` in ``split``: from the first tag after the question's text to the end of its call, where it ends ``on_call``, or of its reply; ``None`` where it cannot be placed."""
        if not first:
            return None
        start = self.tag_after_content(split, first - 1)
        if on_call:
            end = self.call_closed_at(split, last)
        elif (content := split.contents[last]) is not None:
            end = content[1]
        else:
            # A reply without text: it ends where the template's closing after it begins,
            # the last before the next text of the caller's own after the message before it.
            closing = self._template.properties.fragments.assistant_suffix.strip()
            before = split.contents[last - 1]
            after = None if start is None else max(
                start, split.call_ends[last - 1] or start, before[1] if before else start)
            following = next((s for s, _ in split.own if after is not None and s >= after),
                             len(split.text))
            found = (split.text.rfind(closing, after, following)
                     if closing and after is not None else -1)
            end = found if found >= 0 else None
        if start is None or end is None or end <= start:
            return None
        return start, end

    def call_closed_at(self, split: SplitRender, index: int) -> int | None:
        """Where the closing after message ``index``'s last call ends in ``split``; ``None`` where the template writes no call of it or no closing."""
        called = split.call_ends[index]
        closing = self._template.properties.call_close_text
        found = split.text.find(closing, called) if called is not None and closing else -1
        return found + len(closing) if found >= 0 else None

    def template_writes(self, messages: list[dict]) -> bool:
        """Whether the template's render alone stands for ``messages`` in the cache: no call, result or turn kept as generated among them."""
        return not any(m["role"] == "tool" or m.get("tool_calls") or GENERATED_TURN in m
                       for m in messages)

    # ----- tags ---------------------------------------------------------------------
    def tag_after_content(self, split: SplitRender, index: int) -> int | None:
        """Where the first tag after message ``index``'s content begins, before the next text of the caller's own."""
        content = split.contents[index]
        if content is None:
            return None
        return self.tag_from(split.text, content[1], split)

    def tag_from(self, text: str, at: int, split: SplitRender | None = None) -> int | None:
        """Where the first tag at or after ``at`` begins, before the next own text of ``split`` where given."""
        end = next((s for s, _ in split.own if s >= at), len(text)) if split else len(text)
        pieces = split_at_tags(self._reader, text[at:end])
        if pieces is None:
            return None
        for piece, is_tag in pieces:
            if is_tag:
                return at
            at += len(piece)
        return None

    def run_start(self, split: SplitRender | None, index: int, start: int | None) -> int | None:
        """Where the run of text holding message ``index``'s content begins: after the last tag from ``start`` on."""
        content = split.contents[index] if split is not None else None
        if split is None or start is None or content is None:
            return None
        run = None
        for piece, own in split.pieces(start, content[0]):
            if own:
                start += len(piece)
                continue
            pieces = split_at_tags(self._reader, piece)
            if pieces is None:
                return None
            for part, is_tag in pieces:
                start += len(part)
                if is_tag:
                    run = start
        return run


def _without_kept_texts(message: dict) -> dict:
    """``message`` without what the conversation keeps beside it for itself: a generated turn, and the mark of a call that could not be read."""
    calls = message.get("tool_calls") or ()
    if GENERATED_TURN not in message and not any(UNREADABLE_CALL in c for c in calls):
        return message
    message = {key: value for key, value in message.items() if key != GENERATED_TURN}
    if calls:
        message["tool_calls"] = [{key: value for key, value in call.items() if key != UNREADABLE_CALL}
                                 for call in calls]
    return message
