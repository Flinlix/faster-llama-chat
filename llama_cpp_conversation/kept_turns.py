"""The turns kept as the tokens they were generated as: which ones a begin may keep, where they stand, those the template no longer places given up, and how much of one a shortened reply keeps."""

from __future__ import annotations

from collections.abc import Callable

from .conversation_state import ConversationState
from .history_forms import intact_records
from .messages import GENERATED_TURN, system_part
from .rendering import ConversationRendering, GeneratedTurn, generated_turns_in, with_generated_turn
from .resolved_template import ResolvedTemplate
from .loop_variant import LoopVariant
from .cache_targets import CacheTargets


def _text_starts_at(pieces: list[bytes], tags: list[bool], said: bytes) -> int | None:
    """Where in the joined ``pieces`` the text ``said`` begins that they end on, the pieces marked as ``tags`` counting as no text; ``None`` where they do not end on it.

    The place may fall inside a piece, where one token holds the end of what
    comes before the text and its first letters.
    """
    text, at = b"", sum(map(len, pieces))
    for piece, tag in zip(reversed(pieces), reversed(tags)):
        if len(text) >= len(said):
            break
        at -= len(piece)
        if not tag:
            text = piece + text
    return at + len(text) - len(said) if text.endswith(said) else None


def _decodes(data: bytes) -> bool:
    """Whether ``data`` is whole UTF-8 text."""
    try:
        data.decode()
    except UnicodeDecodeError:
        return False
    return True


class KeptTurns:
    """Reads and gives up the conversation's turns kept as generated."""

    def __init__(self, context, rendering: ConversationRendering, targets: CacheTargets, state: ConversationState, template: Callable[[], ResolvedTemplate], turn_open: Callable[[], bool]) -> None:
        self._context = context
        self._rendering = rendering
        self._targets = targets
        self._state = state
        self._template = template
        self._turn_open = turn_open

    def usable(self, generated_turns: dict | None) -> list[dict]:
        """The generated turns of ``generated_turns`` whose tokens are this model's, every one of them; none where they are another model's."""
        if not generated_turns:
            return []
        if generated_turns["model"] != self._context.model_fingerprint():
            self._state.changes.generated_turns_ignored = True
            return []
        intact = intact_records(generated_turns["turns"])
        if len(intact) < len(generated_turns["turns"]):
            # Tokens changed since they were given out are no tokens this model generated.
            self._state.changes.generated_turns_ignored = True
        return intact

    def held_indexes(self) -> set[int]:
        """Where the messages a generated turn's tokens stand for are."""
        return {i for first, last, _ in generated_turns_in(self._state.messages)
                for i in range(first, last + 1)}

    def open_turn_in_thought(self) -> bool:
        """Whether the open turn's newest call, kept as generated, was made inside a thought the model had not closed, as its tokens read."""
        tags, messages = self._template().properties.reasoning, self._state.messages
        calling = next((m for m in reversed(messages) if m.get("tool_calls")), None)
        turn = calling.get(GENERATED_TURN) if calling is not None else None
        if tags is None or turn is None or not turn.on_call or not self._turn_open() \
                or not tags.thought_open.strip():
            return False
        text = "".join(self._context.token_text(token) for token in turn.tokens)
        return text.rfind(tags.thought_open.strip()) > text.rfind(tags.thought_close.strip())

    def open_generated_turn(self) -> tuple[int, GeneratedTurn] | None:
        """The generated turn the open reply goes on with, and where its last message stands: the one ending on its newest call, placed in the cache."""
        messages = self._state.messages
        at = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("tool_calls")),
                  None)
        turn = messages[at].get(GENERATED_TURN) if at is not None else None
        if at is None or turn is None or not turn.on_call or turn not in self._state.turns_at:
            return None
        return at, turn

    def forget_calls(self) -> None:
        """Record the newest reply without the call it made, which could be told nothing; its tokens stay as generated."""
        newest = {k: v for k, v in self._state.messages[-1].items() if k != "tool_calls"}
        turn = newest.get(GENERATED_TURN)
        if turn is not None:
            newest[GENERATED_TURN] = GeneratedTurn(turn.tokens, turn.span, False)
        self._state.messages[-1] = newest
        self._state.turns_at = {}
        self.drop_unplaceable()

    def drop_unplaceable(self) -> bool:
        """Hold each generated turn the template writes where it cannot be placed as the template writes it instead; whether any was.

        Where a window of the newest exchanges stands for the render, only
        the turns it holds are placed: those before it stay as they were.
        """
        messages = self._state.messages
        if not generated_turns_in(messages):
            return False
        found = self._targets.window_of(messages)
        system = system_part(messages)
        rendered = messages if found is None else system + messages[found[0]:]
        offset = 0 if found is None else found[0] - len(system)
        # In the form the cache holds them in: with the generation prompt where a reply follows.
        split = self._rendering.split(rendered, messages[-1]["role"] in ("user", "tool"))
        unplaced = set() if split is None else {
            last + offset for first, last, turn in generated_turns_in(rendered)
            if self._rendering.place(first, last, turn.on_call, split) is None}
        for last in unplaced:
            messages[last] = with_generated_turn(messages[last], None)
        held = {turn for _, _, turn in generated_turns_in(messages)}
        self._state.turns_at = {k: at for k, at in self._state.turns_at.items() if k in held}
        return bool(unplaced)

    def generated_up_to(self, turn: GeneratedTurn, text: str, keep_text: str) -> GeneratedTurn | None:
        """``turn``, ending on a reply's ``text``, as its tokens hold the reply cut to ``keep_text``: its generated tokens up to the last whole one before the cut, the rest of ``keep_text`` tokenized; ``None`` where its tokens do not end on ``text``.

        A tag the model wrote inside the text is no text, as the reply's text
        drops it: it is kept where it stands before the cut and goes with the
        rest. Where the text spells a tag itself, the tokens are read as
        generated.
        """
        spelled = [self._context.token_bytes(token) for token in turn.tokens]
        said = text.encode()
        # The text without the tags, as a reply's text drops one inside it, else as generated.
        pieces = [b"" if self._context.is_special(token) else piece
                  for token, piece in zip(turn.tokens, spelled)]
        if not b"".join(pieces).endswith(said):
            pieces = spelled
        whole = b"".join(pieces)
        if not whole.endswith(said):
            return None
        end = len(whole) - len(said) + len(keep_text.encode())
        n, at = len(pieces), len(whole)
        backing_variant = LoopVariant("finding the last whole token")
        while n and (at > end or not _decodes(whole[at:end]) or (at == end and not pieces[n - 1])):
            backing_variant.passes(n)
            n -= 1
            at -= len(pieces[n])
        rest = whole[at:end].decode()
        tokens = turn.tokens[:n] + tuple(self._rendering.tokenize([(rest, True)]) if rest else ())
        return GeneratedTurn(tokens, turn.span, turn.on_call)

    def generated_to_its_call(self, turn: GeneratedTurn, text: str, span: int) -> GeneratedTurn | None:
        """``turn``, ending the newest message, a reply of ``text``, as the turn of its first ``span`` messages, ending on its newest call: its generated tokens up to that call's closing; ``None`` where they cannot be told.

        The closing is the one right before the template's text from it to the end of
        the newest result's content, the results that follow the call, since a
        result's own closing may read the same.
        """
        messages = self._state.messages
        calling = len(messages) - turn.span + span - 1
        split = self._rendering.split(messages, generation_prompt=False)
        if split is None:
            return None
        closed, result = self._rendering.call_closed_at(split, calling), split.contents[-2]
        if closed is None or result is None or result[1] < closed:
            return None
        pieces = [self._context.token_bytes(token) for token in turn.tokens]
        whole = b"".join(pieces)
        closing = self._template().properties.call_close_text.encode()
        said_at = _text_starts_at(pieces, [self._context.is_special(token) for token in turn.tokens],
                                  text.encode())
        if said_at is None and whole.endswith(text.encode()):
            said_at = len(whole) - len(text.encode())       # a tag the text spells as written
        if said_at is None:
            return None
        end = whole.rfind(split.text[closed:result[1]].encode(), 0, said_at)
        if end < 0 or not whole[:end].endswith(closing):
            return None
        at = 0
        for n, piece in enumerate(pieces):
            if at == end:
                return GeneratedTurn(turn.tokens[:n], span, True)
            at += len(piece)
        return None
