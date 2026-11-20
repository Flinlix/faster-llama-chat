"""What the cache should hold for a list of messages: their render, or the cache up to an anchor and the render of a window after it, where such a window stands for the whole."""

from __future__ import annotations

from dataclasses import dataclass

from .cache_sync import CacheSync
from .conversation_state import Anchor, ConversationState
from .messages import exchange_starts, system_part
from .rendering import ConversationRendering, Target, generated_turns_in
from .split_render import SplitRender


@dataclass(frozen=True)
class _View:
    """Messages as the cache holds them: rendered whole, or as a window of the newest after the cache's anchor.

    The cache holds ``base`` and then the tokens of ``split`` from ``start``
    on; ``messages`` is what was rendered, and the conversation's message
    ``i`` is its ``i - offset``-th, the system message aside.
    """

    split: SplitRender
    messages: list[dict]
    offset: int
    start: int
    base: list[int]
    base_turns_at: dict
    generation_prompt: bool


def repeats_before(messages: list[dict], start: int) -> bool:
    """Whether a question of ``messages`` from ``start`` on repeats the text of a message before it."""
    texts = {m["content"] for m in messages[start:]
             if m["role"] == "user" and isinstance(m.get("content"), str)}
    return any(m.get("content") in texts for m in messages[:start]
               if isinstance(m.get("content"), str))


def closes_turn(message: dict) -> bool:
    """Whether ``message`` is a reply that makes no call, which closes its turn."""
    return message["role"] == "assistant" and not message.get("tool_calls")


class CacheTargets:
    """The target the cache is brought to for a list of messages, and its length, read from the conversation's state and the cache."""

    def __init__(self, rendering: ConversationRendering, state: ConversationState,
                 memory: CacheSync) -> None:
        self._rendering = rendering
        self._state = state
        self._memory = memory

    def window_of(self, messages: list[dict]) -> tuple[int, Anchor] | None:
        """Where a window of ``messages`` begins that stands for their render after one of the cache's anchors, the newest that still stands, and that anchor; ``None`` where no window does.

        The window opens with the exchange the anchor's message ends, and
        holds at most three more; that message is a reply that closes its
        turn and a question follows it, as in the probes that showed windows
        to stand for the whole render, so that nothing a turn left open
        carries past it and nothing after it writes its closing otherwise.
        """
        if not self._state.checks.windows:
            return None
        for anchor in reversed(self._state.anchors):
            found = self.window_after(messages, anchor)
            if found is not None:
                return found, anchor
        return None

    def window_after(self, messages: list[dict], anchor: Anchor) -> int | None:
        """Where a window of ``messages`` after ``anchor`` begins, where one stands for their render."""
        if anchor.index >= len(messages) or messages[anchor.index] is not anchor.message \
                or len(self._memory.tokens) <= anchor.at:
            return None
        starts = exchange_starts(messages)
        opening = [i for i, at in enumerate(starts) if at <= anchor.index]
        if not opening or len(starts) - opening[-1] > 4:
            return None
        start = starts[opening[-1]]
        # As in the probes: a reply that closes its turn, a question after it. Whatever else
        # follows the reply may write its closing otherwise.
        if not closes_turn(messages[anchor.index]) or anchor.index + 1 >= len(messages) \
                or messages[anchor.index + 1]["role"] != "user":
            return None
        if not self._state.checks.tool_windows \
                and not self._rendering.template_writes(messages[start:]):
            return None
        # A template may write a message otherwise where a later one repeats its text, which
        # only a render of the whole conversation shows: now, or when the cache was written.
        if not self._state.checks.repeats_alike and (anchor.written_under_a_repeat
                                                     or repeats_before(messages, anchor.index + 1)):
            return None
        kept = tuple(turn for _, _, turn in generated_turns_in(messages[:anchor.index + 1]))
        if len(kept) != len(anchor.kept) or any(a is not b for a, b in zip(kept, anchor.kept)):
            return None
        return start

    def view(self, messages: list[dict], generation_prompt: bool) -> _View | None:
        """``messages`` rendered as the cache holds them: a window of the newest of them after the newest of the cache's anchors that still opens with the tag the cache holds there, where one stands for the whole, else all of them; ``None`` where the template refuses them."""
        for anchor in reversed(self._state.anchors if self._state.checks.windows else ()):
            start = self.window_after(messages, anchor)
            if start is None:
                continue
            system = system_part(messages)
            window = system + messages[start:]
            split = self._rendering.split(window, generation_prompt)
            tag = (None if split is None
                   else self._rendering.tag_after_content(split, len(system) + anchor.index - start))
            if tag is not None and split is not None:
                view = _View(split, window, start - len(system), tag,
                             list(self._memory.tokens[:anchor.at]),
                             {k: at for k, at in self._state.turns_at.items() if at < anchor.at},
                             generation_prompt)
                # The tag the window opens with is the one the cache holds there, unless
                # what follows the anchor's message is written otherwise now.
                opening = self._rendering.target(window, generation_prompt, start=tag).tokens[:1]
                if opening == list(self._memory.tokens[anchor.at:anchor.at + 1]):
                    return view
        split = self._rendering.split(messages, generation_prompt)
        return None if split is None else _View(split, messages, 0, 0, [], {}, generation_prompt)

    def view_target(self, view: _View, upto: int | None = None) -> Target:
        """What the cache should hold for the messages ``view`` stands for, up to ``upto`` in its render, with where the newest exchange of what it renders opens."""
        rendering, base = self._rendering, len(view.base)
        found = rendering.target(view.messages, view.generation_prompt, upto=upto, start=view.start)

        def at(tag: int) -> int:
            return base + len(rendering.target(view.messages, view.generation_prompt, upto=tag,
                                               start=view.start).tokens)

        starts = [s for s in exchange_starts(view.messages) if s > len(system_part(view.messages))]
        end = len(view.split.text) if upto is None else upto
        tags = [(s - 1, rendering.tag_after_content(view.split, s - 1)) for s in starts[-2:]]
        known = [(i, tag) for i, tag in tags if tag is not None and view.start <= tag <= end]
        anchors = tuple((i + view.offset, at(tag)) for i, tag in known)
        # The exchanges of the conversation, not one a stand-in after it opens.
        conversation = [(i, tag) for i, tag in known
                        if i + 1 + view.offset < len(self._state.messages)]
        return Target(view.base + found.tokens,
                      {**view.base_turns_at, **{k: base + p for k, p in found.turns_at.items()}},
                      anchors, tuple(at(tag) for _, tag in conversation[-2:]))

    def target_of(self, messages: list[dict], generation_prompt: bool) -> Target:
        """What the cache should hold for ``messages``, from a window of the newest of them where one stands for the whole.

        Raises:
            ValueError: As :meth:`ConversationRendering.target` does.
        """
        view = self.view(messages, generation_prompt)
        if view is None:
            return self._rendering.target(messages, generation_prompt)
        return self.view_target(view)

    def length(self, messages: list[dict], generation_prompt: bool) -> int:
        """How many tokens the cache holds for ``messages``, as :meth:`ConversationRendering.length` counts them, from a window of the newest of them where one stands for the whole."""
        if self.window_of(messages) is not None:
            view = self.view(messages, generation_prompt)
            if view is not None and view.base:
                return len(self.view_target(view).tokens)
        return self._rendering.length(messages, generation_prompt)

    def newest_part(self, messages: list[dict]) -> list[dict]:
        """The system message and the window of the newest of ``messages`` that stands for their render, or all of them where none does; what a refusal of the newest is read from."""
        found = self.window_of(messages)
        return messages if found is None else system_part(messages) + messages[found[0]:]
