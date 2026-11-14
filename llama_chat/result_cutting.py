"""Cutting a tool result, or an earlier one of the same round, to the room a reply leaves, the cut told to the model in a note.

:class:`ResultCutting` holds no state of its own: each method is given the
conversation's state, measured with the rendering it was made with.
"""

from __future__ import annotations

from .conversation_state import ConversationState
from .messages import with_result_text
from .tools import ToolCall

# What a result cut to fit the room left ends with, so that the model reads how much of it it got.
RESULT_CUT_NOTE = "\n(cut: {kept} of {total} characters kept)"


def cut_result(text: str, kept: int) -> str:
    """``text`` cut to its first ``kept`` characters, :data:`RESULT_CUT_NOTE` after them."""
    return text[:kept] + RESULT_CUT_NOTE.format(kept=kept, total=len(text))


class ResultCutting:
    """Measures and cuts the results of the conversation state it is given."""

    def __init__(self, rendering) -> None:
        self._rendering = rendering

    def fits(self, state: ConversationState, at: int, content: str, room: int) -> bool:
        """Whether the conversation, its message at ``at``, a result, holding ``content``, fits in ``room`` with the generation prompt."""
        state.messages[at] = {**state.messages[at], "content": content}
        return self._rendering.length(state.messages, generation_prompt=True) <= room

    def longest_fitting(self, state: ConversationState, at: int, text: str, room: int) -> int:
        """How many characters of ``text`` the result at ``at`` keeps, cut, within ``room``."""
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if self.fits(state, at, cut_result(text, middle), room) \
                else (low, middle - 1)
        return low

    def cut_an_earlier_result(self, state: ConversationState, at: int, content: str,
                              room: int) -> tuple[ToolCall, int, int] | None:
        """Cut a result written whole earlier in this round, the newest one that makes room for the one at ``at`` holding ``content``; its call, and how many of its characters it keeps of how many, or ``None`` where none makes room."""
        messages = state.messages
        calling = max(i for i in range(at) if messages[i].get("tool_calls"))
        cut_ids = {call.id for call in state.results_cut}
        for j in range(at - 1, calling, -1):
            earlier = messages[j]
            if earlier["role"] != "tool" or earlier["tool_call_id"] in cut_ids:
                continue
            whole = earlier["content"]
            messages[j] = with_result_text(earlier, cut_result(whole, 0))
            if not self.fits(state, at, content, room):
                messages[j] = earlier
                continue
            low, high = 0, len(whole)
            while low < high:
                middle = (low + high + 1) // 2
                messages[j] = with_result_text(earlier, cut_result(whole, middle))
                low, high = (middle, high) if self.fits(state, at, content, room) \
                    else (low, middle - 1)
            messages[j] = with_result_text(earlier, cut_result(whole, low))
            call, = [c for c in messages[calling]["tool_calls"]
                     if c["id"] == earlier["tool_call_id"]]
            return (ToolCall(call["function"]["name"], call["function"]["arguments"], call["id"]),
                    low, len(whole))
        return None
