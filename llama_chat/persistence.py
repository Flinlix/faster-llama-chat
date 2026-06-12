"""Durable conversation history for :class:`~llama_chat.wrapper.ChatWrapper`.

The store is a *superset* that outlives KV eviction: it keeps every message except
the system prompt forever (user, assistant, and injected context), while the
wrapper's ``MessageTable`` holds only the recent window that fits ``n_ctx``.
:class:`PersistentChat` composes a ``ChatWrapper`` and uses its ``on_message`` hook
to capture each message, reloading prior history at ``begin``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, runtime_checkable

from .config import Config
from .wrapper import ChatWrapper


@runtime_checkable
class ConversationStore(Protocol):
    """Durable store of conversation turns, keyed by conversation id."""

    def load(self, conversation_id: str) -> list[tuple[str, str]]:
        """Return the conversation's ``(role, text)`` turns, oldest first."""
        ...

    def append(self, conversation_id: str, role: str, text: str) -> None:
        """Append one turn to the conversation's durable log."""
        ...


class InMemoryStore:
    """Reference :class:`ConversationStore` backed by a dict (not durable)."""

    def __init__(self) -> None:
        self._convos: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def load(self, conversation_id: str) -> list[tuple[str, str]]:
        return list(self._convos[conversation_id])

    def append(self, conversation_id: str, role: str, text: str) -> None:
        self._convos[conversation_id].append((role, text))


class PersistentChat:
    """A ``ChatWrapper`` whose user/assistant turns are persisted to a store.

    ``begin(conversation_id, system_prompt)`` selects the active conversation, loads
    its prior messages from the store, and prefills the recent ones that fit. Every
    subsequent ``request``/``stream``/``inject`` message is persisted through the
    wrapper's ``on_message`` hook. Only the system prompt is not persisted (it is
    supplied fresh on each ``begin``).
    """

    def __init__(self, store: ConversationStore, config: Config | None = None, **kwargs) -> None:
        self._store = store
        self._cid: str | None = None
        self._chat = ChatWrapper(config, on_message=self._persist, **kwargs)

    def _persist(self, role: str, text: str) -> None:
        assert self._cid is not None, "call begin() before persisting turns"
        self._store.append(self._cid, role, text)

    def begin(self, conversation_id: str, system_prompt: str) -> None:
        self._cid = conversation_id
        self._chat.begin(system_prompt, self._store.load(conversation_id))

    def request(self, text: str, **kwargs):
        return self._chat.request(text, **kwargs)

    def stream(self, text: str, **kwargs):
        return self._chat.stream(text, **kwargs)

    def inject(self, text: str, role: str = "user") -> int:
        return self._chat.inject(text, role)  # persisted via on_message

    def snapshot(self) -> list[dict]:
        return self._chat.snapshot()

    def close(self) -> None:
        self._chat.close()
