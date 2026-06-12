"""llama_chat — a KV-cache managing chat wrapper around llama.cpp.

Three actions, each prefilling as little as possible:

* :meth:`ChatWrapper.begin` — reset and prefill the system prompt + history (optional).
* :meth:`ChatWrapper.inject` — prefill one message as context, no generation.
* :meth:`ChatWrapper.request` — prefill the request text, then generate.

When the cache crosses the threshold, the oldest non-system messages are removed
and the survivors shifted down to reuse their KV without re-prefilling.
"""

from .config import Config
from .context import KVContext
from .messages import Eviction, Message, MessageTable, fit_newest_first
from .persistence import ConversationStore, InMemoryStore, PersistentChat
from .template import TemplateFormatter
from .wrapper import ChatWrapper, ContextOverflowError, Turn

__all__ = [
    "Config",
    "ChatWrapper",
    "ContextOverflowError",
    "KVContext",
    "Turn",
    "Message",
    "MessageTable",
    "Eviction",
    "TemplateFormatter",
    "fit_newest_first",
    "PersistentChat",
    "ConversationStore",
    "InMemoryStore",
]
