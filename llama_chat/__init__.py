"""llama_chat: a KV-cache managing chat wrapper around llama.cpp.

The core actions, each prefilling as little as possible:

* :meth:`ChatWrapper.begin`: reset and prefill the system prompt and history.
* :meth:`ChatWrapper.inject`: prefill one message as context, no generation.
* :meth:`ChatWrapper.generate`: prefill the request text, then generate.
* :meth:`ChatWrapper.generate_stream`: as ``generate``, yielding text deltas.

Crossing either eviction limit removes the oldest non-system messages and shifts
the survivors down, reusing their KV without re-prefilling.

A model whose own template describes tool calls can use them, the tags and the
argument spelling being recovered from that template:

* :meth:`ChatWrapper.answer`: the finished reply, tools run on the way.
* :meth:`ChatWrapper.answer_stream`: the same loop, yielding text deltas.
* :meth:`ChatWrapper.generate_events` and :meth:`ChatWrapper.resume_events`:
  the same exchange driven by the caller.
* :meth:`ChatWrapper.declare_tools`: declare a tool ``begin`` was not given.
"""

from .config import Config
from .context import KVContext
from .messages import Eviction, Message, MessageTable, n_newest_that_fit
from .persistent_wrapper import ConversationStore, InMemoryStore, PersistentChat
from .template import CallBody, TemplateFragments, TemplateFormatter, ToolCallFragments
from .tools import ToolCall, ToolRegistry, ToolSpec, tool
from .wrapper import ChatWrapper, ContextOverflowError, Reply

__all__ = [
    "Config",
    "ChatWrapper",
    "ContextOverflowError",
    "Reply",
    "KVContext",
    "Message",
    "MessageTable",
    "Eviction",
    "n_newest_that_fit",
    "TemplateFragments",
    "TemplateFormatter",
    "ToolCallFragments",
    "CallBody",
    "ToolCall",
    "ToolSpec",
    "ToolRegistry",
    "tool",
    "PersistentChat",
    "ConversationStore",
    "InMemoryStore",
]
