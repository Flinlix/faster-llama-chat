"""llama_chat: a KV-cache managing chat wrapper around llama.cpp.

The core actions, each prefilling as little as possible:

* :meth:`ChatWrapper.begin`: reset and prefill the system prompt and history.
* :meth:`ChatWrapper.inject`: prefill one message as context, no generation.
* :meth:`ChatWrapper.reply_without_tools`: prefill the request text, then
  generate.
* :meth:`ChatWrapper.reply_without_tools_stream`: the same, yielding text
  deltas.
* :meth:`ChatWrapper.truncate_last_reply`: shorten the newest reply to what was
  read or heard of it.

Crossing either eviction limit removes the oldest non-system messages and shifts
the survivors down, reusing their KV without re-prefilling.

A model whose own template describes tool calls can use them, the tags and the
argument spelling being recovered from that template:

* :meth:`ChatWrapper.reply_with_tools`: the finished reply, tools run on the
  way.
* :meth:`ChatWrapper.reply_with_tools_stream`: the same loop, yielding text
  deltas.
* :meth:`ChatWrapper.reply_until_tool` and
  :meth:`ChatWrapper.resume_until_tool`: the same exchange driven by the caller.
* :meth:`ChatWrapper.declare_unknown_tools`: tell the model about a tool it has
  not been told about yet.
"""

from .config import Config, Sampling
from .context import KVContext
from .messages import Eviction, Message, MessageTable, n_newest_that_fit
from .persistent_wrapper import ConversationStore, InMemoryStore, PersistentChat
from .template import CallBody, TemplateFragments, TemplateFormatter, ToolCallFragments
from .tools import ToolCall, ToolRegistry, ToolResult, ToolSpec, ToolTurn, tool
from .wrapper import ChatWrapper, ContextOverflowError, Reply, ReplyStream

__all__ = [
    "Config",
    "Sampling",
    "ChatWrapper",
    "ContextOverflowError",
    "Reply",
    "ReplyStream",
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
    "ToolResult",
    "ToolTurn",
    "ToolSpec",
    "ToolRegistry",
    "tool",
    "PersistentChat",
    "ConversationStore",
    "InMemoryStore",
]
