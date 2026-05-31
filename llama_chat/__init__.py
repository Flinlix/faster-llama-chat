"""llama_chat — a KV-cache scalpel chat wrapper around llama.cpp.

Three actions, each prefilling as little as possible:

* :meth:`ChatWrapper.begin` — reset and prefill the system prompt + recent history.
* :meth:`ChatWrapper.inject` — prefill one message as context, no generation.
* :meth:`ChatWrapper.request` — prefill the request text, then generate.

When the cache crosses the threshold, the oldest non-system messages are removed
and the survivors shifted down to reuse their KV without re-prefilling.
"""

from .config import Config, TemplateConfig
from .messages import Eviction, Message, MessageTable, fit_newest_first
from .template import TemplateFormatter
from .wrapper import ChatWrapper, Turn, detect_template_name

__all__ = [
    "Config",
    "TemplateConfig",
    "ChatWrapper",
    "Turn",
    "Message",
    "MessageTable",
    "Eviction",
    "TemplateFormatter",
    "fit_newest_first",
    "detect_template_name",
]

# Note: KVContext and Generation live in llama_chat.context (they pull in the
# llama.cpp backend lazily); import them from there when needed.

# Note: KVContext (the llama.cpp-backed implementation) is intentionally not
# imported here so that the pure bookkeeping layer can be used and tested
# without llama-cpp-python installed. Import it explicitly when you need it:
#     from llama_chat.context import KVContext
