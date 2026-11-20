"""Every Python example in the README and the guides runs as written, on the stand-in model in place of a model file.

The blocks of one document run in order in one namespace, as a reader would
run them; a model file the example names loads the stand-in instead, and a
block that uses ``chat`` without making it finds one begun.
"""

import asyncio
import logging
import re
from pathlib import Path

import pytest

import llama_cpp_conversation
from llama_cpp_conversation import Config
from llama_cpp_conversation.testing.fake_context import FakeContext, written_call
from llama_cpp_conversation.testing.invented_templates import PLAIN, REASONING, TOOLS

_ROOT = Path(__file__).parent.parent
_DOCUMENTS = ["README.md"] + sorted(str(p.relative_to(_ROOT)) for p in (_ROOT / "docs/guide").glob("*.md"))
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
# The template each document's model writes with, and the calls it makes in each example.
_TEMPLATES = {"docs/guide/tool-calling.md": TOOLS, "docs/guide/reasoning.md": REASONING}
_CALLS = {"docs/guide/tool-calling.md": [[("get_weather", {"city": "Munich"})]] * 2}
# What runs a document's examples that only define something.
_AFTERWARDS = {"docs/guide/streams-and-threads.md": """
async def _read():
    assert (await ask(chat, "Hi")).strip()
    assert "".join([text async for text in stream(chat, "Again")]).strip()
    early = stream(chat, "Once more")
    await early.__anext__()
    await early.aclose()
asyncio.run(_read())
"""}
_ANSWERS = ["Hello there.", "The deadline is Friday.", "A short plan.", "It is mild.",
            "Rayleigh scattering."]


def _examples(document: str) -> list[str]:
    return _BLOCK.findall((_ROOT / document).read_text())


def _script(model: FakeContext, calls: list) -> None:
    """Have ``model`` make ``calls``, each a name and its arguments, in place of its next replies."""
    model.call_script = [written_call(None, model, name, arguments)[1] for name, arguments in calls]


@pytest.mark.parametrize("document", [d for d in _DOCUMENTS if _examples(d)])
def test_every_example_of_a_document_runs(document, monkeypatch, capsys):
    real = llama_cpp_conversation.ChatWrapper
    models = []

    calls: list = []                    # the calls the current example's model makes

    def stand_in(config=None, context=None, **options):
        models.append(FakeContext(_TEMPLATES.get(document, PLAIN), list(_ANSWERS)))
        _script(models[-1], calls)
        return real(config or Config(), context=context or models[-1], **options)

    monkeypatch.setattr(llama_cpp_conversation, "ChatWrapper", stand_in)
    chat = stand_in(Config(context_size=2048))
    chat.begin([])
    levels = {name: logging.getLogger(name).level for name in ("llama_cpp_conversation", "llama_cpp_conversation.engine")}
    handlers = list(logging.getLogger().handlers)
    namespace = {"chat": chat, "asyncio": asyncio}
    try:
        scripted = _CALLS.get(document, [])
        for i, block in enumerate(_examples(document) + [_AFTERWARDS.get(document, "")]):
            calls[:] = scripted[i] if i < len(scripted) else []
            _script(models[-1], calls)
            exec(compile(block, document, "exec"), namespace)
            assert not models[-1].call_script, f"example {i + 1} of {document} made no call its model made"
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
        logging.getLogger().handlers[:] = handlers
        for made in {chat, namespace.get("chat")}:
            made.close()
