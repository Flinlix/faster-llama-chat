"""Start a test template's file from a model file: the keys the model file tells, written to ``tests/templates/STEM.json``.

Usage: ``python -m tests.new_template MODEL.gguf SOURCE STEM``, where ``SOURCE``
is where the model file was published, such as its repository's name. Reads
the vocabulary and the metadata alone, so it loads no weights. The keys the
template's behaviour decides are written as a first guess, and the tests of
``tests/test_real_templates.py`` say which to change; CONTRIBUTING.md says how.
"""

import json
import sys
from pathlib import Path

from llama_cpp_conversation._backend import Backend
from llama_cpp_conversation.cache_context import TOOL_USE_TEMPLATE

_TEMPLATES = Path(__file__).parent / "templates"


def told_by(model_path: str, source: str) -> dict:
    """The keys of a test template's file the model file at ``model_path`` tells, and a first guess at the others."""
    from llama_cpp import Llama

    model = Llama(model_path=model_path, vocab_only=True, verbose=False)
    backend = Backend()
    vocab = backend.vocab(model._model.model)
    template = model.metadata.get("tokenizer.chat_template")
    if not template:
        raise SystemExit(f"{model_path} ships no chat template")
    tools = model.metadata.get(f"tokenizer.chat_template.{TOOL_USE_TEMPLATE}")
    written = template + (tools or "")
    eos = backend.eos_text(vocab)
    tags = {backend.token_to_piece(vocab, token) for token in backend.special_tokens(vocab)}
    known = {
        "source": source,
        "chat_template": template,
        "eos_token": eos,
        "render_variables": {"off": {}, "on": {}},
        "marks_thought": False,
        "can_reason": False,
        "calls_tools": "tools" in written,
        "tags": sorted(tag for tag in tags if tag and (tag in written or tag == eos)),
    }
    if tools:
        known["tool_use_template"] = tools
    return known


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    model_path, source, stem = sys.argv[1:]
    path = _TEMPLATES / f"{stem}.json"
    if path.exists():
        raise SystemExit(f"{path} exists already")
    path.write_text(json.dumps(told_by(model_path, source), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}; now run: python -m pytest -q tests/test_real_templates.py -k {stem}")
