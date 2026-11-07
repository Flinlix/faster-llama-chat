"""The library's modules name no model, family or vendor."""

import json
import re
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).parent.parent / "llama_chat"
_ENGINE_MODULE, _ENGINE_WORDS = "_backend.py", {"llama", "llama-cpp-python"}
_KNOWN_FILES = [*(Path(__file__).parent / "templates").glob("*.json"),
                *(_PACKAGE / "addons").glob("*.json")]


def _families() -> set[str]:
    """The leading letters of each known file's name, such as ``qwen`` for ``qwen3-coder``."""
    return {re.match("[a-z]+", path.stem.lower()).group() for path in _KNOWN_FILES}


def _vendors() -> set[str]:
    """The publisher each known file's ``source`` names before its slash."""
    sources = (json.loads(path.read_text()).get("source", "") for path in _KNOWN_FILES)
    return {source.split("/")[0].lower() for source in sources if "/" in source}


def _names_a_model(word: str, families: set[str], vendors: set[str]) -> bool:
    if word in families or word in vendors:
        return True
    return any(re.fullmatch(re.escape(family) + r"(\d[\d.]*|-.+)", word) for family in families)


@pytest.mark.parametrize("module", sorted(_PACKAGE.rglob("*.py")),
                         ids=lambda path: str(path.relative_to(_PACKAGE)))
def test_a_module_names_no_model_family_or_vendor(module):
    families, vendors = _families(), _vendors()
    words = set(re.findall(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*", module.read_text().lower()))
    if module.name == _ENGINE_MODULE:
        words -= _ENGINE_WORDS                  # the engine's own module may name the engine
    assert not sorted(word for word in words if _names_a_model(word, families, vendors))


@pytest.mark.parametrize("word", ["qwen", "qwen3", "qwen2.5", "gpt-oss", "llama-3", "openai"])
def test_a_model_family_or_vendor_name_is_recognized(word):
    assert _names_a_model(word, _families(), _vendors())


@pytest.mark.parametrize("word", ["llama_chat", "llama.cpp", "llama_cpp", "sampling"])
def test_the_library_and_its_engine_are_not_taken_for_a_model(word):
    assert not _names_a_model(word, _families(), _vendors())
