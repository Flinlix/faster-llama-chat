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


# Publishers and chat formats the known files do not spell out themselves.
_NAMED = {"meta", "ibm", "microsoft", "alibaba", "moonshot", "zhipu", "google", "openai",
          "mistralai", "nvidia", "tencent", "cohere", "deepseek", "chatml", "hermes", "harmony"}


def _vendors() -> set[str]:
    """The publisher each known file's ``source`` names before its slash, and the names above."""
    sources = (json.loads(path.read_text()).get("source", "") for path in _KNOWN_FILES)
    return {source.split("/")[0].lower() for source in sources if "/" in source} | _NAMED


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


def _template_tags() -> set[str]:
    from .template_contexts import TAG
    return {tag for path in (Path(__file__).parent / "templates").glob("*.json")
            for tag in TAG.findall(json.loads(path.read_text())["chat_template"])}


@pytest.mark.parametrize("module", sorted(_PACKAGE.rglob("*.py")),
                         ids=lambda path: str(path.relative_to(_PACKAGE)))
def test_a_module_writes_no_tag_of_a_real_template(module):
    text = module.read_text()
    assert not sorted(tag for tag in _template_tags() if len(tag) > 3 and tag in text)


def test_the_test_doubles_tags_are_no_real_templates():
    from .fake_context import DEFAULT_FRAGMENTS, EMPTY_THOUGHT, THINK_ON
    invented = set(re.findall(r"<[^<>\s]+>", "".join(
        [DEFAULT_FRAGMENTS.system_prefix, DEFAULT_FRAGMENTS.system_suffix,
         DEFAULT_FRAGMENTS.assistant_prefix, EMPTY_THOUGHT, THINK_ON])))
    assert invented and not invented & _template_tags()
