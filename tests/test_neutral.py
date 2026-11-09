"""The library's modules name no model, family or vendor."""

import json
import re
from pathlib import Path

import pytest

from .chat_templates import STEMS, fixture
from .fake_context import TAG

_PACKAGE = Path(__file__).parent.parent / "llama_chat"
_ENGINE_MODULE, _ENGINE_WORDS = "_backend.py", {"llama", "llama-cpp-python"}
_ADDONS = sorted((_PACKAGE / "addons").glob("*.json"))


def _families() -> set[str]:
    """The leading letters of each known file's name, such as ``qwen`` for ``qwen3-coder``."""
    return {re.match("[a-z]+", name.lower()).group()
            for name in [*STEMS, *(path.stem for path in _ADDONS)]}


# Publishers and chat formats the known files do not spell out themselves.
_NAMED = {"meta", "ibm", "microsoft", "alibaba", "moonshot", "zhipu", "google", "openai",
          "mistralai", "nvidia", "tencent", "cohere", "deepseek", "chatml", "hermes", "harmony"}


def _vendors() -> set[str]:
    """The publisher each known file's ``source`` names before its slash, and the names above."""
    sources = [*(fixture(stem).get("source", "") for stem in STEMS),
               *(json.loads(path.read_text()).get("source", "") for path in _ADDONS)]
    return {source.split("/")[0].lower() for source in sources if "/" in source} | _NAMED


# The library's and its engine's own names, which hold a family's letters.
_OWN_NAMES = {"llama_chat", "llama.cpp", "llama_cpp", "llama-cpp-python"}


def _parts(word: str) -> list[str]:
    """``word`` whole and in its parts: split at ``_``, between letters and digits, and before capitals."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])|_", " ", word)
    return [word.lower()] + spaced.lower().split()


# Publishers whose name is also an everyday word, which as a part of a name means the word.
_WORDS_FIRST = {"meta"}


def _names_a_model(word: str, families: set[str], vendors: set[str]) -> bool:
    if word.lower() in _OWN_NAMES:
        return False
    whole, *parts = _parts(word)
    if whole in vendors:
        return True
    for part in [whole, *parts]:
        if part in families or (part in vendors and part not in _WORDS_FIRST):
            return True
        if any(re.fullmatch(re.escape(family) + r"(\d[\d.]*|-.+)", part) for family in families):
            return True
    return False


@pytest.mark.parametrize("module", sorted(_PACKAGE.rglob("*.py")),
                         ids=lambda path: str(path.relative_to(_PACKAGE)))
def test_a_module_names_no_model_family_or_vendor(module):
    families, vendors = _families(), _vendors()
    text = re.sub(r"(?i)llama_chat|llama[._-]cpp(?:-python)?", " ", module.read_text())
    words = set(re.findall(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*", text))
    if module.name == _ENGINE_MODULE:      # the engine's own module may name the engine
        words = {w for w in words if w.lower() not in _ENGINE_WORDS
                 and not w.lower().startswith("llama_")}
    assert not sorted(word for word in words if _names_a_model(word, families, vendors))


@pytest.mark.parametrize("word", ["qwen", "qwen3", "qwen2.5", "gpt-oss", "llama-3", "openai",
                                  "is_qwen", "qwen_tags", "deepseek_v3", "qwen3coder",
                                  "QwenTags", "harmony_format"])
def test_a_model_family_or_vendor_name_is_recognized(word):
    assert _names_a_model(word, _families(), _vendors())


@pytest.mark.parametrize("word", ["llama_chat", "llama.cpp", "llama_cpp", "sampling",
                                  "max_batch_tokens", "ThoughtFilter"])
def test_the_library_and_its_engine_are_not_taken_for_a_model(word):
    assert not _names_a_model(word, _families(), _vendors())


def _template_tags() -> set[str]:
    return {tag for stem in STEMS for tag in TAG.findall(fixture(stem)["chat_template"])}


@pytest.mark.parametrize("module", sorted(_PACKAGE.rglob("*.py")),
                         ids=lambda path: str(path.relative_to(_PACKAGE)))
def test_a_module_writes_no_tag_of_a_real_template(module):
    text = module.read_text()
    assert not sorted(tag for tag in _template_tags() if len(tag) > 3 and tag in text)


def test_the_test_doubles_tags_are_no_real_templates():
    from .fake_context import DEFAULT_FRAGMENTS, EMPTY_THOUGHT, SYSTEM_PREFIX, SYSTEM_SUFFIX, THINK_ON
    invented = set(re.findall(r"<[^<>\s]+>", "".join(
        [SYSTEM_PREFIX, SYSTEM_SUFFIX,
         DEFAULT_FRAGMENTS.assistant_prefix, EMPTY_THOUGHT, THINK_ON])))
    assert invented and not invented & _template_tags()
