"""No file but a model file names a model, family or vendor, or writes a tag of a real template.

A model file is a test template or what its file says of it, in
``tests/templates/``, or an add-on file in ``llama_chat/addons/``. The other
exemptions are listed below.
"""

import functools
import json
import re
from pathlib import Path
from typing import NamedTuple

import pytest

from llama_chat.addon_files import template_fingerprint

from .chat_templates import STEMS, expected, fixture

_ROOT = Path(__file__).parent.parent
_PACKAGE = _ROOT / "llama_chat"
_ADDONS = sorted((_PACKAGE / "addons").glob("*.json"))

# Folders of model files, and files that may name a model: the design history, and the
# files that install the engine, which name the build for a GPU's maker.
_MODEL_FOLDERS = (Path(__file__).parent / "templates", _PACKAGE / "addons")
_EXEMPT = {_ROOT / "design_history.md", _ROOT / "install.sh", _ROOT / "pyproject.toml"}
# The README's install section, which names what gets installed, up to the next section.
_INSTALL_SECTION = re.compile(r"(?ms)^## Install$.*?(?=^## )")

# The library's and its engine's own names, which hold a family's letters.
_OWN_NAMES = ("faster-llama-chat", "llama-cpp-python", "llama_chat", "llama.cpp", "llama_cpp")
_OWN_NAME = re.compile("|".join(map(re.escape, _OWN_NAMES)), re.IGNORECASE)
# The engine's module and its tests, which name the engine's functions: the words its
# module's name begins with, alone or before an underscore.
_ENGINE_FILES = {_PACKAGE / "_backend.py", Path(__file__).parent / "test_backend.py"}
_ENGINE_PREFIX = "llama_cpp".split("_")[0]


def _checked_files() -> list[Path]:
    """Every file of the project that is not hidden, built, exempt or a model file."""
    def skipped(path):
        parts = path.relative_to(_ROOT).parts
        return (any(part.startswith(".") or part == "__pycache__" or part.endswith(".egg-info")
                    for part in parts)
                or path in _EXEMPT or any(folder in path.parents for folder in _MODEL_FOLDERS))
    return [path for path in sorted(_ROOT.rglob("*")) if path.is_file() and not skipped(path)]


def _checked_text(path: Path) -> str:
    """The text of ``path`` that must name no model, the README without its install section."""
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return ""                       # not text, so no name a reader sees
    if path == _ROOT / "README.md":
        assert _INSTALL_SECTION.search(text), "the README's install section is no longer found"
        text = _INSTALL_SECTION.sub("", text)
    return text


def _known_files() -> list[dict]:
    """Every test template's file and every add-on file, as each holds it."""
    return [fixture(stem) for stem in STEMS] + [json.loads(path.read_text()) for path in _ADDONS]


def _family_of(name: str) -> str:
    """The family a model file's name gives: its leading letters, up to its first digit or other sign."""
    return re.match("[a-z]+", name.lower()).group()


def _publishers_in(source: str) -> set[str]:
    """Every publisher a ``source`` names, as the name before each slash."""
    return {found.lower() for found in re.findall(r"([\w.-]+)/", source)}


class _Names(NamedTuple):
    """What the name check looks for, read from the model files."""

    families: set[str]
    vendors: set[str]
    models: set[str]            # each model file's own name, which a compound word may hold
    everyday: set[str]          # families and vendors whose name is also an everyday word


@functools.cache
def _names() -> _Names:
    stems = [*STEMS, *(path.stem for path in _ADDONS)]
    named = {expected(fixture(stem), key) for stem in STEMS for key in ("publisher", "chat_format")}
    vendors = {found for known in _known_files() for found in _publishers_in(known.get("source", ""))}
    return _Names(families={_family_of(stem) for stem in stems},
                  vendors=vendors | (named - {None}),
                  models={stem.lower() for stem in stems},
                  everyday={word for stem in STEMS
                            for word in expected(fixture(stem), "everyday_words")})


def _own_names(stem: str) -> set[str]:
    """The names the check reads from the test template ``stem``'s file and from the add-on file that names its template."""
    known = fixture(stem)
    fingerprint = template_fingerprint(known["chat_template"])
    addons = [json.loads(path.read_text()) | {"stem": path.stem} for path in _ADDONS]
    addons = [addon for addon in addons if fingerprint in addon["templates"]]
    named = {expected(known, key) for key in ("publisher", "chat_format")} - {None}
    return (named | {_family_of(name) for name in [stem, *(addon["stem"] for addon in addons)]}
            | {found for file in [known, *addons] for found in _publishers_in(file.get("source", ""))})


def _spelled(word: str) -> str:
    """``word`` in lower case, a space before each capital that follows a small letter."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word).lower()


def _parts(word: str) -> list[str]:
    """``word`` whole and in its parts: split at ``_``, ``-``, ``.`` and before capitals, and those split again between letters and digits."""
    pieces = re.sub(r"[_.\-]", " ", _spelled(word)).split()
    spaced = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", " ".join(pieces))
    return [word.lower()] + pieces + spaced.split()


def _holds(word: str, name: str) -> bool:
    """Whether ``word`` holds ``name`` between its separators, as a word or a run of words."""
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", _spelled(word)))


def _names_a_model(word: str, names: _Names) -> bool:
    """Whether ``word`` names a model, family or vendor.

    A name that is also an everyday word names one only as the whole word,
    or followed by a version; as a part of a longer name it means the word.
    """
    if word.lower() in _OWN_NAMES:
        return False
    whole, *parts = _parts(word)
    if whole in names.families | names.vendors | names.models:
        return True
    if any(_holds(word, name) for name in names.vendors | names.models if re.search(r"[.\-]", name)):
        return True
    if any(part in (names.families | names.vendors) - names.everyday for part in parts):
        return True
    for family in names.families:
        if family in names.everyday:
            if re.search(rf"(?<![a-z0-9]){re.escape(family)}[-._]?[a-z]?\d", _spelled(word)):
                return True
        elif any(re.fullmatch(re.escape(family) + r"(\d[\d.]*|-.+)", part) for part in [whole, *parts]):
            return True
    return False


def _is_the_engines(word: str) -> bool:
    return word.lower() == _ENGINE_PREFIX or word.lower().startswith(_ENGINE_PREFIX + "_")


@pytest.mark.parametrize("path", _checked_files(), ids=lambda path: str(path.relative_to(_ROOT)))
def test_a_file_names_no_model_family_or_vendor(path):
    words = set(re.findall(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*", _OWN_NAME.sub(" ", _checked_text(path))))
    if path in _ENGINE_FILES:
        words = {word for word in words if not _is_the_engines(word)}
    assert not sorted(word for word in words if _names_a_model(word, _names()))


def _recognized_forms() -> list[str]:
    """Each family, vendor and model as names spell it: alone, versioned, inside an identifier, a compound word or a file name, and in camel case."""
    names = _names()
    forms = []
    for family in sorted(names.families):
        forms += [family, f"{family}3", f"{family}2.5", f"{family}-3", f"{family}3coder",
                  f"{family.capitalize()}-X2.5", f"my-{family}-3"]
        if family not in names.everyday:
            forms += [f"is_{family}", f"{family}_tags", f"{family.capitalize()}Tags",
                      f"my-{family}-model", f"x-{family}", f"a.{family}", f"{family}.gguf",
                      f"{family.capitalize()}-Instruct-3-8B"]
    for vendor in sorted(names.vendors):
        forms += [vendor] if vendor in names.everyday else [vendor, f"{vendor}_format",
                                                            f"x-{vendor}-3"]
    for model in sorted(names.models):
        forms += [model, f"My-{model}-Instruct"] if re.search(r"[.\-]", model) else [model]
    return forms


def test_every_publisher_a_source_names_before_a_slash_is_a_vendor():
    vendors = _names().vendors
    named = {piece.split("/")[0].lower() for known in _known_files()
             for piece in re.split(r"[\s,:;\"()]+", known.get("source", "")) if "/" in piece}
    assert named - {""} <= vendors
    assert not [vendor for vendor in vendors if re.search(r"\s", vendor)]


def test_a_model_family_or_vendor_name_is_recognized_in_every_form():
    assert not [form for form in _recognized_forms() if not _names_a_model(form, _names())]


@pytest.mark.parametrize("stem", STEMS)
def test_an_everyday_word_a_file_lists_is_one_of_its_own_names(stem):
    assert set(expected(fixture(stem), "everyday_words")) <= _own_names(stem)


@pytest.mark.parametrize("word", [*_OWN_NAMES, "sampling", "max_batch_tokens", "ThoughtFilter",
                                  *(form for word in sorted(_names().everyday) for form in (
                                      f"{word}_value", f"{word}-like", f"{word.capitalize()}Plug",
                                      f"the.{word}"))])
def test_the_library_its_engine_and_an_everyday_word_are_not_taken_for_a_model(word):
    assert not _names_a_model(word, _names())


# The keys of a test template's file whose value is one of the template's tags.
_TAG_KEYS = ("eos_token", "turn_end_token", "call_closing_tag")


def _stated_tags(known: dict) -> set[str]:
    """The tags a test template's file states beside its template."""
    stated = {known.get(key) for key in _TAG_KEYS} | set(expected(known, "ends_generation"))
    return stated - {None}


def _template_tags() -> set[str]:
    return {tag for stem in STEMS for tag in (*fixture(stem)["tags"], *_stated_tags(fixture(stem)))}


@pytest.mark.parametrize("stem", STEMS)
def test_every_tag_a_test_templates_file_states_is_among_the_tags_it_lists(stem):
    assert not sorted(_stated_tags(fixture(stem)) - set(fixture(stem)["tags"]))


@pytest.mark.parametrize("path", _checked_files(), ids=lambda path: str(path.relative_to(_ROOT)))
def test_a_file_writes_no_tag_of_a_real_template(path):
    text = _checked_text(path)
    assert not sorted(tag for tag in _template_tags() if len(tag) > 3 and tag in text)


def test_the_invented_templates_tags_are_no_real_templates():
    from .invented_templates import INVENTED
    invented = {tag for template in INVENTED.values() for tag in template["tags"]}
    assert invented and not invented & _template_tags()
