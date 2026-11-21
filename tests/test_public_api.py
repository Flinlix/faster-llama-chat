"""Every public name keeps its signature: what ``llama_cpp_conversation`` exports and what ``llama_cpp_conversation.testing`` offers, compared with ``tests/public_api.txt``.

A change to a signature updates that file in the same commit, with the kind of
release it needs (``docs/guide/compatibility.md``): ``python -m tests.test_public_api``
writes the file anew.
"""

import dataclasses
import inspect
import re
import subprocess
import sys
from pathlib import Path

import llama_cpp_conversation
from llama_cpp_conversation.testing import fake_context, invented_templates

_LIST = Path(__file__).with_name("public_api.txt")


# The special methods a caller uses: making an object, iterating it, and using it in ``with``.
_USED_SPECIAL = ("__init__", "__iter__", "__next__", "__enter__", "__exit__")


def _members(owner) -> list[str]:
    """The signature of each public method and property of the class ``owner``, its own and those it takes from the library's classes it is built on, and its fields where it is a dataclass."""
    lines = []
    if dataclasses.is_dataclass(owner):
        lines += [f"  field {f.name}: {f.type}" for f in dataclasses.fields(owner) if not f.name.startswith("_")]
    found = {}
    for base in reversed(owner.__mro__):
        if base.__module__.split(".")[0] == llama_cpp_conversation.__name__:
            found.update(vars(base))
    for name, member in sorted(found.items()):
        if name.startswith("_") and name not in _USED_SPECIAL:
            continue
        if name == "__init__" and getattr(owner, "_is_protocol", False):
            continue                        # a protocol is satisfied, not made
        if isinstance(member, property):
            lines.append(f"  property {name}")
        elif isinstance(member, (staticmethod, classmethod)) or inspect.isfunction(member):
            function = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
            lines.append(f"  def {name}{inspect.signature(function)}")
    return lines


def _described(name: str, value) -> list[str]:
    """What a caller relies on of the public ``value`` named ``name``."""
    if inspect.isclass(value) and hasattr(value, "__required_keys__"):
        return [f"typed dictionary {name}"] + [
            f"  key {key}: {inspect.formatannotation(kind)}"
            + ("" if key in value.__required_keys__ else " (may be left out)")
            for key, kind in value.__annotations__.items()]
    if inspect.isclass(value):
        bases = ", ".join(base.__name__ for base in value.__bases__ if base is not object)
        return [f"class {name}({bases})"] + _members(value)
    if inspect.isfunction(value):
        return [f"def {name}{inspect.signature(value)}"]
    return [f"{name} = {value!r}"]


def public_api() -> str:
    """Every public name with its signature, one per line."""
    lines = []
    for name in sorted(llama_cpp_conversation.__all__):
        lines += _described(name, getattr(llama_cpp_conversation, name))
    lines += _described("testing.FakeContext", fake_context.FakeContext)
    lines += _described("testing.ScriptedCall", fake_context.ScriptedCall)
    lines += [f"testing.invented_templates.{name}" for name in sorted(vars(invented_templates))
              if name.isupper() and not name.startswith("_")]
    # A default that is a function shows where it lives in memory, which differs every run.
    return re.sub(r" at 0x[0-9a-f]+", "", "\n".join(lines) + "\n")


def test_every_public_signature_is_the_one_listed():
    # Read in a fresh interpreter: the suite's own checks wrap some of these names.
    listed = subprocess.run([sys.executable, "-c", "from tests.test_public_api import public_api; "
                             "print(public_api(), end='')"], cwd=_LIST.parent.parent,
                            capture_output=True, text=True, timeout=120, check=True).stdout
    assert listed == _LIST.read_text(), (
        "a public signature changed: compare with tests/public_api.txt, write it anew with "
        "python -m tests.test_public_api, and name the release it needs")


if __name__ == "__main__":
    _LIST.write_text(public_api())
