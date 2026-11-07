"""A model's own sampling: what its GGUF states, else its family's add-on file, else a default."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .config import Config

# The sampling fields a model may recommend, and the GGUF keys that state them.
SAMPLING_FIELDS = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")
_GGUF_KEYS = {"temperature": "general.sampling.temp", "top_k": "general.sampling.top_k",
              "top_p": "general.sampling.top_p", "min_p": "general.sampling.min_p",
              "repetition_penalty": "general.sampling.penalty_repeat"}
# llama.cpp's own defaults, for a model that recommends nothing.
DEFAULT_SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                    "repetition_penalty": 1.0}
# What a field a recommendation leaves out stands at: switched off.
UNSTATED_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                     "repetition_penalty": 1.0}
FAMILY_FILES = Path(__file__).parent / "families"
_FILE_KEYS = {"templates", "source", "sampling"}


def with_model_sampling(config: Config, metadata: Callable[[str], str | None]) -> Config:
    """``config`` with each sampling field it leaves unset taken from the model.

    One source is taken whole: the GGUF's recommendation, else the add-on
    file's, a field it leaves out switched off; with neither, llama.cpp's
    defaults.

    Args:
        config: What the caller set; a field it sets is kept.
        metadata: Reads one of the model's GGUF metadata values, ``None`` where absent.

    Raises:
        ValueError: If an add-on file for this model is malformed.
    """
    unset = [name for name in SAMPLING_FIELDS if getattr(config, name) is None]
    if not unset:
        return config
    stated = (_stated_in_gguf(metadata)
              or family_sampling(config.model_path, metadata("tokenizer.chat_template")))
    rest = UNSTATED_SAMPLING if stated else DEFAULT_SAMPLING
    return replace(config, **{name: stated.get(name, rest[name]) for name in unset})


def template_fingerprint(template: str) -> str:
    """The fingerprint an add-on file names a chat template by."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def family_sampling(model_path: str | None, template: str | None) -> dict:
    """The sampling a model's add-on file recommends, empty where it has none.

    The file next to the model with its name and ``.json`` comes first, then a
    file shipped in :data:`FAMILY_FILES` that names the model's template.

    Raises:
        ValueError: If the file found is malformed.
    """
    own = Path(model_path).with_suffix(".json") if model_path else None
    if own is not None and own.is_file():
        return _read_family_file(own)["sampling"]
    if template is None:
        return {}
    fingerprint = template_fingerprint(template)
    for path in sorted(FAMILY_FILES.glob("*.json")):
        found = _read_family_file(path)
        if fingerprint in found["templates"]:
            return found["sampling"]
    return {}


def _read_family_file(path: Path) -> dict:
    """An add-on file's contents, checked as a model loads.

    Raises:
        ValueError: If it is not JSON, holds a key no add-on file has, or a
            sampling value :class:`Config` refuses.
    """
    try:
        found = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(found, dict) or not set(found) <= _FILE_KEYS:
        raise ValueError(f"{path} may hold only {sorted(_FILE_KEYS)}")
    sampling = found.get("sampling", {})
    if not isinstance(sampling, dict) or not set(sampling) <= set(SAMPLING_FIELDS):
        raise ValueError(f"{path}: sampling may set only {list(SAMPLING_FIELDS)}")
    try:
        Config(**sampling)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from None
    return {"templates": list(found.get("templates", [])), "sampling": sampling}


def _stated_in_gguf(metadata: Callable[[str], str | None]) -> dict:
    """The sampling the model's GGUF states, as numbers."""
    stated = {}
    for name, key in _GGUF_KEYS.items():
        value = metadata(key)
        if value is None:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        stated[name] = int(number) if name == "top_k" else number
    return stated
