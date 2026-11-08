"""What a model needs beyond its template: its add-on file, and the sampling it runs with.

An add-on file holds what the model's GGUF and its template do not tell, and is
checked as the model loads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from .config import Config

SHIPPED_ADDONS = Path(__file__).parent / "addons"
SAMPLING_FIELDS = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")
CALL_FIELDS = ("opening", "name_close", "closing")
_FILE_KEYS = {"templates", "source", "sampling", "call"}
# The GGUF keys that state the sampling fields a model may recommend.
_GGUF_KEYS = {"temperature": "general.sampling.temp", "top_k": "general.sampling.top_k",
              "top_p": "general.sampling.top_p", "min_p": "general.sampling.min_p",
              "repetition_penalty": "general.sampling.penalty_repeat"}
# llama.cpp's own defaults, for a model that recommends nothing.
DEFAULT_SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                    "repetition_penalty": 1.0}
# What a field a recommendation leaves out stands at: switched off.
UNSTATED_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                     "repetition_penalty": 1.0}


def template_fingerprint(template: str) -> str:
    """The fingerprint an add-on file names a chat template by."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def addon_file(model_path: str | None, template: str | None) -> dict:
    """A model's add-on file, with ``sampling`` and ``call`` always present, possibly empty.

    The file next to the model with its name and ``.json`` comes first, then a
    file shipped in :data:`SHIPPED_ADDONS` that names the model's template.

    Raises:
        ValueError: If the file found is malformed.
    """
    own = Path(model_path).with_suffix(".json") if model_path else None
    if own is not None and own.is_file():
        return read_addon_file(own)
    if template is None:
        return read_addon_file(None)
    fingerprint = template_fingerprint(template)
    for path in sorted(SHIPPED_ADDONS.glob("*.json")):
        found = read_addon_file(path)
        if fingerprint in found["templates"]:
            return found
    return read_addon_file(None)


def read_addon_file(path: Path | None) -> dict:
    """An add-on file's contents, checked; an empty one for ``None``.

    Raises:
        ValueError: If it is not JSON, holds a key no add-on file has, a
            sampling value :class:`Config` refuses, or a call field that is not
            text.
    """
    if path is None:
        return {"templates": [], "sampling": {}, "call": {}}
    try:
        found = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(found, dict) or not set(found) <= _FILE_KEYS:
        raise ValueError(f"{path} may hold only {sorted(_FILE_KEYS)}")
    sampling, call = found.get("sampling", {}), found.get("call", {})
    if not isinstance(sampling, dict) or not set(sampling) <= set(SAMPLING_FIELDS):
        raise ValueError(f"{path}: sampling may set only {list(SAMPLING_FIELDS)}")
    try:
        Config(**sampling)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from None
    if call and (not isinstance(call, dict) or set(call) != set(CALL_FIELDS)
                 or not all(isinstance(value, str) and value for value in call.values())):
        raise ValueError(f"{path}: call must give {list(CALL_FIELDS)}, each as text")
    return {"templates": list(found.get("templates", [])), "sampling": sampling, "call": call}


def with_model_sampling(config: Config, metadata: Callable[[str], str | None],
                        addon: dict) -> Config:
    """``config`` with each sampling field it leaves unset taken from the model.

    One source is taken whole: the GGUF's recommendation, else the add-on
    file's, a field it leaves out switched off; with neither, llama.cpp's
    defaults.

    Args:
        config: What the caller set; a field it sets is kept.
        metadata: Reads one of the model's GGUF metadata values, ``None`` where absent.
        addon: The model's add-on file, as :func:`addon_file` reads it.
    """
    unset = [name for name in SAMPLING_FIELDS if getattr(config, name) is None]
    if not unset:
        return config
    stated = _stated_in_gguf(metadata) or addon["sampling"]
    rest = UNSTATED_SAMPLING if stated else DEFAULT_SAMPLING
    return replace(config, **{name: stated.get(name, rest[name]) for name in unset})


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
