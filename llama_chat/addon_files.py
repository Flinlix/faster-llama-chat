"""What a model needs beyond its template: its add-on file, and the sampling it runs with.

An add-on file holds what the model's GGUF and its template do not tell, and is
checked as the model loads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .config import Config

_log = logging.getLogger(__name__)

SHIPPED_ADDONS = Path(__file__).parent / "addons"
SAMPLING_FIELDS = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty",
                   "presence_penalty", "frequency_penalty")
CALL_FIELDS = ("opening", "name_close", "closing")
_FILE_KEYS = {"templates", "source", "sampling", "call"}
# llama.cpp's own defaults, for a model that recommends nothing.
DEFAULT_SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
# What a field no source states stands at, where another field is stated: switched off.
UNSTATED_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                     "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def template_fingerprint(template: str) -> str:
    """The fingerprint an add-on file names a chat template by."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def addon_file(model_path: str | None, template: str | None) -> dict:
    """A model's add-on file, with ``sampling`` and ``call`` always present, possibly empty.

    The file next to the model with its name and ``.json`` comes first, where
    it names the model's template; one that does not is not used, and a
    warning says why. Then a file shipped in :data:`SHIPPED_ADDONS` that names
    the template.

    Raises:
        ValueError: If the file found is malformed.
    """
    fingerprint = template_fingerprint(template) if template is not None else None
    own = Path(model_path).with_suffix(".json") if model_path else None
    if own is not None and own.is_file():
        unused_because = _not_naming(own, fingerprint)
        if unused_because is None:
            return read_addon_file(own)
        _log.warning("%s is not used as the add-on file of %s: %s", own, model_path,
                     unused_because)
    if fingerprint is None:
        return read_addon_file(None)
    for path in sorted(SHIPPED_ADDONS.glob("*.json")):
        found = read_addon_file(path)
        if fingerprint in found["templates"]:
            return found
    return read_addon_file(None)


def _not_naming(path: Path, fingerprint: str | None) -> str | None:
    """Why the file at ``path`` is not the add-on file of the template ``fingerprint`` names; ``None`` where it is."""
    if fingerprint is None:
        return "the model ships no chat template for it to name"
    try:
        found = json.loads(path.read_text())
    except ValueError:
        return "it is not JSON"
    if not isinstance(found, dict):
        return "it is not a JSON object"
    named = found.get("templates")
    if not isinstance(named, list) or not named:
        return "it names no template: its templates is no list of fingerprints"
    if fingerprint not in named:
        return f"it names other templates than this model's ({fingerprint})"
    return None


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


def with_model_sampling(config: Config, recommended: Mapping[str, float],
                        addon: dict) -> Config:
    """``config`` with each sampling field it leaves unset taken from the model.

    Value by value: the add-on file's, else what the model recommends itself,
    else switched off. A model for which neither states any value runs at
    llama.cpp's defaults.

    Args:
        config: What the caller set; a field it sets is kept.
        recommended: The sampling the model states for itself, as
            :meth:`~llama_chat.cache_context.ModelReader.recommended_sampling` gives it.
        addon: The model's add-on file, as :func:`addon_file` reads it.
    """
    unset = [name for name in SAMPLING_FIELDS if getattr(config, name) is None]
    if not unset:
        return config
    stated = {**recommended, **addon["sampling"]}
    rest = UNSTATED_SAMPLING if stated else DEFAULT_SAMPLING
    return replace(config, **{name: stated.get(name, rest[name]) for name in unset})
