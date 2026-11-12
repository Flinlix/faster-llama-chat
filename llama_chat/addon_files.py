"""What a model needs beyond its template: its add-on file, and the sampling it runs with.

An add-on file holds what the model's GGUF and its template do not tell, and is
checked as the model loads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .config import Config, InvalidSettingError

_log = logging.getLogger(__name__)

SHIPPED_ADDONS = Path(__file__).parent / "addons"
SAMPLING_FIELDS = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty",
                   "presence_penalty", "frequency_penalty", "penalty_window_tokens")
CALL_FIELDS = ("opening", "name_close", "closing")
_FILE_KEYS = {"templates", "source", "sampling", "call", "reasoning_on"}
# llama.cpp's own defaults, for a model that recommends nothing.
DEFAULT_SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05,
                    "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                    "penalty_window_tokens": 64}
# What a field no source states stands at, where another field but the penalty window
# is stated: switched off, and the penalty window at the engine's default length.
UNSTATED_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
                     "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                     "penalty_window_tokens": 64}


def template_fingerprint(template: str) -> str:
    """The fingerprint an add-on file names a chat template by."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def addon_file(model_path: str | None, template: str | None) -> dict:
    """A model's add-on file, with ``sampling``, ``call`` and ``reasoning_on`` always present, possibly empty.

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
        ValueError: Naming the file, if it is not JSON, holds a key no add-on
            file has, templates that are not a list of fingerprints, a source
            that is not text, a sampling value :class:`Config` refuses, its
            type included, a call field that is not text, or a reasoning_on
            that is no object of template variables.
    """
    if path is None:
        return {"templates": [], "sampling": {}, "call": {}, "reasoning_on": {}}
    try:
        found = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(found, dict) or not set(found) <= _FILE_KEYS:
        raise ValueError(f"{path} may hold only {sorted(_FILE_KEYS)}")
    templates = found.get("templates", [])
    if not isinstance(templates, list) or not all(
            isinstance(t, str) and re.fullmatch("[0-9a-f]{64}", t) for t in templates):
        raise ValueError(f"{path}: templates must be a list of SHA-256 fingerprints, "
                         "each as 64 lowercase hex digits")
    if not isinstance(found.get("source", ""), str):
        raise ValueError(f"{path}: source must be text")
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
    reasoning_on = found.get("reasoning_on", {})
    if not isinstance(reasoning_on, dict) or "reasoning_on" in found and not reasoning_on:
        raise ValueError(f"{path}: reasoning_on must be an object naming the template "
                         "variables that turn reasoning on, with their values")
    return {"templates": templates, "sampling": sampling, "call": call,
            "reasoning_on": reasoning_on}


def with_model_sampling(config: Config, recommended: Mapping[str, float],
                        addon: dict) -> Config:
    """``config`` with each sampling field it leaves unset taken from the model.

    Value by value: the add-on file's, else what the model recommends itself,
    else switched off. A model for which neither states any value but the
    penalty window runs at llama.cpp's defaults.

    Args:
        config: What the caller set; a field it sets is kept.
        recommended: The sampling the model states for itself, as
            :meth:`~llama_chat.cache_context.ModelReader.recommended_sampling` gives it.
        addon: The model's add-on file, as :func:`addon_file` reads it.

    Raises:
        ValueError: If a value the model recommends, which neither ``config``
            nor the add-on file replaces, is one the settings refuse, or names
            no model sampling field. Not an
            :class:`~llama_chat.config.InvalidSettingError`, since no setting
            of the caller's is at fault.
    """
    unknown = sorted(set(recommended) - set(SAMPLING_FIELDS))
    if unknown:
        raise ValueError(f"the model's own recommendation (its GGUF) names {unknown}, which "
                         f"are no model sampling fields {list(SAMPLING_FIELDS)}")
    unset = [name for name in SAMPLING_FIELDS if getattr(config, name) is None]
    for name in unset:
        if name in recommended and name not in addon["sampling"]:
            _ensure_recommendation_taken(name, recommended[name])
    stated = {**recommended, **addon["sampling"]}
    rest = UNSTATED_SAMPLING if set(stated) - {"penalty_window_tokens"} else DEFAULT_SAMPLING
    return replace(config, **{name: stated.get(name, rest[name]) for name in unset})


def _ensure_recommendation_taken(name: str, value) -> None:
    """Refuse a value the model recommends for ``name`` that the settings refuse, naming where it comes from.

    Raises:
        ValueError: If :class:`Config` refuses it.
    """
    try:
        Config(**{name: value})
    except InvalidSettingError as refused:
        raise ValueError(
            f"the model's own recommendation (its GGUF) gives {name} = {value!r}, which is "
            f"refused: {refused}. Set {name} in Config, or in the model's add-on file, "
            "to run it with another value") from None
