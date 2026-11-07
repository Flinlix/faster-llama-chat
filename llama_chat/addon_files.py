"""A model's add-on file: what its GGUF and its template do not tell, checked as the model loads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import Config

FAMILY_FILES = Path(__file__).parent / "families"
SAMPLING_FIELDS = ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")
CALL_FIELDS = ("opening", "name_close", "closing")
_FILE_KEYS = {"templates", "source", "sampling", "call"}


def template_fingerprint(template: str) -> str:
    """The fingerprint an add-on file names a chat template by."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def addon_file(model_path: str | None, template: str | None) -> dict:
    """A model's add-on file, with ``sampling`` and ``call`` always present, possibly empty.

    The file next to the model with its name and ``.json`` comes first, then a
    file shipped in :data:`FAMILY_FILES` that names the model's template.

    Raises:
        ValueError: If the file found is malformed.
    """
    own = Path(model_path).with_suffix(".json") if model_path else None
    if own is not None and own.is_file():
        return read_addon_file(own)
    if template is None:
        return read_addon_file(None)
    fingerprint = template_fingerprint(template)
    for path in sorted(FAMILY_FILES.glob("*.json")):
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
