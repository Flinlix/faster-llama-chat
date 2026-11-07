"""A model's own sampling: what its GGUF states, else its add-on file, else a default."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .addon_files import SAMPLING_FIELDS
from .config import Config

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


def with_model_sampling(config: Config, metadata: Callable[[str], str | None],
                        addon: dict) -> Config:
    """``config`` with each sampling field it leaves unset taken from the model.

    One source is taken whole: the GGUF's recommendation, else the add-on
    file's, a field it leaves out switched off; with neither, llama.cpp's
    defaults.

    Args:
        config: What the caller set; a field it sets is kept.
        metadata: Reads one of the model's GGUF metadata values, ``None`` where absent.
        addon: The model's add-on file, as :func:`~llama_chat.addon_files.addon_file`
            reads it.
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
