"""Recover per-role template fragments from a model's own chat template.

This is the inverse of the validation check in
:meth:`~llama_chat.wrapper.ChatWrapper._check_fragments_match_model`: instead of
rendering a probe to verify existing fragments, we render probes with
sentinel content and split the render on the sentinels to *recover* the
``prefix``/``suffix`` wrapped around each role's content.

The render callable is the model's GGUF chat template (see
:meth:`~llama_chat.context.KVContext.render_with_model_template`). Keeping the
logic as a pure function of that callable lets it be tested without loading a
model.
"""

from __future__ import annotations

from typing import Callable

from .template import Fragments

# Plain alphanumeric sentinels survive Jinja ``| trim`` and are never
# HTML-escaped, so they appear verbatim in the render and split cleanly.
_SYS = "SYSMSG"
_USER = "USERMSG"
_ASST = "ASSTMSG"

Render = Callable[[list[dict]], "str | None"]


def extract_fragments(render: Render) -> Fragments | None:
    """Recover template fragments by probing ``render`` with sentinel content.

    Args:
        render: Renders a list of ``{"role", "content"}`` messages through the
            model's chat template, returning the prompt string or ``None`` when
            the template rejects that message shape.

    Returns:
        A :class:`~llama_chat.template.Fragments` recovered from the template, or
        ``None`` when the model ships no usable template.
    """
    user = render([{"role": "user", "content": _USER}])
    if user is None or _USER not in user:
        return None
    user_prefix, _, user_suffix = user.partition(_USER)

    fragments = {
        "user_prefix": user_prefix,
        "user_suffix": user_suffix,
        "trim_content": _detect_trim(render),
    }
    fragments.update(_extract_assistant(render, user_prefix, user_suffix))
    fragments.update(_extract_system(render, user_prefix, user_suffix))
    return Fragments(**fragments)


def _extract_assistant(render: Render, user_prefix: str, user_suffix: str) -> dict:
    """Split off the assistant fragment from a user+assistant render."""
    both = render([
        {"role": "user", "content": _USER},
        {"role": "assistant", "content": _ASST},
    ])
    if both is None or _ASST not in both:
        # Fall back to the user tags so generation still has an open/close pair.
        return {"assistant_prefix": user_prefix, "assistant_suffix": user_suffix}
    # Strip the leading user fragment, then the assistant fragment is what wraps
    # the assistant sentinel.
    tail = both.split(_USER, 1)[1].split(user_suffix, 1)[1]
    asst_prefix, _, asst_suffix = tail.partition(_ASST)
    return {"assistant_prefix": asst_prefix, "assistant_suffix": asst_suffix}


def _extract_system(render: Render, user_prefix: str, user_suffix: str) -> dict:
    """Split off the system fragment, or fall back to the user tags.

    Many templates have no system role (Gemma rejects it) or fold it into the
    first user turn - there is no portable ground truth. When the system
    sentinel does not survive verbatim ahead of the user turn, reuse the user
    tags (the convention llama.cpp uses for Gemma).
    """
    rendered = render([
        {"role": "system", "content": _SYS},
        {"role": "user", "content": _USER},
    ])
    if rendered is None or _SYS not in rendered or _USER not in rendered:
        return {"system_prefix": user_prefix, "system_suffix": user_suffix}
    # The system fragment is everything before the user turn begins.
    head = rendered.split(user_prefix + _USER, 1)[0]
    sys_prefix, _, sys_suffix = head.partition(_SYS)
    return {"system_prefix": sys_prefix, "system_suffix": sys_suffix}


def _detect_trim(render: Render) -> bool:
    """True if the template strips surrounding whitespace from content.

    Renders a user turn whose content is padded with spaces; if a space survives
    next to the sentinel the template emits content verbatim (``trim_content``
    False), otherwise it applies ``| trim``.
    """
    padded = render([{"role": "user", "content": " " + _USER + " "}])
    if padded is None or _USER not in padded:
        return True
    before, _, after = padded.partition(_USER)
    return not (before.endswith(" ") and after.startswith(" "))
