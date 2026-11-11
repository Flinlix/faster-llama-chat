"""A model's own chat template, rendered with its reasoning switch applied."""

from __future__ import annotations

from collections.abc import Mapping

from ._backend import template_names_read, template_renderer
from .template_extract import reasoning_switch


def names_read(text: str) -> frozenset[str]:
    """Every name the template ``text`` reads, a variable it gives a default when undefined included, as its renderer parses it."""
    return template_names_read(template_renderer(text, ""))


class ModelTemplate:
    """The chat template a model ships, and renders of it.

    Built from the template's text and the text of the model's end token,
    which some templates write. Shared by everything that renders a model's
    template, so a test of the template runs the code a loaded model runs.
    """

    def __init__(self, text: str, eos_text: str) -> None:
        self.text = text
        self._renderers = {prompted: template_renderer(text, eos_text, prompted)
                           for prompted in (False, True)}
        self._switch = reasoning_switch(lambda messages, generation_prompt, variables:
                                        self._renderers[generation_prompt](
                                            messages=messages, **variables).prompt)

    def render(self, messages: list[dict], tools: list[dict] | None = None, *,
               generation_prompt: bool = False, reasoning: bool | None = None,
               variables: Mapping[str, object] | None = None) -> str | None:
        """``messages`` as the template writes them, or ``None`` where it refuses that shape.

        Args:
            generation_prompt: Append the prompt a reply is generated after.
            reasoning: Turn the template's reasoning switch on or off, as
                :func:`~llama_chat.template_extract.reasoning_switch` reads it.
                ``None`` leaves it unset.
            variables: Further template variables, by name.
        """
        try:
            return self._renderers[generation_prompt](
                messages=messages, tools=tools, **self._switch(reasoning),
                **(variables or {})).prompt
        except Exception:
            return None
