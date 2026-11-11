"""The test chat templates of ``tests/templates/``, and what the tests that run them all ask, answer and call."""

import json
from pathlib import Path

from llama_chat.tools import ToolSpec


_TEMPLATES = Path(__file__).parent / "templates"
# Every test template, by the name of its file.
STEMS = sorted(path.stem for path in _TEMPLATES.glob("*.json"))


def fixture(stem: str) -> dict:
    """The test template ``stem`` of ``tests/templates/``, as its file holds it."""
    return json.loads((_TEMPLATES / f"{stem}.json").read_text())


# What the template-wide tests ask, answer and call, the same in every file.
QUESTIONS = ["How are you?", "Name a colour.", "And a number?", "Why that one?",
             "Tell me more.", "Shorter, please.", "Thanks!", "One more thing?"]
ANSWERS = [f"Answer number {i}, kept short." for i in range(len(QUESTIONS) + 2)]
WEATHER = ToolSpec(name="get_weather", description="Weather for a city",
                   parameters={"type": "object", "properties": {"city": {"type": "string"}}})


def in_each_thought_form(messages: list[dict]) -> list[list[dict]]:
    """``messages``, each reply's thought held under ``reasoning_content``, and the same with every thought in each other form chat templates read one in: a ``thinking`` field, or a content block ahead of the text."""
    def moved(message, form):
        thought = message.get("reasoning_content")
        if not thought:
            return message
        rest = {key: value for key, value in message.items() if key != "reasoning_content"}
        if form == "field":
            return {**rest, "thinking": thought}
        return {**rest, "content": [{"type": "thinking", "thinking": thought},
                                    {"type": "text", "text": message["content"]}]}
    return [list(messages)] + [[moved(message, form) for message in messages]
                               for form in ("field", "block")]
