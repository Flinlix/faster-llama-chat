"""The test chat templates of ``tests/templates/``, what each one's file says it does, and what the tests that run them all ask, answer and call."""

import json
from pathlib import Path

from llama_chat.tools import ToolSpec


_TEMPLATES = Path(__file__).parent / "templates"
# Every test template, by the name of its file.
STEMS = sorted(path.stem for path in _TEMPLATES.glob("*.json"))


def fixture(stem: str) -> dict:
    """The test template ``stem`` of ``tests/templates/``, as its file holds it."""
    return json.loads((_TEMPLATES / f"{stem}.json").read_text())


# What a test template's file says of it by leaving a key out: what most templates do.
_UNSTATED = {
    "turn_end_token": None,             # the text of the tag that ends a reply
    "turn_end_text": None,              # the text that ends a reply, where a call opens with the start of its closing
    "answer_opening": "",               # what a reply opens with before its answer
    "call_opens_as_text": False,        # a call opens with plain text, no tag
    "writes_empty_reply": True,
    "can_take_system_prompt": True,     # no later question moves a system prompt
    "date_format": None,                # how the date is written, where the template reads the clock
    "writes_date_beside_system_prompt": False,  # the date is written with a system prompt too
    "lost_arguments": {},               # call arguments that do not read back as made, and why
    "card_sampling": None,              # the sampling the model's card recommends
    "card_quote": None,                 # the card's words, which its shipped add-on file quotes
    "fragments": None,                  # the tags around a turn, by the name extraction gives each
    "tool_fragments": None,             # the texts a call opens and closes with
    "call_body": None,                  # how a call is spelled between its tags, by field
    "call_closing_tag": None,           # the tag that closes a call
    "publisher": None,                  # the publisher's name, where the source does not spell it
    "everyday_words": [],               # those of its names that are also everyday words
    "chat_format": None,                # the name the template's chat format goes by
    "ends_generation": [],              # the tags beside its end-of-sequence tag that end a generation in the engine
}


# What every test template's file states.
REQUIRED_KEYS = {"source", "chat_template", "eos_token", "render_variables", "marks_thought",
                 "can_reason", "calls_tools"}
# What a file may state besides: what the stand-in context reads, a template for tools and
# whether the tokenizer adds a sequence opener, and every key a file may leave out.
KNOWN_KEYS = REQUIRED_KEYS | {"tool_use_template", "adds_bos"} | set(_UNSTATED)


def expected(known: dict, key: str):
    """What the test template ``known``, as its file holds it, is known to do under ``key``."""
    return known.get(key, _UNSTATED[key])


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
