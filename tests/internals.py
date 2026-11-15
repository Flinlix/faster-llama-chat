"""What tests read of a wrapper, or set in it, that no public member offers, here and nowhere else.

Each function reads or sets one thing in plain values, so that an internal
change that keeps the behaviour changes this module alone.
"""

import dataclasses
import sys

from llama_chat.cache_sync import _TOKEN_OFFSET
from llama_chat import rendering
from llama_chat.template_reading import TemplateReader


def token_id_range() -> range:
    """The token ids a cache context may hand out: those the search for a dropped stretch can spell."""
    return range(-_TOKEN_OFFSET, sys.maxunicode - _TOKEN_OFFSET + 1)


# ----- the tokens the wrapper holds the cache to be ----------------------------------
def held_tokens(wrapper) -> list[int]:
    """The tokens the wrapper holds the cache to be; ``total_tokens`` is their count."""
    return list(wrapper._conversation._memory.tokens)


def overwrite_held_tokens(wrapper, tokens: list[int]) -> None:
    """Have the wrapper hold the cache to be ``tokens``, by no change it would notice."""
    wrapper._conversation._memory.tokens[:] = tokens


def leave_cache_behind(wrapper, tokens: list[int]) -> None:
    """Have the wrapper hold the cache to be ``tokens``, the cache left for later as an action may leave it."""
    wrapper._conversation._memory.replace(tokens)


def bring_cache_back(wrapper) -> None:
    """Bring the cache back where a failed decode, edit or generation left it unknown, as the next action would."""
    memory = wrapper._conversation._memory
    if not memory.is_known:
        memory.repair()


# ----- the wrapper and the conversation as it is rendered ------------------------------
def context_of(wrapper):
    """The cache context the wrapper runs on, the one it built where it was given none."""
    return wrapper._context


def is_closed(wrapper) -> bool:
    return wrapper._closed


def is_begun(wrapper) -> bool:
    return wrapper._conversation.begun


def rendered_messages(wrapper) -> list[dict]:
    """The messages the conversation is rendered from, the system prompt's included."""
    return [dict(message) for message in wrapper._conversation.messages]


def declarations(wrapper) -> list[dict] | None:
    """The tools declared to the conversation, as a render is given them; ``None`` for none."""
    return [spec.as_declaration() for spec in wrapper._conversation.tools] or None


def uses_tool_template(wrapper) -> bool:
    """Whether the conversation is written with the model's template for tools, as begun with tools."""
    return wrapper._conversation._state.uses_tool_template


def reasons(wrapper) -> bool:
    """Whether the conversation was begun reasoning."""
    return wrapper._conversation.reasoning


def generated_turns_held(wrapper) -> list[tuple[list[int], str, bool]]:
    """Each turn kept as generated: its tokens, the text of the message it ends with, and whether it ends on a call."""
    messages = wrapper._conversation.messages
    return [(list(kept.tokens), (messages[last].get("content") or "").strip(), kept.on_call)
            for _first, last, kept in rendering.generated_turns_in(messages)]


def generated_turn_spans(wrapper) -> list[tuple[int, int]]:
    """Each turn kept as generated, as the indexes of its first and its last message among the rendered messages."""
    return [(first, last) for first, last, _kept in rendering.generated_turns_in(wrapper._conversation.messages)]


def generated_turn_places(wrapper) -> list[tuple[int, int] | None]:
    """Where the template writes each generated turn, in each render that splits, with the generation prompt and without."""
    conversation = wrapper._conversation
    messages = conversation.messages
    places = []
    for generation_prompt in (False, True):
        split = conversation._rendering.split(messages, generation_prompt)
        if split is not None:
            places += [conversation._rendering.place(first, last, kept.on_call, split)
                       for first, last, kept in rendering.generated_turns_in(messages)]
    return places


def done_after_the_newest_reply(wrapper) -> tuple[int, int]:
    """How many tokens the work after the newest reply decoded, and how many history messages it evicted, which the next request counts."""
    conversation = wrapper._conversation
    return conversation._decoded_after_reply, conversation._evicted_after_reply[0]


def keeps_the_rest_after_a_drop(wrapper) -> bool:
    """Whether the template writes the exchanges after the oldest as before once it goes."""
    return wrapper._conversation._state.checks.rest_stays


def renders_tool_turns_in_windows(wrapper) -> bool:
    """Whether windows of the conversation stand for its whole render where they hold turns that called tools."""
    return wrapper._conversation._state.checks.tool_windows


def renders_in_windows(wrapper) -> bool:
    """Whether windows of the conversation stand for its whole render."""
    return wrapper._conversation._state.checks.windows


def last_spliced_question(wrapper) -> str | None:
    """The question last decoded in the stand-in's place, ``None`` where none was."""
    spliced_at = wrapper._conversation._state.spliced_at
    return None if spliced_at is None else spliced_at[2].messages[-1]["content"]


def prompt_text(wrapper) -> str:
    """The template text the newest generation followed."""
    return wrapper._conversation.prompt_text


def pending_calls_held(wrapper) -> list:
    """The calls the wrapper itself holds as owed, not the copies ``pending_calls`` hands out."""
    return wrapper._conversation.pending_calls


# ----- a template read otherwise than it writes ---------------------------------------
def read_fragments_as(monkeypatch, **changes) -> None:
    """Have every template's fragments read with ``changes``, as no template can make them read."""
    read = TemplateReader._resolve_fragments
    monkeypatch.setattr(TemplateReader, "_resolve_fragments",
                        lambda self: dataclasses.replace(read(self), **changes))


def template(wrapper):
    """The template the conversation is written with now."""
    return wrapper._conversation.template


def called_in_thought(wrapper) -> bool:
    """Whether the open turn's newest call was made inside a thought the model had not closed."""
    return wrapper._conversation.called_in_thought


def tool_turns_stay(wrapper) -> bool:
    """Whether the conversation's template keeps a past turn that called tools as written once another exchange follows."""
    return wrapper._conversation._state.checks.tool_turns_stay


def joins_questions(wrapper) -> bool:
    """Whether the conversation writes two questions in a row as one."""
    return wrapper._conversation._joins_questions


def unreadable_result_text(wrapper) -> str:
    """The text a call that could not be read is answered with now."""
    from llama_chat.tools import result_text
    return result_text(wrapper._conversation._state.unreadable_result)
