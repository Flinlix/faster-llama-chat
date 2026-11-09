"""Cache contexts over real chat templates, for tests: a renderer, tags as a vocabulary holds them, and a cache.

The renderer is the one a model is loaded with; only the tokenizer is a
stand-in, one token per character and one per tag, so what is compared is the
text.
"""

import re

from llama_chat.addon_files import addon_file
from llama_chat.model_template import ModelTemplate
from llama_chat.resolved_template import ResolvedTemplate
from llama_chat.tools import ToolSpec

# What the template-wide tests ask, answer and call, the same in every file.
QUESTIONS = ["How are you?", "Name a colour.", "And a number?", "Why that one?",
             "Tell me more.", "Shorter, please.", "Thanks!", "One more thing?"]
ANSWERS = [f"Answer number {i}, kept short." for i in range(len(QUESTIONS) + 2)]
WEATHER = ToolSpec(name="get_weather", description="Weather for a city",
                   parameters={"type": "object", "properties": {"city": {"type": "string"}}})


class TemplateContext:
    """A model reader over one real template: its renderer, and a vocabulary that holds every tag-shaped text.

    The tokenizer gives one token per character and, where special-token
    parsing is on, one per tag-shaped text, each a token of its own from the
    first time it is met: a stand-in for a vocabulary that holds every tag,
    such as one closing each argument of a call. A tag typed into a message,
    which is tokenized with parsing off, stays its characters and cannot
    become the model's own. The tags it lists are those the template spells,
    and the opening of each closing tag among them.
    """

    def __init__(self, fixture: dict) -> None:
        self._model = ModelTemplate(fixture["chat_template"], fixture["eos_token"])
        self._template = fixture["chat_template"]
        self._tag_ids: dict[str, int] = {}
        self._tag_texts: list[str] = []
        found = set(TAG.findall(self._template))
        self._listed = tuple(self._tag_token(tag) for tag in sorted(
            found | {"<" + tag[2:] for tag in found if tag.startswith("</")}))

    def render_with_model_template(self, messages, tools=None, *,
                                   generation_prompt=False, reasoning=None, **variables):
        return self._model.render(messages, tools, generation_prompt=generation_prompt,
                                  reasoning=reasoning, **variables)

    def chat_template_text(self):
        return self._template

    def addon_call(self):
        # The shipped add-on file that names this template, as a model's load finds it.
        return addon_file(None, self._template)["call"]

    def tokenize(self, text, add_special=False, parse_special=True):
        if not parse_special:
            return [ord(c) for c in text]
        tokens, pos = [], 0
        for found in TAG.finditer(text):
            tokens += [ord(c) for c in text[pos:found.start()]]
            tokens.append(self._tag_token(found.group()))
            pos = found.end()
        return tokens + [ord(c) for c in text[pos:]]

    def _tag_token(self, tag):
        """The one token ``tag`` is, given an id of its own the first time it is met."""
        if tag not in self._tag_ids:
            self._tag_ids[tag] = -1 - len(self._tag_texts)
            self._tag_texts.append(tag)
        return self._tag_ids[tag]

    def is_special(self, token):
        return token < 0

    def special_tokens(self):
        return self._listed

    def token_text(self, token):
        return self._tag_texts[-1 - token] if token < 0 else chr(token)


TAG = re.compile(r"<\|[^<>\s]*?\|?>|<[^<>\s|]+\|>|<[/a-z_]+(?::[a-z0-9_]+)?>|\[[A-Z_]+\]|<｜[^｜]+｜>|\]~!?[a-z][\[\]]|\[[a-z]~\[")
# The token that ends a generation: past every character, as a tag's is below them.
END_OF_GENERATION = 0x110000


def call_marks(template, context) -> tuple[str, str]:
    """The texts a call opens and closes with in a render: its own texts, else its tags' text."""
    boundary = template.call_boundary()
    if boundary.call_texts is not None:
        return boundary.call_texts
    opener, closer = boundary.tool_tokens
    return context.token_text(opener), context.token_text(closer)


class ConversationContext(TemplateContext):
    """The same, with a cache a wrapper writes a whole conversation into.

    Each generation is the library's own loop drawing the next reply of
    ``replies``, one token per character, then the end of generation. It
    first hands ``before_reply`` the cache's text, so a test can compare it
    with the template's render of the conversation at that moment.
    """

    def __init__(self, fixture: dict, replies: list[str], can_shift: bool = True) -> None:
        super().__init__(fixture)
        self._fixture = fixture
        self._call_mark_tokens = None
        self.cache: list[int] = []
        self.can_shift = can_shift
        self.can_cut_tail = can_shift
        self.context_size = 1 << 20
        self.resets = 0             # how often the whole cache was emptied
        self.decoded = 0            # every token prefilled
        # ("cut", position), ("shift", start, tokens) and ("restore", position), in order.
        self.edits: list[tuple] = []
        self.rendered: list[int] = []            # how many messages each render was given
        self._replies = list(replies)
        self.before_reply = None

    def render_with_model_template(self, messages, tools=None, *,
                                   generation_prompt=False, reasoning=None, **variables):
        self.rendered.append(len(messages))
        return super().render_with_model_template(messages, tools,
                                                  generation_prompt=generation_prompt,
                                                  reasoning=reasoning, **variables)

    def text_of(self, tokens) -> str:
        return "".join(self.token_text(token) for token in tokens)

    def reset(self):
        self.cache = []
        self.resets += 1

    def warmup(self):
        pass

    def prefill(self, token_ids, start_pos, want_logits):
        assert start_pos == len(self.cache), (start_pos, len(self.cache))
        self.cache.extend(token_ids)
        self.decoded += len(token_ids)

    def save_state(self):
        return list(self.cache)

    def restore_state(self, state, pos):
        assert len(state) == pos
        self.cache = list(state)
        self.edits.append(("restore", pos))

    def remove_from(self, pos):
        assert self.can_cut_tail and 0 <= pos <= len(self.cache)
        self.edits.append(("cut", pos))
        del self.cache[pos:]

    def apply_eviction(self, ev):
        assert self.can_shift and len(self.cache) == ev.old_total
        self.edits.append(("shift", ev.remove_start, ev.shift_delta))
        del self.cache[ev.remove_start:ev.remove_end]

    def sampled_tokens(self, sampling, withheld_tokens):
        if self.before_reply is not None:
            self.before_reply(self.text_of(self.cache))
        scripted = self._replies.pop(0)
        if isinstance(scripted, ScriptedCall):
            yield from self._call_tokens(scripted)
        else:
            yield from (ord(ch) for ch in scripted)
        while True:
            yield END_OF_GENERATION

    def _call_tokens(self, call):
        """The scripted call as the model writes it: its ``before``, then its opening, its body, its closing."""
        if self._call_mark_tokens is None:
            # The template's own call tags as tokens, else its call texts as characters.
            reader = TemplateContext(self._fixture)
            template = ResolvedTemplate(reader)
            marks = call_marks(template, reader)
            by_tags = template.call_boundary().tool_tokens is not None
            self._call_mark_tokens = tuple(self.tokenize(text) if by_tags else [ord(c) for c in text]
                                           for text in marks)
        opening, closing = self._call_mark_tokens
        return self.tokenize(call.before) + opening + [ord(c) for c in call.body] + closing

    def decode_generated(self, token_ids, start_pos):
        assert start_pos == len(self.cache), (start_pos, len(self.cache))
        self.cache.extend(token_ids)

    def ends_generation(self, token):
        return token == END_OF_GENERATION

    def token_bytes(self, token):
        return b"" if token == END_OF_GENERATION else self.token_text(token).encode()

    def close(self):
        pass


class ScriptedCall:
    """A call the conversation context writes in place of a reply, ``body`` between its tags, ``before`` it as reply text."""

    def __init__(self, body: str, before: str = "") -> None:
        self.body = body
        self.before = before
