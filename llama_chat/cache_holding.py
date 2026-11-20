"""Bringing the cache to hold a target, with the anchors and the states kept where the opening, the newest exchanges and the reply begin, and the render for the next question decoded ahead."""

from __future__ import annotations

from collections.abc import Callable

from .cache_sync import CacheSync
from .config import Config
from .conversation_state import Ahead, Anchor, ConversationState, MadeFor, goes_on_from
from .messages import exchange_starts, system_part
from .render_checks import STAND_IN, question_splices
from .rendering import ConversationRendering, Target, generated_turns_in
from .resolved_template import ResolvedTemplate
from .split_render import SplitRender
from .cache_targets import CacheTargets, closes_turn


# What the states a cache that cannot be cut keeps where the newest exchanges begin are kept for.
NEWEST_EXCHANGES = "newest exchanges"

# What the states it keeps where the newest reply begins and where its generation starts are for.
THE_REPLY = "the reply"

# How many tokens after a kept state another is not kept, being decoded from it instead.
_NEAR_STATE_TOKENS = 32


class CacheHolding:
    """Brings the cache to the targets of the conversation's actions and notes where it holds what."""

    def __init__(self, memory: CacheSync, rendering: ConversationRendering, targets: CacheTargets, state: ConversationState, config: Config, template: Callable[[], ResolvedTemplate]) -> None:
        self._memory = memory
        self._rendering = rendering
        self._targets = targets
        self._state = state
        self._config = config
        self._template = template

    def splice(self, question: dict, room: int) -> bool:
        """Add ``question``, decoded in the stand-in's place in the render made ahead, where that is exact; whether it was."""
        ahead = self._state.ahead
        text = question["content"]
        # A template may write a message otherwise where a later one repeats its text, which
        # only a render of the whole conversation shows.
        if (ahead is None or not self.is_current(ahead.made_for)
                or not self._rendering.is_plain(text)
                or (not self._state.checks.repeats_alike
                    and any(m["content"] == text for m in self._state.messages))
                or not question_splices(self._rendering.render, system_part(self._state.messages), text,
                                        self._state.checks.splice_probe)):
            return False
        rest = [(text if own and piece == STAND_IN else piece, own) for piece, own in ahead.rest]
        tokens = self._rendering.tokenize(rest)
        if len(self._memory.tokens) + len(tokens) > room:
            return False
        if self._memory.keeps_saved_states:
            newest = max(self._memory.pinned(NEWEST_EXCHANGES), default=None)
            self._memory.pin([at for at in (newest, len(self._memory.tokens)) if at is not None],
                             NEWEST_EXCHANGES)
            end = len(self._memory.tokens) + len(tokens)
            self.pin_the_reply(self._rendering.tokenize(ahead.reply_opening), tokens, end)
        self._state.messages.append(question)
        self._memory.decode(tokens, want_logits=True)
        self._state.spliced_at = (len(self._memory.tokens) - len(tokens), tokens,
                                  self._made_for_now())
        self._note_prompt(ahead.after_question, ahead.reply_opening, self._memory.tokens)
        return True

    def pin_the_reply(self, opening: list[int], tokens: list[int], end: int) -> None:
        """Keep a state where the reply after ``tokens``, decoded to ``end``, begins, so that a reply written again, or shortened, decodes only itself and its opening."""
        starts = end - len(opening) if opening and tokens[-len(opening):] == opening else end
        self._memory.pin([starts if end - len(tokens) < starts else end]
                         if self.reply_kept_from_its_start else [], THE_REPLY)

    @property
    def reply_kept_from_its_start(self) -> bool:
        """Whether a state is kept where the reply begins: only where the template writes a past reply otherwise than it was generated, which takes the cache back there after every reply. See docs/decisions/0024-saved-states-on-a-cache-that-cannot-cut.md."""
        return bool(self._state.reasoning) or not self._template().properties.writes_reply_as_prompted

    def pin_opening(self) -> None:
        """Keep states saved where the conversation's opening ends, and where the decode ahead of a first question stops, where the cache goes back to saved states."""
        if not self._memory.keeps_saved_states:
            self._memory.pin([], "opening")
            return
        system = system_part(self._state.messages)
        places = {self._rendering.opening_length(system)}
        asked = system + [{"role": "user", "content": STAND_IN}]
        split = self._rendering.split(asked, generation_prompt=True)
        run = self._rendering.run_start(split, len(asked) - 1, 0) if split is not None else None
        if run is not None:
            ahead = len(self._rendering.target(asked, True, upto=run).tokens)
            # Not kept within a few tokens of the opening's state, from which going back there
            # decodes them instead.
            if ahead - min(places) > _NEAR_STATE_TOKENS:
                places.add(ahead)
        self._memory.pin(sorted(places), "opening")

    def render_ahead(self, dropped: bool = False, begins: bool = False) -> None:
        """Decode the render for the next question up to where the question goes; ``begins`` where a begin writes it."""
        messages = self._state.messages + [{"role": "user", "content": STAND_IN}]
        view = self._targets.view(messages, generation_prompt=True)
        split = view.split if view is not None else None
        run = (self._rendering.run_start(split, len(split.contents) - 1, view.start)
               if view is not None and split is not None and split.text.count(STAND_IN) == 1
               else None)
        if run is not None and view is not None and split is not None:
            before = self._targets.view_target(view, upto=run)
            if len(before.tokens) <= self._config.context_size:
                self.sync(before, logits=False, dropped=dropped, begins=begins)
                self.note_ahead(split, run)
                return
        # The next question will be rendered when it comes.
        if self._state.messages and self._rendering.split(self._state.messages, False) is None:
            return
        self.sync(self._targets.target_of(self._state.messages, generation_prompt=False),
                   logits=False, dropped=dropped, begins=begins)

    # ----- the cache against the target -----------------------------------------
    def sync(self, target: Target, *, logits: bool = True, dropped: bool = False,
              begins: bool = False) -> None:
        """Make the cache hold ``target``, and note where its generated turns begin there.

        Args:
            logits: A generation follows the decode at once.
            dropped: Exchanges were dropped, so that the target may go on as
                the cache does after a stretch.
            begins: A begin writes the target, so that the cache may be one
                kept from another conversation.
        """
        if self._memory.keeps_saved_states:
            # States there let a change to the newest exchanges decode only those exchanges.
            places = (self._newest_exchanges_at() if target.exchanges_at is None
                      else self._with_older_pinned(target.exchanges_at))
            if places is not None:
                self._memory.pin(places, NEWEST_EXCHANGES)
        # Declarations that move with the newest question leave a stretch behind, and the
        # next question's move back to where this cut is made, or past it.
        moves = self.moves_declarations()
        # The anchors before the first token the target changes still stand.
        same = self._memory.hold(target.tokens, want_logits=logits,
                                 find_dropped_stretch=dropped or moves, rest_at_cut=moves,
                                 begins=begins)
        self._state.turns_at = target.turns_at
        messages = self._state.messages
        new = {index: Anchor(messages[index], index, at, tuple(
                   turn for _, _, turn in generated_turns_in(messages[:index + 1])))
               for index, at in target.anchors
               if index < len(messages) and closes_turn(messages[index])}
        standing = {a.index: a for a in self._state.anchors
                    if a.at < same and a.index < min(new, default=len(messages))}
        self._state.anchors = tuple(sorted({**standing, **new}.values(),
                                           key=lambda a: a.index)[-2:])

    def _with_older_pinned(self, places: tuple[int, ...]) -> list[int]:
        """``places``, where a window placed the tags of the newest exchanges it holds, with the newest pinned before them where it holds only one."""
        if len(places) >= 2:
            return list(places)
        older = [at for at in self._memory.pinned(NEWEST_EXCHANGES) if not places or at < places[0]]
        return ([max(older)] if older else []) + list(places)

    def _newest_exchanges_at(self) -> list[int] | None:
        """Where in the cache each of the two newest exchanges begins, the tag that opens its question, those that cannot be told left out; ``None`` where neither this action nor the one before rendered the conversation whole in either form, which it is not rendered again for."""
        messages = self._state.messages
        # Not the first after the system turn, whose tag may stand inside it: the opening's
        # state stands there.
        starts = [at for at in exchange_starts(messages)[-2:] if at and messages[at - 1]["role"] != "system"]
        for prompted in (False, True):
            split = self._rendering.split_made(messages, prompted)
            if split is not None:
                break
        else:
            return None
        tags = [self._rendering.tag_after_content(split, at - 1) for at in starts]
        return [len(self._rendering.target(messages, prompted, upto=tag).tokens)
                for tag in tags if tag is not None]

    def moves_declarations(self) -> bool:
        """Whether the declarations move with the newest question, leaving a stretch behind."""
        return bool(self._state.tools) and self._template().properties.declares_before_newest_turn

    def prompted(self) -> Target:
        """The conversation with the generation prompt after it, noting the template text a reply follows."""
        view = self._targets.view(self._state.messages, generation_prompt=True)
        if view is None:
            self._rendering.target(self._state.messages, generation_prompt=True)   # its refusal
        assert view is not None, "a conversation without a view is one the template refuses"
        target, split = self._targets.view_target(view), view.split
        last = max((end for _, end in split.own), default=0)
        self._note_prompt(split.text[last:], self._reply_opening(split), target.tokens)
        return target

    def prompt_text_start(self, target: list[int]) -> int:
        """Where in ``target`` the template text the next reply follows begins, the target's end where it does not end on it."""
        tokens = self._rendering.tokenize([(self._state.prompt_text, False)])
        ends = tokens and list(target[-len(tokens):]) == tokens
        return len(target) - (len(tokens) if ends else 0)

    def _reply_opening(self, split: SplitRender) -> list[tuple[str, bool]]:
        """The render ``split`` from where a reply after its newest message begins as a generated turn."""
        start = self._rendering.tag_after_content(split, len(split.contents) - 1)
        return split.pieces(start) if start is not None else []

    def _note_prompt(self, prompt_text: str, reply_opening: list[tuple[str, bool]],
                     target) -> None:
        """Note the template text a reply follows, and where in ``target`` the reply's generated turn begins: at ``reply_opening``, the render's end."""
        self._state.prompt_text = prompt_text
        tokens = self._rendering.tokenize(reply_opening)
        ends = tokens and list(target[-len(tokens):]) == tokens
        self._state.prompt_start = len(target) - (len(tokens) if ends else 0)

    # ----- what a record made ahead was made for ----------------------------------------
    def note_ahead(self, split: SplitRender, run: int) -> None:
        """Note ``split``, decoded up to ``run``, as the render made ahead for the conversation and the cache as they are now."""
        question = split.contents[-1]
        assert question is not None, "a render made ahead holds the stand-in question"
        self._state.ahead = Ahead(rest=split.pieces(run),
                                   after_question=split.text[question[1]:],
                                   reply_opening=self._reply_opening(split),
                                   made_for=self._made_for_now())

    def _made_for_now(self) -> MadeFor:
        """The conversation and the cache as they are now, for a record made of them."""
        state = self._state
        return MadeFor(tuple(state.messages), tuple(state.tools), state.reasoning,
                        state.checks.opening, self._memory.version)

    def is_current(self, made_for: MadeFor) -> bool:
        """Whether the conversation and the cache are still those ``made_for`` was made for."""
        return (made_for.cache_version == self._memory.version
                and len(made_for.messages) == len(self._state.messages)
                and goes_on_from(self._state, made_for))
