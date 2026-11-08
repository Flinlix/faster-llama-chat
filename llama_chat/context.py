"""KV-cache operations: tokenize, prefill, generate, evict and shift.

Owns the model, context and sampler. What the cache should hold is decided by
the caller.
"""

from __future__ import annotations

import codecs
import os
import threading
from collections.abc import Callable

from ._backend import Backend, set_log_level
from .config import Config, Sampling
from .cache_context import Eviction, GenerationAccumulator
from .reply_text import earliest
from .template import CallBoundary
from .addon_files import addon_file
from .model_sampling import with_model_sampling
from .model_template import ModelTemplate

SEQ = 0  # single active conversation -> single sequence id


class _TextBuffer:
    """Decoded text with a cursor tracking how much has been emitted."""

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0

    def append(self, text: str) -> None:
        self._buf += text

    def emit(self, upto: int) -> str | None:
        if upto > self._emitted:
            delta = self._buf[self._emitted:upto]
            self._emitted = upto
            return delta
        return None

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def text(self) -> str:
        return self._buf

    def __len__(self) -> int:
        return len(self._buf)


class KVContext:
    """Owns the llama.cpp model and context, and performs every cache edit.

    Args:
        config: Runtime configuration.
        on_load_progress: Called during weight loading with a fraction in
            ``[0.0, 1.0]``. Must not re-enter the context and must not raise,
            because a raise aborts the load.

    Raises:
        ValueError: If ``config.model_path`` is unset.
        FileNotFoundError: If it names no file.
        RuntimeError: If the model or the context cannot be created, or this
            build cannot perform one of the cache edits eviction rests on.
    """

    def __init__(
        self, config: Config,
        on_load_progress: Callable[[float], None] | None = None,
    ) -> None:
        if not config.model_path:
            raise ValueError("config.model_path is unset; a model cannot be "
                             "loaded without one")
        if not os.path.isfile(config.model_path):
            raise FileNotFoundError(f"no model file at {config.model_path!r}")
        self._config = config
        set_log_level(config.log_level)  # filter llama.cpp's load-time logs too
        self._backend = Backend()
        self._model = self._backend.load_model(
            config.model_path, config.gpu_layers, progress=on_load_progress)
        self._engine_context = None
        self._sampler = None
        try:
            template = self._backend.metadata_value(self._model, "tokenizer.chat_template")
            addon = addon_file(config.model_path, template)      # read once, for both uses
            self._config = config = with_model_sampling(
                config, lambda key: self._backend.metadata_value(self._model, key), addon)
            self._addon_call = addon["call"]
            kv_type = (self._backend.kv_cache_type_id(config.kv_cache_type)
                       if config.kv_cache_type else None)
            self._engine_context = self._backend.new_context(
                self._model, config.context_size, config.decode_threads, config.max_batch_tokens,
                flash_attn=config.flash_attention, type_k=kv_type, type_v=kv_type,
                kv_cache_on_gpu=config.kv_cache_on_gpu,
                n_threads_batch=config.prefill_threads)
            self._vocab = self._backend.vocab(self._model)
            self._max_batch_tokens = config.max_batch_tokens
            self._settle_cache_edits()
            self._sampler = self._build_sampler(config)
            self._model_template = (ModelTemplate(template, self._backend.eos_text(self._vocab))
                                    if template else None)
            self._tags = None
        except BaseException:
            # Free whatever was created before the failure.
            if self._sampler is not None:
                self._backend.free_sampler(self._sampler)
            if self._engine_context is not None:
                self._backend.free_context(self._engine_context)
            self._backend.free_model(self._model)
            raise
        self._closed = False

    @property
    def config(self) -> Config:
        """The configuration this context runs with, the sampling the model supplies filled in."""
        return self._config

    @property
    def context_size(self) -> int:
        """How many tokens the cache was built to hold."""
        return self._config.context_size

    @property
    def can_shift(self) -> bool:
        """Whether eviction can shift survivors in place, rather than rebuild."""
        return self._can_shift

    @property
    def can_cut_tail(self) -> bool:
        """Whether the cache takes a removal from a position to its end.

        A cache that shifts always does; one that cannot shift may still,
        where only positions, not a running state, keep it from shifting.
        """
        return self._can_cut_tail

    # ----- tokenization --------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False,
                 parse_special: bool = True) -> list[int]:
        return self._backend.tokenize(self._vocab, text, add_special, parse_special)

    def _tags_in(self, text: str) -> list[str]:
        """The texts of the model's own tags in ``text``, in order."""
        return [self.token_text(token) for token in self.tokenize(text, parse_special=True)
                if self._backend.is_special(self._vocab, token)]

    def split_at_tags(self, text: str) -> list[tuple[str, bool]] | None:
        """``text`` as the model's tags and the plain text between them, each marked whether a tag.

        ``None`` where a tag the tokenizer finds cannot be located in the text.
        """
        pieces, at = [], 0
        for tag in self._tags_in(text):
            start = text.find(tag, at)
            if not tag or start < 0:
                return None
            if start > at:
                pieces.append((text[at:start], False))
            pieces.append((tag, True))
            at = start + len(tag)
        if at < len(text):
            pieces.append((text[at:], False))
        return pieces

    def tags(self) -> tuple[str, ...]:
        """The texts of every one of the model's own tags, read once."""
        if self._tags is None:
            self._tags = tuple(self.token_text(token)
                               for token in range(self._backend.n_tokens(self._vocab))
                               if self._backend.is_special(self._vocab, token))
        return self._tags

    def first_special_token(self, text: str) -> int | None:
        """The first of the model's own tags in ``text``, or ``None``.

        The first tag, not the first token: a recovered tool tag carries fixed
        lead-in text. See the README, "Tool calling".
        """
        for token in self.tokenize(text, add_special=False, parse_special=True):
            if self._backend.is_special(self._vocab, token):
                return token
        return None

    def token_text(self, token: int) -> str:
        """The text a single token stands for, the model's own tags included."""
        return self._backend.token_to_piece(self._vocab, token)

    def render_with_model_template(self, messages: list[dict],
                                   tools: list[dict] | None = None, *,
                                   generation_prompt: bool = False,
                                   reasoning: bool | None = None,
                                   **variables) -> str | None:
        """Render ``messages`` with the model's own chat template; see :meth:`ModelTemplate.render`.

        ``None`` where the model ships no template or the template rejects
        this message shape. BOS is left out: the sequence opener is the
        tokenizer's to add.
        """
        if self._model_template is None:
            return None
        return self._model_template.render(messages, tools, generation_prompt=generation_prompt,
                                           reasoning=reasoning, **variables)

    def addon_call(self) -> dict:
        """How the model's add-on file says a generated call is written, if it says."""
        return dict(self._addon_call)

    def chat_template_text(self) -> str | None:
        """The source of the model's own chat template, or ``None`` where it ships none."""
        return None if self._model_template is None else self._model_template.text

    def reset(self) -> None:
        self._backend.kv_clear(self._engine_context)

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None:
        self._backend.decode(self._engine_context, token_ids, start_pos, SEQ, want_logits, self._max_batch_tokens)

    def warmup(self) -> None:
        """Decode a full batch and a single token, then clear the cache again.

        The engine prepares each size of batch the first time it decodes one,
        so without this the first reply pays that cost: a full batch covers
        every prefill, the single token every generated token.
        """
        tok = self._backend.token_eos(self._vocab)
        if tok < 0:  # no EOS; any valid id warms the graph
            tok = 0
        full = min(self._max_batch_tokens, self._config.context_size)
        for tokens in ([tok] * full, [tok]):
            self._backend.decode(self._engine_context, tokens, 0, SEQ, True, self._max_batch_tokens)
            self.reset()  # kv_clear -> cache empty again for the real prefill

    def apply_eviction(self, ev: Eviction) -> None:
        """Remove the evicted tokens, then shift survivors to close the gap.

        Raises:
            RuntimeError: If the cache refuses the removal, which a cache that
                can shift does only when it is corrupted.
        """
        if not self._backend.kv_seq_rm(self._engine_context, SEQ, ev.remove_start,
                                       ev.remove_end):
            raise RuntimeError(f"the cache refused to remove tokens "
                               f"{ev.remove_start} to {ev.remove_end}")
        if ev.remove_end < ev.old_total:
            self._backend.kv_seq_add(
                self._engine_context, SEQ, ev.remove_end, ev.old_total, -ev.shift_delta
            )

    def save_state(self):
        """What a cache that cannot shift needs to be put back to where it rests now.

        ``None`` where this build cannot save it, a rebuild then standing in.
        """
        if not self._backend.can_save_state:
            return None
        return self._backend.save_state(self._engine_context, SEQ)

    def restore_state(self, state, pos: int) -> None:
        """Put the cache back to ``state``, saved when it held ``pos`` tokens.

        Raises:
            RuntimeError: If the cache refuses the state or the cut after it.
        """
        self._backend.restore_state(self._engine_context, SEQ, state)
        self.remove_from(pos)

    def remove_from(self, pos: int) -> None:
        """Remove every cached token from ``pos`` to the end of the sequence.

        Raises:
            RuntimeError: If the cache refuses a partial removal.
        """
        if not self._backend.kv_seq_rm(self._engine_context, SEQ, pos, -1):
            raise RuntimeError(f"the cache refused to remove its tokens from {pos} on")

    # ----- generation ----------------------------------------------------
    def generate(
        self, start_pos: int, max_generated_tokens: int,
        out: GenerationAccumulator | None = None,
        cancel: threading.Event | None = None,
        sampling: Sampling | None = None,
        ends_turn: Callable[[int], bool] | None = None,
        boundary: CallBoundary = CallBoundary(),
    ):
        """Stream a reply, yielding text deltas as tokens are sampled.

        The caller must have prefilled the generation prompt with
        ``want_logits=True``. Bytes run through an incremental UTF-8 decoder, so
        a codepoint split across tokens is never mangled, and the trailing
        characters are held back while they could still begin a call's opening
        or the text that ends a reply.

        Args:
            start_pos: Cache position of the first generated token.
            max_generated_tokens: Hard cap on tokens to generate.
            out: Accumulator to fill. Pass one in to read what reached the cache
                after abandoning the stream; a fresh one is used otherwise.
            cancel: Checked once per token. Setting it, from any thread, ends
                the generation with ``stop_reason == "interrupted"``.
            sampling: This generation's own settings. One that overrides
                anything gets a sampler chain of its own, freed when the
                generation ends; without one the chain built at load is used.
            ends_turn: Whether a sampled token ends the reply as an
                end-of-generation token does, neither cached nor yielded: the
                first of a closing the model writes in several tags.
            boundary: Where a call opens and closes and where a reply ends in
                text. A call is recognized where it opens, none of it is
                yielded, and generation ends where it closes with
                ``stop_reason == "tool_call"``; one the token cap cuts short is
                dropped, and the reply ends with ``"token_limit"``.

        Yields:
            Visible reply text deltas; their concatenation equals ``out.text``.
        """
        own_sampler = None
        if boundary.withheld_tokens:
            own_sampler = self._build_sampler(
                sampling.applied_to(self._config) if sampling is not None else self._config,
                boundary.withheld_tokens)
        elif sampling is not None and sampling.overrides:
            own_sampler = self._build_sampler(sampling.applied_to(self._config))
        else:
            # Penalties count this generation's tokens only, and a seed
            # starts over, as in a chain built for this generation alone.
            self._backend.sampler_reset(self._sampler)
        try:
            yield from self._sample_into(
                own_sampler if own_sampler is not None else self._sampler,
                start_pos, max_generated_tokens, out, cancel, ends_turn, boundary)
        finally:
            if own_sampler is not None:
                self._backend.free_sampler(own_sampler)

    def _sample_into(self, sampler, start_pos: int, max_generated_tokens: int,
                     out: GenerationAccumulator | None, cancel: threading.Event | None,
                     ends_turn: Callable[[int], bool] | None, boundary: CallBoundary):
        """The generation loop of :meth:`generate`, drawing from ``sampler``."""
        generated = out if out is not None else GenerationAccumulator()
        pos = start_pos
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        visible = _TextBuffer()
        call_open, call_close = boundary.tool_tokens or (None, None)
        call = None            # pieces of the call being generated, once one is
        open_text, close_text = boundary.call_texts or (None, None)
        call_opens_as_text = boundary.opens_as_text and open_text is not None
        marks = [mark for mark in (open_text, boundary.turn_end_text) if mark]
        held_chars = max([0] + [len(mark) for mark in marks])
        starts = []            # where each token's text begins in ``visible``
        ended_at_text = False
        marks_from = 0         # where the next search for a mark begins
        held_tag = None        # where a tag a call may follow begins and ends, not shown yet

        for _ in range(max(0, max_generated_tokens)):
            if cancel is not None and cancel.is_set():
                generated.stop_reason = "interrupted"
                break
            tok = self._backend.sample(sampler, self._engine_context)
            ends = self._backend.is_eog(self._vocab, tok) or (
                ends_turn is not None and ends_turn(tok))
            # A call may close with the token that otherwise ends a turn.
            closes_call = call is not None and (tok == call_close or (
                close_text is not None and close_text in "".join(call) + self._backend
                .token_to_piece_bytes(self._vocab, tok).decode("utf-8", errors="replace")))
            if ends and not closes_call and call is not None and close_text is not None:
                # Ended inside a call: the call ends here, closed as the template closes one.
                closing = self.tokenize(close_text, add_special=False, parse_special=True)
                self._backend.decode(self._engine_context, closing, pos, SEQ, True,
                                     self._max_batch_tokens)
                pos += len(closing)
                generated.token_ids.extend(closing)
                generated.stop_reason = "tool_call"
                generated.tool_call = "".join(call)
                break
            if ends and not closes_call:
                generated.stop_reason = "end_of_turn"
                break
            # Cache, then record, then surface: this order is what keeps
            # token_ids equal to the cache across an early close.
            self._backend.decode(self._engine_context, [tok], pos, SEQ, True, self._max_batch_tokens)
            pos += 1
            generated.token_ids.append(tok)
            piece = decoder.decode(self._backend.token_to_piece_bytes(self._vocab, tok))

            if call is not None:
                if tok == call_close:
                    generated.stop_reason = "tool_call"
                    generated.tool_call = "".join(call)
                    break
                call.append(piece)
                if close_text is not None and close_text in "".join(call):
                    generated.stop_reason = "tool_call"
                    generated.tool_call = "".join(call).split(close_text, 1)[0]
                    break
                continue
            if tok == call_open:
                # The held-back tail is ordinary reply text; release it.
                call = []
                upto = len(visible)
            elif marks:
                starts.append(len(visible))
                visible.append(piece)
                if call_opens_as_text and self._backend.is_special(self._vocab, tok) \
                        and piece.strip() not in boundary.thought_tags:
                    held_tag = (starts[-1], len(visible))
                mark_at = earliest(visible.text, marks,
                                   search_from=max(visible.emitted, marks_from))
                while (mark_at is not None and call_opens_as_text
                       and visible.text.startswith(open_text, mark_at)
                       and not self._after_tag(visible.text, starts, generated.token_ids, mark_at)):
                    marks_from = mark_at + 1        # a mention, not a call
                    mark_at = earliest(visible.text, marks, search_from=marks_from)
                if mark_at is not None:
                    upto = mark_at
                    if open_text is not None and visible.text.startswith(open_text, mark_at):
                        call = [visible.text[mark_at + len(open_text):]]
                        if held_tag is not None:
                            upto = held_tag[0]          # the tag the call follows is the call's
                    else:
                        generated.stop_reason = "end_of_turn"
                        ended_at_text = True
                        # Its tokens leave the reply, and the caller cuts them
                        # from the cache: the closing is written whole.
                        if mark_at in starts:
                            kept = starts.index(mark_at)
                            generated.cached_beyond = len(generated.token_ids) - kept
                            del generated.token_ids[kept:]
                            pos = start_pos + kept
                else:
                    upto = len(visible) - (held_chars - 1)
                if held_tag is not None and call is None:
                    following = visible.text[held_tag[1]:].lstrip()
                    if following and not open_text.startswith(following[:len(open_text)]):
                        held_tag = None             # no call follows it: it is the reply's
                    else:
                        upto = min(upto, held_tag[0])
            else:
                visible.append(piece)
                upto = len(visible)
            delta = visible.emit(upto)
            if delta:
                generated.text += delta
                try:
                    yield delta
                except GeneratorExit:
                    generated.stop_reason = "interrupted"
                    raise
            if ended_at_text:
                break
        else:
            generated.stop_reason = "token_limit"

        # Flush the held-back tail unless the reply ended in a call or at its text.
        in_text_call = call is not None and close_text is not None
        if generated.stop_reason != "tool_call" and not ended_at_text and not in_text_call:
            visible.append(decoder.decode(b"", final=True))
            delta = visible.emit(len(visible))
            if delta:
                generated.text += delta
                yield delta

    def _after_tag(self, text: str, starts: list[int], tokens: list[int], at: int) -> bool:
        """Whether ``at`` is the reply's start or follows a tag, whitespace between."""
        before = len(text[:at].rstrip())
        if not before:
            return True
        last = max(i for i, start in enumerate(starts) if start < before)
        return self._backend.is_special(self._vocab, tokens[last])

    # ----- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Free the sampler, context and model. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._backend.free_sampler(self._sampler)
        finally:
            self._backend.free_context(self._engine_context)
            self._backend.free_model(self._model)

    # ----- internals -----------------------------------------------------
    def _verify_cache_edits(self) -> None:
        """Shift the empty cache by nothing, so a build missing the shift fails here.

        The cut probe has already removed and cleared. Which function performs
        an edit is settled per build and per context, so an edit that works
        here works for the rest of the conversation; left to the first
        eviction, a missing one would surface deep into a conversation. A
        cache that cannot shift is never shifted, and some abort the process
        when asked to, so it is not asked here either.

        Raises:
            RuntimeError: If this build offers no function for the shift.
        """
        if self._can_shift:
            self._backend.kv_seq_add(self._engine_context, SEQ, 0, 0, 0)

    def _settle_cache_edits(self) -> None:
        """Which edits this cache takes, each proven on it while it is empty."""
        self._can_cut_tail = self._probe_tail_cut()
        # A running state refuses the cut, and is not shifted whatever the cache reports.
        self._can_shift = self._can_cut_tail and self._backend.can_shift(self._engine_context)
        self._verify_cache_edits()

    def _probe_tail_cut(self) -> bool:
        """Whether three decoded tokens can be cut back to one, cleared either way."""
        # A cut at a running state's last position may clear it whole and succeed.
        token = max(self._backend.token_eos(self._vocab), 0)
        try:
            self._backend.decode(self._engine_context, [token] * 3, 0, SEQ, False,
                                 self._max_batch_tokens)
            return self._backend.kv_seq_rm(self._engine_context, SEQ, 1, -1)
        finally:
            self._backend.kv_clear(self._engine_context)

    def _build_sampler(self, config: Config, withheld: tuple[int, ...] = ()):
        """Assemble the sampler chain ``config`` asks for, in sampling order, ``withheld`` never drawn."""
        backend = self._backend
        chain = backend.sampler_chain()
        if withheld:
            backend.add_withheld(chain, self._vocab, withheld)
        if config.repetition_penalty != 1.0 \
                or config.presence_penalty or config.frequency_penalty:
            backend.add_penalties(chain, self._vocab, config.penalty_window_tokens,
                                  config.repetition_penalty,
                                  config.frequency_penalty,
                                  config.presence_penalty)
        if config.temperature <= 0.0:
            backend.add_greedy(chain)
            return chain
        if config.top_k > 0:
            backend.add_top_k(chain, config.top_k)
        backend.add_top_p(chain, config.top_p)
        if config.min_p > 0.0:
            backend.add_min_p(chain, config.min_p)
        backend.add_temperature(chain, config.temperature)
        # An unset seed asks for a fresh random one on every run.
        seed = config.seed if config.seed is not None else backend.fresh_seed_per_run
        backend.add_distribution(chain, seed)
        return chain

