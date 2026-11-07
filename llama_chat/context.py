"""KV-cache operations: tokenize, prefill, generate, evict and shift.

Owns the model, context and sampler. Which messages to keep is not decided
here; that policy lives in the wrapper.
"""

from __future__ import annotations

import codecs
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from ._backend import Backend, set_log_level
from .config import Config, Sampling
from .messages import Eviction
from .template import TemplateFragments

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


@dataclass
class GenerationAccumulator:
    """What one generation has produced so far, filled as tokens stream in.

    Every token is decoded into the cache before its text is yielded, so a
    caller that abandons the stream still reads exactly what reached the cache.

    Attributes:
        token_ids: Every token decoded into the cache.
        text: Visible reply text so far, trimmed at a stop string if one hit.
        stop_reason: ``"end_of_turn"``, ``"stop_string"``, ``"token_limit"``,
            ``"interrupted"`` or ``"tool_call"``.
        tool_call: What the model generated between its call tags, tags
            excluded, when ``stop_reason`` is ``"tool_call"``. Never part of
            ``text``.
    """

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    stop_reason: str = "token_limit"
    tool_call: str = ""


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
        self._llama_context = None
        self._sampler = None
        try:
            kv_type = (self._backend.kv_cache_type_id(config.kv_cache_type)
                       if config.kv_cache_type else None)
            self._llama_context = self._backend.new_context(
                self._model, config.context_size, config.decode_threads, config.max_batch_tokens,
                flash_attn=config.flash_attention, type_k=kv_type, type_v=kv_type,
                kv_cache_on_gpu=config.kv_cache_on_gpu,
                n_threads_batch=config.prefill_threads)
            self._vocab = self._backend.vocab(self._model)
            self._max_batch_tokens = config.max_batch_tokens
            self._settle_cache_edits()
            self._sampler = self._build_sampler(config)
            self._model_formatter = self._build_model_formatter()
        except BaseException:
            # Free whatever was created before the failure.
            if self._sampler is not None:
                self._backend.free_sampler(self._sampler)
            if self._llama_context is not None:
                self._backend.free_context(self._llama_context)
            self._backend.free_model(self._model)
            raise
        self._closed = False

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

    def tokenize_fragment(self, prefix: str, content: str, suffix: str,
                          add_special: bool = False) -> list[int]:
        """Tokenize one templated message: tags special-on, content special-off.

        ``add_special`` attaches BOS to the leading ``prefix`` only. See the
        README, "Special tokens in message content".
        """
        toks = self.tokenize(prefix, add_special=add_special, parse_special=True)
        toks += self.tokenize(content, add_special=False, parse_special=False)
        toks += self.tokenize(suffix, add_special=False, parse_special=True)
        return toks

    def tokenizes_to_tags(self, text: str) -> bool:
        """Whether ``text`` tokenizes into the model's own tags and nothing else.

        A tag is a control token or one a fine-tune added. See the README,
        "Tool calling".
        """
        toks = self.tokenize(text, add_special=False)
        return bool(toks) and all(self._backend.is_special(self._vocab, t) for t in toks)

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
                                   reasoning: bool | None = None) -> str | None:
        """Render ``messages`` with the model's own GGUF chat template.

        BOS is suppressed, so the render aligns with a fragment prefill.

        Args:
            generation_prompt: Append the prompt a reply is generated after.
            reasoning: Set the template's reasoning switch, ``enable_thinking``,
                the variable chat templates share for it. ``None`` leaves it
                unset.

        Returns:
            The prompt string, or ``None`` where the model ships no template or
            the template rejects this message shape.
        """
        if self._model_formatter is None:
            return None
        switch = {} if reasoning is None else {"enable_thinking": reasoning}
        try:
            return self._model_formatter[generation_prompt](
                messages=messages, tools=tools, **switch).prompt
        except Exception:
            return None

    def extract_fragments(self) -> TemplateFragments | None:
        """Recover the per-role template fragments from the model's own template.

        Returns:
            The fragments, or ``None`` where the model ships no template.
        """
        from .template_extract import extract_fragments
        return extract_fragments(self.render_with_model_template)

    def _build_model_formatter(self):
        """One renderer without the generation prompt and one with it, or ``None``."""
        template = self._backend.metadata_value(self._model, "tokenizer.chat_template")
        if not template:
            return None
        eos = self._backend.eos_text(self._vocab)
        return {prompted: self._backend.template_renderer(template, eos, prompted)
                for prompted in (False, True)}

    # ----- cache edits ---------------------------------------------------
    def reset(self) -> None:
        self._backend.kv_clear(self._llama_context)

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None:
        self._backend.decode(self._llama_context, token_ids, start_pos, SEQ, want_logits, self._max_batch_tokens)

    def warmup(self) -> None:
        """Warm the single-token generation graph, then clear the cache again.

        Prefill warms only the batched graph, so without this the first reply
        pays a one-time graph-build cost.
        """
        tok = self._backend.token_eos(self._vocab)
        if tok < 0:  # no EOS; any valid id warms the graph
            tok = 0
        self._backend.decode(self._llama_context, [tok], 0, SEQ, True, self._max_batch_tokens)
        self.reset()  # kv_clear -> cache empty again for the real prefill

    def apply_eviction(self, ev: Eviction) -> None:
        """Remove the evicted tokens, then shift survivors to close the gap.

        Raises:
            RuntimeError: If the cache refuses the removal, which a cache that
                can shift does only when it is corrupted.
        """
        if not self._backend.kv_seq_rm(self._llama_context, SEQ, ev.remove_start,
                                       ev.remove_end):
            raise RuntimeError(f"the cache refused to remove tokens "
                               f"{ev.remove_start} to {ev.remove_end}")
        if ev.remove_end < ev.old_total:
            self._backend.kv_seq_add(
                self._llama_context, SEQ, ev.remove_end, ev.old_total, -ev.shift_delta
            )

    def save_state(self):
        """What a cache that cannot shift needs to be put back to where it rests now.

        ``None`` where this build cannot save it, a rebuild then standing in.
        """
        if not self._backend.can_save_state:
            return None
        return self._backend.save_state(self._llama_context, SEQ)

    def restore_state(self, state, pos: int) -> None:
        """Put the cache back to ``state``, saved when it held ``pos`` tokens.

        Raises:
            RuntimeError: If the cache refuses the state or the cut after it.
        """
        self._backend.restore_state(self._llama_context, SEQ, state)
        self.remove_from(pos)

    def remove_from(self, pos: int) -> None:
        """Remove every cached token from ``pos`` to the end of the sequence.

        Raises:
            RuntimeError: If the cache refuses a partial removal.
        """
        if not self._backend.kv_seq_rm(self._llama_context, SEQ, pos, -1):
            raise RuntimeError(f"the cache refused to remove its tokens from {pos} on")

    # ----- generation ----------------------------------------------------
    def generate(
        self, start_pos: int, max_generated_tokens: int, stop: list[str],
        out: GenerationAccumulator | None = None,
        cancel: threading.Event | None = None,
        tool_tokens: tuple[int, int] | None = None,
        sampling: Sampling | None = None,
        turn_end: int | None = None,
    ):
        """Stream a reply, yielding text deltas as tokens are sampled.

        The caller must have prefilled the generation prompt with
        ``want_logits=True``. Bytes run through an incremental UTF-8 decoder, so
        a codepoint split across tokens is never mangled, and the trailing
        characters are held back while they could still begin a stop string.

        Args:
            start_pos: Cache position of the first generated token.
            max_generated_tokens: Hard cap on tokens to generate.
            stop: Stop strings; generation ends just before the earliest match.
            out: Accumulator to fill. Pass one in to read what reached the cache
                after abandoning the stream; a fresh one is used otherwise.
            cancel: Checked once per token. Setting it, from any thread, ends
                the generation with ``stop_reason == "interrupted"``.
            tool_tokens: The ``(open, close)`` tokens bracketing a tool call.
                Given them, a call is recognized at its opening token, none of
                it is yielded, and generation ends on the closing token with
                ``stop_reason == "tool_call"``. A call the token cap cuts short
                is dropped, and the reply ends with ``"token_limit"``.
            sampling: This generation's own settings. One that overrides
                anything gets a sampler chain of its own, freed when the
                generation ends; without one the chain built at load is used.
            turn_end: A token that ends the reply as an end-of-generation token
                does, neither cached nor yielded: the first of a closing the
                model writes in several tags.

        Yields:
            Visible reply text deltas; their concatenation equals ``out.text``.
        """
        own_sampler = None
        if sampling is not None and sampling.overrides:
            own_sampler = self._build_sampler(sampling.applied_to(self._config))
        else:
            # Penalties count this generation's tokens only, and a seed
            # starts over, as in a chain built for this generation alone.
            self._backend.sampler_reset(self._sampler)
        try:
            yield from self._sample_into(
                own_sampler if own_sampler is not None else self._sampler,
                start_pos, max_generated_tokens,
                stop, out, cancel, tool_tokens, turn_end)
        finally:
            if own_sampler is not None:
                self._backend.free_sampler(own_sampler)

    def _sample_into(self, sampler, start_pos: int, max_generated_tokens: int,
                     stop: list[str], out: GenerationAccumulator | None,
                     cancel: threading.Event | None,
                     tool_tokens: tuple[int, int] | None, turn_end: int | None = None):
        """The generation loop of :meth:`generate`, drawing from ``sampler``."""
        generated = out if out is not None else GenerationAccumulator()
        pos = start_pos
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        longest_stop_chars = max((len(s) for s in stop if s), default=0)
        visible = _TextBuffer()
        call_open, call_close = tool_tokens if tool_tokens else (None, None)
        call = None            # pieces of the call being generated, once one is

        for _ in range(max(0, max_generated_tokens)):
            if cancel is not None and cancel.is_set():
                generated.stop_reason = "interrupted"
                break
            tok = self._backend.sample(sampler, self._llama_context)
            # A call may close with the token that otherwise ends a turn.
            if (self._backend.is_eog(self._vocab, tok) or tok == turn_end) and not (
                    call is not None and tok == call_close):
                generated.stop_reason = "end_of_turn"
                break
            # Cache, then record, then surface: this order is what keeps
            # token_ids equal to the cache across an early close.
            self._backend.decode(self._llama_context, [tok], pos, SEQ, True, self._max_batch_tokens)
            pos += 1
            generated.token_ids.append(tok)
            piece = decoder.decode(self._backend.token_to_piece_bytes(self._vocab, tok))

            if call is not None:
                if tok == call_close:
                    generated.stop_reason = "tool_call"
                    generated.tool_call = "".join(call)
                    break
                call.append(piece)
                continue
            if tok == call_open:
                # The held-back tail is ordinary reply text; release it.
                call = []
                upto = len(visible)
            else:
                visible.append(piece)
                if longest_stop_chars:
                    # Only the unemitted tail can still begin a stop string, so
                    # the repeated scans stay linear in the reply.
                    hit = _stop_hit(visible.text, stop, search_from=visible.emitted)
                    if hit is not None:
                        generated.stop_reason = "stop_string"
                    upto = (hit if hit is not None
                            else len(visible) - (longest_stop_chars - 1))
                else:
                    upto = len(visible)
            delta = visible.emit(upto)
            if delta:
                generated.text += delta
                try:
                    yield delta
                except GeneratorExit:
                    # A reply that already ended at a stop string keeps that
                    # reason; the close only declined the final resume.
                    if generated.stop_reason != "stop_string":
                        generated.stop_reason = "interrupted"
                    raise
            if generated.stop_reason == "stop_string":
                break
        else:
            generated.stop_reason = "token_limit"

        # Flush the held-back tail unless a stop string truncated the reply.
        if generated.stop_reason not in ("stop_string", "tool_call"):
            visible.append(decoder.decode(b"", final=True))
            delta = visible.emit(len(visible))
            if delta:
                generated.text += delta
                yield delta

    # ----- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Free the sampler, context and model. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._backend.free_sampler(self._sampler)
        finally:
            self._backend.free_context(self._llama_context)
            self._backend.free_model(self._model)

    # ----- internals -----------------------------------------------------
    def _verify_cache_edits(self) -> None:
        """Perform each cache edit once, empty, so a build missing one fails here.

        Removing an empty range and shifting it by nothing changes nothing.
        Which function performs an edit is settled per build and per context,
        so an edit that works here works for the rest of the conversation.
        Left to the first eviction, a missing one would surface deep into a
        conversation. A cache that cannot shift is never shifted, and some
        abort the process when asked to, so it is not asked here either.

        Raises:
            RuntimeError: If this build offers no function for one of them.
        """
        self._backend.kv_seq_rm(self._llama_context, SEQ, 0, 0)
        if self._can_shift:
            self._backend.kv_seq_add(self._llama_context, SEQ, 0, 0, 0)
        self._backend.kv_clear(self._llama_context)

    def _settle_cache_edits(self) -> None:
        """Which edits this cache takes, each proven on it while it is empty."""
        self._can_cut_tail = self._probe_tail_cut()
        # A running state refuses the cut, and is not shifted whatever the cache reports.
        self._can_shift = self._can_cut_tail and self._backend.can_shift(self._llama_context)
        self._verify_cache_edits()

    def _probe_tail_cut(self) -> bool:
        """Whether three decoded tokens can be cut back to one, cleared either way."""
        # A cut at a running state's last position may clear it whole and succeed.
        token = max(self._backend.token_eos(self._vocab), 0)
        try:
            self._backend.decode(self._llama_context, [token] * 3, 0, SEQ, False,
                                 self._max_batch_tokens)
            return self._backend.kv_seq_rm(self._llama_context, SEQ, 1, -1)
        finally:
            self._backend.kv_clear(self._llama_context)

    def _build_sampler(self, config: Config):
        """Assemble the sampler chain ``config`` asks for, in sampling order."""
        backend = self._backend
        chain = backend.sampler_chain()
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


def _stop_hit(text: str, stop: list[str], search_from: int = 0) -> int | None:
    """Return the index where the earliest stop string begins, or ``None``.

    ``search_from`` skips text already known to begin no stop string.
    """
    earliest: int | None = None
    for s in stop:
        if not s:
            continue
        i = text.find(s, search_from)
        if i != -1 and (earliest is None or i < earliest):
            earliest = i
    return earliest
