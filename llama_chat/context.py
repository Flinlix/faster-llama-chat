"""KV-cache operations: tokenize, prefill, the steps a generation takes, evict and shift.

Owns the model and the context, and builds each generation's sampler. What
the cache should hold is decided by the caller, and how a generation goes on
by :mod:`llama_chat.generation`.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from ._backend import Backend, set_log_level
from .config import Config, Sampling, with_request_sampling
from .addon_files import addon_file, with_model_sampling
from .model_template import ModelTemplate

SEQ = 0  # single active conversation -> single sequence id


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
        try:
            template = self._backend.metadata_value(self._model, "tokenizer.chat_template")
            addon = addon_file(config.model_path, template)      # read once, for both uses
            self._config = config = with_model_sampling(
                config, lambda key: self._backend.metadata_value(self._model, key), addon)
            self._addon_call = addon["call"]
            kv_type = self._backend.kv_cache_type_id(config.kv_cache_type)
            self._engine_context = self._backend.new_context(
                self._model, config.context_size, config.decode_threads, config.max_batch_tokens,
                flash_attn=config.flash_attention, type_k=kv_type, type_v=kv_type,
                kv_cache_on_gpu=config.kv_cache_on_gpu,
                n_threads_batch=config.prefill_threads)
            self._vocab = self._backend.vocab(self._model)
            self._max_batch_tokens = config.max_batch_tokens
            self._settle_cache_edits()
            self._warm_up()
            self._model_template = (ModelTemplate(template, self._backend.eos_text(self._vocab))
                                    if template else None)
            self._special_tokens = None
        except BaseException:
            # Free whatever was created before the failure.
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

    def is_special(self, token: int) -> bool:
        """Whether ``token`` is one of the model's own tags."""
        return self._backend.is_special(self._vocab, token)

    def special_tokens(self) -> tuple[int, ...]:
        """Every token that is one of the model's own tags, read once."""
        if self._special_tokens is None:
            self._special_tokens = tuple(token for token in range(self._backend.n_tokens(self._vocab))
                                         if self._backend.is_special(self._vocab, token))
        return self._special_tokens

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

    def _warm_up(self) -> None:
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

    def apply_eviction(self, start: int, end: int) -> None:
        """Remove the evicted tokens, then shift survivors to close the gap.

        Raises:
            RuntimeError: If the cache refuses the removal, which a cache that
                can shift does only when it is corrupted.
        """
        if not self._backend.kv_seq_rm(self._engine_context, SEQ, start, end):
            raise RuntimeError(f"the cache refused to remove tokens {start} to {end}")
        self._backend.kv_seq_add(self._engine_context, SEQ, end, -1, start - end)

    def save_state(self):
        """What a cache that cannot shift needs to be put back to where it rests now."""
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

    # ----- the steps a generation takes ----------------------------------
    def sampled_tokens(self, sampling: Sampling | None, withheld_tokens: tuple[int, ...]):
        """Tokens drawn one at a time from a sampler chain of their own, freed when this is closed."""
        # A chain of its own: penalties count this generation's tokens only, and a seed starts over.
        sampler = self._build_sampler(with_request_sampling(self._config, sampling),
                                      withheld_tokens)
        try:
            while True:
                yield self._backend.sample(sampler, self._engine_context)
        finally:
            self._backend.free_sampler(sampler)

    def decode_generated(self, token_ids: list[int], start_pos: int) -> None:
        self._backend.decode(self._engine_context, token_ids, start_pos, SEQ, True,
                             self._max_batch_tokens)

    def ends_generation(self, token: int) -> bool:
        return self._backend.is_eog(self._vocab, token)

    def token_bytes(self, token: int) -> bytes:
        return self._backend.token_to_piece_bytes(self._vocab, token)

    # ----- lifecycle -----------------------------------------------------
    def close(self) -> None:
        """Free the context and model. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._backend.free_context(self._engine_context)
        self._backend.free_model(self._model)

    # ----- internals -----------------------------------------------------
    def _settle_cache_edits(self) -> None:
        """Which edits this cache takes, each proven on it while it is empty."""
        self._can_cut_tail = self._probe_tail_cut()
        # A running state refuses the cut, and is not shifted whatever the cache reports.
        self._can_shift = self._can_cut_tail and self._backend.can_shift(self._engine_context)
        # Shifted by nothing, so a build missing the shift fails at load, not at an eviction.
        if self._can_shift:
            self._backend.kv_seq_add(self._engine_context, SEQ, 0, 0, 0)

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
        try:
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
        except BaseException:
            backend.free_sampler(chain)
            raise

