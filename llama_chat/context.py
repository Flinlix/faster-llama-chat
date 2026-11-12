"""KV-cache operations: tokenize, prefill, the steps a generation takes, evict and shift.

Owns the model and the context, reads the sampling the model's GGUF
recommends, and builds each generation's sampler. What the cache should hold
is decided by the caller, and how a generation goes on by
:mod:`llama_chat.generation`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from ._backend import Backend
from .cache_context import TOOL_USE_TEMPLATE
from .config import UNQUANTIZED_KV_CACHE_TYPE, Config, InvalidSettingError, Sampling
from .model_template import ModelTemplate

SEQ = 0  # single active conversation -> single sequence id
# The GGUF keys that state the sampling a model recommends, by Sampling's names.
_GGUF_SAMPLING_KEYS = {"temperature": "general.sampling.temp", "top_k": "general.sampling.top_k",
                       "top_p": "general.sampling.top_p", "min_p": "general.sampling.min_p",
                       "repetition_penalty": "general.sampling.penalty_repeat",
                       "penalty_window_tokens": "general.sampling.penalty_last_n"}
# The fields among them that Config takes as whole numbers.
_WHOLE_NUMBER_FIELDS = {"top_k", "penalty_window_tokens"}
# The penalty window a GGUF states to mean the whole context, as llama.cpp reads it.
_WHOLE_CONTEXT_WINDOW = -1


class KVContext:
    """Owns the llama.cpp model and context, and performs every cache edit.

    Args:
        config: Runtime configuration.
        on_load_progress: Called during weight loading with a fraction in
            ``[0.0, 1.0]``. Must not re-enter the context and must not raise,
            because a raise aborts the load.

    Raises:
        InvalidSettingError: If ``config.model_path`` is unset or names no file.
        ValueError: If the chat template the model ships is not valid template
            code.
        RuntimeError: If the model or the context cannot be created, or this
            build cannot perform one of the cache edits eviction rests on.
    """

    def __init__(
        self, config: Config,
        on_load_progress: Callable[[float], None] | None = None,
    ) -> None:
        if not config.model_path:
            raise InvalidSettingError("model_path", "model_path is unset; a model cannot be "
                                                    "loaded without one")
        if not os.path.isfile(config.model_path):
            raise InvalidSettingError("model_path", f"no model file at {config.model_path!r}")
        self._config = config
        self._backend = Backend()
        self._model = self._backend.load_model(
            config.model_path, config.gpu_layers, progress=on_load_progress)
        self._engine_context = None
        try:
            template = self._backend.metadata_value(self._model, "tokenizer.chat_template")
            kv_type = self._backend.kv_cache_type_id(config.kv_cache_type)
            self._engine_context = self._backend.new_context(
                self._model, config.context_size, config.decode_threads, config.max_batch_tokens,
                flash_attn=_flash_attention(config), type_k=kv_type, type_v=kv_type,
                kv_cache_on_gpu=config.kv_cache_on_gpu,
                n_threads_batch=config.prefill_threads)
            self._vocab = self._backend.vocab(self._model)
            self._max_batch_tokens = config.max_batch_tokens
            self._settle_cache_edits()
            self._warm_up()
            # The default template under None, and each further one the GGUF ships by its name.
            texts = {None: template, TOOL_USE_TEMPLATE: self._backend.metadata_value(
                self._model, f"tokenizer.chat_template.{TOOL_USE_TEMPLATE}")}
            self._model_templates = {name: ModelTemplate(text, self._backend.eos_text(self._vocab))
                                     for name, text in texts.items() if text}
            self._special_tokens = None
        except BaseException:
            # Free whatever was created before the failure.
            if self._engine_context is not None:
                self._backend.free_context(self._engine_context)
            self._backend.free_model(self._model)
            raise
        self._closed = False

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
                                   variables: Mapping[str, object] | None = None,
                                   template_name: str | None = None) -> str | None:
        """Render ``messages`` with the model's own chat template of that name; see :meth:`ModelTemplate.render`.

        ``None`` where the model ships no template of that name or the
        template rejects this message shape. BOS is left out: the sequence
        opener is the tokenizer's to add.
        """
        model_template = self._model_templates.get(template_name)
        if model_template is None:
            return None
        return model_template.render(messages, tools, generation_prompt=generation_prompt,
                                     reasoning=reasoning, variables=variables)

    def recommended_sampling(self) -> dict[str, float]:
        """The sampling the model's GGUF states, by :class:`Sampling`'s names.

        A penalty window stated as ``-1`` is the whole context, given as
        :attr:`context_size`.

        Raises:
            ValueError: If the GGUF states one of them as text that is no number.
        """
        stated = {}
        for name, key in _GGUF_SAMPLING_KEYS.items():
            text = self._backend.metadata_value(self._model, key)
            if text is None:
                continue
            try:
                number = float(text)
            except ValueError:
                raise ValueError(f"the model's GGUF states {key} as {text!r}, which is "
                                 "no number") from None
            if name == "penalty_window_tokens" and number == _WHOLE_CONTEXT_WINDOW:
                number = float(self.context_size)
            stated[name] = (int(number) if name in _WHOLE_NUMBER_FIELDS and number.is_integer()
                            else number)
        return stated

    def chat_template_text(self, template_name: str | None = None) -> str | None:
        """The source of the model's own chat template of that name, or ``None`` where it ships none."""
        model_template = self._model_templates.get(template_name)
        return None if model_template is None else model_template.text

    def reset(self) -> None:
        self._backend.kv_clear(self._engine_context)

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None:
        self._backend.decode(self._engine_context, token_ids, start_pos, SEQ, want_logits, self._max_batch_tokens)

    def _warm_up(self) -> None:
        """Decode two tokens and then a single one, then clear the cache again.

        The engine prepares itself on its first decodes, so this keeps that
        cost out of the first generated token; the first larger batch still
        pays for its own size. See design_history.md, "The warm-up decodes two
        tokens and then one".
        """
        tok = self._backend.token_eos(self._vocab)
        if tok < 0:  # no EOS; any valid id warms the graph
            tok = 0
        for tokens in ([tok] * 2, [tok]):
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
    def sampled_tokens(self, sampling: Sampling, withheld_tokens: tuple[int, ...]):
        """Tokens drawn one at a time from a sampler chain of their own, freed when this is closed."""
        # A chain of its own: penalties count this generation's tokens only, and a seed starts over.
        sampler = self._build_sampler(sampling, withheld_tokens)
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
        if not self._can_cut_tail:
            self._probe_saved_state()

    def _probe_saved_state(self) -> None:
        """Save the state one decoded token leaves, decode two more, and go back to it, cleared either way.

        Raises:
            RuntimeError: If this build cannot save a state or go back to one,
                which a cache that cannot be cut goes back with.
        """
        token = max(self._backend.token_eos(self._vocab), 0)
        try:
            self._backend.decode(self._engine_context, [token], 0, SEQ, False,
                                 self._max_batch_tokens)
            state = self.save_state()
            self._backend.decode(self._engine_context, [token] * 2, 1, SEQ, False,
                                 self._max_batch_tokens)
            self.restore_state(state, 1)
        except RuntimeError as error:
            raise RuntimeError(f"this build cannot save and restore the state of a cache "
                               f"that cannot be cut, which it is brought back with: {error}"
                               ) from error
        finally:
            self._backend.kv_clear(self._engine_context)

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

    def _build_sampler(self, sampling: Sampling, withheld: tuple[int, ...] = ()):
        """Assemble the sampler chain ``sampling`` asks for, in sampling order, ``withheld`` never drawn."""
        backend = self._backend
        chain = backend.sampler_chain()
        try:
            if withheld:
                backend.add_withheld(chain, self._vocab, withheld)
            if sampling.repetition_penalty != 1.0 \
                    or sampling.presence_penalty or sampling.frequency_penalty:
                backend.add_penalties(chain, self._vocab, sampling.penalty_window_tokens,
                                      sampling.repetition_penalty,
                                      sampling.frequency_penalty,
                                      sampling.presence_penalty)
            if sampling.temperature <= 0.0:
                backend.add_greedy(chain)
                return chain
            if sampling.top_k > 0:
                backend.add_top_k(chain, sampling.top_k)
            backend.add_top_p(chain, sampling.top_p)
            if sampling.min_p > 0.0:
                backend.add_min_p(chain, sampling.min_p)
            backend.add_temperature(chain, sampling.temperature)
            # An unset seed asks for a fresh random one on every run.
            seed = sampling.seed if sampling.seed is not None else backend.fresh_seed_per_run
            backend.add_distribution(chain, seed)
            return chain
        except BaseException:
            backend.free_sampler(chain)
            raise


def _flash_attention(config: Config) -> bool | None:
    """Flash attention as ``config`` asks for it, on for a quantized cache; ``None`` leaves it to the engine."""
    if config.flash_attention is None and config.kv_cache_type != UNQUANTIZED_KV_CACHE_TYPE:
        return True
    return config.flash_attention
