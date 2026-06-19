"""Configuration for the chat wrapper.

The ``Config`` object is the only thing that differs between deployment targets:
the model (and its chat-template fragments), the context size, and the eviction
threshold. The wrapper logic itself is hardware agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Friendly KV-cache quantization names -> ggml type ids (see ggml.h `enum ggml_type`).
KV_CACHE_GGML_TYPES = {
    "f16": 1,
    "q8_0": 8,
    "q5_1": 7,
    "q5_0": 6,
    "q4_1": 3,
    "q4_0": 2,
}


@dataclass
class Config:
    """Runtime configuration for :class:`~llama_chat.wrapper.ChatWrapper`.

    You can supply a chat template explicitly via ``ChatWrapper(fragments=...)``.

    Attributes:
        model_path: Path to the GGUF model file.
        n_ctx: General context size in tokens (the hard wall - generation must
            never cross it).
        threshold_pct: Fraction of ``n_ctx`` above which prefilled content
            triggers eviction of the oldest messages. Must be in ``(0, 1]``.
        n_gpu_layers: Layers to offload to GPU. ``0`` for a CPU-only build;
            ``-1`` to offload everything on the GPU.
        n_threads: CPU threads for decode (``None`` lets llama.cpp decide).
        n_batch: Maximum tokens submitted to a single ``llama_decode`` call;
            longer prefills are chunked to this size.
        flash_attn: Enable flash attention. Required for a quantized KV cache.
        kv_cache_type: KV-cache quantization, one of ``KV_CACHE_GGML_TYPES``
            (e.g. ``"q8_0"`` to roughly halve cache memory at near-zero quality
            cost) or ``None`` for llama.cpp's f16 default. Quantized types
            require ``flash_attn=True``.
        max_tokens: Upper bound on generated tokens per ``request`` (further
            capped so total never exceeds ``n_ctx``).
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding.
        top_k: Top-k sampling cutoff (``<= 0`` disables).
        top_p: Nucleus sampling cutoff.
        repeat_penalty: Repetition penalty applied over ``repeat_last_n`` tokens.
        repeat_last_n: Window for the repetition penalty.
        stop: Additional stop strings that end generation.
        oversize_policy: What to do when a single message cannot fit under the
            threshold even after evicting everything but the system prompt:
            ``"reject"`` raises, ``"truncate"`` clips the message's content to
            fit (the turn-terminator tag is kept, so the clipped turn still
            closes cleanly).
        min_answer_tokens: Refuse a ``request``/``stream`` (raising
            :class:`~llama_chat.wrapper.ContextOverflowError`) if fewer than this
            many tokens of ``n_ctx`` would remain for the reply after prefilling
            the prompt. ``0`` disables the guard.
        unsafe_content_policy: What to do on a llama.cpp build lacking
            ``llama_vocab_is_control``, where special tokens cannot be classified
            and message content is therefore left unsanitized (untrusted input
            can forge turn boundaries): ``"error"`` raises at construction,
            ``"warn"`` emits a ``RuntimeWarning``, ``"ignore"`` proceeds
            silently. Has no effect on builds that can classify tokens.
    """

    # Gemma 4

    model_path: str = "llama/models/gemma-4-E2B-it-Q4_K_M.gguf"
    n_ctx: int = 4096
    threshold_pct: float = 0.75

    # Backend / hardware
    n_gpu_layers: int = 0
    n_threads: int | None = None
    n_batch: int = 512
    flash_attn: bool = False
    kv_cache_type: str | None = None

    # Generation defaults (overridable per request)
    max_tokens: int = 1024
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    repeat_penalty: float = 1.0  # 1.0 means no penalty
    repeat_last_n: int = 64
    stop: list[str] = field(default_factory=list)

    # Policy
    oversize_policy: str = "reject"
    min_answer_tokens: int = 32
    unsafe_content_policy: str = "error"

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_pct <= 1.0:
            raise ValueError("threshold_pct must be in (0, 1]")
        if self.n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if self.n_batch <= 0:
            raise ValueError("n_batch must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_answer_tokens < 0:
            raise ValueError("min_answer_tokens must be >= 0")
        if self.oversize_policy not in ("reject", "truncate"):
            raise ValueError("oversize_policy must be 'reject' or 'truncate'")
        if self.unsafe_content_policy not in ("error", "warn", "ignore"):
            raise ValueError("unsafe_content_policy must be 'error', 'warn' or 'ignore'")
        if self.kv_cache_type is not None:
            if self.kv_cache_type not in KV_CACHE_GGML_TYPES:
                raise ValueError(
                    f"kv_cache_type must be None or one of {sorted(KV_CACHE_GGML_TYPES)}"
                )
            if not self.flash_attn:
                raise ValueError("a quantized kv_cache_type requires flash_attn=True")

    @property
    def threshold_tokens(self) -> int:
        """Eviction trigger expressed as an absolute token count."""
        return int(self.threshold_pct * self.n_ctx)
