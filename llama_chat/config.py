"""Configuration for the chat wrapper.

The ``Config`` object is the only thing that differs between deployment targets:
model sampling defaults, context management, generation parameters, hardware
configuration, and security policy.
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

# Accepted llama.cpp/ggml log verbosities, least to most restrictive.
LOG_LEVELS = ("debug", "info", "warn", "error", "none")


@dataclass
class Config:
    """Runtime configuration for :class:`~llama_chat.wrapper.ChatWrapper`.

    You can supply a chat template explicitly via ``ChatWrapper(fragments=...)``.

    Attributes:
        model_path: Path to the GGUF model file.
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding.
        top_k: Top-k sampling cutoff (``<= 0`` disables).
        top_p: Nucleus sampling cutoff.
        min_p: Min-p sampling cutoff: drop tokens whose probability is below
            ``min_p`` times the top token's probability. ``0`` disables.
        repetition_penalty: Repetition penalty applied over ``penalty_window``
            tokens (``1.0`` disables).
        presence_penalty: Flat penalty on any token already present in the last
            ``penalty_window`` tokens (``0`` disables).
        frequency_penalty: Penalty scaled by how often a token occurs in the
            last ``penalty_window`` tokens (``0`` disables).
        penalty_window: Window for the repetition/presence/frequency penalties.
        seed: Sampling RNG seed, for reproducible generation; must be in
            ``[0, 2**32 - 2]``. ``None`` draws a fresh random seed per run.
        context_size: General context size in tokens (the hard wall - generation
            must never cross it).
        eviction_threshold: Fraction of ``context_size`` above which prefilled
            content triggers eviction of the oldest messages. Must be in
            ``(0, 1]``.
        max_tokens: Upper bound on generated tokens per ``request`` (further
            capped so total never exceeds ``context_size``).
        oversize_policy: What to do when a single message cannot fit under the
            threshold even after evicting everything but the system prompt:
            ``"reject"`` raises, ``"truncate"`` clips the message's content to
            fit (the turn-terminator tag is kept, so the clipped turn still
            closes cleanly).
        min_reply_tokens: Refuse a ``request``/``stream`` (raising
            :class:`~llama_chat.wrapper.ContextOverflowError`) if fewer than this
            many tokens of ``context_size`` would remain for the reply after
            prefilling the prompt. ``0`` disables the guard.
        stop_strings: Additional stop strings that end generation.
        gpu_layers: Layers to offload to GPU. ``0`` for a CPU-only build;
            ``-1`` to offload everything on the GPU.
        offload_kqv_to_gpu: Whether to offload the KV cache (and the KQV ops) to
            the GPU. ``True`` (llama.cpp's own default) puts the cache in VRAM;
            ``False`` keeps it in host RAM, freeing a large amount of VRAM on long
            contexts at the cost of generation speed. Ignored on a CPU-only build.
        threads: CPU threads for decode (``None`` lets llama.cpp decide).
        batch_size: Maximum tokens submitted to a single ``llama_decode`` call;
            longer prefills are chunked to this size.
        flash_attention: Enable flash attention. Required for a quantized KV
            cache.
        kv_cache_type: KV-cache quantization, one of ``KV_CACHE_GGML_TYPES``
            (e.g. ``"q8_0"`` to roughly halve cache memory at near-zero quality
            cost) or ``None`` for llama.cpp's f16 default. Quantized types
            require ``flash_attention=True``.
        log_level: Verbosity of llama.cpp/ggml's own log output, one of
            ``LOG_LEVELS`` (``"debug"``, ``"info"``, ``"warn"``, ``"error"``,
            ``"none"``). Messages below the chosen level are suppressed;
            ``"none"`` silences llama.cpp entirely, including its load-time
            banners. Applied process-wide before the model loads. Note that
            llama.cpp tags its routine model-loader banners at
            WARN, so ``"warn"`` still leaks the whole load-time wall.
    """

    # Model

    model_path: str = "faster-llama-chat/models/gemma-4-E2B-it-Q4_K_M.gguf"
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    min_p: float = 0.0  # 0.0 means disabled
    repetition_penalty: float = 1.0  # 1.0 means no penalty
    presence_penalty: float = 0.0  # 0.0 means no penalty
    frequency_penalty: float = 0.0  # 0.0 means no penalty
    penalty_window: int = 64
    seed: int | None = None  # None means a fresh random seed per run

    # Context management

    context_size: int = 4096
    eviction_threshold: float = 0.75
    max_tokens: int = 1024
    oversize_policy: str = "reject"
    min_reply_tokens: int = 32

    # Generation

    stop_strings: list[str] = field(default_factory=list)

    # Backend / hardware

    gpu_layers: int = 0
    offload_kqv_to_gpu: bool = False
    threads: int | None = None
    batch_size: int = 512
    flash_attention: bool = False
    kv_cache_type: str | None = None

    # Logging

    log_level: str = "error"

    def __post_init__(self) -> None:
        if not 0.0 < self.eviction_threshold <= 1.0:
            raise ValueError("eviction_threshold must be in (0, 1]")
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.min_reply_tokens < 0:
            raise ValueError("min_reply_tokens must be >= 0")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0 (1.0 disables)")
        if self.seed is not None and not 0 <= self.seed <= 0xFFFFFFFE:
            raise ValueError(
                "seed must be None or an int in [0, 2**32 - 2] "
                "(use None for a random seed)"
            )
        if self.oversize_policy not in ("reject", "truncate"):
            raise ValueError("oversize_policy must be 'reject' or 'truncate'")
        if self.log_level not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {list(LOG_LEVELS)}")
        if self.kv_cache_type is not None:
            if self.kv_cache_type not in KV_CACHE_GGML_TYPES:
                raise ValueError(
                    f"kv_cache_type must be None or one of {sorted(KV_CACHE_GGML_TYPES)}"
                )
            if not self.flash_attention:
                raise ValueError("a quantized kv_cache_type requires flash_attention=True")

    @property
    def threshold_tokens(self) -> int:
        """Eviction trigger expressed as an absolute token count."""
        return int(self.eviction_threshold * self.context_size)
