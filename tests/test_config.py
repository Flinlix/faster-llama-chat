"""Settings refused where they are built."""

import pytest

from llama_chat.config import Config, Sampling


# ----- cache type, flash attention and offload -----------------------------
def test_the_unquantized_kv_cache_type_needs_no_flash_attention():
    # f16 is llama.cpp's own default and the non-flash path holds it, so
    # naming it explicitly must not be refused.
    from llama_chat.config import UNQUANTIZED_KV_CACHE_TYPE
    config = Config(kv_cache_type=UNQUANTIZED_KV_CACHE_TYPE, flash_attention=False)
    assert config.kv_cache_type == UNQUANTIZED_KV_CACHE_TYPE


@pytest.mark.parametrize("quantized", ["q8_0", "q5_1", "q5_0", "q4_1", "q4_0"])
def test_every_quantized_kv_cache_type_requires_flash_attention(quantized):
    with pytest.raises(ValueError, match="flash_attention"):
        Config(kv_cache_type=quantized, flash_attention=False)


def test_kv_cache_type_requires_flash_attn():
    with pytest.raises(ValueError, match="flash_attention"):
        Config(kv_cache_type="q8_0", flash_attention=False)


def test_kv_cache_type_rejects_unknown_name():
    with pytest.raises(ValueError, match="kv_cache_type"):
        Config(kv_cache_type="bogus", flash_attention=True)


def test_kv_cache_type_accepts_known_type_with_flash_attn():
    cfg = Config(kv_cache_type="q8_0", flash_attention=True)
    assert cfg.kv_cache_type == "q8_0"


def test_offload_kqv_to_gpu_defaults_off():
    # KV cache stays in host RAM unless explicitly offloaded to the GPU.
    assert Config().kv_cache_on_gpu is False


@pytest.mark.parametrize("field,value", [
    ("max_batch_tokens", 0),
    ("max_reply_tokens", 0),
    ("min_reply_tokens", -1),
    ("context_size", 0),
    ("evict_above_fraction", 0.0),
    ("max_history_messages", 0),
    ("max_history_messages", -1),
    ("repetition_penalty", 0.0),
    ("seed", -1),
    ("seed", 0xFFFFFFFF),
])
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        Config(**{field: value})


# ----- configurations that could never work --------------------------------
def test_min_reply_tokens_at_the_context_size_is_refused():
    """Every generate would raise, and the first one is too late to find out."""
    with pytest.raises(ValueError, match="min_reply_tokens"):
        Config(context_size=64, min_reply_tokens=64)
    Config(context_size=64, min_reply_tokens=63)        # one token to reply in


def test_an_eviction_threshold_that_rounds_to_nothing_is_refused():
    with pytest.raises(ValueError, match="rounds down to no tokens"):
        Config(context_size=8, evict_above_fraction=0.1)


# ----- a sampling built for one request ------------------------------------
def test_an_empty_sampling_overrides_nothing():
    assert Sampling().overrides == {}
    config = Config(temperature=0.7)
    assert Sampling().applied_to(config) == config


def test_a_sampling_replaces_exactly_the_fields_it_sets():
    applied = Sampling(temperature=0.2, seed=7).applied_to(
        Config(temperature=0.7, top_p=0.9))
    assert (applied.temperature, applied.seed, applied.top_p) == (0.2, 7, 0.9)


@pytest.mark.parametrize("bad", [{"top_p": 0.0}, {"top_p": 1.5}])
def test_an_impossible_top_p_is_refused_where_it_is_built(bad):
    with pytest.raises(ValueError):
        Sampling(**bad)


@pytest.mark.parametrize("settings, field", [
    (dict(top_p=2.0), "top_p"), (dict(seed=-1), "seed"),
    (dict(evict_to_fraction_without_shift=0.0), "evict_to_fraction_without_shift"),
    (dict(min_reply_tokens=5000), "min_reply_tokens")])
def test_a_refused_setting_names_its_field_as_data(settings, field):
    from llama_chat import InvalidSettingError
    with pytest.raises(InvalidSettingError) as refused:
        Config(**settings)
    assert refused.value.field == field and isinstance(refused.value, ValueError)
