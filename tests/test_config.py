"""Settings refused where they are built."""

import pytest

from llama_chat.config import (KV_CACHE_TYPES, UNQUANTIZED_KV_CACHE_TYPE, Config,
                               InvalidSettingError, Sampling, with_request_sampling)


# ----- cache type, flash attention and offload -----------------------------
def test_the_unquantized_kv_cache_type_needs_no_flash_attention():
    # f16 is llama.cpp's own default and the non-flash path holds it, so
    # naming it explicitly must not be refused.
    config = Config(kv_cache_type=UNQUANTIZED_KV_CACHE_TYPE, flash_attention=False)
    assert config.kv_cache_type == UNQUANTIZED_KV_CACHE_TYPE


def test_the_kv_cache_type_defaults_to_the_unquantized_one():
    assert Config(flash_attention=False).kv_cache_type == UNQUANTIZED_KV_CACHE_TYPE


def test_the_kv_cache_type_has_no_unset_value():
    with pytest.raises(ValueError, match="kv_cache_type"):
        Config(kv_cache_type=None)


@pytest.mark.parametrize("quantized", [kind for kind in KV_CACHE_TYPES
                                       if kind != UNQUANTIZED_KV_CACHE_TYPE])
def test_every_quantized_kv_cache_type_requires_flash_attention(quantized):
    with pytest.raises(ValueError, match="flash_attention"):
        Config(kv_cache_type=quantized, flash_attention=False)


def test_kv_cache_type_rejects_unknown_name():
    with pytest.raises(ValueError, match="kv_cache_type"):
        Config(kv_cache_type="bogus", flash_attention=True)


@pytest.mark.parametrize("kind", KV_CACHE_TYPES)
def test_every_kv_cache_type_is_accepted_with_flash_attention(kind):
    assert Config(kv_cache_type=kind, flash_attention=True).kv_cache_type == kind


def test_kv_cache_on_gpu_defaults_off():
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
    config = Config(temperature=0.7)
    assert with_request_sampling(config, Sampling()) == config
    assert with_request_sampling(config, None) == config


def test_a_sampling_replaces_exactly_the_fields_it_sets():
    applied = with_request_sampling(Config(temperature=0.7, top_p=0.9),
                                    Sampling(temperature=0.2, seed=7))
    assert (applied.temperature, applied.seed, applied.top_p) == (0.2, 7, 0.9)


@pytest.mark.parametrize("bad", [{"top_p": 0.0}, {"top_p": 1.5}, {"seed": -1},
                                 {"seed": 2**32 - 1}])
def test_a_request_is_refused_exactly_as_the_configuration_refuses_it(bad):
    with pytest.raises(InvalidSettingError) as by_config:
        Config(**bad)
    with pytest.raises(InvalidSettingError) as by_request:
        Sampling(**bad)
    assert (by_request.value.field, str(by_request.value)) == \
        (by_config.value.field, str(by_config.value))


@pytest.mark.parametrize("settings, field", [
    (dict(top_p=2.0), "top_p"), (dict(seed=-1), "seed"),
    (dict(evict_to_fraction_without_shift=0.0), "evict_to_fraction_without_shift"),
    (dict(min_reply_tokens=5000), "min_reply_tokens")])
def test_a_refused_setting_names_its_field_as_data(settings, field):
    from llama_chat import InvalidSettingError
    with pytest.raises(InvalidSettingError) as refused:
        Config(**settings)
    assert refused.value.field == field and isinstance(refused.value, ValueError)
