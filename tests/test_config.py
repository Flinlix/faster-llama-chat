"""Settings refused where they are built."""

from dataclasses import fields

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


def test_the_kv_cache_follows_the_model_by_default():
    # Each layer's cache where that layer runs, in VRAM for the layers on the GPU.
    assert Config().kv_cache_on_gpu is True


@pytest.mark.parametrize("field,value", [
    ("max_batch_tokens", 0),
    ("max_reply_tokens", 0),
    ("min_reply_tokens", -1),
    ("context_size", 0),
    ("evict_above_fraction", 0.0),
    ("max_history_exchanges", 0),
    ("max_history_exchanges", -1),
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


@pytest.mark.parametrize("model_path", [None, "", "no-such-model.gguf"])
def test_a_model_path_that_names_no_file_is_a_refused_setting(tmp_path, model_path):
    from llama_chat import ChatWrapper
    path = model_path and str(tmp_path / model_path)
    with pytest.raises(InvalidSettingError) as refused:
        ChatWrapper(Config(model_path=path))
    assert refused.value.field == "model_path"


@pytest.mark.parametrize("cap", [0, -1])
def test_a_request_cap_that_allows_no_token_is_refused_where_it_is_given(cap):
    from llama_chat import ChatWrapper
    from tests.fake_context import FakeContext
    from tests.invented_templates import PLAIN
    wrapper = ChatWrapper(Config(context_size=400), context=FakeContext(PLAIN))
    wrapper.begin("s")
    with pytest.raises(InvalidSettingError) as refused:
        wrapper.reply("hi", max_reply_tokens=cap)
    assert refused.value.field == "max_reply_tokens"
    assert str(refused.value) == str(pytest.raises(InvalidSettingError, Config,
                                                   max_reply_tokens=cap).value)


def test_how_far_an_eviction_drops_without_a_shift_is_fixed_and_no_setting():
    from llama_chat.rendered_cache import EVICT_TO_FRACTION_WITHOUT_SHIFT
    assert "evict_to_fraction_without_shift" not in {f.name for f in fields(Config)}
    assert 0.0 < EVICT_TO_FRACTION_WITHOUT_SHIFT <= 1.0


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
    (dict(evict_above_fraction=0.0), "evict_above_fraction"),
    (dict(min_reply_tokens=5000), "min_reply_tokens")])
def test_a_refused_setting_names_its_field_as_data(settings, field):
    from llama_chat import InvalidSettingError
    with pytest.raises(InvalidSettingError) as refused:
        Config(**settings)
    assert refused.value.field == field and isinstance(refused.value, ValueError)
