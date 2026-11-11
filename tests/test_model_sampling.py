"""Where a model's sampling comes from: the caller, its GGUF, its add-on file, a default."""

import json
import re

import pytest

from llama_chat import addon_files
from llama_chat.addon_files import (
    DEFAULT_SAMPLING, SHIPPED_ADDONS, UNSTATED_SAMPLING, addon_file, read_addon_file,
    template_fingerprint, with_model_sampling)
from llama_chat.config import Config
from tests.chat_templates import STEMS, fixture

_TEMPLATE = "{{ messages }}"


def test_what_the_caller_sets_wins_over_everything_the_model_recommends(tmp_path):
    model = tmp_path / "m.gguf"
    (tmp_path / "m.json").write_text(json.dumps({"sampling": {"temperature": 0.3}}))
    config = with_model_sampling(
        Config(model_path=str(model), temperature=0.9),
        {"temperature": 0.6}, addon_file(model, None))
    assert config.temperature == 0.9


def _addon(sampling: dict) -> dict:
    return {"templates": [], "sampling": sampling, "call": {}}


def test_each_value_comes_from_the_add_on_file_else_the_gguf_else_is_switched_off():
    config = with_model_sampling(Config(), {"temperature": 0.6, "top_p": 0.9, "min_p": 0.02},
                                 _addon({"temperature": 0.3, "top_k": 7}))
    assert (config.temperature, config.top_k, config.top_p, config.min_p,
            config.repetition_penalty) == (0.3, 7, 0.9, 0.02, UNSTATED_SAMPLING["repetition_penalty"])


def test_the_gguf_alone_or_the_add_on_file_alone_is_taken_as_it_states():
    stated = {"temperature": 0.6, "top_k": -1}
    from_gguf = with_model_sampling(Config(), stated, _addon({}))
    from_file = with_model_sampling(Config(), {}, _addon(stated))
    assert from_gguf == from_file
    assert (from_file.temperature, from_file.top_k, from_file.top_p) == (
        0.6, -1, UNSTATED_SAMPLING["top_p"])


def test_a_model_that_recommends_nothing_runs_at_the_engines_defaults():
    config = with_model_sampling(Config(), {}, _addon({}))
    assert {name: getattr(config, name) for name in DEFAULT_SAMPLING} == DEFAULT_SAMPLING


def test_what_a_recommendation_leaves_out_stays_switched_off(tmp_path):
    config = with_model_sampling(
        Config(model_path=str(tmp_path / "m.gguf")),
        {"temperature": 1.0, "top_k": 64, "top_p": 0.95}, addon_file(tmp_path / "m.gguf", None))
    assert (config.min_p, config.repetition_penalty) == (UNSTATED_SAMPLING["min_p"], 1.0)


def test_a_shipped_file_applies_to_the_template_it_names(tmp_path, monkeypatch):
    monkeypatch.setattr(addon_files, "SHIPPED_ADDONS", tmp_path)
    (tmp_path / "family.json").write_text(json.dumps(
        {"templates": [template_fingerprint(_TEMPLATE)], "sampling": {"top_p": 0.8}}))
    assert addon_file(None, _TEMPLATE)["sampling"] == {"top_p": 0.8}
    assert addon_file(None, "another template")["sampling"] == {}


@pytest.mark.parametrize("contents", ["not json", json.dumps({"sampler": {}}),
                                      json.dumps({"sampling": {"top_p": 2.0}}),
                                      json.dumps({"sampling": {"seed": 1}})])
def test_a_malformed_add_on_file_is_refused_when_the_model_loads(tmp_path, contents):
    (tmp_path / "m.json").write_text(contents)
    with pytest.raises(ValueError):
        with_model_sampling(
        Config(model_path=str(tmp_path / "m.gguf")),
        {}, addon_file(tmp_path / "m.gguf", None))


@pytest.mark.parametrize("path", sorted(SHIPPED_ADDONS.glob("*.json")), ids=lambda p: p.stem)
def test_every_shipped_file_names_its_templates_and_says_where_its_values_come_from(path):
    found = json.loads(path.read_text())
    assert found["templates"] and all(re.fullmatch("[0-9a-f]{64}", t) for t in found["templates"])
    assert found["source"] and (found.get("sampling") or found.get("call"))
    read_addon_file(path)                       # passes the check a model's load runs


@pytest.mark.parametrize("call", [{"opening": "x"},
                                  {"opening": "x", "name_close": "y", "closing": 3}])
def test_an_add_on_file_with_an_incomplete_call_is_refused(tmp_path, call):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"call": call}))
    with pytest.raises(ValueError, match="call"):
        read_addon_file(path)


def test_every_shipped_file_names_a_template_the_tests_run():
    fixtures = {template_fingerprint(fixture(stem)["chat_template"]) for stem in STEMS}
    for path in SHIPPED_ADDONS.glob("*.json"):
        assert set(json.loads(path.read_text())["templates"]) & fixtures, path.name


def test_the_ministral_3_file_holds_the_temperature_its_card_recommends_for_production():
    path = SHIPPED_ADDONS / "ministral-3.json"
    assert read_addon_file(path)["sampling"] == {"temperature": 0.05}
    assert ("Use a temperature below 0.1 for daily-driver and production environments"
            in json.loads(path.read_text())["source"])
