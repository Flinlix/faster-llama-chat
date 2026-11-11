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


def _metadata(stated=None, template=_TEMPLATE):
    values = {"tokenizer.chat_template": template,
              **{f"general.sampling.{key}": value for key, value in (stated or {}).items()}}
    return values.get


def test_what_the_caller_sets_wins_over_everything_the_model_recommends(tmp_path):
    model = tmp_path / "m.gguf"
    (tmp_path / "m.json").write_text(json.dumps({"sampling": {"temperature": 0.3}}))
    config = with_model_sampling(
        Config(model_path=str(model), temperature=0.9),
        _metadata({"temp": "0.6"}), addon_file(model, None))
    assert config.temperature == 0.9


def test_the_gguf_comes_before_the_add_on_file_and_the_file_before_the_default(tmp_path):
    model = tmp_path / "m.gguf"
    (tmp_path / "m.json").write_text(json.dumps({"sampling": {"temperature": 0.3, "top_k": 7}}))
    config = with_model_sampling(
        Config(model_path=str(model)),
        _metadata({"temp": "0.600000", "top_k": "-1"}), addon_file(model, None))
    assert (config.temperature, config.top_k) == (0.6, -1)
    config = with_model_sampling(
        Config(model_path=str(model)),
        _metadata(), addon_file(model, None))
    assert (config.temperature, config.top_k) == (0.3, 7)
    config = with_model_sampling(
        Config(model_path=str(tmp_path / "other.gguf")),
        _metadata(), addon_file(tmp_path / "other.gguf", None))
    assert config.min_p == DEFAULT_SAMPLING["min_p"]


def test_what_a_recommendation_leaves_out_stays_switched_off(tmp_path):
    config = with_model_sampling(
        Config(model_path=str(tmp_path / "m.gguf")),
        _metadata({"temp": "1.0", "top_k": "64", "top_p": "0.95"}), addon_file(tmp_path / "m.gguf", None))
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
        _metadata(), addon_file(tmp_path / "m.gguf", None))


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
