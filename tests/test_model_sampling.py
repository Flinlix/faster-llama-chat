"""Where a model's sampling comes from: the caller, its GGUF, its add-on file, a default."""

import json
import re

import pytest

from llama_chat import addon_files
from llama_chat.addon_files import (
    DEFAULT_SAMPLING, SHIPPED_ADDONS, UNSTATED_SAMPLING, addon_file, read_addon_file,
    template_fingerprint, with_model_sampling)
from llama_chat.config import Config, InvalidSettingError
from tests.chat_templates import STEMS, expected, fixture

_TEMPLATE = "{{ messages }}"


def _own_file(tmp_path, contents) -> str:
    """The path of a model whose own add-on file holds ``contents``, as JSON where it is not text."""
    (tmp_path / "m.json").write_text(contents if isinstance(contents, str) else json.dumps(contents))
    return str(tmp_path / "m.gguf")


def test_what_the_caller_sets_wins_over_everything_the_model_recommends(tmp_path):
    model = _own_file(tmp_path, {"templates": [template_fingerprint(_TEMPLATE)],
                                 "sampling": {"temperature": 0.3}})
    config = with_model_sampling(
        Config(model_path=model, temperature=0.9),
        {"temperature": 0.6}, addon_file(model, _TEMPLATE))
    assert config.temperature == 0.9


def _addon(sampling: dict) -> dict:
    return {"templates": [], "sampling": sampling, "call": {}, "reasoning_on": {}}


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


@pytest.mark.parametrize("rest", [{"sampler": {}}, {"sampling": {"top_p": 2.0}},
                                  {"sampling": {"seed": 1}}])
def test_a_malformed_add_on_file_that_names_the_template_is_refused_when_the_model_loads(
        tmp_path, rest):
    model = _own_file(tmp_path, {"templates": [template_fingerprint(_TEMPLATE)], **rest})
    with pytest.raises(ValueError):
        addon_file(model, _TEMPLATE)


@pytest.mark.parametrize("contents, why", [
    ({"templates": ["0" * 64], "sampling": {"top_p": 0.5}}, "names other templates"),
    ({"sampling": {"top_p": 0.5}}, "names no template"),
    ("not json", "is not JSON"),
    ('["a list"]', "is not a JSON object"),
    ({"templates": "abc", "sampling": {"top_p": 2.0}}, "names no template"),
])
def test_a_file_next_to_the_model_that_does_not_name_its_template_is_not_used(
        tmp_path, monkeypatch, caplog, contents, why):
    monkeypatch.setattr(addon_files, "SHIPPED_ADDONS", tmp_path / "shipped")
    (tmp_path / "shipped").mkdir()
    (tmp_path / "shipped" / "family.json").write_text(json.dumps(
        {"templates": [template_fingerprint(_TEMPLATE)], "sampling": {"top_p": 0.8}}))
    model = _own_file(tmp_path, contents)
    with caplog.at_level("WARNING", logger="llama_chat"):
        assert addon_file(model, _TEMPLATE)["sampling"] == {"top_p": 0.8}
    assert len(caplog.records) == 1 and why in caplog.records[0].getMessage()
    assert str(tmp_path / "m.json") in caplog.records[0].getMessage()


def test_a_file_next_to_a_model_without_a_template_is_not_used(tmp_path, caplog):
    model = _own_file(tmp_path, {"templates": ["0" * 64], "sampling": {"top_p": 0.5}})
    with caplog.at_level("WARNING", logger="llama_chat"):
        assert addon_file(model, None)["sampling"] == {}
    assert "ships no chat template" in caplog.records[0].getMessage()


def test_a_file_next_to_the_model_that_names_its_template_comes_before_a_shipped_one(
        tmp_path, monkeypatch):
    monkeypatch.setattr(addon_files, "SHIPPED_ADDONS", tmp_path / "shipped")
    (tmp_path / "shipped").mkdir()
    (tmp_path / "shipped" / "family.json").write_text(json.dumps(
        {"templates": [template_fingerprint(_TEMPLATE)], "sampling": {"top_p": 0.8}}))
    model = _own_file(tmp_path, {"templates": ["0" * 64, template_fingerprint(_TEMPLATE)],
                                 "sampling": {"top_p": 0.5}})
    assert addon_file(model, _TEMPLATE)["sampling"] == {"top_p": 0.5}


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


@pytest.mark.parametrize("stem", [stem for stem in STEMS if expected(fixture(stem), "card_sampling")])
def test_a_shipped_file_holds_the_sampling_its_models_card_recommends_and_quotes_the_card(stem):
    known = fixture(stem)
    fingerprint = template_fingerprint(known["chat_template"])
    path, = [path for path in SHIPPED_ADDONS.glob("*.json")
             if fingerprint in json.loads(path.read_text())["templates"]]
    assert read_addon_file(path)["sampling"] == expected(known, "card_sampling")
    assert expected(known, "card_quote") in json.loads(path.read_text())["source"]


@pytest.mark.parametrize("penalty", ["presence_penalty", "frequency_penalty"])
def test_an_add_on_file_may_set_a_presence_or_a_frequency_penalty(tmp_path, penalty):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"sampling": {penalty: 1.5}}))
    found = read_addon_file(path)
    assert getattr(with_model_sampling(Config(), {}, found), penalty) == 1.5
    assert getattr(with_model_sampling(Config(), {"temperature": 0.7}, _addon({})), penalty) == 0.0
    assert getattr(Config(), penalty) is None


@pytest.mark.parametrize("contents, named", [
    ({"sampling": {"temperature": "hot"}}, "temperature"),
    ({"sampling": {"top_k": 2.5}}, "top_k"),
    ({"sampling": {"min_p": True}}, "min_p"),
    ({"templates": "7f0e5290"}, "templates"),
    ({"templates": ["not a fingerprint"]}, "templates"),
    ({"source": ["a", "list"]}, "source"),
])
def test_an_add_on_file_value_of_the_wrong_type_is_refused_naming_the_file(
        tmp_path, contents, named):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError) as refused:
        read_addon_file(path)
    assert str(path) in str(refused.value) and named in str(refused.value)


@pytest.mark.parametrize("stated, named", [({"top_p": 2.0}, "top_p"), ({"top_k": 20.5}, "top_k"),
                                           ({"temperature": "warm"}, "temperature"),
                                           ({"seed": 3}, "seed")])
def test_a_value_the_model_recommends_that_the_settings_refuse_stops_the_load_naming_its_source(
        stated, named):
    with pytest.raises(ValueError) as refused:
        with_model_sampling(Config(), stated, _addon({}))
    assert not isinstance(refused.value, InvalidSettingError)     # no setting of the caller's
    assert "the model's own" in str(refused.value) and named in str(refused.value)


def test_a_refused_value_the_model_recommends_is_no_matter_where_another_source_wins():
    assert with_model_sampling(Config(top_p=0.9), {"top_p": 2.0}, _addon({})).top_p == 0.9
    assert with_model_sampling(Config(), {"top_p": 2.0}, _addon({"top_p": 0.8})).top_p == 0.8


def test_the_penalty_window_comes_from_the_config_else_the_add_on_file_else_the_gguf():
    assert with_model_sampling(Config(penalty_window_tokens=8), {"penalty_window_tokens": 128},
                               _addon({"penalty_window_tokens": 32})).penalty_window_tokens == 8
    assert with_model_sampling(Config(), {"penalty_window_tokens": 128},
                               _addon({"penalty_window_tokens": 32})).penalty_window_tokens == 32
    assert with_model_sampling(Config(), {"penalty_window_tokens": 128},
                               _addon({})).penalty_window_tokens == 128


def test_a_penalty_window_no_source_states_is_the_engines_default_length():
    assert DEFAULT_SAMPLING["penalty_window_tokens"] == 64
    assert with_model_sampling(Config(), {}, _addon({})).penalty_window_tokens == 64
    assert with_model_sampling(Config(), {"repetition_penalty": 1.1},
                               _addon({})).penalty_window_tokens == 64
    assert Config().penalty_window_tokens is None


@pytest.mark.parametrize("source", ["gguf", "add-on file"])
def test_a_model_that_states_only_its_penalty_window_samples_at_the_engines_defaults(source):
    stated = {"penalty_window_tokens": 128}
    config = (with_model_sampling(Config(), stated, _addon({})) if source == "gguf"
              else with_model_sampling(Config(), {}, _addon(stated)))
    assert {name: getattr(config, name) for name in DEFAULT_SAMPLING} == {
        **DEFAULT_SAMPLING, **stated}


def test_an_add_on_file_may_set_the_penalty_window(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"sampling": {"penalty_window_tokens": 256}}))
    assert with_model_sampling(Config(), {}, read_addon_file(path)).penalty_window_tokens == 256
    path.write_text(json.dumps({"sampling": {"penalty_window_tokens": -1}}))
    with pytest.raises(ValueError, match="penalty_window_tokens"):
        read_addon_file(path)


def test_no_two_shipped_files_name_one_template():
    named = [t for path in SHIPPED_ADDONS.glob("*.json")
             for t in json.loads(path.read_text())["templates"]]
    assert len(named) == len(set(named))



def test_an_add_on_file_may_name_the_template_variables_that_turn_reasoning_on(tmp_path):
    model = _own_file(tmp_path, {"templates": [template_fingerprint(_TEMPLATE)],
                                 "reasoning_on": {"reasoning_effort": "high"}})
    assert addon_file(model, _TEMPLATE)["reasoning_on"] == {"reasoning_effort": "high"}
    assert addon_file(None, _TEMPLATE)["reasoning_on"] == {}


@pytest.mark.parametrize("reasoning_on", [{}, ["reasoning_effort"], "high"])
def test_a_reasoning_on_that_names_no_variable_is_refused_naming_the_file(tmp_path, reasoning_on):
    model = _own_file(tmp_path, {"templates": [template_fingerprint(_TEMPLATE)],
                                 "reasoning_on": reasoning_on})
    with pytest.raises(ValueError, match=r"m\.json: reasoning_on"):
        addon_file(model, _TEMPLATE)
