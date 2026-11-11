"""Tests of workflows/run_config.py: the configuration dict and the YAML overrides."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from streameucliddes.analysis.cmd_match_filter import DEFAULT_MATCH_FILTER
from streameucliddes.analysis.known_stream_search import prepare_search
from streameucliddes.catalogs.galstreams_catalog import select_galstreams_in_footprint
from streameucliddes.workflows.run_config import (
    KNOWN_STREAM_SEARCH_CONFIG,
    KNOWN_STREAM_SEARCH_OTHER_ARGUMENTS,
    RunConfig,
)
from streameucliddes.workflows.run_known_stream_search import (
    run_all_known_stream_searches,
    run_known_stream_search,
)

pytestmark = pytest.mark.run_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _defaults():
    """Default values of the arguments of the run and of the functions it calls."""
    defaults = {}
    functions = (prepare_search, run_known_stream_search, run_all_known_stream_searches,
                 select_galstreams_in_footprint)
    for function in functions:
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                defaults.setdefault(name, parameter.default)
    return defaults


# Every key is an argument of the run, with the default value of the functions
# (None of the match filter: the default match filter)
def test_config_is_the_defaults():
    defaults = _defaults()
    assert set(KNOWN_STREAM_SEARCH_CONFIG) <= set(defaults)
    for key, value in KNOWN_STREAM_SEARCH_CONFIG.items():
        expected = DEFAULT_MATCH_FILTER if key == "match_filter" and defaults[key] is None else defaults[key]
        assert value == expected, key
    assert KNOWN_STREAM_SEARCH_CONFIG["match_filter"] is not DEFAULT_MATCH_FILTER  # a copy
    assert set(KNOWN_STREAM_SEARCH_OTHER_ARGUMENTS) <= set(defaults)


# Values given in the file replace the defaults; the others stay; distance_moduli from
# {start, stop, step}, stop included
def test_from_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "search: known_stream\n"
        "n_workers: 8\n"
        "distance_moduli: {start: 16.0, stop: 17.0, step: 0.5}\n"
        "track_names: [AAU-ATLAS-L21]\n"
        "add_ibata2024: true\n"
    )
    config = RunConfig.from_yaml(path)
    assert config.search == "known_stream"
    np.testing.assert_allclose(config.kwargs.pop("distance_moduli"), [16.0, 16.5, 17.0])
    expected = {**KNOWN_STREAM_SEARCH_CONFIG, "n_workers": 8, "track_names": ["AAU-ATLAS-L21"], "add_ibata2024": True}
    expected.pop("distance_moduli")
    assert config.kwargs == expected


# An empty file, or no file: the defaults, as copies
def test_from_yaml_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    for config in (RunConfig.from_yaml(path), RunConfig()):
        assert config.search == "known_stream" and config.kwargs == KNOWN_STREAM_SEARCH_CONFIG
        config.kwargs["match_filter"]["name"] = "changed"
    assert KNOWN_STREAM_SEARCH_CONFIG["match_filter"]["name"] == DEFAULT_MATCH_FILTER["name"]


# The configuration of the repository reads
def test_repository_config():
    config = RunConfig.from_yaml(CONFIGS / "known_stream_search.yaml")
    assert config.search == "known_stream"
    assert set(config.kwargs) >= set(KNOWN_STREAM_SEARCH_CONFIG)


def test_from_yaml_errors(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("search: generic\n")
    with pytest.raises(ValueError, match="only 'known_stream'"):
        RunConfig.from_yaml(path)
    path.write_text("search: known_stream\nsmoothing: 0.3\n")
    with pytest.raises(ValueError, match="smoothing"):
        RunConfig.from_yaml(path)
