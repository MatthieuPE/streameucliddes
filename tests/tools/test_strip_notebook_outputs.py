"""Tests of the git filter stripping the notebook outputs."""

import json

import pytest

pytestmark = pytest.mark.tools


# ===================================================
# Filter mode: stdin to stdout, as git runs it
# ===================================================


# Code cells lose their outputs and execution counts; the rest is kept as is
def test_filter_strips_outputs(run_script, notebook, notebook_raw):
    result = run_script(stdin=notebook_raw)
    assert result.returncode == 0

    cells = json.loads(result.stdout)["cells"]
    assert cells[0] == notebook["cells"][0]                 # markdown untouched
    assert cells[1]["outputs"] == []                        # no outputs
    assert cells[1]["execution_count"] is None              # no count
    assert cells[1]["source"] == notebook["cells"][1]["source"]
    assert "à 5σ".encode() in result.stdout                 # UTF-8, not escaped


# Filtering a stored notebook gives it back unchanged, or git would see it modified
def test_filter_is_idempotent(run_script, notebook_raw):
    once = run_script(stdin=notebook_raw).stdout
    assert run_script(stdin=once).stdout == once


# ===================================================
# Check mode: are notebooks stored as the filter writes them?
# ===================================================


def test_check_fails_on_outputs(run_script, tmp_path, notebook_raw):
    path = tmp_path / "run.ipynb"
    path.write_bytes(notebook_raw)

    result = run_script("--check", str(path))
    assert result.returncode == 1
    assert str(path).encode() in result.stderr              # names the notebook


def test_check_passes_on_stripped(run_script, strip_module, tmp_path, notebook_raw):
    path = tmp_path / "clean.ipynb"
    path.write_bytes(strip_module.stripped(notebook_raw))

    result = run_script("--check", str(path))
    assert result.returncode == 0
    assert result.stdout == result.stderr == b""


# No outputs, but not in the filter's format: git would see it modified
def test_check_fails_on_other_format(run_script, strip_module, tmp_path, notebook):
    path = tmp_path / "unsorted.ipynb"
    raw = strip_module.stripped(json.dumps(notebook).encode())
    path.write_bytes(json.dumps(json.loads(raw), indent=2).encode())

    assert run_script("--check", str(path)).returncode == 1


# CI passes no file when the repository has no notebook
def test_check_without_notebooks(run_script):
    assert run_script("--check").returncode == 0
