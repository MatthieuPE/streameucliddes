"""Fixtures for the tests of tools/strip_notebook_outputs.py."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "tools" / "strip_notebook_outputs.py"


@pytest.fixture(scope="session")
def strip_module():
    """The filter script, imported as a module."""
    spec = importlib.util.spec_from_file_location("strip_notebook_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def run_script():
    """Factory: run the filter script as git does, returning the completed process."""

    def run(*args, stdin=b""):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], input=stdin, capture_output=True
        )

    return run


@pytest.fixture
def notebook():
    """Notebook dict: a markdown cell (non-ASCII), then a code cell that was run."""
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Flux à 5σ"]},
            {
                "cell_type": "code",
                "execution_count": 3,
                "metadata": {},
                "outputs": [
                    {"name": "stdout", "output_type": "stream", "text": ["1\n"]}
                ],
                "source": ["print(1)"],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


@pytest.fixture
def notebook_raw(notebook):
    """The notebook as Jupyter saves it: JSON bytes, outputs included."""
    return json.dumps(notebook, indent=1, ensure_ascii=False).encode("utf-8")
