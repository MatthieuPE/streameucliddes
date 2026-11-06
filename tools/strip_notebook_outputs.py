#!/usr/bin/env python3
"""
Git clean filter: a Jupyter notebook without its outputs.

Reads a notebook on stdin and writes it on stdout with the outputs and execution counts of
its code cells removed, so that git stores the notebooks without their (large, generated)
outputs while the local copies keep them. Uses only the standard library.

Set up once per clone (see .gitattributes):
    git config filter.strip-notebook-outputs.clean "python3 tools/strip_notebook_outputs.py"
    git config filter.strip-notebook-outputs.required true
"""
import json
import sys


def strip_outputs(notebook):
    """The notebook, with no outputs nor execution counts in its code cells."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return notebook


if __name__ == "__main__":
    # Bytes in and out: independent of the locale git runs the filter with
    notebook = json.loads(sys.stdin.buffer.read())
    text = json.dumps(strip_outputs(notebook), indent=1, sort_keys=True, ensure_ascii=False)
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
