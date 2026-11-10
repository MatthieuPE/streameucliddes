#!/usr/bin/env python3
"""
Git clean filter: a Jupyter notebook without its outputs.

Reads a notebook on stdin and writes it on stdout with the outputs and execution counts of
its code cells removed, so that git stores the notebooks without their (large, generated)
outputs while the local copies keep them. Uses only the standard library.

With ``--check NOTEBOOK ...``, writes nothing and exits with status 1 if a notebook is not
stored as the filter writes it, e.g. committed from a clone without the filter (used by CI
on the committed notebooks; local copies keep their outputs, so they fail it).

Set up once per clone (the filter is defined in .gitconfig):
    git config include.path ../.gitconfig
"""
import argparse
import json
import sys
from pathlib import Path


def strip_outputs(notebook):
    """The notebook, with no outputs nor execution counts in its code cells."""
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return notebook


def stripped(raw):
    """Notebook as the filter stores it.

    Parameters
    ----------
    raw : bytes
        Notebook JSON.

    Returns
    -------
    bytes
        Notebook JSON without outputs: UTF-8, indent 1, sorted keys, final newline.
    """
    # Bytes in and out: independent of the locale git runs the filter with
    notebook = strip_outputs(json.loads(raw))
    text = json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def unstripped(paths):
    """Notebooks not stored as the filter writes them.

    Parameters
    ----------
    paths : iterable of str or Path
        Notebook files.

    Returns
    -------
    list of str or Path
        Those whose content differs from its stripped version.
    """
    contents = ((path, Path(path).read_bytes()) for path in paths)
    return [path for path, raw in contents if raw != stripped(raw)]


def main(argv=None):
    """Run the filter (stdin to stdout), or the check of ``--check``.

    Returns
    -------
    int
        Exit status: 1 if ``--check`` finds a notebook not stored as the filter writes it.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--check",
        nargs="*",
        metavar="NOTEBOOK",
        help="check these notebooks instead of filtering stdin",
    )
    args = parser.parse_args(argv)

    if args.check is None:
        sys.stdout.buffer.write(stripped(sys.stdin.buffer.read()))  # filter mode
        return 0

    bad = unstripped(args.check)
    for path in bad:
        print(f"{path}: stored with outputs, or not through the filter", file=sys.stderr)
    if bad:
        print(
            "Set up the filter (git config include.path ../.gitconfig), then store them"
            " again: git add --renormalize <notebook>",
            file=sys.stderr,
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
