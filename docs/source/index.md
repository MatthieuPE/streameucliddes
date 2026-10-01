# streameucliddes

`streameucliddes` is a package for detecting Milky Way stellar streams in
Euclid and DES photometric survey data.

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Building and viewing this site locally

From the repository root, with the `docs` dependency group installed:

```bash
python -m sphinx -b html docs/source docs/_build/html
```

Then either open `docs/_build/html/index.html` directly in a browser, or
serve it for full functionality including search:
`cd docs/_build/html && python -m http.server 8000`, then open
`http://localhost:8000`.

`docs/_build/` and `docs/source/generated/` are gitignored; rebuilding
after an edit just overwrites them.
