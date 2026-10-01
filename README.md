# streameucliddes

Stellar stream detection in Euclid and DES photometric survey data.

Documentation: https://matthieupe.github.io/streameucliddes/

## Development setup

```bash
conda activate streameuclid
pip install -e . --group test --group docs
```

- Run the tests: `pytest`
- Build the docs: `python -m sphinx -b html docs/source docs/_build/html`

The docs are rebuilt on every pull request (downloadable as the `docs-html`
artifact) and deployed to GitHub Pages on every push to `main`.
