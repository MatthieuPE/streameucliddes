# Context for Claude Code: restructure + known-stream matched-filter maps

Read this file fully before any change. Companion file:
`package_structure.md` (target tree and old → new names). Keep both in
`docs/dev/` of the repo.

## 1. Science goal

Make matched-filter (MF) maps and videos for all known streams in the
Euclid DR1 × DES Y6 footprint. Also make generic MF maps over the full
footprint.

1. **Selection.** galstreams streams (Mateu 2023, MNRAS 520, 5225) with
   **dm > 14** along the track, no upper bound (user choice: wider than
   the dm 15–22 of the maps, so as not to miss a stream), crossing the
   footprint. The Ibata et al. 2024 tracks have no distance in galstreams:
   only selected with `add_ibata2024` (default False).
2. **Frame.** phi1/phi2 from the galstreams frame (`strm.stream_frame`),
   used as it is: a gala `GreatCircleICRSFrame`. Transformations with
   gala and astropy (user choice), not with our own rotation matrices.
3. **Known-stream MF.** Isochrone from the stream's age and [Fe/H] when
   known, else a default old, metal-poor grid. Maps in phi1/phi2, **one
   set of maps per isochrone**. **No polynomial background subtraction.**
4. **Generic MF.** Full footprint, generic isochrones, dm 15–22.
   **2D polynomial background of fixed degree** (5, the current default)
   for now. Choosing the degree from the map fluctuations comes later
   (Section 8).
5. **Products.** One map per (isochrone, dm); one video and one file of
   maps per (stream, isochrone).

Method: Rockosi et al. 2002 (AJ 124, 349); Grillmair 2009 (ApJ 693,
1118); Shipp et al. 2018 (ApJ 862, 114).

## 2. Work in three phases. Stop after each one for review.

### Phase 1: restructure only (branch `restructure-package`)

- Apply the tree of `package_structure.md`, **old code only**. Use the
  old → new table (Section 4 there). Do **not** create any `[NEW]` file
  or function.
- Use `git mv` so history follows the files. Separate commits: moves,
  renames, then the fixes below. Small commits, easy to review.
- Allowed logic changes, and only these:
  - delete the duplicate `get_healpy_data_from_table` (diff the two
    copies first, keep the more complete one, report the differences);
  - split `scan_find_stream`: computation → `analysis/generic_stream_search.py`,
    loop + plots + gif → `workflows/run_generic_search.py`;
  - split `plotting.get_hp_map`: computation → `analysis/star_density_maps.py`.
- **Missing functions** (`correct_dust_map`, `plot_overlapped_objects`,
  `plot_single_stream_track`): search the git history
  (`git log -S <name> --all`) and the local machine. Report what you
  find. Do not write them from scratch without asking.
- No compatibility shims. Update all imports in `src/`, `tests/`,
  `notebooks/`, `figures/`, `docs/`.
- `DEFAULT_DATA_PATH` uses `Path(__file__).parents[3]`: `catalogs/` is at
  the same depth as `analysis/`, so it still works. Check it.
- **Checks before asking for review:**
  - `pytest` passes. Adapt tests to new paths and names only. Do not
    change what a test checks. Rename test files to match modules
    (e.g. `test_fit.py` → `test_background_polynomial_fit.py`). Update
    the pytest markers in `pyproject.toml`.
  - Notebooks (`notebooks/*.ipynb`, `figures/*.ipynb`): run each with
    `jupyter nbconvert --to notebook --execute --output-dir /tmp/nbcheck`
    (never overwrite them). If a cell fails only because data is
    missing on this machine, say so: it is not a restructure failure.
    All import cells must run.
  - This grep returns nothing:
    `grep -rn "data_manipulation\|euclid_des_preparation\|analysis\.fit\|analysis\.pipeline\|plotting\.plotting\|analysis\.utils\|streameucliddes\.utils\|streameucliddes import utils" src tests notebooks figures docs`
  - Docs build: `python -m sphinx -b html docs/source docs/_build/html`.
- End with a short summary for the user: moved files, renamed functions,
  logic changes, missing functions found or not, notebook results.

### Phase 2: empty stubs for known streams (branch `known-stream-stubs`, from phase 1)

- Start only after the user approves phase 1.
- Create every `[NEW]` file and function of `package_structure.md`.
- Each stub: full signature with type hints, numpy-style docstring
  (purpose, parameters, returns, code to reuse from Section 4,
  references), body `raise NotImplementedError`.
- Add `catalogs/stream_populations.csv` with the header only.
- Add one test per new module that only checks the import.
- Add `notebooks/known_stream_search_example.ipynb`: an example of how
  to use the known-stream search, written against the stubs. Markdown
  explains each step (galstreams streams in the footprint, stream frame,
  isochrone, map per dm, video, file of maps); code cells call the public
  functions in the order a user would. Only the import cell runs at this
  stage: the other cells call stubs. Stored without outputs (git filter).
- No implementation. The user reviews and may reorganise the stubs.

### Phase 3: fill the stubs (later, on request)

- Fill only the stubs the user names, one function or module at a time.
- **Do not** rename, move, merge or delete stubs. Do not change a
  signature. If one looks wrong, propose a change and wait.
- Private helpers (`_name`) in the same module are fine. New public
  functions: ask first.
- Write a real test for each filled function.
- Reuse code before writing new code (Section 4).
- Once the functions it calls are filled, make
  `notebooks/known_stream_search_example.ipynb` run end to end on a
  validation stream of Section 7.

## 3. Where to find celestial_currents

A collaborator's package with stream-search tools for DELVE DR3
(DES g/r). It is **on this machine**, not in the repo. Find it with:

```bash
python -c "import celestial_currents, os; print(os.path.dirname(celestial_currents.__file__))"
find ~ -maxdepth 6 -type d -name celestial_currents -not -path "*/.Trash/*" 2>/dev/null
```

Read its `CLAUDE.md` and `README.md` first.

- **Read** it freely. **Never edit** it.
- **Never import** it from `streameucliddes`: hard-coded cluster paths
  (`survey.py::_DELVE_LOCAL`, `investigate/postprocess.py::_LVDB_ROOT`)
  and heavy dependencies (lsdb, hats, skyproj).
- **Copy and adapt** what you need. Credit it in the docstring:
  `Adapted from celestial_currents <file>::<function>.`
- `visualization.py` and `investigate/visualization.py` are two
  versions. The top-level one is newer (`fit_bkg_poly`, `mask_phi12_disks`).

## 4. Reuse map for the new stubs (phases 2 and 3)

[C] = celestial_currents (paths from `src/celestial_currents/`).
[U] = this repo, **names after phase 1**.

| New stub | Reuse |
|------|-------|
| `stream_frames.radec_to_phi12`, `phi12_to_radec` | gala frames + `SkyCoord.transform_to` (numpy arrays in degrees in and out) |
| `catalogs.isochrone_parameters.*` | [U] `unit_conversions.convert_FeH_to_z`; [C] grid in `configs/delve_dr3_mz_scan.yaml` |
| `galstreams_catalog.load_galstreams` | [C] `galstreams.MWStreams(verbose=False)` in `notebooks/create_projections.ipynb`; [U] `unit_conversions.convert_kpc_to_DM` |
| `galstreams_track_length_in_footprint`, `select_galstreams_in_footprint` | [C] `visualization.py::_find_overlapping_streams` |
| `galstreams_phi12_window` | [C] `visualization.py::get_stream_phi12_bounds` |
| `galstreams_dm_along_track` | none; interpolate galstreams track distances in phi1 |
| `survey_footprint.*` | [U] footprint mask in `star_density_healpix` and `background_polynomial_fit.process_data`; [C] `visualization.py::transform_healpix_to_custom_frame` |
| `cmd_match_filter.select_stars_in_match_filter` | [U] `build_match_filter_polygon` + `streamobs.match_filter.is_in_match_filter` (see `_count_selected` in `injected_stream_snr`); [C] DES g−r alternative `search/matched_filter.py::get_filter_splines_from_color_mag`, `filter_data` |
| `star_density_maps.star_density_phi12` | [C] `_compute_spline_phi12` in `notebooks/candidate_investigations/full_search/template.ipynb` (histogram2d + Gaussian smoothing) |
| `known_stream_search.*` | [U] `generic_search_dm_scan` pattern (cuts computed once); [C] `ProcessPoolExecutor` sweep cells in the template notebook |
| `plot_known_stream_search.*` | [C] `visualization.py::plot_custom_stream_at_mu`, `_find_overlapping_streams` (rewrite with imshow on the grid) |
| `make_videos.make_gif_from_plots` | [C] `visualization.py::make_gif_from_plots` (copy as is). **User choice: all new gifs use this function**, not [U] `make_gif` |
| `make_videos.make_mp4` | `imageio` + `imageio-ffmpeg`; same interface as `make_gif_from_plots` |
| `run_config.RunConfig` | [C] `config.py::RunConfig.from_yaml` |
| `save_generic_search_healpix_maps`, `save_known_stream_search_maps` | [C] output block of `scripts/run_matched_filter_maps.py::main` |
| CLI scripts | [C] `scripts/run_matched_filter_maps.py` |

## 5. Known gaps

- **galstreams has no age or [Fe/H].** Curated CSV. Sources: Li et al.
  2022 (ApJ 928, 30); Martin et al. 2022 (MNRAS 516, 5331); Bonaca &
  Price-Whelan 2024 (arXiv:2405.19410). Never invent values: NaN →
  default grid.
- **Footprint.** Start from star counts; use DES Y6 and Euclid DR1
  coverage maps when available.
- **Edges.** Normalise smoothed counts by the smoothed footprint mask.
- **Euclid isochrones.** Check `streamobs` has them for each (age, Z).
- **Dust.** The Euclid × DES magnitudes are already dereddened: DES uses
  `BDF_MAG_<B>_CORRECTED` (SFD98, by DES); Euclid is dereddened in
  `euclid_des_catalog.add_magnitudes` (`EBV_SFD98_des` × the streamobs
  coefficients). Do not correct them again. `generic_search_single_dm`
  defaults to `use_dust_corrected=True`: pass `False` on this catalog.
  `correct_dust_map` stays missing (it raises) unless it is needed.

## 6. Repo conventions

- Env: `conda activate streameuclid` (celestial uses `search26`).
- Stream frames: gala `GreatCircleICRSFrame` (galstreams `stream_frame`,
  or `GreatCircleICRSFrame.from_endpoints` for another stream). Use gala
  for the transformations as much as possible.
- Tests: `pytest`, one marker per module in `pyproject.toml`.
- Docs: Sphinx autosummary (recursive), `docs/source/api.rst`.
- Numpy-style docstrings, short sentences.
- HEALPix: always pass `nest=` explicitly. Celestial uses NESTED.
- Angles in degrees. Distances as distance modulus unless named `_kpc`.
- New dependencies (galstreams, gala, pyyaml, imageio): add to
  `pyproject.toml` and tell the user.
- Commit and push only on the phase branch. Never on `main`.

## 7. Validation (phase 3)

- Test known-stream maps first on known DES streams with a DES-only
  footprint (e.g. ATLAS, Phoenix, Turranburra; Shipp et al. 2018).
- Inject a stream with `injection.inject_stream_catalog` on a known
  track; check it appears at phi2 ≈ 0.

## 8. Later: left out of the first version

Start simple (user choice, phase 2 review). Add these once the simple
version works, and only on request:

- **Polynomial degree from the map fluctuations** (generic search):
  choose among degrees 3–5, e.g. by the lowest BIC (Bayesian Information
  Criterion) on a Poisson deviance, ties going to the lower degree. Reuse:
  [U] `fit_background_model`; [C] `visualization.py::fit_bkg_poly`.
- **Masking known objects in the maps**: disks around the dwarf galaxies
  and globular clusters of the LVDB (Local Volume Database; Pace 2024,
  arXiv:2411.07424). Reuse: [C] `visualization.py::mask_phi12_disks`;
  [C] `investigate/postprocess.py::build_lvdb_mask` (LVDB path as an
  argument).
- **Several isochrones in one map** (e.g. the union of their filters),
  instead of one set of maps per isochrone.
