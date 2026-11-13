# Plan for Claude Code: stream-candidate investigation tools

Repository: `streameucliddes` (checked on `main`, commit fe06c49, 2026-11-13;
re-checked against `main` 2796f5c after the review below).
Read this file fully before any change. Copy it to `docs/dev/` of the repo.
Read also `docs/dev/package_structure.md` (layering rule, naming rules).

## 0. Goal and design rules

Investigate one stream candidate in a notebook:

1. Frame from two sky endpoints.
2. Matched-filter (MF) map in that frame.
3. Stream track and width; on-stream and off-stream regions.
4. CMDs (Hess diagrams) of the on, off and on − off stars, in stars/deg².
5. Isochrone overlay for an age, a Z or [Fe/H], and a distance modulus.
6. The stream track drawn on the MF maps, to show which stream is studied.

**Simplicity first.** This is an investigation tool. It will be refactored
often. So:

- Two layers. (1) Plain functions returning plain `dict`s: each can be
  called alone and tuned. (2) One thin class, `CandidateInvestigation`
  (Section 4, C7), that holds every input and result and calls the
  functions in order. The class adds no science: each method is a few
  lines calling one function. Plots take the class or the dicts.
- No caching, no workflow, no YAML config, no script. The notebook is the
  driver.
- Short functions, one job each. Default values as module constants at the
  top of each file.
- Reuse existing functions. Write new code only when nothing exists.
  Never import `celestial_currents`; copy small pieces and cite them in
  the docstring (`Notes`).
- Keep the layering rule: `analysis/` computes, `plotting/` draws only.
- Numpy docstrings, as in the rest of the package. Cite the literature in
  `References`.

## 1. Agreed decisions (user, 2026-11-13)

- Rename `known_stream_search` → `stream_frame_search`: it already works for
  any frame and window, without a galstreams track.
- Track: constant phi2 (`phi2_centre`) for now. A phi2(phi1) polynomial may
  come later.
- Width: Gaussian sigma in phi2, fitted with a constant background
  (Shipp et al. 2018, ApJ 862, 114).
- On region: |phi2 − phi2_centre| ≤ 1.4σ (`on_n_sigma`), over `phi1_range` (user choice).
- Off region: two bands, one on each side, at 2.5σ to 4.5σ from
  `phi2_centre`, over the same `phi1_range`.
- One class holding all quantities, so the user can run the steps without
  calling each function by hand, and pass it to the plots (user, 2026-11-13).
- **Frame check first** (user, 2026-11-13): one of the first notebook steps
  shows, side by side, the MF map of the loaded box with the chosen great
  circle and window, and the zoomed MF map in the stream frame. The user
  then adjusts the endpoints by hand and re-runs this one cell, until the
  frame follows the stream. This loop must be fast and need one call.
- Example notebook on Euclid DR1 south, candidate endpoints
  (ra, dec) = (77, −40) and (70, −43) deg (user, 2026-11-13).
- CMDs normalized per deg². Areas computed inside the footprint and outside
  the masked objects (LMC, dwarfs, clusters, galaxies).
- Isochrones: the ugali Marigo et al. 2017 (PARSEC) model, the same as the
  match filter (Marigo et al. 2017, ApJ 835, 77). [Fe/H] → Z with the
  existing `unit_conversions.convert_FeH_to_z` (Bressan et al. 2012).
- Frame: gala `GreatCircleICRSFrame.from_endpoints` (Price-Whelan 2017,
  JOSS 2, 388). Do **not** copy celestial `frame_utils.stream_frame_from_endpoints`:
  it builds a `SkyOffsetFrame` with `rotation=0` and ignores the pole, so
  phi1 is not along the stream (checked in its source).

Review of the plan (user, 2026-11-13, second round):

- Nothing is built in the notebook: the class builds the excluded HEALPix
  mask (footprint OR masked objects).
- Masks done one way (see "Masks" below): one HEALPix mask of the excluded
  sky, RING, at the object-mask nside, shared by the stars, the maps and the
  areas.
- The phi2 and phi1 profiles are densities (stars/deg²): each bin divided by
  its area outside the excluded mask (Euclid tile edges and masked objects
  would otherwise make dips that bias the width fit). One function for both
  (`phi12_density_profile`, Sect. 4 C4).
- Isochrone: reuse `streamobs.match_filter._isochrone_locus`, the function
  `build_match_filter` calls (same model by construction).
- phi1 margin of the window: `padding_phi1_deg = 5.0`, the same name and
  default as `galstreams_phi12_window`. Initial sigma of the width fit 0.2
  deg, as a module constant.
- Rename where a name is unclear or clashes (e.g. class fields vs functions).

Facts checked (2026-11-13):

- gala 1.10.1, `from_endpoints(c1, c2)`: origin (phi1 = 0) at the midpoint
  of the arc. For (77, −40), (70, −43): endpoint_1 at phi1 = −3.02 deg,
  endpoint_2 at +3.02 deg, both at phi2 ~ 1e−15 deg: phi1 increases from
  endpoint_1 to endpoint_2.
- `convert_FeH_to_z(-1.9)` = 1.96e−4.
- Euclid DR1 is in `data/euclid_dr1`. The candidate is ~30 deg from the LMC,
  outside its 21 deg mask.

### Masks

`prepare_search` builds the object mask at `object_mask_nside(bin_size_deg)`
(2048 for 0.1 deg bins), RING, but keeps only its version on the phi1/phi2
grid. The footprint mask is at nside 512 (from the stars, or
`cand.sky_map["footprint_mask"]`). One way for both:

- `PreparedSearch` gets one more field, `object_healpix_mask` (RING, at
  `object_mask_nside`), the mask `prepare_search` already builds (minimal
  hook, default path unchanged).
- One function, `analysis.build_masks.excluded_healpix_mask(footprint_mask,
  object_healpix_mask, *, nest)`: footprint brought to the object nside and
  RING (`hp.ud_grade`, exact when going up), OR the objects. Returns
  `(mask, nside)`, RING.
- The class keeps it in `excluded_healpix_mask`; `region_area_deg2` and the
  profiles use it. Stars (dropped by object pixel in `prepare_search`), maps
  (bins of the grid) and areas then see the same objects.
- `set_endpoints` speed: the object mask does not depend on the frame. Time
  `prepare_search` in S3; if building the object mask dominates, let
  `prepare_search` take an `object_healpix_mask` already built (report first).

## 2. What already exists (reuse, do not rewrite)

| Need | Existing function |
|---|---|
| Stars in an (ra, dec) box | `catalogs.euclid_catalog.get_euclid_stars("euclid_dr1", ra_range=..., dec_range=...)` |
| phi1/phi2 window → (ra, dec) box | celestial `frame_utils.phi12_bounds_to_radec_bounds` (copy the idea, sample the window edges) |
| MF map of the loaded box (HEALPix) | `analysis.generic_stream_search.generic_search_single_dm` (`density`, `residual`, `footprint_mask`, `nest`) |
| Sky map plot with tracks | `plotting.plot_generic_search.plot_generic_search_residual_map(result, tracks=[(label, ra, dec), ...])` |
| (ra, dec) ↔ (phi1, phi2) | `stream_frames.radec_to_phi12`, `phi12_to_radec` |
| Grid of a window | `stream_frames.phi12_bin_centres` |
| Stars prepared in the frame (cuts, masks, phi1/phi2) | `analysis.stream_frame_search.prepare_search` → `search.stars` |
| MF map at one dm / over a dm grid | `stream_frame_search_single_dm`, `stream_frame_search_dm_scan` |
| MF selection | `analysis.cmd_match_filter.select_stars_in_match_filter` |
| Magnitude columns of a match filter | `cmd_match_filter.match_filter_columns` |
| MF polygons | `cmd_match_filter.match_filter_polygons` |
| Footprint HEALPix mask | `catalogs.survey_footprint.survey_footprint_healpix_mask` |
| Masked objects HEALPix mask | `analysis.build_masks.overlapped_objects_healpix_mask`, `object_mask_nside` |
| MF map plot, with a track | `plotting.plot_stream_frame_search.plot_stream_frame_search_single_dm(result, track_phi12=...)` |
| Track on a sky (skyproj) map | `plotting.plot_stream_frame_search._split_at_map_edge` (edge splitting), used in `plot_generic_search_residual_map` with `KNOWN_TRACK_STYLE` |
| CMD 2D histogram | `plotting.plot_cmd.plot_CMD`, `plot_match_filter` |
| Z of a stream, default isochrone | `catalogs.isochrone_parameters` |
| Isochrone locus, as the match filter | `streamobs.match_filter._isochrone_locus(model, age, Z, band_1, survey_1, band_2, survey_2)` |
| Hess difference, plot style | celestial `investigate/hess.py`: `compute_difference_cmd`, `plot_difference_cmd` (RdBu_r, symmetric limits, optional smoothing) |

## 3. Phases. Stop after each one for review.

### Phase A: rename (branch `rename-stream-frame-search`) — done 2026-11-13

Moves and renames only, no logic change. Use `git mv`.

| Old | New |
|---|---|
| `analysis/known_stream_search.py` | `analysis/stream_frame_search.py` |
| `known_stream_search_single_dm` | `stream_frame_search_single_dm` |
| `known_stream_search_dm_scan` | `stream_frame_search_dm_scan` |
| `known_stream_search_dm_along_track` | `stream_frame_search_dm_along_track` (still galstreams only; say it in the docstring) |
| `plotting/plot_known_stream_search.py` | `plotting/plot_stream_frame_search.py` |
| `plot_known_stream_search_single_dm` | `plot_stream_frame_search_single_dm` |
| `tests/test_known_stream_search.py`, `tests/test_plot_known_stream_search.py` | `test_stream_frame_search.py`, `test_plot_stream_frame_search.py` |

- `prepare_search`, `PreparedSearch` and `plot_galstreams_selection` keep
  their names.
- `workflows/run_known_stream_search.py`, `scripts/run_known_stream_search.py`
  and `configs/known_stream_search.yaml` keep their names: they are about
  known streams. Update their imports only.
- Update the module docstrings ("known-stream search" → "search in a
  stream frame", known streams being one use).
- Update all imports and mentions: `src/`, `tests/`, `notebooks/`,
  `docs/source/api.rst`, `docs/dev/*.md`, `README.md`. Check with
  `grep -rn "known_stream_search_single_dm\|known_stream_search_dm\|analysis.known_stream_search\|plot_known_stream_search"`.
- No compatibility aliases.
- Checks: `pytest` passes; `notebooks/known_stream_search_example.ipynb`
  runs (`jupyter nbconvert --to notebook --execute --output-dir /tmp/nbcheck`,
  never overwrite). Missing data on the machine is not a failure: report it.

### Phase B: empty stubs (branch `candidate-investigation`)

Done 2026-11-13. Changes from Section 4, made while writing the stubs:

- `isochrone_cmd_track` is in `analysis/cmd_match_filter.py` (next to the
  polygons, constant `ISOCHRONE_MODEL`), not in `catalogs/isochrone_parameters.py`:
  it needs the match filter, and `catalogs/` must not import `analysis/`.
- One plotting module, `plotting/plot_candidate_investigation.py` (user,
  2026-11-13): the tracks, regions and profiles, and the `plot_candidate_*`
  wrappers. No `plot_stream_regions.py`.
- `draw_stream_track`, `draw_stream_track_on_sky`, `draw_stream_regions`
  return the lines they add (the regions test counts 3); `plot_isochrone`
  and `plot_match_filter_polygon` return their line; `plot_*` return
  `(fig, ax)`. `draw_stream_track` has `phi2_offset_deg`.
- `on_off_phi1_profile` has a `phi1_range` for its bins (default:
  `regions["phi1_range"]`), so that the profile can extend over the window
  and show where the stream stops.
- Second review (user, 2026-11-13): window code in `stream_frames.py` only,
  one window format (`phi12_window_around` shared with
  `galstreams_phi12_window`; regions as windows); no `stream_regions.py` nor
  `hess_diagrams.py`; Hess diagrams computed in `cmd_match_filter.py` and
  drawn by `plot_hess` only, which `plot_CMD` will call. Section 4 is
  rewritten for this layout.
- `run` renamed `run_all_steps` (user, 2026-11-13, a more explicit name):
  `run_all_steps(catalog, *, footprint_mask=None, nest=False)`: every other input comes
  from the fields.
- `PreparedSearch.object_healpix_mask` (Masks, Sect. 1) is not added yet: it
  changes `prepare_search`, so it comes with S3. `excluded_healpix_mask` is a
  stub in `build_masks.py`.

Create every `[NEW]` function of Section 4 with its full signature and
numpy docstring, and a body `raise NotImplementedError`. Create the test
files with the test names and `pytest.mark.skip(reason="stub")`. Create the
notebook of Section 5 with its markdown cells and empty code cells. Add the
new modules to `docs/source/api.rst`. `pytest` must pass (skips allowed).
Stop for review of names and signatures.

### Phase C: fill, one notebook step at a time (same branch)

The notebook drives the work. The class is filled **together** with the
functions, not at the end. The user asks for one step at a time (S1 to S9
below). For each step, fill together:

- the functions of Section 4 that the step needs (and only those),
- the class method(s) of C7 that call them,
- the plot(s) the step needs,
- the notebook cell(s) of the step, which use the class method.

Then:

1. Write the tests of the functions and of the method (synthetic data).
2. `pytest` on the new tests, then the whole suite once.
3. Run the notebook up to this step on Euclid DR1 if the data are there.
4. Report: what was reused, what is new, test results, the step's figure.
   Then stop. The user runs the cell, checks it, and asks for the next step.

| Step | Notebook section (Sect. 5) | Functions | Class | Plots |
|---|---|---|---|---|
| S1 | 1–2 Inputs, frame, window | C1: `stream_frame_from_endpoints`, `phi12_window_radec_box`, `phi12_window_outline` (`phi12_window_around` done) | dataclass fields, `__post_init__` | — |
| S2 | 3 Catalog, MF map of the loaded box | (existing `generic_search_single_dm`) | `compute_sky_map` | (existing `plot_generic_search_residual_map`, + `ax` argument) |
| S3 | 4 Frame check, endpoints by hand | (existing `prepare_search`, `stream_frame_search_single_dm`); C3: `excluded_healpix_mask` | `prepare`, `compute_zoom_map`, `set_endpoints` | `plot_candidate_frame_check`, `draw_stream_track_on_sky`, `draw_stream_track` |
| S4 | 5 MF maps over dm, in the frame | — | `scan_distance_moduli`, `track_phi12`, `track_radec` | `plot_candidate_map` (without regions) |
| S5 | 6 Width | C3: `unmasked_area_deg2`; C4: `phi12_density_profile`, `fit_stream_width` | `fit_width` | `plot_phi2_profile` |
| S6 | 7 Regions | C1: `in_phi12_window`; C4: `stream_regions`, `region_windows`, `region_area_deg2`, `on_off_phi1_profile` | `define_regions` | `draw_stream_regions`, `plot_phi1_profile`; regions in `plot_candidate_map` |
| S7 | 8 CMDs | C5: `hess_diagram`, `difference_hess` | `compute_hess` | `plot_hess` (and `plot_CMD` through it), `plot_on_off_difference_hess`, `plot_candidate_hess` |
| S8 | 9 Isochrone | C5: `isochrone_cmd_track` | `compute_isochrone` | `plot_isochrone`, `plot_match_filter_polygon`; isochrone in `plot_candidate_hess` |
| S9 | 10–11 Summary, all at once | — | `summary`, `run_all_steps` | `plot_candidate_overview` |

Sections C1–C7 below describe the functions; the table says when each is filled.

## 4. New functions, by module

Layout after the second review of the stubs (user, 2026-11-13): window code
in one place (`stream_frames.py`), one window format, few modules. The
docstrings of the stubs are the reference for the signatures; this section
says what goes where and what each test checks. Plain dicts are described
by their keys.

A **window** is `(phi1_min, phi1_max, phi2_min, phi2_max)` (deg): the window
of the maps, the on region and each off band.

### C1. Windows and frame — `stream_frames.py`

- `phi12_window_around(phi1, phi2, *, padding_phi1_deg=5, padding_phi2_deg=2,
  min_phi2_half_width_deg=5)`: the core of `galstreams_phi12_window`, moved
  here (done in Phase B; `galstreams_phi12_window` calls it, unchanged).
  The candidate window is `phi12_window_around` of its two endpoints:
  phi1 between them ± 5 deg, phi2 ±5 deg (min half-width).
- `in_phi12_window(phi1, phi2, window)`: stars in a window (on region, off
  bands).
- `phi12_window_outline(window, *, n_points_per_edge=50)`: closed outline in
  phi1/phi2; on the sky with `phi12_to_radec`.
- `phi12_window_radec_box(stream_frame, window, *, margin_deg=0.5)`: (ra,
  dec) box from the outline, for `get_euclid_stars`; warns across ra = 0.
- `stream_frame_from_endpoints(endpoint_1, endpoint_2)`: gala
  `GreatCircleICRSFrame.from_endpoints` (Sect. 1, facts). The great circle
  on a sky map is `phi12_to_radec(phi1, 0, frame)`: no function.
- Tests: window around points (done); `in_phi12_window` edges; outline
  closed; box contains the window, wrap warning; endpoints at phi2 = 0 (1e-8
  deg), phi1 symmetric, endpoint_1 negative.

### C2. MF map of the candidate — no new function

`prepare_search(catalog, stream_frame=frame, window=window, footprint_mask=..., nest=..., isochrone=..., distance_moduli=...)`,
then `stream_frame_search_single_dm(search=search, distance_modulus=dm)` or
`stream_frame_search_dm_scan(search=search)`. If something blocks a frame
without a track, report it before changing `prepare_search`.

### C3. Masks and areas — `analysis/build_masks.py`, `catalogs/survey_footprint.py`

- `build_masks.excluded_healpix_mask(footprint_mask, object_healpix_mask, *, nest)`
  → `(mask, nside)`, RING (Sect. 1, Masks).
- `survey_footprint.unmasked_area_deg2(stream_frame, phi1_edges, phi2_edges,
  excluded_healpix_mask, *, step_deg=0.01)`: area of each cell outside the
  mask (fine grid, cos(phi2) weights, pixel look-up as
  `survey_footprint_phi12_mask`, sharing a private helper). A window is one
  cell.
- Tests: footprint upgraded to the object nside; OR of the objects; no
  objects; coarser objects rejected. No mask: area Δφ1 · (sin φ2,max − sin
  φ2,min) · (180/π)² within 1 %; half the sky masked: about half; cells sum
  to the whole window.

### C4. Regions, profiles, width, and the class — `analysis/candidate_investigation.py`

No `stream_regions.py` (user, 2026-11-13: it would only hold the regions
dict, the profiles and the width fit, all candidate-specific). Functions
first, then the class (C7).

```python
DEFAULT_ON_N_SIGMA = 1.4          # user choice
DEFAULT_OFF_N_SIGMA = (2.5, 4.5)  # user choice
DEFAULT_SIGMA_GUESS_DEG = 0.2
DEFAULT_PHI2_BIN_DEG = 0.05
DEFAULT_PHI1_BIN_DEG = 0.5

stream_regions(phi1_range, phi2_centre=0.0, sigma=DEFAULT_SIGMA_GUESS_DEG, *, on_n_sigma, off_n_sigma)
    # plain dict, edit it by hand
region_windows(regions)            # {"on": window, "off": (below, above)}, computed each time
region_area_deg2(regions, stream_frame, excluded_healpix_mask)   # {"on", "off"} deg²
phi12_density_profile(phi1, phi2, windows, stream_frame, excluded_healpix_mask, *,
                      axis, bin_deg, axis_range=None)
    # one function for both profiles: counts in bins along phi1 or phi2,
    # over one or more windows, / unmasked area of each bin -> stars/deg²
fit_stream_width(profile, *, phi2_centre_guess=0.0, sigma_guess=DEFAULT_SIGMA_GUESS_DEG)
    # Gaussian + constant, curve_fit, Poisson errors (Shipp et al. 2018)
on_off_phi1_profile(phi1, phi2, regions, stream_frame, excluded_healpix_mask, *,
                    phi1_range=None, bin_deg=DEFAULT_PHI1_BIN_DEG)
    # on and off (two bands) profiles along phi1, and on - off
```

- No track function: the track is straight, `CandidateInvestigation.track_phi12`
  and `track_radec` draw it (phi2 = phi2_centre over phi1_range; phi2 = 0
  between the endpoints before the regions).
- Tests (synthetic): regions keys, overlap rejected; windows disjoint;
  region areas analytic; uniform stars with half the window masked give a
  flat phi2 density; two windows add up; uniform + Gaussian stream → centre
  and σ recovered within 3 errors; on − off along phi1 ≈ 0 without stream,
  positive only where the stream is.

### C5. CMDs, Hess diagrams, isochrone — `analysis/cmd_match_filter.py`

Computed here, drawn by `plotting/plot_cmd.py` (`analysis/` computes,
`plotting/` draws). No `hess_diagrams.py`.

- `hess_diagram(stars, mag_cols, *, mask=None, area_deg2=None,
  color_bins=None, mag_bins=None)`: 2D histogram of colour `mag_cols[0] −
  mag_cols[1]` and magnitude `mag_cols[0]` (the convention of the polygons
  and of `plot_CMD`); counts, or stars/deg²/bin with `area_deg2`. Defaults:
  `DEFAULT_COLOR_RANGE = (-1.0, 1.5)`, `DEFAULT_MAG_RANGE = (17, 27)`, bins
  0.04 and 1/6 mag (celestial `hess.py`). Shipp et al. 2018, Sect. 5.
- `difference_hess(on_hess, off_hess)`: on − off, errors in quadrature;
  `ValueError` if the bins differ.
- `isochrone_cmd_track(distance_modulus, age, metallicity=None, *, feh=None,
  match_filter=None, color_index=0)`: `streamobs.match_filter._isochrone_locus`
  (model `ISOCHRONE_MODEL = "Marigo2017"`, band_1 = reference band, surveys
  as `match_filter_polygons`), + dm. Here, not in `catalogs/`: it needs the
  match filter, and `catalogs/` must not import `analysis/`.
- Tests: counts without area; one star in 2 deg² → 0.5 in one bin; mask;
  on and off from one uniform population, areas 1 and 3 → |diff|/error ~
  N(0, 1); different bins rejected. Isochrone: dm 20 = dm 15 + 5 mag, same
  colour; feh −1.9 → Z ≈ 0.0002; MSTO part inside the polygon; exactly one
  of Z and feh.

### C6. Plots — `plotting/plot_cmd.py`, `plotting/plot_candidate_investigation.py`

In `plot_cmd.py`:

- `plot_hess(hess, *, ax=None, difference=False, cmap=None, norm=None,
  vlim=None, smoothing_bins=0.0, colorbar=True, title=None)`: the only
  function drawing CMD histograms (counts, per deg², or on − off with RdBu_r
  and symmetric limits ±vlim, default 98th percentile of |density|).
- `plot_CMD` keeps its signature and defaults (used by the notebooks); in
  S7 its body becomes `hess_diagram(...)` then `plot_hess(...)` (user,
  2026-11-13: one drawing path).
- `plot_on_off_difference_hess`, `plot_isochrone`, `plot_match_filter_polygon`
  (factored out of `plot_match_filter`, which then calls it).

In `plot_candidate_investigation.py` (one module, user 2026-11-13):

- `draw_stream_track(ax, phi1, phi2, *, label=None, phi2_offset_deg=0.0,
  **line_kwargs)`, `draw_stream_track_on_sky(ax, ra, dec, ...)` (split at the
  map edge with `plot_stream_frame_search._split_at_map_edge`),
  `draw_stream_regions(ax, regions, ...)` (`phi12_window_outline` of
  `region_windows`): return the lines they add. Make
  `plot_stream_frame_search_single_dm(track_phi12=...)` call
  `draw_stream_track` (one style for all tracks).
- `plot_phi2_profile(profile, fit=None, *, ax=None)`,
  `plot_phi1_profile(profile, regions=None, *, ax=None)`: `(fig, ax)`.
- Wrappers taking the class: `plot_candidate_frame_check(cand, *,
  sky_kwargs=None, zoom_kwargs=None)` (left: `cand.sky_map` residual with the
  great circle, window outline and endpoints; right: `cand.zoom_map` with
  the endpoints and phi2 = 0; same dm in the title),
  `plot_candidate_map(cand, distance_modulus=None, *, ax=None,
  show_regions=True)`, `plot_candidate_hess(cand, which="difference", *,
  ax=None, isochrone=True, match_filter=False)`, `plot_candidate_overview(cand)`.
- `plot_generic_search_residual_map` gets an optional `ax` (S2), so that the
  frame check is one figure with two panels; old behaviour by default.
- Tests: each function runs on synthetic inputs; `plot_hess(difference=True)`
  has symmetric limits; `draw_stream_regions` adds 3 outlines. Style:
  `plot_style`, `CATEGORICAL_COLORS`.

### C7. The class — `CandidateInvestigation` in `analysis/candidate_investigation.py`

Fields (inputs): `name`, `endpoint_1`, `endpoint_2`, `age`, `metallicity`
(or `feh`), `distance_modulus`, `distance_moduli`, `match_filter`,
`padding_phi1_deg`, `padding_phi2_deg`, `min_phi2_half_width_deg`,
`on_n_sigma`, `off_n_sigma`, `prepare_kwargs`. Fields (results, None until
computed): `stream_frame`, `window`, `radec_box`, `catalog` (a reference,
no copy), `search`, `excluded_healpix_mask`, `sky_map`, `zoom_map`, `maps`,
`phi2_profile`, `width_fit`, `regions`, `areas`, `phi1_profile`, `on_hess`,
`off_hess`, `diff_hess`, `isochrone_track`.

Methods: `__post_init__` (frame, window, box), `compute_sky_map`,
`prepare`, `set_endpoints`, `compute_zoom_map`, `scan_distance_moduli`,
`fit_width`, `define_regions`, `compute_hess`, `compute_isochrone`,
`run_all_steps(catalog, *, footprint_mask=None, nest=False)` (renamed from
`run`, user 2026-11-13), `track_phi12`, `track_radec`, `summary`.

- Each method: check that the previous step was done (clear `ValueError`
  naming the missing step), call one or two functions, store the result,
  return `self`.
- Defaults when an argument is None: `phi1_range` → the endpoints' phi1;
  `phi2_centre`, `sigma` → `width_fit`. `distance_modulus` has no default:
  the user picks it by eye after the scan; `fit_width` raises if it is
  missing.
- `prepare`: footprint, when not given, `sky_map["footprint_mask"]` (with
  its `nest`), else from the stars (`survey_footprint_healpix_mask(ra, dec,
  512, nest=False)`); `excluded_healpix_mask` from
  `build_masks.excluded_healpix_mask(footprint, search.object_healpix_mask,
  nest=...)`.
- `define_regions` raises if the off bands leave the window.
- `set_endpoints` must be fast: the sky map is kept (frame independent);
  only `prepare_search` on the stored catalog and one zoom map. If the new
  window leaves the loaded box, a clear error saying to load a larger box.
- Tests: a synthetic catalog (uniform + Gaussian stream along the endpoints
  arc) through `run_all_steps`; σ and phi2_centre recovered; steps out of
  order raise; `summary` keys.

## 5. Example notebook — `notebooks/candidate_investigation_example.ipynb`

Data: Euclid DR1 south (`get_euclid_stars("euclid_dr1", ra_range, dec_range)`,
Euclid only, no DES). Candidate endpoints: (ra, dec) = (77, −40) and
(70, −43) deg. Default isochrone: 13 Gyr, Z = 0.0002
(`default_isochrone_grid()[0]`). Default match filter: VIS vs VIS − H.

The notebook uses the class step by step (one method per cell), so each
result can be looked at and tuned before the next step. Where a step needs
tuning beyond the class arguments, the cell shows the direct function call
and puts the result back in the class (e.g. `cand.regions = stream_regions(...)`).
A last cell shows `cand.run_all_steps(...)` doing everything at once.

Sections (one markdown title each, a few code cells):

1. **Inputs.** `CandidateInvestigation(name, endpoint_1=(77, -40),
   endpoint_2=(70, -43), age=13, metallicity=0.0002, ...)`. All inputs in
   one cell.
2. **Frame and window.** Set at construction (`cand.stream_frame`,
   `cand.window`); print the (ra, dec) box.
3. **Catalog and MF map of the loaded box.** Load Euclid DR1 south once, in a
   box generously larger than the candidate (so the endpoints can move). The
   "sky map" is the MF map of this box, not of the whole survey.
   `cand.compute_sky_map(catalog, distance_modulus=...)`; show it with the
   great circle (`plot_generic_search_residual_map`). Computed once.
4. **Frame check (main loop).** `cand.prepare()`, `cand.compute_zoom_map()`,
   `plot_candidate_frame_check(cand)`. Then one cell the user edits and
   re-runs:
   `cand.set_endpoints((77, -40), (70, -43)).compute_zoom_map(); plot_candidate_frame_check(cand)`.
   Assert that the window contains the off bands.
5. **MF maps over dm, in the frame.** `cand.scan_distance_moduli()`; plot a
   few dm with `plot_candidate_map`; keep the best dm by eye
   (`cand.distance_modulus = ...`).
6. **Width.** MF-selected stars → `phi12_density_profile` along phi2 → `fit_stream_width` →
   `plot_phi2_profile`.
7. **Regions.** `stream_regions` (phi1_range by eye, then from
   `on_off_phi1_profile`); `draw_stream_regions` on the MF map; `region_area_deg2`.
8. **CMDs.** `hess_diagram` on and off (same bins), `difference_hess`,
   `plot_on_off_difference_hess`; then the difference alone with
   `plot_hess(difference=True)`.
9. **Isochrone.** `isochrone_cmd_track` + `plot_isochrone` on the difference
   Hess; optional `plot_match_filter_polygon`. Try 2–3 (age, [Fe/H], dm).
10. **Summary.** `cand.summary()`; `plot_candidate_overview(cand)`.
11. **All at once.** A new `CandidateInvestigation` with the chosen dm and
    `cand.run_all_steps(catalog)`, to check it gives the same results.

Notebooks are committed without outputs (git filter in `.gitconfig`).

## 6. Out of scope for now

- phi2(phi1) polynomial track; significance of the on − off excess;
  isochrone fit to the difference Hess; saving candidates to files;
  workflows and configs. Add later if the investigation needs them.

## References

- Hess diagrams on − off: Shipp et al. 2018, ApJ 862, 114.
- Matched filter: Rockosi et al. 2002, AJ 124, 349; Grillmair 2009, ApJ 693, 1118.
- Isochrones: Marigo et al. 2017, ApJ 835, 77; Bressan et al. 2012, MNRAS 427, 127.
- Stream frames: Price-Whelan 2017, JOSS 2, 388 (gala).
- Normalized convolution (existing maps): Knutsson & Westin 1993.
- Off band 2.5σ–4.5σ and on band 1.4σ: user choices, no reference.
