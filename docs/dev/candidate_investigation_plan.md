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
  would otherwise make dips that bias the width fit). Hence the names
  `phi2_density_profile`, `phi1_density_profile`.
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
- `phi1_density_profile` has a `phi1_range` for its bins (default:
  `regions["phi1_range"]`), so that the profile can extend over the window
  and show where the stream stops.
- Default Hess ranges: `DEFAULT_COLOR_RANGE = (-1.0, 1.5)`,
  `DEFAULT_MAG_RANGE = (17.0, 27.0)` (VIS − H, VIS), those of the CMDs of
  the known-stream notebook.
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
| S1 | 1–2 Inputs, frame, window | C1: `stream_frame_from_endpoints`, `endpoints_phi12_window`, `phi12_window_radec_box`, `great_circle_radec`, `window_outline_radec` | dataclass fields, `__post_init__` | — |
| S2 | 3 Catalog, MF map of the loaded box | (existing `generic_search_single_dm`) | `compute_sky_map` | (existing `plot_generic_search_residual_map`, + `ax` argument) |
| S3 | 4 Frame check, endpoints by hand | (existing `prepare_search`, `stream_frame_search_single_dm`); `excluded_healpix_mask` (Masks, Sect. 1) | `prepare`, `compute_zoom_map`, `set_endpoints` | `plot_candidate_frame_check`, `draw_stream_track_on_sky`, `draw_stream_track` |
| S4 | 5 MF maps over dm, in the frame | C3: `stream_track_phi12`, `stream_track_radec` | `scan_distance_moduli`, `track_phi12`, `track_radec` | `plot_candidate_map` (without regions) |
| S5 | 6 Width | C3: `phi2_density_profile`, `fit_stream_width` | `fit_width` | `plot_phi2_profile` |
| S6 | 7 Regions | C3: `stream_regions`, `on_stream_mask`, `off_stream_mask`, `phi1_density_profile`, `region_area_deg2` | `define_regions` | `draw_stream_regions`, `plot_phi1_profile`; regions in `plot_candidate_map` |
| S7 | 8 CMDs | C4: `hess_diagram`, `difference_hess` | `compute_hess` | `plot_hess`, `plot_on_off_difference_hess`, `plot_candidate_hess` |
| S8 | 9 Isochrone | C5: `isochrone_cmd_track` | `compute_isochrone` | `plot_isochrone`, `plot_match_filter_polygon`; isochrone in `plot_candidate_hess` |
| S9 | 10–11 Summary, all at once | — | `summary`, `run_all_steps` | `plot_candidate_overview` |

Sections C1–C7 below describe the functions; the table says when each is filled.

## 4. New functions

`[NEW]` = new. Plain dicts are described by their keys.

### C1. Frame from endpoints — `stream_frames.py`

```python
def stream_frame_from_endpoints(endpoint_1, endpoint_2):
    """Great-circle frame through two (ra, dec) points (deg), phi1 along the arc.
    gala GreatCircleICRSFrame.from_endpoints. Returns the frame."""

def endpoints_phi12_window(stream_frame, endpoint_1, endpoint_2, *,
                           phi2_halfwidth_deg=DEFAULT_PHI2_HALFWIDTH_DEG,
                           padding_phi1_deg=DEFAULT_PADDING_PHI1_DEG):
    """(phi1_min, phi1_max, phi2_min, phi2_max) around the arc between the endpoints.
    padding_phi1_deg on each side in phi1 (5.0, as galstreams_phi12_window)."""

def great_circle_radec(stream_frame, *, phi1_range=(-180.0, 180.0), n_points=721):
    """(ra, dec) of the great circle phi2 = 0 of a frame, for sky maps."""

def window_outline_radec(stream_frame, window, *, n_points_per_edge=50):
    """(ra, dec) of the outline of a phi1/phi2 window, for sky maps."""

def phi12_window_radec_box(stream_frame, window, *, margin_deg=0.5):
    """(ra_range, dec_range) of the box containing a phi1/phi2 window, for
    get_euclid_stars. Edges of the window sampled, converted with phi12_to_radec,
    min/max plus a margin. Handles the ra = 0/360 wrap (warn if crossed)."""
```

- gala's default origin is the midpoint (phi1 = 0), endpoint_1 at negative
  phi1 (Sect. 1, facts checked). Write it in the docstring.
- `DEFAULT_PHI2_HALFWIDTH_DEG = 3.0`: wide enough for the off bands
  (4.5σ for σ up to ~0.5 deg) plus a margin. No reference: a practical choice.
- Tests: both endpoints at phi2 = 0 (to 1e-8 deg); their phi1 symmetric
  around 0; window contains them; round trip with `phi12_to_radec`.

### C2. MF map of the candidate — no new function

`prepare_search(catalog, stream_frame=frame, window=window, footprint_mask=..., nest=..., isochrone=..., distance_moduli=...)`,
then `stream_frame_search_single_dm(search=search, distance_modulus=dm)` or
`stream_frame_search_dm_scan(search=search)`. Only notebook cells. If
something blocks a frame without a track, report it before changing
`prepare_search`.

### C3. Regions and width — `analysis/stream_regions.py` [NEW module]

```python
DEFAULT_ON_N_SIGMA = 1.4          # user choice
DEFAULT_OFF_N_SIGMA = (2.5, 4.5)  # user choice
DEFAULT_PHI2_BIN_DEG = 0.05
DEFAULT_SIGMA_GUESS_DEG = 0.2    # initial sigma of the width fit

def stream_regions(phi1_range, phi2_centre=0.0, sigma=DEFAULT_SIGMA_GUESS_DEG, *,
                   on_n_sigma=DEFAULT_ON_N_SIGMA, off_n_sigma=DEFAULT_OFF_N_SIGMA):
    """Plain dict {phi1_range, phi2_centre, sigma, on_n_sigma, off_n_sigma}.
    Edit it by hand in the notebook if needed."""

def on_stream_mask(phi1, phi2, regions):
    """True for |phi2 - phi2_centre| <= on_n_sigma*sigma and phi1 in phi1_range."""

def off_stream_mask(phi1, phi2, regions):
    """True for off_n_sigma[0]*sigma <= |phi2 - phi2_centre| <= off_n_sigma[1]*sigma, phi1 in range."""

def phi2_density_profile(phi1, phi2, phi1_range, stream_frame, excluded_healpix_mask, *,
                         phi2_range, bin_deg=DEFAULT_PHI2_BIN_DEG, step_deg=0.01):
    """Density of stars (e.g. MF-selected, stars/deg²) in phi2 bins, over phi1_range:
    counts / area of each bin outside excluded_healpix_mask (unmasked_area_deg2).
    Dict {phi2_centres, counts, area_deg2, density, error, bin_deg}."""

def fit_stream_width(profile, *, phi2_centre_guess=0.0, sigma_guess=DEFAULT_SIGMA_GUESS_DEG):
    """Gaussian + constant fitted to a phi2 density profile (scipy.optimize.curve_fit,
    Poisson errors sqrt(counts)/area). Dict {phi2_centre, sigma, amplitude, background, errors, model}."""

def phi1_density_profile(phi1, phi2, regions, stream_frame, excluded_healpix_mask, *,
                         bin_deg=0.5, step_deg=0.01):
    """On − off density (stars/deg²) along phi1, each phi1 bin of the on region and
    of the off bands divided by its own area outside the mask (unmasked_area_deg2),
    to see where the stream starts and stops. Dict {phi1_centres, on, off, difference, error}."""

def unmasked_area_deg2(stream_frame, phi1_edges, phi2_edges, excluded_healpix_mask, *,
                       step_deg=0.01):
    """Area (deg²) of each phi1/phi2 cell outside excluded_healpix_mask (RING).
    Fine grid of step_deg, cos(phi2) weights, -> (ra, dec) -> HEALPix pixel
    (nside from the mask size) -> kept if not excluded. Shape (n_phi1, n_phi2).
    Shared by region_area_deg2 and the two profiles."""

def region_area_deg2(regions, stream_frame, excluded_healpix_mask, *, step_deg=0.01):
    """Areas (deg²) of the on and off regions outside excluded_healpix_mask
    (unmasked_area_deg2 on the region edges). Dict {on, off}."""

def stream_track_phi12(regions, n_points=100):
    """(phi1, phi2) of the straight track: phi2 = phi2_centre over phi1_range."""

def stream_track_radec(regions, stream_frame, n_points=100):
    """Same track in (ra, dec), for sky maps (phi12_to_radec)."""
```

- `excluded_healpix_mask` = footprint mask (True outside) OR masked objects,
  RING, at `object_mask_nside(bin_size_deg)`, from
  `build_masks.excluded_healpix_mask` (Sect. 1, Masks). Built by the class,
  never in the notebook.
- The window must contain the off bands: check it in the notebook with an
  `assert`, not in the package.
- Tests (synthetic): uniform stars + Gaussian stream of known centre and σ →
  `fit_stream_width` recovers them within 3 errors; on and off masks are
  disjoint; `unmasked_area_deg2` / `region_area_deg2` with no mask matches the analytic area
  Δφ1 · (sin φ2,max − sin φ2,min) · (180/π)² within 1 %; with half the sky
  masked, about half the area; a uniform population with half the sky
  masked gives a flat `phi2_density_profile`.

### C4. Hess diagrams — `analysis/hess_diagrams.py` [NEW module]

```python
DEFAULT_COLOR_BIN = 0.04     # celestial hess.py defaults
DEFAULT_MAG_BIN = 1 / 6

def hess_diagram(stars, mask, area_deg2, *, match_filter=None, color_index=0,
                 color_bins=None, mag_bins=None):
    """2D histogram of colour (reference - color_bands[color_index]) and magnitude
    (reference band) of stars[mask], divided by area_deg2.
    Dict {density (stars/deg²/bin, shape (n_color, n_mag)), counts, error
    (sqrt(counts)/area), color_edges, mag_edges, area_deg2, n_stars,
    color_label, mag_label}."""

def difference_hess(on_hess, off_hess):
    """on - off densities, errors in quadrature. Same dict keys.
    Raises ValueError if the bin edges differ."""
```

- Columns from `match_filter_columns(match_filter)`, as the MF; `stars` is
  `search.stars`.
- Default bins: from the MF polygon extent, or fixed ranges as module
  constants. Same bins for on and off (the notebook passes them).
- Tests: on and off drawn from the same uniform population, areas 1 and 3 →
  difference consistent with 0 (|diff| / error ~ N(0, 1)); density of a
  single star in a 2 deg² region is 0.5 in one bin.
- Background-subtracted Hess diagrams as in Shipp et al. 2018, Sect. 5.

### C5. Isochrone — `catalogs/isochrone_parameters.py`

```python
def isochrone_cmd_track(distance_modulus, age, metallicity=None, *, feh=None,
                        match_filter=None, color_index=0):
    """Colour and magnitude of an isochrone in the bands of a match filter,
    at a distance modulus. Exactly one of metallicity (Z) or feh.
    Dict {color, mag, age, metallicity, feh, distance_modulus}."""
```

- Call `streamobs.match_filter._isochrone_locus(isochrone_model, age, Z,
  band_1, survey_1, band_2, survey_2)`, the function `build_match_filter`
  calls (model "Marigo2017", band_1 = reference band, so colour = reference −
  colour band, as the polygons). Absolute magnitude + distance_modulus. Private
  in streamobs: say it in the docstring. The track and the MF polygon then
  come from the same model.
- `feh` → Z with `unit_conversions.convert_FeH_to_z`. Check that it matches
  the Marigo et al. 2017 scale used elsewhere (module docstring of
  `isochrone_parameters`).
- Tests: the track at dm = 20 is the track at dm = 15 shifted by 5 mag in
  magnitude, same colour; `feh=-1.9` gives Z ≈ 0.0002 (within the grid
  step); the track lies inside the MF polygon at the same dm (most points
  of the main sequence turn-off region).

### C6. Plots — `plotting/plot_cmd.py` and `plotting/plot_candidate_investigation.py` [NEW module]

In `plot_cmd.py`:

```python
def plot_hess(hess, *, ax=None, difference=False, cmap=None, vlim=None,
              smoothing_bins=0.0, colorbar=True, title=None):
    """One Hess diagram (on, off or on - off). Magnitude axis inverted.
    difference=True: diverging cmap (RdBu_r), symmetric limits ±vlim
    (default: 98th percentile of |density|). Colour bar 'stars / deg² / bin'.
    Returns (fig, ax)."""

def plot_on_off_difference_hess(on_hess, off_hess, diff_hess, *, axes=None, **kwargs):
    """Three panels side by side, same axes limits; calls plot_hess three times."""

def plot_isochrone(ax, isochrone_track, *, color="k", lw=1.0, label=None):
    """Draws isochrone_cmd_track output on a CMD axes. Default label:
    age, Z ([Fe/H]), dm."""

def plot_match_filter_polygon(ax, distance_modulus, isochrone=None, match_filter=None,
                              *, color_index=0, **line_kwargs):
    """Outline of the MF polygon on an existing CMD axes
    (factored out of plot_match_filter, which then calls it)."""
```

In `plotting/plot_candidate_investigation.py`:

```python
def draw_stream_track(ax, phi1, phi2, *, label=None, **line_kwargs):
    """Track on a phi1/phi2 map (e.g. from plot_stream_frame_search_single_dm).
    Thin dashed line, offset-able not to hide the stream."""

def draw_stream_track_on_sky(ax, ra, dec, *, label=None, **line_kwargs):
    """Track on a skyproj sky map (plot_generic_search_result,
    plot_2D_density_from_hp), split at the map edge as in plot_generic_search."""

def draw_stream_regions(ax, regions, *, on_color=..., off_color=...):
    """Outlines of the on region and the two off bands on a phi1/phi2 map."""

def plot_phi2_profile(profile, fit=None, *, ax=None):
    """Counts vs phi2, with the fitted Gaussian + background and ±σ lines."""

def plot_phi1_profile(profile, regions=None, *, ax=None):
    """On - off density vs phi1; phi1_range shaded."""
```

- `draw_stream_track` and `draw_stream_track_on_sky` take arrays, so they
  also draw galstreams tracks or the endpoints arc. Make
  `plot_stream_frame_search_single_dm(track_phi12=...)` call
  `draw_stream_track` (one style for all tracks).
- Tests: each function runs on synthetic inputs and returns the axes;
  `plot_hess(difference=True)` has symmetric colour limits; the number of
  lines added by `draw_stream_regions` is 3 outlines.
- Figure style: `plot_style`, colour-blind friendly colours
  (`CATEGORICAL_COLORS`).

### C7. The class — `analysis/candidate_investigation.py` [NEW module]

Stub written in Phase B; each method filled in the step that needs it (table of Phase C).

```python
@dataclass
class CandidateInvestigation:
    """All inputs and results of one candidate investigation.
    Mutable: change a field and re-run the later steps."""
    # inputs
    name: str
    endpoint_1: tuple[float, float]          # (ra, dec) deg
    endpoint_2: tuple[float, float]
    age: float = 13.0
    metallicity: float | None = 0.0002       # Z; or feh
    feh: float | None = None
    distance_modulus: float | None = None    # chosen after the dm scan
    distance_moduli: Sequence[float] | None = None
    match_filter: dict | None = None
    phi2_halfwidth_deg: float = DEFAULT_PHI2_HALFWIDTH_DEG
    on_n_sigma: float = DEFAULT_ON_N_SIGMA
    off_n_sigma: tuple[float, float] = DEFAULT_OFF_N_SIGMA
    prepare_kwargs: dict = field(default_factory=dict)   # to prepare_search
    # results, None until computed
    stream_frame: Any = None
    window: tuple | None = None
    radec_box: tuple | None = None
    search: PreparedSearch | None = None
    excluded_healpix_mask: NDArray | None = None   # RING, build_masks.excluded_healpix_mask
    catalog: Any = None                      # reference to the stars (no copy), kept to re-prepare
    sky_map: dict | None = None              # generic_search_single_dm result on the loaded box, frame independent
    zoom_map: dict | None = None             # stream_frame_search_single_dm at sky_map's dm
    maps: dict | None = None                 # {dm: result of stream_frame_search_single_dm}
    phi2_profile: dict | None = None        # phi2_density_profile result
    width_fit: dict | None = None
    regions: dict | None = None
    areas: dict | None = None
    phi1_profile: dict | None = None        # phi1_density_profile result
    on_hess: dict | None = None
    off_hess: dict | None = None
    diff_hess: dict | None = None
    isochrone_track: dict | None = None

    def __post_init__(self): ...            # frame, window, radec_box (C1)
    def compute_sky_map(self, catalog, distance_modulus, **generic_kwargs): ...  # MF map of the loaded box, once
    def prepare(self, catalog=None, *, footprint_mask=None, nest=False): ...  # prepare_search + excluded mask; stores catalog
    def set_endpoints(self, endpoint_1, endpoint_2, **window_kwargs): ...  # new frame, window, box; re-prepare
                                            # from the stored catalog; clears every later result; returns self
    def compute_zoom_map(self, distance_modulus=None): ...              # one map in the frame (default: sky_map dm)
    def scan_distance_moduli(self, distance_moduli=None): ...           # fills maps
    def fit_width(self, phi1_range=None): ...  # MF-selected stars at distance_modulus -> profile, fit
    def define_regions(self, phi1_range=None, phi2_centre=None, sigma=None): ...  # regions, areas, phi1_profile
    def compute_hess(self, *, color_bins=None, mag_bins=None): ...      # on, off, diff
    def compute_isochrone(self, *, age=None, metallicity=None, feh=None, distance_modulus=None): ...
    def run_all_steps(self, catalog, **kwargs): ...   # every step in order
    def track_phi12(self): ...              # stream_track_phi12(regions), or the endpoints arc before regions
    def track_radec(self): ...
    def summary(self): ...                  # dict of the key numbers
```

- Each method: check that the previous step was done (clear
  `ValueError` naming the missing step), call one or two functions of
  C1–C6, store the result, return `self` (so calls can be chained).
- Defaults when an argument is None: `phi1_range` → the endpoints' phi1;
  `phi2_centre`, `sigma` → `width_fit`. `distance_modulus` has no default:
  the user picks it by eye after the scan; `fit_width` raises if it is
  missing.
- `prepare`: footprint, when not given, `sky_map["footprint_mask"]` (with
  its `nest`), else from the stars (`survey_footprint_healpix_mask(ra, dec,
  512, nest=False)`); `excluded_healpix_mask` from
  `build_masks.excluded_healpix_mask(footprint, search.object_healpix_mask,
  nest=...)` (Sect. 1, Masks).
- `set_endpoints` must be fast: the map of the loaded box is not recomputed
  (it does not depend on the frame); only `prepare_search` on the stored
  catalog (an (ra, dec) box larger than the window) and one zoom map. If the
  new window leaves the loaded box, raise a clear error saying to reload a
  larger box.
- `plot_generic_search_residual_map` makes its own figure. Add an optional
  `ax` argument (skyproj axes made from it) so that the frame check is one
  figure with two panels. Small change, keep the old behaviour by default.
- Plots accept the class (thin wrappers in `plotting/plot_candidate_investigation.py`):

```python
def plot_candidate_frame_check(cand, *, sky_kwargs=None, zoom_kwargs=None):
    """The endpoints check. Left: cand.sky_map (residual) with the great
    circle, the window outline and the two endpoints
    (plot_generic_search_residual_map, tracks=...). Right: cand.zoom_map in
    phi1/phi2 with the endpoints marked and the phi2 = 0 line. Same dm, said in
    the title. Returns (fig, axes)."""

def plot_candidate_map(cand, distance_modulus=None, *, ax=None, show_regions=True):
    """plot_stream_frame_search_single_dm + draw_stream_track + draw_stream_regions."""
def plot_candidate_hess(cand, which="difference", *, ax=None, isochrone=True, match_filter=False):
    """One Hess ('on', 'off' or 'difference') with the isochrone on top."""
def plot_candidate_overview(cand):
    """Map with regions, phi2 profile, phi1 profile, difference Hess + isochrone."""
```

- Tests: a synthetic catalog (uniform + Gaussian stream along the endpoints
  arc) through `run_all_steps`; σ and phi2_centre recovered; calling
  `compute_hess` before `define_regions` raises; `summary` keys.

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
6. **Width.** MF-selected stars → `phi2_density_profile` → `fit_stream_width` →
   `plot_phi2_profile`.
7. **Regions.** `stream_regions` (phi1_range by eye, then from
   `phi1_density_profile`); `draw_stream_regions` on the MF map; `region_area_deg2`.
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
