# Target structure of `streameucliddes`

Goal: one place per kind of work, no duplicates, explicit names.
A module name says what it holds. A function name says what it returns
or does, and reuses the words of its module.

## 1. Problems in the current code

Checked on `main` (2026-11-06).

1. **Duplicate.** `get_healpy_data_from_table` is in `utils.py:106` and
   `analysis/data_manipulation.py:330`.
2. **Unclear module names.** `analysis/data_manipulation.py` holds the SNR
   study, the match-filter polygon, the background catalog and HEALPix
   maps. `utils.py` mixes an ugali patch, unit conversions and HEALPix.
3. **Analysis imports plotting.** `analysis/pipeline.py` imports
   `plotting.plotting` (frames and gif in `scan_find_stream`).
4. **Computation in plotting.** `plotting.get_hp_map` builds a map.
5. **Missing functions.** `data_manipulation.correct_dust_map`,
   `plot_overlapped_objects` and `plot_single_stream_track` are called
   but not defined in the package.
6. **Hard-coded frame.** `GC_FRAME_DEFAULT` in `data_manipulation.py:30`.
7. **One 752-line plotting file** with five unrelated topics.

## 2. Layering rule

Lower layers never import higher layers.

```
core modules → catalogs/ → analysis/ → plotting/ → workflows/
```

- **Core modules**: unit conversions, stream frames, isochrone parameters.
- **`catalogs/`**: read data, apply cuts, dust, footprint, galstreams.
- **`analysis/`**: compute only. Returns arrays and dicts. Never draws, never writes files.
- **`plotting/`**: draw only. Never filters or fits.
- **`workflows/`**: loops over dm or streams, configs, saving files, videos. Scripts and notebooks call these.
- `injection/` stays as it is, beside `analysis/`.

## 3. Target tree

Legend: `(old: …)` = moved or renamed from existing code (phase 1).
`[NEW]` = new stub (phase 2). Names in phase 2 are proposals; the user
reviews them.

```
src/streameucliddes/
  __init__.py
  _ugali_patches.py              cache_isochrone_grids            (old: utils.py)
  unit_conversions.py            mag_to_flux                      (old: utils.magToFlux)
                                 flux_to_mag                      (old: utils.fluxToMag)
                                 convert_DM_to_kpc, convert_kpc_to_DM,
                                 convert_FeH_to_z                 (old: utils.py)
  stream_frames.py               [NEW] radec_to_phi12, phi12_to_radec (gala frames)
                                 [NEW] phi12_bin_centres

  catalogs/
    euclid_des_catalog.py        open_eucliddes_data, add_magnitudes,
                                 get_eucliddes_stars, available_columns
                                 (old: analysis/euclid_des_preparation.py)
    catalog_cuts.py              get_cut_mask, cut_flow    (old: analysis/utils.py)
    dust_correction.py           correct_dust_map          (old: missing, locate it)
    survey_footprint.py          [NEW] survey_footprint_healpix_mask
                                 [NEW] survey_footprint_phi12_mask
    galstreams_catalog.py        [NEW] load_galstreams
                                 [NEW] galstreams_track_length_in_footprint
                                 [NEW] select_galstreams_in_footprint
                                 [NEW] galstreams_default_tracks (one track per stream)
                                 [NEW] galstreams_phi12_window
                                 [NEW] galstreams_dm_along_track
    isochrone_parameters.py      [NEW] default_isochrone_grid
                                 [NEW] known_stream_isochrone_parameters
                                 [NEW] load_stream_population_table
    stream_populations.csv       [NEW] age, [Fe/H], Z, references

  analysis/
    cmd_match_filter.py          build_match_filter_polygon
                                 (old: data_manipulation._match_filter_polygon)
                                 [NEW] select_stars_in_match_filter, match_filter_polygons
                                 (selection only; preparation in catalog_preparation)
    catalog_preparation.py       [NEW] prepare_match_filter_catalog, check_columns,
                                 default_error_cuts; resolve_mag_columns and
                                 ensure_dust_corrected (moved from generic_stream_search)
    star_density_maps.py         star_density_healpix
                                 (old: get_healpy_data_from_table, one copy,
                                  + compute part of plotting.get_hp_map)
                                 [NEW] star_density_phi12
    background_polynomial_fit.py fit_background_model, polyfit2d,
                                 fit_background_model_polyfit2d, process_data,
                                 process_data_polyfit2d, polynomial_formula
                                 (old: analysis/fit.py)
    generic_stream_search.py     generic_search_single_dm  (old: pipeline.find_stream)
                                 generic_search_dm_scan    (old: compute part of
                                                            pipeline.scan_find_stream)
                                 [NEW] prepare_generic_search, PreparedGenericSearch,
                                 generic_search_cmd_stars
                                 (generic-search branch: API mirrors known_stream_search;
                                  shared match filter, no copy of the catalog, chunks)
    known_stream_search.py       [NEW] known_stream_search_single_dm
                                 [NEW] known_stream_search_dm_scan
                                 [NEW] known_stream_search_dm_along_track
    injected_stream_snr.py       get_stream_snr, scan_snr_grid + helpers
                                 (old: data_manipulation.py)

  injection/                     unchanged

  plotting/
    plot_style.py                unchanged
    plot_cmd.py                  plot_CMD                       (old: plotting.py)
                                 [NEW] plot_match_filter        (polygons on the CMD; gif:
                                       frames + make_gif_from_plots)
    plot_sky_density.py          plot_2D_density, plot_2D_density_from_hp,
                                 plot_overlapped_objects, plot_single_stream_track
                                 (old: plotting.py; last two missing, locate them)
    plot_generic_search.py       plot_generic_search_result
                                 (old: plotting.plot_results_find_stream; generic-search
                                  branch: CMD selection, map, background, corrected map)
                                 [NEW] plot_generic_search_residual_map (corrected map
                                       alone, known stream tracks faint and dashed)
    plot_known_stream_search.py  [NEW] plot_known_stream_search_single_dm
                                 [NEW] plot_galstreams_selection
    plot_injected_stream_snr.py  plot_snr_contours, plot_snr_maps + helpers
                                 (old: plotting.py)
    make_videos.py               make_gif                       (old: plotting.make_gif)
                                 [NEW] make_gif_from_plots      (copy of celestial
                                       visualization.make_gif_from_plots; used for
                                       all new gifs, user choice; no mp4,
                                       as celestial_currents)

  workflows/
    run_config.py                [NEW] KNOWN_STREAM_SEARCH_CONFIG, GENERIC_SEARCH_CONFIG (dicts
                                       of the run arguments, to copy and update),
                                       RunConfig (YAML overrides)
    run_generic_search.py        run_generic_search
                                 (old: loop, plots and gif of scan_find_stream; gifs of the
                                  four panels and of the corrected map with known tracks)
                                 [NEW] save_generic_search_healpix_maps
    run_known_stream_search.py   [NEW] run_known_stream_search
                                 [NEW] run_all_known_stream_searches
                                 [NEW] save_known_stream_search_maps

scripts/                         [NEW] run_generic_search.py, run_known_stream_search.py
configs/                         [NEW] generic_search.yaml, known_stream_search.yaml
notebooks/                       [NEW] known_stream_search_example.ipynb (usage example)
                                 [NEW] generic_search_example.ipynb (fake data first,
                                       then the Euclid x DES stars)
```

Naming rules used:
- `generic_search_*` and `known_stream_search_*` mirror each other: `_single_dm` gives one map; `_dm_scan` goes over a dm grid one dm at a time (it yields `(dm, result)`), so that the caller can drop each map before the next one is computed. The workflows save all the maps of a run in one file (`save_*_maps`). Known-stream maps: one set per isochrone.
- `star_density_*` = smoothed star counts, `*_healpix` or `*_phi12` says the grid.
- `galstreams_*` = reads galstreams. `survey_footprint_*` = footprint.
- `plot_<thing>` in `plot_<module>.py`; `run_<thing>` in `run_<thing>.py`.
- Old camelCase (`magToFlux`) → snake_case. Keep `plot_CMD` and `convert_DM_to_kpc` (acronyms).

## 4. Old → new map (phase 1 only)

| Old | New |
|---|---|
| `utils.cache_isochrone_grids` | `_ugali_patches.cache_isochrone_grids` |
| `utils.magToFlux`, `fluxToMag` | `unit_conversions.mag_to_flux`, `flux_to_mag` |
| `utils.convert_*` | `unit_conversions.convert_*` |
| `utils.get_healpy_data_from_table` | `analysis.star_density_maps.star_density_healpix` |
| `analysis.data_manipulation.get_healpy_data_from_table` | same (delete the duplicate; keep the more complete one) |
| `analysis.euclid_des_preparation.*` | `catalogs.euclid_des_catalog.*` |
| `analysis.utils.*` | `catalogs.catalog_cuts.*` |
| `analysis.data_manipulation._match_filter_polygon` | `analysis.cmd_match_filter.build_match_filter_polygon` |
| `analysis.data_manipulation` (SNR part) | `analysis.injected_stream_snr` |
| `analysis.fit.*` | `analysis.background_polynomial_fit.*` |
| `analysis.pipeline.find_stream` | `analysis.generic_stream_search.generic_search_single_dm` |
| `analysis.pipeline.scan_find_stream` | compute: `analysis.generic_stream_search.generic_search_dm_scan`; loop/plots/gif: `workflows.run_generic_search.run_generic_search` |
| `analysis.pipeline.SimpleLonLatProj` | `analysis.background_polynomial_fit.SimpleLonLatProj` |
| `plotting.plotting.plot_CMD` | `plotting.plot_cmd.plot_CMD` |
| `plotting.plotting.plot_2D_density*` | `plotting.plot_sky_density.*` |
| `plotting.plotting.get_hp_map` | compute → `star_density_healpix`; drawing → `plot_sky_density` |
| `plotting.plotting.plot_results_find_stream` | `plotting.plot_generic_search.plot_generic_search_result` |
| `plotting.plotting.plot_snr_*` + helpers | `plotting.plot_injected_stream_snr.*` |
| `plotting.plotting.make_gif` | `plotting.make_videos.make_gif` |
| `analysis.data_manipulation.GC_FRAME_DEFAULT` | `analysis.injected_stream_snr` (kept as is; only used there) |

Name `catalogs/`, not `data/`, to avoid a clash with the repo-level
`data/` folder used by `DEFAULT_DATA_PATH`.
