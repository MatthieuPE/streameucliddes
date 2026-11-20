# Module for stream injection using streamobs

from streamobs.observed import StreamInjector

from streameucliddes.catalogs.euclid_catalog import VIS_TEMPLFIT_PARTNER_BANDS

from .generate import generate_stream_catalog

# Warning of the match filters with VIS template fits on injected stars (see
# add_vis_templfit_magnitudes)
INJECTED_VIS_TEMPLFIT_WARNING = (
    "Injected stars have one VIS magnitude, simulated by streamobs: their VIS_TO_<band> "
    "template-fitting magnitudes are copies of it, so a match filter with "
    "vis_photometry='templfit' sees the same VIS as with 'psf' on them (not on the data)."
)


def inject_stream(
    stream_catalog=None, stream_config=None, seed=None, rng=None, **kwargs
):
    """
    Inject a stream into a survey using the streamobs package.

    Parameters
    ----------
    stream_catalog : np.ndarray
        A mock catalog of stars for the stream.
    stream_config : dict
        The stream configuration (the ``stream`` section consumed by
        :class:`~streamobs.model.StreamModel`). Used to generate the catalog
        when ``stream_catalog`` is None, and by the injector to sample the
        missing columns (distances, true magnitudes).
    seed : int, optional
        Random seed for reproducibility.
    rng : np.random.Generator, optional
        Random number generator instance.
    survey : str, Survey, dict, or list
            One survey or several. Every survey is namespaced by its own
            :attr:`~streamobs.surveys.Survey.namespace` (``{name}_{release}``,
            or just ``{name}`` with no release). Accepted forms:

            - a survey-name string (e.g. ``'lsst'``), a
              ``{"survey": ..., "release": ...}`` spec dict, or a pre-loaded
              :class:`~streamobs.surveys.Survey` — a single survey;
            - a list/tuple of such specs;
            - a ``{key: spec}`` dict — the keys are containers only and are
              **ignored**; the namespace is re-derived from each loaded survey.
    Returns
    -------
    injected_catalog : np.ndarray
        The catalog of stars after injection, with VIS template-fitting magnitudes
        that are copies of VIS (:func:`add_vis_templfit_magnitudes`).
    """

    if stream_catalog is None:
        if stream_config is None:
            raise ValueError("Either stream_catalog or stream_config must be provided.")
        else:
            # kwargs are injection options (survey, bands, ...), not generation ones
            stream_catalog = generate_stream_catalog(
                stream_config=stream_config, seed=seed, rng=rng
            )

    injected_catalog = inject_stream_catalog(
        stream_catalog, stream_config=stream_config, seed=seed, rng=rng, **kwargs
    )

    return injected_catalog


def inject_stream_catalog(
    stream_catalog,
    stream_config=None,
    seed=None,
    gc_frame=None,
    survey=[{"survey": "euclid", "release": "q1"},
        {"survey": "des", "release": "yr6"},
    ],
    bands={"des_yr6": ["g", "r", "i"], "euclid_q1": ["VIS", "Y", "J", "H"]},
    **kwargs,
):

    injector = StreamInjector(survey=survey)
    stream = injector.inject(
        stream_catalog,
        stream_config=stream_config,
        gc_frame=gc_frame,
        seed=seed,
        bands=bands,
        **kwargs,
    )

    add_vis_templfit_magnitudes(stream)
    metadata = {"gc_frame": injector._last_gc_frame, "seed": seed}

    return stream, metadata


def add_vis_templfit_magnitudes(catalog):
    """
    Add VIS template-fitting magnitudes to injected stars, copies of their VIS, in place.

    streamobs simulates one VIS magnitude per star; the data have also the VIS
    template-fitting magnitudes ``<namespace>_VIS_TO_<band>_obs`` / ``_err``
    (``catalogs.euclid_catalog.vis_band``), read by the match filters with
    ``"vis_photometry": "templfit"``. They are set here to the VIS magnitude and
    error of each Euclid namespace, for <band> in ``VIS_TEMPLFIT_PARTNER_BANDS``:
    the difference between the two photometries (about 0.07 mag in DR1) is not
    simulated (see ``INJECTED_VIS_TEMPLFIT_WARNING``).

    Parameters
    ----------
    catalog : pandas.DataFrame
        Injected stars, with ``<namespace>_VIS_obs`` / ``_err`` columns.

    Returns
    -------
    pandas.DataFrame
        ``catalog``, with the new columns.
    """
    namespaces = [c.removesuffix("_VIS_obs") for c in catalog.columns
                  if c.startswith("euclid") and c.endswith("_VIS_obs")]
    for namespace in namespaces:
        for band in VIS_TEMPLFIT_PARTNER_BANDS:
            for kind in ("obs", "err"):
                if f"{namespace}_VIS_{kind}" in catalog.columns:
                    catalog[f"{namespace}_VIS_TO_{band}_{kind}"] = catalog[f"{namespace}_VIS_{kind}"]
    return catalog
