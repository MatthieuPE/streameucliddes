# Module for stream injection using streamobs

from streamobs.observed import StreamInjector

from .generate import generate_stream_catalog


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
        The catalog of stars after injection.
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
    bands={"des_yr6": ["g", "r", "i"], "euclid_q1": ["VIS", "Y", "J"]},
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

    metadata = {"gc_frame": injector._last_gc_frame, "seed": seed}

    return stream, metadata
