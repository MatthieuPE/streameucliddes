"""Stream candidates: their table, their phi1/phi2 windows and the stars inside.

Copied from the package streameucliddes, so the windows are those of the
candidate investigation. Needs numpy, pandas, astropy and gala.

The table has one row per candidate: ``id``, two points along the stream
(``endpoint_1_ra``, ``endpoint_1_dec``, ``endpoint_2_ra``, ``endpoint_2_dec``,
deg) and the best distance modulus (``best_dm``). The frame of a candidate
is the great circle through its endpoints (phi1 along the stream, both
endpoints at phi2 = 0); its window ``(phi1_min, phi1_max, phi2_min,
phi2_max)`` goes 5 deg beyond the endpoints in phi1 and at least +-5 deg in
phi2.

Example, with ``stars`` a DataFrame with ra, dec columns (deg)::

    candidate = get_candidate(load_candidates("candidates_info.csv"), 3)
    frame, window, radec_box = candidate_window(candidate)
    mask = stars_in_candidate_window(stars, frame, window)  # True for the stars inside
    phi1, phi2 = radec_to_phi12(stars["ra"][mask], stars["dec"][mask], frame)
"""

from __future__ import annotations

import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import numpy as np
import pandas as pd

PADDING_PHI1_DEG = 5.0  # window: beyond the endpoints in phi1
PADDING_PHI2_DEG = 2.0  # window: beyond the endpoints in phi2...
MIN_PHI2_HALF_WIDTH_DEG = 5.0  # ...and at least this on each side


# ===================================================
# Table
# ===================================================


def load_candidates(path):
    """The table of candidates (CSV), ``id`` as integers."""
    candidates = pd.read_csv(path)
    candidates["id"] = candidates["id"].astype(int)
    return candidates


def get_candidate(candidates, candidate_id):
    """The row of a candidate id; ValueError if absent."""
    rows = candidates[candidates["id"] == int(candidate_id)]
    if rows.empty:
        raise ValueError(f"No candidate with id {candidate_id}; the ids are {candidates['id'].tolist()}.")
    return rows.iloc[0]


# ===================================================
# Frame: (ra, dec) <-> (phi1, phi2)
# ===================================================


def stream_frame_from_endpoints(endpoint_1, endpoint_2):
    """gala great-circle frame through two (ra, dec) points (deg): phi1 = 0 at their midpoint, both at phi2 = 0."""
    point_1 = coord.SkyCoord(ra=endpoint_1[0] * u.deg, dec=endpoint_1[1] * u.deg)
    point_2 = coord.SkyCoord(ra=endpoint_2[0] * u.deg, dec=endpoint_2[1] * u.deg)
    return gc.GreatCircleICRSFrame.from_endpoints(point_1, point_2)


def radec_to_phi12(ra, dec, stream_frame):
    """(ra, dec) to (phi1, phi2) in a stream frame (deg); phi1 in [-180, 180)."""
    ra, dec = np.broadcast_arrays(np.asarray(ra, dtype=float), np.asarray(dec, dtype=float))
    stars = coord.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    in_frame = stars.transform_to(stream_frame).represent_as(coord.UnitSphericalRepresentation)
    return in_frame.lon.wrap_at(180 * u.deg).deg, in_frame.lat.deg


def phi12_to_radec(phi1, phi2, stream_frame):
    """(phi1, phi2) in a stream frame to (ra, dec) (deg); ra in [0, 360)."""
    phi1, phi2 = np.broadcast_arrays(phi1, phi2)
    positions = coord.UnitSphericalRepresentation(lon=phi1 * u.deg, lat=phi2 * u.deg)
    icrs = stream_frame.realize_frame(positions).transform_to(coord.ICRS())
    return icrs.ra.deg, icrs.dec.deg


# ===================================================
# Windows in phi1/phi2 and their (ra, dec) boxes
# ===================================================


def phi12_window_around(phi1, phi2, *, padding_phi1_deg=PADDING_PHI1_DEG, padding_phi2_deg=PADDING_PHI2_DEG,
                        min_phi2_half_width_deg=MIN_PHI2_HALF_WIDTH_DEG):
    """Window ``(phi1_min, phi1_max, phi2_min, phi2_max)`` around points: their extent plus the paddings,
    at least ``min_phi2_half_width_deg`` on each side of their centre in phi2."""
    phi1, phi2 = np.asarray(phi1, dtype=float), np.asarray(phi2, dtype=float)
    phi2_centre = (phi2.min() + phi2.max()) / 2
    phi2_half_width = max((phi2.max() - phi2.min()) / 2 + padding_phi2_deg, min_phi2_half_width_deg)
    return (float(phi1.min() - padding_phi1_deg), float(phi1.max() + padding_phi1_deg),
            float(phi2_centre - phi2_half_width), float(phi2_centre + phi2_half_width))


def in_phi12_window(phi1, phi2, window):
    """Mask of the points inside a phi1/phi2 window, edges included."""
    phi1_min, phi1_max, phi2_min, phi2_max = window
    phi1, phi2 = np.asarray(phi1), np.asarray(phi2)
    return (phi1 >= phi1_min) & (phi1 <= phi1_max) & (phi2 >= phi2_min) & (phi2 <= phi2_max)


def phi12_window_outline(window, n_points_per_edge=50):
    """(phi1, phi2) of the closed outline of a window, sampled along its edges (curved in ra/dec)."""
    phi1_min, phi1_max, phi2_min, phi2_max = window
    step = np.linspace(0.0, 1.0, n_points_per_edge, endpoint=False)
    phi1_span, phi2_span = phi1_max - phi1_min, phi2_max - phi2_min
    phi1 = np.concatenate([phi1_min + step * phi1_span, np.full(n_points_per_edge, phi1_max),
                           phi1_max - step * phi1_span, np.full(n_points_per_edge, phi1_min), [phi1_min]])
    phi2 = np.concatenate([np.full(n_points_per_edge, phi2_min), phi2_min + step * phi2_span,
                           np.full(n_points_per_edge, phi2_max), phi2_max - step * phi2_span, [phi2_min]])
    return phi1, phi2


def phi12_window_radec_box(stream_frame, window, margin_deg=0.5):
    """``((ra_min, ra_max), (dec_min, dec_max))`` containing a window, plus a margin (deg).

    ``ra_min > ra_max`` across ra = 0/360; every ra if the window contains a pole.
    """
    ra, dec = phi12_to_radec(*phi12_window_outline(window), stream_frame)
    dec_range = (max(float(dec.min()) - margin_deg, -90.0), min(float(dec.max()) + margin_deg, 90.0))
    pole_phi1, pole_phi2 = radec_to_phi12([0.0, 0.0], [90.0, -90.0], stream_frame)
    north_inside, south_inside = in_phi12_window(pole_phi1, pole_phi2, window)
    if north_inside or south_inside:
        return (0.0, 360.0), (-90.0 if south_inside else dec_range[0], 90.0 if north_inside else dec_range[1])
    centre_ra, _ = phi12_to_radec((window[0] + window[1]) / 2, (window[2] + window[3]) / 2, stream_frame)
    relative_ra = (ra - centre_ra + 180.0) % 360.0 - 180.0  # around the window centre: no jump at 0/360
    if relative_ra.max() - relative_ra.min() + 2 * margin_deg >= 360.0:
        return (0.0, 360.0), dec_range
    return ((float((centre_ra + relative_ra.min() - margin_deg) % 360.0),
             float((centre_ra + relative_ra.max() + margin_deg) % 360.0)), dec_range)


def in_radec_box(ra, dec, ra_range, dec_range):
    """Mask of the points inside an (ra, dec) box (deg); ``ra_min > ra_max``: across ra = 0/360."""
    ra, dec = np.asarray(ra, dtype=float) % 360.0, np.asarray(dec, dtype=float)
    ra_min, ra_max = ra_range
    if ra_max - ra_min >= 360.0:
        in_ra = np.ones(ra.shape, dtype=bool)
    elif ra_min <= ra_max:
        in_ra = (ra >= ra_min) & (ra <= ra_max)
    else:
        in_ra = (ra >= ra_min) | (ra <= ra_max)
    return in_ra & (dec >= dec_range[0]) & (dec <= dec_range[1])


# ===================================================
# A candidate: frame, window, box; the stars inside
# ===================================================


def candidate_window(candidate, **window_kwargs):
    """Frame, phi1/phi2 window and (ra, dec) box of a candidate (a row of the table).

    ``window_kwargs``: those of ``phi12_window_around``, e.g. ``min_phi2_half_width_deg=8`` for a wider window.
    """
    endpoint_1 = (float(candidate["endpoint_1_ra"]), float(candidate["endpoint_1_dec"]))
    endpoint_2 = (float(candidate["endpoint_2_ra"]), float(candidate["endpoint_2_dec"]))
    frame = stream_frame_from_endpoints(endpoint_1, endpoint_2)
    window = phi12_window_around(*radec_to_phi12([endpoint_1[0], endpoint_2[0]], [endpoint_1[1], endpoint_2[1]], frame),
                                 **window_kwargs)
    return frame, window, phi12_window_radec_box(frame, window)


def stars_in_candidate_window(stars, frame, window, ra_column="ra", dec_column="dec"):
    """Mask of the stars of a DataFrame inside a window: the (ra, dec) box first, then phi1/phi2 for those only."""
    ra, dec = stars[ra_column].to_numpy(), stars[dec_column].to_numpy()
    mask = in_radec_box(ra, dec, *phi12_window_radec_box(frame, window))
    in_box = np.flatnonzero(mask)
    mask[in_box] = in_phi12_window(*radec_to_phi12(ra[in_box], dec[in_box], frame), window)
    return mask
