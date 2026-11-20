#!/usr/bin/env python
"""Compare the Euclid GAL_EBV column of the DR1 catalog with the Planck 2013 and SFD98 maps.

The Euclid documentation describes GAL_EBV only as "E(B-V) ... according to the
reference Planck map". This script samples sources in every partition of the local
DR1 HATS catalog and compares GAL_EBV with:

- the E(B-V) map of Planck Collaboration: Abergel et al. 2014 (A&A 571, A11), file
  HFI_CompMap_ThermalDustModel_2048_R1.20.fits (field EBV, from the dust radiance,
  for extragalactic studies), the map used by Gatto et al. (Euclid DR1 dwarf
  galaxies, Sect. 2), read with dustmaps' PlanckQuery (Green 2018, JOSS 3, 695);
- the E(B-V) from the 353 GHz optical depth of the same file (PlanckQuery
  component "tau"), to show which Planck field GAL_EBV follows;
- the EBV_SFD98 column of the catalog (Schlegel, Finkbeiner & Davis 1998).

It prints the statistics and saves the sample (parquet) and a figure in
``data/dust_comparison``.

The Planck file (1.6 GB) comes from the Planck Legacy Archive, as in
``dustmaps.planck.fetch``::

    curl -L -o data/dustmaps/planck/HFI_CompMap_ThermalDustModel_2048_R1.20.fits \\
        "https://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_CompMap_ThermalDustModel_2048_R1.20.fits"

(md5 8d804f4e64e709f476a63f0dfed1fd11).

Example::

    python scripts/compare_dust_maps.py --per-partition 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm

from streameucliddes.catalogs.euclid_catalog import DATA_DIR
from streameucliddes.plotting import plot_style

CATALOG_DIR = DATA_DIR / "euclid_dr1" / "euclid_dr1_mer"
PLANCK_MAP = DATA_DIR / "dustmaps" / "planck" / "HFI_CompMap_ThermalDustModel_2048_R1.20.fits"
OUTPUT_DIR = DATA_DIR / "dust_comparison"
COLUMNS = ["RIGHT_ASCENSION", "DECLINATION", "GAL_EBV", "EBV_SFD98"]
FIGURE_WIDTH = 7.0  # in, full page
SAME_VALUE_TOLERANCE = 1e-5  # mag: GAL_EBV is float32 in the catalog


def sample_catalog(catalog_dir, per_partition, seed=0):
    """``per_partition`` random sources (all if fewer) of each partition, :data:`COLUMNS` only."""
    files = sorted(Path(catalog_dir).glob("dataset/Norder=*/Dir=*/Npix=*.parquet"))
    rng = np.random.default_rng(seed)
    parts = []
    t0 = time.time()
    for k, path in enumerate(files, 1):
        table = pq.read_table(path, columns=COLUMNS)
        rows = rng.choice(len(table), min(per_partition, len(table)), replace=False)
        parts.append(table.take(np.sort(rows)).to_pandas())
        if k % 500 == 0 or k == len(files):
            print(f"read {k}/{len(files)} partitions ({time.time() - t0:,.0f} s)", flush=True)
    return pd.concat(parts, ignore_index=True)


def add_planck_ebv(sample, planck_map=PLANCK_MAP):
    """Add the Planck 2013 E(B-V) (``EBV_PLANCK13``) and tau-based E(B-V) (``EBV_PLANCK13_TAU``)."""
    from dustmaps.planck import PlanckQuery

    coords = SkyCoord(sample["RIGHT_ASCENSION"].to_numpy() * u.deg,
                      sample["DECLINATION"].to_numpy() * u.deg)
    sample["EBV_PLANCK13"] = PlanckQuery(map_fname=str(planck_map), component="extragalactic")(coords)
    sample["EBV_PLANCK13_TAU"] = PlanckQuery(map_fname=str(planck_map), component="tau")(coords)
    return sample


def comparison_statistics(sample):
    """GAL_EBV compared with each map: differences and ratios."""
    gal_ebv = sample["GAL_EBV"].to_numpy(dtype=float)
    rows = {}
    for column in ["EBV_PLANCK13", "EBV_PLANCK13_TAU", "EBV_SFD98"]:
        other = sample[column].to_numpy(dtype=float)
        difference = gal_ebv - other
        ratio = gal_ebv / other
        rows[column] = {
            "n": len(sample),
            "median difference": np.median(difference),
            "max |difference|": np.max(np.abs(difference)),
            f"fraction |difference| > {SAME_VALUE_TOLERANCE:g}": np.mean(np.abs(difference) > SAME_VALUE_TOLERANCE),
            "median ratio": np.median(ratio),
            "ratio 16%": np.percentile(ratio, 16),
            "ratio 84%": np.percentile(ratio, 84),
        }
    return pd.DataFrame(rows).T


def plot_comparison(sample, path):
    """GAL_EBV vs the Planck 2013 E(B-V) and tau-based E(B-V), and GAL_EBV / SFD98 vs SFD98."""
    gal_ebv = sample["GAL_EBV"].to_numpy(dtype=float)
    sfd = sample["EBV_SFD98"].to_numpy(dtype=float)
    limits = (0.0, np.percentile(np.concatenate([gal_ebv, sfd]), 99.9))

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH, 2.4))
    for ax, column, label in [
        (axes[0], "EBV_PLANCK13", r"$E(B-V)$ Planck 2013 [mag]"),
        (axes[1], "EBV_PLANCK13_TAU", r"$E(B-V)$ Planck 2013 from $\tau_{353}$ [mag]"),
    ]:
        other = sample[column].to_numpy(dtype=float)
        hexbin = ax.hexbin(other, gal_ebv, gridsize=60, extent=(*limits, *limits), mincnt=1,
                           norm=LogNorm(vmin=1, vmax=len(sample) / 10), cmap="viridis",
                           rasterized=True)
        ax.plot(limits, limits, color="black", ls="--", lw=0.8)
        ax.set(xlim=limits, ylim=limits, xlabel=label)
        ax.text(0.96, 0.04, rf"max $|\Delta|$ = {np.max(np.abs(gal_ebv - other)):.2g} mag"
                f"\n{len(sample):,} sources", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8)
    axes[0].set_ylabel(r"Euclid GAL$\_$EBV [mag]")
    axes[1].tick_params(labelleft=False)

    ax = axes[2]
    ax.hexbin(sfd, gal_ebv / sfd, gridsize=60, extent=(*limits, 0.5, 2.0), mincnt=1,
              norm=LogNorm(vmin=1, vmax=len(sample) / 10), cmap="viridis", rasterized=True)
    bins = np.linspace(*limits, 21)
    centers = 0.5 * (bins[1:] + bins[:-1])
    which = np.digitize(sfd, bins) - 1
    medians = [np.median((gal_ebv / sfd)[which == i]) if np.any(which == i) else np.nan
               for i in range(len(centers))]
    ax.plot(centers, medians, color=plot_style.CATEGORICAL_COLORS[0], lw=1.2, label="median")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set(xlim=limits, ylim=(0.5, 2.0), xlabel=r"$E(B-V)$ SFD98 [mag]",
           ylabel=r"GAL$\_$EBV / SFD98")
    ax.legend(frameon=False, loc="upper right")

    fig.colorbar(hexbin, ax=axes.tolist(), label="Sources", pad=0.01)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--catalog", type=Path, default=CATALOG_DIR, help="DR1 HATS catalog")
    ap.add_argument("--planck-map", type=Path, default=PLANCK_MAP)
    ap.add_argument("--per-partition", type=int, default=200, help="sources sampled per partition")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    sample = add_planck_ebv(sample_catalog(args.catalog, args.per_partition), args.planck_map)
    sample.to_parquet(args.output / "gal_ebv_sample.parquet", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", "{:.6g}".format):
        print(comparison_statistics(sample))
    for suffix in ("pdf", "png"):
        plot_comparison(sample, args.output / f"gal_ebv_vs_planck_sfd98.{suffix}")


if __name__ == "__main__":
    main()
