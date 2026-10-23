"""Draw the figures on the "Your first fit" page.

Run from the repository root:
    python tools/docs/first_fit_figures.py                     # the data figure only
    python tools/docs/first_fit_figures.py --run results/WASP-39_SOSS_ORDER1 --joint results/WASP-39_joint

The plotting code is the same as the code shown on docs/quickstart.md, with the
documentation house style applied on top. The data figure needs only the bundled
FITS file; the others read the CSV products of a finished first fit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits

from generate_tutorial_figures import COLOR, OUT, ROOT, finish, style


def data_figure():
    with fits.open(ROOT / "examples/data/WASP-39_soss_binned8.fits") as hdul:
        time = hdul["TIME"].data          # BMJD_TDB, as in every JWST file
        wave = hdul["WAVE O1"].data       # microns
        flux = hdul["FLUX O1"].data       # (integration, wavelength)

    hours = (time - time[0]) * 24
    white = flux.sum(axis=1)
    white /= np.median(white[:100])
    image = flux / np.median(flux[:100], axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 5.4), sharex=True)
    ax1.plot(hours, white, ".k", ms=2)
    ax1.set_ylabel("white-light flux")
    ax2.pcolormesh(hours, wave, image.T, vmin=0.97, vmax=1.01, shading="nearest")
    ax2.set_xlabel("hours since the first integration")
    ax2.set_ylabel("wavelength (µm)")
    finish(fig, "first_fit_data.png")


def whitelight_figure(out):
    wl = pd.read_csv(next(out.glob("*_whitelight_lightcurve.csv")))
    good = wl["is_outlier"] == 0
    hours = wl["time_from_t0_hr"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(hours[good], wl["detrended_flux"][good], ".k", ms=2, alpha=0.5)
    ax1.plot(hours, wl["transit_model"], "C1", lw=1.5)
    ax1.set_ylabel("detrended flux")
    ax2.plot(hours[good], wl["residual_ppm"][good], ".k", ms=2, alpha=0.5)
    ax2.axhline(0, color="C1", lw=1.5)
    ax2.set_xlabel("hours from mid-transit")
    ax2.set_ylabel("residual (ppm)")
    finish(fig, "first_fit_whitelight.png")


def spectrum_figure(out):
    spec = pd.read_csv(next(out.glob("*_R100_spectrum.csv")))

    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    ax.errorbar(spec["wavelength"], spec["depth_ppm00"],
                xerr=spec["wavelength_err"], yerr=spec["depth_err_ppm00"],
                fmt="o", ms=3, color="k")
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("transit depth (ppm)")
    finish(fig, "first_fit_spectrum.png")

    # The front page and README show the same spectrum in the house colour.
    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    ax.errorbar(spec["wavelength"], spec["depth_ppm00"],
                xerr=spec["wavelength_err"], yerr=spec["depth_err_ppm00"],
                fmt="o", ms=3.5, mfc="white", mec=COLOR, ecolor=COLOR, elinewidth=0.9)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("transit depth (ppm)")
    finish(fig, "soss_spectrum.png")


def both_orders_figure(out):
    fig, ax = plt.subplots(figsize=(6.9, 3.2))
    for order in (1, 2):
        spec = pd.read_csv(next((out / f"joint_soss_order{order}").glob("*_R100_spectrum.csv")))
        ax.errorbar(spec["wavelength"], spec["depth_ppm00"],
                    xerr=spec["wavelength_err"], yerr=spec["depth_err_ppm00"],
                    fmt="o", ms=3, label=f"order {order}")
    ax.legend()
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("transit depth (ppm)")
    finish(fig, "soss_both_orders.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, help="output directory of a finished first fit")
    parser.add_argument("--joint", type=Path, help="output directory of examples/joint_soss_orders.yaml")
    args = parser.parse_args()
    style()
    data_figure()
    if args.run is not None:
        whitelight_figure(args.run)
        spectrum_figure(args.run)
    if args.joint is not None:
        both_orders_figure(args.joint)


if __name__ == "__main__":
    main()
