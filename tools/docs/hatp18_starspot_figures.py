"""Turn a finished HAT-P-18 b starspot run into the tutorial figures.

Usage::

    python tools/docs/hatp18_starspot_figures.py /path/to/HAT-P-18_SOSS_ORDER1_STARSPOT

Writes to ``docs/_static``:

- ``hatp18_starspot_whitelight.png``: white-light data, fitted model, and
  residuals, with an inset on the spot crossing.
- ``hatp18_starspot_spectrum.png``: the R = 100 transmission spectrum.
- ``hatp18_starspot_contrast.png``: the fitted spot contrast per wavelength
  bin.

The script reads only the CSV tables written by ``fit_jwst.py``
(``*_whitelight_timeseries.csv``, ``*_R100.csv``, ``*_stellar_spots.csv``),
so it can run on any machine without JAX.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_static"
SPOT_WINDOW_DAYS = 0.02  # half-width of the crossing inset around its centre


def _one(results: Path, pattern: str) -> Path:
    matches = sorted(results.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {results}")
    return matches[0]


def _finish(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def whitelight(results: Path, spot_center: float | None) -> Path:
    table = pd.read_csv(_one(results, "*_whitelight_timeseries.csv"))
    good = table["is_outlier"] == 0
    t = table.loc[good, "time_bjd"].to_numpy()
    flux = table.loc[good, "flux"].to_numpy()
    err = table.loc[good, "flux_err"].to_numpy()
    model = table.loc[good, "bestfit_model"].to_numpy()
    resid = table.loc[good, "residual_ppm"].to_numpy()
    t_ref = np.floor(t.min())
    x = t - t_ref

    if spot_center is None:
        # The crossing is the largest positive bump in the transit-only
        # residual when the spot is removed; fall back to the in-transit
        # point where the fitted model deviates most from a smooth transit.
        in_transit = model < np.median(model) - 0.25 * (np.max(model) - np.min(model))
        window = np.where(in_transit)[0]
        spot_center = t[window[np.argmax(np.gradient(model)[window])]] if window.size else t[len(t) // 2]

    fig, (ax_lc, ax_res) = plt.subplots(
        2, 1, figsize=(8.6, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    ax_lc.errorbar(x, flux, yerr=err, fmt=".", color="0.25", ms=4, lw=0.6, zorder=1)
    ax_lc.plot(x, model, color="#76538e", lw=1.8, zorder=2, label="spotted-star model")
    ax_lc.set_ylabel("Relative flux")
    ax_lc.legend(frameon=False, loc="lower right")

    inset = ax_lc.inset_axes([0.36, 0.42, 0.3, 0.36])
    sel = np.abs(t - spot_center) < SPOT_WINDOW_DAYS
    inset.errorbar(x[sel], flux[sel], yerr=err[sel], fmt=".", color="0.25", ms=4, lw=0.6)
    inset.plot(x[sel], model[sel], color="#76538e", lw=1.8)
    inset.set_title("spot crossing", fontsize=9)
    inset.tick_params(labelsize=7)

    ax_res.axhline(0.0, color="0.6", lw=0.8)
    ax_res.plot(x, resid, ".", color="0.25", ms=4)
    ax_res.set_ylabel("Residual [ppm]")
    ax_res.set_xlabel(f"Time [BJD - {t_ref:.0f}]")
    for ax in (ax_lc, ax_res):
        ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, "hatp18_starspot_whitelight.png")


def spectrum(results: Path) -> Path:
    table = pd.read_csv(_one(results, "*_R100.csv"))
    good = table[["wavelength", "depth_ppm00", "depth_err_ppm00"]].notna().all(axis=1)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.errorbar(
        table.loc[good, "wavelength"], table.loc[good, "depth_ppm00"],
        xerr=table.loc[good, "wavelength_err"], yerr=table.loc[good, "depth_err_ppm00"],
        fmt="o", color="#76538e", ms=3.5, lw=0.9,
    )
    ax.set(xlabel="Wavelength [micron]", ylabel="Transit depth [ppm]")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, "hatp18_starspot_spectrum.png")


def contrast(results: Path) -> Path:
    table = pd.read_csv(_one(results, "*_stellar_spots.csv"))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for spot, rows in table.groupby("spot_index"):
        ax.errorbar(
            rows["wavelength"], rows["contrast"], xerr=rows["wavelength_err"],
            yerr=np.vstack([rows["contrast_err_low"], rows["contrast_err_high"]]),
            fmt="o", ms=3.5, lw=0.9, label=f"spot {spot + 1}",
        )
    ax.set(xlabel="Wavelength [micron]", ylabel="Spot contrast")
    if table["spot_index"].nunique() > 1:
        ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, "hatp18_starspot_contrast.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="output_dir of the finished fit")
    parser.add_argument(
        "--spot-center", type=float, default=None,
        help="BJD of the spot crossing for the inset (default: detect from the model)",
    )
    args = parser.parse_args()
    for path in (
        whitelight(args.results, args.spot_center),
        spectrum(args.results),
        contrast(args.results),
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
