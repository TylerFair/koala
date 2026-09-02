#!/usr/bin/env python3
"""Make publication-quality plots from the STELLARINFORMED fit products.

For every fit directory directly below ``--input-root``, this script writes a
combined white-light and transmission-spectrum figure.  The default output is
a flat publication package whose informative filenames identify the planet,
instrument, observing mode, detector/order, and visit where applicable.

The script understands the version suffixes used by multi-visit fits and gives
preference to Rreference/Rnative products.  MIRI ``pixN`` and explicit ``RN``
products are supported as fallbacks for runs that use those labels instead.
PDF artists remain vectorized.  Before plotting a spectrum, isolated points
whose uncertainty exceeds a configurable multiple of the local neighboring
median are excluded and written to an audit CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

try:
    from cmcrameri import cm as cmc
except ImportError:  # pragma: no cover - the plotting environment has cmcrameri
    cmc = None


DEFAULT_INPUT_ROOT = Path("/scratch/midway3/tfairnington/STELLARINFORMED")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "LIGHTCURVE_AND_SPECTRA_PLOTS"
PRODUCT_RE = re.compile(r"_(Rreference|Rnative|pix\d+|R\d+)(?:_V\d+)?$", re.IGNORECASE)
VERSION_RE = re.compile(r"(?:^|_)V(\d+)(?:_|$)", re.IGNORECASE)
FILE_VERSION_RE = re.compile(r"_V(\d+)$", re.IGNORECASE)

INK = "#20242B"
MUTED_INK = "#53606D"
OUTLIER_COLOR = "#B9BEC5"
ZERO_COLOR = "#8D949C"
MODEL_COLOR = "#D1495B"
RESIDUAL_BIN_MINUTES = 5.0
ANNOTATION_BOX = {
    "boxstyle": "round,pad=0.28",
    "facecolor": "white",
    "edgecolor": "#D6DBE1",
    "linewidth": 0.7,
    "alpha": 0.90,
}


@dataclass(frozen=True)
class RunProducts:
    run_dir: Path
    white_path: Path
    spectrum_path: Path
    product_label: str
    target: str
    instrument: str
    visit: int | None


@dataclass(frozen=True)
class PreparedSpectrum:
    wavelength: np.ndarray
    xerr: np.ndarray
    depth_ppm: np.ndarray
    precision_ppm: np.ndarray
    clipped_rows: list[dict[str, str]]


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.0,
            "axes.labelsize": 12.0,
            "axes.labelweight": "medium",
            "axes.linewidth": 1.15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def visit_from_name(name: str) -> int | None:
    match = VERSION_RE.search(name)
    return int(match.group(1)) if match else None


def target_from_name(name: str) -> str:
    match = re.search(r"_(?:G\d{3}[HM]|SOSS|PRISM|MIRI)", name, re.IGNORECASE)
    return name[: match.start()] if match else name


def planet_name(target: str) -> str:
    """Return the conventional planet label used in titles and filenames."""

    return target if target[-1:].lower() in {"b", "c", "d", "e", "f"} else f"{target}b"


def instrument_slug(run_name: str) -> str:
    """Build a compact, filesystem-safe instrument identifier for one run."""

    upper = run_name.upper()
    if "SOSS" in upper:
        order_match = re.search(r"ORDER(\d+)", upper)
        order = f"_ORDER{order_match.group(1)}" if order_match else ""
        slug = f"NIRISS_SOSS{order}"
    elif "PRISM" in upper:
        detector = "_NRS2" if "NRS2" in upper else "_NRS1" if "NRS1" in upper else ""
        slug = f"NIRSPEC_PRISM{detector}"
    else:
        grating_match = re.search(r"(?:^|_)(G(?:140|235|395)[HM])(?:_|$)", upper)
        if grating_match:
            detector = "_NRS2" if "NRS2" in upper else "_NRS1" if "NRS1" in upper else ""
            slug = f"NIRSPEC_{grating_match.group(1)}{detector}"
        elif "MIRI" in upper or "LRS" in upper:
            slug = "MIRI_LRS"
        else:
            raise ValueError(f"Could not derive an instrument filename from {run_name}")

    visit = visit_from_name(run_name)
    return f"{slug}_V{visit}" if visit is not None else slug


def publication_basename(products: RunProducts) -> str:
    return f"{planet_name(products.target)}_{instrument_slug(products.run_dir.name)}"


def instrument_from_name(name: str) -> str:
    upper = name.upper()
    visit_match = VERSION_RE.search(name)
    visit_text = f" · Visit {visit_match.group(1)}" if visit_match else ""
    detector = ""
    if "NRS1" in upper:
        detector = " NRS1"
    elif "NRS2" in upper:
        detector = " NRS2"

    if "SOSS" in upper:
        order_match = re.search(r"ORDER(\d+)", upper)
        order = f" Order {order_match.group(1)}" if order_match else ""
        return f"JWST/NIRISS SOSS{order}{visit_text}"
    if "PRISM" in upper:
        return f"JWST/NIRSpec PRISM{detector}{visit_text}"
    for grating in ("G140H", "G235H", "G395M", "G395H"):
        if grating in upper:
            return f"JWST/NIRSpec {grating}{detector}{visit_text}"
    if "MIRI" in upper or "LRS" in upper:
        return f"JWST/MIRI LRS{visit_text}"
    return name.replace("_", " ")


def instrument_color(name: str) -> tuple[float, float, float, float] | str:
    upper = name.upper()
    if "ORDER2" in upper:
        sample = 0.26
    elif "ORDER1" in upper or "SOSS" in upper:
        sample = 0.18
    elif "PRISM" in upper:
        sample = 0.38
    elif "G140H" in upper:
        sample = 0.50
    elif "G235H" in upper:
        sample = 0.58
    elif "G395M" in upper:
        sample = 0.68
    elif "G395H" in upper and "NRS1" in upper:
        sample = 0.76
    elif "G395H" in upper:
        sample = 0.82
    elif "MIRI" in upper or "LRS" in upper:
        sample = 0.90
    else:
        sample = 0.65
    return cmc.managua(sample) if cmc is not None else plt.get_cmap("viridis")(sample)


def choose_version(candidates: list[Path], visit: int | None) -> list[Path]:
    if not candidates:
        return []
    if visit is not None:
        matched = [path for path in candidates if path.stem.upper().endswith(f"_V{visit}")]
        if matched:
            return matched
    unversioned = [path for path in candidates if FILE_VERSION_RE.search(path.stem) is None]
    return unversioned or candidates


def product_score(label: str) -> tuple[int, int]:
    lower = label.lower()
    if lower == "rreference":
        return (4, 0)
    if lower == "rnative":
        return (4, 0)
    if lower.startswith("pix"):
        # Fewer detector pixels per bin means the higher-resolution MIRI product.
        pixels = int(lower[3:])
        return (3, -pixels)
    if lower.startswith("r") and lower[1:].isdigit():
        return (2, int(lower[1:]))
    return (0, 0)


def select_white_path(run_dir: Path, visit: int | None) -> Path | None:
    candidates = sorted(run_dir.glob("*_whitelight_timeseries*.csv"))
    selected = choose_version(candidates, visit)
    return sorted(selected)[0] if selected else None


def select_spectrum_path(run_dir: Path, visit: int | None) -> tuple[Path | None, str | None]:
    candidates: list[tuple[Path, str]] = []
    for path in run_dir.glob("*.csv"):
        match = PRODUCT_RE.search(path.stem)
        if match:
            candidates.append((path, match.group(1)))
    selected_paths = set(choose_version([path for path, _label in candidates], visit))
    selected = [(path, label) for path, label in candidates if path in selected_paths]
    if not selected:
        return None, None
    path, label = max(selected, key=lambda item: (product_score(item[1]), item[0].name))
    return path, label


def discover_runs(input_root: Path) -> tuple[list[RunProducts], list[tuple[str, str]]]:
    runs: list[RunProducts] = []
    skipped: list[tuple[str, str]] = []
    for run_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        visit = visit_from_name(run_dir.name)
        white_path = select_white_path(run_dir, visit)
        spectrum_path, product_label = select_spectrum_path(run_dir, visit)
        if white_path is None and spectrum_path is None:
            skipped.append((run_dir.name, "no direct fit products (auxiliary directory)"))
            continue
        missing = []
        if white_path is None:
            missing.append("white-light timeseries")
        if spectrum_path is None:
            missing.append("native/reference spectrum")
        if missing:
            skipped.append((run_dir.name, "missing " + " and ".join(missing)))
            continue
        assert spectrum_path is not None and product_label is not None and white_path is not None
        runs.append(
            RunProducts(
                run_dir=run_dir,
                white_path=white_path,
                spectrum_path=spectrum_path,
                product_label=product_label,
                target=target_from_name(run_dir.name),
                instrument=instrument_from_name(run_dir.name),
                visit=visit,
            )
        )
    return runs, skipped


def finite_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def add_figure_heading(fig: plt.Figure, products: RunProducts, detail: str) -> None:
    fig.text(0.125, 0.965, products.target, ha="left", va="top", fontsize=16.0, weight="bold", color=INK)
    fig.text(
        0.125,
        0.925,
        f"{products.instrument} · {detail}",
        ha="left",
        va="top",
        fontsize=10.8,
        color=MUTED_INK,
    )


def line_with_gaps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    finite = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    x_sorted = x_sorted[finite]
    y_sorted = y_sorted[finite]
    if x_sorted.size < 3:
        return x_sorted, y_sorted
    steps = np.diff(x_sorted)
    positive_steps = steps[steps > 0]
    if positive_steps.size == 0:
        return x_sorted, y_sorted
    gap_limit = 8.0 * np.nanmedian(positive_steps)
    gap_indices = np.flatnonzero(steps > gap_limit)
    if gap_indices.size == 0:
        return x_sorted, y_sorted
    return np.insert(x_sorted, gap_indices + 1, np.nan), np.insert(y_sorted, gap_indices + 1, np.nan)


def robust_symmetric_limit(values: np.ndarray, errors: np.ndarray | None = None) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    centered = values - np.nanmedian(values)
    mad_sigma = 1.4826 * np.nanmedian(np.abs(centered))
    percentile = np.nanpercentile(np.abs(values), 99.5)
    error_floor = 0.0
    if errors is not None:
        finite_errors = errors[np.isfinite(errors) & (errors >= 0)]
        if finite_errors.size:
            error_floor = 2.0 * np.nanmedian(finite_errors)
    limit = max(4.5 * mad_sigma, percentile, error_floor, 1.0)
    return 1.08 * float(limit)


def set_flux_limits(ax: plt.Axes, flux: np.ndarray, model: np.ndarray, errors: np.ndarray) -> None:
    finite_flux = flux[np.isfinite(flux)]
    finite_model = model[np.isfinite(model)]
    if finite_flux.size == 0 or finite_model.size == 0:
        return
    low, high = np.nanpercentile(finite_flux, [0.2, 99.8])
    low = min(low, float(np.nanmin(finite_model)))
    high = max(high, float(np.nanmax(finite_model)))
    finite_errors = errors[np.isfinite(errors) & (errors >= 0)]
    if finite_errors.size:
        pad_error = float(np.nanmedian(finite_errors))
        low -= pad_error
        high += pad_error
    span = high - low
    if not np.isfinite(span) or span <= 0:
        span = max(abs(high) * 1e-4, 1e-4)
    ax.set_ylim(low - 0.08 * span, high + 0.08 * span)


def configure_precision_axis(
    ax: plt.Axes,
    precision_ppm: np.ndarray,
    pad_fraction: float = 0.10,
) -> float:
    """Use the precision panel consistently without forcing an unhelpful zero."""

    finite = precision_ppm[np.isfinite(precision_ppm) & (precision_ppm >= 0)]
    if finite.size == 0:
        ax.set_ylim(0.0, 1.0)
        return np.nan
    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    median = float(np.nanmedian(finite))
    span = maximum - minimum
    if not np.isfinite(span) or span <= 0:
        pad = max(0.10 * max(abs(median), 1.0), 1.0)
    else:
        pad = pad_fraction * span
    lower = max(0.0, minimum - pad)
    upper = maximum + pad
    if upper <= lower:
        upper = lower + 1.0
    ax.set_ylim(lower, upper)
    return median


def bin_residuals(
    time_hr: np.ndarray,
    residual_ppm: np.ndarray,
    uncertainty_ppm: np.ndarray,
    bin_minutes: float = RESIDUAL_BIN_MINUTES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance bin residuals on a fixed time grid."""

    finite = (
        np.isfinite(time_hr)
        & np.isfinite(residual_ppm)
        & np.isfinite(uncertainty_ppm)
        & (uncertainty_ppm > 0)
    )
    time_hr = time_hr[finite]
    residual_ppm = residual_ppm[finite]
    uncertainty_ppm = uncertainty_ppm[finite]
    if time_hr.size < 2:
        return np.array([]), np.array([]), np.array([])

    bin_width_hr = bin_minutes / 60.0
    start = np.floor(np.nanmin(time_hr) / bin_width_hr) * bin_width_hr
    bin_index = np.floor((time_hr - start) / bin_width_hr).astype(int)
    binned_time: list[float] = []
    binned_residual: list[float] = []
    binned_uncertainty: list[float] = []
    for index in np.unique(bin_index):
        selected = bin_index == index
        if np.count_nonzero(selected) < 2:
            continue
        weights = 1.0 / np.square(uncertainty_ppm[selected])
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        binned_time.append(float(np.sum(weights * time_hr[selected]) / weight_sum))
        binned_residual.append(float(np.sum(weights * residual_ppm[selected]) / weight_sum))
        binned_uncertainty.append(float(np.sqrt(1.0 / weight_sum)))
    return (
        np.asarray(binned_time, dtype=float),
        np.asarray(binned_residual, dtype=float),
        np.asarray(binned_uncertainty, dtype=float),
    )


def plot_residual_bins(
    ax: plt.Axes,
    time_hr: np.ndarray,
    residual_ppm: np.ndarray,
    uncertainty_ppm: np.ndarray,
) -> None:
    bin_time, bin_value, bin_error = bin_residuals(time_hr, residual_ppm, uncertainty_ppm)
    if bin_time.size == 0:
        return
    ax.errorbar(
        bin_time,
        bin_value,
        yerr=bin_error,
        fmt="o",
        ms=3.8,
        mfc=INK,
        mec="white",
        mew=0.45,
        ecolor=INK,
        elinewidth=0.9,
        capsize=1.5,
        label=f"{RESIDUAL_BIN_MINUTES:g}-min bins",
        zorder=10,
    )
    ax.legend(
        loc="lower left",
        fontsize=8.3,
        handlelength=1.0,
        handletextpad=0.45,
        borderaxespad=0.4,
    )


def plot_white_light(products: RunProducts, output_dir: Path, formats: Iterable[str], dpi: int) -> None:
    frame = pd.read_csv(products.white_path)
    required = {"flux", "flux_err", "bestfit_model", "residual_ppm"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{products.white_path.name} lacks columns: {', '.join(sorted(missing))}")

    if "time_from_t0_hr" in frame:
        time_hr = finite_array(frame, "time_from_t0_hr")
    elif "time_bjd" in frame:
        time_bjd = finite_array(frame, "time_bjd")
        time_hr = 24.0 * (time_bjd - np.nanmedian(time_bjd))
    else:
        raise ValueError(f"{products.white_path.name} has no time column")
    flux = finite_array(frame, "flux")
    flux_err = finite_array(frame, "flux_err")
    model = finite_array(frame, "bestfit_model")
    residual_ppm = finite_array(frame, "residual_ppm")
    if "is_outlier" in frame:
        outlier = pd.to_numeric(frame["is_outlier"], errors="coerce").fillna(0).to_numpy(dtype=bool)
    else:
        outlier = np.zeros(len(frame), dtype=bool)

    base_finite = np.isfinite(time_hr) & np.isfinite(flux) & np.isfinite(model) & np.isfinite(residual_ppm)
    good = base_finite & ~outlier
    bad = base_finite & outlier
    if not np.any(good):
        raise ValueError(f"{products.white_path.name} contains no finite, unmasked points")

    color = instrument_color(products.run_dir.name)
    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.07},
    )
    n_good = int(np.count_nonzero(good))
    if n_good <= 3000:
        ax.errorbar(
            time_hr[good],
            flux[good],
            yerr=flux_err[good],
            fmt="o",
            ms=2.7,
            mfc="white",
            mec=color,
            mew=0.75,
            ecolor=color,
            elinewidth=0.45,
            alpha=0.65,
            capsize=0,
            zorder=2,
        )
        ax_res.errorbar(
            time_hr[good],
            residual_ppm[good],
            yerr=1e6 * flux_err[good],
            fmt="o",
            ms=2.3,
            mfc="white",
            mec=color,
            mew=0.65,
            ecolor=color,
            elinewidth=0.4,
            alpha=0.62,
            capsize=0,
            zorder=2,
        )
    else:
        marker_size = 3.0 if n_good < 10000 else 1.5
        alpha = 0.42 if n_good < 10000 else 0.28
        ax.scatter(
            time_hr[good], flux[good], s=marker_size, color=color, alpha=alpha,
            linewidths=0, zorder=2,
        )
        ax_res.scatter(
            time_hr[good], residual_ppm[good], s=marker_size, color=color, alpha=alpha,
            linewidths=0, zorder=2,
        )

    if np.any(bad):
        ax.scatter(
            time_hr[bad], flux[bad], s=12, marker="x", color=OUTLIER_COLOR,
            linewidths=0.7, zorder=1,
        )
        ax_res.scatter(
            time_hr[bad], residual_ppm[bad], s=12, marker="x", color=OUTLIER_COLOR,
            linewidths=0.7, zorder=1,
        )

    model_x, model_y = line_with_gaps(time_hr[base_finite], model[base_finite])
    ax.plot(model_x, model_y, color=MODEL_COLOR, lw=1.8, zorder=0.5)
    plot_residual_bins(ax_res, time_hr[good], residual_ppm[good], 1e6 * flux_err[good])
    ax_res.axhline(0.0, color=ZERO_COLOR, lw=1.0, ls=(0, (4, 3)), zorder=12)

    ax.set_ylabel("Relative Flux")
    ax_res.set_ylabel("Residual [ppm]")
    ax_res.set_xlabel("Time From Transit Midpoint [hr]")
    set_flux_limits(ax, flux[good], model[good], flux_err[good])
    residual_limit = robust_symmetric_limit(residual_ppm[good], 1e6 * flux_err[good])
    ax_res.set_ylim(-residual_limit, residual_limit)
    ax_res.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    rms = float(np.sqrt(np.nanmean(np.square(residual_ppm[good]))))
    median_uncertainty = float(np.nanmedian(1e6 * flux_err[good]))
    ax_res.text(
        0.985,
        0.92,
        f"RMS = {rms:,.0f} ppm  ·  median $\\sigma$ = {median_uncertainty:,.0f} ppm",
        transform=ax_res.transAxes,
        ha="right",
        va="top",
        fontsize=9.4,
        color=MUTED_INK,
    )
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=5.0, mfc="white", mec=color, mew=1.0, label="Observed flux"),
        Line2D([0], [0], color=MODEL_COLOR, lw=1.8, label="Best-fit model"),
    ]
    if np.any(bad):
        handles.append(Line2D([0], [0], marker="x", ls="", ms=5.0, color=OUTLIER_COLOR, label="Excluded"))
    ax.legend(handles=handles, loc="lower left", ncol=len(handles), fontsize=9.5, handlelength=1.8, columnspacing=1.2)

    for axis in (ax, ax_res):
        axis.tick_params(top=False, right=False)
        axis.margins(x=0.015)
    add_figure_heading(fig, products, "White-light transit")
    fig.align_ylabels((ax, ax_res))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.11, top=0.86, hspace=0.07)
    save_figure(fig, output_dir / "publication_whitelight", formats, dpi)


def find_spectral_column(frame: pd.DataFrame, prefix: str) -> str:
    exact = [column for column in frame.columns if column.lower() == prefix.lower()]
    if exact:
        return exact[0]
    numbered = sorted(column for column in frame.columns if column.lower().startswith(prefix.lower()))
    if numbered:
        return numbered[0]
    raise ValueError(f"no column matching {prefix}")


def find_precision_outliers(
    precision_ppm: np.ndarray,
    factor: float,
    neighbor_window: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify isolated uncertainty spikes relative to nearby spectral bins.

    The local baseline is the median of up to ``neighbor_window`` points on
    either side, explicitly excluding the point being tested.  This preserves
    smooth wavelength-dependent changes in precision while catching isolated,
    very large error bars.
    """

    precision_ppm = np.asarray(precision_ppm, dtype=float)
    baseline = np.full(precision_ppm.shape, np.nan, dtype=float)
    ratio = np.full(precision_ppm.shape, np.nan, dtype=float)
    for index in range(precision_ppm.size):
        left = precision_ppm[max(0, index - neighbor_window) : index]
        right = precision_ppm[index + 1 : index + 1 + neighbor_window]
        neighbors = np.concatenate((left, right))
        neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
        if neighbors.size < 2:
            continue
        baseline[index] = float(np.nanmedian(neighbors))
        if baseline[index] > 0:
            ratio[index] = precision_ppm[index] / baseline[index]
    outlier = np.isfinite(ratio) & (ratio > factor)
    return outlier, baseline, ratio


def find_depth_outliers(
    depth_ppm: np.ndarray,
    precision_ppm: np.ndarray,
    sigma_threshold: float,
    neighbor_window: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify only extreme, isolated depth excursions.

    Significance is measured against the local median depth using the
    quadrature sum of the point uncertainty and the neighboring median
    uncertainty.  A high default threshold avoids clipping plausible narrow
    spectral structure.
    """

    depth_ppm = np.asarray(depth_ppm, dtype=float)
    precision_ppm = np.asarray(precision_ppm, dtype=float)
    baseline = np.full(depth_ppm.shape, np.nan, dtype=float)
    significance = np.full(depth_ppm.shape, np.nan, dtype=float)
    for index in range(depth_ppm.size):
        neighbor_depths = np.concatenate(
            (
                depth_ppm[max(0, index - neighbor_window) : index],
                depth_ppm[index + 1 : index + 1 + neighbor_window],
            )
        )
        neighbor_errors = np.concatenate(
            (
                precision_ppm[max(0, index - neighbor_window) : index],
                precision_ppm[index + 1 : index + 1 + neighbor_window],
            )
        )
        finite = np.isfinite(neighbor_depths) & np.isfinite(neighbor_errors) & (neighbor_errors > 0)
        if np.count_nonzero(finite) < 2 or precision_ppm[index] <= 0:
            continue
        baseline[index] = float(np.nanmedian(neighbor_depths[finite]))
        neighbor_precision = float(np.nanmedian(neighbor_errors[finite]))
        combined_precision = np.hypot(precision_ppm[index], neighbor_precision)
        if combined_precision > 0:
            significance[index] = abs(depth_ppm[index] - baseline[index]) / combined_precision
    outlier = np.isfinite(significance) & (significance > sigma_threshold)
    return outlier, baseline, significance


def prepare_spectrum(
    products: RunProducts,
    precision_outlier_factor: float,
    depth_outlier_sigma: float,
) -> PreparedSpectrum:
    frame = pd.read_csv(products.spectrum_path)
    try:
        depth_column = find_spectral_column(frame, "depth_ppm")
        precision_column = find_spectral_column(frame, "depth_err_ppm")
    except ValueError as exc:
        raise ValueError(f"{products.spectrum_path.name}: {exc}") from exc
    if "wavelength" not in frame:
        raise ValueError(f"{products.spectrum_path.name} has no wavelength column")

    wavelength = finite_array(frame, "wavelength")
    wavelength_err = finite_array(frame, "wavelength_err") if "wavelength_err" in frame else np.full_like(wavelength, np.nan)
    depth_ppm = finite_array(frame, depth_column)
    precision_ppm = finite_array(frame, precision_column)
    finite = np.isfinite(wavelength) & np.isfinite(depth_ppm) & np.isfinite(precision_ppm)
    finite &= precision_ppm >= 0
    if not np.any(finite):
        raise ValueError(f"{products.spectrum_path.name} contains no finite spectral points")
    source_rows = np.flatnonzero(finite)
    wavelength = wavelength[finite]
    wavelength_err = wavelength_err[finite]
    depth_ppm = depth_ppm[finite]
    precision_ppm = precision_ppm[finite]
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    wavelength_err = wavelength_err[order]
    depth_ppm = depth_ppm[order]
    precision_ppm = precision_ppm[order]
    source_rows = source_rows[order]

    precision_outlier, local_baseline, precision_ratio = find_precision_outliers(
        precision_ppm,
        factor=precision_outlier_factor,
    )
    depth_outlier, local_depth_baseline, depth_significance = find_depth_outliers(
        depth_ppm,
        precision_ppm,
        sigma_threshold=depth_outlier_sigma,
    )
    spectral_outlier = precision_outlier | depth_outlier
    clipped_rows = [
        {
            "run": products.run_dir.name,
            "product_label": products.product_label,
            "source_file": str(products.spectrum_path),
            "source_row_zero_based": str(int(source_rows[index])),
            "wavelength_micron": f"{wavelength[index]:.12g}",
            "depth_ppm": f"{depth_ppm[index]:.12g}",
            "precision_ppm": f"{precision_ppm[index]:.12g}",
            "neighbor_median_precision_ppm": f"{local_baseline[index]:.12g}",
            "precision_ratio": f"{precision_ratio[index]:.8g}",
            "neighbor_median_depth_ppm": f"{local_depth_baseline[index]:.12g}",
            "depth_local_significance_sigma": f"{depth_significance[index]:.8g}",
            "clip_reason": ";".join(
                reason
                for condition, reason in (
                    (precision_outlier[index], "precision_ratio"),
                    (depth_outlier[index], "depth_significance"),
                )
                if condition
            ),
            "precision_threshold_factor": f"{precision_outlier_factor:g}",
            "depth_threshold_sigma": f"{depth_outlier_sigma:g}",
        }
        for index in np.flatnonzero(spectral_outlier)
    ]
    keep = ~spectral_outlier
    if not np.any(keep):
        raise ValueError(f"{products.spectrum_path.name}: precision clipping removed every point")
    wavelength = wavelength[keep]
    wavelength_err = wavelength_err[keep]
    depth_ppm = depth_ppm[keep]
    precision_ppm = precision_ppm[keep]
    xerr = np.where(np.isfinite(wavelength_err) & (wavelength_err >= 0), wavelength_err, 0.0)
    return PreparedSpectrum(
        wavelength=wavelength,
        xerr=xerr,
        depth_ppm=depth_ppm,
        precision_ppm=precision_ppm,
        clipped_rows=clipped_rows,
    )


def plot_spectrum(
    products: RunProducts,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    precision_outlier_factor: float,
    depth_outlier_sigma: float,
) -> list[dict[str, str]]:
    spectrum = prepare_spectrum(products, precision_outlier_factor, depth_outlier_sigma)
    wavelength = spectrum.wavelength
    xerr = spectrum.xerr
    depth_ppm = spectrum.depth_ppm
    precision_ppm = spectrum.precision_ppm

    color = instrument_color(products.run_dir.name)
    fig, (ax, ax_precision) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.0], "hspace": 0.07},
    )
    ax.errorbar(
        wavelength,
        depth_ppm,
        xerr=xerr,
        yerr=precision_ppm,
        fmt="o",
        ms=4.0,
        mfc="white",
        mec=color,
        mew=1.0,
        ecolor=color,
        elinewidth=1.0,
        capsize=0,
        alpha=0.88,
        zorder=3,
    )
    ax_precision.plot(wavelength, precision_ppm, color=color, lw=1.35, alpha=0.75, zorder=1)
    ax_precision.scatter(
        wavelength,
        precision_ppm,
        s=14,
        facecolor="white",
        edgecolor=color,
        linewidth=0.9,
        zorder=2,
    )

    ax.set_ylabel("Transit Depth [ppm]")
    ax_precision.set_ylabel("Precision [ppm]")
    ax_precision.set_xlabel("Wavelength [$\\mu$m]")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.0f}"))
    ax_precision.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.0f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_precision.yaxis.set_major_locator(MaxNLocator(nbins=5))

    lower = float(np.nanmin(depth_ppm - precision_ppm))
    upper = float(np.nanmax(depth_ppm + precision_ppm))
    depth_span = upper - lower
    if not np.isfinite(depth_span) or depth_span <= 0:
        depth_span = max(abs(upper) * 0.05, 1.0)
    ax.set_ylim(lower - 0.08 * depth_span, upper + 0.08 * depth_span)
    median_precision = configure_precision_axis(ax_precision, precision_ppm)
    left = float(np.nanmin(wavelength - xerr))
    right = float(np.nanmax(wavelength + xerr))
    wavelength_span = right - left
    ax_precision.set_xlim(left - 0.015 * wavelength_span, right + 0.015 * wavelength_span)

    ax_precision.text(
        0.985,
        0.90,
        f"Median precision = {median_precision:,.0f} ppm",
        transform=ax_precision.transAxes,
        ha="right",
        va="top",
        fontsize=9.4,
        color=MUTED_INK,
    )
    detail = f"{products.product_label} transmission spectrum"
    add_figure_heading(fig, products, detail)
    for axis in (ax, ax_precision):
        axis.tick_params(top=False, right=False)
    fig.align_ylabels((ax, ax_precision))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.11, top=0.86, hspace=0.07)
    save_figure(fig, output_dir / "publication_spectrum", formats, dpi)
    return spectrum.clipped_rows


def plot_combined(
    products: RunProducts,
    output_stem: Path,
    formats: Iterable[str],
    dpi: int,
    precision_outlier_factor: float,
    depth_outlier_sigma: float,
    spine_style: str,
) -> list[dict[str, str]]:
    """Plot white-light and spectral results side by side in one figure."""

    if spine_style not in {"two", "four"}:
        raise ValueError(f"Unknown spine style: {spine_style}")

    white = pd.read_csv(products.white_path)
    required = {"flux", "flux_err", "bestfit_model", "residual_ppm"}
    missing = required.difference(white.columns)
    if missing:
        raise ValueError(f"{products.white_path.name} lacks columns: {', '.join(sorted(missing))}")
    if "time_from_t0_hr" in white:
        time_hr = finite_array(white, "time_from_t0_hr")
    elif "time_bjd" in white:
        time_bjd = finite_array(white, "time_bjd")
        time_hr = 24.0 * (time_bjd - np.nanmedian(time_bjd))
    else:
        raise ValueError(f"{products.white_path.name} has no time column")
    flux = finite_array(white, "flux")
    flux_err = finite_array(white, "flux_err")
    model = finite_array(white, "bestfit_model")
    residual_ppm = finite_array(white, "residual_ppm")
    if "is_outlier" in white:
        outlier = pd.to_numeric(white["is_outlier"], errors="coerce").fillna(0).to_numpy(dtype=bool)
    else:
        outlier = np.zeros(len(white), dtype=bool)
    base_finite = np.isfinite(time_hr) & np.isfinite(flux) & np.isfinite(model) & np.isfinite(residual_ppm)
    good = base_finite & ~outlier
    bad = base_finite & outlier
    if not np.any(good):
        raise ValueError(f"{products.white_path.name} contains no finite, unmasked points")

    spectrum = prepare_spectrum(products, precision_outlier_factor, depth_outlier_sigma)
    wavelength = spectrum.wavelength
    xerr = spectrum.xerr
    depth_ppm = spectrum.depth_ppm
    precision_ppm = spectrum.precision_ppm
    color = instrument_color(products.run_dir.name)

    fig = plt.figure(figsize=(15.2, 6.5))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[3.15, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.07,
        wspace=0.28,
    )
    ax_white = fig.add_subplot(grid[0, 0])
    ax_residual = fig.add_subplot(grid[1, 0], sharex=ax_white)
    ax_spectrum = fig.add_subplot(grid[0, 1])
    ax_precision = fig.add_subplot(grid[1, 1], sharex=ax_spectrum)

    n_good = int(np.count_nonzero(good))
    if n_good <= 3000:
        ax_white.errorbar(
            time_hr[good],
            flux[good],
            yerr=flux_err[good],
            fmt="o",
            ms=2.5,
            mfc="white",
            mec=color,
            mew=0.7,
            ecolor=color,
            elinewidth=0.4,
            alpha=0.65,
            capsize=0,
            zorder=2,
        )
        ax_residual.errorbar(
            time_hr[good],
            residual_ppm[good],
            yerr=1e6 * flux_err[good],
            fmt="o",
            ms=2.1,
            mfc="white",
            mec=color,
            mew=0.6,
            ecolor=color,
            elinewidth=0.35,
            alpha=0.60,
            capsize=0,
            zorder=2,
        )
    else:
        marker_size = 2.8 if n_good < 10000 else 1.4
        alpha = 0.40 if n_good < 10000 else 0.27
        ax_white.scatter(
            time_hr[good], flux[good], s=marker_size, color=color, alpha=alpha,
            linewidths=0, zorder=2,
        )
        ax_residual.scatter(
            time_hr[good], residual_ppm[good], s=marker_size, color=color, alpha=alpha,
            linewidths=0, zorder=2,
        )
    if np.any(bad):
        ax_white.scatter(
            time_hr[bad], flux[bad], s=11, marker="x", color=OUTLIER_COLOR,
            linewidths=0.65, zorder=1,
        )
        ax_residual.scatter(
            time_hr[bad], residual_ppm[bad], s=11, marker="x", color=OUTLIER_COLOR,
            linewidths=0.65, zorder=1,
        )
    model_x, model_y = line_with_gaps(time_hr[base_finite], model[base_finite])
    ax_white.plot(model_x, model_y, color=MODEL_COLOR, lw=1.75, zorder=0.5)
    plot_residual_bins(ax_residual, time_hr[good], residual_ppm[good], 1e6 * flux_err[good])
    ax_residual.axhline(0.0, color=ZERO_COLOR, lw=1.0, ls=(0, (4, 3)), zorder=12)
    ax_white.set_ylabel("Relative Flux")
    ax_residual.set_ylabel("Residual [ppm]")
    ax_residual.set_xlabel("Time From Transit Midpoint [hr]")
    set_flux_limits(ax_white, flux[good], model[good], flux_err[good])
    residual_limit = robust_symmetric_limit(residual_ppm[good], 1e6 * flux_err[good])
    ax_residual.set_ylim(-residual_limit, residual_limit)
    ax_residual.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax_white.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_white.ticklabel_format(axis="y", style="plain", useOffset=False)
    rms = float(np.sqrt(np.nanmean(np.square(residual_ppm[good]))))
    median_uncertainty = float(np.nanmedian(1e6 * flux_err[good]))
    ax_residual.text(
        0.02,
        0.91,
        f"RMS = {rms:,.0f} ppm  ·  median $\\sigma$ = {median_uncertainty:,.0f} ppm",
        transform=ax_residual.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color=MUTED_INK,
        bbox=ANNOTATION_BOX,
        zorder=10,
    )
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=4.8, mfc="white", mec=color, mew=1.0, label="Observed flux"),
        Line2D([0], [0], color=MODEL_COLOR, lw=1.75, label="Best-fit model"),
    ]
    if np.any(bad):
        handles.append(Line2D([0], [0], marker="x", ls="", ms=4.8, color=OUTLIER_COLOR, label="Excluded"))
    ax_white.legend(
        handles=handles,
        loc="lower left",
        ncol=len(handles),
        fontsize=9.0,
        handlelength=1.7,
        columnspacing=1.0,
    )

    ax_spectrum.errorbar(
        wavelength,
        depth_ppm,
        xerr=xerr,
        yerr=precision_ppm,
        fmt="o",
        ms=3.8,
        mfc="white",
        mec=color,
        mew=0.95,
        ecolor=color,
        elinewidth=0.95,
        capsize=0,
        alpha=0.88,
        zorder=3,
    )
    ax_precision.plot(wavelength, precision_ppm, color=color, lw=1.3, alpha=0.75, zorder=1)
    ax_precision.scatter(
        wavelength,
        precision_ppm,
        s=13,
        facecolor="white",
        edgecolor=color,
        linewidth=0.85,
        zorder=2,
    )
    ax_spectrum.set_ylabel("Transit Depth [ppm]")
    ax_precision.set_ylabel("Precision [ppm]")
    ax_precision.set_xlabel("Wavelength [$\\mu$m]")
    ax_spectrum.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:.0f}"))
    ax_precision.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:,.0f}"))
    ax_spectrum.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_precision.yaxis.set_major_locator(MaxNLocator(nbins=5))
    lower = float(np.nanmin(depth_ppm - precision_ppm))
    upper = float(np.nanmax(depth_ppm + precision_ppm))
    depth_span = upper - lower
    if not np.isfinite(depth_span) or depth_span <= 0:
        depth_span = max(abs(upper) * 0.05, 1.0)
    ax_spectrum.set_ylim(lower - 0.08 * depth_span, upper + 0.08 * depth_span)
    median_precision = configure_precision_axis(ax_precision, precision_ppm)
    left = float(np.nanmin(wavelength - xerr))
    right = float(np.nanmax(wavelength + xerr))
    wavelength_span = right - left
    ax_precision.set_xlim(left - 0.015 * wavelength_span, right + 0.015 * wavelength_span)
    ax_precision.text(
        0.02,
        0.90,
        f"Median precision = {median_precision:,.0f} ppm",
        transform=ax_precision.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color=MUTED_INK,
        bbox=ANNOTATION_BOX,
        zorder=10,
    )

    four_spines = spine_style == "four"
    for axis in (ax_white, ax_residual, ax_spectrum, ax_precision):
        axis.spines["top"].set_visible(four_spines)
        axis.spines["right"].set_visible(four_spines)
        axis.tick_params(
            top=four_spines,
            right=four_spines,
            labeltop=False,
            labelright=False,
        )
    # Each upper/lower pair shares x; show ticks and the label only once below.
    ax_white.tick_params(axis="x", labelbottom=False)
    ax_spectrum.tick_params(axis="x", labelbottom=False)
    ax_white.margins(x=0.015)
    ax_residual.margins(x=0.015)
    fig.text(
        0.065,
        0.963,
        planet_name(products.target),
        ha="left",
        va="top",
        fontsize=17.0,
        weight="bold",
        color=INK,
    )
    fig.text(0.065, 0.915, products.instrument, ha="left", va="top", fontsize=11.2, color=MUTED_INK)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.11, top=0.83, hspace=0.07, wspace=0.28)
    save_figure(fig, output_stem, formats, dpi)
    return spectrum.clipped_rows


def save_figure(fig: plt.Figure, stem: Path, formats: Iterable[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = stem.with_suffix(f".{extension}")
        if extension == "png":
            fig.savefig(output_path, dpi=dpi)
        else:
            fig.savefig(output_path)
    plt.close(fig)


def write_manifest(
    output_root: Path,
    rows: list[dict[str, str]],
    skipped: list[tuple[str, str]],
    clipped_rows: list[dict[str, str]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "target",
        "instrument",
        "product_label",
        "white_source",
        "spectrum_source",
        "clipped_spectral_points",
        "layout",
        "spines",
        "output_stem",
        "output_directory",
        "status",
    ]
    with (output_root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    clipped_fieldnames = [
        "run",
        "product_label",
        "source_file",
        "source_row_zero_based",
        "wavelength_micron",
        "depth_ppm",
        "precision_ppm",
        "neighbor_median_precision_ppm",
        "precision_ratio",
        "neighbor_median_depth_ppm",
        "depth_local_significance_sigma",
        "clip_reason",
        "precision_threshold_factor",
        "depth_threshold_sigma",
    ]
    with (output_root / "clipped_spectrum_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=clipped_fieldnames)
        writer.writeheader()
        writer.writerows(clipped_rows)
    with (output_root / "skipped.txt").open("w") as handle:
        for name, reason in skipped:
            handle.write(f"{name}: {reason}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
        help="Output formats (default: png pdf).",
    )
    parser.add_argument("--dpi", type=int, default=450, help="PNG resolution (default: 450 dpi; minimum: 400).")
    parser.add_argument(
        "--layout",
        choices=("combined", "separate", "both"),
        default="combined",
        help="Write a merged figure, separate figures, or both (default: combined).",
    )
    parser.add_argument(
        "--spines",
        choices=("two", "four", "both"),
        default="four",
        help="Spine style for combined figures (default: four).",
    )
    parser.add_argument(
        "--precision-outlier-factor",
        type=float,
        default=5.0,
        help="Exclude spectral points with uncertainty above this multiple of the local median (default: 5).",
    )
    parser.add_argument(
        "--depth-outlier-sigma",
        type=float,
        default=10.0,
        help="Exclude isolated depth excursions above this local significance (default: 10 sigma).",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="REGEX",
        help="Only process run-directory names matching this regex (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List selected inputs without plotting.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input_root.is_dir():
        print(f"Input root does not exist: {args.input_root}", file=sys.stderr)
        return 2
    if args.dpi < 400:
        print("--dpi must be at least 400 for publication output", file=sys.stderr)
        return 2
    if not np.isfinite(args.precision_outlier_factor) or args.precision_outlier_factor <= 1.0:
        print("--precision-outlier-factor must be finite and greater than 1", file=sys.stderr)
        return 2
    if not np.isfinite(args.depth_outlier_sigma) or args.depth_outlier_sigma < 5.0:
        print("--depth-outlier-sigma must be finite and at least 5", file=sys.stderr)
        return 2

    apply_publication_style()
    runs, skipped = discover_runs(args.input_root)
    if args.only:
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in args.only]
        runs = [run for run in runs if any(pattern.search(run.run_dir.name) for pattern in patterns)]
    basenames = [publication_basename(products) for products in runs]
    duplicate_basenames = sorted({name for name in basenames if basenames.count(name) > 1})
    if duplicate_basenames:
        print(
            "Non-unique publication filename(s): " + ", ".join(duplicate_basenames),
            file=sys.stderr,
        )
        return 2
    print(f"Selected {len(runs)} complete runs from {args.input_root}", flush=True)
    for name, reason in skipped:
        print(f"Skipping {name}: {reason}", flush=True)
    if args.dry_run:
        for products in runs:
            print(
                f"{products.run_dir.name}: {products.white_path.name} | "
                f"{products.spectrum_path.name} ({products.product_label}) -> "
                f"{publication_basename(products)}"
            )
        return 0

    rows: list[dict[str, str]] = []
    all_clipped_rows: list[dict[str, str]] = []
    failures: list[tuple[str, str]] = []
    for index, products in enumerate(runs, start=1):
        output_dir = args.output_root / products.run_dir.name
        base_name = publication_basename(products)
        status = "ok"
        clipped_rows: list[dict[str, str]] = []
        try:
            if args.layout in ("separate", "both"):
                plot_white_light(products, output_dir, args.formats, args.dpi)
                clipped_rows = plot_spectrum(
                    products,
                    output_dir,
                    args.formats,
                    args.dpi,
                    args.precision_outlier_factor,
                    args.depth_outlier_sigma,
                )
            if args.layout in ("combined", "both"):
                spine_styles = ("two", "four") if args.spines == "both" else (args.spines,)
                for spine_index, spine_style in enumerate(spine_styles):
                    variant_suffix = "_TWO_SPINES" if spine_style == "two" else ""
                    output_stem = args.output_root / f"{base_name}{variant_suffix}"
                    combined_clipped_rows = plot_combined(
                        products,
                        output_stem,
                        args.formats,
                        args.dpi,
                        args.precision_outlier_factor,
                        args.depth_outlier_sigma,
                        spine_style,
                    )
                    if args.layout == "combined" and spine_index == 0:
                        clipped_rows = combined_clipped_rows
                    elif len(combined_clipped_rows) != len(clipped_rows):
                        raise RuntimeError("figure variants produced different clipping results")
            all_clipped_rows.extend(clipped_rows)
        except Exception as exc:
            status = f"ERROR: {exc}"
            failures.append((products.run_dir.name, str(exc)))
        rows.append(
            {
                "run": products.run_dir.name,
                "target": products.target,
                "instrument": products.instrument,
                "product_label": products.product_label,
                "white_source": str(products.white_path),
                "spectrum_source": str(products.spectrum_path),
                "clipped_spectral_points": str(len(clipped_rows)),
                "layout": args.layout,
                "spines": args.spines if args.layout in ("combined", "both") else "n/a",
                "output_stem": str(args.output_root / base_name) if args.layout in ("combined", "both") else "",
                "output_directory": str(args.output_root if args.layout == "combined" else output_dir),
                "status": status,
            }
        )
        print(
            f"[{index:02d}/{len(runs):02d}] {products.run_dir.name}: {status}; "
            f"clipped {len(clipped_rows)} spectral point(s)",
            flush=True,
        )

    write_manifest(args.output_root, rows, skipped + failures, all_clipped_rows)
    if failures:
        print(f"Finished with {len(failures)} failed runs; see {args.output_root / 'skipped.txt'}", file=sys.stderr)
        return 1
    print(f"Wrote publication plots to {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
