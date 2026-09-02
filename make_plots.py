#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import fnmatch
from pathlib import Path
import re
import sys
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

try:
    import scienceplots  # noqa: F401
except Exception:
    scienceplots = None

try:
    from cmcrameri import cm as cmc
except Exception:
    cmc = None


PLOT_DPI = 250
SPECTRUM_MARKER = "o"
SPECTRUM_MS = 5.5
SPECTRUM_ELW = 1.0
WATERFALL_ALPHA = 0.9
WATERFALL_LW = 0.9
DEFAULT_MAX_WATERFALL_CHANNELS = 24
MODEL_COLOR = "k"
ZERO_LINE_COLOR = "0.40"
LIGHTCURVE_COLOR = "mediumorchid"
SPECTRUM_COLOR = LIGHTCURVE_COLOR
CURRENT_PALETTE = "orchid"
AVAILABLE_PALETTES = ("orchid", "batlow", "bamako", "lajolla", "tokyo", "oslo", "hawaii", "lipari")
JWST_GRATING_TOKEN_RE = re.compile(r"G\d{3}[HM]")
JWST_DETECTOR_TOKEN_RE = re.compile(r"NRS[12]")


def _cmc_sample(name: str, x: float) -> str:
    if cmc is None or not hasattr(cmc, name):
        return "mediumorchid"
    return getattr(cmc, name)(x)


PALETTE_TO_COLOR = {
    "orchid": "mediumorchid",
    "batlow": _cmc_sample("batlow", 0.72),
    "bamako": _cmc_sample("bamako", 0.68),
    "lajolla": _cmc_sample("lajolla", 0.58),
    "tokyo": _cmc_sample("tokyo", 0.66),
    "oslo": _cmc_sample("oslo", 0.78),
    "hawaii": _cmc_sample("hawaii", 0.63),
    "lipari": _cmc_sample("lipari", 0.70),
}


def set_plot_palette(name: str) -> None:
    global LIGHTCURVE_COLOR, SPECTRUM_COLOR, CURRENT_PALETTE
    if name not in PALETTE_TO_COLOR:
        raise ValueError(f"Unknown palette: {name}")
    CURRENT_PALETTE = name
    LIGHTCURVE_COLOR = PALETTE_TO_COLOR[name]
    SPECTRUM_COLOR = LIGHTCURVE_COLOR


def accent_color_for_label(label: str | None) -> str:
    if CURRENT_PALETTE == "orchid" or label is None:
        return LIGHTCURVE_COLOR

    ulabel = label.upper()
    # Related shades within the same sequential colormap.
    if "NIRISS" in ulabel or "SOSS" in ulabel:
        if "ORDER2" in ulabel or "O2" in ulabel:
            sample = 0.26
        else:
            sample = 0.18
    elif "PRISM" in ulabel:
        sample = 0.38
    elif "G140H" in ulabel:
        sample = 0.50
    elif "G235H" in ulabel:
        sample = 0.58
    elif "G395M" in ulabel:
        sample = 0.68
    elif "G395H" in ulabel:
        sample = 0.80
    elif "LRS" in ulabel or "MIRI" in ulabel:
        sample = 0.88
    else:
        sample = 0.70
    return _cmc_sample(CURRENT_PALETTE, sample)


def apply_publication_style() -> None:
    if scienceplots is not None:
        plt.style.use(["science", "nature"])
    else:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "axes.spines.top": True,
                "axes.spines.right": True,
                "legend.frameon": False,
            }
        )
    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.4,
            "axes.labelpad": 12.0,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "xtick.major.pad": 8.0,
            "ytick.major.pad": 8.0,
            "figure.constrained_layout.use": True,
        }
    )
    plt.rcParams["savefig.bbox"] = None


def parse_scalar_like(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return np.nan
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            try:
                return float(stripped)
            except Exception:
                return value
        if isinstance(parsed, (list, tuple, np.ndarray)):
            if len(parsed) == 0:
                return np.nan
            if len(parsed) == 1:
                return float(parsed[0])
            return np.asarray(parsed, dtype=float)
        if isinstance(parsed, (int, float)):
            return float(parsed)
        return parsed
    return value


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        out[col] = [parse_scalar_like(v) for v in df[col]]
    return out


@dataclass
class SpectroProduct:
    label: str
    bestfit_path: Path
    noisebin_path: Path | None


@dataclass
class PandexoCurve:
    wavelength_um: np.ndarray
    sigma_ppm: np.ndarray


def find_products(run_dir: Path) -> tuple[Path | None, Path | None, list[SpectroProduct]]:
    whitelight_params = None
    whitelight_series = None
    spectro: list[SpectroProduct] = []
    for bestfit_path in sorted(run_dir.glob("*_bestfit_params.csv")):
        name = bestfit_path.name
        if "_whitelight_bestfit_params.csv" in name:
            whitelight_params = bestfit_path
            continue
        label = name.replace("_bestfit_params.csv", "")
        noise_matches = sorted(run_dir.glob(f"*_{label}_noisebin.csv"))
        noisebin_path = noise_matches[0] if noise_matches else None
        spectro.append(
            SpectroProduct(
                label=label,
                bestfit_path=bestfit_path,
                noisebin_path=noisebin_path,
            )
        )
    series_matches = sorted(run_dir.glob("*_whitelight_timeseries.csv"))
    if series_matches:
        whitelight_series = series_matches[0]
    return whitelight_params, whitelight_series, spectro


def _has_jwst_mode_tokens(tokens: list[str]) -> bool:
    for i in range(len(tokens) - 1):
        if tokens[i] == "SOSS" and re.fullmatch(r"ORDER[12]", tokens[i + 1]):
            return True
        if tokens[i] == "MIRI" and tokens[i + 1] == "LRS":
            return True
        if JWST_GRATING_TOKEN_RE.fullmatch(tokens[i]) and JWST_DETECTOR_TOKEN_RE.fullmatch(tokens[i + 1]):
            return True
    return "PRISM" in tokens


def looks_like_jwst_run_dir_name(name: str) -> bool:
    tokens = name.upper().split("_")
    if len(tokens) < 4:
        return False
    if "POWER2" not in tokens:
        return False
    return _has_jwst_mode_tokens(tokens)


def has_posthoc_products(run_dir: Path) -> bool:
    patterns = (
        "*_bestfit_params.csv",
        "*_whitelight_timeseries.csv",
        "*_noisebin.csv",
    )
    return any(any(run_dir.glob(pattern)) for pattern in patterns)


def _matches_glob_filters(name: str, include_globs: Iterable[str] | None = None, exclude_globs: Iterable[str] | None = None) -> bool:
    include = [pattern for pattern in (include_globs or ["*"]) if pattern]
    exclude = [pattern for pattern in (exclude_globs or []) if pattern]
    if not any(fnmatch.fnmatch(name, pattern) for pattern in include):
        return False
    if any(fnmatch.fnmatch(name, pattern) for pattern in exclude):
        return False
    return True


def discover_matching_run_dirs(
    section_dir: Path,
    include_globs: Iterable[str] | None = None,
    exclude_globs: Iterable[str] | None = None,
) -> list[Path]:
    matches: list[Path] = []
    for child in sorted(section_dir.iterdir()):
        if not child.is_dir():
            continue
        if not _matches_glob_filters(child.name, include_globs=include_globs, exclude_globs=exclude_globs):
            continue
        if not looks_like_jwst_run_dir_name(child.name):
            continue
        if not has_posthoc_products(child):
            continue
        matches.append(child)
    return matches


def resolve_run_dirs(
    input_path: Path,
    all_matching: bool,
    include_name_globs: Iterable[str] | None,
    exclude_name_globs: Iterable[str] | None,
) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    if all_matching:
        matches = discover_matching_run_dirs(
            input_path,
            include_globs=include_name_globs,
            exclude_globs=exclude_name_globs,
        )
        if not matches:
            raise FileNotFoundError(
                f"No matching JWST post-hoc run directories found under {input_path} "
                f"with include globs {list(include_name_globs or ['*'])!r}"
            )
        return matches

    if has_posthoc_products(input_path):
        return [input_path]

    matching_children = discover_matching_run_dirs(
        input_path,
        include_globs=include_name_globs,
        exclude_globs=exclude_name_globs,
    )
    if matching_children:
        raise FileNotFoundError(
            f"{input_path} does not contain post-hoc CSV products directly. "
            f"It does contain {len(matching_children)} matching child run directories; "
            f"use --all-matching to process them."
        )
    raise FileNotFoundError(f"No post-hoc CSV products found in {input_path}")


def filter_spectro_products(
    spectro_products: Iterable[SpectroProduct],
    include_globs: Iterable[str] | None = None,
    exclude_globs: Iterable[str] | None = None,
) -> list[SpectroProduct]:
    return [
        product
        for product in spectro_products
        if _matches_glob_filters(product.label, include_globs=include_globs, exclude_globs=exclude_globs)
    ]


def format_pm(value, err_low=None, err_high=None, fmt=".6f") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if err_low is None or err_high is None:
        return format(float(value), fmt)
    if not np.isfinite(err_low) or not np.isfinite(err_high):
        return format(float(value), fmt)
    return f"{format(float(value), fmt)} -{format(float(err_low), fmt)} +{format(float(err_high), fmt)}"


def _infer_half_widths(wavelengths: np.ndarray) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.size <= 1:
        return np.full_like(wavelengths, 0.05, dtype=float)
    mids = 0.5 * (wavelengths[1:] + wavelengths[:-1])
    edges = np.empty(wavelengths.size + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = wavelengths[0] - (mids[0] - wavelengths[0])
    edges[-1] = wavelengths[-1] + (wavelengths[-1] - mids[-1])
    return 0.5 * (edges[1:] - edges[:-1])


def _normalize_pandexo_wavelengths_um(wavelengths: np.ndarray) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=float)
    finite = wavelengths[np.isfinite(wavelengths)]
    if finite.size == 0:
        return wavelengths
    median = float(np.nanmedian(finite))
    if median > 100.0:
        return wavelengths / 1e4
    return wavelengths


def _normalize_pandexo_sigma_ppm(sigmas: np.ndarray) -> np.ndarray:
    sigmas = np.asarray(sigmas, dtype=float)
    finite = np.abs(sigmas[np.isfinite(sigmas)])
    if finite.size == 0:
        return sigmas
    median = float(np.nanmedian(finite))
    if median < 1e-2:
        return sigmas * 1e6
    return sigmas


def load_pandexo_curve(path: Path) -> PandexoCurve:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        wave_col = next(
            (c for c in ("wavelength_um", "wavelength", "wave", "wl", "lambda_um") if c in df.columns),
            None,
        )
        sigma_col = next(
            (
                c
                for c in (
                    "depth_err00",
                    "depth_err_ppm",
                    "depth_err",
                    "sigma_ppm",
                    "error_ppm",
                    "precision_ppm",
                    "sigma_depth",
                    "pandexo_err",
                    "error",
                    "err",
                )
                if c in df.columns
            ),
            None,
        )
        if wave_col is None or sigma_col is None:
            raise ValueError(f"{path} must contain a wavelength column and a sigma/error column")
        wave = df[wave_col].to_numpy(dtype=float)
        sigma = df[sigma_col].to_numpy(dtype=float)
    else:
        try:
            arr = np.genfromtxt(path, names=True, dtype=None, encoding=None)
        except Exception:
            arr = None
        wave = None
        sigma = None
        if arr is not None and getattr(arr, "dtype", None) is not None and arr.dtype.names:
            names = list(arr.dtype.names)
            wave_name = next((n for n in names if n.lower() in {"wavelength", "wavelength_um", "wave", "wl", "lambda"}), None)
            sigma_name = next(
                (
                    n
                    for n in names
                    if n.lower()
                    in {"sigma_ppm", "error_ppm", "precision_ppm", "sigma_depth", "pandexo_err", "error", "err"}
                ),
                None,
            )
            if wave_name is not None and sigma_name is not None:
                wave = np.asarray(arr[wave_name], dtype=float)
                sigma = np.asarray(arr[sigma_name], dtype=float)
        if wave is None or sigma is None:
            raw = np.loadtxt(path)
            if raw.ndim == 1:
                raw = raw[None, :]
            if raw.shape[1] < 2:
                raise ValueError(f"{path} must have at least two columns")
            wave = np.asarray(raw[:, 0], dtype=float)
            sigma = np.asarray(raw[:, -1], dtype=float)

    wave = _normalize_pandexo_wavelengths_um(wave)
    sigma = _normalize_pandexo_sigma_ppm(sigma)
    good = np.isfinite(wave) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(good):
        raise ValueError(f"{path} has no valid finite PandExo wavelength/sigma rows")
    order = np.argsort(wave[good])
    return PandexoCurve(wavelength_um=wave[good][order], sigma_ppm=sigma[good][order])


def bin_pandexo_to_data_grid(
    pandexo: PandexoCurve,
    data_wavelength_um: np.ndarray,
    data_half_width_um: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(data_wavelength_um, dtype=float)
    if data_half_width_um is None:
        half_widths = _infer_half_widths(centers)
    else:
        half_widths = np.asarray(data_half_width_um, dtype=float)
        bad = ~np.isfinite(half_widths) | (half_widths <= 0)
        if np.any(bad):
            half_widths = half_widths.copy()
            half_widths[bad] = _infer_half_widths(centers)[bad]

    lows = centers - half_widths
    highs = centers + half_widths
    out = np.full_like(centers, np.nan, dtype=float)
    pw = pandexo.wavelength_um
    ps = pandexo.sigma_ppm

    for i, (lo, hi) in enumerate(zip(lows, highs)):
        in_bin = (pw >= lo) & (pw <= hi)
        if np.any(in_bin):
            sigma_vals = ps[in_bin]
            out[i] = 1.0 / np.sqrt(np.sum(1.0 / np.square(sigma_vals)))
        else:
            out[i] = np.interp(centers[i], pw, ps, left=np.nan, right=np.nan)
    good = np.isfinite(out)
    return centers[good], out[good]


def pick_reference_t0(whitelight_df: pd.DataFrame | None) -> float | None:
    if whitelight_df is None or "t0" not in whitelight_df.columns or whitelight_df.empty:
        return None
    t0 = whitelight_df.iloc[0]["t0"]
    return float(t0) if np.isfinite(t0) else None


def plot_whitelight_summary(whitelight_df: pd.DataFrame, outpath: Path) -> None:
    row = whitelight_df.iloc[0]
    text = (
        f"P = {format_pm(row.get('period'), None, None, '.8f')} d\n"
        f"Duration = {format_pm(row.get('duration'), row.get('duration_err_low'), row.get('duration_err_high'), '.6f')} d\n"
        f"t0 = {format_pm(row.get('t0'), row.get('t0_err_low'), row.get('t0_err_high'), '.6f')}\n"
        f"b = {format_pm(row.get('b'), row.get('b_err_low'), row.get('b_err_high'), '.5f')}\n"
        f"Rp/R* = {format_pm(row.get('rors'), row.get('rors_err_low'), row.get('rors_err_high'), '.6f')}\n"
        f"Depth = {format_pm(1e6 * row.get('depths'), 1e6 * row.get('depths_err_low'), 1e6 * row.get('depths_err_high'), '.1f')} ppm"
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axis("off")
    ax.text(0.02, 0.95, text, ha="left", va="top", transform=ax.transAxes)
    fig.savefig(outpath, dpi=PLOT_DPI)
    plt.close(fig)


def format_instrument_stamp(label: str) -> str:
    parts = label.split("_")
    if len(parts) < 4:
        return label.replace("_", " ")
    inst_map = {
        "NIRSPEC": "NIRSpec",
        "NIRISS": "NIRISS",
        "MIRI": "MIRI",
    }
    instrument = inst_map.get(parts[1].upper(), parts[1])
    disperser = parts[2].upper()
    order = parts[3].upper()
    return f"JWST {instrument}/{disperser} {order}"


def format_instrument_stamp_display(label: str) -> str:
    stamp = format_instrument_stamp(label)
    if plt.rcParams.get("text.usetex", False):
        return rf"\textbf{{{stamp}}}"
    return stamp


def choose_symmetric_residual_ticks(residual_ppm: np.ndarray, residual_err_ppm: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    finite = np.isfinite(residual_ppm)
    if residual_err_ppm is not None:
        finite &= np.isfinite(residual_err_ppm)
        extent = np.nanmax(np.abs(residual_ppm[finite]) + residual_err_ppm[finite]) if np.any(finite) else np.nan
    else:
        extent = np.nanmax(np.abs(residual_ppm[finite])) if np.any(finite) else np.nan
    if not np.isfinite(extent) or extent <= 0:
        mid = 500.0
    else:
        target_mid = extent / 2.0
        candidates = []
        for power in range(-1, 7):
            scale = 10.0**power
            for mantissa in (1.0, 2.5, 5.0):
                candidates.append(mantissa * scale)
        mid = next((cand for cand in candidates if cand >= target_mid), candidates[-1])
    ticks = np.array([-2.0 * mid, -mid, 0.0, mid, 2.0 * mid], dtype=float)
    return 2.0 * mid, ticks


def enforce_min_major_ticks(ax, x: bool = True, y: bool = True) -> None:
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))


def plot_whitelight_curve(series_df: pd.DataFrame, label: str, outpath: Path) -> None:
    accent = accent_color_for_label(label)
    x = np.asarray(series_df["time_from_t0_hr"], dtype=float)
    if not np.any(np.isfinite(x)):
        x = np.asarray(series_df["time_bjd"], dtype=float) - np.nanmedian(np.asarray(series_df["time_bjd"], dtype=float))
        x = x * 24.0
    flux = np.asarray(series_df["flux"], dtype=float)
    flux_err = np.asarray(series_df["flux_err"], dtype=float)
    model = np.asarray(series_df["bestfit_model"], dtype=float)
    residual_ppm = np.asarray(series_df["residual_ppm"], dtype=float)
    outlier = np.asarray(series_df.get("is_outlier", np.zeros_like(x)), dtype=int).astype(bool)
    residual_err_ppm = flux_err * 1e6
    good = ~outlier

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    if np.any(outlier):
        axes[0].plot(x[outlier], flux[outlier], "o", ms=2.6, mfc="0.82", mec="0.82", mew=0.0, alpha=0.8, zorder=1)
    axes[0].plot(
        x[good],
        flux[good],
        linestyle="none",
        marker="o",
        ms=3.6,
        mfc="white",
        mec=accent,
        mew=1.7,
        zorder=2,
    )
    axes[0].plot(x, model, color=MODEL_COLOR, lw=1.9, zorder=3)
    axes[0].set_ylabel("Normalized Flux")
    y_ticks_main = axes[0].get_yticks()
    stamp_y = y_ticks_main[1] if len(y_ticks_main) >= 2 else np.nanmin(flux[good])
    axes[0].text(
        0.02,
        stamp_y,
        format_instrument_stamp_display(label),
        transform=blended_transform_factory(axes[0].transAxes, axes[0].transData),
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
    )

    if np.any(outlier):
        axes[1].plot(x[outlier], residual_ppm[outlier], "o", ms=2.6, mfc="0.82", mec="0.82", mew=0.0, alpha=0.8, zorder=1)
    axes[1].errorbar(
        x[good],
        residual_ppm[good],
        yerr=residual_err_ppm[good],
        fmt="o",
        ms=3.6,
        mfc="white",
        mec=accent,
        mew=1.7,
        ecolor=accent,
        elinewidth=1.7,
        capsize=0,
        linestyle="none",
        zorder=2,
    )
    axes[1].axhline(0.0, color=ZERO_LINE_COLOR, lw=1.7, ls="--", zorder=3)
    axes[1].set_xlabel("Time from Mid-Transit [hr]")
    axes[1].set_ylabel("Residuals [ppm]")
    residual_limit, residual_ticks = choose_symmetric_residual_ticks(residual_ppm[good], residual_err_ppm[good])
    axes[1].set_ylim(-residual_limit, residual_limit)
    axes[1].set_yticks(residual_ticks)
    fig.align_ylabels(axes)

    for ax in axes:
        enforce_min_major_ticks(ax, x=True, y=True)
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.4)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False, top=False, right=False)
    axes[1].set_yticks(residual_ticks)

    fig.savefig(outpath, dpi=PLOT_DPI)
    plt.close(fig)


def plot_spectrum_precision(
    params_df: pd.DataFrame,
    label: str,
    outpath: Path,
    pandexo_curve: PandexoCurve | None = None,
) -> None:
    accent = accent_color_for_label(label)
    wl = np.asarray(params_df["wavelength"], dtype=float)
    wl_err = (
        np.asarray(params_df["wavelength_err"], dtype=float)
        if "wavelength_err" in params_df.columns
        else _infer_half_widths(wl)
    )
    depth_ppm = np.asarray(params_df.get("depth_ppm", 1e6 * np.asarray(params_df["depth"], dtype=float)), dtype=float)
    depth_err = np.asarray(params_df.get("depth_err_ppm", 1e6 * np.asarray(params_df.get("depth_err", np.nan), dtype=float)), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    color = accent

    axes[0].errorbar(
        wl,
        depth_ppm,
        yerr=depth_err,
        fmt="o",
        ms=3.6,
        mfc="white",
        mec=color,
        mew=1.7,
        ecolor=color,
        elinewidth=1.7,
        capsize=0,
        linestyle="none",
    )
    axes[0].set_ylabel("Transit Depth [ppm]")

    axes[1].plot(wl, depth_err, color=color, lw=1.4)
    if pandexo_curve is not None:
        p_wl, p_sigma = bin_pandexo_to_data_grid(pandexo_curve, wl, wl_err)
        if p_wl.size > 0:
            axes[1].plot(p_wl, p_sigma, color="k", lw=1.5, ls="--")
    axes[1].set_xlabel("Wavelength [$\\mu$m]")
    axes[1].set_ylabel("Precision [ppm]")
    fig.align_ylabels(axes)

    axes[1].set_xlim(np.nanmin(wl) - 0.1, np.nanmax(wl) + 0.1)

    for ax in axes:
        enforce_min_major_ticks(ax, x=True, y=True)
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.4)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False, top=False, right=False)

    fig.savefig(outpath, dpi=PLOT_DPI)
    plt.close(fig)


def plot_noise_binning_from_csv(noise_df: pd.DataFrame, label: str, outpath: Path) -> None:
    if noise_df.empty:
        return
    accent = accent_color_for_label(label)
    bins = np.asarray(noise_df["bin_size_points"], dtype=float)
    measured = np.asarray(noise_df["measured_rms"], dtype=float)
    p16 = np.asarray(noise_df["measured_rms_p16"], dtype=float)
    p84 = np.asarray(noise_df["measured_rms_p84"], dtype=float)
    expected = np.asarray(noise_df["expected_white_rms"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.loglog(bins, expected, color=ZERO_LINE_COLOR, lw=1.7, ls="--")
    ax.fill_between(bins, p16, p84, color=accent, alpha=0.18)
    ax.loglog(
        bins,
        measured,
        color=accent,
        lw=1.8,
        marker="o",
        ms=3.4,
        mfc="white",
        mec=accent,
        mew=1.5,
    )
    ax.set_xlabel("Bin Size [points]")
    ax.set_ylabel("RMS [ppm]")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=6))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.4)
    ax.tick_params(axis="both", which="minor", bottom=False, left=False, top=False, right=False)
    fig.savefig(outpath, dpi=PLOT_DPI)
    plt.close(fig)


def build_plot_suite(
    run_dir: Path,
    outdir: Path,
    max_waterfall_channels: int,
    pandexo_path: Path | None = None,
    palette: str = "orchid",
    include_product_globs: Iterable[str] | None = None,
    exclude_product_globs: Iterable[str] | None = None,
) -> None:
    apply_publication_style()
    set_plot_palette(palette)
    outdir.mkdir(parents=True, exist_ok=True)
    run_label = run_dir.name
    for stale_name in (
        f"{run_label}_summary.png",
        f"{run_label}_spectrum.png",
        f"{run_label}_noiseplot.png",
    ):
        stale_path = outdir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    pandexo_curve = load_pandexo_curve(pandexo_path) if pandexo_path is not None else None

    whitelight_path, whitelight_series_path, spectro_products = find_products(run_dir)
    spectro_products = filter_spectro_products(
        spectro_products,
        include_globs=include_product_globs,
        exclude_globs=exclude_product_globs,
    )
    if whitelight_path is None and whitelight_series_path is None and not spectro_products:
        raise FileNotFoundError(f"No post-hoc CSV products found in {run_dir}")
    whitelight_df = None
    if whitelight_path is not None:
        whitelight_df = normalize_dataframe(pd.read_csv(whitelight_path))
    if whitelight_series_path is not None:
        whitelight_series_df = normalize_dataframe(pd.read_csv(whitelight_series_path))
        wl_label = whitelight_series_path.stem.replace("_whitelight_timeseries", "")
        plot_whitelight_curve(whitelight_series_df, wl_label, outdir / f"{run_label}_lightcurves.png")

    single_spectro_product = len(spectro_products) == 1
    for product in spectro_products:
        params_df = normalize_dataframe(pd.read_csv(product.bestfit_path))
        stem = product.label
        spectrum_name = f"{run_label}_spectrum.png" if single_spectro_product else f"{run_label}_{stem}_spectrum.png"
        noise_name = f"{run_label}_noiseplot.png" if single_spectro_product else f"{run_label}_{stem}_noiseplot.png"
        plot_spectrum_precision(params_df, stem, outdir / spectrum_name, pandexo_curve=pandexo_curve)
        if product.noisebin_path is not None:
            noise_df = normalize_dataframe(pd.read_csv(product.noisebin_path))
            plot_noise_binning_from_csv(noise_df, stem, outdir / noise_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate publication-style JWST plots from saved post-hoc CSV products."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="A single run directory, or a section/root directory when --all-matching is set.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory for regenerated plots. In single-run mode this defaults to "
            "<run_dir>/plots. In --all-matching mode, omitting this keeps plots inside each "
            "matched run directory."
        ),
    )
    parser.add_argument(
        "--all-matching",
        action="store_true",
        help=(
            "Treat run_dir as a section/root directory and process every immediate child "
            "directory whose name matches the JWST retrieval naming convention and which "
            "contains post-hoc CSV products."
        ),
    )
    parser.add_argument(
        "--name-glob",
        action="append",
        default=None,
        help=(
            "Optional fnmatch include filter for child directory names when using "
            "--all-matching. May be repeated, for example 'WASP-121_*' or "
            "'*_SOSS_ORDER1_*'."
        ),
    )
    parser.add_argument(
        "--exclude-name-glob",
        action="append",
        default=None,
        help=(
            "Optional fnmatch exclude filter for child directory names when using "
            "--all-matching. May be repeated, for example '*_R20*'."
        ),
    )
    parser.add_argument(
        "--product-glob",
        action="append",
        default=None,
        help=(
            "Optional fnmatch include filter for spectral product labels inside each run "
            "directory. May be repeated, for example '*_R20' and '*_Rreference'."
        ),
    )
    parser.add_argument(
        "--exclude-product-glob",
        action="append",
        default=None,
        help=(
            "Optional fnmatch exclude filter for spectral product labels inside each run "
            "directory. May be repeated, for example '*_R20'."
        ),
    )
    parser.add_argument(
        "--max-waterfall-channels",
        type=int,
        default=DEFAULT_MAX_WATERFALL_CHANNELS,
        help="Maximum number of channels to show in each waterfall plot.",
    )
    parser.add_argument(
        "--pandexo",
        type=Path,
        default=None,
        help="Optional PandExo curve file (csv/txt). Plotted as a dashed black curve in the precision panel after binning to the data grid.",
    )
    parser.add_argument(
        "--palette",
        choices=AVAILABLE_PALETTES,
        default="orchid",
        help="Accent palette for the plotted data. Default keeps the current mediumorchid look.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(
        args.run_dir,
        all_matching=args.all_matching,
        include_name_globs=args.name_glob,
        exclude_name_globs=args.exclude_name_glob,
    )
    base_outdir = args.outdir.expanduser().resolve() if args.outdir is not None else None
    pandexo_path = args.pandexo.expanduser().resolve() if args.pandexo is not None else None
    failures: list[tuple[Path, Exception]] = []

    for run_dir in run_dirs:
        outdir = base_outdir if base_outdir is not None else run_dir / "plots"
        try:
            build_plot_suite(
                run_dir,
                outdir,
                max_waterfall_channels=args.max_waterfall_channels,
                pandexo_path=pandexo_path,
                palette=args.palette,
                include_product_globs=args.product_glob,
                exclude_product_globs=args.exclude_product_glob,
            )
        except Exception as exc:
            failures.append((run_dir, exc))
            print(f"[FAILED] {run_dir}: {exc}", file=sys.stderr)
            continue
        print(f"Wrote plots for {run_dir.name} to {outdir}")

    if failures:
        failed_names = ", ".join(run_dir.name for run_dir, _ in failures)
        raise SystemExit(
            f"Completed {len(run_dirs) - len(failures)}/{len(run_dirs)} directories; "
            f"failures: {failed_names}"
        )


if __name__ == "__main__":
    main()

