#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import make_posthoc_jwst_plot_suite as posthoc


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "HAT-P-18_FINAL_PLOTS"
MODE_CMAP_NAME = "managua"
COMBINED_COMPARISON_PALETTE = "managua"
LIGHTCURVE_RESIDUAL_LIMIT_SCALE = 0.5
LIGHTCURVE_PALETTE = "managua"


@dataclass(frozen=True)
class ModeSpec:
    slug: str
    display_name: str
    combined_stem: str
    color_sample: float
    fallback_color: str
    run_names: tuple[str, ...]


@dataclass(frozen=True)
class InstrumentOverlaySpec:
    slug: str
    lightcurve_filename: str
    spectrum_filename: str


MODE_SPECS = (
    ModeSpec(
        slug="uniform",
        display_name="Uniform",
        combined_stem="HAT-P-18_UNIFORM",
        color_sample=0.18,
        fallback_color="#334d8d",
        run_names=(
            "HAT-P-18_SOSS_ORDER1_UNIFORMLD_POWER2_SPOT_R100",
            "HAT-P-18_SOSS_ORDER2_UNIFORMLD_POWER2_SPOT_R50",
            "HAT-P-18_G395M_NRS1_UNIFORMLD_POWER2_LINEAR_R300",
            "HAT-P-18_MIRI_LRS_UNIFORMLD_POWER2_EXPLINEAR_PIX6",
        ),
    ),
    ModeSpec(
        slug="widegaussian",
        display_name="Wide Gaussian",
        combined_stem="HAT-P-18_WIDEGAUSSIAN",
        color_sample=0.38,
        fallback_color="#43a6ba",
        run_names=(
            "HAT-P-18_SOSS_ORDER1_WIDEGAUSSIANLD_POWER2_SPOT_R100",
            "HAT-P-18_SOSS_ORDER2_WIDEGAUSSIANLD_POWER2_SPOT_R50",
            "HAT-P-18_G395M_NRS1_WIDEGAUSSIANLD_POWER2_LINEAR_R300",
            "HAT-P-18_MIRI_LRS_WIDEGAUSSIANLD_POWER2_EXPLINEAR_PIX6",
        ),
    ),
    ModeSpec(
        slug="stellarinformed",
        display_name="Stellar Marginalized (Fiducial)",
        combined_stem="HAT-P-18_STELLARINFORMED",
        color_sample=0.62,
        fallback_color="#88c86c",
        run_names=(
            "HAT-P-18_SOSS_ORDER1_STELLARPRIORLD_POWER2_SPOT_R100",
            "HAT-P-18_SOSS_ORDER2_STELLARPRIORLD_POWER2_SPOT_R50",
            "HAT-P-18_G395M_NRS1_STELLARPRIORLD_POWER2_LINEAR_R300",
            "HAT-P-18_MIRI_LRS_STELLARPRIORLD_POWER2_EXPLINEAR_PIX6",
        ),
    ),
    ModeSpec(
        slug="fixed",
        display_name="Fixed",
        combined_stem="HAT-P-18_FIXED",
        color_sample=0.82,
        fallback_color="#f0c64f",
        run_names=(
            "HAT-P-18_SOSS_ORDER1_FIXEDLD_POWER2_SPOT_R100",
            "HAT-P-18_SOSS_ORDER2_FIXEDLD_POWER2_SPOT_R50",
            "HAT-P-18_G395M_NRS1_FIXEDLD_POWER2_LINEAR_R300",
            "HAT-P-18_MIRI_LRS_FIXEDLD_POWER2_EXPLINEAR_PIX6",
        ),
    ),
)

INSTRUMENT_OVERLAY_SPECS = (
    InstrumentOverlaySpec(
        "soss_order1_r100",
        "HAT-P-18_SOSS_ORDER1_R100_all_ld_modes_lightcurves.png",
        "HAT-P-18_SOSS_ORDER1_R100_all_ld_modes_spectrum.png",
    ),
    InstrumentOverlaySpec(
        "soss_order2_r50",
        "HAT-P-18_SOSS_ORDER2_R50_all_ld_modes_lightcurves.png",
        "HAT-P-18_SOSS_ORDER2_R50_all_ld_modes_spectrum.png",
    ),
    InstrumentOverlaySpec(
        "g395m_r300",
        "HAT-P-18_G395M_R300_all_ld_modes_lightcurves.png",
        "HAT-P-18_G395M_R300_all_ld_modes_spectrum.png",
    ),
    InstrumentOverlaySpec(
        "miri_pix6",
        "HAT-P-18_MIRI_PIX6_all_ld_modes_lightcurves.png",
        "HAT-P-18_MIRI_PIX6_all_ld_modes_spectrum.png",
    ),
)


def _mode_color(spec: ModeSpec):
    sampled = posthoc._cmc_sample(MODE_CMAP_NAME, spec.color_sample)
    if sampled == "mediumorchid":
        return spec.fallback_color
    return sampled


def _instrument_palette_sample_for_run_name(run_name: str) -> float:
    upper_name = run_name.upper()
    if "SOSS_ORDER2" in upper_name or "ORDER2" in upper_name:
        return 0.26
    if "SOSS_ORDER1" in upper_name or "ORDER1" in upper_name:
        return 0.18
    if "PRISM" in upper_name:
        return 0.38
    if "G140H" in upper_name:
        return 0.50
    if "G235H" in upper_name:
        return 0.58
    if "G395M" in upper_name:
        return 0.68
    if "G395H" in upper_name:
        return 0.80
    if "MIRI" in upper_name or "LRS" in upper_name:
        return 0.88
    return 0.70


def _combined_mode_color(spec: ModeSpec, instrument_index: int | None = None) -> str:
    return _combined_mode_color_in_palette(spec, COMBINED_COMPARISON_PALETTE, instrument_index)


def _combined_mode_color_in_palette(spec: ModeSpec, palette_name: str, instrument_index: int | None = None) -> str:
    if spec.slug == "stellarinformed":
        sample = 0.68 if instrument_index is None else _instrument_palette_sample_for_run_name(spec.run_names[instrument_index])
        return posthoc._cmc_sample(palette_name, sample)
    sample_map = {
        "uniform": 0.08,
        "widegaussian": 0.38,
        "fixed": 0.54,
    }
    sample = sample_map.get(spec.slug, spec.color_sample)
    return posthoc._cmc_sample(palette_name, sample)


def _combined_plot_order(items):
    priority = {
        "uniform": 0,
        "widegaussian": 1,
        "fixed": 2,
        "stellarinformed": 3,
    }
    return sorted(items, key=lambda item: priority.get(item[0].slug, 99))


def _combined_mode_alpha(spec: ModeSpec) -> float:
    if spec.slug == "stellarinformed":
        return 0.82
    return 1.0


def _segment_match_key(segment) -> tuple[str, int, float]:
    wl = np.asarray(segment.params_df["wavelength"], dtype=float)
    if wl.size == 0:
        return (segment.product_label, 0, np.nan)
    return (segment.product_label, wl.size, round(float(np.nanmedian(wl)), 6))


def make_errorbar_legend_handle(color: str, label: str, *, ms: float = 7.0, lw: float = 1.2) -> ErrorbarContainer:
    point = Line2D(
        [0],
        [0],
        marker="o",
        ms=ms,
        mfc=color,
        mec=color,
        mew=0.0,
        linestyle="none",
    )
    barline = LineCollection([[(0.0, -1.0), (0.0, 1.0)]], colors=[color], linewidths=[lw])
    return ErrorbarContainer((point, (), (barline,)), has_xerr=False, has_yerr=True, label=label)


def add_mode_legend(ax, mode_entries, extra_handles=None) -> None:
    handles = [
        make_errorbar_legend_handle(color, spec.display_name)
        for spec, color in mode_entries
    ]
    if extra_handles:
        handles.extend(extra_handles)
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=min(2, len(handles)),
        frameon=False,
        fancybox=False,
        handlelength=1.0,
        handleheight=0.9,
        handletextpad=0.55,
        columnspacing=1.15,
        borderpad=0.0,
        borderaxespad=0.2,
        prop={"size": 11.0, "weight": "semibold"},
    )
    frame = legend.get_frame()
    frame.set_linewidth(0.0)
    frame.set_facecolor("none")
    frame.set_edgecolor("none")


def add_line_legend(ax, entries, *, ncol: int = 1) -> None:
    handles = [
        Line2D([0], [0], color=color, lw=lw, ls=ls, label=label)
        for label, color, ls, lw in entries
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=max(1, ncol),
        frameon=False,
        fancybox=False,
        borderaxespad=0.2,
        prop={"size": 11.0, "weight": "semibold"},
    )
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_edgecolor("none")


def draw_mode_sidebar(bar_ax, mode_entries) -> None:
    bar_ax.set_xlim(0.0, 1.0)
    bar_ax.set_ylim(0.0, 1.0)
    bar_ax.axis("off")

    shell = FancyBboxPatch(
        (0.05, 0.04),
        0.34,
        0.90,
        boxstyle="round,pad=0.015,rounding_size=0.045",
        linewidth=2.2,
        edgecolor="0.18",
        facecolor="white",
    )
    bar_ax.add_patch(shell)

    inner_x = 0.09
    inner_y = 0.07
    inner_w = 0.26
    inner_h = 0.84
    section_h = inner_h / len(mode_entries)

    for idx, (spec, color) in enumerate(reversed(mode_entries)):
        y0 = inner_y + idx * section_h
        bar_ax.add_patch(
            Rectangle(
                (inner_x, y0),
                inner_w,
                section_h,
                facecolor=color,
                edgecolor="white",
                linewidth=1.3,
            )
        )
        y_mid = y0 + 0.5 * section_h
        bar_ax.plot([0.39, 0.48], [y_mid, y_mid], color="0.25", lw=1.3, solid_capstyle="round")
        bar_ax.text(
            0.52,
            y_mid,
            spec.display_name,
            ha="left",
            va="center",
            fontsize=12.5,
            fontweight="semibold",
            color="0.15",
        )

    arrow = Polygon(
        [[0.22, -0.02], [0.10, 0.07], [0.34, 0.07]],
        closed=True,
        facecolor="0.18",
        edgecolor="0.18",
        lw=0.0,
    )
    bar_ax.add_patch(arrow)


def _clean_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def _sorted_segments(segments):
    return sorted(
        segments,
        key=lambda segment: float(np.nanmin(np.asarray(segment.params_df["wavelength"], dtype=float))),
    )


def _extract_series(params_df):
    wl = np.asarray(params_df["wavelength"], dtype=float)
    depth_ppm = np.asarray(
        params_df.get("depth_ppm", 1e6 * np.asarray(params_df["depth"], dtype=float)),
        dtype=float,
    )
    depth_err = np.asarray(
        params_df.get("depth_err_ppm", 1e6 * np.asarray(params_df.get("depth_err", np.nan), dtype=float)),
        dtype=float,
    )
    wl_err = (
        np.asarray(params_df["wavelength_err"], dtype=float)
        if "wavelength_err" in params_df.columns
        else posthoc._infer_half_widths(wl)
    )
    return wl, wl_err, depth_ppm, depth_err


def _segment_reference_key(segment) -> tuple[str, int, float, float]:
    wl = np.asarray(segment.params_df["wavelength"], dtype=float)
    return (
        str(segment.product_label),
        int(wl.size),
        float(np.nanmin(wl)) if wl.size else np.nan,
        float(np.nanmax(wl)) if wl.size else np.nan,
    )


def _interp_reference_to_target(target_wl, reference_wl, reference_values):
    return np.interp(target_wl, reference_wl, reference_values)


def _choose_mad_residual_ticks(residual_ppm, residual_err_ppm=None):
    residual_ppm = np.asarray(residual_ppm, dtype=float)
    finite = np.isfinite(residual_ppm)
    if not np.any(finite):
        return 20.0, np.array([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=float)

    center = np.nanmedian(residual_ppm[finite])
    mad = np.nanmedian(np.abs(residual_ppm[finite] - center))
    if residual_err_ppm is not None:
        residual_err_ppm = np.asarray(residual_err_ppm, dtype=float)
        finite_err = finite & np.isfinite(residual_err_ppm)
        err_floor = 2.0 * np.nanmedian(residual_err_ppm[finite_err]) if np.any(finite_err) else 0.0
    else:
        err_floor = 0.0

    target_limit = max(4.0 * mad, err_floor, 5.0)
    candidates = []
    for power in range(-1, 7):
        scale = 10.0**power
        for mantissa in (1.0, 2.0, 2.5, 5.0):
            candidates.append(mantissa * scale)
    limit = next((cand for cand in candidates if cand >= target_limit), candidates[-1])
    ticks = np.array([-limit, -0.5 * limit, 0.0, 0.5 * limit, limit], dtype=float)
    return limit, ticks


def _load_whitelight_series(run_dir: Path):
    _, series_path, _ = posthoc.find_products(run_dir)
    if series_path is None:
        raise FileNotFoundError(f"No white-light timeseries CSV found in {run_dir}")
    df = posthoc.normalize_dataframe(pd.read_csv(series_path))
    label = series_path.stem.replace("_whitelight_timeseries", "")
    return label, df


def _timeseries_x_hours(series_df):
    x = np.asarray(series_df["time_from_t0_hr"], dtype=float)
    if np.any(np.isfinite(x)):
        return x
    bjd = np.asarray(series_df["time_bjd"], dtype=float)
    return (bjd - np.nanmedian(bjd)) * 24.0


def plot_lightcurve_mode_overlay(mode_outputs, instrument_index: int, outpath: Path) -> None:
    posthoc.apply_publication_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.2),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    all_residuals = []
    all_residual_errs = []
    all_fluxes = []
    all_flux_errs = []
    stamp_label = None

    ordered_outputs = [
        (spec, _combined_mode_color_in_palette(spec, LIGHTCURVE_PALETTE, instrument_index), run_dirs, _mode_dir)
        for spec, _color, run_dirs, _mode_dir in _combined_plot_order(mode_outputs)
    ]

    for zorder, (spec, color, run_dirs, _mode_dir) in enumerate(ordered_outputs, start=1):
        run_dir = run_dirs[instrument_index]
        label, series_df = _load_whitelight_series(run_dir)
        if stamp_label is None:
            stamp_label = label
        x = _timeseries_x_hours(series_df)
        flux = np.asarray(series_df.get("detrended_flux", series_df["flux"]), dtype=float)
        flux_err = np.asarray(series_df["flux_err"], dtype=float)
        model = np.asarray(series_df.get("transit_model", series_df["bestfit_model"]), dtype=float)
        residual_ppm = np.asarray(series_df["residual_ppm"], dtype=float)
        outlier = np.asarray(series_df.get("is_outlier", np.zeros_like(x)), dtype=int).astype(bool)
        good = ~outlier
        residual_err_ppm = flux_err * 1e6

        axes[0].errorbar(
            x[good],
            flux[good],
            yerr=flux_err[good],
            fmt="o",
            ls="",
            ms=3.0,
            mfc="white",
            mec=color,
            mew=1.3,
            ecolor=color,
            elinewidth=1.0,
            capsize=0.0,
            zorder=zorder,
        )
        axes[0].plot(x, model, color=color, lw=1.5, zorder=zorder + 0.2)
        axes[1].errorbar(
            x[good],
            residual_ppm[good],
            yerr=residual_err_ppm[good],
            fmt="o",
            ls="",
            ms=2.8,
            mfc="white",
            mec=color,
            mew=1.1,
            ecolor=color,
            elinewidth=0.95,
            capsize=0.0,
            zorder=zorder,
        )
        all_residuals.append(residual_ppm[good])
        all_residual_errs.append(residual_err_ppm[good])
        all_fluxes.append(flux[good])
        all_flux_errs.append(flux_err[good])
    axes[0].set_ylabel("Detrended Flux")
    axes[1].set_xlabel("Time From Transit Midpoint [hr]")
    axes[1].set_ylabel("Residual [ppm]")
    axes[1].axhline(0.0, color=posthoc.ZERO_LINE_COLOR, lw=1.0, ls="--", zorder=0)

    residual_limit, residual_ticks = posthoc.choose_symmetric_residual_ticks(
        np.concatenate(all_residuals),
        np.concatenate(all_residual_errs),
    )
    residual_limit *= LIGHTCURVE_RESIDUAL_LIMIT_SCALE
    residual_ticks = residual_ticks * LIGHTCURVE_RESIDUAL_LIMIT_SCALE
    axes[1].set_ylim(-residual_limit, residual_limit)
    axes[1].set_yticks(residual_ticks)
    if stamp_label is not None and all_fluxes:
        posthoc.set_lightcurve_label_band(
            axes[0],
            stamp_label,
            np.concatenate(all_fluxes),
            np.concatenate(all_flux_errs),
        )
    if all_residuals:
        posthoc.add_residual_rms_stamp(axes[1], np.concatenate(all_residuals))
    fig.align_ylabels(axes)
    add_mode_legend(axes[0], [(spec, color) for spec, color, _run_dirs, _mode_dir in ordered_outputs])

    for ax in axes:
        posthoc.enforce_min_major_ticks(ax, x=True, y=True)
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.4)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False, top=False, right=False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.11, top=0.90, hspace=0.08)

    fig.savefig(outpath, dpi=posthoc.PLOT_DPI)
    plt.close(fig)


def plot_lightcurve_mode_overlay_with_sidebar(mode_outputs, instrument_index: int, outpath: Path) -> None:
    posthoc.apply_publication_style()
    fig = plt.figure(figsize=(10.8, 6.3), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[5.8, 1.55], hspace=0.08)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    axes = [ax_top, ax_bottom]
    bar_ax = fig.add_subplot(gs[:, 1])
    all_residuals = []
    all_residual_errs = []
    all_fluxes = []
    all_flux_errs = []
    stamp_label = None

    ordered_outputs = [
        (spec, _combined_mode_color_in_palette(spec, LIGHTCURVE_PALETTE, instrument_index), run_dirs, _mode_dir)
        for spec, _color, run_dirs, _mode_dir in _combined_plot_order(mode_outputs)
    ]

    for zorder, (spec, color, run_dirs, _mode_dir) in enumerate(ordered_outputs, start=1):
        run_dir = run_dirs[instrument_index]
        label, series_df = _load_whitelight_series(run_dir)
        if stamp_label is None:
            stamp_label = label
        x = _timeseries_x_hours(series_df)
        flux = np.asarray(series_df.get("detrended_flux", series_df["flux"]), dtype=float)
        flux_err = np.asarray(series_df["flux_err"], dtype=float)
        model = np.asarray(series_df.get("transit_model", series_df["bestfit_model"]), dtype=float)
        residual_ppm = np.asarray(series_df["residual_ppm"], dtype=float)
        outlier = np.asarray(series_df.get("is_outlier", np.zeros_like(x)), dtype=int).astype(bool)
        good = ~outlier
        residual_err_ppm = flux_err * 1e6

        axes[0].errorbar(
            x[good],
            flux[good],
            yerr=flux_err[good],
            fmt="o",
            ls="",
            ms=3.0,
            mfc="white",
            mec=color,
            mew=1.3,
            ecolor=color,
            elinewidth=1.0,
            capsize=0.0,
            zorder=zorder,
        )
        axes[0].plot(x, model, color=color, lw=1.5, zorder=zorder + 0.2)
        axes[1].errorbar(
            x[good],
            residual_ppm[good],
            yerr=residual_err_ppm[good],
            fmt="o",
            ls="",
            ms=2.8,
            mfc="white",
            mec=color,
            mew=1.1,
            ecolor=color,
            elinewidth=0.95,
            capsize=0.0,
            zorder=zorder,
        )
        all_residuals.append(residual_ppm[good])
        all_residual_errs.append(residual_err_ppm[good])
        all_fluxes.append(flux[good])
        all_flux_errs.append(flux_err[good])

    axes[0].set_ylabel("Detrended Flux")
    axes[1].set_xlabel("Time From Transit Midpoint [hr]")
    axes[1].set_ylabel("Residual [ppm]")
    axes[1].axhline(0.0, color=posthoc.ZERO_LINE_COLOR, lw=1.0, ls="--", zorder=0)

    residual_limit, residual_ticks = posthoc.choose_symmetric_residual_ticks(
        np.concatenate(all_residuals),
        np.concatenate(all_residual_errs),
    )
    residual_limit *= LIGHTCURVE_RESIDUAL_LIMIT_SCALE
    residual_ticks = residual_ticks * LIGHTCURVE_RESIDUAL_LIMIT_SCALE
    axes[1].set_ylim(-residual_limit, residual_limit)
    axes[1].set_yticks(residual_ticks)
    if stamp_label is not None and all_fluxes:
        posthoc.set_lightcurve_label_band(
            axes[0],
            stamp_label,
            np.concatenate(all_fluxes),
            np.concatenate(all_flux_errs),
        )
    if all_residuals:
        posthoc.add_residual_rms_stamp(axes[1], np.concatenate(all_residuals))
    fig.align_ylabels(axes)
    draw_mode_sidebar(bar_ax, [(spec, color) for spec, color, _run_dirs, _mode_dir in ordered_outputs])

    for ax in axes:
        posthoc.enforce_min_major_ticks(ax, x=True, y=True)
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.4)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False, top=False, right=False)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.11, top=0.98, hspace=0.08)

    fig.savefig(outpath, dpi=posthoc.PLOT_DPI)
    plt.close(fig)


def plot_mode_overlay(mode_segments, outpath: Path, force_log_xscale: bool | None = None) -> None:
    posthoc.apply_publication_style()
    errorbar_scale = posthoc.spectrum_errorbar_scale_for_path(outpath)
    flat_segments = [segment for _spec, _color, segments in mode_segments for segment in segments]
    min_wl = min(float(np.nanmin(np.asarray(segment.params_df["wavelength"], dtype=float))) for segment in flat_segments)
    max_wl = max(float(np.nanmax(np.asarray(segment.params_df["wavelength"], dtype=float))) for segment in flat_segments)
    use_multipanel = (
        force_log_xscale is None
        and posthoc.should_use_multipanel_combined_spectrum(
            [segment.product_label for segment in flat_segments],
            min_wl,
            max_wl,
        )
    )
    use_multipanel = False
    use_hybrid_xscale = (
        force_log_xscale is None
        and posthoc.should_use_hybrid_wavelength_xscale(
            [segment.product_label for segment in flat_segments],
            min_wl,
            max_wl,
        )
    )
    if use_multipanel:
        fig, panel_axes = posthoc.create_multipanel_spectrum_figure()
        panel_limits = {
            "mid": (2.5, 5.0),
            "short": (0.6, 2.5),
            "long": (5.0, 12.0),
        }
        ordered_mode_segments = _combined_plot_order(mode_segments)
        marker_face = posthoc.spectrum_marker_facecolor_for_path(outpath, color)
        precision_series = []
        pandexo_curves = {}
        for zorder, (spec, color, segments) in enumerate(ordered_mode_segments, start=1):
            alpha = _combined_mode_alpha(spec)
            for segment in _sorted_segments(segments):
                wl, wl_err, depth_ppm, depth_err = _extract_series(segment.params_df)
                panel_axes["full"].errorbar(
                    posthoc.transform_hybrid_wavelength(wl),
                    depth_ppm,
                    yerr=depth_err,
                    fmt="o",
                    ms=3.1,
                    mfc=marker_face,
                    mec=color,
                    mew=0.8,
                    ecolor=color,
                    elinewidth=1.4,
                    capsize=0,
                    alpha=alpha,
                    zorder=zorder,
                )
                for name, (x0, x1) in panel_limits.items():
                    mask = (wl >= x0) & (wl <= x1)
                    if not np.any(mask):
                        continue
                    panel_axes[name].errorbar(
                        wl[mask],
                        depth_ppm[mask],
                        yerr=depth_err[mask],
                        fmt="o",
                        ms=3.1,
                        mfc=marker_face,
                        mec=color,
                        mew=0.8,
                        ecolor=color,
                        elinewidth=1.4,
                        capsize=0,
                        alpha=alpha,
                        zorder=zorder,
                    )

                panel_axes["precision"].plot(
                    posthoc.transform_hybrid_wavelength(wl),
                    depth_err,
                    color=color,
                    lw=1.4,
                    alpha=alpha,
                    zorder=zorder,
                )
                precision_series.append(depth_err)
                if segment.pandexo_curve is not None and segment.product_label not in pandexo_curves:
                    pandexo_curves[segment.product_label] = (segment.pandexo_curve, wl.copy(), wl_err.copy())

        for product_label in sorted(pandexo_curves):
            pandexo_curve, wl, wl_err = pandexo_curves[product_label]
            p_wl, p_sigma = posthoc.bin_pandexo_to_data_grid(pandexo_curve, wl, wl_err)
            if p_wl.size > 0:
                panel_axes["precision"].plot(
                    posthoc.transform_hybrid_wavelength(p_wl),
                    p_sigma,
                    color="k",
                    lw=1.7,
                    ls="--",
                    zorder=8,
                )
                precision_series.append(p_sigma)

        xmin, xmax = posthoc.expanded_spectrum_xlim(min_wl, max_wl)
        posthoc.apply_hybrid_wavelength_ticks(panel_axes["full"], xmin, xmax)
        posthoc.apply_zoom_wavelength_ticks(panel_axes["mid"], 2.5, 5.0)
        posthoc.apply_zoom_wavelength_ticks(panel_axes["short"], 0.6, 2.5)
        posthoc.apply_zoom_wavelength_ticks(panel_axes["long"], 5.0, 12.0)
        posthoc.apply_hybrid_wavelength_ticks(panel_axes["precision"], xmin, xmax)

        posthoc.apply_transit_depth_ppm_axis(panel_axes["full"])
        posthoc.apply_transit_depth_ppm_axis(panel_axes["mid"])
        posthoc.apply_transit_depth_ppm_axis(panel_axes["short"])
        panel_axes["long"].set_ylabel("Transit Depth [ppm]")
        panel_axes["long"].yaxis.set_major_formatter(FuncFormatter(posthoc._format_transit_depth_ppm))
        panel_axes["precision"].set_ylabel("Precision [ppm]")
        panel_axes["precision"].set_xlabel("Wavelength [micron]")
        posthoc.set_precision_axis_limits(panel_axes["precision"], *precision_series)

        add_mode_legend(
            panel_axes["full"],
            [(spec, color) for spec, color, _segments in ordered_mode_segments],
        )

        for ax in panel_axes.values():
            posthoc.enforce_min_major_ticks(ax, x=False, y=True)
            ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.4)
            ax.tick_params(axis="both", which="minor", bottom=False, left=False, top=False, right=False)
        posthoc.apply_special_spectrum_tweaks(panel_axes["full"], outpath)

        fig.align_ylabels(list(panel_axes.values()))
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.08, top=0.96, hspace=0.16, wspace=0.20)
        fig.savefig(outpath, dpi=posthoc.PLOT_DPI)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.4),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    residual_series = []
    marker_face = posthoc.spectrum_marker_facecolor_for_path(outpath, "#000000")

    ordered_mode_segments = _combined_plot_order(mode_segments)
    fiducial_entry = next(((spec, color, segments) for spec, color, segments in ordered_mode_segments if spec.slug == "stellarinformed"), None)
    fiducial_map = {}
    if fiducial_entry is not None:
        _fid_spec, _fid_color, fid_segments = fiducial_entry
        fiducial_map = {_segment_match_key(segment): segment for segment in fid_segments}

    for zorder, (spec, color, segments) in enumerate(ordered_mode_segments, start=1):
        alpha = _combined_mode_alpha(spec)
        for segment in _sorted_segments(segments):
            wl, wl_err, depth_ppm, depth_err = _extract_series(segment.params_df)
            plot_wl = posthoc.transform_hybrid_wavelength(wl) if use_hybrid_xscale else wl
            axes[0].errorbar(
                plot_wl,
                depth_ppm,
                yerr=depth_err,
                fmt="o",
                ms=3.1,
                mfc=marker_face,
                mec=color,
                mew=0.8,
                ecolor=color,
                elinewidth=1.4,
                capsize=0,
                alpha=alpha,
                zorder=zorder,
            )
            if spec.slug != "stellarinformed" and fiducial_map:
                fid_segment = fiducial_map.get(_segment_match_key(segment))
                if fid_segment is not None:
                    fid_wl, _fid_wl_err, fid_depth_ppm, _fid_depth_err = _extract_series(fid_segment.params_df)
                    residual_ppm = np.interp(wl, fid_wl, fid_depth_ppm) - depth_ppm
                    residual_series.append(residual_ppm)
                    axes[1].plot(
                        plot_wl,
                        residual_ppm,
                        marker="o",
                        ms=3.1,
                        mfc=color,
                        mec=color,
                        mew=0.0,
                        linestyle="none",
                        alpha=alpha,
                        zorder=zorder,
                    )

    posthoc.apply_transit_depth_ppm_axis(axes[0])
    axes[1].set_xlabel("Wavelength [micron]")
    axes[1].set_ylabel("Residual [ppm]")
    axes[1].axhline(0.0, color=posthoc.ZERO_LINE_COLOR, lw=1.0, ls="--", zorder=0)
    axes[1].set_ylim(-500.0, 500.0)
    use_log_xscale = (
        posthoc.should_use_log_spectrum_xscale(
            segment.product_label for _spec, _color, segments in mode_segments for segment in segments
        )
        if force_log_xscale is None
        else force_log_xscale
    )
    custom_x_ticks = False
    if use_hybrid_xscale:
        outname = outpath.name.upper()
        if outname == "HAT-P-18_ALL_LD_MODES_COMBINED_SPECTRUM.PNG":
            xmin, xmax = 0.55, 12.5
        else:
            xmin, xmax = posthoc.expanded_spectrum_xlim(min_wl, max_wl)
        posthoc.apply_hybrid_wavelength_ticks(axes[0], xmin, xmax)
        posthoc.apply_hybrid_wavelength_ticks(axes[1], xmin, xmax)
        custom_x_ticks = True
    elif use_log_xscale:
        xmin, xmax = posthoc.expanded_spectrum_xlim(min_wl, max_wl)
        posthoc.apply_plain_log_wavelength_ticks(axes[0], xmin, xmax)
        posthoc.apply_plain_log_wavelength_ticks(axes[1], xmin, xmax)
        custom_x_ticks = True
    else:
        xmin, xmax = posthoc.expanded_spectrum_xlim(min_wl, max_wl)
        axes[1].set_xlim(xmin, xmax)
    fig.align_ylabels(axes)
    add_mode_legend(
        axes[0],
        [(spec, color) for spec, color, _segments in ordered_mode_segments],
    )

    for ax in axes:
        posthoc.enforce_min_major_ticks(ax, x=not custom_x_ticks, y=True)
        ax.tick_params(axis="both", which="major", direction="in", top=False, right=False, length=6, width=1.4)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False, top=False, right=False)
    posthoc.apply_special_spectrum_tweaks(axes[0], outpath)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.11, top=0.92, hspace=0.08)

    fig.savefig(outpath, dpi=posthoc.PLOT_DPI)
    plt.close(fig)


def plot_instrument_spectrum_mode_overlay(mode_segments, instrument_index: int, outpath: Path) -> None:
    instrument_mode_segments = [
        (spec, _combined_mode_color(spec, instrument_index), [segments[instrument_index]])
        for spec, _color, segments in mode_segments
    ]
    plot_mode_overlay(instrument_mode_segments, outpath)


def write_manifest(mode_outputs, outpath: Path) -> None:
    lines = []
    for spec, color, run_dirs, mode_dir in mode_outputs:
        lines.append(f"[{spec.display_name}]")
        lines.append(f"subdir = {mode_dir}")
        lines.append(f"color = {color}")
        for run_dir in run_dirs:
            lines.append(f"run = {run_dir.name}")
        lines.append("")
    outpath.write_text("\n".join(lines).rstrip() + "\n")


def _segment_is_miri(segment) -> bool:
    token = f"{segment.run_label}_{segment.product_label}".upper()
    return ("MIRI" in token) or ("LRS" in token)


def _exclude_miri_segments(segments):
    return [segment for segment in segments if not _segment_is_miri(segment)]


def desired_product_globs_for_run(run_name: str) -> tuple[str, ...]:
    upper_name = run_name.upper()
    if upper_name.endswith("_R100"):
        return ("*_R100",)
    if upper_name.endswith("_R50"):
        return ("*_R50",)
    if upper_name.endswith("_R300"):
        return ("*_R300",)
    if upper_name.endswith("_PIX6"):
        return ("*pix6",)
    raise ValueError(f"Could not infer the desired spectral product label for {run_name}")


def build_mode_bundle(
    spec: ModeSpec,
    *,
    output_slug: str | None = None,
    combined_stem: str | None = None,
    plot_palette: str = "orchid",
    accent_color: str | None = None,
):
    color = _mode_color(spec)
    run_dirs = [ROOT / name for name in spec.run_names]
    missing = [run_dir for run_dir in run_dirs if not run_dir.is_dir()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing expected HAT-P-18 run directories for {spec.slug}: {missing_names}")

    mode_dir = OUTPUT_ROOT / (output_slug or spec.slug)
    _clean_output_dir(mode_dir)
    combined_segments = []

    for run_dir in run_dirs:
        run_segments = posthoc.build_plot_suite(
            run_dir,
            mode_dir,
            max_waterfall_channels=posthoc.DEFAULT_MAX_WATERFALL_CHANNELS,
            palette=plot_palette,
            accent_color=accent_color,
            include_product_globs=desired_product_globs_for_run(run_dir.name),
            lightcurve_residual_scale=LIGHTCURVE_RESIDUAL_LIMIT_SCALE,
            rms_in_top_panel=False,
            lightcurve_palette=LIGHTCURVE_PALETTE,
        )
        combined_segments.extend(run_segments)

    posthoc.apply_publication_style()
    posthoc.set_plot_palette(plot_palette)
    if accent_color is not None:
        posthoc.set_explicit_accent_color(accent_color)
    combined_path = mode_dir / f"{combined_stem or spec.combined_stem}_combined_spectrum.png"
    posthoc.plot_combined_spectrum_precision(
        combined_segments,
        combined_path,
        force_multipanel=False,
    )
    niriss_nirspec_segments = _exclude_miri_segments(combined_segments)
    if niriss_nirspec_segments:
        niriss_nirspec_path = mode_dir / f"{combined_stem or spec.combined_stem}_niriss_nirspec_combined_spectrum.png"
        posthoc.plot_combined_spectrum_precision(
            niriss_nirspec_segments,
            niriss_nirspec_path,
            force_log_xscale=True,
        )
    return color, run_dirs, mode_dir, combined_segments


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_dir = OUTPUT_ROOT / "combined"
    _clean_output_dir(combined_dir)

    mode_outputs = []
    overlay_inputs = []
    for spec in MODE_SPECS:
        bundle_kwargs = {}
        if spec.slug == "stellarinformed":
            bundle_kwargs = {"plot_palette": "managua", "accent_color": None}
        else:
            bundle_kwargs = {"plot_palette": "managua", "accent_color": _mode_color(spec)}
        color, run_dirs, mode_dir, segments = build_mode_bundle(spec, **bundle_kwargs)
        mode_outputs.append((spec, color, run_dirs, mode_dir))
        overlay_inputs.append((spec, _combined_mode_color(spec), segments))
        print(f"Wrote {spec.display_name} plots to {mode_dir}")

    stellar_spec = next(spec for spec in MODE_SPECS if spec.slug == "stellarinformed")
    _acton_color, _acton_run_dirs, acton_dir, _acton_segments = build_mode_bundle(
        stellar_spec,
        output_slug="stellarinformed_acton",
        combined_stem="HAT-P-18_STELLARINFORMED_ACTON",
        plot_palette="acton",
        accent_color=None,
    )
    print(f"Wrote Stellar Informed acton alternative to {acton_dir}")

    plot_mode_overlay(overlay_inputs, combined_dir / "HAT-P-18_all_ld_modes_combined_spectrum.png")
    niriss_nirspec_overlay_inputs = [
        (spec, color, _exclude_miri_segments(segments))
        for spec, color, segments in overlay_inputs
    ]
    niriss_nirspec_overlay_inputs = [
        (spec, color, segments)
        for spec, color, segments in niriss_nirspec_overlay_inputs
        if segments
    ]
    plot_mode_overlay(
        niriss_nirspec_overlay_inputs,
        combined_dir / "HAT-P-18_all_ld_modes_niriss_nirspec_combined_spectrum.png",
        force_log_xscale=True,
    )
    for index, instrument_spec in enumerate(INSTRUMENT_OVERLAY_SPECS):
        plot_instrument_spectrum_mode_overlay(
            overlay_inputs,
            instrument_index=index,
            outpath=combined_dir / instrument_spec.spectrum_filename,
        )
    write_manifest(mode_outputs, OUTPUT_ROOT / "manifest.txt")
    write_manifest(mode_outputs, combined_dir / "manifest.txt")
    print(f"Wrote mode-comparison plots to {combined_dir}")


if __name__ == "__main__":
    main()
