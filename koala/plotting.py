
from koala.console import debug, info
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import jax.numpy as jnp
from models.common import apply_systematics, compute_transit_model_auto
from models.harmonica.core import _ALL_ODD_COEFF_SPECS
from .plotting_style import (
    DATA_MARKER_STYLE,
    FIGURE_WIDTH,
    FONT_SIZE,
    LABEL_SIZE,
    TITLE_SIZE,
    scaled_figsize,
    ERRORBAR_COLOR,
    LIGHTCURVE_COLOR,
    MODEL_COLOR,
    PLOT_DPI,
    OUTLIER_MARKER_STYLE,
    ZERO_LINE_COLOR,
    _infer_half_widths,
    accent_color_for_label,
    add_instrument_stamp,
    apply_publication_style,
    choose_symmetric_residual_ticks,
    related_accent_colors,
    save_figure,
    style_axis,
    figure_title,
)

# Use the full catalogue here so plotting can handle any saved results.
HARMONICA_ODD_HARMONICS = tuple(name for name, _ in _ALL_ODD_COEFF_SPECS)
_JAXOPLANET_EVAL_METADATA = (
    "_surface_model",
    "_stellar_spots",
    "stellar_rotation_period",
    "stellar_inclination",
    "stellar_phase",
    "_jaxoplanet_kernel",
    "_ld_profile",
    "_power2_exact",
    "_power2_quadrature_order",
    "_transit_phase_offsets",
    "_transit_phase_mask",
    "_transit_window_indices",
)


def _resolve_plot_geometry_mode(map_params, transit_params):
    param_method = transit_params.get("param_method")
    if param_method is None:
        param_method = "a_rs" if "a_rs" in map_params else "duration"
    transit_engine = transit_params.get("transit_engine")
    if transit_engine is None:
        has_power2_ld = all((key in map_params for key in ('c_ld', 'alpha_ld')))
        has_quadratic_ld = all((key in map_params for key in ('u1_ld', 'u2_ld')))
        transit_engine = (
            "harmonica"
            if has_power2_ld or has_quadratic_ld
            else "jaxoplanet"
        )
    return transit_engine, param_method


def _single_curve_transit_signal(t, map_params, transit_params, idx):
    periods = np.atleast_1d(transit_params["period"])
    bs = np.atleast_1d(map_params["b"])
    t0s = np.atleast_1d(map_params["t0"])
    if t0s.ndim == 2:
        t0s = t0s[idx]
    rors_i_all_planets = np.atleast_1d(map_params["rors"][idx])
    transit_engine, param_method = _resolve_plot_geometry_mode(map_params, transit_params)

    params = {
        "period": jnp.asarray(periods),
        "t0": jnp.asarray(t0s),
        "b": jnp.asarray(bs),
        "rors": jnp.asarray(rors_i_all_planets),
    }

    # Harmonica always consumes physical a/Rs geometry, even when the white-light
    # fit was parameterized by duration.  Jaxoplanet, by contrast, must retain
    # the same duration-vs-Keplerian choice as the sampled model.
    if transit_engine == "harmonica" or "a_rs" in map_params:
        params["a_rs"] = jnp.asarray(np.atleast_1d(map_params["a_rs"]))
        params["ecc"] = jnp.asarray(
            np.atleast_1d(map_params.get("ecc", np.zeros_like(periods)))
        )
        params["omega"] = jnp.asarray(
            np.atleast_1d(map_params.get("omega", np.zeros_like(periods)))
        )
    else:
        params["duration"] = jnp.asarray(np.atleast_1d(map_params["duration"]))

    if transit_engine == "harmonica":
        if all(key in map_params for key in ("c_ld", "alpha_ld")):
            params["c_ld"] = jnp.asarray(map_params["c_ld"][idx])
            params["alpha_ld"] = jnp.asarray(map_params["alpha_ld"][idx])
        elif all(key in map_params for key in ("u1_ld", "u2_ld")):
            params["u1_ld"] = jnp.asarray(map_params["u1_ld"][idx])
            params["u2_ld"] = jnp.asarray(map_params["u2_ld"][idx])
        else:
            raise ValueError('Harmonica plotting requires either c_ld/alpha_ld or u1_ld/u2_ld.')
        for harmonic_name in HARMONICA_ODD_HARMONICS:
            if harmonic_name in map_params:
                params[harmonic_name] = jnp.asarray(
                    np.atleast_1d(map_params[harmonic_name][idx])
                )
    else:
        params["u"] = jnp.asarray(map_params["u"][idx])
        for name in ('c1', 'c2'):
            if name in map_params:
                params[name] = jnp.asarray(map_params[name][idx])
        for name in _JAXOPLANET_EVAL_METADATA:
            if name in map_params:
                params[name] = map_params[name]
            elif name in transit_params:
                params[name] = transit_params[name]
        for name in (
            "eclipse_depth", "phase_c1", "phase_d1", "stellar_spot_contrast",
        ):
            if name in map_params:
                params[name] = jnp.asarray(map_params[name][idx])
        basis = map_params.get("_surface_basis")
        if basis is not None:
            basis_time = map_params.get("_surface_basis_time")
            if basis_time is None or not np.array_equal(np.asarray(basis_time), np.asarray(t)):
                raise ValueError("The plotting surface basis must match the plotted time grid.")
            if basis.baseline.shape[-1] != len(t):
                raise ValueError("The plotting surface basis has the wrong number of cadences.")
            params["_surface_basis"] = type(basis)(*(
                None if field is None else (
                    field[idx] if basis.baseline.ndim > 1 else field
                )
                for field in basis
            ))

    return np.asarray(compute_transit_model_auto(params, jnp.asarray(t)))


def _broadcast_error(error, size):
    values = np.asarray(error, dtype=float)
    if values.ndim == 0:
        return np.full(size, float(values))
    values = np.ravel(values)
    if values.size == 1:
        return np.full(size, float(values[0]))
    if values.size != size:
        raise ValueError(f"Expected one uncertainty or {size} values, got {values.size}.")
    return values


def _time_from_midtransit_hours(time, t0_reference=None):
    time = np.asarray(time, dtype=float)
    if t0_reference is None:
        reference = float(np.nanmedian(time))
    else:
        reference = float(np.ravel(np.asarray(t0_reference, dtype=float))[0])
    return (time - reference) * 24.0


def _first_value(values, index=0):
    array = np.asarray(values)
    if array.ndim == 0:
        return float(array)
    return float(array.ravel()[index])


def _format_pm(value, err_low=None, err_high=None, fmt=".6f"):
    value = float(value)
    if err_low is None or err_high is None:
        return format(value, fmt)
    return (
        f"{format(value, fmt)} "
        f"-{format(float(err_low), fmt)} +{format(float(err_high), fmt)}"
    )


def _white_light_residual_panel(ax, x, residual_ppm, error_ppm, outlier, accent):
    """Draw residuals and choose a symmetric ppm scale."""
    good = ~outlier
    if np.any(outlier):
        ax.plot(x[outlier], residual_ppm[outlier], zorder=1, **OUTLIER_MARKER_STYLE)
    ax.errorbar(
        x[good], residual_ppm[good], yerr=error_ppm[good], color=accent, ecolor=accent,
        zorder=2, **DATA_MARKER_STYLE,
    )
    ax.axhline(0.0, color=ZERO_LINE_COLOR, lw=1.7, ls="--", zorder=3)
    ax.set_xlabel("Time from Mid-Transit [hr]")
    ax.set_ylabel("Residuals [ppm]")
    limit, ticks = choose_symmetric_residual_ticks(residual_ppm[good], error_ppm[good])
    ax.set_ylim(-limit, limit)
    ax.set_yticks(ticks)
    return ticks


def plot_whitelight_curve(
    time, flux, flux_err, bestfit_model, filename, instrument_label=None, t0_reference=None,
    outlier_mask=None, flux_label='Normalized Flux',
):
    """Write the house-style white-light curve and residual panels."""
    apply_publication_style()
    time = np.asarray(time, dtype=float)
    x = _time_from_midtransit_hours(time, t0_reference)
    flux = np.asarray(flux, dtype=float)
    model = np.asarray(bestfit_model, dtype=float)
    error = _broadcast_error(flux_err, time.size)
    if outlier_mask is None:
        outlier = np.zeros(time.size, dtype=bool)
    else:
        outlier = np.asarray(outlier_mask, dtype=bool)
        if outlier.shape != time.shape:
            raise ValueError("outlier_mask must match the white-light time axis.")
    good = ~outlier
    residual_ppm = (flux - model) * 1e6
    error_ppm = error * 1e6
    accent = accent_color_for_label(instrument_label or str(filename))

    fig, axes = plt.subplots(
        2, 1, figsize=scaled_figsize(8.8, 6.2), sharex=True, gridspec_kw={'height_ratios': [3, 1]},
    )
    if np.any(outlier):
        axes[0].plot(x[outlier], flux[outlier], zorder=1, **OUTLIER_MARKER_STYLE)
    axes[0].plot(
        x[good],
        flux[good],
        marker=DATA_MARKER_STYLE["fmt"],
        color=accent,
        zorder=2,
        **{key: value for key, value in DATA_MARKER_STYLE.items() if key not in {"fmt", "elinewidth", "capsize"}},
    )
    axes[0].plot(x, model, color=MODEL_COLOR, lw=1.9, zorder=3)
    axes[0].set_ylabel(flux_label)
    main_ticks = axes[0].get_yticks()
    stamp_y = main_ticks[1] if len(main_ticks) >= 2 else np.nanmin(flux[good])
    add_instrument_stamp(
        axes[0], instrument_label or str(filename), fontsize=TITLE_SIZE, y_data=stamp_y,
    )

    ticks = _white_light_residual_panel(axes[1], x, residual_ppm, error_ppm, outlier, accent)
    for ax in axes:
        style_axis(ax)
    axes[1].set_yticks(ticks)
    fig.align_ylabels(axes)
    save_figure(fig, filename)
    return fig


def plot_whitelight_residuals(
    time, residual, flux_err, filename, instrument_label=None, t0_reference=None,
    outlier_mask=None,
):
    """Write the retained standalone white-light residual diagnostic."""
    apply_publication_style()
    time = np.asarray(time, dtype=float)
    x = _time_from_midtransit_hours(time, t0_reference)
    residual_ppm = np.asarray(residual, dtype=float) * 1e6
    error_ppm = _broadcast_error(flux_err, time.size) * 1e6
    outlier = (
        np.zeros(time.size, dtype=bool)
        if outlier_mask is None
        else np.asarray(outlier_mask, dtype=bool)
    )
    good = ~outlier
    accent = accent_color_for_label(instrument_label or str(filename))
    fig, ax = plt.subplots(figsize=scaled_figsize(8.8, 4.0))
    ticks = _white_light_residual_panel(ax, x, residual_ppm, error_ppm, outlier, accent)
    add_instrument_stamp(ax, instrument_label or str(filename))
    style_axis(ax)
    ax.set_yticks(ticks)
    save_figure(fig, filename)
    return fig


def plot_whitelight_summary(params, filename, instrument_label=None):
    """Write a compact one-annotation white-light parameter summary."""
    apply_publication_style()
    rors = np.atleast_1d(np.asarray(params["rors"], dtype=float))
    lines = [str(instrument_label or "White-light fit")]
    period = np.atleast_1d(np.asarray(params.get("period", np.nan), dtype=float))
    for index in range(rors.size):
        prefix = f"Planet {index + 1}: " if rors.size > 1 else ""
        lines.append(f"{prefix}P = {_format_pm(period[min(index, period.size - 1)], fmt='.8f')} d")
        for name, label, scale, precision, unit in (
            ('duration', 'Duration', 1.0, '.6f', ' d'),
            ('t0', 't0', 1.0, '.6f', ''),
            ('b', 'b', 1.0, '.5f', ''),
            ('rors', 'Rp/R*', 1.0, '.6f', ''),
            ('depths', 'Depth', 1e6, '.1f', ' ppm'),
        ):
            values = params.get(name, rors**2) if name == 'depths' else params[name]
            formatted = _format_pm(
                scale * _first_value(values, index),
                scale * _first_value(params.get(f'{name}_err_low', 0.0), index),
                scale * _first_value(params.get(f'{name}_err_high', 0.0), index), precision,
            )
            lines.append(f"{label} = {formatted}{unit}")
        if index != rors.size - 1:
            lines.append("")

    fig_height = max(3.4, 2.8 * rors.size)
    fig, ax = plt.subplots(figsize=scaled_figsize(7.2, fig_height))
    ax.axis("off")
    ax.text(
        0.02, 0.96, '\n'.join(lines), ha='left', va='top', transform=ax.transAxes, fontsize=TITLE_SIZE,
        linespacing=1.45,
    )
    save_figure(fig, filename)
    return fig


def plot_spectrum_precision(
    wavelengths, depth_ppm, depth_err_ppm, filename, instrument_label=None,
    wavelength_err=None, asymmetric_depth_err=None,
):
    """Write a transmission spectrum with its precision panel underneath."""
    apply_publication_style()
    wavelengths = np.asarray(wavelengths, dtype=float)
    depth_ppm = np.asarray(depth_ppm, dtype=float)
    precision = np.asarray(depth_err_ppm, dtype=float)
    if wavelength_err is None:
        wavelength_err = _infer_half_widths(wavelengths)
    else:
        wavelength_err = np.asarray(wavelength_err, dtype=float)
    yerr = precision if asymmetric_depth_err is None else np.asarray(asymmetric_depth_err, dtype=float)
    accent = accent_color_for_label(instrument_label or str(filename))

    fig, axes = plt.subplots(
        2, 1, figsize=scaled_figsize(8.8, 6.2), sharex=True, gridspec_kw={'height_ratios': [3, 1]},
    )
    axes[0].errorbar(
        wavelengths, depth_ppm, xerr=wavelength_err, yerr=yerr, color=accent, ecolor=accent,
        zorder=2, **DATA_MARKER_STYLE,
    )
    axes[0].set_ylabel("Transit Depth [ppm]")
    add_instrument_stamp(axes[0], instrument_label or str(filename), fontsize=TITLE_SIZE)
    axes[1].plot(wavelengths, precision, color=accent, lw=1.9)
    axes[1].set_xlabel("Wavelength [$\\mu$m]")
    axes[1].set_ylabel("Precision [ppm]")
    if wavelengths.size:
        padding = max(0.02, 0.5 * float(np.nanmedian(wavelength_err)))
        axes[1].set_xlim(np.nanmin(wavelengths) - padding, np.nanmax(wavelengths) + padding)
    for ax in axes:
        style_axis(ax)
    fig.align_ylabels(axes)
    save_figure(fig, filename)
    return fig


def plot_noise_binning_from_csv(noise_df, filename, instrument_label=None):
    """Write the house-style temporal noise-binning diagnostic."""
    apply_publication_style()
    bins = np.asarray(noise_df["bin_size_points"], dtype=float)
    measured = np.asarray(noise_df["measured_rms"], dtype=float)
    p16 = np.asarray(noise_df["measured_rms_p16"], dtype=float)
    p84 = np.asarray(noise_df["measured_rms_p84"], dtype=float)
    expected = np.asarray(noise_df["expected_white_rms"], dtype=float)
    accent = accent_color_for_label(instrument_label or str(filename))

    fig, ax = plt.subplots(figsize=scaled_figsize(6.8, 4.8))
    ax.loglog(bins, expected, color=ZERO_LINE_COLOR, lw=1.7, ls="--")
    ax.fill_between(bins, p16, p84, color=accent, alpha=0.18)
    ax.loglog(
        bins, measured, color=accent, lw=1.8, marker='o', ms=3.4, mfc='white', mec=accent,
        mew=1.5,
    )
    ax.set_xlabel("Bin Size [points]")
    ax.set_ylabel("RMS [ppm]")
    add_instrument_stamp(ax, instrument_label or str(filename), fontsize=TITLE_SIZE)
    style_axis(ax, x_locator=False, y_locator=False)
    save_figure(fig, filename)
    return fig

def _simple_channel_trend(t, map_params, index, detrend_type):
    """Evaluate the trend families used by the standalone MAP grids."""
    t = np.asarray(t, dtype=float)
    shifted = t - np.nanmin(t)
    if detrend_type == "none":
        return np.ones_like(t)
    c = _first_value(map_params.get("c", 1.0), index)
    v = _first_value(map_params.get("v", 0.0), index)
    trend = c + v * shifted
    if detrend_type in {"quadratic", "quadratic_spot", "quadratic+spot"}:
        trend = trend + _first_value(map_params.get("v2", 0.0), index) * shifted**2
    if detrend_type in {"explinear", "explinear_spectroscopic"}:
        amplitude = _first_value(map_params.get("A", 0.0), index)
        tau = max(abs(_first_value(map_params.get("tau", 1.0), index)), 1e-12)
        trend = trend + amplitude * np.exp(-shifted / tau)
    if "spot" in detrend_type:
        for suffix in ("", "2") if "2spot" in detrend_type else ("",):
            amplitude = _first_value(map_params.get(f"spot_amp{suffix}", 0.0), index)
            centre = _first_value(map_params.get(f"spot_mu{suffix}", 0.0), index)
            width = max(abs(_first_value(map_params.get(f"spot_sigma{suffix}", 1.0), index)), 1e-12)
            trend = trend + amplitude * np.exp(-0.5 * ((t - centre) / width) ** 2)
    return trend


def _map_full_models(t, map_params, transit_params, count, detrend_type):
    return np.asarray(
        [
            apply_systematics(
                _single_curve_transit_signal(t, map_params, transit_params, index),
                _simple_channel_trend(t, map_params, index, detrend_type),
            )
            for index in range(count)
        ]
    )


def _channel_error(jitter, index, size):
    values = np.asarray(jitter)
    if values.ndim <= 1:
        return _broadcast_error(values[index] if values.size > 1 else values, size)
    return _broadcast_error(values[index], size)


def plot_map_fits(
    t, indiv_y, jitter, wavelengths, map_params, transit_params, filename, ncols=3,
    detrend_type='linear',
):
    """Plot the per-channel MAP fits in a clean, shared-axis grid."""
    apply_publication_style()
    t = np.asarray(t, dtype=float)
    data = np.asarray(indiv_y, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    count = wavelengths.size
    nrows = int(np.ceil(count / ncols))
    x = _time_from_midtransit_hours(t, map_params.get("t0"))
    models = _map_full_models(t, map_params, transit_params, count, detrend_type)
    accent = accent_color_for_label(str(filename))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=scaled_figsize(8.8, max(3.0, 2.55 * nrows)), sharex=True, sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for index in range(count):
        ax = flat_axes[index]
        ax.errorbar(
            x, data[index], yerr=_channel_error(jitter, index, t.size), color=accent,
            ecolor=accent, zorder=2, **DATA_MARKER_STYLE,
        )
        ax.plot(x, models[index], color=MODEL_COLOR, lw=1.9, zorder=3)
        style_axis(ax)
        ax.label_outer()
    for ax in flat_axes[count:]:
        ax.set_visible(False)
    add_instrument_stamp(flat_axes[0], str(filename), fontsize=TITLE_SIZE)
    fig.supxlabel("Time from Mid-Transit [hr]")
    fig.supylabel("Normalized Flux")
    save_figure(fig, filename)
    return fig


def plot_map_residuals(
    t, indiv_y, jitter, wavelengths, map_params, transit_params, filename, ncols=3,
    detrend_type='linear',
):
    """Plot per-channel residuals in ppm with a shared symmetric scale."""
    apply_publication_style()
    t = np.asarray(t, dtype=float)
    data = np.asarray(indiv_y, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    count = wavelengths.size
    nrows = int(np.ceil(count / ncols))
    x = _time_from_midtransit_hours(t, map_params.get("t0"))
    models = _map_full_models(t, map_params, transit_params, count, detrend_type)
    residuals = (data - models) * 1e6
    errors = np.asarray([_channel_error(jitter, index, t.size) for index in range(count)]) * 1e6
    limit, ticks = choose_symmetric_residual_ticks(residuals, errors)
    accent = accent_color_for_label(str(filename))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=scaled_figsize(8.8, max(3.0, 2.35 * nrows)), sharex=True, sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for index in range(count):
        ax = flat_axes[index]
        ax.errorbar(
            x, residuals[index], yerr=errors[index], color=accent, ecolor=accent, zorder=2,
            **DATA_MARKER_STYLE,
        )
        ax.axhline(0.0, color=ZERO_LINE_COLOR, lw=1.7, ls="--", zorder=3)
        ax.set_ylim(-limit, limit)
        ax.set_yticks(ticks)
        style_axis(ax)
        ax.set_yticks(ticks)
        ax.label_outer()
    for ax in flat_axes[count:]:
        ax.set_visible(False)
    add_instrument_stamp(flat_axes[0], str(filename), fontsize=TITLE_SIZE)
    fig.supxlabel("Time from Mid-Transit [hr]")
    fig.supylabel("Residuals [ppm]")
    save_figure(fig, filename)
    return fig


def plot_transmission_spectrum(wavelengths, rors_posterior, filename):
    """One PNG with a depth panel for each fitted planet."""
    from .diagnostic_figures import spectrum
    filename = str(filename)
    if not filename.endswith('.png'):
        filename += '.png'
    return spectrum(wavelengths, rors_posterior, filename, label=Path(filename).stem)


def _summary_channel_trend(
    t, map_params, index, detrend_type, gp_trend=None, spot_trend=None, spot_trend2=None,
    jump_trend=None, exp_trend=None,
):
    t = np.asarray(t, dtype=float)
    shifted = t - np.nanmin(t)
    c = _first_value(map_params.get("c", 1.0), index)
    if "gp_spectroscopic" in detrend_type:
        trend = np.full_like(t, c)
        if "linear" in detrend_type:
            trend += _first_value(map_params.get("v", 0.0), index) * shifted
        if "quadratic" in detrend_type:
            trend += _first_value(map_params.get("v2", 0.0), index) * shifted**2
        return trend + _first_value(map_params["A_gp"], index) * np.asarray(gp_trend)
    if set(detrend_type.split("+")) == {"quadratic", "2spot_spectroscopic"}:
        return (c + _first_value(map_params.get("v", 0.0), index) * shifted
                + _first_value(map_params.get("v2", 0.0), index) * shifted**2
                + _first_value(map_params["A_spot"], index) * np.asarray(spot_trend)
                + _first_value(map_params["A_spot2"], index) * np.asarray(spot_trend2))
    if detrend_type == "2spot_spectroscopic":
        return (
            c
            + _first_value(map_params["A_spot"], index) * np.asarray(spot_trend)
            + _first_value(map_params["A_spot2"], index) * np.asarray(spot_trend2)
        )
    if "spot_spectroscopic" in detrend_type:
        trend = c + _first_value(map_params["A_spot"], index) * np.asarray(spot_trend)
        if "linear" in detrend_type:
            trend += _first_value(map_params.get("v", 0.0), index) * shifted
        if "quadratic" in detrend_type:
            trend += _first_value(map_params.get("v2", 0.0), index) * shifted**2
        if "explinear" in detrend_type and exp_trend is not None:
            trend += _first_value(map_params["A"], index) * np.asarray(exp_trend)
        if "linear_discontinuity" in detrend_type:
            trend += _first_value(map_params["A_jump"], index) * np.asarray(jump_trend)
        return trend
    if detrend_type == "explinear_spectroscopic":
        return (
            c
            + _first_value(map_params.get("v", 0.0), index) * shifted
            + _first_value(map_params["A"], index) * np.asarray(exp_trend)
        )
    if detrend_type == "linear_discontinuity_spectroscopic":
        return (
            c
            + _first_value(map_params.get("v", 0.0), index) * shifted
            + _first_value(map_params["A_jump"], index) * np.asarray(jump_trend)
        )
    if detrend_type == "linear_discontinuity":
        return (
            c
            + _first_value(map_params.get("v", 0.0), index) * shifted
            + np.where(
                t > _first_value(map_params['t_jump'], index),
                _first_value(map_params['jump'], index), 0.0,
            )
        )
    return _simple_channel_trend(t, map_params, index, detrend_type)


def plot_wavelength_offset_summary(
    t, indiv_y, jitter, wavelengths, map_params, transit_params, filename,
    detrend_type='linear', use_hours=True, residual_scale=2.0, gp_trend=None, spot_trend=None,
    spot_trend2=None, jump_trend=None, exp_trend=None,
):
    """Write the wavelength-offset light-curve and residual summary."""
    apply_publication_style()
    t = np.asarray(t, dtype=float)
    data = np.asarray(indiv_y, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    count = data.shape[0]
    if count > 10:
        targets = np.linspace(np.nanmin(wavelengths), np.nanmax(wavelengths), 10)
        indices = np.unique([np.nanargmin(np.abs(wavelengths - value)) for value in targets])
    else:
        indices = np.arange(count)
    indices = indices[np.argsort(wavelengths[indices])]
    selected = len(indices)

    t0 = _first_value(map_params["t0"])
    scale = 24.0 if use_hours else 1.0
    x = (t - t0) * scale
    unit = "hr" if use_hours else "d"
    selected_rors = np.asarray(map_params["rors"])[indices]
    spacing_rors = selected_rors if selected_rors.ndim == 1 else selected_rors[:, 0]
    depths = np.asarray(spacing_rors, dtype=float) ** 2
    depth_median = max(float(np.nanmedian(depths)), 1e-5)
    depth_max = max(float(np.nanmax(depths)), depth_median)
    step = 0.5 * depth_median
    offsets = np.arange(selected) * step
    colors = related_accent_colors(str(filename), selected)

    fig, (ax_data, ax_residual) = plt.subplots(
        1, 2, figsize=scaled_figsize(10.8, max(5.2, 0.58 * selected + 1.0)),
        gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.08}, sharey=True,
    )
    marker_style = {
        "linestyle": "none",
        "marker": "o",
        "ms": 2.8,
        "mfc": "white",
        "mew": 1.1,
        "alpha": 0.85,
        "rasterized": True,
    }
    for position, index in enumerate(indices):
        offset = offsets[position]
        transit = _single_curve_transit_signal(t, map_params, transit_params, index)
        trend = _summary_channel_trend(
            t, map_params, index, detrend_type, gp_trend=gp_trend, spot_trend=spot_trend,
            spot_trend2=spot_trend2, jump_trend=jump_trend, exp_trend=exp_trend,
        )
        model = apply_systematics(transit, trend)
        residual = data[index] - model
        color = colors[position]
        ax_data.plot(x, data[index] - offset, color=color, mec=color, **marker_style)
        ax_data.plot(x, model - offset, color=MODEL_COLOR, lw=1.4, zorder=3)
        baseline = 1.0 - offset
        ax_residual.plot(
            x, baseline + residual_scale * residual, color=color, mec=color, **marker_style,
        )
        ax_residual.axhline(baseline, color=ZERO_LINE_COLOR, ls='--', lw=1.0, zorder=0)

    top = 1.0 + 0.15 * depth_max
    bottom = np.nanmin(1.0 - offsets - depths) - 0.10 * depth_max
    ax_data.set_ylim(bottom, top)
    for ax in (ax_data, ax_residual):
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        ax.set_xlabel(f"Time from Mid-Transit [{unit}]")
        style_axis(ax)
    ax_data.set_ylabel("Normalized Flux + Offset")
    ax_residual.tick_params(labelleft=False)
    add_instrument_stamp(ax_data, str(filename), fontsize=TITLE_SIZE)
    fig.align_ylabels((ax_data, ax_residual))
    save_figure(fig, filename)
    return fig


def _plot_limb_panel(ax, wavelength, wavelength_err, median, err_lo, err_hi, color, label=None):
    ax.plot(wavelength, median, color=color, lw=1.9, zorder=2)
    ax.errorbar(
        wavelength, median, xerr=wavelength_err, yerr=np.vstack((err_lo, err_hi)), color=color,
        ecolor=ERRORBAR_COLOR, label=label, zorder=3, **DATA_MARKER_STYLE,
    )


def plot_harmonica_limb_spectra(limb_df, filename, instrument_label=None):
    """Write the three-panel neutral-index Harmonica limb spectrum figure."""
    apply_publication_style()
    wavelength = np.asarray(limb_df["wavelength"], dtype=float)
    wavelength_err = np.asarray(
        limb_df.get("wavelength_err", _infer_half_widths(wavelength)), dtype=float
    )
    color = accent_color_for_label(instrument_label or str(filename))
    fig = plt.figure(figsize=scaled_figsize(8.8, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.23, wspace=0.10)
    ax_total = fig.add_subplot(grid[0, :])
    ax_one = fig.add_subplot(grid[1, 0])
    ax_two = fig.add_subplot(grid[1, 1], sharey=ax_one)

    for ax, name, label in (
        (ax_total, 'depth_total_area', 'Limb-averaged spectrum'),
        (ax_one, 'depth_one', None),
        (ax_two, 'depth_two', None),
    ):
        _plot_limb_panel(
            ax, wavelength, wavelength_err,
            *(np.asarray(limb_df[f'{name}_{stat}'], dtype=float)
              for stat in ('median', 'err_lo', 'err_hi')),
            color, label=label,
        )
    figure_title(ax_total, instrument_label or str(filename), "Limb depths")
    ax_total.set_ylabel("Transit Depth [ppm]")
    ax_one.set_ylabel("Transit Depth [ppm]")
    ax_one.set_xlabel("Wavelength [$\\mu$m]")
    ax_two.set_xlabel("Wavelength [$\\mu$m]")
    ax_one.set_title("Terminator One", fontsize=TITLE_SIZE, pad=9)
    ax_two.set_title("Terminator Two", fontsize=TITLE_SIZE, pad=9)
    for ax, tag in zip((ax_total, ax_one, ax_two), ("a", "b", "c")):
        ax.text(
            0.025, 0.95, tag, transform=ax.transAxes, ha='left', va='top', fontweight='bold',
            fontsize=TITLE_SIZE,
        )
        style_axis(ax)
    one_low = np.asarray(limb_df["depth_one_median"], dtype=float) - np.asarray(
        limb_df["depth_one_err_lo"], dtype=float
    )
    one_high = np.asarray(limb_df["depth_one_median"], dtype=float) + np.asarray(
        limb_df["depth_one_err_hi"], dtype=float
    )
    two_low = np.asarray(limb_df["depth_two_median"], dtype=float) - np.asarray(
        limb_df["depth_two_err_lo"], dtype=float
    )
    two_high = np.asarray(limb_df["depth_two_median"], dtype=float) + np.asarray(
        limb_df["depth_two_err_hi"], dtype=float
    )
    low = float(np.nanmin(np.concatenate((one_low, two_low))))
    high = float(np.nanmax(np.concatenate((one_high, two_high))))
    padding = 0.08 * (high - low) if high > low else max(1.0, 0.02 * abs(high))
    ax_one.set_ylim(low - padding, high + padding)
    ax_two.tick_params(labelleft=False)
    ax_total.legend(loc="best", frameon=False)
    fig.align_ylabels((ax_total, ax_one))
    save_figure(fig, filename)
    return fig


def plot_harmonica_transmission_strings(
    theta, radius_curves, reference_radius, filename, instrument_label=None,
):
    """Plot wavelength-channel transmission strings in the house style."""
    apply_publication_style()
    theta = np.asarray(theta, dtype=float)
    curves = np.atleast_2d(np.asarray(radius_curves, dtype=float))
    color = accent_color_for_label(instrument_label or str(filename))
    fig, ax = plt.subplots(figsize=scaled_figsize(8.8, 7.6))
    for curve in curves:
        ax.plot(
            curve * np.cos(theta), curve * np.sin(theta), color='0.78', lw=0.9, alpha=0.6,
            zorder=1,
        )
    median_curve = np.nanmedian(curves, axis=0)
    ax.plot(
        median_curve * np.cos(theta), median_curve * np.sin(theta), color=color, lw=1.9,
        label='Median transmission string', zorder=3,
    )
    reference = float(np.nanmedian(np.asarray(reference_radius, dtype=float)))
    ax.plot(
        reference * np.cos(theta), reference * np.sin(theta), color=ZERO_LINE_COLOR, ls='--',
        lw=1.4, label='Reference circle', zorder=2,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / stellar radii")
    ax.set_ylabel("y / stellar radii")
    figure_title(ax, instrument_label or str(filename), "Limb profiles")
    ax.legend(loc="lower left", frameon=False)
    style_axis(ax)
    save_figure(fig, filename)
    return fig


def plot_harmonica_transmission_posterior(
    theta, posterior_radius_curves, median_radius_curve, reference_radius, filename,
    instrument_label=None,
):
    """Plot posterior transmission strings with one coloured median curve."""
    apply_publication_style()
    theta = np.asarray(theta, dtype=float)
    curves = np.atleast_2d(np.asarray(posterior_radius_curves, dtype=float))
    median_curve = np.asarray(median_radius_curve, dtype=float)
    color = accent_color_for_label(instrument_label or str(filename))
    fig, ax = plt.subplots(figsize=scaled_figsize(8.8, 7.6))
    for curve in curves:
        ax.plot(
            curve * np.cos(theta), curve * np.sin(theta), color='0.72', lw=0.8, alpha=0.08,
            zorder=1,
        )
    ax.plot(
        median_curve * np.cos(theta), median_curve * np.sin(theta), color=color, lw=1.9,
        label='Median transmission string', zorder=3,
    )
    reference = float(reference_radius)
    ax.plot(
        reference * np.cos(theta), reference * np.sin(theta), color=ZERO_LINE_COLOR, ls='--',
        lw=1.4, label='Reference circle', zorder=2,
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / stellar radii")
    ax.set_ylabel("y / stellar radii")
    figure_title(ax, instrument_label or str(filename), "Limb outline")
    ax.legend(loc="lower left", frameon=False)
    style_axis(ax)
    save_figure(fig, filename)
    return fig


_CORNER_LABELS = {
    "period": r"$P$ [d]",
    "t0": r"$t_0$ [d]",
    "rors": r"$R_p/R_\star$",
    "b": r"$b$",
    "duration": r"$T_{14}$ [d]",
    "a_rs": r"$a/R_\star$",
    "c1": r"$c_1$",
    "c2": r"$c_2$",
    "u1": r"$u_1$",
    "u2": r"$u_2$",
    "c": r"$c$",
    "v": r"$v$",
    "v2": r"$v_2$",
    "v3": r"$v_3$",
    "v4": r"$v_4$",
    "A": r"$A$",
    "tau": r"$\tau$ [d]",
    "spot_amp": r"$A_\mathrm{spot}$",
    "spot_mu": r"$t_\mathrm{spot}$ [d]",
    "spot_sigma": r"$\sigma_\mathrm{spot}$ [d]",
    "spot_amp2": r"$A_\mathrm{spot,2}$",
    "spot_mu2": r"$t_\mathrm{spot,2}$ [d]",
    "spot_sigma2": r"$\sigma_\mathrm{spot,2}$ [d]",
    "t_jump": r"$t_\mathrm{jump}$ [d]",
    "jump": r"jump",
    "width": r"width [d]",
    "error": r"$\sigma_\mathrm{jit}$",
    "GP_log_sigma": r"$\ln\sigma_\mathrm{GP}$",
    "GP_log_rho": r"$\ln\rho_\mathrm{GP}$",
}
_CORNER_ORDER = (
    "t0", "rors", "b", "duration", "a_rs", "period", "c1", "c2", "u1", "u2", "c", "v",
    "v2", "v3", "v4", "A", "tau", "spot_amp", "spot_mu", "spot_sigma",
    "spot_amp2", "spot_mu2", "spot_sigma2", "t_jump", "jump", "width",
    "error", "GP_log_sigma", "GP_log_rho",
)
# Surface-model sites shown when present (labels fall back to the name).
_CORNER_SURFACE = (
    "eclipse_depth", "phase_c1", "phase_d1",
    "stellar_spot_contrast", "stellar_rotation_period",
)
# Only physical, directly interpretable sites are plotted. Latent
# reparameterisations (cos i, delta, log-jitter, u+/u- coordinates, log tau)
# and deterministic duplicates (depths, inclination, width in minutes) are
# left out; a_rs is shown only when duration is not. The trend, jitter and GP
# sites are left out too, so the figure stays legible at print size; their
# values are in the white-light parameter table.
_CORNER_NUISANCE = {
    "c", "v", "v2", "v3", "v4", "A", "tau", "spot_amp", "spot_mu", "spot_sigma",
    "spot_amp2", "spot_mu2", "spot_sigma2", "t_jump", "jump", "width",
    "error", "GP_log_sigma", "GP_log_rho",
}
_CORNER_KEEP = (set(_CORNER_ORDER) | set(_CORNER_SURFACE)) - _CORNER_NUISANCE


def _corner_columns(samples):
    """Return (columns, labels) of scalar posterior sites for a corner plot."""
    def stem_of(name):
        if "_" in name and name.rsplit("_", 1)[1].isdigit():
            return name.rsplit("_", 1)[0]
        return name

    has_duration = any(stem_of(n) == "duration" for n in samples)
    columns = {}
    for name, value in samples.items():
        stem = stem_of(name)
        if name.startswith("_") or (stem not in _CORNER_KEEP and name != "u"):
            continue
        if stem == "a_rs" and has_duration:
            continue
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            columns[name] = arr
        elif arr.ndim == 2 and arr.shape[1] <= 4 and arr.shape[0] > arr.shape[1]:
            # Small vector sites (e.g. quadratic ``u``) become one column each.
            base = "u" if name == "u" else name
            for k in range(arr.shape[1]):
                columns[f"{base}{k + 1}"] = arr[:, k]

    def rank(name):
        stem = name.rsplit("_", 1)[0] if name[-1].isdigit() and "_" in name else name
        for key in (name, stem):
            if key in _CORNER_ORDER:
                return _CORNER_ORDER.index(key)
        return len(_CORNER_ORDER)

    multi_planet = sum(n.startswith("rors_") for n in columns) > 1
    out_cols, labels = [], []
    for name in sorted(columns, key=lambda n: (rank(n), n)):
        arr = columns[name]
        if not np.all(np.isfinite(arr)) or np.all(arr == arr[0]):
            continue
        stem, suffix = name, ""
        if "_" in name and name.rsplit("_", 1)[1].isdigit():
            stem, suffix = name.rsplit("_", 1)
        label = _CORNER_LABELS.get(stem, _CORNER_LABELS.get(name, name.replace("_", " ")))
        if suffix and multi_planet and stem in {"t0", "rors"}:
            planet = int(suffix) + 1
            label = (r"$t_{0,%d}$ [d]" % planet if stem == "t0"
                     else r"$R_{p,%d}/R_\star$" % planet)
        if stem == "t0":
            ref = float(np.nanmedian(arr))
            arr = (arr - ref) * 86400.0
            label = label.replace("[d]", "offset [s]")
        elif stem == "duration":
            arr = arr * 24.0
            label = label.replace("[d]", "[h]")
        out_cols.append(arr)
        labels.append(label)
    return out_cols, labels


def plot_whitelight_corner(samples, filename, instrument_label=None, max_draws=6000):
    """Corner plot of the sampled white-light parameters.

    Only scalar posterior sites are shown (geometry, limb darkening, trend
    coefficients, jitter, and GP hyperparameters when present); derived
    duplicates are dropped. Returns the path written, or ``None`` when the
    optional :mod:`corner` package is missing or there is nothing to plot.
    """
    try:
        import corner
    except ImportError:
        debug("corner is not installed; skipping the white-light corner plot.")
        return None
    columns, labels = _corner_columns(samples)
    if len(columns) < 2:
        return None
    data = np.column_stack(columns)
    if data.shape[0] > max_draws:
        rng = np.random.default_rng(0)
        data = data[rng.choice(data.shape[0], max_draws, replace=False)]

    apply_publication_style()
    color = accent_color_for_label(instrument_label) if instrument_label else LIGHTCURVE_COLOR
    title = None
    if instrument_label:
        from .plotting_style import figure_title_text
        title = figure_title_text(instrument_label, 'White-light posterior')
    fig = print_corner(data, labels, color, title)
    save_figure(fig, filename)
    return filename


class _InnerTicks(MaxNLocator):
    """Two ticks, neither near the ends of the panel, so rotated labels don't run into each other."""

    def __init__(self):
        super().__init__(nbins=3, steps=[1, 2, 2.5, 5, 10])

    def __call__(self):
        low, high = sorted(self.axis.get_view_interval())
        margin = 0.15 * (high - low)
        ticks = [tick for tick in super().__call__() if low + margin <= tick <= high - margin]
        return ticks if len(ticks) <= 2 else [ticks[0], ticks[-1]]


def print_corner(data, labels, color, title=None):
    """A corner plot at full text width, with the house ticks and print-size text."""
    import corner
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH), constrained_layout=False)
    fig = corner.corner(
        data, fig=fig, labels=labels, color=color, bins=35, smooth=1.0, smooth1d=1.0,
        quantiles=(0.16, 0.5, 0.84), show_titles=False, label_kwargs={'fontsize': LABEL_SIZE},
        levels=(0.393, 0.865), plot_datapoints=False, plot_density=False, fill_contours=True,
        hist_kwargs={'linewidth': 1.2}, max_n_ticks=3, labelpad=0.45,
    )
    size = int(round(np.sqrt(len(fig.axes))))
    for index, ax in enumerate(fig.axes):
        if not ax.get_frame_on():
            continue
        ax.xaxis.set_major_locator(_InnerTicks())
        if index % size != index // size:
            ax.yaxis.set_major_locator(_InnerTicks())
        ax.tick_params(labelsize=FONT_SIZE, direction='in', top=True, right=True, length=4, width=1.0)
        for spine in ax.spines.values():
            spine.set_visible(True)
        # Absolute tick values, never a "+1" offset legend.
        for axis in (ax.xaxis, ax.yaxis):
            try:
                axis.get_major_formatter().set_useOffset(False)
            except AttributeError:
                pass
    # corner's margins assume two inches per panel; leave room for the rotated tick labels instead.
    side = fig.get_figwidth()
    fig.subplots_adjust(left=1.35 / side, bottom=1.35 / side, right=1 - 0.1 / side,
                        top=1 - (0.4 if title else 0.1) / side, wspace=0.06, hspace=0.06)
    if title:
        fig.suptitle(title, fontsize=TITLE_SIZE, y=1 - 0.08 / side, va='top')
    return fig
