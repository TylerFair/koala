"""Opt-in timing spectra, conditional on the fixed white-light orbit.

Positive dt0 means a later transit. The trailing response multiplies
``R_trailing - R_leading`` when the two timing responses are antisymmetric.
Radii are in stellar-radius units; a visit has a free achromatic time offset.
"""
from pathlib import Path
from .output_paths import OutputPaths
import json
import numpy as np

SECONDS_PER_DAY = 86400.0
from .readers import TIME_OFFSET


def analytic_time_shift_coefficient(period, a_rs, b):
    """PLATON circular half-disk coefficient C/v, seconds per stellar radius.

    This geometric linearization assumes a full transit of two semicircular
    limbs; it is not a numerical, limb-darkening-dependent calibration.
    The circular formula is undefined for a grazing stellar-centre chord.
    """
    period, a_rs, b = np.broadcast_arrays(
        np.asarray(period, float), np.asarray(a_rs, float), np.asarray(b, float),
    )
    if (np.any(~np.isfinite(period + a_rs + b)) or np.any(period <= 0)
            or np.any(a_rs <= 1) or np.any(np.abs(b) >= 1)):
        raise ValueError('PLATON coefficient requires P>0, a/R*>1 and |b|<1.')
    s = np.sqrt(1 - (1 - b**2) / a_rs**2)
    sin_theta = b / s
    if np.any(np.abs(sin_theta) >= 1):
        raise ValueError('PLATON coefficient requires a full stellar-centre chord.')
    theta = np.arcsin(sin_theta)
    cosine = np.cos(theta)
    c = 3 * (np.pi**2 / 4 - theta**2 + cosine**2) / (16 * cosine)
    velocity = 2 * np.pi * a_rs * s / (period * SECONDS_PER_DAY)
    return c / velocity


def time_shift_table(wavelengths, wavelength_err, samples, geometry, *, responses):
    """One row per channel/planet; medians and central 68% half-widths.

    Koala works in BJD_TDB - 2400000. Only the absolute epoch output receives
    the JD offset; relative times use the sampled small coordinate directly.
    """
    import pandas as pd
    radius = np.asarray(samples['rors'], float)
    dt = np.asarray(samples['dt0'], float)
    if radius.ndim == 2:
        radius, dt = radius[..., None], dt[..., None]
    if dt.shape != radius.shape or radius.ndim != 3:
        raise ValueError('rors and dt0 must have shape (draw, channel, planet).')
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(dt)):
        raise ValueError('Timing products require finite posterior draws.')
    n_planets = radius.shape[-1]
    if len(responses) != radius.shape[1] or any(len(row) != n_planets for row in responses):
        raise ValueError('One exact response is required per channel and planet.')
    t0 = np.broadcast_to(np.atleast_1d(geometry.T0_BASE), (n_planets,))
    ecc = np.broadcast_to(np.atleast_1d(geometry.HARMONICA_ECC), (n_planets,))
    # The circular PLATON expression must never be mislabeled for eccentric
    # systems. Timing inference itself supports the full Keplerian orbit.
    coefficient = np.full(n_planets, np.nan)
    circular = ecc == 0
    period = np.broadcast_to(np.atleast_1d(geometry.PERIOD_FIXED), (n_planets,))
    a = np.broadcast_to(np.atleast_1d(geometry.A_RS_BASE), (n_planets,))
    b = np.broadcast_to(np.atleast_1d(geometry.B_BASE), (n_planets,))
    valid = circular & (np.abs(b) < 1) & (a > 1)
    if np.any(valid):
        coefficient[valid] = analytic_time_shift_coefficient(period[valid], a[valid], b[valid])
    rows = []
    for channel, (wave, width) in enumerate(zip(wavelengths, wavelength_err)):
        for planet in range(n_planets):
            depth = radius[:, channel, planet] ** 2
            offset = dt[:, channel, planet] * SECONDS_PER_DAY
            dq = np.percentile(depth, [16, 50, 84])
            tq = np.percentile(offset, [16, 50, 84])
            correlation = np.corrcoef(depth, offset)[0, 1]
            rows.append(dict(
                wavelength=float(wave), wavelength_err=float(width),
                wave_lo_um=float(wave - width), wave_hi_um=float(wave + width),
                planet=planet, depth=dq[1], depth_err=(dq[2]-dq[0])/2,
                t0=t0[planet] + TIME_OFFSET + tq[1]/SECONDS_PER_DAY,
                t0_bjd=t0[planet] + TIME_OFFSET + tq[1]/SECONDS_PER_DAY,
                t0_err_s=(tq[2]-tq[0])/2, dt0_s=tq[1],
                corr_depth_t0=correlation,
                **responses[channel][planet],
                time_shift_coefficient_analytic=coefficient[planet],
                time_shift_coefficient_difference_percent=(
                    100 * (responses[channel][planet]['time_shift_coefficient'] / coefficient[planet] - 1)
                    if np.isfinite(coefficient[planet]) else np.nan),
            ))
    return pd.DataFrame(rows)


def save_time_shift_products(wavelengths, wavelength_err, samples, prefix, geometry, *,
                             responses, response_seconds=None):
    """Append opt-in timing columns to the spectrum and save a timing plot."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from .artifacts import _atomic_dataframe_csv
    table = time_shift_table(wavelengths, wavelength_err, samples, geometry, responses=responses)
    paths, stage = OutputPaths.from_stem(prefix)
    _atomic_dataframe_csv(table, paths.table(stage, 'timing'), index=False)
    spectrum = pd.read_csv(paths.table(stage, 'spectrum'), float_precision='round_trip')
    columns = ('depth', 'depth_err', 't0', 't0_bjd', 't0_err_s', 'dt0_s',
               'corr_depth_t0', 'time_shift_coefficient',
               'd_dt0_dRL_s_per_Rstar', 'd_dt0_dRT_s_per_Rstar',
               'd_depth_dRL', 'd_depth_dRT', 'antisymmetry_check',
               'time_shift_coefficient_analytic', 'time_shift_coefficient_difference_percent',
               'radius_response_relative_error', 'n_retained')
    for planet, part in table.groupby('planet', sort=True):
        suffix = '' if table.planet.nunique() == 1 else f'{planet:02d}'
        for column in columns:
            spectrum[column + suffix] = part[column].to_numpy()
    _atomic_dataframe_csv(spectrum, paths.table(stage, 'spectrum'), index=False)
    from .plotting_style import apply_publication_style, style_axis, save_figure, figure_title, accent_color_for_label
    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, constrained_layout=True)
    for planet, part in table.groupby('planet', sort=True):
        color = accent_color_for_label(paths.prefix)
        axes[0].errorbar(part.wavelength, part.depth*1e6, yerr=part.depth_err*1e6,
                         fmt='o', ms=4, color=color, mfc='white', label=f'Planet {planet + 1}')
        axes[1].errorbar(part.wavelength, part.dt0_s, yerr=part.t0_err_s,
                         fmt='o', ms=4, color=color, mfc='white')
    axes[0].set_ylabel('Depth [ppm]')
    axes[1].axhline(0, color='.5', lw=.8)
    axes[1].set(xlabel='Wavelength [µm]', ylabel=r'$t_0 - t_{0,white}$ [s]')
    if table.planet.nunique() > 1:
        axes[0].legend(fontsize=10)
    for ax in axes:
        style_axis(ax)
    from .diagnostic_figures import grid_title
    figure_title(axes[0], paths.prefix, grid_title(stage) + ' | Timing')
    save_figure(fig, paths.figure(stage or paths.prefix, 'timing'))
    metadata = dict(
        time_scale='BJD_TDB', uncertainty='central 68 percent interval half-width',
        reference_t0_bjd=(np.atleast_1d(geometry.T0_BASE) + TIME_OFFSET).tolist(),
        conditional_on_fixed_white_orbit=True,
        white_epoch_uncertainty_added_to_each_bin=False,
        time_shift_coefficient_units='seconds per stellar radius',
        time_shift_coefficient_method='two-half-disk boundary integral and spherical-fit linear response',
        response_formula='dtheta/dRh = solve(J.T W J + P, J.T W dF/dRh)',
        evaluation='posterior sample-site medians; jitter conditioned at its median',
        nuisance_parameters='all fitted LD and trend coordinates; Gaussian prior curvature included',
        time_sampling='all retained integrations; no time binning',
        limb_division='perpendicular to instantaneous sky-projected Keplerian velocity',
        leading='morning; along projected velocity; first across stellar limb at ingress',
        trailing='evening; opposite projected velocity',
        sign='positive dt0 is later; coefficient is d(dt0_s)/dR_trailing',
        linear_prediction='dt0_s = d_dt0_dRL_s_per_Rstar * delta_RL + d_dt0_dRT_s_per_Rstar * delta_RT',
        analytic_comparison='PLATON circular C/v; NaN where the full-chord circular expression is undefined',
        percentage_difference='100 * (exact / analytic - 1)',
        calibration='exact to first order at a spherical planet; finite contrasts can be nonlinear',
        radius_response_check_tolerance=1e-9,
        radius_response_check_passed=bool(np.all(table.radius_response_relative_error <= 1e-9)),
        radius_response_check_metric='relative Euclidean norm over retained integrations',
        response_seconds_per_bin=response_seconds,
    )
    Path(paths.table(stage, 'timing', 'json')).write_text(json.dumps(metadata, indent=2) + '\n')
