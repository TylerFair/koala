"""Outputs helpers."""

import os
import sys
import glob
import csv
import json
import pickle
import hashlib
import uuid
import re
import time
import tempfile
import difflib
import warnings
import logging
import inspect
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from plotting import (
    plot_harmonica_limb_spectra,
    plot_harmonica_transmission_posterior,
    plot_harmonica_transmission_strings,
    plot_map_fits,
    plot_map_residuals,
    plot_noise_binning_from_csv,
    plot_transmission_spectrum,
    plot_wavelength_offset_summary,
    plot_whitelight_curve,
    plot_whitelight_residuals,
    plot_whitelight_summary,
)
import argparse
import yaml
import jaxopt
import arviz as az
from createdatacube import SpectroData, process_spectroscopy_data
from matplotlib.widgets import Slider, Button, TextBox
from jaxoplanet.experimental import calc_poly_coeffs
import tinygp
from models.common import _to_f64, _tree_to_f64, get_I_power2, compute_transit_model_auto
from models.ld_parameterization import Power2MaxtedTransform
from models.sing_ld import (
    SING_TABULATED_OFFSET,
    SING_TABULATED_SCATTER,
    estimate_gray_offset,
    quadratic_to_sing,
    write_offset_artifact,
)
from models.harmonica.core import (
    harmonica_a_rs_from_duration,
    harmonica_cos_i_from_geometry,
    harmonica_impact_param_from_cos_i,
    harmonica_duration_from_geometry,
    harmonica_duration_from_cos_i,
    harmonica_odd_coeff_specs,
    HARMONICA_SPECTRO_RORS_MIN,
    HARMONICA_SPECTRO_RORS_MAX,
)
from models.trends import (
    spot_crossing, compute_lc_linear, compute_lc_quadratic, compute_lc_cubic,
    compute_lc_quartic, compute_lc_linear_discontinuity, compute_lc_explinear,
    compute_lc_spot, compute_lc_2spot, compute_lc_none
)
from models.gp import (
    compute_lc_gp_mean, compute_lc_linear_gp_mean, compute_lc_quadratic_gp_mean,
    compute_lc_cubic_gp_mean, compute_lc_quartic_gp_mean, compute_lc_explinear_gp_mean
)
from models.detrend import resolve_detrend_kernel
from models.trend_marginal import (
    marginalized_trend_coefficient_names,
    materialize_marginalized_trend_samples,
)
from surface_outputs import save_surface_results
from models.limb_darkening_config import (
    LD_PRIORS,
    resolve_ld_prior,
    validate_ld_profile,
)
from .constants import *
from .constants import (
    _JUMP_WIDTH_DAYS, _VALID_HARMONICA_SPECTRO_PARAMETERIZATIONS,
    _JAXOPLANET_STATIC_EVAL_KEYS, _SURFACE_STATIC_EVAL_KEYS,
    _JAXOPLANET_DYNAMIC_EVAL_KEYS,
)
from .config import (
    UnknownFlagWarning, _validate_flag_keys, _build_gaussian_trend_prior,
    _has_stellar_ld_uncertainties, _resolve_ld_prior_mode,
    _resolve_ld_parameterization, _spectro_detrend_type,
    _has_single_spot_spectroscopic, load_config, _resolve_stage_vmap_width,
    _resolve_stage_mcmc_kwargs, _resolve_whitelight_laplace_options,
    _resolve_whitelight_mass_matrix, _resolve_whitelight_trend_parameterization,
    _resolve_compile_cache_options, _resolve_ld_prior_cache_options,
    _resolve_harmonica_stage_nuts_kwargs,
    _resolve_jaxoplanet_spectro_nuts_kwargs,
)
from .data import _pad_spectro_cadences_exact, jax_bin_lightcurve
from .artifacts import (
    _update_checkpoint_hash, _science_artifact_fingerprint,
    _science_artifact_manifest_matches, _write_science_artifact_manifest,
    _file_content_identity, _optional_file_content_identity,
    _directory_metadata_identity, _atomic_save_npy, _atomic_savez,
    _atomic_savez_compressed, _atomic_dataframe_csv,
)
from models.harmonica.core import (
    _ALL_ODD_COEFF_SPECS,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    harmonica_half_area_area_radius_and_q,
)

from .constants import *
from .artifacts import _atomic_dataframe_csv

def _param_at(params, name, idx=None):
    if name not in params:
        return None
    val = params[name]
    return val[idx] if idx is not None else val


def _poly_trend_np(params, t_shift, order, idx=None):
    trend = _param_at(params, "c", idx) + _param_at(params, "v", idx) * t_shift
    if order >= 2:
        trend = trend + _param_at(params, "v2", idx) * t_shift**2
    if order >= 3:
        trend = trend + _param_at(params, "v3", idx) * t_shift**3
    if order >= 4:
        trend = trend + _param_at(params, "v4", idx) * t_shift**4
    return trend


def _soft_step_np(t, t_jump, width=_JUMP_WIDTH_DAYS):
    scaled = np.clip((np.asarray(t) - t_jump) / width, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def _trend_from_params_np(detrend_type, time, params, idx=None, gp_trend=None, spot_trend=None, spot_trend2=None, jump_trend=None, exp_trend=None):
    t_shift = time - np.min(time)
    if detrend_type == 'none':
        return np.ones_like(time)

    poly_order = 0
    if 'quartic' in detrend_type:
        poly_order = 4
    elif 'cubic' in detrend_type:
        poly_order = 3
    elif 'quadratic' in detrend_type:
        poly_order = 2
    elif 'linear' in detrend_type or detrend_type in {'spot', '2spot'}:
        poly_order = 1

    if 'spectroscopic' in detrend_type:
        trend = _param_at(params, "c", idx) if poly_order == 0 else _poly_trend_np(params, t_shift, poly_order, idx)
        if 'explinear_spectroscopic' in detrend_type:
            trend = trend + _param_at(params, "A", idx) * exp_trend
        elif 'explinear' in detrend_type:
            trend = trend + _param_at(params, "A", idx) * np.exp(-t_shift / _param_at(params, "tau", idx))
        if 'gp_spectroscopic' in detrend_type:
            trend = trend + _param_at(params, "A_gp", idx) * gp_trend
        if '2spot_spectroscopic' in detrend_type:
            trend = (
                trend
                + _param_at(params, "A_spot", idx) * spot_trend
                + _param_at(params, "A_spot2", idx) * spot_trend2
            )
        elif 'spot_spectroscopic' in detrend_type:
            trend = trend + _param_at(params, "A_spot", idx) * spot_trend
        if 'linear_discontinuity_spectroscopic' in detrend_type:
            trend = trend + _param_at(params, "A_jump", idx) * jump_trend
        return trend

    trend = _param_at(params, "c", idx) if poly_order == 0 else _poly_trend_np(params, t_shift, poly_order, idx)
    if detrend_type == 'linear_discontinuity':
        if jump_trend is None:
            width = _param_at(params, "width", idx)
            if width is None:
                width = _JUMP_WIDTH_DAYS
            jump_trend = _param_at(params, "jump", idx) * _soft_step_np(
                time, _param_at(params, "t_jump", idx), width
            )
        trend = trend + jump_trend
    elif detrend_type == 'explinear':
        trend = trend + _param_at(params, "A", idx) * np.exp(-t_shift / _param_at(params, "tau", idx))
    elif 'spot' in detrend_type and '2spot' not in detrend_type:
        if spot_trend is None:
            spot_trend = spot_crossing(time, _param_at(params, "spot_amp", idx), _param_at(params, "spot_mu", idx), _param_at(params, "spot_sigma", idx))
        trend = trend + spot_trend
    elif '2spot' in detrend_type:
        if spot_trend is None:
            spot_trend = spot_crossing(time, _param_at(params, "spot_amp", idx), _param_at(params, "spot_mu", idx), _param_at(params, "spot_sigma", idx))
            spot_trend = spot_trend + spot_crossing(time, _param_at(params, "spot_amp2", idx), _param_at(params, "spot_mu2", idx), _param_at(params, "spot_sigma2", idx))
        trend = trend + spot_trend
    return trend


def plot_noise_binning_robust(residuals, dt_seconds, outpath, title=None):
    """Write robust RMS-vs-bin-size statistics in the shared house style."""
    del dt_seconds, title  # Retained in the public signature for compatibility.
    bins, measured, p16, p84, expected, _ = _noise_binning_stats(residuals)
    noise_df = pd.DataFrame(
        {
            "bin_size_points": bins,
            "measured_rms": measured * 1e6,
            "measured_rms_p16": p16 * 1e6,
            "measured_rms_p84": p84 * 1e6,
            "expected_white_rms": expected * 1e6,
        }
    )
    plot_noise_binning_from_csv(noise_df, outpath, instrument_label=str(outpath))
    print(f"Saved robust noise-binning plot to {outpath}")


def _noise_binning_stats(residuals, n_bins=30, max_bin=None):
    residuals = np.array(residuals)
    if residuals.ndim == 1:
        residuals = residuals[None, :]
    n_channels, n_times = residuals.shape
    if max_bin is None:
        max_bin = max(1, n_times // 4)
    bins = np.unique(np.round(np.logspace(0, np.log10(max_bin), n_bins)).astype(int))
    sigma_b_channels = np.full((n_channels, len(bins)), np.nan)
    for i in range(n_channels):
        r = residuals[i, :]
        for j, b in enumerate(bins):
            m = (n_times // b) * b
            if m < b:
                continue
            r_trunc = r[:m].reshape(-1, b)
            means = r_trunc.mean(axis=1)
            if means.size >= 2:
                sigma_b_channels[i, j] = np.std(means, ddof=0)

    sigma_med = np.nanmedian(sigma_b_channels, axis=0)
    sigma_16 = np.nanpercentile(sigma_b_channels, 16, axis=0)
    sigma_84 = np.nanpercentile(sigma_b_channels, 84, axis=0)
    sigma1_channels = np.nanstd(residuals, axis=1, ddof=0)
    sigma1_med = np.nanmedian(sigma1_channels)
    expected_white_per_channel = sigma1_channels[:, None] / np.sqrt(bins)
    expected_white_median = sigma1_med / np.sqrt(bins)
    return bins, sigma_med, sigma_16, sigma_84, expected_white_median, expected_white_per_channel


def save_noise_binning_data(residuals, csv_path, to_ppm=True):
    bins, sigma_med, sigma_16, sigma_84, expected_white_med, _ = _noise_binning_stats(residuals)
    factor = 1e6 if to_ppm else 1.0
    df = pd.DataFrame({
        "bin_size_points": bins,
        "measured_rms": sigma_med * factor,
        "measured_rms_p16": sigma_16 * factor,
        "measured_rms_p84": sigma_84 * factor,
        "expected_white_rms": expected_white_med * factor,
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved noise-binning data to {csv_path}")


def save_whitelight_timeseries(time, flux, flux_err, bestfit_model, output_csv,
                               t0_reference=None, transit_model=None, trend_model=None,
                               residual=None, outlier_mask=None,
                               gp_flux=None, gp_err=None, gp_trend=None):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    bestfit_model = np.asarray(bestfit_model, dtype=float)
    if residual is None:
        residual = flux - bestfit_model
    residual = np.asarray(residual, dtype=float)

    if outlier_mask is None:
        outlier_mask = np.zeros_like(time, dtype=bool)
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    if outlier_mask.shape != time.shape:
        outlier_mask = np.zeros_like(time, dtype=bool)

    if t0_reference is None:
        time_from_t0_hr = np.full_like(time, np.nan, dtype=float)
    else:
        time_from_t0_hr = (time - float(t0_reference)) * 24.0

    data = {
        "time_bjd": time,
        "time_from_t0_hr": time_from_t0_hr,
        "flux": flux,
        "flux_err": flux_err,
        "bestfit_model": bestfit_model,
        "residual": residual,
        "residual_ppm": residual * 1e6,
        "is_outlier": outlier_mask.astype(int),
    }
    if transit_model is not None:
        data["transit_model"] = np.asarray(transit_model, dtype=float)
    if trend_model is not None:
        trend_model = np.asarray(trend_model, dtype=float)
        data["trend_model"] = trend_model
        # F = (1 + transit) * trend, so the detrended flux is flux / trend.
        data["detrended_flux"] = flux / trend_model
    if gp_flux is not None:
        data["gp_flux"] = np.asarray(gp_flux, dtype=float)
    if gp_err is not None:
        data["gp_err"] = np.asarray(gp_err, dtype=float)
    if gp_trend is not None:
        data["gp_trend"] = np.asarray(gp_trend, dtype=float)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved white-light time series to {output_csv}")


def _save_mcmc_diagnostics(
    mcmc, max_tree_depth=None, output_path=None, extra_diagnostics=None,
    samples_override=None, extra_fields_override=None,
):
    """Print and optionally persist diagnostics that expose gradient failures."""
    samples = jax.device_get(
        mcmc.get_samples() if samples_override is None else samples_override
    )
    extra = jax.device_get(
        mcmc.get_extra_fields()
        if extra_fields_override is None
        else extra_fields_override
    )
    diagnostics = {}

    diverging = np.asarray(extra.get("diverging", []), dtype=bool)
    diagnostics["num_draws"] = int(diverging.size) if diverging.size else int(
        next(iter(samples.values())).shape[0]
    )
    diagnostics["num_divergences"] = int(np.sum(diverging))

    finite_elements = 0
    total_elements = 0
    nonfinite_sites = []
    for name, values in samples.items():
        values = np.asarray(values)
        finite = np.isfinite(values)
        finite_elements += int(np.sum(finite))
        total_elements += int(finite.size)
        if not np.all(finite):
            nonfinite_sites.append(name)
    diagnostics["finite_sample_fraction"] = (
        finite_elements / total_elements if total_elements else 1.0
    )
    diagnostics["nonfinite_sample_sites"] = nonfinite_sites

    for name in ("accept_prob", "potential_energy", "num_steps"):
        if name not in extra:
            continue
        values = np.asarray(extra[name])
        diagnostics[f"{name}_mean"] = float(np.nanmean(values))
        diagnostics[f"{name}_median"] = float(np.nanmedian(values))
        diagnostics[f"{name}_max"] = float(np.nanmax(values))
        diagnostics[f"{name}_nonfinite"] = int(np.sum(~np.isfinite(values)))

    if max_tree_depth is not None and "num_steps" in extra:
        max_steps = 2 ** int(max_tree_depth) - 1
        diagnostics["max_tree_depth"] = int(max_tree_depth)
        diagnostics["num_tree_depth_saturations"] = int(
            np.sum(np.asarray(extra["num_steps"]) >= max_steps)
        )

    diagnostics["warning"] = bool(
        diagnostics["num_divergences"]
        or diagnostics["nonfinite_sample_sites"]
        or diagnostics.get("potential_energy_nonfinite", 0)
    )
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    print("MCMC numerical diagnostics:", diagnostics)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(diagnostics, stream, indent=2, sort_keys=True)
        print(f"Saved MCMC numerical diagnostics to {output_path}")
    return diagnostics


def compute_aic(n, residuals, k):
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    aic = 2*k + n * np.log(rss/n)
    return aic


def get_asym_errors(data, axis=0):
    p16, p50, p84 = jnp.nanpercentile(data, jnp.array([16, 50, 84]), axis=axis)
    err_low = p50 - p16
    err_high = p84 - p50
    return p50, err_low, err_high


def compute_u_from_c(c1, c2):
    POLY_DEGREE = 12
    MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
    profile = get_I_power2(c1, c2, MUS)
    return calc_poly_coeffs(MUS, profile, poly_degree=POLY_DEGREE)


def fit_polynomial(x, y, poly_orders):
    best_order = None
    best_aic = np.inf
    best_coeffs = None
    for deg in poly_orders:
        y_med = np.median(y, axis=0)
        coeffs = np.polyfit(x, y_med, deg)
        pred = np.polyval(coeffs, x)
        aic = compute_aic(len(x), y_med - pred, k=deg+1)
        if aic < best_aic:
            best_aic = aic
            best_order = deg
            best_coeffs = coeffs
    return best_coeffs, best_order, best_aic


def plot_poly_fit(x, y, coeffs, order, xlabel, ylabel, title, save_path):
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)
    y_med = np.median(y, axis=0)
    y_err = np.std(y, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y_med, yerr=y_err, fmt='o', ls='', ecolor='k', mfc='k', mec='k')
    ax.plot(x_fit, y_fit, '-', label=f'Poly fit (deg {order})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved polynomial fit plot to {save_path}")


def save_results(wavelengths,wavelength_err,  samples, csv_filename):
    depth_chain = samples['rors']**2
    depth_median = np.nanmedian(depth_chain, axis=0)
    depth_err = np.std(depth_chain, axis=0)
    depth_ppm = depth_median * 1e6
    depth_err_ppm = depth_err * 1e6
    if depth_median.ndim == 1:
        depth_median = depth_median[:, np.newaxis]
        depth_err = depth_err[:, np.newaxis]
        depth_ppm = depth_ppm[:, np.newaxis]
        depth_err_ppm = depth_err_ppm[:, np.newaxis]
    n_planets = depth_median.shape[1]
    header_cols = ["wavelength", "wavelength_err"]
    for i in range(n_planets):
        header_cols.append(f"depth{i:02d}")
        header_cols.append(f"depth_err{i:02d}")
        header_cols.append(f"depth_ppm{i:02d}")
        header_cols.append(f"depth_err_ppm{i:02d}")
    sampler_used = getattr(samples, "sampler_used", None)
    if sampler_used is not None:
        if len(sampler_used) != len(wavelengths):
            raise ValueError("sampler_used provenance does not match wavelength axis")
        header_cols.append("sampler_used")
    header = ",".join(header_cols)
    output_cols = [wavelengths, wavelength_err]
    for i in range(n_planets):
        output_cols.append(depth_median[:, i])
        output_cols.append(depth_err[:, i])
        output_cols.append(depth_ppm[:, i])
        output_cols.append(depth_err_ppm[:, i])
    if sampler_used is None:
        output_data = np.column_stack(output_cols)
        np.savetxt(csv_filename, output_data, delimiter=",", header=header, comments="")
    else:
        frame = pd.DataFrame(np.column_stack(output_cols), columns=header_cols[:-1])
        frame["sampler_used"] = sampler_used
        _atomic_dataframe_csv(frame, csv_filename, index=False)
    print(f"Transmission spectroscopy data saved to {csv_filename}")


def summarize_joint_geometry(samples, joint_geometry):
    """Summarise shared-geometry sites sampled jointly across channels.

    ``joint_geometry`` maps a site name (``t0``, ``b``, ``duration``,
    ``a_rs``) to a mapping with ``prior_center``, ``prior_sigma`` and
    ``whitelight_std`` arrays (one entry per planet).  Returns the per-row
    columns to broadcast into the channel table and the long-format rows of
    the ``*_joint_geometry.csv`` product.
    """
    columns = {}
    rows = []
    for name, reference in joint_geometry.items():
        if name not in samples:
            raise ValueError(
                f"Joint geometry site {name!r} is missing from the posterior samples."
            )
        draws = np.asarray(samples[name], dtype=float)
        draws = draws.reshape(draws.shape[0], -1)
        n_planets = draws.shape[1]
        center = np.broadcast_to(
            np.atleast_1d(np.asarray(reference['prior_center'], dtype=float)),
            (n_planets,),
        )
        sigma = np.broadcast_to(
            np.atleast_1d(np.asarray(reference['prior_sigma'], dtype=float)),
            (n_planets,),
        )
        wl_std = np.broadcast_to(
            np.atleast_1d(np.asarray(reference['whitelight_std'], dtype=float)),
            (n_planets,),
        )
        for planet in range(n_planets):
            med, low, high = get_asym_errors(draws[:, planet])
            std = float(np.std(draws[:, planet]))
            label = name if n_planets == 1 else f"{name}_{planet}"
            columns.update({
                label: med,
                f'{label}_err': std,
                f'{label}_err_low': low,
                f'{label}_err_high': high,
            })
            rows.append({
                'parameter': name,
                'planet': planet,
                'median': med,
                'std': std,
                'err_low': low,
                'err_high': high,
                'prior_center': float(center[planet]),
                'prior_sigma': float(sigma[planet]),
                'whitelight_median': float(center[planet]),
                'whitelight_std': float(wl_std[planet]),
                'shift_in_whitelight_sigma': (
                    float((med - center[planet]) / wl_std[planet])
                    if wl_std[planet] > 0 else np.nan
                ),
            })
    return columns, rows


def save_detailed_fit_results(time, flux, flux_err, wavelengths, wavelengths_err, samples, map_params,
                               transit_params, detrend_type, output_prefix,
                               total_error_fit=None, gp_trend=None, spot_trend=None, jump_trend=None,
                               joint_geometry=None):
    n_wavelengths = len(wavelengths)
    n_times = len(time)
    print(f"Saving detailed fit results to {output_prefix}_*.csv")
    param_rows = []
    shared_columns = {}
    if joint_geometry:
        shared_columns, joint_rows = summarize_joint_geometry(samples, joint_geometry)
        joint_df = pd.DataFrame(joint_rows)
        joint_path = f"{output_prefix}_joint_geometry.csv"
        joint_df.to_csv(joint_path, index=False)
        print(f"Saved joint geometry posterior to {joint_path}")
        for entry in joint_rows:
            print(
                f"  joint {entry['parameter']}[{entry['planet']}] = "
                f"{entry['median']:.6f} +/- {entry['std']:.6f} "
                f"(white light {entry['whitelight_median']:.6f} +/- "
                f"{entry['whitelight_std']:.6f}, shift "
                f"{entry['shift_in_whitelight_sigma']:+.2f} sigma)"
            )
    channel_depth_errors = np.nanstd(np.asarray(samples['rors']) ** 2, axis=0)
    if channel_depth_errors.ndim > 1:
        channel_depth_errors = np.nanmax(channel_depth_errors, axis=-1)
    typical_depth_error = np.nanmedian(channel_depth_errors)
    depth_error_ratios = channel_depth_errors / max(typical_depth_error, 1e-30)
    
    def get_stats_local(data_slice):
        med, low, high = get_asym_errors(data_slice)
        return med, np.std(data_slice), low, high

    for i in range(n_wavelengths):
        rors_med, rors_std, rors_l, rors_h = get_stats_local(samples['rors'][:, i])
        depth_med, depth_std, depth_l, depth_h = get_stats_local(samples['rors'][:, i]**2)
        
        row = {
            'wavelength': wavelengths[i],
            'wavelength_err': wavelengths_err[i],
            'rors': rors_med,
            'rors_err': rors_std,
            'rors_err_low': rors_l,
            'rors_err_high': rors_h,
            'depth': depth_med,
            'depth_err': depth_std,
            'depth_err_low': depth_l,
            'depth_err_high': depth_h,
            'depth_ppm': depth_med * 1e6,
            'depth_err_ppm': depth_std * 1e6,
            'depth_err_low_ppm': depth_l * 1e6,
            'depth_err_high_ppm': depth_h * 1e6,
            'depth_error_ratio_to_median': depth_error_ratios[i],
            'depth_error_outlier_gt5x': bool(depth_error_ratios[i] > 5.0),
        }
        
        if 'u' in samples:
            u1_med, u1_std, u1_l, u1_h = get_stats_local(samples['u'][:, i, 0])
            u2_med, u2_std, u2_l, u2_h = get_stats_local(samples['u'][:, i, 1])
            row.update({
                'u1': u1_med, 'u1_err': u1_std, 'u1_err_low': u1_l, 'u1_err_high': u1_h,
                'u2': u2_med, 'u2_err': u2_std, 'u2_err_low': u2_l, 'u2_err_high': u2_h
            })

        if 'c1' in samples:
            c1_med, c1_std, c1_l, c1_h = get_stats_local(samples['c1'][:, i])
            row.update({'c1': c1_med, 'c1_err': c1_std, 'c1_err_low': c1_l, 'c1_err_high': c1_h})
            row.update({'u1': c1_med, 'u1_err': c1_std, 'u1_err_low': c1_l, 'u1_err_high': c1_h})

        if 'c2' in samples:
            c2_med, c2_std, c2_l, c2_h = get_stats_local(samples['c2'][:, i])
            row.update({'c2': c2_med, 'c2_err': c2_std, 'c2_err_low': c2_l, 'c2_err_high': c2_h})
            row.update({'u2': c2_med, 'u2_err': c2_std, 'u2_err_low': c2_l, 'u2_err_high': c2_h})

        if 'u1' in samples and 'u2' in samples:
            u1_med, u1_std, u1_l, u1_h = get_stats_local(samples['u1'][:, i])
            u2_med, u2_std, u2_l, u2_h = get_stats_local(samples['u2'][:, i])
            row.update({
                'u1': u1_med, 'u1_err': u1_std,
                'u1_err_low': u1_l, 'u1_err_high': u1_h,
                'u2': u2_med, 'u2_err': u2_std,
                'u2_err_low': u2_l, 'u2_err_high': u2_h,
            })

        for harmonic_name in HARMONICA_ODD_HARMONICS:
            if harmonic_name in samples:
                harm_med, harm_std, harm_l, harm_h = get_stats_local(samples[harmonic_name][:, i])
                row.update({
                    harmonic_name: harm_med,
                    f'{harmonic_name}_err': harm_std,
                    f'{harmonic_name}_err_low': harm_l,
                    f'{harmonic_name}_err_high': harm_h,
                })

        if detrend_type != 'none':
            c_med, c_std, c_l, c_h = get_stats_local(samples['c'][:, i])
            row.update({'c': c_med, 'c_err': c_std, 'c_err_low': c_l, 'c_err_high': c_h})
            if 'v' in samples:
                v_med, v_std, v_l, v_h = get_stats_local(samples['v'][:, i])
                row.update({'v': v_med, 'v_err': v_std, 'v_err_low': v_l, 'v_err_high': v_h})
        
        for key in ['v2', 'v3', 'v4', 'A', 'tau', 't_jump', 'jump',
                    'spot_amp', 'spot_mu', 'spot_sigma', 'spot_amp2', 'spot_mu2', 'spot_sigma2',
                    'A_gp', 'A_spot', 'A_spot2', 'A_jump']:
            if key in samples:
                med, std, low, high = get_stats_local(samples[key][:, i])
                row.update({
                    key: med,
                    f'{key}_err': std,
                    f'{key}_err_low': low,
                    f'{key}_err_high': high
                })
            
        row.update(shared_columns)
        param_rows.append(row)

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(f"{output_prefix}_bestfit_params.csv", index=False)
    outlier_indices = np.flatnonzero(depth_error_ratios > 5.0)
    if outlier_indices.size:
        print(
            "WARNING: depth uncertainties exceed 5x the channel median at "
            f"channel indices {outlier_indices.tolist()}."
        )

    time = np.asarray(time)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)
    wavelengths = np.asarray(wavelengths)
    wavelengths_err = np.asarray(wavelengths_err)

    meta_path = f"{output_prefix}_wavelengths.csv"
    meta_df = pd.DataFrame({
        "wavelength_center": wavelengths,
        "wavelength_err": wavelengths_err,
    })
    meta_df.to_csv(meta_path, index=False)

    wide_path = f"{output_prefix}_lightcurves_wide.csv"
    columns = ["time"]
    data_cols = [time]
    for i in range(n_wavelengths):
        columns.append(f"flux_{i:03d}")
        columns.append(f"flux_err_{i:03d}")
        data_cols.append(flux[i])
        data_cols.append(flux_err[i])

    wide_data = np.column_stack(data_cols)
    wide_df = pd.DataFrame(wide_data, columns=columns)
    wide_df.to_csv(wide_path, index=False)
    print(f"Saved wavelength metadata to {meta_path}")
    print(f"Saved wide light curves to {wide_path}")
    return params_df, None


def get_robust_sigma(x):
    """Helper to calculate sigma using MAD (robust to outliers)."""
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad


def calculate_beta_metrics(residuals, dt, cut_factor=5.0):
    residuals = np.array(residuals)
    ndata = len(residuals)
    
    sigma1 = get_robust_sigma(residuals)
    
    cadence_min = dt / 60.0
    
    max_bin_n = ndata // int(cut_factor) 
    
    bin_sizes_points = np.unique(np.logspace(0, np.log10(max_bin_n), 300).astype(int))
    
    measured_rms = []
    expected_rms = []
    bin_sizes_min = []
    betas = []
    
    for N in bin_sizes_points:
        cutoff = ndata - (ndata % N)
        binned_res = residuals[:cutoff].reshape(-1, N).mean(axis=1)
        
        sigma_N_measured = get_robust_sigma(binned_res)
        
        sigma_N_theory = sigma1 / np.sqrt(N)
        
        measured_rms.append(sigma_N_measured)
        expected_rms.append(sigma_N_theory)
        bin_sizes_min.append(N * cadence_min)
        
        betas.append(sigma_N_measured / sigma_N_theory)

    beta_final = np.median(betas)
    
    return beta_final, np.array(bin_sizes_min), np.array(measured_rms), np.array(expected_rms)


def run_beta_monte_carlo(residuals, dt, n_sims=500):
    ndata = len(residuals)
    
    sigma1 = get_robust_sigma(residuals)
    
    mc_betas = []
    
    _, ref_bin_sizes, _, _ = calculate_beta_metrics(residuals, dt)
    all_sim_rms = np.zeros((n_sims, len(ref_bin_sizes)))
    
    for i in range(n_sims):
        synth_res = np.random.normal(0, sigma1, ndata)
        b_sim, _, rms_sim, _ = calculate_beta_metrics(synth_res, dt)
        
        mc_betas.append(b_sim)
        
        if len(rms_sim) == len(ref_bin_sizes):
            all_sim_rms[i, :] = rms_sim
            
    rms_low_1sig = np.percentile(all_sim_rms, 16, axis=0)
    rms_high_1sig = np.percentile(all_sim_rms, 84, axis=0)
    rms_low_2sig = np.percentile(all_sim_rms, 5, axis=0)
    rms_high_2sig = np.percentile(all_sim_rms, 95, axis=0)

    return np.array(mc_betas), rms_low_1sig, rms_high_1sig, rms_low_2sig, rms_high_2sig
