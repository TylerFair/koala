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
from functools import partial

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.stats import norm
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.7
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from plotting import plot_map_fits, plot_map_residuals, plot_transmission_spectrum, plot_wavelength_offset_summary
import argparse
import yaml
import jaxopt
import arviz as az
from createdatacube import SpectroData, process_spectroscopy_data
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from jaxoplanet.experimental import calc_poly_coeffs
import tinygp
import matplotlib.cm as cm
from models.common import _to_f64, _tree_to_f64, get_I_power2, compute_transit_model_auto
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

TREND_PARAMS = [
    'c', 'v', 'v2', 'v3', 'v4', 
    'A', 'tau', 
    'spot_amp', 'spot_mu', 'spot_sigma', 
    'spot_amp2', 'spot_mu2', 'spot_sigma2',
    't_jump', 'jump', 
    'A_gp', 'A_spot', 'A_spot2', 'A_jump'
]

LD_PRIOR_MODES = {'fixed', 'widegaussian', 'informed', 'uniform'}


def _build_gaussian_trend_prior(detrend_type, num_channels, init_params, flags):
    """Build explicit per-channel priors for analytic trend marginalization."""
    names = marginalized_trend_coefficient_names(detrend_type)
    configured_means = dict(flags.get('trend_prior_means', {}) or {})
    configured_scales = dict(flags.get('trend_prior_scales', {}) or {})
    default_means = {
        'c': 1.0,
        'v': 0.0,
        'v2': 0.0,
        'v3': 0.0,
        'v4': 0.0,
        'A': 0.0,
        'A_spot': 1.0,
        'A_spot2': 1.0,
        'A_jump': 1.0,
    }
    default_scales = {
        'c': 0.1,
        'v': 0.1,
        'v2': 0.1,
        'v3': 0.1,
        'v4': 0.1,
        'A': 0.1,
        'A_spot': 0.75,
        'A_spot2': 0.75,
        'A_jump': 0.75,
    }

    means = []
    scales = []
    for name in names:
        source_mean = configured_means.get(
            name, init_params.get(name, default_means[name])
        )
        mean = jnp.asarray(source_mean, dtype=jnp.float64)
        if mean.ndim == 0:
            mean = jnp.broadcast_to(mean, (num_channels,))
        if mean.shape != (num_channels,):
            raise ValueError(
                f"Trend prior mean for {name!r} must be scalar or shape "
                f"({num_channels},); received {mean.shape}."
            )
        scale = jnp.asarray(
            configured_scales.get(name, default_scales[name]),
            dtype=jnp.float64,
        )
        if scale.ndim == 0:
            scale = jnp.broadcast_to(scale, (num_channels,))
        if scale.shape != (num_channels,):
            raise ValueError(
                f"Trend prior scale for {name!r} must be scalar or shape "
                f"({num_channels},); received {scale.shape}."
            )
        if not bool(jnp.all(jnp.isfinite(mean))):
            raise ValueError(f"Trend prior mean for {name!r} is non-finite.")
        if not bool(jnp.all(jnp.isfinite(scale) & (scale > 0))):
            raise ValueError(f"Trend prior scale for {name!r} must be finite and > 0.")
        means.append(mean)
        scales.append(scale)
    return jnp.stack(means, axis=1), jnp.stack(scales, axis=1), names


def _has_stellar_ld_uncertainties(stellar_cfg):
    return all(k in stellar_cfg for k in ('teff_sigma', 'logg_sigma', 'feh_sigma'))


def _resolve_ld_prior_mode(flags, stellar_cfg, ld_profile):
    raw_mode = flags.get('ld_prior', None)
    legacy_fix_ld = bool(flags.get('fix_ld', False))

    if raw_mode is None:
        if legacy_fix_ld:
            return 'fixed'
        if ld_profile == 'power2' and _has_stellar_ld_uncertainties(stellar_cfg):
            return 'informed'
        return 'widegaussian'

    mode = str(raw_mode).strip().lower().replace('-', '').replace('_', '')
    alias_map = {
        'fixed': 'fixed',
        'widegaussian': 'widegaussian',
        'gaussian': 'widegaussian',
        'free': 'widegaussian',
        'informed': 'informed',
        'stellarprior': 'informed',
        'stellar': 'informed',
        'sing': 'sing',
        'uniform': 'uniform',
    }
    if mode not in alias_map:
        raise ValueError(
            "flags.ld_prior must be one of {'fixed', 'widegaussian', 'informed', 'sing', 'uniform'}. "
            f"Received '{raw_mode}'."
        )
    resolved = alias_map[mode]
    if legacy_fix_ld and resolved != 'fixed':
        print(
            f"[LD prior] flags.ld_prior='{resolved}' overrides legacy flags.fix_ld={legacy_fix_ld}.",
            flush=True,
        )
    if resolved == 'informed':
        if ld_profile != 'power2':
            raise ValueError("flags.ld_prior: 'informed' currently supports only flags.ld_profile: 'power2'.")
        if not _has_stellar_ld_uncertainties(stellar_cfg):
            raise ValueError(
                "flags.ld_prior: 'informed' requires stellar.teff_sigma, stellar.logg_sigma, and stellar.feh_sigma."
            )
    if resolved == 'sing' and ld_profile != 'quadratic':
        raise ValueError("flags.ld_prior: 'sing' requires flags.ld_profile: 'quadratic'.")
    return resolved

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

_JUMP_WIDTH_DAYS = 1e-4

def _soft_step_np(t, t_jump, width=_JUMP_WIDTH_DAYS):
    return 0.5 * (1.0 + np.tanh((t - t_jump) / width))

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
            jump_trend = _param_at(params, "jump", idx) * _soft_step_np(time, _param_at(params, "t_jump", idx))
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

def _align_trend_to_time(trend, trend_time, target_time):
    trend = np.asarray(trend)
    target_time = np.asarray(target_time)
    trend_time = np.asarray(trend_time)
    if len(trend) == len(target_time):
        return trend
    return np.interp(target_time, trend_time, trend)

def _spectro_detrend_type(detrending_type, fixed_timescale=False):
    """Map WL detrending type to spectroscopic equivalent where applicable."""
    detrend_type = detrending_type
    if fixed_timescale and 'explinear' in detrend_type and 'explinear_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('explinear', 'explinear_spectroscopic')
    if 'gp' in detrend_type and 'gp_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('gp', 'gp_spectroscopic')
    if 'linear_discontinuity' in detrend_type and 'linear_discontinuity_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('linear_discontinuity', 'linear_discontinuity_spectroscopic')
    if 'spot' in detrend_type and 'spot_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('spot', 'spot_spectroscopic')
    return detrend_type


def _validate_interpolated_trend_mode(
    interpolate_trend,
    *,
    transit_engine,
    detrending_type,
):
    """Fail closed for fixed HR trends that the interpolation path can build."""

    if not interpolate_trend:
        return
    spectro_detrend_type = _spectro_detrend_type(str(detrending_type))
    if transit_engine != "jaxoplanet" or spectro_detrend_type != "linear":
        raise ValueError(
            "flags.interpolate_trend currently supports only "
            "flags.transit_engine='jaxoplanet' with "
            "flags.detrending_type='linear'. The low-to-high-resolution "
            "handoff currently constructs only the fixed c and v columns; "
            "using it with polynomial, exponential, spot, discontinuity, GP, "
            "or Harmonica trends would be incomplete."
        )


def _has_single_spot_spectroscopic(detrend_type):
    return 'spot_spectroscopic' in detrend_type and '2spot_spectroscopic' not in detrend_type

def _bin_time_series_numpy(time, y, yerr=None, dt_seconds=120.0, t0=None, method="weighted"):
    """
    Bin a time-series (or stack of time-series) onto a fixed cadence.

    Parameters
    ----------
    time : (N,) array
        Time array (assumed to be in DAYS if it's MJD/BJD-style; we convert dt_seconds->dt_days).
    y : (N,) or (C, N) array
        Flux (or residuals, etc.). If 2D, bins along the last axis.
    yerr : (N,) or (C, N) array or None
        Uncertainties. If None, uses unweighted mean and returns None for binned errors.
    dt_seconds : float
        Bin width in seconds.
    t0 : float or None
        Bin reference start time. If None, uses min(time).
    method : {"weighted","mean"}
        "weighted": inverse-variance weighted mean (needs yerr)
        "mean": simple mean; error propagated as sqrt(sum(err^2))/N if yerr provided

    Returns
    -------
    t_b : (M,) array
    y_b : (M,) or (C, M) array
    yerr_b : same shape as y_b or None
    counts : (M,) array
    """
    time = np.asarray(time)
    y = np.asarray(y)

    if t0 is None:
        t0 = np.nanmin(time)

    dt_days = float(dt_seconds) / 86400.0
    if dt_days <= 0:
        raise ValueError("dt_seconds must be > 0")

    idx = np.floor((time - t0) / dt_days).astype(np.int64)

    valid = np.isfinite(time)
    if y.ndim == 1:
        valid &= np.isfinite(y)
    else:
        valid &= np.all(np.isfinite(y), axis=0)

    if yerr is not None:
        yerr = np.asarray(yerr)

        if y.ndim == 1 and yerr.ndim == 2:
            if yerr.shape == (1, y.shape[0]):
                yerr = yerr[0]
            elif yerr.shape == (y.shape[0], 1):
                yerr = yerr[:, 0]

        if yerr.ndim == 0:
            yerr = np.full_like(y, float(yerr))

        elif y.ndim == 2 and yerr.ndim == 1:
            if yerr.shape[0] == y.shape[1]:
                yerr = np.broadcast_to(yerr[None, :], y.shape)
            elif yerr.shape[0] == y.shape[0]:
                yerr = np.broadcast_to(yerr[:, None], y.shape)

        if yerr.ndim == 1:
            valid &= np.isfinite(yerr) & (yerr > 0)
        else:
            valid &= np.all(np.isfinite(yerr) & (yerr > 0), axis=0)


    time_v = time[valid]
    idx_v = idx[valid]

    idx0 = idx_v.min()
    idx_v = idx_v - idx0
    nbins = int(idx_v.max()) + 1

    counts = np.bincount(idx_v, minlength=nbins).astype(float)
    keep = counts > 0

    t_sum = np.bincount(idx_v, weights=time_v, minlength=nbins)
    t_b_full = np.full(nbins, np.nan)
    t_b_full[keep] = t_sum[keep] / counts[keep]
    t_b = t_b_full[keep]
    counts_b = counts[keep]

    def _bin_1d(y1, e1):
        if method == "weighted" and e1 is not None:
            w = 1.0 / (e1 * e1)
            wsum = np.bincount(idx_v, weights=w, minlength=nbins)
            ysum = np.bincount(idx_v, weights=w * y1, minlength=nbins)
            yb_full = np.full(nbins, np.nan)
            eb_full = np.full(nbins, np.nan)
            yb_full[keep] = ysum[keep] / wsum[keep]
            eb_full[keep] = np.sqrt(1.0 / wsum[keep])
            yb = yb_full[keep]
            eb = eb_full[keep]
            return yb, eb
        else:
            ysum = np.bincount(idx_v, weights=y1, minlength=nbins)
            yb_full = np.full(nbins, np.nan)
            yb_full[keep] = ysum[keep] / counts[keep]
            yb = yb_full[keep]
            if e1 is None:
                return yb, None
            esum2 = np.bincount(idx_v, weights=e1 * e1, minlength=nbins)
            eb = np.sqrt(esum2[keep]) / counts_b
            return yb, eb

    if y.ndim == 1:
        y_v = y[valid]
        e_v = None if yerr is None else (yerr[valid] if yerr.ndim == 1 else yerr[:, valid])
        y_b, e_b = _bin_1d(y_v, e_v if (e_v is not None and np.ndim(e_v) == 1) else None)
        return t_b, y_b, e_b, counts_b

    C = y.shape[0]
    y_b_list = []
    e_b_list = [] if yerr is not None else None

    for c in range(C):
        y_v = y[c, valid]
        e_v = None if yerr is None else (yerr[c, valid] if yerr.ndim == 2 else None)
        yb, eb = _bin_1d(y_v, e_v)
        y_b_list.append(yb)
        if e_b_list is not None:
            e_b_list.append(eb)

    y_b = np.stack(y_b_list, axis=0)
    e_b = None if e_b_list is None else np.stack(e_b_list, axis=0)
    return t_b, y_b, e_b, counts_b


def bin_spectrodata_in_time(data, dt_seconds=120.0, method="weighted", bin_whitelight=True, bin_spectroscopic=True):
    """
    In-place time-binning for SpectroData:
      - wl_time, wl_flux, wl_flux_err
      - time, flux_lr, flux_err_lr
      - time, flux_hr, flux_err_hr
    """
    n0_wl = len(data.wl_time)
    n0_spec = len(data.time)

    def _timebin_debug(label, time, wl_time, flux_lr, flux_hr, wl_flux):
        def _summarize_time(t):
            if t is None or len(t) == 0:
                return "n=0"
            dt_med = np.nanmedian(np.diff(t)) if len(t) > 1 else np.nan
            return f"n={len(t)}, range=[{np.nanmin(t):.6f}, {np.nanmax(t):.6f}], dt_med={dt_med:.6e}"

        def _shape_or_none(arr):
            return None if arr is None else arr.shape

        def _nan_count(arr):
            return None if arr is None else int(np.isnan(arr).sum())

        print(f"=== TIME BINNING DEBUG ({label}) ===")
        print(f"wl_time: {_summarize_time(wl_time)}")
        print(f"time: {_summarize_time(time)}")
        print(f"wl_flux shape: {_shape_or_none(wl_flux)} nan_count: {_nan_count(wl_flux)}")
        print(f"flux_lr shape: {_shape_or_none(flux_lr)} nan_count: {_nan_count(flux_lr)}")
        print(f"flux_hr shape: {_shape_or_none(flux_hr)} nan_count: {_nan_count(flux_hr)}")
        if flux_lr is not None:
            print(f"n_channels_lr: {flux_lr.shape[0]}")
        if flux_hr is not None:
            print(f"n_channels_hr: {flux_hr.shape[0]}")

    _timebin_debug("pre", data.time, data.wl_time, data.flux_lr, data.flux_hr, data.wl_flux)

    t_b_wl = data.wl_time
    if bin_whitelight:
        t_b_wl, wl_b, wlerr_b, _ = _bin_time_series_numpy(
            data.wl_time, data.wl_flux, data.wl_flux_err,
            dt_seconds=dt_seconds, method=method
        )
        data.wl_time = t_b_wl
        data.wl_flux = wl_b
        data.wl_flux_err = wlerr_b

    if bin_spectroscopic:
        t0_ref = np.nanmin(data.wl_time)
        t_b_lr, flr_b, flrerr_b, _ = _bin_time_series_numpy(
            data.time, data.flux_lr, data.flux_err_lr,
            dt_seconds=dt_seconds, method=method, t0=t0_ref
        )
        t_b_hr, fhr_b, fhrerr_b, _ = _bin_time_series_numpy(
            data.time, data.flux_hr, data.flux_err_hr,
            dt_seconds=dt_seconds, method=method, t0=t0_ref
        )
        data.time = t_b_lr
        data.flux_lr = flr_b
        data.flux_err_lr = flrerr_b
        data.flux_hr = fhr_b
        data.flux_err_hr = fhrerr_b

    if bin_whitelight:
        print(f"[time_binning wl] {n0_wl} -> {len(data.wl_time)} points (dt={dt_seconds:.1f}s, method={method})")
    if bin_spectroscopic:
        print(f"[time_binning spec] {n0_spec} -> {len(data.time)} points (dt={dt_seconds:.1f}s, method={method})")
    _timebin_debug("post", data.time, data.wl_time, data.flux_lr, data.flux_hr, data.wl_flux)
    return data

DTYPE = jnp.float64


def _pad_spectro_cadences_exact(t, y, yerr, multiple=256):
    """Pad cadence tails with an explicit zero-weight likelihood mask."""
    t = jnp.asarray(t, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    yerr = jnp.asarray(yerr, dtype=jnp.float64)
    original = int(t.shape[0])
    padded = ((original + int(multiple) - 1) // int(multiple)) * int(multiple)
    count = padded - original
    mask = jnp.arange(padded) < original
    if count == 0:
        return t, y, yerr, mask
    cadence = t[-1] - t[-2] if original > 1 else jnp.asarray(1.0)
    tail_t = t[-1] + cadence * jnp.arange(1, count + 1, dtype=jnp.float64)
    padded_t = jnp.concatenate((t, tail_t))
    padded_y = jnp.pad(y, ((0, 0), (0, count)), constant_values=1.0)
    padded_yerr = jnp.pad(yerr, ((0, 0), (0, count)), constant_values=1.0)
    return padded_t, padded_y, padded_yerr, mask

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def jax_bin_lightcurve(time, flux, duration, points_per_transit=20):
    dt = duration / points_per_transit
    t_min = jnp.min(time)
    t_max = jnp.max(time)
    total_range = t_max - t_min
    num_bins = jnp.ceil(total_range / dt).astype(int) + 1
    bin_indices = jnp.clip(((time - t_min) / dt).astype(int), 0, num_bins - 1)
    flux_sums = jnp.zeros(num_bins)
    time_sums = jnp.zeros(num_bins)
    counts = jnp.zeros(num_bins)
    flux_sums = flux_sums.at[bin_indices].add(flux)
    time_sums = time_sums.at[bin_indices].add(time)
    counts = counts.at[bin_indices].add(1.0)
    binned_flux = jnp.where(counts > 0, flux_sums / counts, jnp.nan)
    binned_time = jnp.where(counts > 0, time_sums / counts, jnp.nan)
    return binned_time, binned_flux
    
def plot_noise_binning_robust(residuals, dt_seconds, outpath, title=None):
    """Robust RMS binning plot using MAD-based sigma + MC white noise bands."""
    residuals = np.array(residuals)
    if residuals.ndim == 1:
        residuals = residuals[None, :]
    n_channels, n_times = residuals.shape
    dt = float(dt_seconds)

    all_measured, all_expected, all_betas = [], [], []
    ref_bins_min = None
    for i in range(n_channels):
        beta, bins_min, meas, exp = calculate_beta_metrics(residuals[i], dt)
        all_measured.append(meas)
        all_expected.append(exp)
        all_betas.append(beta)
        if ref_bins_min is None:
            ref_bins_min = bins_min

    median_ch = np.argmin(np.abs(np.array(all_betas) - np.median(all_betas)))
    _, rms_lo1, rms_hi1, rms_lo2, rms_hi2 = run_beta_monte_carlo(residuals[median_ch], dt, n_sims=500)

    measured_stack = np.array(all_measured)
    expected_stack = np.array(all_expected)
    med_measured = np.nanmedian(measured_stack, axis=0) * 1e6
    p16_measured = np.nanpercentile(measured_stack, 16, axis=0) * 1e6
    p84_measured = np.nanpercentile(measured_stack, 84, axis=0) * 1e6
    med_expected = np.nanmedian(expected_stack, axis=0) * 1e6
    beta_median = np.median(all_betas)

    plt.figure(figsize=(7, 5))
    plt.loglog(ref_bins_min, med_expected, 'k--', lw=1.5, label=r'Theory $1/\sqrt{N}$')
    plt.fill_between(ref_bins_min, rms_lo2 * 1e6, rms_hi2 * 1e6, color='gray', alpha=0.2, label=r'White Noise ($2\sigma$)')
    plt.fill_between(ref_bins_min, rms_lo1 * 1e6, rms_hi1 * 1e6, color='gray', alpha=0.4, label=r'White Noise ($1\sigma$)')
    plt.loglog(ref_bins_min, med_measured, color='teal', lw=2, marker='o', ms=4, label=f'Median $\\beta$={beta_median:.2f}')
    plt.fill_between(ref_bins_min, p16_measured, p84_measured, color='teal', alpha=0.2)
    plt.xlabel('Bin Size (minutes)')
    plt.ylabel('RMS (ppm)')
    if title:
        plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, which='both', alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
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
        data["detrended_flux"] = flux - trend_model
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


def _geometry_chain_quality(grouped_samples):
    """Return exact-MCMC quality metrics for white-light geometry sites."""
    aliases = {
        "t0": ("t0_0", "t0"),
        "b": ("b_0", "b"),
        "duration": ("logD_0", "duration_0", "duration", "logD"),
        "rors": ("rors_0", "rors"),
    }
    ess = {}
    rhat = {}
    for label, names in aliases.items():
        name = next((candidate for candidate in names if candidate in grouped_samples), None)
        if name is None:
            continue
        values = jnp.asarray(grouped_samples[name], dtype=jnp.float64)
        if values.ndim < 2:
            values = values.reshape((1, values.shape[0]))
        site_ess = numpyro.diagnostics.effective_sample_size(values)
        ess[label] = float(np.nanmin(np.asarray(jax.device_get(site_ess))))
        if values.shape[0] > 1:
            site_rhat = numpyro.diagnostics.split_gelman_rubin(values)
            rhat[label] = float(np.nanmax(np.asarray(jax.device_get(site_rhat))))
    return ess, rhat


def _continue_mcmc_until_geometry_gate(
    mcmc,
    rng_key,
    run_args,
    run_kwargs,
    *,
    min_ess=400.0,
    max_divergences=0,
    max_extra_blocks=3,
):
    """Append post-warmup draws until the white-light geometry gate passes."""
    sample_blocks = [jax.device_get(mcmc.get_samples(group_by_chain=True))]
    extra_blocks = [jax.device_get(mcmc.get_extra_fields(group_by_chain=True))]
    extra_count = 0
    while True:
        grouped = {
            name: np.concatenate([np.asarray(block[name]) for block in sample_blocks], axis=1)
            for name in sample_blocks[0]
        }
        grouped_extra = {
            name: np.concatenate([np.asarray(block[name]) for block in extra_blocks], axis=1)
            for name in extra_blocks[0]
        }
        ess, rhat = _geometry_chain_quality(grouped)
        divergences = int(np.sum(np.asarray(grouped_extra.get("diverging", []))))
        passes = bool(ess) and min(ess.values()) >= float(min_ess)
        passes = passes and divergences <= int(max_divergences)
        print(
            "White-light quality gate: "
            f"ESS={ess}, rhat={rhat or 'n/a'}, divergences={divergences}, "
            f"pass={passes}, extra_blocks={extra_count}."
        )
        if passes or extra_count >= int(max_extra_blocks):
            flat = {
                name: values.reshape((-1,) + values.shape[2:])
                for name, values in grouped.items()
            }
            flat_extra = {
                name: values.reshape((-1,) + values.shape[2:])
                for name, values in grouped_extra.items()
            }
            return flat, grouped, flat_extra, {
                "whitelight_geometry_bulk_ess": ess,
                "whitelight_geometry_rhat": rhat,
                "whitelight_quality_gate_passed": passes,
                "whitelight_extra_blocks": extra_count,
                "whitelight_total_retained_draws": int(
                    next(iter(flat.values())).shape[0]
                ),
            }
        extra_count += 1
        print(
            "WHITE-LIGHT QUALITY GATE NOT MET; continuing the same warmed "
            f"chain with retained block {extra_count}/{max_extra_blocks}."
        )
        mcmc.post_warmup_state = mcmc.last_state
        rng_key, block_key = jax.random.split(rng_key)
        continuation_kwargs = dict(run_kwargs)
        continuation_kwargs.pop("init_params", None)
        mcmc.run(block_key, *run_args, **continuation_kwargs)
        sample_blocks.append(jax.device_get(mcmc.get_samples(group_by_chain=True)))
        extra_blocks.append(jax.device_get(mcmc.get_extra_fields(group_by_chain=True)))


def _build_numpyro_mcmc(model, init_params, nuts_kwargs, mcmc_kwargs):
    """Build one shape-specialized MCMC runner.

    NumPyro's ``jit_model_args=True`` path makes model values dynamic while
    array shapes remain static.  The same runner can therefore be reused for
    equal-width spectroscopic chunks: every call still performs an independent
    warmup/adaptation, but the expensive NUTS transition program is compiled
    only once for that resident GPU shape.
    """
    nuts_defaults = {
        "regularize_mass_matrix": False,
        "init_strategy": numpyro.infer.init_to_value(values=init_params),
        "target_accept_prob": 0.9,
    }
    nuts_defaults.update(nuts_kwargs)

    mcmc_defaults = {
        "num_warmup": 1000,
        "num_samples": 1000,
        "progress_bar": True,
        "jit_model_args": True,
    }
    mcmc_defaults.update(mcmc_kwargs)
    return (
        numpyro.infer.MCMC(
            numpyro.infer.NUTS(model, **nuts_defaults),
            **mcmc_defaults,
        ),
        nuts_defaults,
        mcmc_defaults,
    )


def get_samples(model, key, t, yerr, indiv_y, init_params, nuts_kwargs=None,
                mcmc_kwargs=None, diagnostics_path=None, _mcmc_runner=None,
                **model_kwargs):
    t = _to_f64(t)
    yerr = _to_f64(yerr)
    indiv_y = _to_f64(indiv_y)
    init_params = _tree_to_f64(init_params)
    model_kwargs = _tree_to_f64(model_kwargs)
    nuts_kwargs = dict(nuts_kwargs or {})
    mcmc_kwargs = dict(mcmc_kwargs or {})

    if _mcmc_runner is None:
        mcmc, nuts_defaults, _ = _build_numpyro_mcmc(
            model, init_params, nuts_kwargs, mcmc_kwargs
        )
    else:
        mcmc = _mcmc_runner
        # The compiled transition program is reusable, but each wavelength
        # block must still start from its own physical initial values. NumPyro
        # stores this strategy on the kernel rather than accepting constrained
        # values in MCMC.run().
        # Match the fresh-runner path exactly: an explicitly supplied
        # strategy takes precedence, otherwise use this block's physical
        # initial values.  ``NUTS.init`` calls ``initialize_model`` on every
        # ``MCMC.run``, so replacing the strategy here does not reuse the
        # previous block's initial state or adaptation.
        mcmc.sampler._init_strategy = nuts_kwargs.get(
            "init_strategy",
            numpyro.infer.init_to_value(values=init_params),
        )
        nuts_defaults = {
            "regularize_mass_matrix": False,
            "target_accept_prob": 0.9,
        }
        nuts_defaults.update(nuts_kwargs)
    mcmc.run(
        key,
        t,
        yerr,
        y=indiv_y,
        extra_fields=("diverging", "accept_prob", "potential_energy", "num_steps"),
        **model_kwargs,
    )
    _save_mcmc_diagnostics(
        mcmc,
        max_tree_depth=nuts_defaults.get("max_tree_depth"),
        output_path=diagnostics_path,
    )
    return mcmc.get_samples()

def _slice_by_channel(value, sl, num_lcs):
    if value is None:
        return None
    if isinstance(value, (np.ndarray, jnp.ndarray)) and value.ndim > 0 and value.shape[0] == num_lcs:
        return value[sl]
    return value


def _take_by_channel(value, channel_indices, num_lcs):
    """Take arbitrary wavelength channels from a channel-varying value."""
    if value is None:
        return None
    if (
        isinstance(value, (np.ndarray, jnp.ndarray))
        and value.ndim > 0
        and value.shape[0] == num_lcs
    ):
        return value[jnp.asarray(channel_indices, dtype=jnp.int32)]
    return value


JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS = (
    "mu_depths",
    "trend_fixed",
    "ld_interpolated",
    "ld_fixed",
    "mu_u_ld",
    "sigma_u_ld",
    "precomputed_yerr_per_lc",
    "trend_prior_mean",
    "trend_prior_scale",
)

HARMONICA_CHANNEL_VARYING_MODEL_KWARGS = (
    "mu_depths",
    "ld_interpolated",
    "ld_fixed",
    "mu_u_ld",
    "sigma_u_ld",
)

CHUNK_CHECKPOINT_SCHEMA_VERSION = 4
CHUNK_CHECKPOINT_TARGET_REVISION = "production-defaults-sampler-swap-v4"
SAMPLING_WORKLOAD_SCHEMA_VERSION = 1
SPECTRO_GRADIENT_DIAGNOSTIC_SCHEMA_VERSION = 1
SCIENCE_ARTIFACT_SCHEMA_VERSION = 1
SCIENCE_ARTIFACT_TARGET_REVISION = "wl-lr-target-2026-08-12"
WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION = 1
WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION = "retained-draw-data-loglik-v1"
SPECTRO_DATA_TARGET_REVISION = "createdatacube-positive-yerr-2026-08-13-v2"
POWER2_LD_CACHE_TARGET_REVISION = "stellar-grid-direct-power2-v2"
SAMPLER_STAGE_INPUT_SCHEMA_VERSION = 1


class SamplerInputsDumpExit(RuntimeError):
    """Internal clean-stop signal after the requested first high-res dump."""


_DUMPED_FIRST_HIGHRES_STAGE = False


def _callable_identity(value):
    return (
        f"{getattr(value, '__module__', type(value).__module__)}."
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


def _tree_to_numpy_for_pickle(value):
    """Copy JAX/NumPy pytrees to host values without changing structure."""
    if isinstance(value, dict):
        return {
            key: _tree_to_numpy_for_pickle(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_tree_to_numpy_for_pickle(item) for item in value)
    if isinstance(value, list):
        return [_tree_to_numpy_for_pickle(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if callable(value):
        # NumPyro initialization strategies are top-level callables/partials
        # and remain pickleable.  Keeping them is necessary for an exact
        # offline replay when a caller explicitly overrides init_strategy.
        return value
    try:
        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError):
        return value
    if array.ndim == 0:
        return array.item()
    return array


def _build_spectroscopic_model(builder, **builder_kwargs):
    """Build a model and retain the exact importable builder invocation."""
    model = builder(**builder_kwargs)
    model.__sampler_input_builder__ = {
        "identity": _callable_identity(builder),
        "kwargs": _tree_to_numpy_for_pickle(builder_kwargs),
        "callable_is_model": False,
    }
    return model


def _sampler_model_builder_spec(model):
    spec = getattr(model, "__sampler_input_builder__", None)
    if spec is not None:
        return _tree_to_numpy_for_pickle(spec)
    # This fallback supports importable model callables in focused tests and
    # external users. Production spectroscopic models always take the builder
    # path above, so their construction kwargs are never inferred.
    return {
        "identity": _callable_identity(model),
        "kwargs": {},
        "callable_is_model": True,
    }


def _sampler_initial_potential(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    nuts_kwargs,
    model_kwargs,
):
    """Evaluate the exact NumPyro potential used to initialize joint NUTS."""
    init_strategy = dict(nuts_kwargs or {}).get(
        "init_strategy",
        numpyro.infer.init_to_value(values=init_params),
    )
    model_info = numpyro.infer.util.initialize_model(
        key,
        model,
        init_strategy=init_strategy,
        dynamic_args=False,
        model_args=(t, yerr),
        model_kwargs={"y": indiv_y, **model_kwargs},
        validate_grad=False,
    )
    potential = model_info.potential_fn(model_info.param_info.z)
    return (
        float(np.asarray(jax.device_get(potential))),
        _tree_to_numpy_for_pickle(model_info.param_info.z),
    )


def _safe_stage_label(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return label or "spectroscopic_stage"


def _maybe_dump_sampler_inputs(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    *,
    nuts_kwargs,
    mcmc_kwargs,
    chunk_size,
    use_chunked,
    sampler_backend,
    channel_varying_kwargs,
    checkpoint_signature,
    dump_metadata,
    model_kwargs,
):
    """Atomically dump one replayable spectroscopic stage when requested."""
    dump_directory = os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
    if not dump_directory:
        return None

    metadata = dict(dump_metadata or {})
    stage_kind = str(
        metadata.get(
            "stage_kind",
            (checkpoint_signature or {}).get("stage", "spectroscopic"),
        )
    )
    stage_label = _safe_stage_label(
        metadata.get("stage_label", stage_kind)
    )
    num_channels = int(np.shape(indiv_y)[0])
    num_cadences = int(np.shape(indiv_y)[1])
    active_cadences = metadata.get("active_window_cadences", num_cadences)
    metadata.update(
        {
            "stage_label": stage_label,
            "stage_kind": stage_kind,
            "num_channels": num_channels,
            "num_cadences": num_cadences,
            "active_window_cadences": int(active_cadences),
        }
    )

    potential, unconstrained_init = _sampler_initial_potential(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        nuts_kwargs,
        model_kwargs,
    )
    effective_chunk_size = (
        int(chunk_size)
        if chunk_size is not None
        else num_channels
    )
    payload = {
        "schema_version": SAMPLER_STAGE_INPUT_SCHEMA_VERSION,
        "model_builder": _sampler_model_builder_spec(model),
        "rng_key": _tree_to_numpy_for_pickle(key),
        "t": _tree_to_numpy_for_pickle(t),
        "yerr": _tree_to_numpy_for_pickle(yerr),
        "indiv_y": _tree_to_numpy_for_pickle(indiv_y),
        "init_params": _tree_to_numpy_for_pickle(init_params),
        "nuts_kwargs": _tree_to_numpy_for_pickle(dict(nuts_kwargs or {})),
        "mcmc_kwargs": _tree_to_numpy_for_pickle(dict(mcmc_kwargs or {})),
        "chunk_size": effective_chunk_size,
        "configured_chunk_size": (
            None if chunk_size is None else int(chunk_size)
        ),
        "use_chunked": bool(use_chunked),
        "sampler_backend": str(sampler_backend),
        "channel_varying_kwargs": tuple(channel_varying_kwargs),
        "model_kwargs": _tree_to_numpy_for_pickle(model_kwargs),
        "checkpoint_signature": _tree_to_numpy_for_pickle(
            checkpoint_signature
        ),
        "initial_potential": potential,
        "unconstrained_init": unconstrained_init,
        "meta": _tree_to_numpy_for_pickle(metadata),
    }

    dump_directory = os.path.abspath(os.path.expanduser(dump_directory))
    os.makedirs(dump_directory, exist_ok=True)
    output_path = os.path.join(
        dump_directory, f"{stage_label}_inputs.pkl"
    )
    temporary_path = f"{output_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary_path, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, output_path)
    print(
        "Dumped replayable spectroscopic sampler inputs to "
        f"{output_path} (initial potential={potential:.17g})",
        flush=True,
    )

    global _DUMPED_FIRST_HIGHRES_STAGE
    is_highres = stage_kind.lower() in {
        "high_resolution", "highres", "high-resolution"
    }
    if (
        is_highres
        and not _DUMPED_FIRST_HIGHRES_STAGE
        and os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS_EXIT", "0") == "1"
    ):
        _DUMPED_FIRST_HIGHRES_STAGE = True
        raise SamplerInputsDumpExit(output_path)
    if is_highres:
        _DUMPED_FIRST_HIGHRES_STAGE = True
    return output_path


def _update_checkpoint_hash(hasher, value):
    """Add a deterministic, type-aware value encoding to ``hasher``."""
    if value is None:
        hasher.update(b"none;")
        return
    if isinstance(value, (str, bytes)):
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        hasher.update(b"text:")
        hasher.update(str(len(encoded)).encode("ascii"))
        hasher.update(b":")
        hasher.update(encoded)
        return
    if isinstance(value, (bool, int, float, np.generic)):
        hasher.update(
            f"scalar:{type(value).__name__}:{value!r};".encode("utf-8")
        )
        return
    if isinstance(value, dict):
        hasher.update(b"dict{")
        for key in sorted(value, key=lambda item: str(item)):
            _update_checkpoint_hash(hasher, str(key))
            _update_checkpoint_hash(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, (tuple, list)):
        hasher.update(f"sequence:{type(value).__name__}[".encode("ascii"))
        for item in value:
            _update_checkpoint_hash(hasher, item)
        hasher.update(b"]")
        return
    if callable(value):
        identity = (
            f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        )
        hasher.update(f"callable:{identity};".encode("utf-8"))
        return

    try:
        array = np.asarray(jax.device_get(value))
    except Exception:
        identity = f"{type(value).__module__}.{type(value).__qualname__}"
        hasher.update(f"object:{identity};".encode("utf-8"))
        return
    contiguous = np.ascontiguousarray(array)
    hasher.update(
        (
            f"array:{contiguous.dtype.str}:"
            f"{','.join(str(size) for size in contiguous.shape)}:"
        ).encode("ascii")
    )
    hasher.update(contiguous.tobytes(order="C"))


def _chunk_checkpoint_fingerprint(
    model,
    rng_key,
    t,
    yerr,
    indiv_y,
    init_params,
    *,
    chunk_size,
    sampler_backend,
    nuts_kwargs,
    mcmc_kwargs,
    channel_varying_kwargs,
    checkpoint_signature,
    model_kwargs,
    spectro_min_depth_ess=0.0,
    spectro_max_divergences=0,
):
    """Fingerprint the posterior target, inputs, and sampling configuration."""
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema": CHUNK_CHECKPOINT_SCHEMA_VERSION,
            "target_revision": CHUNK_CHECKPOINT_TARGET_REVISION,
            "model": (
                f"{getattr(model, '__module__', type(model).__module__)}."
                f"{getattr(model, '__qualname__', type(model).__qualname__)}"
            ),
            # Builder keyword arguments (prior choices such as jitter_prior)
            # must participate so checkpoints from a different model
            # configuration are never reused.
            "model_builder": _sampler_model_builder_spec(model),
            "checkpoint_signature": checkpoint_signature,
            "rng_key": rng_key,
            "chunk_size": int(chunk_size),
            "sampler_backend": str(sampler_backend),
            "nuts_kwargs": dict(nuts_kwargs or {}),
            "mcmc_kwargs": dict(mcmc_kwargs or {}),
            "channel_varying_kwargs": tuple(channel_varying_kwargs),
            "t": t,
            "yerr": yerr,
            "y": indiv_y,
            "init_params": init_params,
            "model_kwargs": model_kwargs,
            "spectro_min_depth_ess": float(spectro_min_depth_ess),
            "spectro_max_divergences": int(spectro_max_divergences),
        },
    )
    return hasher.hexdigest()


def _sampling_workload_fingerprint(
    model,
    t,
    yerr,
    indiv_y,
    init_params,
    *,
    sampler_backend,
    nuts_kwargs,
    channel_varying_kwargs,
    checkpoint_signature,
    model_kwargs,
):
    """Fingerprint the science target independently of draws and batching.

    A short pilot and a production run may intentionally use different random
    keys, warmup/sample counts, or resident widths.  They may not silently use
    different data, priors, model code, or transition-kernel controls.
    """

    repository_root = os.path.dirname(os.path.abspath(__file__))
    relative_sources = (
        "fit_jwst.py",
        "models/channel_batching.py",
        "models/independent_nuts.py",
        "models/independent_hmc.py",
        "models/laplace_is.py",
        "models/trend_marginal.py",
        "models/linear_marginalization.py",
        "models/trends.py",
        "models/jaxoplanet/builder.py",
        "models/jaxoplanet/core.py",
        "models/jaxoplanet/limb_dark_streamed.py",
    )
    source_sha256 = {}
    for relative_path in relative_sources:
        path = os.path.join(repository_root, relative_path)
        if os.path.isfile(path):
            with open(path, "rb") as stream:
                source_sha256[relative_path] = hashlib.sha256(
                    stream.read()
                ).hexdigest()

    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema_version": SAMPLING_WORKLOAD_SCHEMA_VERSION,
            "checkpoint_target_revision": CHUNK_CHECKPOINT_TARGET_REVISION,
            "model": (
                f"{getattr(model, '__module__', type(model).__module__)}."
                f"{getattr(model, '__qualname__', type(model).__qualname__)}"
            ),
            "sampler_backend": str(sampler_backend),
            "nuts_kwargs": dict(nuts_kwargs or {}),
            "channel_varying_kwargs": tuple(channel_varying_kwargs),
            "checkpoint_signature": checkpoint_signature,
            "t": t,
            "yerr": yerr,
            "y": indiv_y,
            "init_params": init_params,
            "model_kwargs": model_kwargs,
            "source_sha256": source_sha256,
        },
    )
    return hasher.hexdigest()


def _annotate_sampler_diagnostics(
    path,
    *,
    sampling_workload_fingerprint,
    sampler_backend,
):
    """Atomically bind a sampler diagnostic to its exact science workload."""

    if path is None or sampling_workload_fingerprint is None:
        return
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Sampler did not write its requested diagnostics file: {path}"
        )
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    payload.update(
        {
            "diagnostic_provenance_schema_version": 1,
            "sampling_workload_fingerprint_sha256": str(
                sampling_workload_fingerprint
            ),
            "sampler_backend": str(sampler_backend),
        }
    )
    def _strict_json_value(value):
        """Represent optional non-finite diagnostics as JSON null."""
        if isinstance(value, dict):
            return {key: _strict_json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_strict_json_value(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    payload = _strict_json_value(payload)
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _write_or_validate_checkpoint_manifest(
    checkpoint_dir,
    logical_prefix,
    effective_prefix,
    fingerprint,
    checkpoint_signature,
):
    manifest_path = os.path.join(
        checkpoint_dir, f"{effective_prefix}.manifest.json"
    )
    payload = {
        "schema_version": CHUNK_CHECKPOINT_SCHEMA_VERSION,
        "logical_prefix": logical_prefix,
        "effective_prefix": effective_prefix,
        "fingerprint_sha256": fingerprint,
        "checkpoint_signature": checkpoint_signature,
    }
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != payload:
            raise RuntimeError(
                "Chunk checkpoint manifest does not match the current fit: "
                f"{manifest_path}"
            )
        return manifest_path

    temporary = f"{manifest_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, manifest_path)
    return manifest_path


def _science_artifact_fingerprint(stage, payload):
    """Fingerprint cached preprocessing products that feed a later fit."""
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema": SCIENCE_ARTIFACT_SCHEMA_VERSION,
            "target_revision": SCIENCE_ARTIFACT_TARGET_REVISION,
            "stage": str(stage),
            "payload": payload,
        },
    )
    return hasher.hexdigest()


def _science_artifact_manifest_matches(path, fingerprint):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        payload.get("schema_version") == SCIENCE_ARTIFACT_SCHEMA_VERSION
        and payload.get("target_revision") == SCIENCE_ARTIFACT_TARGET_REVISION
        and payload.get("fingerprint_sha256") == fingerprint
    )


def _write_science_artifact_manifest(path, stage, fingerprint):
    payload = {
        "schema_version": SCIENCE_ARTIFACT_SCHEMA_VERSION,
        "target_revision": SCIENCE_ARTIFACT_TARGET_REVISION,
        "stage": str(stage),
        "fingerprint_sha256": fingerprint,
    }
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _posterior_num_draws(posterior_samples):
    """Validate a flattened posterior sample mapping and return its draw count."""
    if not posterior_samples:
        raise ValueError("Posterior samples are empty.")
    draw_counts = {}
    for name, values in posterior_samples.items():
        array = np.asarray(jax.device_get(values))
        if array.ndim == 0:
            raise ValueError(
                f"Posterior site '{name}' is missing a leading draw dimension."
            )
        draw_counts[name] = int(array.shape[0])
    unique_counts = set(draw_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            "Posterior sites have inconsistent draw counts: "
            f"{draw_counts}."
        )
    num_draws = unique_counts.pop()
    if num_draws < 1:
        raise ValueError("Posterior samples contain no retained draws.")
    return num_draws


def _sum_data_log_likelihood_per_draw(
    model,
    posterior_samples,
    *model_args,
    batch_size=64,
    log_likelihood_fn=None,
    **model_kwargs,
):
    """Return the summed observed-data log likelihood for every retained draw.

    This deliberately evaluates NumPyro observation sites rather than reusing
    NUTS ``potential_energy``, which also contains priors and unconstrained-
    coordinate Jacobians.  Evaluation is chunked to bound white-light memory.
    """
    if log_likelihood_fn is None:
        from numpyro.infer.util import log_likelihood as log_likelihood_fn

    num_draws = _posterior_num_draws(posterior_samples)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("White-light log-likelihood batch_size must be >= 1.")

    totals = np.empty(num_draws, dtype=np.float64)
    for start in range(0, num_draws, batch_size):
        end = min(start + batch_size, num_draws)
        sample_batch = {
            name: values[start:end]
            for name, values in posterior_samples.items()
        }
        pointwise = log_likelihood_fn(
            model,
            sample_batch,
            *model_args,
            batch_ndims=1,
            **model_kwargs,
        )
        if not pointwise:
            raise RuntimeError(
                "White-light model exposed no observed likelihood sites."
            )

        batch_total = np.zeros(end - start, dtype=np.float64)
        for site_name, site_values in pointwise.items():
            site_array = np.asarray(jax.device_get(site_values), dtype=np.float64)
            if site_array.ndim == 0 or site_array.shape[0] != end - start:
                raise RuntimeError(
                    "Unexpected log-likelihood shape for observation site "
                    f"'{site_name}': {site_array.shape}."
                )
            batch_total += site_array.reshape(end - start, -1).sum(axis=1)
        totals[start:end] = batch_total
    return totals


def _select_max_likelihood_retained_draw(
    posterior_samples,
    summed_log_likelihood,
    *,
    num_chains=1,
):
    """Select one coherent finite posterior draw by observed-data likelihood."""
    num_draws = _posterior_num_draws(posterior_samples)
    scores = np.asarray(summed_log_likelihood, dtype=np.float64)
    if scores.shape != (num_draws,):
        raise ValueError(
            "summed_log_likelihood must contain exactly one value per retained "
            f"draw; got {scores.shape} for {num_draws} draws."
        )

    finite_draw = np.isfinite(scores)
    for values in posterior_samples.values():
        array = np.asarray(jax.device_get(values))
        finite_draw &= np.isfinite(array).reshape(num_draws, -1).all(axis=1)
    if not np.any(finite_draw):
        raise RuntimeError(
            "No retained white-light draw has finite samples and finite data "
            "log likelihood."
        )

    safe_scores = np.where(finite_draw, scores, -np.inf)
    flat_index = int(np.argmax(safe_scores))
    num_chains = int(num_chains)
    if num_chains < 1 or num_draws % num_chains:
        raise ValueError(
            f"Cannot map {num_draws} flattened draws onto {num_chains} chains."
        )
    draws_per_chain = num_draws // num_chains
    selected = {
        name: values[flat_index:flat_index + 1]
        for name, values in posterior_samples.items()
    }
    return {
        "flat_draw_index": flat_index,
        "chain_index": flat_index // draws_per_chain,
        "draw_index": flat_index % draws_per_chain,
        "summed_data_log_likelihood": float(scores[flat_index]),
        "samples": selected,
    }


def _geometry_from_white_light_samples(
    samples,
    derive_geometry,
    period,
    *,
    ecc=0.0,
    omega=0.0,
):
    """Extract a complete, internally consistent geometry from sample draws."""
    periods = np.atleast_1d(np.asarray(period, dtype=np.float64))
    eccs = np.atleast_1d(np.asarray(ecc, dtype=np.float64))
    omegas = np.atleast_1d(np.asarray(omega, dtype=np.float64))
    if eccs.size == 1 and periods.size > 1:
        eccs = np.repeat(eccs, periods.size)
    if omegas.size == 1 and periods.size > 1:
        omegas = np.repeat(omegas, periods.size)

    derived = derive_geometry(samples, periods, ecc=eccs, omega=omegas)
    geometry = {
        name: []
        for name in (
            "period", "duration", "t0", "b", "a_rs", "cos_i",
            "inclination", "rors",
        )
    }
    for planet_index, period_value in enumerate(periods):
        required = {
            "duration": derived.get(f"duration_{planet_index}"),
            "t0": samples.get(f"t0_{planet_index}"),
            "b": derived.get(f"b_{planet_index}"),
            "a_rs": derived.get(f"a_rs_{planet_index}"),
            "cos_i": derived.get(f"cos_i_{planet_index}"),
            "rors": samples.get(f"rors_{planet_index}"),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                f"White-light draw is missing geometry for planet "
                f"{planet_index}: {missing}."
            )
        values = {
            name: float(np.asarray(jax.device_get(value)).reshape(-1)[0])
            for name, value in required.items()
        }
        inclination = derived.get(f"inc_{planet_index}")
        if inclination is None:
            inclination_value = float(
                np.arccos(np.clip(values["cos_i"], 0.0, 1.0))
            )
        else:
            inclination_value = float(
                np.asarray(jax.device_get(inclination)).reshape(-1)[0]
            )
        values["inclination"] = inclination_value
        values["period"] = float(period_value)
        if not np.all(np.isfinite(list(values.values()))):
            raise RuntimeError(
                f"Selected white-light geometry for planet {planet_index} "
                "contains non-finite values."
            )
        for name in geometry:
            geometry[name].append(values[name])
    return geometry


def _geometry_from_white_light_medians(bestfit_params, period):
    """Build the compatibility handoff from the reported posterior medians."""
    periods = np.atleast_1d(np.asarray(period, dtype=np.float64))
    geometry = {
        "period": periods.tolist(),
        "duration": np.atleast_1d(np.asarray(bestfit_params["duration"], dtype=float)).tolist(),
        "t0": np.atleast_1d(np.asarray(bestfit_params["t0"], dtype=float)).tolist(),
        "b": np.atleast_1d(np.asarray(bestfit_params["b"], dtype=float)).tolist(),
        "a_rs": np.atleast_1d(np.asarray(bestfit_params["a_rs"], dtype=float)).tolist(),
        "cos_i": np.atleast_1d(np.asarray(bestfit_params["cos_i"], dtype=float)).tolist(),
        "rors": np.atleast_1d(np.asarray(bestfit_params["rors"], dtype=float)).tolist(),
    }
    geometry["inclination"] = np.arccos(
        np.clip(np.asarray(geometry["cos_i"], dtype=float), 0.0, 1.0)
    ).tolist()
    lengths = {name: len(values) for name, values in geometry.items()}
    if set(lengths.values()) != {len(periods)}:
        raise RuntimeError(
            "White-light median geometry has inconsistent planet dimensions: "
            f"{lengths}."
        )
    if not all(np.all(np.isfinite(values)) for values in geometry.values()):
        raise RuntimeError("White-light median geometry contains non-finite values.")
    return geometry


def _selected_geometry_primitives(samples, num_planets):
    """Serialize the primitive coordinates that identify a selected draw."""
    names = []
    for planet_index in range(int(num_planets)):
        names.extend(
            f"{base}_{planet_index}"
            for base in (
                "t0", "rors", "_b", "logD", "log_a_rs", "duration",
                "b", "a_rs", "cos_i", "inc",
            )
        )
    primitives = {}
    for name in names:
        if name not in samples:
            continue
        value = np.asarray(jax.device_get(samples[name]))
        primitives[name] = (
            float(value.reshape(-1)[0])
            if value.size == 1
            else value.reshape(-1).astype(float).tolist()
        )
    return primitives


def _whitelight_geometry_handoff_fingerprint(payload):
    hasher = hashlib.sha256()
    _update_checkpoint_hash(
        hasher,
        {
            "schema": WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION,
            "target_revision": WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION,
            "payload": payload,
        },
    )
    return hasher.hexdigest()


def _write_whitelight_geometry_handoff(path, payload):
    """Atomically persist a self-validating white-light geometry handoff."""
    payload = dict(payload)
    payload["schema_version"] = WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION
    payload["target_revision"] = WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint_sha256"
    }
    payload["artifact_fingerprint_sha256"] = (
        _whitelight_geometry_handoff_fingerprint(fingerprint_payload)
    )
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return payload


def _load_whitelight_geometry_handoff(
    path,
    *,
    expected_posterior_fingerprint,
    expected_estimator,
):
    """Load a handoff only if its schema, fingerprint, and target all match."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version")
        != WHITELIGHT_GEOMETRY_HANDOFF_SCHEMA_VERSION
        or payload.get("target_revision")
        != WHITELIGHT_GEOMETRY_HANDOFF_TARGET_REVISION
        or payload.get("posterior_fingerprint_sha256")
        != expected_posterior_fingerprint
        or payload.get("estimator") != expected_estimator
    ):
        return None
    expected_artifact_fingerprint = payload.get("artifact_fingerprint_sha256")
    fingerprint_payload = {
        key: value
        for key, value in payload.items()
        if key != "artifact_fingerprint_sha256"
    }
    if expected_artifact_fingerprint != _whitelight_geometry_handoff_fingerprint(
        fingerprint_payload
    ):
        return None
    geometry = payload.get("geometry")
    required = {
        "period", "duration", "t0", "b", "a_rs", "cos_i",
        "inclination", "rors",
    }
    if not isinstance(geometry, dict) or not required.issubset(geometry):
        return None
    try:
        lengths = {
            len(np.atleast_1d(np.asarray(geometry[name], dtype=float)))
            for name in required
        }
        finite = all(
            np.all(np.isfinite(np.asarray(geometry[name], dtype=float)))
            for name in required
        )
    except (TypeError, ValueError):
        return None
    if len(lengths) != 1 or next(iter(lengths), 0) < 1 or not finite:
        return None
    if expected_estimator == "max_likelihood_draw":
        try:
            num_retained_draws = int(payload["num_retained_draws"])
            num_chains = int(payload["num_chains"])
            flat_draw_index = int(payload["selected_flat_draw_index"])
            chain_index = int(payload["selected_chain_index"])
            draw_index = int(payload["selected_draw_index"])
            summed_log_likelihood = float(
                payload["summed_data_log_likelihood"]
            )
        except (KeyError, TypeError, ValueError):
            return None
        if (
            num_retained_draws < 1
            or num_chains < 1
            or num_retained_draws % num_chains
            or flat_draw_index < 0
            or flat_draw_index >= num_retained_draws
            or chain_index != flat_draw_index // (num_retained_draws // num_chains)
            or draw_index != flat_draw_index % (num_retained_draws // num_chains)
            or not np.isfinite(summed_log_likelihood)
        ):
            return None
    elif any(
        payload.get(name) is not None
        for name in (
            "selected_flat_draw_index", "selected_chain_index",
            "selected_draw_index", "summed_data_log_likelihood",
        )
    ):
        return None
    return payload


def _file_content_identity(path, block_size=8 * 1024 * 1024):
    """Return a stable source identity including a complete content digest."""
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    before = os.stat(resolved)
    digest = hashlib.sha256()
    with open(resolved, "rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    after = os.stat(resolved)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"Source file changed while hashing: {resolved}")
    return {
        "realpath": resolved,
        "size": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _optional_file_content_identity(path):
    if path is None:
        return None
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    if not os.path.isfile(resolved):
        return {"realpath": resolved, "missing": True}
    return _file_content_identity(resolved)


def _directory_metadata_identity(path):
    """Fingerprint a model-grid tree without rereading multi-GB grid files."""
    resolved = os.path.realpath(os.path.expanduser(str(path)))
    if not os.path.isdir(resolved):
        return {"realpath": resolved, "missing": True}
    entries = []
    for root, dirs, files in os.walk(resolved):
        dirs.sort()
        for name in sorted(files):
            full_path = os.path.join(root, name)
            stat = os.stat(full_path)
            entries.append(
                (
                    os.path.relpath(full_path, resolved),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                )
            )
    hasher = hashlib.sha256()
    _update_checkpoint_hash(hasher, entries)
    return {
        "realpath": resolved,
        "file_count": len(entries),
        "metadata_sha256": hasher.hexdigest(),
    }


def _atomic_save_npy(path, values):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npy"
    np.save(temporary, values)
    os.replace(temporary, path)


def _atomic_savez(path, **values):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
    np.savez(temporary, **values)
    os.replace(temporary, path)


def _atomic_savez_compressed(path, **values):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}.npz"
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _atomic_dataframe_csv(frame, path, *, index=False):
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    frame.to_csv(temporary, index=index)
    os.replace(temporary, path)


def _chunk_ranges_for(num_lcs, chunk_size):
    return [
        (start, min(start + chunk_size, num_lcs))
        for start in range(0, num_lcs, chunk_size)
    ]


def _chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end):
    return os.path.join(checkpoint_dir, f"{checkpoint_prefix}_chunk_{start}_{end}.pkl")


class _SamplerSamples(dict):
    """Sample mapping carrying non-array per-channel sampler provenance."""

    def __init__(self, *args, sampler_used=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_used = None if sampler_used is None else list(sampler_used)


def _load_chunk_samples(chunk_file):
    with open(chunk_file, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "samples" in payload and "sampler_used" in payload:
        return _SamplerSamples(payload["samples"], sampler_used=payload["sampler_used"])
    return payload


def _gradient_diagnostic_path(chunk_file):
    return os.path.splitext(chunk_file)[0] + ".gradient.json"


def _load_current_gradient_diagnostic(path, checkpoint_fingerprint):
    """Return a diagnostic only when it belongs to the current target."""
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(result, dict):
        return None
    if (
        result.get("diagnostic_schema_version")
        != SPECTRO_GRADIENT_DIAGNOSTIC_SCHEMA_VERSION
        or result.get("checkpoint_fingerprint_sha256")
        != checkpoint_fingerprint
        or not isinstance(result.get("passed"), bool)
        or not isinstance(result.get("all_finite"), bool)
    ):
        return None
    return result


def _run_or_validate_gradient_diagnostic(
    model,
    init_chunk,
    t,
    yerr_chunk,
    y_chunk,
    *,
    model_kwargs,
    gradient_path,
    checkpoint_fingerprint,
    strict,
    chunk_label,
):
    """Run once per requested target, reusing only a fingerprinted result."""
    cached = _load_current_gradient_diagnostic(
        gradient_path, checkpoint_fingerprint
    )
    if cached is not None:
        print(
            f"  chunk {chunk_label} - gradient diagnostic "
            f"{'PASSED' if cached['passed'] else 'WARNING'} (validated cache)"
        )
        if strict and not cached["passed"]:
            raise RuntimeError(
                f"Spectroscopic gradient diagnostic failed for chunk {chunk_label}."
            )
        return cached

    try:
        from diagnose_nuts_gradient import diagnose_gradient_quality

        result = diagnose_gradient_quality(
            model,
            init_chunk,
            t,
            yerr_chunk,
            y_chunk,
            model_kwargs=model_kwargs,
            num_directions=3,
        )
        result = dict(result)
        result["diagnostic_schema_version"] = (
            SPECTRO_GRADIENT_DIAGNOSTIC_SCHEMA_VERSION
        )
        result["checkpoint_fingerprint_sha256"] = checkpoint_fingerprint
        print(
            f"  chunk {chunk_label} - gradient diagnostic "
            f"{'PASSED' if result['passed'] else 'WARNING'} "
            f"(finite={result['all_finite']}, "
            "max directional relative error="
            f"{result['max_directional_relative_error']:.3e})"
        )
        if gradient_path is not None:
            temporary = (
                f"{gradient_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
            os.replace(temporary, gradient_path)
        if strict and not result["passed"]:
            raise RuntimeError(
                f"Spectroscopic gradient diagnostic failed for chunk {chunk_label}."
            )
        return result
    except Exception as error:
        if strict:
            raise
        print(
            f"  chunk {chunk_label} - gradient diagnostic unavailable: {error}"
        )
        return None


def _concatenate_chunk_samples(samples_chunks):
    samples = {}
    for key in samples_chunks[0].keys():
        arrays = [chunk[key] for chunk in samples_chunks]
        samples[key] = np.concatenate(arrays, axis=1)
    sampler_used = []
    for chunk in samples_chunks:
        provenance = getattr(chunk, "sampler_used", None)
        if provenance is None:
            sampler_used = []
            break
        sampler_used.extend(provenance)
    return _SamplerSamples(samples, sampler_used=sampler_used or None)


def _spectro_failed_lanes(samples, diagnostics_path, min_depth_ess, max_divergences):
    """Return failed lanes and auditable gate metrics for an exact-MCMC run."""
    depth_values = samples.get("depths")
    if depth_values is None and "rors" in samples:
        depth_values = jnp.asarray(samples["rors"]) ** 2
    if depth_values is None:
        return np.asarray([], dtype=int), {"depth_ess_per_channel": None,
                                           "num_divergences_per_channel": None}
    depth_values = np.asarray(jax.device_get(depth_values))
    lane_ess = []
    for lane in range(depth_values.shape[1]):
        value = jnp.asarray(depth_values[:, lane])
        ess = numpyro.diagnostics.effective_sample_size(value[None, ...])
        lane_ess.append(float(np.nanmin(np.asarray(jax.device_get(ess)))))
    divergences = np.zeros(depth_values.shape[1], dtype=int)
    if diagnostics_path is not None and os.path.isfile(diagnostics_path):
        with open(diagnostics_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        per_lane = payload.get("num_divergences_per_channel")
        if per_lane is not None and len(per_lane) == depth_values.shape[1]:
            divergences = np.asarray(per_lane, dtype=int)
        elif int(payload.get("num_divergences", 0)) > int(max_divergences):
            # A joint diagnostic cannot localize the event, so conservatively
            # reject every lane in that selective attempt.
            divergences[:] = int(payload["num_divergences"])
    failed = np.flatnonzero(
        (np.asarray(lane_ess) < float(min_depth_ess))
        | (divergences > int(max_divergences))
    )
    return failed, {
        "depth_ess_per_channel": lane_ess,
        "num_divergences_per_channel": divergences.tolist(),
    }


def _spectro_sampler_swap_order(primary):
    if primary == "independent_nuts":
        return ("independent_hmc", "joint_nuts")
    if primary == "independent_hmc":
        return ("independent_nuts", "joint_nuts")
    return ()


def _resolve_parallel_chunk_job(parallel_job_count=None, parallel_job_index=None):
    if parallel_job_count is None:
        env_count = os.getenv("JWSTJAXFIT_CHUNK_TASK_COUNT")
        if env_count is None:
            env_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if env_count is not None:
            parallel_job_count = int(env_count)

    if parallel_job_index is None:
        env_idx = os.getenv("JWSTJAXFIT_CHUNK_TASK_ID")
        if env_idx is not None:
            parallel_job_index = int(env_idx)
        else:
            slurm_idx = os.getenv("SLURM_ARRAY_TASK_ID")
            if slurm_idx is not None:
                slurm_min = int(os.getenv("SLURM_ARRAY_TASK_MIN", slurm_idx))
                parallel_job_index = int(slurm_idx) - slurm_min

    return parallel_job_count, parallel_job_index

def get_samples_chunked(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    chunk_size,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    chunk_mode="serial",
    parallel_job_count=None,
    parallel_job_index=None,
    output_dir=None,
    checkpoint_prefix=None,
    sampler_backend="joint_nuts",
    laplace_is_kwargs=None,
    channel_varying_kwargs=(),
    checkpoint_signature=None,
    sampling_workload_fingerprint=None,
    gradient_diagnostic_mode="off",
    gradient_diagnostic_strict=False,
    spectro_min_depth_ess=0.0,
    spectro_max_divergences=0,
    _joint_mcmc_runner_cache=None,
    _independent_mcmc_runner_cache=None,
    **model_kwargs,
):
    """
    Run MCMC in chunks with checkpoint support for wall-time resilience.

    Parameters:
    -----------
    output_dir : str, optional
        Directory to save checkpoint files. If None, no checkpointing is used.
    checkpoint_prefix : str, optional
        Prefix for checkpoint filenames (e.g., 'WASP-39_PRISM_native').
        Checkpoints saved as: {output_dir}/chunks/{checkpoint_prefix}_chunk_{start}_{end}.pkl
    """
    num_lcs = indiv_y.shape[0]
    chunk_mode = str(chunk_mode).lower()
    gradient_diagnostic_mode = str(gradient_diagnostic_mode).lower()
    if gradient_diagnostic_mode not in {"off", "first", "each"}:
        raise ValueError(
            "gradient_diagnostic_mode must be one of {'off', 'first', 'each'}."
        )
    if gradient_diagnostic_strict and gradient_diagnostic_mode == "off":
        raise ValueError(
            "gradient_diagnostic_strict=True requires "
            "gradient_diagnostic_mode='first' or 'each'."
        )
    if sampler_backend != "joint_nuts" and checkpoint_prefix is not None:
        # Never silently reuse samples made by a different transition kernel
        # when comparing or switching sampler backends.
        checkpoint_prefix = f"{checkpoint_prefix}_{sampler_backend}"
    if chunk_mode not in {"serial", "parallel", "combine"}:
        raise ValueError(
            "chunk_mode must be one of {'serial', 'parallel', 'combine'}; "
            f"received {chunk_mode!r}."
        )
    print(
        f"Running chunked MCMC: {num_lcs} channels in blocks of {chunk_size} "
        f"(mode={chunk_mode})"
    )

    # Set up checkpointing
    use_checkpoints = (output_dir is not None and checkpoint_prefix is not None)
    checkpoint_fingerprint = None
    if use_checkpoints:
        checkpoint_dir = os.path.join(output_dir, 'chunks')
        os.makedirs(checkpoint_dir, exist_ok=True)
        logical_checkpoint_prefix = checkpoint_prefix
        checkpoint_fingerprint = _chunk_checkpoint_fingerprint(
            model,
            jax.random.key_data(key),
            t,
            yerr,
            indiv_y,
            init_params,
            chunk_size=chunk_size,
            sampler_backend=sampler_backend,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            channel_varying_kwargs=channel_varying_kwargs,
            checkpoint_signature=checkpoint_signature,
            model_kwargs=model_kwargs,
            spectro_min_depth_ess=spectro_min_depth_ess,
            spectro_max_divergences=spectro_max_divergences,
        )
        checkpoint_prefix = (
            f"{logical_checkpoint_prefix}_cfg{checkpoint_fingerprint[:12]}"
        )
        manifest_path = _write_or_validate_checkpoint_manifest(
            checkpoint_dir,
            logical_checkpoint_prefix,
            checkpoint_prefix,
            checkpoint_fingerprint,
            checkpoint_signature,
        )
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(
            "Checkpoint fingerprint: "
            f"{checkpoint_fingerprint[:12]} ({manifest_path})"
        )
    elif chunk_mode != "serial":
        raise ValueError("chunk_mode='parallel' or 'combine' requires output_dir and checkpoint_prefix.")

    all_chunk_ranges = _chunk_ranges_for(num_lcs, chunk_size)

    if chunk_mode == "parallel":
        parallel_job_count, parallel_job_index = _resolve_parallel_chunk_job(
            parallel_job_count=parallel_job_count,
            parallel_job_index=parallel_job_index,
        )
        if parallel_job_count is None or parallel_job_index is None:
            raise ValueError(
                "chunk_mode='parallel' requires chunk_parallel_job_count/chunk_parallel_job_index "
                "or SLURM/JWSTJAXFIT chunk task environment variables."
            )
        if parallel_job_count < 1:
            raise ValueError("chunk_parallel_job_count must be >= 1.")
        if not (0 <= parallel_job_index < parallel_job_count):
            raise ValueError(
                "chunk_parallel_job_index must satisfy "
                f"0 <= index < count; received index={parallel_job_index}, "
                f"count={parallel_job_count}."
            )
        target_chunk_indices = [
            idx for idx in range(len(all_chunk_ranges))
            if idx % parallel_job_count == parallel_job_index
        ]
        print(
            f"Parallel chunk job {parallel_job_index + 1}/{parallel_job_count} "
            f"assigned chunk indices {target_chunk_indices}"
        )
        if not target_chunk_indices:
            print("No chunks assigned to this job; exiting without computation.")
            return None
    else:
        target_chunk_indices = list(range(len(all_chunk_ranges)))

    if chunk_mode == "combine":
        missing = []
        corrupt = []
        samples_chunks = []
        for start, end in all_chunk_ranges:
            chunk_file = _chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end)
            if not os.path.exists(chunk_file):
                missing.append((start, end))
                continue
            try:
                samples_chunks.append(_load_chunk_samples(chunk_file))
            except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
                corrupt.append((start, end, error))
        if missing:
            raise FileNotFoundError(
                "Cannot combine chunk checkpoints because the following chunk files are missing: "
                + ", ".join(f"{start}:{end}" for start, end in missing)
            )
        if corrupt:
            raise RuntimeError(
                "Cannot combine corrupt chunk checkpoint(s): "
                + ", ".join(
                    f"{start}:{end} ({type(error).__name__})"
                    for start, end, error in corrupt
                )
            )
        if gradient_diagnostic_strict:
            required_indices = (
                range(len(all_chunk_ranges))
                if gradient_diagnostic_mode == "each"
                else range(min(1, len(all_chunk_ranges)))
            )
            invalid = []
            for chunk_idx in required_indices:
                start, end = all_chunk_ranges[chunk_idx]
                chunk_file = _chunk_checkpoint_path(
                    checkpoint_dir, checkpoint_prefix, start, end
                )
                result = _load_current_gradient_diagnostic(
                    _gradient_diagnostic_path(chunk_file),
                    checkpoint_fingerprint,
                )
                if result is None or not result["passed"]:
                    invalid.append((start, end))
            if invalid:
                raise RuntimeError(
                    "Strict spectroscopic gradient diagnostics are missing, "
                    "stale, or failed for chunk(s): "
                    + ", ".join(f"{start}:{end}" for start, end in invalid)
                )
        print(f"\nConcatenating {len(samples_chunks)} chunks...")
        samples = _concatenate_chunk_samples(samples_chunks)
        print(f"All chunks complete! Final shape: {list(samples.values())[0].shape}")
        return samples

    samples_chunks = []
    # One runner per exact resident/padded width. Every numerical chunk value
    # remains a dynamic argument; only the shape-specialized transition
    # programs are reused.
    joint_mcmc_runners = (
        {}
        if _joint_mcmc_runner_cache is None
        else _joint_mcmc_runner_cache
    )
    independent_mcmc_runners = (
        {}
        if _independent_mcmc_runner_cache is None
        else _independent_mcmc_runner_cache
    )

    # Derive every chunk's stream from its global index.  This makes serial,
    # resumed, and array-parallel execution assign exactly the same distinct
    # key to a given chunk, independent of which earlier checkpoints exist.
    base_key = key
    for chunk_idx in target_chunk_indices:
        start, end = all_chunk_ranges[chunk_idx]
        sl = slice(start, end)

        init_chunk = {
            k: _slice_by_channel(v, sl, num_lcs)
            for k, v in init_params.items()
        }
        channel_varying_set = frozenset(channel_varying_kwargs)
        kwargs_chunk = {
            k: (
                _slice_by_channel(v, sl, num_lcs)
                if k in channel_varying_set else v
            )
            for k, v in model_kwargs.items()
        }
        yerr_chunk = yerr[sl]
        y_chunk = indiv_y[sl]
        chunk_file = None
        if use_checkpoints:
            chunk_file = _chunk_checkpoint_path(
                checkpoint_dir, checkpoint_prefix, start, end
            )

        run_gradient_diagnostic = (
            gradient_diagnostic_mode == "each"
            or (gradient_diagnostic_mode == "first" and chunk_idx == 0)
        )
        if run_gradient_diagnostic:
            gradient_path = (
                _gradient_diagnostic_path(chunk_file)
                if chunk_file is not None else None
            )
            _run_or_validate_gradient_diagnostic(
                model,
                init_chunk,
                t,
                yerr_chunk,
                y_chunk,
                model_kwargs=kwargs_chunk,
                gradient_path=gradient_path,
                checkpoint_fingerprint=checkpoint_fingerprint,
                strict=gradient_diagnostic_strict,
                chunk_label=f"{start}:{end}",
            )

        # Check if this chunk already exists
        if use_checkpoints:
            if os.path.exists(chunk_file):
                print(f"  chunk {start}:{end} - LOADING from checkpoint")
                try:
                    samples_chunk = _load_chunk_samples(chunk_file)
                except (OSError, EOFError, pickle.UnpicklingError, ValueError) as error:
                    print(
                        f"  chunk {start}:{end} - checkpoint unreadable "
                        f"({type(error).__name__}); recomputing atomically"
                    )
                else:
                    samples_chunks.append(samples_chunk)
                    continue  # Skip computation for this chunk

        # Compute this chunk
        key_chunk = jax.random.fold_in(base_key, chunk_idx)
        print(f"  chunk {start}:{end} - COMPUTING ({end - start} channels)")
        diagnostics_path = (
            os.path.splitext(chunk_file)[0] + ".diagnostics.json"
            if use_checkpoints else None
        )
        if sampler_backend in {
            "independent_nuts", "independent_hmc", "laplace_is"
        }:
            from models.independent_nuts import (
                build_independent_nuts_runner,
                get_samples_independent,
            )

            varying_for_chunk = tuple(
                name for name in channel_varying_kwargs
                if name in kwargs_chunk
            )
            if sampler_backend == "independent_hmc":
                from models.independent_hmc import (
                    build_independent_hmc_runner,
                    get_samples_independent_hmc,
                )
                independent_sampler = get_samples_independent_hmc
                independent_builder = build_independent_hmc_runner
                sampler_label = "fixed-step HMC"
                backend_options = {}
            elif sampler_backend == "laplace_is":
                from models.laplace_is import (
                    build_laplace_is_runner,
                    get_samples_laplace_is,
                )
                independent_sampler = get_samples_laplace_is
                independent_builder = build_laplace_is_runner
                sampler_label = "Laplace-IS"
                backend_options = dict(laplace_is_kwargs or {})
            else:
                independent_sampler = get_samples_independent
                independent_builder = build_independent_nuts_runner
                sampler_label = "NUTS"
                backend_options = {}
            padded_width = int(chunk_size)
            runner_key = (
                sampler_backend,
                padded_width,
                varying_for_chunk,
            )
            runner = independent_mcmc_runners.get(runner_key)
            if runner is None:
                runner = independent_builder(
                    model,
                    nuts_kwargs=nuts_kwargs,
                    mcmc_kwargs=mcmc_kwargs,
                    lane_width=padded_width,
                    channel_varying_kwargs=varying_for_chunk,
                    **backend_options,
                )
                independent_mcmc_runners[runner_key] = runner
            else:
                print(
                    f"  chunk {start}:{end} - reusing compiled independent "
                    f"{sampler_label} runner (padded width {padded_width})"
                )
            print(
                f"  chunk {start}:{end} - independent GPU {sampler_label} "
                f"({end - start} active lanes, padded width {padded_width})"
            )
            samples_chunk = independent_sampler(
                model,
                key_chunk,
                t,
                yerr_chunk,
                y_chunk,
                init_chunk,
                nuts_kwargs=nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs,
                diagnostics_path=diagnostics_path,
                lane_width=padded_width,
                channel_varying_kwargs=varying_for_chunk,
                _runner=runner,
                **backend_options,
                **kwargs_chunk,
            )
        elif sampler_backend == "joint_nuts":
            runner = None
            if bool(dict(mcmc_kwargs or {}).get("jit_model_args", True)):
                resident_width = end - start
                runner = joint_mcmc_runners.get(resident_width)
                if runner is None:
                    runner, _, _ = _build_numpyro_mcmc(
                        model,
                        _tree_to_f64(init_chunk),
                        dict(nuts_kwargs or {}),
                        dict(mcmc_kwargs or {}),
                    )
                    joint_mcmc_runners[resident_width] = runner
                else:
                    print(
                        f"  chunk {start}:{end} - reusing compiled "
                        f"joint-NUTS runner (width {resident_width})"
                    )
            samples_chunk = get_samples(
                model,
                key_chunk,
                t,
                yerr_chunk,
                y_chunk,
                init_chunk,
                nuts_kwargs=nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs,
                diagnostics_path=diagnostics_path,
                _mcmc_runner=runner,
                **kwargs_chunk,
            )
        else:
            raise ValueError(
                "sampler_backend must be one of "
                "{'joint_nuts', 'independent_nuts', 'independent_hmc', "
                "'laplace_is'}; "
                f"received {sampler_backend!r}."
            )
        sampler_used = [sampler_backend] * (end - start)
        gate_attempts = []
        if sampler_backend in {"independent_nuts", "independent_hmc"}:
            failed, gate = _spectro_failed_lanes(
                samples_chunk, diagnostics_path, spectro_min_depth_ess,
                spectro_max_divergences,
            )
            gate_attempts.append({"sampler": sampler_backend, **gate})
            for attempt_index, fallback_backend in enumerate(
                _spectro_sampler_swap_order(sampler_backend), start=1
            ):
                if not failed.size:
                    break
                selected = failed.copy()
                print(
                    f"  chunk {start}:{end} - gate failed in local lanes "
                    f"{selected.tolist()}; re-running only those lanes with "
                    f"{fallback_backend}."
                )

                def _select_lanes(value):
                    try:
                        array = jnp.asarray(value)
                    except (TypeError, ValueError):
                        return value
                    if array.ndim and array.shape[0] == end - start:
                        return array[selected]
                    return value

                fallback_init = {name: _select_lanes(value)
                                 for name, value in init_chunk.items()}
                fallback_kwargs = {
                    name: (_select_lanes(value)
                           if name in channel_varying_set else value)
                    for name, value in kwargs_chunk.items()
                }
                fallback_path = (
                    None if diagnostics_path is None else
                    diagnostics_path.replace(
                        ".json", f".swap{attempt_index}_{fallback_backend}.json"
                    )
                )
                fallback_nuts = dict(nuts_kwargs or {})
                if fallback_backend == "independent_hmc":
                    from models.independent_hmc import get_samples_independent_hmc
                    fallback_nuts.pop("max_tree_depth", None)
                    fallback_nuts.pop("laplace_max_tree_depth", None)
                    fallback_nuts.update(
                        mass_matrix="laplace", laplace_warmup=150,
                        laplace_target_accept=0.85,
                        laplace_start_at_map=True, num_steps=8,
                        trajectory_jitter=0.25,
                    )
                    fallback = get_samples_independent_hmc(
                        model, jax.random.fold_in(key_chunk, 99173 + attempt_index),
                        t, yerr_chunk[selected], y_chunk[selected], fallback_init,
                        nuts_kwargs=fallback_nuts, mcmc_kwargs=mcmc_kwargs,
                        diagnostics_path=fallback_path, lane_width=chunk_size,
                        channel_varying_kwargs=varying_for_chunk, **fallback_kwargs,
                    )
                elif fallback_backend == "independent_nuts":
                    from models.independent_nuts import get_samples_independent
                    fallback_nuts.pop("num_steps", None)
                    fallback_nuts.pop("trajectory_jitter", None)
                    fallback_nuts.update(
                        mass_matrix="laplace", laplace_warmup=150,
                        laplace_target_accept=(
                            0.99 if float((nuts_kwargs or {}).get(
                                "laplace_target_accept", 0.95)) >= 0.99 else 0.95
                        ),
                        laplace_start_at_map=True,
                    )
                    fallback = get_samples_independent(
                        model, jax.random.fold_in(key_chunk, 99173 + attempt_index),
                        t, yerr_chunk[selected], y_chunk[selected], fallback_init,
                        nuts_kwargs=fallback_nuts, mcmc_kwargs=mcmc_kwargs,
                        diagnostics_path=fallback_path, lane_width=chunk_size,
                        channel_varying_kwargs=varying_for_chunk, **fallback_kwargs,
                    )
                else:
                    adaptive_nuts = {
                        key: value for key, value in fallback_nuts.items()
                        if key in {"dense_mass", "regularize_mass_matrix",
                                   "target_accept_prob", "max_tree_depth"}
                    }
                    adaptive_nuts.update(target_accept_prob=0.95)
                    fallback = get_samples(
                        model, jax.random.fold_in(key_chunk, 99173 + attempt_index),
                        t, yerr_chunk[selected], y_chunk[selected], fallback_init,
                        nuts_kwargs=adaptive_nuts, mcmc_kwargs=mcmc_kwargs,
                        diagnostics_path=fallback_path, **fallback_kwargs,
                    )
                fallback_failed, fallback_gate = _spectro_failed_lanes(
                    fallback, fallback_path, spectro_min_depth_ess,
                    spectro_max_divergences,
                )
                gate_attempts.append({
                    "sampler": fallback_backend,
                    "input_local_lanes": selected.tolist(),
                    **fallback_gate,
                })
                samples_chunk = {
                    name: jnp.asarray(values).at[:, selected].set(fallback[name])
                    for name, values in samples_chunk.items()
                }
                for lane in selected:
                    sampler_used[int(lane)] = fallback_backend
                failed = selected[fallback_failed]
            if failed.size:
                raise RuntimeError(
                    f"chunk {start}:{end} failed the spectroscopic ESS/divergence "
                    f"gate after adaptive NUTS in local lanes {failed.tolist()}"
                )
        samples_chunk = _SamplerSamples(samples_chunk, sampler_used=sampler_used)
        _annotate_sampler_diagnostics(
            diagnostics_path,
            sampling_workload_fingerprint=sampling_workload_fingerprint,
            sampler_backend=sampler_backend,
        )
        if diagnostics_path is not None and os.path.isfile(diagnostics_path):
            with open(diagnostics_path, "r", encoding="utf-8") as stream:
                diagnostic_payload = json.load(stream)
            diagnostic_payload["sampler_used"] = sampler_used
            diagnostic_payload["sampler_gate_attempts"] = gate_attempts
            temporary = f"{diagnostics_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(diagnostic_payload, stream, indent=2, sort_keys=True)
            os.replace(temporary, diagnostics_path)
        samples_chunk = jax.device_get(samples_chunk)
        samples_chunks.append(samples_chunk)

        # Save checkpoint immediately after computation
        if use_checkpoints:
            chunk_file = _chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end)
            temporary_chunk = (
                f"{chunk_file}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            )
            with open(temporary_chunk, 'wb') as f:
                pickle.dump(
                    {"samples": dict(samples_chunk),
                     "sampler_used": samples_chunk.sampler_used}, f
                )
            os.replace(temporary_chunk, chunk_file)
            print(f"  chunk {start}:{end} - SAVED checkpoint")

    if chunk_mode == "parallel":
        existing = [
            os.path.exists(_chunk_checkpoint_path(checkpoint_dir, checkpoint_prefix, start, end))
            for start, end in all_chunk_ranges
        ]
        done = sum(existing)
        print(
            f"Parallel chunk job complete: {done}/{len(all_chunk_ranges)} checkpoint files present. "
            "Run again with chunk_mode='combine' after all chunks are finished."
        )
        return None

    # Concatenate all chunks
    print(f"\nConcatenating {len(samples_chunks)} chunks...")
    samples = _concatenate_chunk_samples(samples_chunks)
    print(f"All chunks complete! Final shape: {list(samples.values())[0].shape}")
    return samples


def _load_channel_batch_plan(path, num_channels):
    """Load and validate a difficulty-batched wavelength execution plan."""
    from models.channel_batching import ChannelBatchPlan

    with open(os.fspath(path), "r", encoding="utf-8") as stream:
        plan = ChannelBatchPlan.from_manifest(json.load(stream))
    expected = tuple(range(int(num_channels)))
    if plan.original_channel_indices != expected:
        raise ValueError(
            f"Channel batch plan {path!s} targets indices "
            f"{plan.original_channel_indices}, but this stage has channels "
            f"{expected}. Regenerate the pilot plan for this exact dataset."
        )
    return plan


def _resolve_stage_channel_batch_plan(flags, stage_name, num_channels):
    """Resolve an optional generic or stage-specific plan manifest."""
    stage_name = str(stage_name).lower()
    path = flags.get(
        f"{stage_name}_batch_plan",
        flags.get("spectro_batch_plan"),
    )
    if path in {None, ""}:
        return None
    plan = _load_channel_batch_plan(path, num_channels)
    print(
        f"Using {stage_name} channel batch plan {path} "
        f"(fingerprint={plan.fingerprint_sha256[:12]})."
    )
    return plan


def _resolve_stage_vmap_width(
    flags,
    stage_name,
    default_width,
    *,
    sampler_backend,
    trend_inference,
    num_cadences,
    active_transit_cadences,
    mcmc_kwargs,
    sampler_kwargs,
    transit_engine,
    ld_profile,
    ld_mode,
    detrend_type,
    param_method,
    n_planets,
    transit_window,
):
    """Load a measured width only for its exact runtime/science workload."""
    from models.channel_batching import WidthSelection

    stage_name = str(stage_name).lower()
    path = flags.get(
        f"{stage_name}_width_selection",
        flags.get("spectro_width_selection"),
    )
    if path in {None, ""}:
        return default_width
    with open(os.fspath(path), "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    selection = WidthSelection.from_manifest(payload)
    provenance = selection.provenance
    if not provenance:
        raise ValueError(
            f"Width selection {path!s} has no workload provenance. Regenerate "
            "it with the current benchmark harness; legacy width-only "
            "manifests are deliberately not fit-consumable."
        )

    expected_scalars = {
        "artifact_kind": "jwst_spectro_width_selection",
        "benchmark_platform": "gpu",
        "sampler_backend": str(sampler_backend),
        "trend_inference": str(trend_inference),
        "num_cadences": int(num_cadences),
        "num_warmup": int(mcmc_kwargs["num_warmup"]),
        "num_samples": int(mcmc_kwargs["num_samples"]),
    }
    mismatches = []
    for name, expected in expected_scalars.items():
        actual = provenance.get(name)
        if actual != expected:
            mismatches.append(f"{name}: measured={actual!r}, fit={expected!r}")

    expected_model = {
        "transit_engine": str(transit_engine),
        "ld_profile": str(ld_profile),
        "ld_mode": str(ld_mode),
        "detrend_type": str(detrend_type),
        "param_method": str(param_method),
        "n_planets": int(n_planets),
        "transit_window": str(transit_window),
        "window_cadences": int(active_transit_cadences),
    }
    measured_model = provenance.get("model", {})
    for name, expected in expected_model.items():
        actual = measured_model.get(name)
        if actual != expected:
            mismatches.append(
                f"model.{name}: measured={actual!r}, fit={expected!r}"
            )

    expected_sampler = {
        "max_tree_depth": (
            None
            if sampler_backend == "independent_hmc"
            else int(sampler_kwargs.get("max_tree_depth", 10))
        ),
        "hmc_num_steps": (
            int(sampler_kwargs["num_steps"])
            if sampler_backend == "independent_hmc" else None
        ),
        "target_accept": float(
            sampler_kwargs.get("target_accept_prob", 0.8)
        ),
        "dense_mass": bool(sampler_kwargs.get("dense_mass", False)),
        "regularize_mass_matrix": bool(
            sampler_kwargs.get("regularize_mass_matrix", True)
        ),
    }
    measured_sampler = provenance.get("sampler_options", {})
    for name, expected in expected_sampler.items():
        actual = measured_sampler.get(name)
        if isinstance(expected, float) and actual is not None:
            equal = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        else:
            equal = actual == expected
        if not equal:
            mismatches.append(
                f"sampler_options.{name}: measured={actual!r}, fit={expected!r}"
            )

    measured_runtime = provenance.get("runtime", {})
    if measured_runtime.get("jax_enable_x64") is not True:
        mismatches.append("runtime.jax_enable_x64 was not true")
    device = jax.devices()[0]
    current_device_kind = getattr(device, "device_kind", None)
    if measured_runtime.get("device_kind") != current_device_kind:
        mismatches.append(
            "runtime.device_kind: measured="
            f"{measured_runtime.get('device_kind')!r}, fit={current_device_kind!r}"
        )

    measured_sources = provenance.get("source_sha256", {})
    for relative_path, measured_hash in measured_sources.items():
        source_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), relative_path
        )
        if not os.path.isfile(source_path):
            mismatches.append(f"source {relative_path!r} is missing")
            continue
        with open(source_path, "rb") as source_stream:
            current_hash = hashlib.sha256(source_stream.read()).hexdigest()
        if current_hash != measured_hash:
            mismatches.append(f"source {relative_path!r} changed since measurement")

    if mismatches:
        raise ValueError(
            f"Measured {stage_name} width selection {path!s} does not match "
            "this fit workload:\n  - " + "\n  - ".join(mismatches)
        )
    try:
        device_stats = device.memory_stats() or {}
    except Exception:
        device_stats = {}
    current_limit = device_stats.get("bytes_limit")
    if current_limit is not None:
        current_budget = int(float(current_limit) * selection.memory_fraction)
        if selection.selected.peak_memory_bytes > current_budget:
            raise ValueError(
                f"Measured width {selection.selected.width} from {path!s} "
                f"peaked at {selection.selected.peak_memory_bytes} bytes, above "
                f"the current device budget of {current_budget} bytes. Run the "
                "width sweep on this GPU instead of reusing this manifest."
            )
    selected_width = int(selection.selected.width)
    print(
        f"Using measured {stage_name} resident width {selected_width} from "
        f"{path} (peak={selection.selected.peak_memory_bytes} bytes, "
        f"throughput={selection.selected.throughput:.4g}, "
        f"fingerprint={selection.fingerprint_sha256[:12]})."
    )
    return selected_width


def get_samples_channel_plan(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    channel_batch_plan,
    *,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    chunk_mode="serial",
    parallel_job_count=None,
    parallel_job_index=None,
    output_dir=None,
    checkpoint_prefix=None,
    sampler_backend="joint_nuts",
    laplace_is_kwargs=None,
    channel_varying_kwargs=(),
    checkpoint_signature=None,
    sampling_workload_fingerprint=None,
    gradient_diagnostic_mode="off",
    gradient_diagnostic_strict=False,
    **model_kwargs,
):
    """Run arbitrary, difficulty-bucketed batches and restore wavelength order.

    The plan is normally produced from a short independent-NUTS pilot.  Hard
    channels can use width one while ordinary channels retain a larger fixed
    resident width, preventing one pathological trajectory from serializing
    every GPU lane.
    """
    from models.channel_batching import ChannelBatchPlan, restore_sample_mapping

    if not isinstance(channel_batch_plan, ChannelBatchPlan):
        channel_batch_plan = ChannelBatchPlan.from_manifest(channel_batch_plan)
    num_lcs = int(indiv_y.shape[0])
    expected = tuple(range(num_lcs))
    if channel_batch_plan.original_channel_indices != expected:
        raise ValueError(
            "channel_batch_plan must cover exactly the current zero-based "
            f"channel indices {expected}."
        )
    chunk_mode = str(chunk_mode).lower()
    if chunk_mode not in {"serial", "parallel", "combine"}:
        raise ValueError(
            "chunk_mode must be one of {'serial', 'parallel', 'combine'}."
        )
    if chunk_mode != "serial" and (
        output_dir is None or checkpoint_prefix is None
    ):
        raise ValueError(
            "Planned parallel/combine execution requires output_dir and "
            "checkpoint_prefix."
        )

    plan_tag = channel_batch_plan.fingerprint_sha256[:12]
    print(
        "Running difficulty-batched MCMC: "
        f"{num_lcs} channels in {len(channel_batch_plan.batches)} batches "
        f"(plan={plan_tag}, mode={chunk_mode})"
    )
    if chunk_mode == "parallel":
        parallel_job_count, parallel_job_index = _resolve_parallel_chunk_job(
            parallel_job_count=parallel_job_count,
            parallel_job_index=parallel_job_index,
        )
        if parallel_job_count is None or parallel_job_index is None:
            raise ValueError(
                "chunk_mode='parallel' requires parallel job count/index or "
                "the corresponding SLURM/JWSTJAXFIT environment variables."
            )
        if parallel_job_count < 1 or not (
            0 <= parallel_job_index < parallel_job_count
        ):
            raise ValueError("Invalid planned-batch parallel job count/index.")
        batch_positions = [
            index
            for index in range(len(channel_batch_plan.batches))
            if index % parallel_job_count == parallel_job_index
        ]
        print(
            f"Parallel planned-batch job {parallel_job_index + 1}/"
            f"{parallel_job_count} assigned batches {batch_positions}"
        )
    else:
        batch_positions = list(range(len(channel_batch_plan.batches)))

    samples_by_batch = []
    # Planned batches invoke get_samples_chunked separately for checkpointing,
    # but batches with the same resident width should still share their
    # shape-specialized transition programs for the lifetime of this plan.
    joint_mcmc_runner_cache = {}
    independent_mcmc_runner_cache = {}
    for batch_index in batch_positions:
        batch = channel_batch_plan.batches[batch_index]
        indices = batch.channel_indices
        batch_init = {
            name: _take_by_channel(value, indices, num_lcs)
            for name, value in init_params.items()
        }
        channel_varying_set = frozenset(channel_varying_kwargs)
        batch_kwargs = {
            name: (
                _take_by_channel(value, indices, num_lcs)
                if name in channel_varying_set else value
            )
            for name, value in model_kwargs.items()
        }
        batch_yerr = _take_by_channel(yerr, indices, num_lcs)
        batch_y = _take_by_channel(indiv_y, indices, num_lcs)
        batch_key = jax.random.fold_in(key, batch_index)
        batch_prefix = (
            None
            if checkpoint_prefix is None
            else f"{checkpoint_prefix}_plan{plan_tag}_{batch.batch_id}"
        )
        batch_signature = {
            "parent": checkpoint_signature,
            "channel_batch_plan_fingerprint": (
                channel_batch_plan.fingerprint_sha256
            ),
            "channel_batch": batch.to_manifest(),
        }
        if gradient_diagnostic_mode == "each":
            batch_gradient_mode = "each"
        elif gradient_diagnostic_mode == "first" and batch_index == 0:
            batch_gradient_mode = "first"
        else:
            batch_gradient_mode = "off"
        print(
            f"  planned batch {batch.batch_id}: active={list(indices)}, "
            f"resident_width={batch.lane_width}, "
            f"quarantined={batch.quarantined}"
        )
        samples = get_samples_chunked(
            model,
            batch_key,
            t,
            batch_yerr,
            batch_y,
            batch_init,
            batch.lane_width,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            # Outer orchestration assigns whole planned batches.  Each inner
            # call owns exactly one checkpointed active batch.
            chunk_mode="combine" if chunk_mode == "combine" else "serial",
            output_dir=output_dir,
            checkpoint_prefix=batch_prefix,
            sampler_backend=sampler_backend,
            laplace_is_kwargs=laplace_is_kwargs,
            channel_varying_kwargs=channel_varying_kwargs,
            checkpoint_signature=batch_signature,
            sampling_workload_fingerprint=sampling_workload_fingerprint,
            gradient_diagnostic_mode=batch_gradient_mode,
            gradient_diagnostic_strict=(
                gradient_diagnostic_strict and batch_gradient_mode != "off"
            ),
            _joint_mcmc_runner_cache=joint_mcmc_runner_cache,
            _independent_mcmc_runner_cache=(
                independent_mcmc_runner_cache
            ),
            **batch_kwargs,
        )
        samples_by_batch.append(samples)

    if chunk_mode == "parallel":
        print(
            "Parallel planned-batch job complete. Run with chunk_mode='combine' "
            "after all batch jobs finish."
        )
        return None
    return restore_sample_mapping(
        samples_by_batch,
        channel_batch_plan,
        channel_axis=1,
    )

MCMC_KWARGS = {"num_warmup": 1000, "num_samples": 1000}

def _resolve_stage_mcmc_kwargs(flags, stage_name, default_warmup=1000, default_samples=1000):
    stage_key = str(stage_name).lower()
    num_warmup = int(flags.get(f"{stage_key}_num_warmup", default_warmup))
    num_samples = int(flags.get(f"{stage_key}_num_samples", default_samples))
    if (
        stage_key == "lowres"
        and os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
        and os.getenv("FIT_JWST_DUMP_BRIDGE_LOWRES", "0") == "1"
    ):
        num_warmup = int(
            os.getenv("FIT_JWST_DUMP_BRIDGE_LOWRES_WARMUP", "2")
        )
        num_samples = int(
            os.getenv("FIT_JWST_DUMP_BRIDGE_LOWRES_SAMPLES", "2")
        )
    if num_warmup < 0:
        raise ValueError(f"flags.{stage_key}_num_warmup must be >= 0.")
    if num_samples < 1:
        raise ValueError(f"flags.{stage_key}_num_samples must be >= 1.")
    return {
        "num_warmup": num_warmup,
        "num_samples": num_samples,
    }


def _resolve_whitelight_laplace_options(flags):
    """Resolve production white-light Laplace preconditioning controls."""
    mass_matrix = str(flags.get("whitelight_mass_matrix", "laplace")).lower()
    if mass_matrix not in {"adaptive", "laplace"}:
        raise ValueError(
            "flags.whitelight_mass_matrix must be 'adaptive' or 'laplace'."
        )
    options = {
        "mass_matrix": mass_matrix,
        "warmup": int(flags.get("whitelight_laplace_warmup", 200)),
        "target_accept": float(
            flags.get("whitelight_laplace_target_accept", 0.9)
        ),
        "max_tree_depth": int(
            flags.get("whitelight_laplace_max_tree_depth", 10)
        ),
        "trust_radius": float(
            flags.get("whitelight_laplace_trust_radius", 5.0)
        ),
        "hessian_method": str(
            flags.get(
                "whitelight_laplace_hessian_method", "finite_difference"
            )
        ).lower(),
    }
    if options["warmup"] < 0:
        raise ValueError("flags.whitelight_laplace_warmup must be >= 0.")
    if not 0.0 < options["target_accept"] < 1.0:
        raise ValueError(
            "flags.whitelight_laplace_target_accept must be between 0 and 1."
        )
    if options["max_tree_depth"] < 1 or options["trust_radius"] <= 0.0:
        raise ValueError("Invalid white-light Laplace depth/trust-radius setting.")
    if options["hessian_method"] not in {"exact", "finite_difference"}:
        raise ValueError(
            "flags.whitelight_laplace_hessian_method must be 'exact' or "
            "'finite_difference'."
        )
    return options


def _resolve_harmonica_stage_nuts_kwargs(
    flags,
    stage_prefix,
    *,
    default_dense_mass,
    default_regularize_mass_matrix=True,
    default_max_tree_depth=8,
    default_target_accept=0.8,
):
    max_tree_depth = int(flags.get(f"{stage_prefix}_max_tree_depth", default_max_tree_depth))
    target_accept = float(flags.get(f"{stage_prefix}_target_accept", default_target_accept))
    if max_tree_depth < 1:
        raise ValueError(f"flags.{stage_prefix}_max_tree_depth must be >= 1.")
    if not (0.0 < target_accept < 1.0):
        raise ValueError(f"flags.{stage_prefix}_target_accept must be between 0 and 1.")
    result = {
        "dense_mass": bool(flags.get(f"{stage_prefix}_dense_mass", default_dense_mass)),
        "regularize_mass_matrix": bool(
            flags.get(f"{stage_prefix}_regularize_mass_matrix", default_regularize_mass_matrix)
        ),
        "max_tree_depth": max_tree_depth,
        "target_accept_prob": target_accept,
    }
    independent = str(flags.get('spectro_sampler', 'independent_nuts')).lower() in {
        'independent_nuts', 'independent_hmc'
    }
    if independent:
        result = _resolve_jaxoplanet_spectro_nuts_kwargs(
            flags,
            stage_prefix,
            result,
            independent=True,
            hmc=str(flags.get('spectro_sampler', 'independent_nuts')).lower() == 'independent_hmc',
        )
    return result


def _resolve_jaxoplanet_spectro_nuts_kwargs(
    flags,
    stage_prefix,
    defaults,
    *,
    independent=False,
    hmc=False,
):
    """Resolve validated low/high-resolution Jaxoplanet sampler controls."""
    result = dict(defaults)
    generic_depth = flags.get('spectro_max_tree_depth', result.get('max_tree_depth', 10))
    generic_accept = flags.get('spectro_target_accept', result.get('target_accept_prob', 0.8))
    max_tree_depth = int(flags.get(f'{stage_prefix}_max_tree_depth', generic_depth))
    if (
        stage_prefix == "lowres"
        and os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
        and os.getenv("FIT_JWST_DUMP_BRIDGE_LOWRES", "0") == "1"
    ):
        max_tree_depth = int(
            os.getenv("FIT_JWST_DUMP_BRIDGE_LOWRES_MAX_TREE_DEPTH", "2")
        )
    target_accept = float(flags.get(f'{stage_prefix}_target_accept', generic_accept))
    if max_tree_depth < 1:
        raise ValueError(f"flags.{stage_prefix}_max_tree_depth must be >= 1.")
    if not (0.0 < target_accept < 1.0):
        raise ValueError(f"flags.{stage_prefix}_target_accept must be between 0 and 1.")
    result['target_accept_prob'] = target_accept
    result['regularize_mass_matrix'] = bool(
        flags.get(
            f'{stage_prefix}_regularize_mass_matrix',
            result.get('regularize_mass_matrix', True),
        )
    )
    result['dense_mass'] = bool(
        flags.get(
            f'{stage_prefix}_dense_mass',
            True if independent else result.get('dense_mass', False),
        )
    )
    if independent:
        mass_matrix = str(
            flags.get(
                f'{stage_prefix}_mass_matrix',
                flags.get('spectro_mass_matrix', result.get('mass_matrix', 'adaptive')),
            )
        ).lower()
        if mass_matrix not in {'adaptive', 'laplace'}:
            raise ValueError(
                f"flags.{stage_prefix}_mass_matrix must be 'adaptive' or "
                f"'laplace'; got {mass_matrix!r}."
            )
        result['mass_matrix'] = mass_matrix
        if mass_matrix == 'laplace':
            laplace_warmup = int(
                flags.get(
                    f'{stage_prefix}_laplace_warmup',
                    flags.get('spectro_laplace_warmup', 150),
                )
            )
            laplace_target_accept = float(
                flags.get(
                    f'{stage_prefix}_laplace_target_accept',
                    flags.get('spectro_laplace_target_accept', 0.85 if hmc else 0.95),
                )
            )
            laplace_max_tree_depth = int(
                flags.get(
                    f'{stage_prefix}_laplace_max_tree_depth',
                    flags.get('spectro_laplace_max_tree_depth', 10),
                )
            )
            if laplace_warmup < 0:
                raise ValueError(
                    f"flags.{stage_prefix}_laplace_warmup must be >= 0."
                )
            if not (0.0 < laplace_target_accept < 1.0):
                raise ValueError(
                    f"flags.{stage_prefix}_laplace_target_accept must be "
                    "between 0 and 1."
                )
            if not hmc and laplace_max_tree_depth < 1:
                raise ValueError(
                    f"flags.{stage_prefix}_laplace_max_tree_depth must be >= 1."
                )
            result.update(
                laplace_warmup=laplace_warmup,
                laplace_target_accept=laplace_target_accept,
                laplace_start_at_map=bool(
                    flags.get(
                        f'{stage_prefix}_laplace_start_at_map',
                        flags.get('spectro_laplace_start_at_map', False),
                    )
                ),
                laplace_hessian_method=str(
                    flags.get(
                        f'{stage_prefix}_laplace_hessian_method',
                        flags.get('spectro_laplace_hessian_method', 'exact'),
                    )
                ).lower(),
                laplace_fd_relative_step=float(
                    flags.get(
                        f'{stage_prefix}_laplace_fd_relative_step',
                        flags.get('spectro_laplace_fd_relative_step', 2.0e-4),
                    )
                ),
                laplace_fuse_program=bool(
                    flags.get(
                        f'{stage_prefix}_laplace_fuse_program',
                        flags.get('spectro_laplace_fuse_program', False),
                    )
                ),
                laplace_trust_radius=float(
                    flags.get(
                        f'{stage_prefix}_laplace_trust_radius',
                        flags.get('spectro_laplace_trust_radius', 5.0),
                    )
                ),
                laplace_map_decrement_tolerance=float(
                    flags.get(
                        f'{stage_prefix}_laplace_map_decrement_tolerance',
                        flags.get(
                            'spectro_laplace_map_decrement_tolerance', 1.0e-4
                        ),
                    )
                ),
            )
            if not hmc:
                result['laplace_max_tree_depth'] = laplace_max_tree_depth
    if hmc:
        generic_steps = int(flags.get('spectro_hmc_num_steps', 16))
        num_steps = int(flags.get(f'{stage_prefix}_hmc_num_steps', generic_steps))
        if num_steps < 1:
            raise ValueError(f"flags.{stage_prefix}_hmc_num_steps must be >= 1.")
        result.pop('max_tree_depth', None)
        result['num_steps'] = num_steps
        result['trajectory_jitter'] = float(
            flags.get(
                f'{stage_prefix}_hmc_trajectory_jitter',
                flags.get('spectro_hmc_trajectory_jitter', 0.0),
            )
        )
    else:
        result['max_tree_depth'] = max_tree_depth
    return result


def _resolve_laplace_is_stage_kwargs(flags, stage_prefix):
    """Resolve generic and stage-specific Laplace-IS controls."""
    stage_prefix = str(stage_prefix).lower()

    def value(name, default):
        return flags.get(
            f'{stage_prefix}_laplace_is_{name}',
            flags.get(
                f'spectro_laplace_is_{name}',
                flags.get(f'laplace_is_{name}', default),
            ),
        )

    def boolean_value(name, default):
        raw = value(name, default)
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {'true', 'yes', 'on', '1'}:
                return True
            if normalized in {'false', 'no', 'off', '0'}:
                return False
            raise ValueError(
                f"flags.{stage_prefix}_laplace_is_{name} must be boolean."
            )
        return bool(raw)

    result = {
        'laplace_is_output': str(value('output', 'imh')).lower(),
        'laplace_is_num_draws': int(value('num_draws', 4096)),
        'laplace_is_rounds': int(value('rounds', 2)),
        'laplace_is_draw_chunk_size': int(value('draw_chunk_size', 256)),
        'laplace_is_student_df': float(value('student_df', 3.0)),
        'laplace_is_scale_inflation': float(value('scale_inflation', 1.5)),
        'laplace_is_wide_fraction': float(value('wide_fraction', 0.0)),
        'laplace_is_wide_scale': float(value('wide_scale', 3.0)),
        'laplace_is_map_maxiter': int(value('map_maxiter', 200)),
        'laplace_is_map_tol': float(value('map_tol', 1.0e-4)),
        'laplace_is_trust_radius': float(value('trust_radius', 5.0)),
        'laplace_is_khat_threshold': float(value('khat_threshold', 0.7)),
        'laplace_is_min_ess': float(value('min_ess', 400.0)),
        'laplace_is_min_ess_fraction': float(value('min_ess_fraction', 0.2)),
        'laplace_is_min_imh_acceptance': float(
            value('min_imh_acceptance', 0.2)
        ),
        'laplace_is_imh_thin': int(value('imh_thin', 8)),
        'laplace_is_fallback': boolean_value('fallback', True),
        'laplace_is_force': boolean_value('force', False),
    }
    if result['laplace_is_output'] not in {'imh', 'resample'}:
        raise ValueError(
            f"flags.{stage_prefix}_laplace_is_output must be 'imh' or "
            "'resample'."
        )
    if (
        result['laplace_is_num_draws'] < 6
        or result['laplace_is_rounds'] < 0
        or result['laplace_is_draw_chunk_size'] < 1
        or result['laplace_is_map_maxiter'] < 1
        or result['laplace_is_imh_thin'] < 1
    ):
        raise ValueError(
            f"flags.{stage_prefix}_laplace_is_* draw/MAP counts are invalid."
        )
    if (
        result['laplace_is_student_df'] <= 2.0
        or result['laplace_is_scale_inflation'] <= 0.0
        or not 0.0 <= result['laplace_is_wide_fraction'] < 1.0
        or result['laplace_is_wide_scale'] < 1.0
        or result['laplace_is_map_tol'] <= 0.0
        or result['laplace_is_trust_radius'] <= 0.0
    ):
        raise ValueError(
            f"flags.{stage_prefix}_laplace_is_* proposal/MAP values are invalid."
        )
    if not 0.0 <= result['laplace_is_min_ess_fraction'] <= 1.0:
        raise ValueError(
            f"flags.{stage_prefix}_laplace_is_min_ess_fraction must be in [0, 1]."
        )
    if not 0.0 <= result['laplace_is_min_imh_acceptance'] <= 1.0:
        raise ValueError(
            f"flags.{stage_prefix}_laplace_is_min_imh_acceptance must be in [0, 1]."
        )
    return result

def _run_sampling_stage(
    model,
    key,
    t,
    yerr,
    indiv_y,
    init_params,
    nuts_kwargs=None,
    mcmc_kwargs=None,
    use_chunked=False,
    chunk_size=None,
    chunk_mode="serial",
    parallel_job_count=None,
    parallel_job_index=None,
    output_dir=None,
    checkpoint_prefix=None,
    sampler_backend="joint_nuts",
    laplace_is_kwargs=None,
    channel_varying_kwargs=(),
    checkpoint_signature=None,
    channel_batch_plan=None,
    gradient_diagnostic_mode="off",
    gradient_diagnostic_strict=False,
    spectro_min_depth_ess=0.0,
    spectro_max_divergences=0,
    compile_box=False,
    dump_metadata=None,
    **model_kwargs,
):
    if compile_box:
        original_cadences = int(jnp.asarray(t).shape[0])
        t, indiv_y, yerr, likelihood_mask = _pad_spectro_cadences_exact(
            t, indiv_y, yerr, multiple=256
        )
        padded_kwargs = dict(model_kwargs)
        for name, value in model_kwargs.items():
            try:
                array = jnp.asarray(value)
            except (TypeError, ValueError):
                continue
            if array.ndim and array.shape[-1] == original_cadences:
                pad = int(t.shape[0]) - original_cadences
                padded_kwargs[name] = jnp.pad(
                    array,
                    [(0, 0)] * (array.ndim - 1) + [(0, pad)],
                    constant_values=0.0,
                )
        padded_kwargs["likelihood_mask"] = likelihood_mask
        model_kwargs = padded_kwargs
        print(
            f"Compile box cadence bucket: {original_cadences} -> {t.shape[0]} "
            "with an exact likelihood mask."
        )
    laplace_is_kwargs = dict(laplace_is_kwargs or {})
    laplace_is_force = bool(laplace_is_kwargs.pop('laplace_is_force', False))
    stage_ld_mode = (
        None if checkpoint_signature is None
        else checkpoint_signature.get('ld_mode')
    )
    if (
        sampler_backend == 'laplace_is'
        and str(stage_ld_mode).lower() in {'widegaussian', 'uniform'}
        and not laplace_is_force
    ):
        print(
            "Laplace-IS disabled for "
            f"ld_mode={stage_ld_mode!r}: routing the whole chunk through "
            "finite-difference Laplace-metric independent NUTS. Set "
            "laplace_is_force: true to override."
        )
        sampler_backend = 'independent_nuts'
        nuts_kwargs = dict(nuts_kwargs or {})
        nuts_kwargs.update(
            mass_matrix='laplace',
            laplace_hessian_method='finite_difference',
            laplace_fd_relative_step=2.0e-4,
            laplace_map_iterations=200,
            laplace_map_decrement_tolerance=1.0e-4,
            laplace_trust_radius=5.0,
            laplace_warmup=150,
            laplace_target_accept=0.95,
            laplace_max_tree_depth=6,
            laplace_start_at_map=True,
        )
        laplace_is_kwargs = None
    _maybe_dump_sampler_inputs(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        chunk_size=chunk_size,
        use_chunked=use_chunked,
        sampler_backend=sampler_backend,
        channel_varying_kwargs=channel_varying_kwargs,
        checkpoint_signature=checkpoint_signature,
        dump_metadata=dump_metadata,
        model_kwargs=model_kwargs,
    )
    sampling_workload_fingerprint = _sampling_workload_fingerprint(
        model,
        t,
        yerr,
        indiv_y,
        init_params,
        sampler_backend=sampler_backend,
        nuts_kwargs=nuts_kwargs,
        channel_varying_kwargs=channel_varying_kwargs,
        checkpoint_signature=checkpoint_signature,
        model_kwargs=model_kwargs,
    )
    if channel_batch_plan is not None:
        from models.channel_batching import ChannelBatchPlan

        if not isinstance(channel_batch_plan, ChannelBatchPlan):
            channel_batch_plan = ChannelBatchPlan.from_manifest(
                channel_batch_plan
            )
        provenance = getattr(channel_batch_plan, "provenance", None)
        if not provenance:
            raise ValueError(
                "The channel batch plan has no workload provenance. Rerun "
                "the pilot with the current fit code and rebuild the plan."
            )
        if provenance.get("artifact_kind") != (
            "jwst_spectro_channel_batch_plan"
        ):
            raise ValueError("Unrecognized channel batch plan provenance.")
        measured_fingerprint = provenance.get(
            "sampling_workload_fingerprint_sha256"
        )
        if measured_fingerprint != sampling_workload_fingerprint:
            raise ValueError(
                "The channel batch plan was measured for a different data, "
                "model, prior, or sampler workload. Rerun the pilot for this "
                "exact spectroscopic stage."
            )
    if use_chunked:
        effective_chunk_size = 1 if chunk_size is None else int(chunk_size)
        if effective_chunk_size < 1:
            raise ValueError("chunk_size must be >= 1 when chunked sampling is enabled.")
        if channel_batch_plan is not None:
            return get_samples_channel_plan(
                model,
                key,
                t,
                yerr,
                indiv_y,
                init_params,
                channel_batch_plan,
                nuts_kwargs=nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs,
                chunk_mode=chunk_mode,
                parallel_job_count=parallel_job_count,
                parallel_job_index=parallel_job_index,
                output_dir=output_dir,
                checkpoint_prefix=checkpoint_prefix,
                sampler_backend=sampler_backend,
                laplace_is_kwargs=laplace_is_kwargs,
                channel_varying_kwargs=channel_varying_kwargs,
                checkpoint_signature=checkpoint_signature,
                sampling_workload_fingerprint=(
                    sampling_workload_fingerprint
                ),
                gradient_diagnostic_mode=gradient_diagnostic_mode,
                gradient_diagnostic_strict=gradient_diagnostic_strict,
                spectro_min_depth_ess=spectro_min_depth_ess,
                spectro_max_divergences=spectro_max_divergences,
                **model_kwargs,
            )
        return get_samples_chunked(
            model,
            key,
            t,
            yerr,
            indiv_y,
            init_params,
            effective_chunk_size,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            chunk_mode=chunk_mode,
            parallel_job_count=parallel_job_count,
            parallel_job_index=parallel_job_index,
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
            sampler_backend=sampler_backend,
            laplace_is_kwargs=laplace_is_kwargs,
            channel_varying_kwargs=channel_varying_kwargs,
            checkpoint_signature=checkpoint_signature,
            sampling_workload_fingerprint=sampling_workload_fingerprint,
            gradient_diagnostic_mode=gradient_diagnostic_mode,
            gradient_diagnostic_strict=gradient_diagnostic_strict,
            spectro_min_depth_ess=spectro_min_depth_ess,
            spectro_max_divergences=spectro_max_divergences,
            **model_kwargs,
        )
    if str(gradient_diagnostic_mode).lower() != "off":
        from diagnose_nuts_gradient import diagnose_gradient_quality

        try:
            gradient_result = diagnose_gradient_quality(
                model,
                init_params,
                t,
                yerr,
                indiv_y,
                model_kwargs=model_kwargs,
                num_directions=3,
            )
            print(
                "Spectroscopic gradient diagnostic "
                f"{'PASSED' if gradient_result['passed'] else 'WARNING'} "
                f"(finite={gradient_result['all_finite']}, "
                "max directional relative error="
                f"{gradient_result['max_directional_relative_error']:.3e})"
            )
            if output_dir is not None and checkpoint_prefix is not None:
                gradient_path = os.path.join(
                    output_dir, f"{checkpoint_prefix}_gradient_diagnostics.json"
                )
                temporary_gradient = (
                    f"{gradient_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
                )
                with open(temporary_gradient, "w", encoding="utf-8") as stream:
                    json.dump(gradient_result, stream, indent=2, sort_keys=True)
                os.replace(temporary_gradient, gradient_path)
            if gradient_diagnostic_strict and not gradient_result["passed"]:
                raise RuntimeError("Spectroscopic gradient diagnostic failed.")
        except Exception as error:
            if gradient_diagnostic_strict:
                raise
            print(f"Spectroscopic gradient diagnostic unavailable: {error}")
    if sampler_backend in {
        "independent_nuts", "independent_hmc", "laplace_is"
    }:
        if sampler_backend == "independent_hmc":
            from models.independent_hmc import get_samples_independent_hmc
            independent_sampler = get_samples_independent_hmc
            backend_options = {}
        elif sampler_backend == "laplace_is":
            from models.laplace_is import get_samples_laplace_is
            independent_sampler = get_samples_laplace_is
            backend_options = dict(laplace_is_kwargs or {})
        else:
            from models.independent_nuts import get_samples_independent
            independent_sampler = get_samples_independent
            backend_options = {}

        return independent_sampler(
            model,
            key,
            t,
            yerr,
            indiv_y,
            init_params,
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            lane_width=int(indiv_y.shape[0]),
            channel_varying_kwargs=tuple(
                name for name in channel_varying_kwargs
                if name in model_kwargs
            ),
            **backend_options,
            **model_kwargs,
        )
    if sampler_backend != "joint_nuts":
        raise ValueError(
            "sampler_backend must be one of "
            "{'joint_nuts', 'independent_nuts', 'independent_hmc', "
            "'laplace_is'}; "
            f"received {sampler_backend!r}."
        )
    return get_samples(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        **model_kwargs,
    )

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

def _get_ld_mode_bounds(instrument, order=None):
    if instrument == 'NIRSPEC/G395H':
        return "JWST_NIRSpec_G395H", 28700.0, 51700.0
    elif instrument == 'NIRSPEC/G235H':
        return "JWST_NIRSpec_G235H", 17000.0, 30600.0
    elif instrument == 'NIRSPEC/G140H':
        return "JWST_NIRSpec_G140H-f100", 10000.0, 18000.0
    elif instrument == 'NIRSPEC/G395M':
        return "JWST_NIRSpec_G395M", 28700.0, 51700.0
    elif instrument == 'NIRSPEC/PRISM':
        return "JWST_NIRSpec_Prism", 5000.0, 55000.0
    elif instrument == 'NIRISS/SOSS':
        if order == 2:
            return f"JWST_NIRISS_SOSSo{order}", 6300.0, 8500.0
        return f"JWST_NIRISS_SOSSo{order}", 8300.0, 28100.0
    elif instrument == 'MIRI/LRS':
        return 'JWST_MIRI_LRS', 50000.0, 120000.0
    raise ValueError(f"Unsupported instrument for LD: {instrument}")


def _clip_ld_range(intended_min, intended_max, wl_min, wl_max):
    width = intended_max - intended_min
    if intended_max > wl_max:
        if width > 0:
            range_min = max(wl_max - width, wl_min)
            range_max = wl_max
        else:
            range_min = wl_max
            range_max = wl_max
    elif intended_min < wl_min:
        if width > 0:
            range_min = wl_min
            range_max = min(wl_min + width, wl_max)
        else:
            range_min = wl_min
            range_max = wl_min
    else:
        range_min = intended_min
        range_max = intended_max

    if range_min >= range_max:
        raise ValueError(
            f"Wavelength range {intended_min}-{intended_max} A has no overlap with "
            f"instrument range {wl_min}-{wl_max} A."
        )
    return range_min, range_max


def _gaussian_pdf(x, mu, sigma):
    if sigma <= 0:
        return 1.0 if np.isclose(x, mu) else 0.0
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z)


def _build_axis(mu, sigma, n_grid, nsigma):
    if sigma <= 0:
        return np.array([mu], dtype=float)
    if n_grid == 1:
        return np.array([mu], dtype=float)
    return mu + sigma * np.linspace(-nsigma, nsigma, n_grid)


def _combine_weighted_means_and_sigmas(mus, weights):
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    mean = np.sum(w[:, None] * mus, axis=0)
    second_moment = np.sum(w[:, None] * (mus ** 2), axis=0)
    star_var = np.maximum(0.0, second_moment - mean ** 2)
    star_sigma = np.sqrt(star_var)
    fit_sigma = np.zeros_like(star_sigma)
    total_sigma = star_sigma
    return mean, total_sigma, fit_sigma, star_sigma


def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None, ld_profile='quadratic', return_sigmas=False, ld_mu_min=None):
    mode, wl_min, wl_max = _get_ld_mode_bounds(instrument, order)

    wavelengths = np.asarray(wavelengths, dtype=float)
    has_vector_err = hasattr(wavelength_err, '__len__') and np.asarray(wavelength_err).ndim > 0
    wavelength_err = np.asarray(wavelength_err, dtype=float) if has_vector_err else float(wavelength_err)

    U_mu = []
    U_sig = []

    if has_vector_err:
        for i in range(len(wavelengths)):
            wl_angstrom = wavelengths[i] * 1e4
            err_angstrom = wavelength_err[i] * 1e4
            intended_min = wl_angstrom - err_angstrom
            intended_max = wl_angstrom + err_angstrom
            range_min, range_max = _clip_ld_range(intended_min, intended_max, wl_min, wl_max)

            if return_sigmas:
                if ld_profile == 'quadratic':
                    coeffs, sigmas = sld.compute_quadratic_ld_coeffs(
                        wavelength_range=[range_min, range_max],
                        mode=mode,
                        **({"mu_min": float(ld_mu_min)} if ld_mu_min is not None else {}),
                        return_sigmas=True,
                    )
                elif ld_profile == 'power2':
                    coeffs, sigmas = sld.compute_power2_ld_coeffs(
                        wavelength_range=[range_min, range_max],
                        mode=mode,
                        return_sigmas=True,
                    )
                else:
                    raise ValueError(f"Unknown ld_profile: {ld_profile}")
                U_mu.append(coeffs)
                U_sig.append(sigmas)
            else:
                if ld_profile == 'quadratic':
                    U_mu.append(sld.compute_quadratic_ld_coeffs(
                        wavelength_range=[range_min, range_max],
                        mode=mode,
                        **({"mu_min": float(ld_mu_min)} if ld_mu_min is not None else {}),
                        return_sigmas=False,
                    ))
                elif ld_profile == 'power2':
                    U_mu.append(sld.compute_power2_ld_coeffs(
                        wavelength_range=[range_min, range_max],
                        mode=mode,
                        return_sigmas=False,
                    ))
                else:
                    raise ValueError(f"Unknown ld_profile: {ld_profile}")
        U_mu = jnp.array(U_mu)
        if return_sigmas:
            U_sig = jnp.array(U_sig)
            return U_mu, U_sig
    else:
        intended_min = np.min(wavelengths) * 1e4
        intended_max = np.max(wavelengths) * 1e4
        range_min, range_max = _clip_ld_range(intended_min, intended_max, wl_min, wl_max)
        if return_sigmas:
            if ld_profile == 'quadratic':
                U_mu, U_sig = sld.compute_quadratic_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    **({"mu_min": float(ld_mu_min)} if ld_mu_min is not None else {}),
                    return_sigmas=True,
                )
            elif ld_profile == 'power2':
                U_mu, U_sig = sld.compute_power2_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    return_sigmas=True,
                )
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
            return jnp.array(U_mu), jnp.array(U_sig)
        else:
            if ld_profile == 'quadratic':
                U_mu = sld.compute_quadratic_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    **({"mu_min": float(ld_mu_min)} if ld_mu_min is not None else {}),
                    return_sigmas=False,
                )
            elif ld_profile == 'power2':
                U_mu = sld.compute_power2_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    return_sigmas=False,
                )
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        U_mu = jnp.array(U_mu)
    return U_mu


def build_sing_ld_prior(quadratic_coefficients, flags, stellar_cfg, offsets_override=None):
    """Return per-channel (l, delta) centers/scales for the Sing prior.

    A fitted offset artifact can be supplied with ``flags.ld_sing_offset_path``.
    If fit mode has no readable artifact yet, the documented Stagger population
    offset is used explicitly as the safe first-pass fallback.
    """
    coeff = np.asarray(quadratic_coefficients, dtype=float)
    scalar = coeff.ndim == 1
    coeff = np.atleast_2d(coeff)
    model_l, model_delta = quadratic_to_sing(coeff[:, 0], coeff[:, 1])
    requested = str(flags.get('ld_sing_offset', 'fit')).strip().lower()
    if requested not in {'fit', 'tabulated'}:
        raise ValueError("flags.ld_sing_offset must be 'fit' or 'tabulated'.")
    offsets = dict(SING_TABULATED_OFFSET)
    source = 'tabulated'
    if offsets_override is not None:
        offsets = {
            'l': float(offsets_override['l']),
            'delta': float(offsets_override['delta']),
        }
        source = 'newly-fitted gray calibration'
    artifact_path = flags.get('ld_sing_offset_path')
    if offsets_override is not None:
        pass
    elif requested == 'fit' and artifact_path:
        try:
            with open(artifact_path) as handle:
                artifact = json.load(handle)
            offsets = {'l': float(artifact['l']), 'delta': float(artifact['delta'])}
            source = f"fit:{artifact_path}"
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            print(f"[Sing LD] fitted gray-offset artifact unavailable ({error}); using tabulated fallback.", flush=True)
    elif requested == 'fit':
        print("[Sing LD] no fitted gray-offset artifact supplied; using tabulated fallback for this first pass.", flush=True)
    centers = np.column_stack((model_l + offsets['l'], model_delta + offsets['delta']))
    scales = np.broadcast_to(
        np.array([SING_TABULATED_SCATTER['l'], SING_TABULATED_SCATTER['delta']]),
        centers.shape,
    ).copy()
    print(
        f"[Sing LD] offset source={source}: delta_l={offsets['l']:+.6f}, "
        f"delta_delta={offsets['delta']:+.6f}; mu_min={float(stellar_cfg.get('ld_mu_min', 0.2)):.3f}",
        flush=True,
    )
    if scalar:
        return jnp.asarray(centers[0]), jnp.asarray(scales[0])
    return jnp.asarray(centers), jnp.asarray(scales)


def _assess_sing_gray_fit(samples, diagnostics_paths, min_ess=100.0):
    """Validate the free-LD calibration before it can inform another fit."""
    if 'limb_l' not in samples or 'limb_delta' not in samples:
        return False, {'reason': "physical free-LD fit did not return limb_l and limb_delta"}
    physical = np.stack(
        (np.asarray(samples['limb_l'], dtype=float),
         np.asarray(samples['limb_delta'], dtype=float)), axis=-1,
    )
    if physical.ndim != 3 or physical.shape[-1] != 2 or not np.all(np.isfinite(physical)):
        return False, {'reason': f"invalid free-LD posterior shape/values: {physical.shape}"}
    try:
        ess_values = [
            float(az.ess(physical[None, :, channel, coefficient], method='bulk'))
            for channel in range(physical.shape[1])
            for coefficient in range(physical.shape[2])
        ]
        min_bulk_ess = float(np.nanmin(ess_values))
    except Exception as error:
        return False, {'reason': f"bulk ESS calculation failed: {error}"}
    num_divergences = 0
    missing_diagnostics = []
    for path in diagnostics_paths:
        try:
            with open(path) as handle:
                num_divergences += int(json.load(handle).get('num_divergences', 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            missing_diagnostics.append(path)
    result = {
        'min_bulk_ess': min_bulk_ess,
        'required_min_bulk_ess': float(min_ess),
        'num_divergences': int(num_divergences),
        'missing_diagnostics': missing_diagnostics,
    }
    passed = not missing_diagnostics and num_divergences == 0 and min_bulk_ess >= min_ess
    return passed, result


def get_or_build_power2_ld_prior(stellar_cfg, wavelengths, wavelength_err, instrument, order=None, output_dir='.', cache_label='ld'):
    ld_prior_model = stellar_cfg.get('ld_prior_model', stellar_cfg.get('ld_model', 'stagger'))
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')
    ld_interpolate_type = stellar_cfg.get('ld_interpolate_type', 'trilinear')
    teff = float(stellar_cfg['teff'])
    logg = float(stellar_cfg['logg'])
    feh = float(stellar_cfg['feh'])
    teff_sigma = float(stellar_cfg.get('teff_sigma', 0.0))
    logg_sigma = float(stellar_cfg.get('logg_sigma', 0.0))
    feh_sigma = float(stellar_cfg.get('feh_sigma', 0.0))
    n_grid = int(stellar_cfg.get('ld_prior_n_grid', 5))
    nsigma = float(stellar_cfg.get('ld_prior_nsigma', 3.0))
    min_sigma = float(stellar_cfg.get('ld_prior_min_sigma', 1e-4))
    if n_grid < 1:
        raise ValueError("stellar.ld_prior_n_grid must be an integer >= 1.")
    if nsigma < 0.0:
        raise ValueError("stellar.ld_prior_nsigma must be >= 0.")
    if min_sigma <= 0.0:
        raise ValueError("stellar.ld_prior_min_sigma must be > 0.")

    wavelengths_np = np.asarray(wavelengths, dtype=float)
    has_vector_err = hasattr(wavelength_err, '__len__') and np.asarray(wavelength_err).ndim > 0
    wavelength_err_np = np.asarray(wavelength_err, dtype=float) if has_vector_err else np.array([float(wavelength_err)], dtype=float)

    cache_dir = stellar_cfg.get('ld_prior_cache_dir', os.path.join(output_dir, 'ld_prior_cache'))
    os.makedirs(cache_dir, exist_ok=True)
    mode, wl_min, wl_max = _get_ld_mode_bounds(instrument, order)
    implementation_path = sys.modules[StellarLimbDarkening.__module__].__file__
    digest = hashlib.sha256()
    _update_checkpoint_hash(
        digest,
        {
            "target_revision": POWER2_LD_CACHE_TARGET_REVISION,
            "wavelengths": wavelengths_np,
            "wavelength_err": wavelength_err_np,
            "instrument": instrument,
            "order": order,
            "mode": mode,
            "mode_bounds": (wl_min, wl_max),
            "ld_prior_model": ld_prior_model,
            "ld_data_path": os.path.realpath(os.path.expanduser(ld_data_path)),
            "ld_data_identity": _directory_metadata_identity(ld_data_path),
            "ld_interpolate_type": ld_interpolate_type,
            "ld_implementation": _file_content_identity(implementation_path),
            "stellar": (teff, teff_sigma, logg, logg_sigma, feh, feh_sigma),
            "grid": (n_grid, nsigma),
            "min_sigma": min_sigma,
            "cache_label": cache_label,
        },
    )
    cache_name = f"{instrument.replace('/', '_')}_{cache_label}_{digest.hexdigest()[:12]}.csv"
    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        expected_rows = len(wavelengths_np) if has_vector_err else 1
        required_columns = {
            'c1_mean', 'c2_mean', 'c1_sigma_star', 'c2_sigma_star'
        }
        try:
            df = pd.read_csv(cache_path)
            missing_columns = required_columns.difference(df.columns)
            if missing_columns:
                raise ValueError(
                    "missing column(s): " + ", ".join(sorted(missing_columns))
                )
            if len(df) != expected_rows:
                raise ValueError(
                    f"expected {expected_rows} rows, found {len(df)}"
                )
            mu = np.column_stack(
                [df['c1_mean'].to_numpy(float), df['c2_mean'].to_numpy(float)]
            )
            sigma = np.column_stack(
                [
                    df['c1_sigma_star'].to_numpy(float),
                    df['c2_sigma_star'].to_numpy(float),
                ]
            )
            if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
                raise ValueError("contains non-finite coefficients or uncertainties")
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(
                f"[LD prior] Ignoring invalid cache {cache_path}: {error}; "
                "rebuilding atomically.",
                flush=True,
            )
        else:
            print(
                f"[LD prior] Loading cached power2 grid for {instrument} "
                f"({cache_label}) from {cache_path}",
                flush=True,
            )
            sigma = np.maximum(sigma, min_sigma)
            if not has_vector_err:
                return jnp.array(mu[0]), jnp.array(sigma[0])
            return jnp.array(mu), jnp.array(sigma)

    teff_axis = _build_axis(teff, teff_sigma, n_grid, nsigma)
    logg_axis = _build_axis(logg, logg_sigma, n_grid, nsigma)
    feh_axis = _build_axis(feh, feh_sigma, n_grid, nsigma)

    combos = []
    for mh_i in feh_axis:
        for teff_i in teff_axis:
            for logg_i in logg_axis:
                weight = (
                    _gaussian_pdf(mh_i, feh, feh_sigma)
                    * _gaussian_pdf(teff_i, teff, teff_sigma)
                    * _gaussian_pdf(logg_i, logg, logg_sigma)
                )
                combos.append((mh_i, teff_i, logg_i, weight))

    print(
        f"[LD prior] Building power2 grid for {instrument} ({cache_label}) with {len(combos)} stellar combinations using model={ld_prior_model}",
        flush=True,
    )

    successful_mu = []
    weights = []
    for idx, (mh_i, teff_i, logg_i, weight) in enumerate(combos, start=1):
        print(
            f"[LD prior {cache_label}] [{idx}/{len(combos)}] M/H={mh_i:.4f} Teff={teff_i:.1f} logg={logg_i:.3f}",
            flush=True,
        )
        sld_grid = StellarLimbDarkening(
            M_H=mh_i,
            Teff=teff_i,
            logg=logg_i,
            ld_model=ld_prior_model,
            ld_data_path=ld_data_path,
            interpolate_type=ld_interpolate_type,
            verbose=0,
        )
        mu_i = get_limb_darkening(
            sld_grid,
            wavelengths_np,
            wavelength_err if has_vector_err else float(wavelength_err_np[0]),
            instrument,
            order=order,
            ld_profile='power2',
            return_sigmas=False,
        )
        mu_i = np.asarray(mu_i, dtype=float)
        if mu_i.ndim == 1:
            mu_i = mu_i[None, :]
        successful_mu.append(mu_i)
        weights.append(weight)

    mu_stack = np.stack(successful_mu, axis=0)
    weights = np.asarray(weights, dtype=float)
    weights /= np.sum(weights)

    c1_mean, c1_tot, c1_fit, c1_star = _combine_weighted_means_and_sigmas(mu_stack[:, :, 0], weights)
    c2_mean, c2_tot, c2_fit, c2_star = _combine_weighted_means_and_sigmas(mu_stack[:, :, 1], weights)

    c1_star = np.maximum(c1_star, min_sigma)
    c2_star = np.maximum(c2_star, min_sigma)
    # `_combine_weighted_means_and_sigmas` currently defines total scatter as
    # stellar scatter. Apply the floor to both aliases so a newly built prior
    # is numerically identical to the same prior loaded from CSV.
    c1_tot = c1_star.copy()
    c2_tot = c2_star.copy()

    if has_vector_err:
        out_df = pd.DataFrame({
            'wavelength': wavelengths_np,
            'wavelength_err': wavelength_err_np,
            'c1_mean': c1_mean,
            'c1_sigma_total': c1_tot,
            'c1_sigma_fit': c1_fit,
            'c1_sigma_star': c1_star,
            'c2_mean': c2_mean,
            'c2_sigma_total': c2_tot,
            'c2_sigma_fit': c2_fit,
            'c2_sigma_star': c2_star,
        })
        _atomic_dataframe_csv(out_df, cache_path, index=False)
        print(f"[LD prior] Saved power2 grid cache to {cache_path}", flush=True)
        return jnp.array(np.column_stack([c1_mean, c2_mean])), jnp.array(np.column_stack([c1_tot, c2_tot]))
    else:
        out_df = pd.DataFrame({
            'wavelength': [float(np.mean(wavelengths_np))],
            'wavelength_err': [float(wavelength_err_np[0])],
            'c1_mean': [float(c1_mean[0])],
            'c1_sigma_total': [float(c1_tot[0])],
            'c1_sigma_fit': [float(c1_fit[0])],
            'c1_sigma_star': [float(c1_star[0])],
            'c2_mean': [float(c2_mean[0])],
            'c2_sigma_total': [float(c2_tot[0])],
            'c2_sigma_fit': [float(c2_fit[0])],
            'c2_sigma_star': [float(c2_star[0])],
        })
        _atomic_dataframe_csv(out_df, cache_path, index=False)
        print(f"[LD prior] Saved power2 grid cache to {cache_path}", flush=True)
        return jnp.array([c1_mean[0], c2_mean[0]]), jnp.array([c1_tot[0], c2_tot[0]])


def _parse_scalar_like(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            stripped = stripped[1:-1].strip()
        if stripped == '':
            return np.nan
        return float(stripped)
    return float(value)


def _smooth_series(y, window):
    window = int(window)
    if window <= 1:
        return np.asarray(y, dtype=float)
    return pd.Series(np.asarray(y, dtype=float)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def _load_custom_power2_ld_curve(path, target_wavelengths, smooth_window=1):
    path = os.path.expanduser(str(path))
    try:
        raw = pd.read_csv(path, sep=None, engine='python', comment='#')
    except Exception:
        raw = pd.read_csv(path, comment='#')
    if raw.shape[1] <= 1:
        raw = pd.read_csv(path, delim_whitespace=True, comment='#')

    parsed = pd.DataFrame()
    for col in raw.columns:
        try:
            parsed[col] = [_parse_scalar_like(v) for v in raw[col]]
        except Exception:
            parsed[col] = raw[col]

    lower_map = {str(c).strip().lower(): c for c in parsed.columns}
    wcol = lower_map.get('wavelength') or lower_map.get('wavelength_um') or lower_map.get('wl')
    c1col = lower_map.get('c1')
    c2col = lower_map.get('c2')

    if wcol is None or c1col is None or c2col is None:
        numeric_cols = [c for c in parsed.columns if pd.api.types.is_numeric_dtype(parsed[c])]
        if len(numeric_cols) < 3:
            raise ValueError(
                f"Custom LD file {path} must provide wavelength, c1, and c2 columns."
            )
        wcol, c1col, c2col = numeric_cols[:3]

    df = pd.DataFrame({
        'wavelength': pd.to_numeric(parsed[wcol], errors='coerce'),
        'c1': pd.to_numeric(parsed[c1col], errors='coerce'),
        'c2': pd.to_numeric(parsed[c2col], errors='coerce'),
    }).dropna().sort_values('wavelength')

    if df.empty:
        raise ValueError(f"Custom LD file {path} has no usable wavelength/c1/c2 rows.")

    target_wavelengths = np.asarray(target_wavelengths, dtype=float)
    src_wavelengths = df['wavelength'].to_numpy(float)
    if target_wavelengths.min() < src_wavelengths.min() - 1e-9 or target_wavelengths.max() > src_wavelengths.max() + 1e-9:
        raise ValueError(
            f"Custom LD file {path} spans {src_wavelengths.min():.4f}-{src_wavelengths.max():.4f} um, "
            f"but target grid spans {target_wavelengths.min():.4f}-{target_wavelengths.max():.4f} um."
        )

    c1_smooth = _smooth_series(df['c1'].to_numpy(float), smooth_window)
    c2_smooth = _smooth_series(df['c2'].to_numpy(float), smooth_window)

    c1_interp = np.interp(target_wavelengths, src_wavelengths, c1_smooth)
    c2_interp = np.interp(target_wavelengths, src_wavelengths, c2_smooth)
    return jnp.column_stack([c1_interp, c2_interp])

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

def _coerce_wavelength_axis(wavelengths, wavelength_err=None):
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wavelength_err is None:
        wl_err = np.zeros_like(wl)
    else:
        wl_err = np.atleast_1d(np.asarray(wavelength_err, dtype=float))
        if wl_err.size == 1 and wl.size > 1:
            wl_err = np.repeat(wl_err, wl.size)
        if wl_err.size != wl.size:
            raise ValueError(
                f"wavelength_err must be scalar or length {wl.size}, got {wl_err.shape}."
            )
    return wl, wl_err

HARMONICA_INIT_ODD_COEFF = 1e-4
HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION = 2
HARMONICA_LIMB_PRODUCT_CONVENTION = (
    "theta=0:evening/leading;theta=pi:morning/trailing"
)
# Full catalogue of possible odd harmonic names (used by helper functions).
# The active subset is set per-run via the config's harmonica_max_order.
from models.harmonica.core import (
    _ALL_ODD_COEFF_SPECS,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    harmonica_half_area_area_radius_and_q,
)
HARMONICA_ODD_HARMONICS = tuple(name for name, _ in _ALL_ODD_COEFF_SPECS)
_VALID_HARMONICA_SPECTRO_PARAMETERIZATIONS = frozenset(
    {'fractional', 'delta_r', 'half_area'}
)


def _harmonica_frac_site(name):
    return f"{name}_frac"


def _harmonica_coeff_to_frac(coeff, radius):
    radius_arr = np.asarray(radius, dtype=float)
    scale = max(float(np.nanmin(np.abs(radius_arr))), 1e-6)
    return float(coeff) / scale


def _harmonica_coeff_array_to_frac(coeff, radius):
    coeff_arr = np.asarray(coeff, dtype=float)
    radius_arr = np.asarray(radius, dtype=float)
    return coeff_arr / np.maximum(np.abs(radius_arr), 1e-6)


def _harmonica_coeff_array_to_delta_r(coeff):
    return 2.0 * np.asarray(coeff, dtype=float)


def _harmonica_coefficients_to_half_area_init(a0, a1):
    """Convert WL coefficients to an interior, sampler-supported init."""
    area_radius, q = harmonica_half_area_area_radius_and_q(a0, a1)
    area_radius = np.asarray(area_radius, dtype=float)
    q = np.asarray(q, dtype=float)
    # Stay materially inside the truncated distribution support.  Merely
    # applying ``nextafter`` to the endpoint can still round through the
    # inverse transform to +/-inf during NumPyro initialization.
    interior_limit = float(HARMONICA_HALF_AREA_CONVEX_Q_LIMIT) * (1.0 - 1e-10)
    finite_radius = np.sqrt(
        np.maximum(
            0.0,
            np.asarray(a0, dtype=float) ** 2
            + 0.5 * np.asarray(a1, dtype=float) ** 2,
        )
    )
    radius_low = HARMONICA_SPECTRO_RORS_MIN * (1.0 + 1e-10)
    radius_high = HARMONICA_SPECTRO_RORS_MAX * (1.0 - 1e-10)
    area_radius = np.where(np.isfinite(area_radius), area_radius, finite_radius)
    area_radius = np.clip(
        np.nan_to_num(
            area_radius,
            nan=radius_low,
            posinf=radius_high,
            neginf=radius_low,
        ),
        radius_low,
        radius_high,
    )
    q = np.nan_to_num(q, nan=0.0, posinf=interior_limit, neginf=-interior_limit)
    return area_radius, np.clip(q, -interior_limit, interior_limit)


def _validate_harmonica_spectro_parameterization(value, max_harmonic_order):
    """Normalize and validate the configured Harmonica spectroscopic shape."""
    value = str(value).lower()
    if value not in _VALID_HARMONICA_SPECTRO_PARAMETERIZATIONS:
        choices = "{'fractional', 'delta_r', 'half_area'}"
        raise ValueError(
            "flags.harmonica_spectro_parameterization must be one of "
            f"{choices}. Received '{value}'."
        )
    if value == 'half_area' and int(max_harmonic_order) != 1:
        raise ValueError(
            "flags.harmonica_spectro_parameterization='half_area' requires "
            "flags.harmonica_max_order: 1."
        )
    return value


def _harmonica_model_radius_samples(samples, parameterization):
    """Select the radius consumed by the Harmonica forward model.

    ``half_area`` deliberately stores the total-area-equivalent radius in the
    usual ``rors`` site, while its deterministic ``a0`` is the mean polar
    radius required by the Harmonica transmission string.  Legacy
    parameterizations continue to use ``rors`` directly.
    """
    if parameterization == 'half_area':
        if 'a0' not in samples:
            raise KeyError(
                "half_area posterior samples must contain deterministic 'a0'."
            )
        return samples['a0']
    return samples['rors']


def _harmonica_checkpoint_prefix(base_prefix, transit_engine, parameterization):
    """Keep opt-in half-area chunks separate from legacy checkpoint families."""
    if transit_engine == 'harmonica' and parameterization == 'half_area':
        return f"{base_prefix}_half_area"
    return base_prefix


def _harmonica_spectro_artifact_stem(base_stem, transit_engine, parameterization):
    """Isolate half-area LR reuse artifacts from legacy parameterizations."""
    return _harmonica_checkpoint_prefix(
        base_stem, transit_engine, parameterization
    )


def _coerce_harmonica_sample_matrix(samples, planet_index=0):
    arr = np.asarray(samples)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    if arr.ndim == 3:
        return arr[:, :, planet_index]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported harmonica sample shape: {arr.shape}")


def _extract_harmonica_limb_samples(rors_samples, harmonic_samples, planet_index=0):
    a0 = np.asarray(rors_samples)

    if a0.ndim == 1:
        a0 = a0[:, np.newaxis]
    elif a0.ndim == 3:
        a0 = a0[:, :, planet_index]
    elif a0.ndim != 2:
        raise ValueError(f"Unsupported rors sample shape: {a0.shape}")

    coeff_samples = {}
    for name in HARMONICA_ODD_HARMONICS:
        if name not in harmonic_samples:
            continue
        coeff = _coerce_harmonica_sample_matrix(
            harmonic_samples[name], planet_index=planet_index
        )
        if coeff.shape[1] == 1 and a0.shape[1] > 1:
            coeff = np.repeat(coeff, a0.shape[1], axis=1)
        if a0.shape != coeff.shape:
            raise ValueError(
                f"rors and {name} sample shapes must match after broadcasting, got {a0.shape} and {coeff.shape}."
            )
        coeff_samples[name] = coeff

    return a0, coeff_samples


def _sum_harmonica_odd_samples(coeff_samples):
    if not coeff_samples:
        return 0.0
    total = None
    for coeff in coeff_samples.values():
        total = coeff if total is None else total + coeff
    return total


def _harmonica_limb_product_samples(a0_samples, coeff_samples):
    """Return Catwoman-comparable limb depths and endpoint diagnostics.

    Harmonica's odd-cosine transmission string is

    ``r(theta) = a0 + sum_n a_n cos(n theta)``.

    Harmonica measures ``theta`` from the orbital-velocity direction, so
    ``evening``/``leading`` denotes the half centred on ``theta=0`` and
    ``morning``/``trailing`` the opposite half centred on ``theta=pi``.
    Each representative limb depth is twice that half's area divided by pi,
    so it is directly comparable to the depth of a circular semicircle.  For
    N_c=1 this gives

    ``D_evening = a0**2 + a1**2/2 + 4*a0*a1/pi`` and
    ``D_morning = a0**2 + a1**2/2 - 4*a0*a1/pi``.

    The implementation also handles the supported higher odd cosine terms
    exactly.  Orthogonality makes their squared contributions add in
    quadrature, while their signed half-area coefficient is
    ``sum_n a_n sin(n*pi/2)/n``.
    """
    a0 = np.asarray(a0_samples, dtype=float)
    odd_sum = np.zeros_like(a0)
    odd_square_sum = np.zeros_like(a0)
    area_asymmetry_coefficient = np.zeros_like(a0)

    for name, coeff_samples_for_order in coeff_samples.items():
        order = int(name[1:])
        if order % 2 != 1:
            raise ValueError(
                f"Harmonica limb products require odd cosine terms; got {name}."
            )
        coeff = np.asarray(coeff_samples_for_order, dtype=float)
        if coeff.shape != a0.shape:
            raise ValueError(
                f"a0 and {name} sample shapes must match, got {a0.shape} and {coeff.shape}."
            )
        odd_sum = odd_sum + coeff
        odd_square_sum = odd_square_sum + coeff**2
        half_area_sign = (-1.0) ** ((order - 1) // 2)
        area_asymmetry_coefficient = (
            area_asymmetry_coefficient + coeff * half_area_sign / order
        )

    total_area_depth = a0**2 + 0.5 * odd_square_sum
    half_area_contrast = (4.0 / np.pi) * a0 * area_asymmetry_coefficient
    leading_endpoint_radius = a0 + odd_sum
    trailing_endpoint_radius = a0 - odd_sum

    return {
        "depth_morning": total_area_depth - half_area_contrast,
        "depth_evening": total_area_depth + half_area_contrast,
        "depth_total_area": total_area_depth,
        "depth_leading_endpoint": leading_endpoint_radius**2,
        "depth_trailing_endpoint": trailing_endpoint_radius**2,
        "asymmetry_coefficient": area_asymmetry_coefficient,
        "endpoint_delta_r": leading_endpoint_radius - trailing_endpoint_radius,
    }


def _harmonica_percentile_summary(samples):
    """Return posterior median and asymmetric 16th/84th percentile errors."""
    median = np.nanpercentile(samples, 50, axis=0)
    err_lo = median - np.nanpercentile(samples, 16, axis=0)
    err_hi = np.nanpercentile(samples, 84, axis=0) - median
    return median, err_lo, err_hi


def _harmonica_r_vector_from_values(a0, coeff_values):
    active_orders = [
        int(name[1:])
        for name in HARMONICA_ODD_HARMONICS
        if name in coeff_values
    ]
    highest_order = max(active_orders, default=1)
    r = np.zeros(2 * highest_order + 1, dtype=float)
    r[0] = a0
    for name, value in coeff_values.items():
        order = int(name[1:])
        r[2 * order - 1] = value
    return r


def _harmonica_sample_payload(samples):
    return {name: samples[name] for name in HARMONICA_ODD_HARMONICS if name in samples}


def _has_harmonica_odd_samples(samples):
    return any(name in samples for name in HARMONICA_ODD_HARMONICS)


def _augment_harmonica_params(params, a_rs, ecc, omega, a1=None, a3=None, a5=None,
                              c_ld=None, alpha_ld=None, u1_ld=None, u2_ld=None):
    out = dict(params)
    out["a_rs"] = jnp.asarray(a_rs)
    out["ecc"] = jnp.asarray(ecc)
    out["omega"] = jnp.asarray(omega)
    if a1 is not None:
        out["a1"] = jnp.asarray(a1)
    if a3 is not None:
        out["a3"] = jnp.asarray(a3)
    if a5 is not None:
        out["a5"] = jnp.asarray(a5)
    if c_ld is not None:
        out["c_ld"] = jnp.asarray(c_ld)
    if alpha_ld is not None:
        out["alpha_ld"] = jnp.asarray(alpha_ld)
    if u1_ld is not None:
        out["u1_ld"] = jnp.asarray(u1_ld)
    if u2_ld is not None:
        out["u2_ld"] = jnp.asarray(u2_ld)
    return out


_JAXOPLANET_STATIC_EVAL_KEYS = frozenset(
    {"_jaxoplanet_kernel", "_ld_profile"}
)
_JAXOPLANET_DYNAMIC_EVAL_KEYS = (
    "_transit_phase_offsets",
    "_transit_phase_mask",
    "_transit_window_indices",
)


def _posterior_quadratic_ld_median(samples, fallback):
    """Return direct quadratic ``u1,u2`` coefficients for post-fit models.

    Jaxoplanet stores sampled direct quadratic coefficients in the joint ``u``
    site, whereas Harmonica exposes fixed coefficients as separate ``u1`` and
    ``u2`` deterministic sites.  Prefer the joint posterior whenever present;
    it is the actual coefficient vector used by the Jaxoplanet likelihood.
    """
    fallback = jnp.asarray(fallback, dtype=jnp.float64)
    if fallback.ndim == 0 or fallback.shape[-1] != 2:
        raise ValueError(
            "Quadratic limb-darkening fallback must have final dimension 2."
        )

    if "u" in samples:
        direct_u = jnp.asarray(samples["u"], dtype=jnp.float64)
        if direct_u.ndim < 2 or direct_u.shape[-1] != 2:
            raise ValueError(
                "Quadratic posterior site 'u' must have final dimension 2."
            )
        return jnp.nanmedian(direct_u, axis=0)

    u1 = (
        jnp.nanmedian(jnp.asarray(samples["u1"], dtype=jnp.float64), axis=0)
        if "u1" in samples
        else fallback[..., 0]
    )
    u2 = (
        jnp.nanmedian(jnp.asarray(samples["u2"], dtype=jnp.float64), axis=0)
        if "u2" in samples
        else fallback[..., 1]
    )
    return jnp.stack((u1, u2), axis=-1)


def _attach_jaxoplanet_eval_metadata(
    params,
    t,
    *,
    jaxoplanet_kernel,
    ld_profile,
    transit_window_optimization="off",
):
    """Attach routing and time-axis-specific acceleration data to diagnostics.

    Phase offsets and window indices are rebuilt for *t*, so a post-fit model
    cannot accidentally reuse indices from a pre-clipping cadence axis.
    """
    from models.jaxoplanet.core import (
        build_transit_phase_offsets,
        build_transit_window_indices,
    )

    out = dict(params)
    for name in (*_JAXOPLANET_STATIC_EVAL_KEYS, *_JAXOPLANET_DYNAMIC_EVAL_KEYS):
        out.pop(name, None)
    out["_jaxoplanet_kernel"] = str(jaxoplanet_kernel)
    out["_ld_profile"] = str(ld_profile)

    duration_geometry = "duration" in out and "a_rs" not in out
    if not duration_geometry:
        return out

    phase_offsets, phase_mask = build_transit_phase_offsets(
        t, out["period"], out["t0"], out["duration"]
    )
    out["_transit_phase_offsets"] = phase_offsets
    out["_transit_phase_mask"] = phase_mask
    if transit_window_optimization == "auto":
        indices = build_transit_window_indices(
            np.asarray(t),
            np.asarray(out["period"]),
            np.asarray(out["t0"]),
            np.asarray(out["duration"]),
        )
        out["_transit_window_indices"] = jnp.asarray(indices, dtype=jnp.int32)
    elif transit_window_optimization != "off":
        raise ValueError(
            "transit_window_optimization must be one of {'auto', 'off'}."
        )
    return out


def _select_transit_eval_params(
    params,
    transit_engine,
    param_method,
    *,
    t=None,
    jaxoplanet_kernel=None,
    ld_profile=None,
    transit_window_optimization="off",
):
    out = dict(params)
    if transit_engine != 'jaxoplanet':
        return out
    if param_method == 'a_rs':
        out.pop('duration', None)
    else:
        out.pop('a_rs', None)
        out.pop('ecc', None)
        out.pop('omega', None)
        out.pop('cos_i', None)
        out.pop('inc', None)
    if t is not None and jaxoplanet_kernel is not None and ld_profile is not None:
        out = _attach_jaxoplanet_eval_metadata(
            out,
            t,
            jaxoplanet_kernel=jaxoplanet_kernel,
            ld_profile=ld_profile,
            transit_window_optimization=transit_window_optimization,
        )
    return out


def _harmonica_cosi_from_b(b, a_rs, ecc, omega):
    return jnp.clip(
        harmonica_cos_i_from_geometry(
            jnp.asarray(b, dtype=jnp.float64),
            jnp.asarray(a_rs, dtype=jnp.float64),
            ecc=jnp.asarray(ecc, dtype=jnp.float64),
            omega=jnp.asarray(omega, dtype=jnp.float64),
        ),
        0.0,
        1.0,
    )


def build_harmonica_limb_dataframe(wavelengths, wavelength_err, rors_samples, harmonic_samples,
                                   bandpass_min=None, bandpass_max=None, planet_index=0):
    wl_arr, wl_err_arr = _coerce_wavelength_axis(wavelengths, wavelength_err)
    a0_samp, coeff_samp = _extract_harmonica_limb_samples(
        rors_samples, harmonic_samples, planet_index=planet_index
    )
    if wl_arr.size != a0_samp.shape[1]:
        raise ValueError(
            f"Number of wavelengths ({wl_arr.size}) does not match sample axis ({a0_samp.shape[1]})."
        )

    limb_products = _harmonica_limb_product_samples(a0_samp, coeff_samp)
    depth_products_ppm = {
        name: values * 1e6
        for name, values in limb_products.items()
        if name.startswith("depth_")
    }
    a0_depth_samp = (a0_samp**2) * 1e6

    product_summaries = {
        name: _harmonica_percentile_summary(values)
        for name, values in depth_products_ppm.items()
    }
    morning_med, morning_lo, morning_hi = product_summaries["depth_morning"]
    evening_med, evening_lo, evening_hi = product_summaries["depth_evening"]
    total_med, total_lo, total_hi = product_summaries["depth_total_area"]
    leading_endpoint_med, leading_endpoint_lo, leading_endpoint_hi = (
        product_summaries["depth_leading_endpoint"]
    )
    trailing_endpoint_med, trailing_endpoint_lo, trailing_endpoint_hi = (
        product_summaries["depth_trailing_endpoint"]
    )
    a0_depth_med, a0_depth_lo, a0_depth_hi = _harmonica_percentile_summary(
        a0_depth_samp
    )

    asymmetry_med, asymmetry_lo, asymmetry_hi = _harmonica_percentile_summary(
        limb_products["asymmetry_coefficient"]
    )
    delta_r_med, delta_r_lo, delta_r_hi = _harmonica_percentile_summary(
        limb_products["endpoint_delta_r"]
    )

    # Raw Rp/Rs statistics (needed for transmission string plots)
    a0_raw_med, a0_raw_lo, a0_raw_hi = _harmonica_percentile_summary(a0_samp)

    limb_df = pd.DataFrame({
        "wavelength": wl_arr,
        "wavelength_err": wl_err_arr,
        "planet_index": int(planet_index),
        "limb_product_schema_version": HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION,
        "limb_product_convention": HARMONICA_LIMB_PRODUCT_CONVENTION,
        # Catwoman-comparable, full-circle-equivalent depths from each half-area.
        "depth_morning_median": morning_med,
        "depth_morning_err_lo": morning_lo,
        "depth_morning_err_hi": morning_hi,
        "depth_evening_median": evening_med,
        "depth_evening_err_lo": evening_lo,
        "depth_evening_err_hi": evening_hi,
        "depth_total_area_median": total_med,
        "depth_total_area_err_lo": total_lo,
        "depth_total_area_err_hi": total_hi,
        # Extrema at theta=0 and theta=pi retained as explicit diagnostics.
        "depth_leading_endpoint_median": leading_endpoint_med,
        "depth_leading_endpoint_err_lo": leading_endpoint_lo,
        "depth_leading_endpoint_err_hi": leading_endpoint_hi,
        "depth_trailing_endpoint_median": trailing_endpoint_med,
        "depth_trailing_endpoint_err_lo": trailing_endpoint_lo,
        "depth_trailing_endpoint_err_hi": trailing_endpoint_hi,
        "asymmetry_coefficient_median": asymmetry_med,
        "asymmetry_coefficient_err_lo": asymmetry_lo,
        "asymmetry_coefficient_err_hi": asymmetry_hi,
        "endpoint_delta_r_median": delta_r_med,
        "endpoint_delta_r_err_lo": delta_r_lo,
        "endpoint_delta_r_err_hi": delta_r_hi,
        # a0**2 is retained for compatibility but is not the total silhouette area.
        "depth_a0_median": a0_depth_med,
        "depth_a0_err_lo": a0_depth_lo,
        "depth_a0_err_hi": a0_depth_hi,
        "a0_median": a0_raw_med,
        "a0_err_lo": a0_raw_lo,
        "a0_err_hi": a0_raw_hi,
    })
    for name, coeff in coeff_samp.items():
        coeff_med, coeff_lo, coeff_hi = _harmonica_percentile_summary(coeff)
        limb_df[f"{name}_median"] = coeff_med
        limb_df[f"{name}_err_lo"] = coeff_lo
        limb_df[f"{name}_err_hi"] = coeff_hi
    if bandpass_min is not None:
        limb_df["bandpass_min"] = bandpass_min
    if bandpass_max is not None:
        limb_df["bandpass_max"] = bandpass_max
    return limb_df, a0_samp, coeff_samp


def _save_harmonica_limb_posterior_samples(
    path,
    wavelengths,
    wavelength_err,
    a0_samples,
    coefficient_samples,
    planet_index=0,
):
    """Persist the joint limb posterior without marginalizing its covariance."""
    wl_arr, wl_err_arr = _coerce_wavelength_axis(wavelengths, wavelength_err)
    a0_samp = np.asarray(a0_samples, dtype=np.float64)
    coeff_samp = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in coefficient_samples.items()
    }
    if a0_samp.ndim != 2 or a0_samp.shape[1] != wl_arr.size:
        raise ValueError(
            "a0_samples must have shape (draw, wavelength) matching wavelengths."
        )
    for name, values in coeff_samp.items():
        if values.shape != a0_samp.shape:
            raise ValueError(
                f"{name} samples have shape {values.shape}; expected {a0_samp.shape}."
            )

    products = _harmonica_limb_product_samples(a0_samp, coeff_samp)
    payload = {
        "limb_product_schema_version": np.asarray(
            HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION, dtype=np.int64
        ),
        "limb_product_convention": np.asarray(HARMONICA_LIMB_PRODUCT_CONVENTION),
        "sample_axes": np.asarray("draw,wavelength"),
        "depth_units": np.asarray("fractional_stellar_area"),
        "wavelength_units": np.asarray("micron"),
        "planet_index": np.asarray(planet_index, dtype=np.int64),
        "wavelength": wl_arr,
        "wavelength_err": wl_err_arr,
        "a0": a0_samp,
    }
    payload.update(coeff_samp)
    payload.update(products)
    _atomic_savez_compressed(path, **payload)
    print(f"Saved joint limb posterior samples to {path}")

def save_harmonica_limb_products(
    wavelengths,
    wavelength_err,
    rors_samples,
    harmonic_samples,
    csv_path,
    limb_spectrum_path,
    transmission_strings_path,
    title_prefix,
    bandpass_min=None,
    bandpass_max=None,
    posterior_strings_path=None,
    posterior_samples_path=None,
    planet_index=0,
):
    from harmonica import HarmonicaTransit

    limb_df, a0_samp, coeff_samp = build_harmonica_limb_dataframe(
        wavelengths,
        wavelength_err,
        rors_samples,
        harmonic_samples,
        bandpass_min=bandpass_min,
        bandpass_max=bandpass_max,
        planet_index=planet_index,
    )
    _atomic_dataframe_csv(limb_df, csv_path, index=False)
    print(f"Saved limb spectra to {csv_path}")

    if posterior_samples_path is not None:
        _save_harmonica_limb_posterior_samples(
            posterior_samples_path,
            wavelengths,
            wavelength_err,
            a0_samp,
            coeff_samp,
            planet_index=planet_index,
        )

    wl_arr = limb_df["wavelength"].values
    wl_spacing = np.nanmedian(np.diff(wl_arr)) if wl_arr.size > 1 else 0.0
    offset = 0.4 * wl_spacing if wl_arr.size > 1 else max(
        0.01,
        0.15 * float(np.nanmax(limb_df["wavelength_err"].values) if limb_df["wavelength_err"].values.size else 0.0),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        wl_arr - offset,
        limb_df["depth_evening_median"].values,
        yerr=[limb_df["depth_evening_err_lo"].values, limb_df["depth_evening_err_hi"].values],
        fmt="o",
        ms=3,
        lw=1,
        mfc="red",
        mec="k",
        mew=1.2,
        ecolor="red",
        capsize=0,
        label="Evening limb (leading half-area equivalent)",
    )
    ax.errorbar(
        wl_arr + offset,
        limb_df["depth_morning_median"].values,
        yerr=[limb_df["depth_morning_err_lo"].values, limb_df["depth_morning_err_hi"].values],
        fmt="o",
        ms=3,
        lw=1,
        mfc="blue",
        mec="k",
        mew=1.2,
        ecolor="blue",
        capsize=0,
        label="Morning limb (trailing half-area equivalent)",
    )
    ax.set_xlabel("Wavelength [$\\mu$m]", fontsize=12)
    ax.set_ylabel("Transit Depth [ppm]", fontsize=12)
    ax.set_title(
        f"{title_prefix} - Representative Evening/Morning Limb Spectra",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(limb_spectrum_path, dpi=200)
    plt.close(fig)
    print(f"Saved limb spectra plot to {limb_spectrum_path}")

    theta = np.linspace(-np.pi, np.pi, 1000)
    ht = HarmonicaTransit()

    a0_med = limb_df["a0_median"].values
    coeff_med = {
        name: limb_df[f"{name}_median"].values
        for name in HARMONICA_ODD_HARMONICS
        if f"{name}_median" in limb_df
    }

    if wl_arr.size > 1:
        colours = plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(wl_arr)))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal", "datalim")
        for i in range(len(wl_arr)):
            r_i = _harmonica_r_vector_from_values(
                a0_med[i],
                {name: values[i] for name, values in coeff_med.items()},
            )
            ht.set_planet_transmission_string(r_i.copy())
            rp = ht.get_planet_transmission_string(theta)
            ax.plot(
                rp * np.cos(theta),
                rp * np.sin(theta),
                color=colours[i],
                lw=1.4,
                label=f"{wl_arr[i]:.3f} $\\mu$m",
            )
        r0_ref = float(np.nanmedian(a0_med))
        ax.plot(
            r0_ref * np.cos(theta),
            r0_ref * np.sin(theta),
            c="#b9b9b9",
            ls="--",
            lw=0.8,
            label="Reference circle",
        )
        ax.set_xlabel("x / stellar radii", fontsize=12)
        ax.set_ylabel("y / stellar radii", fontsize=12)
        ax.set_title(f"{title_prefix} - Transmission Strings", fontsize=13)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
        plt.tight_layout()
        plt.savefig(transmission_strings_path, dpi=200)
        plt.close(fig)
        print(f"Saved transmission string plot to {transmission_strings_path}")
    else:
        posterior_strings_path = transmission_strings_path

    if posterior_strings_path is not None:
        i_show = len(wl_arr) // 2
        n_draw = min(200, a0_samp.shape[0])
        draw_idx = np.random.choice(a0_samp.shape[0], n_draw, replace=False)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_aspect("equal", "datalim")
        for j in draw_idx:
            r_sample = _harmonica_r_vector_from_values(
                a0_samp[j, i_show],
                {
                    name: coeff[j, i_show]
                    for name, coeff in coeff_samp.items()
                },
            )
            ht.set_planet_transmission_string(r_sample.copy())
            rp = ht.get_planet_transmission_string(theta)
            ax.plot(
                rp * np.cos(theta),
                rp * np.sin(theta),
                c=cm.BuGn(0.8),
                alpha=0.05,
            )
        r_med = _harmonica_r_vector_from_values(
            a0_med[i_show],
            {name: values[i_show] for name, values in coeff_med.items()},
        )
        ht.set_planet_transmission_string(r_med.copy())
        rp_med = ht.get_planet_transmission_string(theta)
        ax.plot(
            rp_med * np.cos(theta),
            rp_med * np.sin(theta),
            c=cm.inferno(0.1),
            lw=2.0,
            label="Median transmission string",
        )
        ax.plot(
            a0_med[i_show] * np.cos(theta),
            a0_med[i_show] * np.sin(theta),
            c="#b9b9b9",
            ls="--",
            label="Reference circle",
        )
        ax.set_xlabel("x / stellar radii", fontsize=13)
        ax.set_ylabel("y / stellar radii", fontsize=13)
        if wl_arr.size > 1:
            ax.set_title(
                f"{title_prefix} - Recovered transmission string ({wl_arr[i_show]:.3f} $\\mu$m)",
                fontsize=13,
            )
        else:
            ax.set_title(f"{title_prefix} - Recovered transmission string", fontsize=13)
        ax.legend(loc="lower left", fontsize=12)
        plt.tight_layout()
        plt.savefig(posterior_strings_path, dpi=200)
        plt.close(fig)
        print(f"Saved transmission string posterior plot to {posterior_strings_path}")

    return limb_df

def save_detailed_fit_results(time, flux, flux_err, wavelengths, wavelengths_err, samples, map_params,
                               transit_params, detrend_type, output_prefix,
                               total_error_fit=None, gp_trend=None, spot_trend=None, jump_trend=None):
    n_wavelengths = len(wavelengths)
    n_times = len(time)
    print(f"Saving detailed fit results to {output_prefix}_*.csv")
    param_rows = []
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
def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instrument = cfg['instrument']
    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        nrs = cfg['nrs']
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
    
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']
    flags = cfg.get('flags', {})
    # Production defaults validated by the 2026-09 acceleration campaign.
    # setdefault preserves every explicit legacy/user selection.
    is_prism = instrument == 'NIRSPEC/PRISM'
    is_explinear = 'explinear' in str(flags.get('detrending_type', 'linear'))
    flags.setdefault('spectro_sampler', 'independent_nuts')
    flags.setdefault('spectro_mass_matrix', 'laplace')
    flags.setdefault('spectro_laplace_warmup', 150)
    flags.setdefault('spectro_laplace_target_accept', 0.99 if (is_prism or is_explinear) else 0.95)
    flags.setdefault('spectro_laplace_max_tree_depth', 6 if is_prism else 5)
    flags.setdefault('spectro_laplace_trust_radius', 5.0)
    flags.setdefault('spectro_laplace_hessian_method', 'finite_difference')
    flags.setdefault('spectro_laplace_start_at_map', True)
    flags.setdefault('spectro_hmc_num_steps', 8)
    flags.setdefault('spectro_hmc_trajectory_jitter', 0.25)
    flags.setdefault('spectro_min_depth_ess', 400)
    flags.setdefault('spectro_max_divergences', 0)
    flags.setdefault('whitelight_mass_matrix', 'laplace')
    flags.setdefault('whitelight_laplace_target_accept', 0.99 if is_prism else 0.9)
    flags.setdefault('whitelight_min_ess', 400)
    flags.setdefault('whitelight_max_divergences', 0)
    flags.setdefault('whitelight_max_extra_blocks', 3)
    flags.setdefault('whitelight_geometry_estimator', 'posterior_median')
    flags.setdefault('spectro_fixed_timescale_trends', True)
    if bool(flags.get('compile_box', False)):
        compilation_cache_dir = str(
            flags.get(
                'jax_compilation_cache_dir',
                '/scratch/midway3/tfairnington/jax_cache',
            )
        )
        os.makedirs(compilation_cache_dir, exist_ok=True)
        jax.config.update('jax_compilation_cache_dir', compilation_cache_dir)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)
        print(
            "Compile box enabled: persistent JAX cache at "
            f"{compilation_cache_dir}."
        )
    resolution = cfg.get('resolution', None)
    pixels = cfg.get('pixels', None)
    
    if resolution is None:
        if pixels is None: raise ValueError('Must Specify Resolutions or Pixels')
        bins = pixels
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)
    elif pixels is None:
        bins = resolution
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)

    outlier_clip = cfg.get('outlier_clip', {})
    planet_str = planet_cfg['name']
    mask_integrations_start = outlier_clip.get('mask_integrations_start', None)
    mask_integrations_end = outlier_clip.get('mask_integrations_end', None)

    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    fits_file = os.path.join(input_dir, cfg.get('fits_file'))
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    detrending_type = flags.get('detrending_type', 'linear')
    interpolate_trend = flags.get('interpolate_trend', False)
    interpolate_ld = flags.get('interpolate_ld', False)
    need_lowres = flags.get('need_lowres', True)
    mask_start = flags.get('mask_start', False)
    mask_end = flags.get('mask_end', False)
    spot_amp = flags.get('spot_amp', 0.0)
    spot_mu = flags.get('spot_center', 0.0)
    spot_sigma = flags.get('spot_width', 0.0)
    spot_amp2 = flags.get('spot_amp2', flags.get('spot_amp_2', 0.0))
    spot_mu2 = flags.get('spot_center2', flags.get('spot_center_2', 0.0))
    spot_sigma2 = flags.get('spot_width2', flags.get('spot_width_2', 0.0))
    t_jump_guess = flags.get('t_jump_guess', None)
    jump_guess = flags.get('jump_guess', 0.0)
    hr_custom_ld_path = flags.get('hr_custom_ld_path', None)
    hr_custom_ld_smooth_window = int(flags.get('hr_custom_ld_smooth_window', 1) or 1)
    save_trace = flags.get('save_whitelight_trace', False)
    whitelight_geometry_estimator = str(
        flags.get('whitelight_geometry_estimator', 'posterior_median')
    ).lower()
    if whitelight_geometry_estimator not in {
        'max_likelihood_draw', 'posterior_median'
    }:
        raise ValueError(
            "flags.whitelight_geometry_estimator must be one of "
            "{'max_likelihood_draw', 'posterior_median'}. "
            f"Received '{whitelight_geometry_estimator}'."
        )
    whitelight_log_likelihood_batch_size = int(
        flags.get('whitelight_log_likelihood_batch_size', 64)
    )
    if whitelight_log_likelihood_batch_size < 1:
        raise ValueError(
            "flags.whitelight_log_likelihood_batch_size must be >= 1."
        )
    transit_engine = flags.get('transit_engine', 'jaxoplanet')
    spectro_fixed_timescale_trends = bool(
        flags.get('spectro_fixed_timescale_trends', True)
    )
    if (spectro_fixed_timescale_trends and 'explinear' in detrending_type
            and transit_engine != 'jaxoplanet'):
        raise ValueError(
            "flags.spectro_fixed_timescale_trends currently supports only "
            "flags.transit_engine='jaxoplanet'."
        )
    _validate_interpolated_trend_mode(
        interpolate_trend,
        transit_engine=transit_engine,
        detrending_type=detrending_type,
    )
    spectro_sampler = str(flags.get('spectro_sampler', 'independent_nuts')).lower()
    spectro_jitter_prior = str(flags.get('spectro_jitter_prior', 'lognormal')).lower()
    if spectro_jitter_prior not in {'log_uniform', 'lognormal'}:
        raise ValueError(
            "flags.spectro_jitter_prior must be 'log_uniform' or 'lognormal'. "
            f"Received '{spectro_jitter_prior}'."
        )
    spectro_jitter_prior_scale = float(flags.get('spectro_jitter_prior_scale', 2.0))
    spectro_jitter_prior_center = float(flags.get('spectro_jitter_prior_center', 0.5))
    if spectro_jitter_prior_scale <= 0.0 or spectro_jitter_prior_center <= 0.0:
        raise ValueError(
            "flags.spectro_jitter_prior_scale and flags.spectro_jitter_prior_center must be > 0."
        )
    spectro_ld_parameterization = str(
        flags.get('spectro_ld_parameterization', 'coefficients')
    ).lower()
    whitelight_ld_parameterization = str(
        flags.get('whitelight_ld_parameterization', 'coefficients')
    ).lower()
    for name, value in (
        ('spectro_ld_parameterization', spectro_ld_parameterization),
        ('whitelight_ld_parameterization', whitelight_ld_parameterization),
    ):
        if value not in {'coefficients', 'decorrelated'}:
            raise ValueError(
                f"flags.{name} must be 'coefficients' or 'decorrelated'."
            )
    if spectro_sampler not in {
        'joint_nuts', 'independent_nuts', 'independent_hmc', 'laplace_is'
    }:
        raise ValueError(
            "flags.spectro_sampler must be one of "
            "{'joint_nuts', 'independent_nuts', 'independent_hmc', "
            "'laplace_is'}. "
            f"Received '{spectro_sampler}'."
        )
    spectro_mass_matrix = str(
        flags.get('spectro_mass_matrix', 'laplace')
    ).lower()
    if spectro_mass_matrix not in {'adaptive', 'laplace'}:
        raise ValueError(
            "flags.spectro_mass_matrix must be 'adaptive' or 'laplace'. "
            f"Received '{spectro_mass_matrix}'."
        )
    if spectro_mass_matrix == 'laplace' and spectro_sampler not in {
        'independent_nuts', 'independent_hmc'
    }:
        raise ValueError(
            "flags.spectro_mass_matrix='laplace' requires "
            "flags.spectro_sampler='independent_nuts' or 'independent_hmc'."
        )
    if spectro_sampler == 'laplace_is' and transit_engine != 'jaxoplanet':
        raise ValueError(
            f"flags.spectro_sampler='{spectro_sampler}' currently supports only "
            "flags.transit_engine='jaxoplanet'."
        )
    if spectro_sampler in {'independent_nuts', 'independent_hmc'} and transit_engine not in {
        'jaxoplanet', 'harmonica'
    }:
        raise ValueError(
            f"flags.spectro_sampler='{spectro_sampler}' does not support "
            f"flags.transit_engine={transit_engine!r}."
        )
    transit_window_optimization = str(
        flags.get('transit_window_optimization', 'auto')
    ).lower()
    if transit_window_optimization not in {'auto', 'off'}:
        raise ValueError(
            "flags.transit_window_optimization must be one of {'auto', 'off'}. "
            f"Received '{transit_window_optimization}'."
        )
    jaxoplanet_kernel = str(
        flags.get('jaxoplanet_kernel', 'auto')
    ).lower()
    if jaxoplanet_kernel not in {
        'auto', 'stock', 'streamed', 'fused', 'quadratic_specialized',
        'quadratic_local_jvp', 'native_power2'
    }:
        raise ValueError(
            "flags.jaxoplanet_kernel must be one of "
            "{'auto', 'stock', 'streamed', 'fused', "
            "'quadratic_specialized', 'quadratic_local_jvp', "
            "'native_power2'}. "
            f"Received '{jaxoplanet_kernel}'."
        )
    spectro_sampling_mode_raw = str(flags.get('spectro_sampling_mode', 'auto')).lower()
    if spectro_sampling_mode_raw not in {'auto', 'joint', 'independent'}:
        raise ValueError(
            "flags.spectro_sampling_mode must be one of {'auto', 'joint', 'independent'}. "
            f"Received '{spectro_sampling_mode_raw}'."
        )
    if spectro_sampling_mode_raw == 'auto':
        spectro_sampling_mode = 'independent' if transit_engine == 'harmonica' else 'joint'
    else:
        spectro_sampling_mode = spectro_sampling_mode_raw
    vmap_chunk = flags.get('vmap_chunk', False)
    vmap_chunk_size = None
    if isinstance(vmap_chunk, (int, float)) and not isinstance(vmap_chunk, bool):
        vmap_chunk_size = int(vmap_chunk)
    elif vmap_chunk is True:
        vmap_chunk_size = 50
    elif spectro_sampler in {
        'independent_nuts', 'independent_hmc', 'laplace_is'
    }:
        vmap_chunk_size = 40
        print(
            f"flags.spectro_sampler='{spectro_sampler}' without "
            "flags.vmap_chunk; defaulting to 40 resident GPU lanes."
        )
    elif spectro_sampling_mode == 'independent':
        vmap_chunk_size = 1
        print(
            "flags.spectro_sampling_mode='independent' without flags.vmap_chunk; "
            "defaulting spectroscopic chunk size to 1 channel per job."
        )
    if vmap_chunk_size is not None and vmap_chunk_size < 1:
        raise ValueError("flags.vmap_chunk must resolve to an integer >= 1.")
    # Measured widths are resolved only when their stage is actually about to
    # run, after the requested JAX backend and exact cadence/model workload are
    # known.  This avoids initializing a GPU for a CPU/prep-only job and keeps
    # irrelevant stage manifests from aborting the run.
    vmap_chunk_size_lr = vmap_chunk_size
    vmap_chunk_size_hr = vmap_chunk_size
    analysis_stage = str(
        os.getenv('JWSTJAXFIT_ANALYSIS_STAGE', flags.get('analysis_stage', 'all'))
    ).lower()
    chunk_mode = str(
        os.getenv('JWSTJAXFIT_CHUNK_MODE', flags.get('chunk_mode', 'serial'))
    ).lower()
    chunk_parallel_job_count = flags.get('chunk_parallel_job_count', None)
    chunk_parallel_job_index = flags.get('chunk_parallel_job_index', None)
    if chunk_parallel_job_count is not None:
        chunk_parallel_job_count = int(chunk_parallel_job_count)
    if chunk_parallel_job_index is not None:
        chunk_parallel_job_index = int(chunk_parallel_job_index)
    if analysis_stage not in {'all', 'whitelight', 'prep', 'highres'}:
        raise ValueError(
            "analysis_stage must be one of "
            "{'all', 'whitelight', 'prep', 'highres'}. "
            f"Received '{analysis_stage}'."
        )
    if chunk_mode not in {'serial', 'parallel', 'combine'}:
        raise ValueError(
            "flags.chunk_mode must be one of {'serial', 'parallel', 'combine'}. "
            f"Received '{chunk_mode}'."
        )
    spectro_gradient_diagnostic = str(
        flags.get('spectro_gradient_diagnostic', 'first')
    ).lower()
    if spectro_gradient_diagnostic not in {'off', 'first', 'each'}:
        raise ValueError(
            "flags.spectro_gradient_diagnostic must be one of "
            "{'off', 'first', 'each'}."
        )
    spectro_gradient_diagnostic_strict = bool(
        flags.get('spectro_gradient_diagnostic_strict', False)
    )
    trend_inference = str(
        flags.get('trend_inference', 'sampled_uniform')
    ).lower()
    if trend_inference not in {'sampled_uniform', 'gaussian_marginalized'}:
        raise ValueError(
            "flags.trend_inference must be one of "
            "{'sampled_uniform', 'gaussian_marginalized'}."
        )
    if trend_inference == 'gaussian_marginalized' and transit_engine != 'jaxoplanet':
        raise ValueError(
            "Gaussian trend marginalization currently supports only the "
            "Jaxoplanet spectroscopic model."
        )
    if trend_inference == 'gaussian_marginalized' and detrending_type == 'none':
        raise ValueError(
            "flags.trend_inference='gaussian_marginalized' requires at "
            "least one conditionally linear trend coefficient."
        )
    whitelight_mcmc_kwargs = _resolve_stage_mcmc_kwargs(flags, 'whitelight')
    whitelight_laplace_options = _resolve_whitelight_laplace_options(flags)
    lowres_mcmc_kwargs = _resolve_stage_mcmc_kwargs(flags, 'lowres')
    highres_mcmc_kwargs = _resolve_stage_mcmc_kwargs(flags, 'highres')
    lowres_laplace_is_kwargs = _resolve_laplace_is_stage_kwargs(
        flags, 'lowres'
    )
    highres_laplace_is_kwargs = _resolve_laplace_is_stage_kwargs(
        flags, 'highres'
    )
    ld_profile = flags.get('ld_profile', 'quadratic')
    ld_prior_mode = _resolve_ld_prior_mode(flags, stellar_cfg, ld_profile)
    if hr_custom_ld_path and ld_profile != 'power2':
        raise ValueError("flags.hr_custom_ld_path currently supports only flags.ld_profile: 'power2'.")
    if transit_engine == 'harmonica' and ld_profile not in {'power2', 'quadratic'}:
        raise ValueError(
            "The harmonica engine supports flags.ld_profile in "
            "{'power2', 'quadratic'}. "
            f"Received '{ld_profile}'."
        )
    if (
        transit_engine == 'harmonica'
        and ld_profile == 'quadratic'
        and ld_prior_mode not in {'fixed', 'sing'}
    ):
        raise ValueError(
            "Harmonica quadratic limb darkening supports fixed direct u1/u2 "
            "coefficients or flags.ld_prior: 'sing'."
        )
    if ld_prior_mode == 'uniform' and ld_profile != 'power2':
        print(
            "[LD prior] flags.ld_prior='uniform' was requested for a non-power2 profile; "
            "the underlying model will sample the free LD coefficients uniformly within its native bounds.",
            flush=True,
        )
    max_harmonic_order = int(flags.get('harmonica_max_order', 1))
    harmonica_spectro_parameterization = (
        _validate_harmonica_spectro_parameterization(
            flags.get('harmonica_spectro_parameterization', 'delta_r'),
            max_harmonic_order,
        )
    )
    harmonica_spectro_fit_jitter = bool(
        flags.get('harmonica_spectro_fit_jitter', True)
    )
    harmonica_spectro_odd_frac_sigma = float(
        flags.get('harmonica_spectro_odd_frac_sigma', 0.1)
    )
    if (
        not np.isfinite(harmonica_spectro_odd_frac_sigma)
        or harmonica_spectro_odd_frac_sigma <= 0
    ):
        raise ValueError(
            "flags.harmonica_spectro_odd_frac_sigma must be finite and > 0."
        )
    harmonica_wl_nuts_kwargs = _resolve_harmonica_stage_nuts_kwargs(
        flags, 'harmonica_wl', default_dense_mass=True
    )
    harmonica_lr_nuts_kwargs = _resolve_harmonica_stage_nuts_kwargs(
        flags, 'harmonica_lr', default_dense_mass=True
    )
    harmonica_hr_nuts_kwargs = _resolve_harmonica_stage_nuts_kwargs(
        flags, 'harmonica_hr', default_dense_mass=False
    )

    # Engine dispatch: import the right model builders, NUTS settings, and geometry deriver.
    # Preserve compatibility with older YAMLs that still use the harmonica-specific
    # flag name and previously defaulted to the a_rs white-light parameterization.
    legacy_param_method = flags.get('harmonica_wl_parameterization')
    explicit_param_method = flags.get('param_method')
    if transit_engine == 'harmonica':
        if explicit_param_method is not None:
            param_method = explicit_param_method
        elif legacy_param_method is not None:
            param_method = legacy_param_method
        else:
            param_method = 'a_rs'
    else:
        param_method = explicit_param_method if explicit_param_method is not None else 'duration'
    if jaxoplanet_kernel == 'native_power2':
        if transit_engine != 'jaxoplanet':
            raise ValueError(
                "flags.jaxoplanet_kernel='native_power2' requires "
                "flags.transit_engine='jaxoplanet'."
            )
        if ld_profile != 'power2':
            raise ValueError(
                "flags.jaxoplanet_kernel='native_power2' requires "
                "flags.ld_profile='power2'."
            )
        if param_method != 'duration':
            raise ValueError(
                "flags.jaxoplanet_kernel='native_power2' supports only "
                "flags.param_method='duration'; a_rs geometry is unsupported."
            )
        if interpolate_ld:
            raise ValueError(
                "flags.jaxoplanet_kernel='native_power2' does not support "
                "flags.interpolate_ld: true because that path supplies "
                "interpolated polynomial rather than direct Power-2 coefficients."
            )
    if transit_engine == 'harmonica':
        from models.harmonica import create_whitelight_model, create_vectorized_model, NUTS_KWARGS, derive_geometry
        _engine_wl_kw = {
            'max_harmonic_order': max_harmonic_order,
            'param_method': param_method,
            'ld_profile': ld_profile,
        }
        _engine_spectro_kw = {
            'max_harmonic_order': max_harmonic_order,
            'odd_parameterization': harmonica_spectro_parameterization,
            'fit_jitter': harmonica_spectro_fit_jitter,
            'odd_frac_sigma': harmonica_spectro_odd_frac_sigma,
            'ld_profile': ld_profile,
        }
    else:
        from models.jaxoplanet import (
            NUTS_KWARGS,
            build_transit_window_indices,
            create_vectorized_model,
            create_whitelight_model,
            derive_geometry,
        )
        _engine_wl_kw = {
            'ld_profile': ld_profile,
            'param_method': param_method,
            'jaxoplanet_kernel': jaxoplanet_kernel,
            'ld_parameterization': whitelight_ld_parameterization,
        }
        _engine_spectro_kw = {
            'jitter_prior': spectro_jitter_prior,
            'jitter_prior_scale': spectro_jitter_prior_scale,
            'jitter_prior_center': spectro_jitter_prior_center,
            'ld_profile': ld_profile,
            'param_method': param_method,
            'transit_window': transit_window_optimization,
            'jaxoplanet_kernel': jaxoplanet_kernel,
            'ld_parameterization': spectro_ld_parameterization,
        }
        if trend_inference == 'gaussian_marginalized' and 'gp' in detrending_type:
            raise ValueError(
                "flags.trend_inference='gaussian_marginalized' does not yet "
                "support GP detrending."
            )
        independent_backend = spectro_sampler in {
            'independent_nuts', 'independent_hmc', 'laplace_is'
        }
        hmc_backend = spectro_sampler == 'independent_hmc'
        jaxoplanet_lr_nuts_kwargs = _resolve_jaxoplanet_spectro_nuts_kwargs(
            flags,
            'lowres',
            NUTS_KWARGS,
            independent=independent_backend,
            hmc=hmc_backend,
        )
        jaxoplanet_hr_nuts_kwargs = _resolve_jaxoplanet_spectro_nuts_kwargs(
            flags,
            'highres',
            NUTS_KWARGS,
            independent=independent_backend,
            hmc=hmc_backend,
        )

    HARMONICA_ODD_HARMONICS = tuple(
        name for name, _ in harmonica_odd_coeff_specs(max_harmonic_order)
    )

    host_device = cfg.get('host_device', 'gpu').lower()
    if (
        os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
        and os.getenv("FIT_JWST_DUMP_FORCE_CPU", "0") == "1"
    ):
        host_device = "cpu"
        print("Sampler-input dump requested a CPU runtime override.")
    # The old C++ FFI custom calls are CPU-only. N_c<=1 power-2 and standard
    # direct-u1/u2 quadratic transits use the pure-JAX path on GPU.
    if transit_engine == 'harmonica' and host_device != 'cpu':
        if max_harmonic_order > 1:
            print(
                f"Harmonica N_c>{1} requires CPU-only FFI custom calls; "
                f"overriding host_device='{host_device}' to 'cpu'."
            )
            host_device = 'cpu'
        else:
            print(
                f"Using pure JAX harmonica GPU path "
                f"(N_c=1 {ld_profile} LD)."
            )
    jax.config.update('jax_platform_name', host_device)
    numpyro.set_platform(host_device)
    actual_backend = jax.default_backend()
    if actual_backend != host_device:
        raise RuntimeError(
            f"Requested JAX backend '{host_device}', but JAX selected "
            f"'{actual_backend}'. Available devices: {jax.devices()}"
        )
    print(f"JAX backend verified: {actual_backend}; devices={jax.devices()}")
    master_seed = int(os.getenv("FIT_JWST_SEED", flags.get("random_seed", 555)))
    key_master = jax.random.PRNGKey(master_seed)
    print(f"Master random seed: {master_seed}")

    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    periods = jnp.atleast_1d(planet_cfg['period'])
    n_planets = len(periods)
    if transit_engine == 'harmonica' and n_planets != 1:
        raise NotImplementedError(
            "Harmonica production fitting and limb-product export currently "
            "support exactly one planet; refusing to silently export only "
            "planet_index=0."
        )
    if 'duration' in planet_cfg:
        durations = jnp.atleast_1d(planet_cfg['duration'])
    elif 'a_rs' in planet_cfg:
        _a_rs_tmp = jnp.atleast_1d(jnp.asarray(planet_cfg['a_rs'], dtype=jnp.float64))
        _b_tmp = jnp.atleast_1d(jnp.asarray(planet_cfg['b'], dtype=jnp.float64))
        _rors_tmp = jnp.atleast_1d(jnp.asarray(planet_cfg['rprs'], dtype=jnp.float64))
        _ecc_tmp = jnp.atleast_1d(jnp.asarray(planet_cfg.get('ecc', 0.0), dtype=jnp.float64))
        _omega_tmp = jnp.atleast_1d(jnp.asarray(planet_cfg.get('omega', 0.0), dtype=jnp.float64))
        durations = harmonica_duration_from_geometry(
            periods, _a_rs_tmp, _b_tmp, _rors_tmp, ecc=_ecc_tmp, omega=_omega_tmp,
        )
        print(f"Computed duration prior from a_rs geometry: {durations}")
    else:
        raise KeyError(
            "'planet.duration' is required unless 'planet.a_rs' is provided."
        )
    t0s = jnp.atleast_1d(planet_cfg['t0'])
    bs = jnp.atleast_1d(planet_cfg['b'])
    rors = jnp.atleast_1d(planet_cfg['rprs'])
    depths = rors**2

    PERIOD_FIXED = periods
    PRIOR_DUR = durations
    PRIOR_T0 = t0s
    PRIOR_B = bs
    PRIOR_RPRS = rors
    PRIOR_DEPTH = depths

    def _planet_cfg_array(key, default):
        arr = np.atleast_1d(np.asarray(planet_cfg.get(key, default), dtype=float))
        if arr.size == 1 and n_planets > 1:
            arr = np.repeat(arr, n_planets)
        elif arr.size != n_planets:
            raise ValueError(
                f"`planet.{key}` must be scalar or length {n_planets}, got shape {arr.shape}."
            )
        return jnp.asarray(arr, dtype=jnp.float64)

    # Shared orbital geometry inputs used by harmonica and jaxoplanet a_rs parameterizations.
    HARMONICA_ECC = _planet_cfg_array('ecc', 0.0)
    HARMONICA_OMEGA = _planet_cfg_array('omega', 0.0)
    if 'a_rs' in planet_cfg:
        HARMONICA_A_RS = _planet_cfg_array('a_rs', None)
    else:
        HARMONICA_A_RS = harmonica_a_rs_from_duration(
            periods, PRIOR_DUR, PRIOR_B, PRIOR_RPRS,
            ecc=HARMONICA_ECC, omega=HARMONICA_OMEGA,
        )
    HARMONICA_COS_I = _harmonica_cosi_from_b(PRIOR_B, HARMONICA_A_RS, HARMONICA_ECC, HARMONICA_OMEGA)
    HARMONICA_A_RS_PRIOR_MIN = _planet_cfg_array(
        'a_rs_prior_min', np.maximum(2.0, 0.5 * np.asarray(HARMONICA_A_RS))
    )
    HARMONICA_A_RS_PRIOR_MAX = _planet_cfg_array(
        'a_rs_prior_max', np.maximum(10.0, 2.0 * np.asarray(HARMONICA_A_RS))
    )
    HARMONICA_INC = jnp.arccos(HARMONICA_COS_I)
    HARMONICA_INC_PRIOR_MIN = _planet_cfg_array('inc_prior_min', 0.0)
    HARMONICA_INC_PRIOR_MAX = _planet_cfg_array('inc_prior_max', np.pi / 2.0)
    if jnp.any(HARMONICA_A_RS_PRIOR_MIN <= 0.0):
        raise ValueError("`planet.a_rs_prior_min` must be > 0.")
    if jnp.any(HARMONICA_A_RS_PRIOR_MAX <= HARMONICA_A_RS_PRIOR_MIN):
        raise ValueError("`planet.a_rs_prior_max` must exceed `planet.a_rs_prior_min`.")
    if jnp.any((HARMONICA_ECC < 0.0) | (HARMONICA_ECC >= 1.0)):
        raise ValueError("`planet.ecc` must satisfy 0 <= ecc < 1.")

    stellar_feh = stellar_cfg['feh']
    stellar_teff = stellar_cfg['teff']
    stellar_logg = stellar_cfg['logg']
    ld_model = stellar_cfg.get('ld_model', 'mps1')
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')
    ld_interpolate_type = stellar_cfg.get('ld_interpolate_type', 'trilinear')
    sld = StellarLimbDarkening(
        M_H=stellar_feh, Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
        ld_data_path=ld_data_path,
        interpolate_type=ld_interpolate_type,
    )

    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        mini_instrument = f'nrs{nrs}'
    elif instrument == 'NIRISS/SOSS':
        mini_instrument = f'order{order}'
    else:
        mini_instrument = ''

    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    if bins == resolution:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}LR_{high_resolution_bins}HR.pkl'
    elif bins == pixels:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}pix_{high_resolution_bins}pix.pkl'

    lr_bin_str = f'R{low_resolution_bins}' if bins == resolution else f'pix{low_resolution_bins}'
    hr_bin_str = f'R{high_resolution_bins}' if bins == resolution else f'pix{high_resolution_bins}'
    lr_artifact_stem = _harmonica_spectro_artifact_stem(
        f"{instrument_full_str}_{lr_bin_str}",
        transit_engine,
        harmonica_spectro_parameterization,
    )
    hr_artifact_stem = _harmonica_spectro_artifact_stem(
        f"{instrument_full_str}_{hr_bin_str}",
        transit_engine,
        harmonica_spectro_parameterization,
    )

    reference_grid_identities = {}
    for section_name, section in (("resolution", resolution), ("pixels", pixels)):
        if not isinstance(section, dict):
            continue
        for key_name in ("reference_grid", "reference_grid_lr"):
            if section.get(key_name) is not None:
                reference_grid_identities[f"{section_name}.{key_name}"] = (
                    _optional_file_content_identity(section[key_name])
                )
    spectro_data_manifest = f"{spectro_data_file}.manifest.json"
    spectro_data_fingerprint = _science_artifact_fingerprint(
        "spectro_data",
        {
            "target_revision": SPECTRO_DATA_TARGET_REVISION,
            "source_fits": _file_content_identity(fits_file),
            "instrument": instrument,
            "order": cfg.get("order"),
            "nrs": cfg.get("nrs"),
            "resolution": resolution,
            "pixels": pixels,
            "planet_t0": planet_cfg.get("t0"),
            "planet_duration": planet_cfg.get("duration"),
            "wavelength_filter": cfg.get("wavelength_filter", {}),
            "wavelength_masks": cfg.get("wavelength_masks"),
            "mask_start": mask_start,
            "mask_end": mask_end,
            "mask_integrations_start": mask_integrations_start,
            "mask_integrations_end": mask_integrations_end,
            "reference_grids": reference_grid_identities,
        },
    )
    reuse_spectro_data = bool(
        os.path.exists(spectro_data_file)
        and _science_artifact_manifest_matches(
            spectro_data_manifest, spectro_data_fingerprint
        )
    )
    data = None
    if reuse_spectro_data:
        try:
            data = SpectroData.load(spectro_data_file)
            print("Reusing fingerprinted spectroscopy data cache.")
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError):
            print("Spectroscopy data cache is unreadable; rebuilding atomically.")
    if data is None:
        data = process_spectroscopy_data(
            instrument,
            input_dir,
            output_dir,
            planet_str,
            cfg,
            fits_file,
            mask_start,
            mask_end,
            mask_integrations_start,
            mask_integrations_end,
        )
        temporary_data = (
            f"{spectro_data_file}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        data.save(temporary_data)
        os.replace(temporary_data, spectro_data_file)
        _write_science_artifact_manifest(
            spectro_data_manifest, "spectro_data", spectro_data_fingerprint
        )
    
    print("Data loaded.")
    print(f"Data length: {data.time.shape}")
    
    timebin_cfg = cfg.get("time_binning", {})
    do_timebin = timebin_cfg.get("enabled", False) or flags.get("bin_time", False)

    if do_timebin:
        dt_seconds = (
            timebin_cfg.get("dt_seconds", None)
            or flags.get("bin_dt_seconds", None)
            or 120.0
        )
        method = (
            timebin_cfg.get("method", None)
            or flags.get("bin_method", None)
            or "weighted"
        )
        bin_whitelight = timebin_cfg.get("whitelight", True) if "whitelight" in timebin_cfg else flags.get("bin_whitelight", True)
        bin_spectroscopic = timebin_cfg.get("spectroscopic", True) if "spectroscopic" in timebin_cfg else flags.get("bin_spectroscopic", True)

        data = bin_spectrodata_in_time(
            data,
            dt_seconds=float(dt_seconds),
            method=str(method),
            bin_whitelight=bool(bin_whitelight),
            bin_spectroscopic=bool(bin_spectroscopic)
        )
        print('Finished Binning! Binned data length: ', data.time.shape)



    if ld_prior_mode == 'informed':
        U_mu_wl, U_sigma_wl = get_or_build_power2_ld_prior(
            stellar_cfg, data.wavelengths_unbinned, 0.0, instrument,
            order=order if instrument == 'NIRISS/SOSS' else None,
            output_dir=output_dir,
            cache_label='whitelight'
        )
    else:
        if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'MIRI/LRS', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
            U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned,0.0, instrument, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
        elif instrument == 'NIRISS/SOSS':
            U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
        U_sigma_wl = None

    wl_mask_path = f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy'
    wl_params_path = f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv'
    wl_gp_path = f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv'
    wl_limb_csv_path = f"{output_dir}/{instrument_full_str}_whitelight_limb_spectra.csv"
    wl_limb_samples_path = f"{output_dir}/{instrument_full_str}_whitelight_limb_posterior_samples.npz"
    wl_manifest_path = f'{output_dir}/{instrument_full_str}_whitelight_artifacts.manifest.json'
    wl_geometry_handoff_path = (
        f'{output_dir}/{instrument_full_str}_whitelight_geometry_handoff.json'
    )
    wl_artifact_fingerprint = _science_artifact_fingerprint(
        "whitelight",
        {
            "config": cfg,
            "time": data.wl_time,
            "flux": data.wl_flux,
            "flux_err": data.wl_flux_err,
            "wavelengths_unbinned": data.wavelengths_unbinned,
            "transit_engine": transit_engine,
            "ld_profile": ld_profile,
            "ld_prior_mode": ld_prior_mode,
            "detrending_type": detrending_type,
            "param_method": param_method,
            "geometry_estimator": whitelight_geometry_estimator,
            "whitelight_sigma": whitelight_sigma,
            "ld_coefficients": U_mu_wl,
            "ld_uncertainties": U_sigma_wl,
        },
    )
    wl_geometry_handoff = _load_whitelight_geometry_handoff(
        wl_geometry_handoff_path,
        expected_posterior_fingerprint=wl_artifact_fingerprint,
        expected_estimator=whitelight_geometry_estimator,
    )
    required_wl_limb_products_exist = bool(
        transit_engine != 'harmonica'
        or not HARMONICA_ODD_HARMONICS
        or (
            os.path.exists(wl_limb_csv_path)
            and os.path.exists(wl_limb_samples_path)
        )
    )
    stringcheck = bool(
        os.path.exists(wl_mask_path)
        and os.path.exists(wl_params_path)
        and ('gp' not in detrending_type or os.path.exists(wl_gp_path))
        and required_wl_limb_products_exist
        and wl_geometry_handoff is not None
        and _science_artifact_manifest_matches(
            wl_manifest_path, wl_artifact_fingerprint
        )
    )
    if not stringcheck and (
        os.path.exists(wl_mask_path) or os.path.exists(wl_params_path)
    ):
        print(
            "Existing white-light products or geometry handoff are unverified "
            "or do not match the current data/configuration; recomputing them."
        )

    if not stringcheck or ('gp' in detrending_type):
        if not stringcheck or not os.path.exists(wl_gp_path):
            plt.scatter(data.wl_time, data.wl_flux, c='k', s=6, alpha=0.5)
            plt.savefig(f'{output_dir}/stuff.png')
            plt.close()
            print('Fitting whitelight for outliers and bestfit parameters')
            hyper_params_wl = {
                "duration": PRIOR_DUR,
                "t0": PRIOR_T0,
                'period': PERIOD_FIXED,
                'u': U_mu_wl,
                'a_rs': HARMONICA_A_RS,
                'a_rs_prior_min': HARMONICA_A_RS_PRIOR_MIN,
                'a_rs_prior_max': HARMONICA_A_RS_PRIOR_MAX,
                'inc_prior_min': HARMONICA_INC_PRIOR_MIN,
                'inc_prior_max': HARMONICA_INC_PRIOR_MAX,
                'ecc': HARMONICA_ECC,
                'omega': HARMONICA_OMEGA,
            }
            if '2spot' in detrending_type:
                hyper_params_wl['spot_guess'] = spot_mu
                hyper_params_wl['spot_guess2'] = spot_mu2
            elif 'spot' in detrending_type:
                hyper_params_wl['spot_guess'] = spot_mu
            if 'linear_discontinuity' in detrending_type:
                hyper_params_wl['t_jump_guess'] = t_jump_guess if t_jump_guess is not None else 0.5 * (jnp.min(data.wl_time) + jnp.max(data.wl_time))
                hyper_params_wl['jump_guess'] = jump_guess

            hyper_params_wl['u'] = U_mu_wl
            if ld_profile == 'power2' and ld_prior_mode == 'informed' and U_sigma_wl is not None:
                hyper_params_wl['u_sigma'] = U_sigma_wl

            init_params_wl = {
                'c': 1.0,
                'v': 0.0,
                'log_jitter': jnp.log(1e-4),
                'b': PRIOR_B,
                'rors': PRIOR_RPRS
            }
            init_params_wl['u'] = U_mu_wl
            # Power-2 models sample the native (c, alpha) coefficients at the
            # latent sites ``c1`` and ``c2``.  Supplying only ``u`` silently
            # falls back to NumPyro's default initialization and defeats the
            # stellar-informed starting point for both transit engines.
            if ld_profile == 'power2' and ld_prior_mode != 'fixed':
                init_params_wl['c1'] = U_mu_wl[0]
                init_params_wl['c2'] = U_mu_wl[1]
            if transit_engine == 'harmonica':
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    init_params_wl[_harmonica_frac_site(harmonic_name)] = _harmonica_coeff_to_frac(
                        HARMONICA_INIT_ODD_COEFF,
                        PRIOR_RPRS,
                    )

            for i in range(n_planets):
                if param_method == 'a_rs':
                    init_params_wl[f'log_a_rs_{i}'] = jnp.log(HARMONICA_A_RS[i])
                    init_params_wl[f'_b_{i}'] = PRIOR_B[i]
                else:
                    init_params_wl[f'logD_{i}'] = jnp.log(PRIOR_DUR[i])
                    init_params_wl[f'_b_{i}'] = PRIOR_B[i]
                init_params_wl[f't0_{i}'] = PRIOR_T0[i]
                init_params_wl[f'rors_{i}'] = PRIOR_RPRS[i]

            if 'quadratic' in detrending_type:
                init_params_wl['v2'] = 0.0
            if 'cubic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0
            if 'quartic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0; init_params_wl['v4'] = 0.0
            if 'explinear' in detrending_type:
                init_params_wl['A'] = 0.001
                # ``tau`` is deterministic; ``log_tau`` is the actual latent
                # site and its prior spans 1e-3--1e-1 days.
                init_params_wl['log_tau'] = jnp.log(1e-2)
            if 'gp' in detrending_type:
                init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_rho'] = jnp.log(0.1)
            if 'linear_discontinuity' in detrending_type:
                if t_jump_guess is not None:
                    init_params_wl['t_jump'] = t_jump_guess
                else:
                    init_params_wl['t_jump'] = 0.5 * (jnp.min(data.wl_time) + jnp.max(data.wl_time))
                init_params_wl['jump'] = jump_guess
            if 'spot' in detrending_type:
                init_params_wl['spot_amp'] = spot_amp
                init_params_wl['spot_mu'] = spot_mu
                init_params_wl['spot_sigma'] = spot_sigma
            if '2spot' in detrending_type:
                init_params_wl['spot_amp2'] = spot_amp2
                init_params_wl['spot_mu2'] = spot_mu2
                init_params_wl['spot_sigma2'] = spot_sigma2

            # Step 1 of the Sing recipe measures LD freely.  The resulting
            # artifact is deliberately consumed only by the spectroscopic fit.
            wl_ld_mode = 'uniform' if ld_prior_mode == 'sing' else ld_prior_mode
            whitelight_model_for_run = create_whitelight_model(
                detrend_type=detrending_type,
                n_planets=n_planets,
                ld_mode=wl_ld_mode,
                **_engine_wl_kw,
            )

            def _get_preopt_sanity_model(params, t_vals):
                if 'gp' in detrending_type:
                    return compute_lc_linear(params, t_vals)
                try:
                    return resolve_detrend_kernel(detrending_type)(params, t_vals)
                except KeyError:
                    return compute_lc_linear(params, t_vals)

            def _build_preopt_physical_params(init_params, n_planets_eval):
                params_eval = hyper_params_wl.copy()
                params_eval.update(init_params)

                def _ensure_len(value, default):
                    arr = jnp.atleast_1d(jnp.asarray(params_eval.get(value, default)))
                    if arr.size == 1 and n_planets_eval > 1:
                        return jnp.repeat(arr, n_planets_eval)
                    return arr

                params_eval["period"] = _ensure_len("period", PERIOD_FIXED)
                params_eval["t0"] = _ensure_len("t0", PRIOR_T0)
                params_eval["b"] = _ensure_len("b", PRIOR_B)
                params_eval["rors"] = _ensure_len("rors", PRIOR_RPRS)

                if all(f"t0_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["t0"] = jnp.array([init_params[f"t0_{i}"] for i in range(n_planets_eval)])
                if all(f"_b_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["b"] = jnp.array([jnp.abs(init_params[f"_b_{i}"]) for i in range(n_planets_eval)])
                if all(f"rors_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["rors"] = jnp.array([init_params[f"rors_{i}"] for i in range(n_planets_eval)])
                elif all(f"depths_{i}" in init_params for i in range(n_planets_eval)):
                    params_eval["rors"] = jnp.array([jnp.sqrt(init_params[f"depths_{i}"]) for i in range(n_planets_eval)])

                if transit_engine == 'harmonica':
                    params_eval["a_rs"] = _ensure_len("a_rs", HARMONICA_A_RS)
                    if all(f"log_a_rs_{i}" in init_params for i in range(n_planets_eval)):
                        params_eval["a_rs"] = jnp.array([jnp.exp(init_params[f"log_a_rs_{i}"]) for i in range(n_planets_eval)])
                    elif all(f"logD_{i}" in init_params for i in range(n_planets_eval)):
                        params_eval["duration"] = jnp.array([jnp.exp(init_params[f"logD_{i}"]) for i in range(n_planets_eval)])
                        params_eval["a_rs"] = harmonica_a_rs_from_duration(
                            params_eval["period"],
                            params_eval["duration"],
                            params_eval["b"],
                            params_eval["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    # Compatibility fallback for older direct-inc initializations.
                    if all(f"inc_{i}" in init_params for i in range(n_planets_eval)):
                        incs = jnp.array([init_params[f"inc_{i}"] for i in range(n_planets_eval)])
                        params_eval["inc"] = incs
                        params_eval["cos_i"] = jnp.cos(incs)
                        params_eval["b"] = harmonica_impact_param_from_cos_i(
                            params_eval["cos_i"], params_eval["a_rs"],
                            ecc=hyper_params_wl['ecc'], omega=hyper_params_wl['omega'],
                        )
                        params_eval["duration"] = harmonica_duration_from_cos_i(
                            params_eval["period"],
                            params_eval["a_rs"],
                            params_eval["cos_i"],
                            params_eval["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    else:
                        params_eval["duration"] = harmonica_duration_from_geometry(
                            params_eval["period"],
                            params_eval["a_rs"],
                            params_eval["b"],
                            params_eval["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                        params_eval["cos_i"] = _harmonica_cosi_from_b(
                            params_eval["b"], params_eval["a_rs"],
                            hyper_params_wl['ecc'], hyper_params_wl['omega'],
                        )
                    if ld_profile == "power2":
                        if "c1" in init_params:
                            params_eval["c_ld"] = init_params["c1"]
                        if "c2" in init_params:
                            params_eval["alpha_ld"] = init_params["c2"]
                    else:
                        params_eval["u1_ld"] = U_mu_wl[0]
                        params_eval["u2_ld"] = U_mu_wl[1]
                else:
                    params_eval["duration"] = _ensure_len("duration", PRIOR_DUR)
                    if all(f"logD_{i}" in init_params for i in range(n_planets_eval)):
                        params_eval["duration"] = jnp.array([jnp.exp(init_params[f"logD_{i}"]) for i in range(n_planets_eval)])
                    if jaxoplanet_kernel == "native_power2":
                        params_eval.pop("a_rs", None)
                        params_eval.pop("ecc", None)
                        params_eval.pop("omega", None)
                        params_eval["c1"] = init_params.get("c1", U_mu_wl[0])
                        params_eval["c2"] = init_params.get("c2", U_mu_wl[1])
                        params_eval["_jaxoplanet_kernel"] = "native_power2"
                        params_eval["_ld_profile"] = "power2"

                return params_eval

            try:
                params_preopt = _build_preopt_physical_params(init_params_wl, n_planets)
                flux_preopt = _get_preopt_sanity_model(params_preopt, data.wl_time)
                finite_preopt = bool(jnp.all(jnp.isfinite(flux_preopt)))
                print(
                    "Pre-opt init model check:",
                    {
                        "finite": finite_preopt,
                        "flux_min": float(jnp.nanmin(flux_preopt)),
                        "flux_max": float(jnp.nanmax(flux_preopt)),
                    },
                )

                fig, axes = plt.subplots(
                    2, 1, figsize=(10, 7), sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                axes[0].scatter(data.wl_time, data.wl_flux, c='k', s=10, alpha=0.3, label='Data')
                axes[0].plot(data.wl_time, flux_preopt, color='tab:red', lw=2.0, label='Init Model')
                axes[0].set_ylabel('Flux')
                axes[0].legend()
                axes[0].set_title(f'Pre-Optimization Check: {detrending_type}')

                preopt_resid = data.wl_flux - flux_preopt
                axes[1].axhline(0.0, color='0.5', lw=1.0, ls='--')
                axes[1].scatter(data.wl_time, preopt_resid, c='tab:blue', s=8, alpha=0.4)
                axes[1].set_xlabel('Time (BJD)')
                axes[1].set_ylabel('Resid.')

                plt.tight_layout()
                preopt_plot_path = f'{output_dir}/00_{instrument_full_str}_preopt_init_check.png'
                plt.savefig(preopt_plot_path)
                plt.close(fig)
                print(f"Saved pre-optimization sanity check plot to: {preopt_plot_path}")
            except Exception as e:
                print(f"Could not create pre-optimization sanity plot: {e}")
            
            if 'gp' in detrending_type:
                print("--- Running Pre-Fit with Linear Detrending to stabilize GP ---")
                whitelight_model_prefit = create_whitelight_model(
                    detrend_type='linear',
                    n_planets=n_planets,
                    ld_mode=wl_ld_mode,
                    **_engine_wl_kw,
                )
                init_params_prefit = init_params_wl.copy()
                init_params_prefit.pop('GP_log_sigma', None)
                init_params_prefit.pop('GP_log_rho', None)
                soln = optimx.optimize(whitelight_model_prefit, start=init_params_prefit)(
                    key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl
                )
            else:
                soln =  optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            
            
            print("Plotting Initial vs. Optimized Model sanity check...")
            
            def _get_sanity_model(params, t_vals):
                if 'gp' in detrending_type:
                    return compute_lc_linear(params, t_vals)
                try:
                    return resolve_detrend_kernel(detrending_type)(params, t_vals)
                except KeyError:
                    return compute_lc_linear(params, t_vals)
            def _soln_to_physical_params(soln, base_params, n_planets=1):
                p = dict(base_params)
                for key, value in soln.items():
                    if key != "obs":
                        p[key] = value

                def have_all(prefix):
                    return all(f"{prefix}_{i}" in p for i in range(n_planets))

                have_log_a_rs = have_all("log_a_rs")
                have_logD = have_all("logD")

                if have_all("t0"):
                    p["t0"] = jnp.array([p[f"t0_{i}"] for i in range(n_planets)])
                if have_all("b"):
                    p["b"] = jnp.array([p[f"b_{i}"] for i in range(n_planets)])
                elif have_all("_b"):
                    p["b"] = jnp.array([jnp.abs(p[f"_b_{i}"]) for i in range(n_planets)])
                if have_all("rors"):
                    p["rors"] = jnp.array([p[f"rors_{i}"] for i in range(n_planets)])
                elif have_all("depths"):
                    p["rors"] = jnp.array([jnp.sqrt(p[f"depths_{i}"]) for i in range(n_planets)])

                if transit_engine == 'harmonica':
                    if have_all("duration"):
                        p["duration"] = jnp.array([p[f"duration_{i}"] for i in range(n_planets)])
                    elif have_logD:
                        p["duration"] = jnp.array([jnp.exp(p[f"logD_{i}"]) for i in range(n_planets)])
                    if have_all("a_rs"):
                        p["a_rs"] = jnp.array([p[f"a_rs_{i}"] for i in range(n_planets)])
                    elif have_log_a_rs:
                        p["a_rs"] = jnp.array([jnp.exp(p[f"log_a_rs_{i}"]) for i in range(n_planets)])
                else:
                    if have_all("duration"):
                        p["duration"] = jnp.array([p[f"duration_{i}"] for i in range(n_planets)])
                    elif have_logD:
                        p["duration"] = jnp.array([jnp.exp(p[f"logD_{i}"]) for i in range(n_planets)])
                    if have_all("a_rs"):
                        p["a_rs"] = jnp.array([p[f"a_rs_{i}"] for i in range(n_planets)])
                    elif have_log_a_rs:
                        p["a_rs"] = jnp.array([jnp.exp(p[f"log_a_rs_{i}"]) for i in range(n_planets)])
                    if "a_rs" not in p and "duration" in p and "b" in p:
                        p["a_rs"] = harmonica_a_rs_from_duration(
                            p["period"],
                            p["duration"],
                            p["b"],
                            p["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    if "duration" not in p and "a_rs" in p and "b" in p:
                        p["duration"] = harmonica_duration_from_geometry(
                            p["period"],
                            p["a_rs"],
                            p["b"],
                            p["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    if "a_rs" in p and "b" in p:
                        p["cos_i"] = _harmonica_cosi_from_b(
                            p["b"], p["a_rs"], hyper_params_wl['ecc'], hyper_params_wl['omega']
                        )
                        p["inc"] = jnp.arccos(jnp.clip(p["cos_i"], 0.0, 1.0))
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    frac_key = _harmonica_frac_site(harmonic_name)
                    if harmonic_name not in p and frac_key in soln and "rors" in p:
                        p[harmonic_name] = soln[frac_key] * jnp.maximum(jnp.min(jnp.asarray(p["rors"])), 1e-6)
                if transit_engine == 'harmonica' and "rors" in p:
                    have_inc = have_all("inc")
                    if have_inc and "a_rs" in p and "b" not in p:
                        incs = jnp.array([p[f"inc_{i}"] for i in range(n_planets)])
                        p["inc"] = incs
                        p["cos_i"] = jnp.cos(incs)
                        p["b"] = harmonica_impact_param_from_cos_i(
                            p["cos_i"], p["a_rs"],
                            ecc=hyper_params_wl['ecc'], omega=hyper_params_wl['omega'],
                        )
                        p["duration"] = harmonica_duration_from_cos_i(
                            p["period"],
                            p["a_rs"],
                            p["cos_i"],
                            p["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    else:
                        if "duration" in p and "b" in p and (have_logD or "a_rs" not in p):
                            p["a_rs"] = harmonica_a_rs_from_duration(
                                p["period"],
                                p["duration"],
                                p["b"],
                                p["rors"],
                                ecc=hyper_params_wl['ecc'],
                                omega=hyper_params_wl['omega'],
                            )
                        if "a_rs" in p and "b" in p and (have_log_a_rs or "duration" not in p):
                            p["duration"] = harmonica_duration_from_geometry(
                                p["period"],
                                p["a_rs"],
                                p["b"],
                                p["rors"],
                                ecc=hyper_params_wl['ecc'],
                                omega=hyper_params_wl['omega'],
                            )
                        if "a_rs" in p and "b" in p:
                            p["cos_i"] = _harmonica_cosi_from_b(
                                p["b"], p["a_rs"], hyper_params_wl['ecc'], hyper_params_wl['omega']
                            )
                        if "duration" not in p and "cos_i" in p:
                            p["duration"] = harmonica_duration_from_cos_i(
                                p["period"],
                                p["a_rs"],
                                p["cos_i"],
                                p["rors"],
                                ecc=hyper_params_wl['ecc'],
                                omega=hyper_params_wl['omega'],
                            )
                    if ld_profile == "power2" and "c1" in p and "c2" in p:
                        p["c_ld"] = p["c1"]
                        p["alpha_ld"] = p["c2"]
                    elif ld_profile == "quadratic":
                        p["u1_ld"] = U_mu_wl[0]
                        p["u2_ld"] = U_mu_wl[1]

                return p

            try:

                params_complete = hyper_params_wl.copy()
                params_complete.update(init_params_wl)

                n_planets_sanity = len(jnp.atleast_1d(PERIOD_FIXED))

                def _ensure_len(value, n):
                    arr = jnp.atleast_1d(value)
                    if arr.size == 1 and n > 1:
                        return jnp.repeat(arr, n)
                    return arr

                params_complete["period"] = _ensure_len(params_complete.get("period", PERIOD_FIXED), n_planets_sanity)
                params_complete["duration"] = _ensure_len(params_complete.get("duration", PRIOR_DUR), n_planets_sanity)
                params_complete["t0"] = _ensure_len(params_complete.get("t0", PRIOR_T0), n_planets_sanity)
                params_complete["b"] = _ensure_len(params_complete.get("b", PRIOR_B), n_planets_sanity)
                params_complete["rors"] = _ensure_len(params_complete.get("rors", PRIOR_RPRS), n_planets_sanity)
                if transit_engine == 'harmonica':
                    if all(f'inc_{i}' in init_params_wl for i in range(n_planets_sanity)):
                        # Compatibility fallback for older direct-inc initializations.
                        params_complete["a_rs"] = jnp.array([jnp.exp(init_params_wl[f'log_a_rs_{i}']) for i in range(n_planets_sanity)])
                        incs = jnp.array([init_params_wl[f'inc_{i}'] for i in range(n_planets_sanity)])
                        params_complete["inc"] = incs
                        params_complete["cos_i"] = jnp.cos(incs)
                        params_complete["b"] = harmonica_impact_param_from_cos_i(
                            params_complete["cos_i"], params_complete["a_rs"],
                            ecc=hyper_params_wl['ecc'], omega=hyper_params_wl['omega'],
                        )
                        params_complete["duration"] = harmonica_duration_from_cos_i(
                            params_complete["period"],
                            params_complete["a_rs"],
                            params_complete["cos_i"],
                            params_complete["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                    elif all(f'log_a_rs_{i}' in init_params_wl for i in range(n_planets_sanity)):
                        params_complete["a_rs"] = jnp.array([jnp.exp(init_params_wl[f'log_a_rs_{i}']) for i in range(n_planets_sanity)])
                        params_complete["duration"] = harmonica_duration_from_geometry(
                            params_complete["period"],
                            params_complete["a_rs"],
                            params_complete["b"],
                            params_complete["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                        params_complete["cos_i"] = _harmonica_cosi_from_b(
                            params_complete["b"],
                            params_complete["a_rs"],
                            hyper_params_wl['ecc'],
                            hyper_params_wl['omega'],
                        )
                    else:
                        params_complete["a_rs"] = harmonica_a_rs_from_duration(
                            params_complete["period"],
                            params_complete["duration"],
                            params_complete["b"],
                            params_complete["rors"],
                            ecc=hyper_params_wl['ecc'],
                            omega=hyper_params_wl['omega'],
                        )
                        params_complete["cos_i"] = _harmonica_cosi_from_b(
                            params_complete["b"],
                            params_complete["a_rs"],
                            hyper_params_wl['ecc'],
                            hyper_params_wl['omega'],
                        )
                    if ld_profile == "power2":
                        if "c_ld" not in params_complete and "c1" in params_complete:
                            params_complete["c_ld"] = params_complete["c1"]
                        if "alpha_ld" not in params_complete and "c2" in params_complete:
                            params_complete["alpha_ld"] = params_complete["c2"]
                    else:
                        params_complete["u1_ld"] = U_mu_wl[0]
                        params_complete["u2_ld"] = U_mu_wl[1]
                elif all(f'log_a_rs_{i}' in init_params_wl for i in range(n_planets_sanity)):
                    params_complete["a_rs"] = jnp.array([jnp.exp(init_params_wl[f'log_a_rs_{i}']) for i in range(n_planets_sanity)])
                    params_complete["duration"] = harmonica_duration_from_geometry(
                        params_complete["period"],
                        params_complete["a_rs"],
                        params_complete["b"],
                        params_complete["rors"],
                        ecc=hyper_params_wl['ecc'],
                        omega=hyper_params_wl['omega'],
                    )
                    params_complete["cos_i"] = _harmonica_cosi_from_b(
                        params_complete["b"],
                        params_complete["a_rs"],
                        hyper_params_wl['ecc'],
                        hyper_params_wl['omega'],
                    )

                key_sanity = jax.random.PRNGKey(0) if key_master is None else key_master
                keys = jax.random.split(key_sanity, num=3)

                if n_planets_sanity == 1:
                    if param_method == 'a_rs':
                        _opt_sites = ["log_a_rs_0", "t0_0", "_b_0"]
                    else:
                        _opt_sites = ["logD_0", "t0_0", "_b_0"]
                else:
                    _opt_sites = None
                stage1 = optimx.optimize(
                    whitelight_model_for_run,
                    sites=_opt_sites,
                    start=init_params_wl,
                )
                soln = stage1(keys[0], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                stage2_sites = ["rors_0"]
                if wl_ld_mode != "fixed":
                    if transit_engine == 'harmonica':
                        stage2_sites.extend(["c1", "c2"])
                    elif ld_profile == "quadratic":
                        stage2_sites.append("u")
                    elif ld_profile == "power2":
                        stage2_sites.extend(["c1", "c2"])
                if n_planets_sanity != 1:
                    stage2_sites = None

                stage2 = optimx.optimize(
                    whitelight_model_for_run,
                    sites=stage2_sites,
                    start=soln,
                )
                soln = stage2(keys[1], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                stage3 = optimx.optimize(
                    whitelight_model_for_run,
                    start=soln,
                )
                soln = stage3(keys[2], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                params_opt = _soln_to_physical_params(soln, params_complete, n_planets=n_planets_sanity)
                if "rors" not in params_opt and "depths" in params_opt:
                    params_opt["rors"] = jnp.sqrt(params_opt["depths"])

                flux_init = _get_sanity_model(params_complete, data.wl_time)
                flux_opt  = _get_sanity_model(params_opt,      data.wl_time)
                print('Optimized Init Params:', params_opt)    
                plt.figure(figsize=(10, 6))
                plt.scatter(data.wl_time, data.wl_flux, c='k', s=10, alpha=0.3, label='Data')
                
                plt.plot(data.wl_time, flux_init, color='red', linestyle='--', lw=2, alpha=0.7, 
                         label='Init Guess')
                plt.plot(data.wl_time, flux_opt, color='blue', lw=2.5, 
                         label='Optimized (Gradient)')
                
                plt.title(f'Sanity Check: {detrending_type} (GP Pre-fit: {"gp" in detrending_type})')
                plt.xlabel('Time (BJD)')
                plt.ylabel('Flux')
                plt.legend()
                plt.tight_layout()
                
                sanity_plot_path = f'{output_dir}/00_{instrument_full_str}_init_vs_opt_check.png'
                plt.savefig(sanity_plot_path)
                plt.close()
                print(f"Saved sanity check plot to: {sanity_plot_path}")
                
            except Exception as e:
                print(f"Could not create sanity plot: {e}")

            # --- Gradient diagnostic (runs before NUTS) ---
            if bool(flags.get('pre_nuts_gradient_diagnostic', True)):
                try:
                    from diagnose_nuts_gradient import diagnose_gradient_quality
                    print("\n>>> Running pre-NUTS gradient diagnostic...")
                    diag_result = diagnose_gradient_quality(
                        whitelight_model_for_run, soln,
                        data.wl_time, data.wl_flux_err, data.wl_flux, hyper_params_wl,
                    )
                    diag_path = os.path.join(
                        output_dir, 'pre_nuts_gradient_diagnostics.json'
                    )
                    with open(diag_path, 'w', encoding='utf-8') as stream:
                        json.dump(diag_result, stream, indent=2, sort_keys=True)
                    print(
                        ">>> Gradient diagnostic "
                        f"{'PASSED' if diag_result['passed'] else 'WARNING'}: "
                        f"finite={diag_result['all_finite']}, "
                        "max directional relative error="
                        f"{diag_result['max_directional_relative_error']:.3e}."
                    )
                    print(f">>> Saved gradient diagnostic to {diag_path}\n")
                except Exception as e:
                    print(f">>> Gradient diagnostic failed: {e}\n")

            wl_nuts_kwargs = {
                "init_strategy": numpyro.infer.init_to_value(values=soln),
                **(harmonica_wl_nuts_kwargs if transit_engine == 'harmonica' else NUTS_KWARGS),
            }
            if transit_engine == 'jaxoplanet':
                wl_nuts_kwargs["dense_mass"] = bool(
                    flags.get("whitelight_dense_mass", wl_nuts_kwargs["dense_mass"])
                )
            laplace_preparation = None
            laplace_diagnostics = {}
            wl_run_init_params = None
            adaptive_wl_nuts_kwargs = dict(wl_nuts_kwargs)
            adaptive_wl_mcmc_kwargs = dict(whitelight_mcmc_kwargs)
            if (
                transit_engine == 'jaxoplanet'
                and whitelight_laplace_options["mass_matrix"] == "laplace"
            ):
                from models.independent_nuts import prepare_laplace_metric

                preparation_start = time.perf_counter()
                laplace_preparation = prepare_laplace_metric(
                    whitelight_model_for_run,
                    key_master,
                    soln,
                    data.wl_time,
                    data.wl_flux_err,
                    model_kwargs={
                        "y": data.wl_flux,
                        "prior_params": hyper_params_wl,
                    },
                    hessian_method=whitelight_laplace_options["hessian_method"],
                    trust_radius=whitelight_laplace_options["trust_radius"],
                    max_iterations=int(
                        flags.get("whitelight_laplace_map_iterations", 200)
                    ),
                    # White-light time/flux coordinates naturally span a
                    # wider curvature range than per-channel spectroscopy;
                    # the 1e-8 lane floor would truncate real geometry modes.
                    eigenvalue_floor=1.0e-12,
                )
                laplace_preparation.inverse_mass_matrix.block_until_ready()
                preparation_wall = time.perf_counter() - preparation_start
                wl_nuts_kwargs.update(
                    dense_mass=True,
                    inverse_mass_matrix=laplace_preparation.inverse_mass_matrix,
                    adapt_mass_matrix=False,
                    target_accept_prob=whitelight_laplace_options["target_accept"],
                    max_tree_depth=whitelight_laplace_options["max_tree_depth"],
                )
                whitelight_mcmc_kwargs = {
                    **whitelight_mcmc_kwargs,
                    "num_warmup": whitelight_laplace_options["warmup"],
                }
                wl_run_init_params = laplace_preparation.unconstrained_map
                laplace_diagnostics = {
                    "mass_matrix": "laplace",
                    "laplace_preparation_wall_seconds": preparation_wall,
                    "map_gradient_norm": float(laplace_preparation.gradient_norm),
                    "map_newton_decrement": float(laplace_preparation.newton_decrement),
                    "map_iterations": int(laplace_preparation.iterations),
                    "map_condition_number": float(laplace_preparation.condition_number),
                    "map_hessian_min_eigenvalue": float(
                        laplace_preparation.hessian_min_eigenvalue
                    ),
                    "laplace_hessian_method": whitelight_laplace_options[
                        "hessian_method"
                    ],
                }
            if (
                os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
                and os.getenv("FIT_JWST_DUMP_WHITELIGHT_MAX_TREE_DEPTH")
            ):
                wl_nuts_kwargs["max_tree_depth"] = int(
                    os.environ["FIT_JWST_DUMP_WHITELIGHT_MAX_TREE_DEPTH"]
                )
                print(
                    "Sampler-input dump white-light max_tree_depth override: "
                    f"{wl_nuts_kwargs['max_tree_depth']}"
                )
            if transit_engine == 'harmonica':
                print(
                    "Using harmonica white-light NUTS settings: "
                    f"dense_mass={wl_nuts_kwargs['dense_mass']}, "
                    f"regularize_mass_matrix={wl_nuts_kwargs['regularize_mass_matrix']}, "
                    f"max_tree_depth={wl_nuts_kwargs['max_tree_depth']}, "
                    f"target_accept_prob={wl_nuts_kwargs['target_accept_prob']}"
                )
            else:
                print(
                    "NUTS settings: "
                    f"dense_mass={wl_nuts_kwargs['dense_mass']}, "
                    f"regularize_mass_matrix={wl_nuts_kwargs['regularize_mass_matrix']}, "
                    f"target_accept_prob={wl_nuts_kwargs['target_accept_prob']}"
                )
            print(data.wavelengths_hr)
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(whitelight_model_for_run, **wl_nuts_kwargs),
                progress_bar=True,
                jit_model_args=True,
                **whitelight_mcmc_kwargs,
            )
            sampling_start = time.perf_counter()
            wl_run_args = (data.wl_time, data.wl_flux_err)
            wl_run_kwargs = {
                "y": data.wl_flux,
                "prior_params": hyper_params_wl,
                "init_params": wl_run_init_params,
                "extra_fields": (
                    "diverging", "accept_prob", "potential_energy", "num_steps"
                ),
            }
            mcmc.run(key_master, *wl_run_args, **wl_run_kwargs)
            jax.block_until_ready(next(iter(mcmc.get_samples().values())))
            laplace_diagnostics["sampling_wall_seconds"] = (
                time.perf_counter() - sampling_start
            )
            (
                wl_samples,
                wl_samples_grouped,
                wl_extra_fields,
                wl_quality_diagnostics,
            ) = _continue_mcmc_until_geometry_gate(
                mcmc,
                key_master,
                wl_run_args,
                wl_run_kwargs,
                min_ess=float(flags.get("whitelight_min_ess", 400)),
                max_divergences=int(flags.get("whitelight_max_divergences", 0)),
                max_extra_blocks=int(flags.get("whitelight_max_extra_blocks", 3)),
            )
            laplace_diagnostics.update(wl_quality_diagnostics)
            laplace_diagnostics.setdefault("mass_matrix", "adaptive")
            if (
                not wl_quality_diagnostics["whitelight_quality_gate_passed"]
                and laplace_diagnostics["mass_matrix"] == "laplace"
            ):
                print(
                    "WARNING: LAPLACE WHITE-LIGHT CHAIN FAILED THE QUALITY "
                    "GATE; AUTOMATICALLY FALLING BACK TO ADAPTIVE NUTS."
                )
                _, adaptive_key = jax.random.split(key_master)
                mcmc = numpyro.infer.MCMC(
                    numpyro.infer.NUTS(
                        whitelight_model_for_run, **adaptive_wl_nuts_kwargs
                    ),
                    progress_bar=True,
                    jit_model_args=True,
                    **adaptive_wl_mcmc_kwargs,
                )
                adaptive_run_kwargs = dict(wl_run_kwargs)
                adaptive_run_kwargs["init_params"] = None
                mcmc.run(adaptive_key, *wl_run_args, **adaptive_run_kwargs)
                (
                    wl_samples,
                    wl_samples_grouped,
                    wl_extra_fields,
                    wl_quality_diagnostics,
                ) = _continue_mcmc_until_geometry_gate(
                    mcmc,
                    adaptive_key,
                    wl_run_args,
                    adaptive_run_kwargs,
                    min_ess=float(flags.get("whitelight_min_ess", 400)),
                    max_divergences=int(
                        flags.get("whitelight_max_divergences", 0)
                    ),
                    max_extra_blocks=int(
                        flags.get("whitelight_max_extra_blocks", 3)
                    ),
                )
                laplace_diagnostics.update(wl_quality_diagnostics)
                laplace_diagnostics["mass_matrix"] = "adaptive_fallback"
                laplace_diagnostics["whitelight_laplace_fell_back"] = True
            if not wl_quality_diagnostics["whitelight_quality_gate_passed"]:
                print(
                    "WARNING: WHITE-LIGHT MCMC FAILED ITS ESS/DIVERGENCE GATE "
                    "AFTER ALL EXTENSION BLOCKS."
                )
            _save_mcmc_diagnostics(
                mcmc,
                max_tree_depth=wl_nuts_kwargs.get("max_tree_depth"),
                output_path=os.path.join(
                    output_dir, "whitelight_mcmc_diagnostics.json"
                ),
                extra_diagnostics=laplace_diagnostics,
                samples_override=wl_samples,
                extra_fields_override=wl_extra_fields,
            )
            inf_data = az.from_dict(posterior=wl_samples_grouped)
            if save_trace:
                az.to_netcdf(
                    inf_data,
                    os.path.join(
                        output_dir,
                        f'whitelight_trace_{n_planets}planets.nc',
                    ),
                )
            grouped_reference = np.asarray(
                jax.device_get(next(iter(wl_samples_grouped.values())))
            )
            if grouped_reference.ndim < 2:
                raise RuntimeError(
                    "Grouped white-light posterior is missing its draw dimension."
                )
            num_wl_chains = int(grouped_reference.shape[0])
            if (
                grouped_reference.shape[0] * grouped_reference.shape[1]
                != _posterior_num_draws(wl_samples)
            ):
                raise RuntimeError(
                    "Flattened and grouped white-light posterior draw counts "
                    "do not agree."
                )
            selected_wl_draw = None
            if whitelight_geometry_estimator == 'max_likelihood_draw':
                summed_wl_log_likelihood = _sum_data_log_likelihood_per_draw(
                    whitelight_model_for_run,
                    wl_samples,
                    data.wl_time,
                    data.wl_flux_err,
                    y=data.wl_flux,
                    prior_params=hyper_params_wl,
                    batch_size=whitelight_log_likelihood_batch_size,
                )
                selected_wl_draw = _select_max_likelihood_retained_draw(
                    wl_samples,
                    summed_wl_log_likelihood,
                    num_chains=num_wl_chains,
                )
                print(
                    "Selected white-light maximum-likelihood retained draw: "
                    f"flat={selected_wl_draw['flat_draw_index']}, "
                    f"chain={selected_wl_draw['chain_index']}, "
                    f"draw={selected_wl_draw['draw_index']}, "
                    "summed data log likelihood="
                    f"{selected_wl_draw['summed_data_log_likelihood']:.8g}."
                )
            print(az.summary(inf_data, var_names=None, round_to=7))

            bestfit_params_wl = {
                'period': PERIOD_FIXED,
            }
            def set_param_stats(name, data_samples, axis=0):
                med, low, high = get_asym_errors(data_samples, axis=axis)
                bestfit_params_wl[name] = med
                bestfit_params_wl[f'{name}_err_low'] = low
                bestfit_params_wl[f'{name}_err_high'] = high
            def set_fixed_param_stats(name, values):
                values = jnp.asarray(values)
                zeros = jnp.zeros_like(values)
                bestfit_params_wl[name] = values
                bestfit_params_wl[f'{name}_err'] = zeros
                bestfit_params_wl[f'{name}_err_low'] = zeros
                bestfit_params_wl[f'{name}_err_high'] = zeros

            if ld_profile == 'power2':
                if 'c1' in wl_samples and 'c2' in wl_samples:
                    set_param_stats('c1', wl_samples['c1'], axis=0)
                    set_param_stats('c2', wl_samples['c2'], axis=0)
                else:
                    set_fixed_param_stats('c1', U_mu_wl[0])
                    set_fixed_param_stats('c2', U_mu_wl[1])
                POLY_DEGREE = 12
                MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
                power2_profile = get_I_power2(bestfit_params_wl['c1'], bestfit_params_wl['c2'], MUS)
                u_poly = calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
                bestfit_params_wl['u'] = u_poly
                if transit_engine == 'harmonica':
                    bestfit_params_wl['c_ld'] = bestfit_params_wl['c1']
                    bestfit_params_wl['alpha_ld'] = bestfit_params_wl['c2']
            else:
                if 'u' in wl_samples:
                    set_param_stats('u', wl_samples['u'], axis=0)
                else:
                    set_fixed_param_stats('u', U_mu_wl)
                if transit_engine == 'harmonica':
                    if 'u1' in wl_samples and 'u2' in wl_samples:
                        set_param_stats('u1', wl_samples['u1'], axis=0)
                        set_param_stats('u2', wl_samples['u2'], axis=0)
                    else:
                        set_fixed_param_stats('u1', U_mu_wl[0])
                        set_fixed_param_stats('u2', U_mu_wl[1])
                    bestfit_params_wl['u1_ld'] = bestfit_params_wl['u1']
                    bestfit_params_wl['u2_ld'] = bestfit_params_wl['u2']
            set_fixed_param_stats('ecc', HARMONICA_ECC)
            set_fixed_param_stats('omega', HARMONICA_OMEGA)
            if transit_engine == 'harmonica':
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    if harmonic_name in wl_samples:
                        set_param_stats(harmonic_name, wl_samples[harmonic_name])

            # Post-hoc geometry derivation (engine-agnostic).
            derived = derive_geometry(wl_samples, PERIOD_FIXED, ecc=HARMONICA_ECC, omega=HARMONICA_OMEGA)

            durations_fit, t0s_fit, bs_fit, rors_fit = [], [], [], []
            cos_is_fit, a_rss_fit = [], []
            durations_err, t0s_err, bs_err, rors_err, depths_err = [], [], [], [], []
            cos_is_err, a_rss_err = [], []
            durations_err_low, t0s_err_low, bs_err_low, rors_err_low, depths_err_low = [], [], [], [], []
            cos_is_err_low, a_rss_err_low = [], []
            durations_err_high, t0s_err_high, bs_err_high, rors_err_high, depths_err_high = [], [], [], [], []
            cos_is_err_high, a_rss_err_high = [], []

            for i in range(n_planets):
                dur_s = derived[f'duration_{i}']
                b_s = derived[f'b_{i}']
                a_rs_s = derived[f'a_rs_{i}']
                cos_i_s = derived[f'cos_i_{i}']

                med_d, low_d, high_d = get_asym_errors(dur_s)
                med_t0, low_t0, high_t0 = get_asym_errors(wl_samples[f't0_{i}'])
                med_b, low_b, high_b = get_asym_errors(b_s)
                med_r, low_r, high_r = get_asym_errors(wl_samples[f'rors_{i}'])
                med_depth, low_depth, high_depth = get_asym_errors(wl_samples[f'rors_{i}']**2)
                med_cos_i, low_cos_i, high_cos_i = get_asym_errors(cos_i_s)
                med_a_rs, low_a_rs, high_a_rs = get_asym_errors(a_rs_s)

                durations_fit.append(med_d); t0s_fit.append(med_t0)
                bs_fit.append(med_b); rors_fit.append(med_r)
                cos_is_fit.append(med_cos_i); a_rss_fit.append(med_a_rs)

                durations_err.append(jnp.std(dur_s))
                t0s_err.append(jnp.std(wl_samples[f't0_{i}']))
                bs_err.append(jnp.std(b_s))
                rors_err.append(jnp.std(wl_samples[f'rors_{i}']))
                depths_err.append(jnp.std(wl_samples[f'rors_{i}']**2))
                cos_is_err.append(jnp.std(cos_i_s))
                a_rss_err.append(jnp.std(a_rs_s))

                durations_err_low.append(low_d); durations_err_high.append(high_d)
                t0s_err_low.append(low_t0); t0s_err_high.append(high_t0)
                bs_err_low.append(low_b); bs_err_high.append(high_b)
                rors_err_low.append(low_r); rors_err_high.append(high_r)
                depths_err_low.append(low_depth); depths_err_high.append(high_depth)
                cos_is_err_low.append(low_cos_i); cos_is_err_high.append(high_cos_i)
                a_rss_err_low.append(low_a_rs); a_rss_err_high.append(high_a_rs)

            bestfit_params_wl['duration'] = jnp.array(durations_fit)
            bestfit_params_wl['t0'] = jnp.array(t0s_fit)
            bestfit_params_wl['b'] = jnp.array(bs_fit)
            bestfit_params_wl['rors'] = jnp.array(rors_fit)
            bestfit_params_wl['depths'] = jnp.array(rors_fit)**2
            bestfit_params_wl['cos_i'] = jnp.array(cos_is_fit)
            bestfit_params_wl['a_rs'] = jnp.array(a_rss_fit)

            bestfit_params_wl['duration_err'] = jnp.array(durations_err)
            bestfit_params_wl['t0_err'] = jnp.array(t0s_err)
            bestfit_params_wl['b_err'] = jnp.array(bs_err)
            bestfit_params_wl['rors_err'] = jnp.array(rors_err)
            bestfit_params_wl['depths_err'] = jnp.array(depths_err)
            bestfit_params_wl['cos_i_err'] = jnp.array(cos_is_err)
            bestfit_params_wl['a_rs_err'] = jnp.array(a_rss_err)

            bestfit_params_wl['duration_err_low'] = jnp.array(durations_err_low)
            bestfit_params_wl['duration_err_high'] = jnp.array(durations_err_high)
            bestfit_params_wl['t0_err_low'] = jnp.array(t0s_err_low)
            bestfit_params_wl['t0_err_high'] = jnp.array(t0s_err_high)
            bestfit_params_wl['b_err_low'] = jnp.array(bs_err_low)
            bestfit_params_wl['b_err_high'] = jnp.array(bs_err_high)
            bestfit_params_wl['rors_err_low'] = jnp.array(rors_err_low)
            bestfit_params_wl['rors_err_high'] = jnp.array(rors_err_high)
            bestfit_params_wl['depths_err_low'] = jnp.array(depths_err_low)
            bestfit_params_wl['depths_err_high'] = jnp.array(depths_err_high)
            bestfit_params_wl['cos_i_err_low'] = jnp.array(cos_is_err_low)
            bestfit_params_wl['cos_i_err_high'] = jnp.array(cos_is_err_high)
            bestfit_params_wl['a_rs_err_low'] = jnp.array(a_rss_err_low)
            bestfit_params_wl['a_rs_err_high'] = jnp.array(a_rss_err_high)

            set_param_stats('error', wl_samples['error'])
            
            if detrending_type != 'none':
                set_param_stats('c', wl_samples['c'])
                if detrending_type != 'gp': 
                    set_param_stats('v', wl_samples['v'])
                if 'v' in wl_samples and 'v' not in bestfit_params_wl:
                     set_param_stats('v', wl_samples['v'])

            if 'v2' in wl_samples: set_param_stats('v2', wl_samples['v2'])
            if 'v3' in wl_samples: set_param_stats('v3', wl_samples['v3'])
            if 'v4' in wl_samples: set_param_stats('v4', wl_samples['v4'])
            if 'explinear' in detrending_type:
                set_param_stats('A', wl_samples['A'])
                set_param_stats('tau', wl_samples['tau'])
            if 'spot' in detrending_type:
                set_param_stats('spot_amp', wl_samples['spot_amp'])
                set_param_stats('spot_mu', wl_samples['spot_mu'])
                set_param_stats('spot_sigma', wl_samples['spot_sigma'])
            if '2spot' in detrending_type:
                set_param_stats('spot_amp2', wl_samples['spot_amp2'])
                set_param_stats('spot_mu2', wl_samples['spot_mu2'])
                set_param_stats('spot_sigma2', wl_samples['spot_sigma2'])
            if 'linear_discontinuity' in detrending_type:
                set_param_stats('t_jump', wl_samples['t_jump'])
                set_param_stats('jump', wl_samples['jump'])
    
            if 'gp' in detrending_type:
                set_param_stats('GP_log_sigma', wl_samples['GP_log_sigma'])
                set_param_stats('GP_log_rho', wl_samples['GP_log_rho'])

            if whitelight_geometry_estimator == 'max_likelihood_draw':
                selected_wl_samples = selected_wl_draw['samples']
                wl_handoff_geometry = _geometry_from_white_light_samples(
                    selected_wl_samples,
                    derive_geometry,
                    PERIOD_FIXED,
                    ecc=HARMONICA_ECC,
                    omega=HARMONICA_OMEGA,
                )
                wl_handoff_payload = {
                    "estimator": whitelight_geometry_estimator,
                    "posterior_fingerprint_sha256": wl_artifact_fingerprint,
                    "transit_engine": transit_engine,
                    "param_method": param_method,
                    "num_retained_draws": _posterior_num_draws(wl_samples),
                    "num_chains": num_wl_chains,
                    "selected_flat_draw_index": selected_wl_draw['flat_draw_index'],
                    "selected_chain_index": selected_wl_draw['chain_index'],
                    "selected_draw_index": selected_wl_draw['draw_index'],
                    "summed_data_log_likelihood": (
                        selected_wl_draw['summed_data_log_likelihood']
                    ),
                    "selected_primitive_parameters": (
                        _selected_geometry_primitives(
                            selected_wl_samples, n_planets
                        )
                    ),
                    "geometry": wl_handoff_geometry,
                }
            else:
                wl_handoff_geometry = _geometry_from_white_light_medians(
                    bestfit_params_wl, PERIOD_FIXED
                )
                wl_handoff_payload = {
                    "estimator": whitelight_geometry_estimator,
                    "posterior_fingerprint_sha256": wl_artifact_fingerprint,
                    "transit_engine": transit_engine,
                    "param_method": param_method,
                    "num_retained_draws": _posterior_num_draws(wl_samples),
                    "num_chains": num_wl_chains,
                    "selected_flat_draw_index": None,
                    "selected_chain_index": None,
                    "selected_draw_index": None,
                    "summed_data_log_likelihood": None,
                    "selected_primitive_parameters": {},
                    "geometry": wl_handoff_geometry,
                }
            spot_trend, spot_trend2, jump_trend = None, None, None
            if 'spot' in detrending_type and '2spot' not in detrending_type:
                spot_trend = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp"],
                    bestfit_params_wl["spot_mu"],
                    jnp.abs(bestfit_params_wl["spot_sigma"])
                )
            if '2spot' in detrending_type:
                spot_trend = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp"],
                    bestfit_params_wl["spot_mu"],
                    jnp.abs(bestfit_params_wl["spot_sigma"])
                )
                spot_trend2 = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp2"],
                    bestfit_params_wl["spot_mu2"],
                    jnp.abs(bestfit_params_wl["spot_sigma2"])
                )
            if 'linear_discontinuity' in detrending_type:
                jump_trend = bestfit_params_wl["jump"] * _soft_step_np(np.array(data.wl_time), bestfit_params_wl["t_jump"])

            model_eval_params_wl = _select_transit_eval_params(
                bestfit_params_wl,
                transit_engine=transit_engine,
                param_method=param_method,
                t=data.wl_time,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
            )

            if 'gp' in detrending_type:
                if 'quartic' in detrending_type:
                    gp_mean_func = compute_lc_quartic_gp_mean
                elif 'cubic' in detrending_type:
                    gp_mean_func = compute_lc_cubic_gp_mean
                elif 'quadratic' in detrending_type:
                    gp_mean_func = compute_lc_quadratic_gp_mean
                elif 'explinear' in detrending_type:
                    gp_mean_func = compute_lc_explinear_gp_mean
                elif 'linear' in detrending_type:
                    gp_mean_func = compute_lc_linear_gp_mean
                else:
                    gp_mean_func = compute_lc_gp_mean

                wl_kernel = tinygp.kernels.quasisep.Matern32(
                    scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                    sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
                )
                wl_gp = tinygp.GaussianProcess(
                    wl_kernel,
                    data.wl_time,
                    diag=bestfit_params_wl['error']**2,
                    mean=partial(gp_mean_func, model_eval_params_wl),
                )
                cond_gp = wl_gp.condition(data.wl_flux, data.wl_time).gp
                mu, var = cond_gp.loc, cond_gp.variance
                wl_transit_model = mu
                
                planet_model_only = compute_transit_model_auto(model_eval_params_wl, data.wl_time)
                trend_flux_total = mu - planet_model_only - 1.0
                
                parametric_mean_val = gp_mean_func(model_eval_params_wl, data.wl_time)
                gp_stochastic_component = mu - parametric_mean_val

            else:
                try:
                    wl_transit_model = resolve_detrend_kernel(detrending_type)(model_eval_params_wl, data.wl_time)
                except KeyError:
                    print('Error with model, not defined!')
                    exit()

            wl_residual = data.wl_flux - wl_transit_model
            wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))
            wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma
            wl_sigma_post_clip = 1.4826 * jnp.nanmedian(jnp.abs(wl_residual[~wl_mad_mask] - jnp.nanmedian(wl_residual[~wl_mad_mask])))
            spec_good_mask = (~wl_mad_mask if len(wl_mad_mask) == len(data.time)
                              else np.ones(len(data.time), dtype=bool))

            plt.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
            plt.scatter(data.wl_time, data.wl_flux, s=6, c='k', zorder=1, alpha=0.5)

            plt.savefig(f"{output_dir}/11_{instrument_full_str}_whitelightmodel.png")
            plt.close()

            plt.scatter(data.wl_time, wl_residual, s=6, c='k')
            plt.title('WL Pre-outlier rejection residual')
            plt.savefig(f"{output_dir}/12_{instrument_full_str}_whitelightresidual.png")
            plt.close()

            t_masked = data.wl_time[~wl_mad_mask]
            f_masked = data.wl_flux[~wl_mad_mask]
            # The diagnostic evaluator carries time-axis-specific phase
            # offsets. Rebuild them after outlier clipping instead of reusing
            # the full-cadence arrays with ``t_masked``.
            masked_model_eval_params_wl = _select_transit_eval_params(
                bestfit_params_wl,
                transit_engine=transit_engine,
                param_method=param_method,
                t=t_masked,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
            )

            if 'gp' in detrending_type:
                planet_model_masked = compute_transit_model_auto(masked_model_eval_params_wl, t_masked)
                mu_masked = mu[~wl_mad_mask] 
                total_trend_at_points = mu_masked - planet_model_masked
                detrended_flux = f_masked - (total_trend_at_points - 1.0)
                gp_stochastic_at_masked = gp_stochastic_component[~wl_mad_mask]

            else:
                trend = _trend_from_params_np(
                    detrending_type,
                    np.array(t_masked),
                    bestfit_params_wl
                )
                detrended_flux = f_masked - trend + 1.0

            plt.scatter(t_masked, detrended_flux, c='k', s=6, alpha=0.5)
            plt.title(f'Detrended WLC: Sigma {round(wl_sigma_post_clip*1e6)} PPM')
            plt.savefig(f'{output_dir}/14_{instrument_full_str}_whitelightdetrended.png')
            plt.close()

            transit_only_model = compute_transit_model_auto(masked_model_eval_params_wl, t_masked) + 1.0
            residuals_detrended = detrended_flux - transit_only_model 

            
            fig = plt.figure(figsize=(26, 12))
            
            gs = gridspec.GridSpec(3, 4, figure=fig, 
                                height_ratios=[1, 1, 1.5], 
                                width_ratios=[1, 1, 1.3, 1.3], 
                                hspace=0.3, wspace=0.25)

            duration_bin = jnp.max(jnp.atleast_1d(bestfit_params_wl['duration']))
            b_time, b_flux = jax_bin_lightcurve(jnp.array(data.wl_time), 
                                                jnp.array(data.wl_flux), 
                                                duration_bin)
            
            b_time_det, b_flux_det = jax_bin_lightcurve(jnp.array(t_masked), 
                                                        jnp.array(detrended_flux), 
                                                        duration_bin)
            b_time_det, b_res_det = jax_bin_lightcurve(jnp.array(t_masked), 
                                            jnp.array(residuals_detrended), 
                                            duration_bin)
            bin_style = dict(c='darkviolet', edgecolors='darkslateblue', s=40,  zorder=10, label='Binned (8/dur)')

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
            ax1.scatter(np.array(b_time), np.array(b_flux), **bin_style)
            ax1.set_title('Raw Light Curve', fontsize=14)
            ax1.set_ylabel('Flux', fontsize=12)
            ax1.tick_params(labelbottom=False)

            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax2.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
            ax2.scatter(np.array(b_time), np.array(b_flux), **bin_style)
            ax2.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
            ax2.set_title('Raw Light Curve + Best-fit Model', fontsize=14)
            ax2.set_ylabel('Flux', fontsize=12)
            ax2.tick_params(labelbottom=False)

            gs_nested = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[2, 0], 
                height_ratios=[2, 1],
                hspace=0.0
            )

            ax3_top = fig.add_subplot(gs_nested[0], sharex=ax1)
            ax3_top.scatter(t_masked, detrended_flux, c='k', s=4, alpha=0.2, label='Detrended Data')
            ax3_top.plot(t_masked, transit_only_model, color="mediumorchid", lw=2, zorder=3, label='Transit Model')
            ax3_top.scatter(np.array(b_time_det), np.array(b_flux_det), **bin_style)
            ax3_top.set_ylabel('Normalized Flux', fontsize=12)
            ax3_top.set_title('Detrended Light Curve', fontsize=14)
            plt.setp(ax3_top.get_xticklabels(), visible=False)

            ax3_bot = fig.add_subplot(gs_nested[1], sharex=ax3_top)
            ax3_bot.scatter(t_masked, residuals_detrended * 1e6, c='k', s=4, alpha=0.2)
            ax3_bot.axhline(0, color='mediumorchid', lw=4, zorder=3, linestyle='--')
            ax3_bot.scatter(np.array(b_time_det), np.array(b_res_det) * 1e6 , **bin_style)
            ax3_bot.set_ylabel('Res. (ppm)', fontsize=10)
            ax3_bot.set_xlabel('Time (BJD)', fontsize=12)

            dt = np.median(np.diff(data.wl_time)) * 86400 
            residuals_arr = np.array(wl_residual[~wl_mad_mask])
            beta, bin_sizes_min, measured_rms, expected_rms = calculate_beta_metrics(residuals_arr, dt)
            mc_betas, rms_lo_1, rms_hi_1, rms_lo_2, rms_hi_2 = run_beta_monte_carlo(residuals_arr, dt, n_sims=500)

            mu_sim, std_sim = norm.fit(mc_betas)
            z_score = (beta - mu_sim) / std_sim

            print(f"Beta: {beta:.4f}")
            print(f"Measured Beta: {beta:.3f}")
            print(f"MC Mean Beta:  {mu_sim:.3f}")
            print(f"MC Std Dev:    {std_sim:.3f}")
            print(f"Significance:  {z_score:.2f} sigma")

            ax_rms = fig.add_subplot(gs[0:2, 1])
            ax_rms.loglog(bin_sizes_min, expected_rms * 1e6, 'k--', lw=1.5, label='Theory $1/\sqrt{N}$')
            ax_rms.fill_between(bin_sizes_min, rms_lo_2 * 1e6, rms_hi_2 * 1e6, color='gray', alpha=0.2, label='White Noise ($2\sigma$)')
            ax_rms.fill_between(bin_sizes_min, rms_lo_1 * 1e6, rms_hi_1 * 1e6, color='gray', alpha=0.4, label='White Noise ($1\sigma$)')
            ax_rms.loglog(bin_sizes_min, measured_rms * 1e6, color='teal', lw=2, marker='o', markersize=5, label=f'Data (Beta={beta:.2f})')
            ax_rms.set_xlabel('Bin Size (minutes)', fontsize=12)
            ax_rms.set_ylabel('RMS (ppm)', fontsize=12)
            ax_rms.set_title('Time-Correlated Noise', fontsize=14)
            ax_rms.grid(True, which="both", alpha=0.2)

            ax_beta = fig.add_subplot(gs[2, 1])
            n, bins, patches = ax_beta.hist(mc_betas, bins=30, color='silver', alpha=0.6, density=True, label='Simulated White Noise')
            xmin, xmax = ax_beta.get_xlim()
            x_plot = np.linspace(xmin, xmax, 100)
            p_plot = norm.pdf(x_plot, mu_sim, std_sim)
            ax_beta.plot(x_plot, p_plot, 'k--', linewidth=2, label='Gaussian Fit')
            ax_beta.axvline(beta, color='teal', lw=3, label=f'Measured: {beta:.2f}')
            sig_color = 'green' if abs(z_score) < 2.0 else ('orange' if abs(z_score) < 3.0 else 'firebrick')
            ax_beta.text(0.95, 0.85, f"Significance: {z_score:.1f}$\sigma$", 
                       transform=ax_beta.transAxes, ha='right', fontsize=14, color=sig_color, fontweight='bold')
            ax_beta.set_xlabel('Beta Factor', fontsize=12)
            ax_beta.set_ylabel('Probability Density', fontsize=12)
            ax_beta.set_title("Beta Significance Test", fontsize=14)

            gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 2:], wspace=0.05)
            
            ax_hr_flux = fig.add_subplot(gs_right[0])
            ax_hr_res = fig.add_subplot(gs_right[1], sharey=ax_hr_flux) 

            n_hr_bins = data.flux_hr.shape[0]
            hr_indices = np.linspace(0, n_hr_bins-1, 10, dtype=int)
            colors = cm.turbo(np.linspace(0, 1, 10))
            
            est_depth = np.nanmedian(bestfit_params_wl['depths'])
            if est_depth < 1e-4: offset_step = 0.0025
            elif est_depth < 1e-3: offset_step = 0.0075
            else: offset_step = 0.02

            wl_time_vector = np.array(data.time[spec_good_mask])
            wl_model_vector_params = _select_transit_eval_params(
                bestfit_params_wl,
                transit_engine=transit_engine,
                param_method=param_method,
                t=wl_time_vector,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
            )
            wl_model_vector = np.array(
                compute_transit_model_auto(
                    wl_model_vector_params, jnp.array(wl_time_vector)
                )
                + 1.0
            )
            time_center = float(bestfit_params_wl['t0'][0])
            res_zoom_factor = 1.0 

            for idx_i, bin_idx in enumerate(hr_indices):
                raw_flux_hr = data.flux_hr[bin_idx]
                flux_hr_masked = raw_flux_hr[spec_good_mask]
                
                baseline_norm = np.nanmedian(flux_hr_masked[:50])
                norm_flux_hr = flux_hr_masked / baseline_norm
                
                residuals_hr_check = norm_flux_hr - wl_model_vector
                
                mad_ppm = 1.4826 * np.nanmedian(np.abs(residuals_hr_check)) * 1e6

                y_offset = idx_i * offset_step
                
                ax_hr_flux.scatter(wl_time_vector - time_center, norm_flux_hr + y_offset, 
                                 color=colors[idx_i], s=5, alpha=0.6, edgecolors='none')
                
                ax_hr_flux.plot(wl_time_vector - time_center, wl_model_vector + y_offset, 
                              color='dimgray', lw=2.0, alpha=0.3, zorder=2)
                ax_hr_flux.plot(wl_time_vector - time_center, wl_model_vector + y_offset, 
                              color=colors[idx_i], lw=1.0, alpha=0.9, linestyle='-', zorder=3)
                
                wl_val = data.wavelengths_hr[bin_idx]
                annotation_y = 1.0 + y_offset + (offset_step * 0.33)
                ax_hr_flux.text(wl_time_vector.min() - time_center, annotation_y, 
                              f"{wl_val:.2f} $\mu$m", 
                              fontsize=9, fontweight='bold', color=colors[idx_i])

                res_plotted = (residuals_hr_check * res_zoom_factor) + 1.0 + y_offset
                
                ax_hr_res.scatter(wl_time_vector - time_center, res_plotted, 
                                color=colors[idx_i], s=5, alpha=0.6, edgecolors='none')
                
                ax_hr_res.text(wl_time_vector.min() - time_center, annotation_y, 
                               f"$\sigma$={int(mad_ppm)} ppm", 
                               fontsize=9, fontweight='bold', color=colors[idx_i])
                
                ax_hr_res.axhline(1.0 + y_offset, color='black', linestyle='--', lw=1, alpha=0.3)

            ax_hr_flux.set_xlabel("Time from Mid-Transit (days)", fontsize=12)
            ax_hr_res.set_xlabel("Time from Mid-Transit (days)", fontsize=12)
            
            ax_hr_flux.set_yticks([])
            ax_hr_res.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/15_{instrument_full_str}_whitelight_summary.png')
            plt.close(fig)

            transit_only_full = np.asarray(
                compute_transit_model_auto(model_eval_params_wl, data.wl_time) + 1.0,
                dtype=float,
            )
            bestfit_model_full = np.asarray(wl_transit_model, dtype=float)
            trend_model_full = bestfit_model_full - transit_only_full
            gp_flux_full = np.asarray(mu, dtype=float) if 'gp' in detrending_type else None
            gp_err_full = np.asarray(jnp.sqrt(var), dtype=float) if 'gp' in detrending_type else None
            gp_trend_full = np.asarray(gp_stochastic_component, dtype=float) if 'gp' in detrending_type else None
            save_whitelight_timeseries(
                time=data.wl_time,
                flux=data.wl_flux,
                flux_err=data.wl_flux_err,
                bestfit_model=bestfit_model_full,
                output_csv=f'{output_dir}/{instrument_full_str}_whitelight_timeseries.csv',
                t0_reference=float(np.asarray(bestfit_params_wl['t0']).ravel()[0]) if np.size(bestfit_params_wl['t0']) else None,
                transit_model=transit_only_full,
                trend_model=trend_model_full,
                residual=np.asarray(wl_residual, dtype=float),
                outlier_mask=np.asarray(wl_mad_mask, dtype=bool),
                gp_flux=gp_flux_full,
                gp_err=gp_err_full,
                gp_trend=gp_trend_full,
            )

            
            _atomic_save_npy(wl_mask_path, wl_mad_mask)
            
            if 'gp' in detrending_type:
                df = pd.DataFrame({
                    'wl_flux': data.wl_flux, 
                    'gp_flux': mu,
                    'gp_err': jnp.sqrt(var), 
                    'gp_trend': gp_stochastic_component
                }) 
                _atomic_dataframe_csv(df, wl_gp_path, index=True)
            
            rows = []
            for i in range(n_planets):
                row = {
                    'planet_id': i,
                    'period': bestfit_params_wl['period'][i],
                    'duration': bestfit_params_wl['duration'][i],
                    't0': bestfit_params_wl['t0'][i],
                    'b': bestfit_params_wl['b'][i],
                    'rors': bestfit_params_wl['rors'][i],
                    'depths': bestfit_params_wl['depths'][i],
                    'duration_err': bestfit_params_wl['duration_err'][i],
                    't0_err': bestfit_params_wl['t0_err'][i],
                    'b_err': bestfit_params_wl['b_err'][i],
                    'rors_err': bestfit_params_wl['rors_err'][i],
                    'depths_err': bestfit_params_wl['depths_err'][i],
                    'duration_err_low': bestfit_params_wl['duration_err_low'][i],
                    'duration_err_high': bestfit_params_wl['duration_err_high'][i],
                    't0_err_low': bestfit_params_wl['t0_err_low'][i],
                    't0_err_high': bestfit_params_wl['t0_err_high'][i],
                    'b_err_low': bestfit_params_wl['b_err_low'][i],
                    'b_err_high': bestfit_params_wl['b_err_high'][i],
                    'rors_err_low': bestfit_params_wl['rors_err_low'][i],
                    'rors_err_high': bestfit_params_wl['rors_err_high'][i],
                    'depths_err_low': bestfit_params_wl['depths_err_low'][i],
                    'depths_err_high': bestfit_params_wl['depths_err_high'][i],
                }
                if 'a_rs' in bestfit_params_wl:
                    row['a_rs'] = bestfit_params_wl['a_rs'][i]
                    row['a_rs_err'] = bestfit_params_wl['a_rs_err'][i]
                    row['a_rs_err_low'] = bestfit_params_wl['a_rs_err_low'][i]
                    row['a_rs_err_high'] = bestfit_params_wl['a_rs_err_high'][i]
                if 'cos_i' in bestfit_params_wl:
                    row['cos_i'] = bestfit_params_wl['cos_i'][i]
                    row['cos_i_err'] = bestfit_params_wl['cos_i_err'][i]
                    row['cos_i_err_low'] = bestfit_params_wl['cos_i_err_low'][i]
                    row['cos_i_err_high'] = bestfit_params_wl['cos_i_err_high'][i]
                if 'ecc' in bestfit_params_wl:
                    row['ecc'] = bestfit_params_wl['ecc'][i]
                    row['ecc_err'] = bestfit_params_wl['ecc_err'][i]
                    row['ecc_err_low'] = bestfit_params_wl['ecc_err_low'][i]
                    row['ecc_err_high'] = bestfit_params_wl['ecc_err_high'][i]
                if 'omega' in bestfit_params_wl:
                    row['omega'] = bestfit_params_wl['omega'][i]
                    row['omega_err'] = bestfit_params_wl['omega_err'][i]
                    row['omega_err_low'] = bestfit_params_wl['omega_err_low'][i]
                    row['omega_err_high'] = bestfit_params_wl['omega_err_high'][i]
                
                def add_scalar_param(name):
                    if name in bestfit_params_wl:
                        row[name] = bestfit_params_wl[name]
                        if f'{name}_err_low' in bestfit_params_wl:
                            row[f'{name}_err_low'] = bestfit_params_wl[f'{name}_err_low']
                            row[f'{name}_err_high'] = bestfit_params_wl[f'{name}_err_high']

                if detrending_type != 'none':
                    add_scalar_param('c')
                    add_scalar_param('v')
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    add_scalar_param(harmonic_name)
                
                if 'c1' in bestfit_params_wl:
                    add_scalar_param('c1')
                    add_scalar_param('c2')
                    row['u1'] = bestfit_params_wl['c1']
                    row['u2'] = bestfit_params_wl['c2']
                    row['u1_err_low'] = bestfit_params_wl['c1_err_low']
                    row['u1_err_high'] = bestfit_params_wl['c1_err_high']
                    row['u2_err_low'] = bestfit_params_wl['c2_err_low']
                    row['u2_err_high'] = bestfit_params_wl['c2_err_high']
                else:
                    row['u1'] = bestfit_params_wl['u'][0]
                    row['u2'] = bestfit_params_wl['u'][1]
                    if 'u_err_low' in bestfit_params_wl:
                        row['u1_err_low'] = bestfit_params_wl['u_err_low'][0]
                        row['u1_err_high'] = bestfit_params_wl['u_err_high'][0]
                        row['u2_err_low'] = bestfit_params_wl['u_err_low'][1]
                        row['u2_err_high'] = bestfit_params_wl['u_err_high'][1]
                
                add_scalar_param('v2')
                add_scalar_param('v3')
                add_scalar_param('v4')
                add_scalar_param('A')
                add_scalar_param('tau')
                add_scalar_param('spot_amp')
                add_scalar_param('spot_mu')
                add_scalar_param('spot_sigma')
                add_scalar_param('spot_amp2')
                add_scalar_param('spot_mu2')
                add_scalar_param('spot_sigma2')
                add_scalar_param('t_jump')
                add_scalar_param('jump')
                add_scalar_param('GP_log_sigma')
                add_scalar_param('GP_log_rho')
                
                rows.append(row)
        
            df = pd.DataFrame(rows)
            _atomic_dataframe_csv(df, wl_params_path, index=False)
            bestfit_params_wl_df = pd.read_csv(wl_params_path)
            if transit_engine == 'harmonica' and _has_harmonica_odd_samples(wl_samples):
                band_wl_raw = np.asarray(data.wavelengths_hr)
                band_wl_err_raw = np.asarray(data.wavelengths_err_hr)
                if band_wl_raw.size == 0:
                    band_wl_raw = np.asarray(data.wavelengths_lr)
                    band_wl_err_raw = np.asarray(data.wavelengths_err_lr)
                band_wl_arr, band_wl_err_arr = _coerce_wavelength_axis(
                    band_wl_raw,
                    band_wl_err_raw if band_wl_raw.size > 0 else None,
                )
                bandpass_min = float(np.nanmin(band_wl_arr - band_wl_err_arr))
                bandpass_max = float(np.nanmax(band_wl_arr + band_wl_err_arr))
                save_harmonica_limb_products(
                    wavelengths=0.5 * (bandpass_min + bandpass_max),
                    wavelength_err=0.5 * (bandpass_max - bandpass_min),
                    rors_samples=wl_samples['rors_0'],
                    harmonic_samples=_harmonica_sample_payload(wl_samples),
                    csv_path=wl_limb_csv_path,
                    limb_spectrum_path=f"{output_dir}/16_{instrument_full_str}_whitelight_limb_spectra.png",
                    transmission_strings_path=f"{output_dir}/17_{instrument_full_str}_whitelight_transmission_string.png",
                    posterior_samples_path=wl_limb_samples_path,
                    title_prefix=f"{planet_str} - Whitelight",
                    bandpass_min=bandpass_min,
                    bandpass_max=bandpass_max,
                )

            wl_geometry_handoff = _write_whitelight_geometry_handoff(
                wl_geometry_handoff_path, wl_handoff_payload
            )
            print(
                "Saved white-light fixed-geometry handoff to "
                f"{wl_geometry_handoff_path}."
            )
            _write_science_artifact_manifest(
                wl_manifest_path, "whitelight", wl_artifact_fingerprint
            )
        else:
            print(f'GP trends already exist...')
            wl_mad_mask = np.load(wl_mask_path)
            bestfit_params_wl_df = pd.read_csv(wl_params_path)
    else:
        print(f'Whitelight outliers and bestfit parameters already exist...')
        wl_mad_mask = np.load(wl_mask_path)
        bestfit_params_wl_df = pd.read_csv(wl_params_path)

    if wl_geometry_handoff is None:
        raise RuntimeError(
            "White-light fitting completed without a valid fixed-geometry "
            "handoff artifact."
        )
    fixed_geometry = wl_geometry_handoff["geometry"]
    DURATION_BASE = np.asarray(fixed_geometry["duration"], dtype=float)
    T0_BASE = np.asarray(fixed_geometry["t0"], dtype=float)
    B_BASE = np.asarray(fixed_geometry["b"], dtype=float)
    RORS_BASE = np.asarray(fixed_geometry["rors"], dtype=float)
    A_RS_BASE = np.asarray(fixed_geometry["a_rs"], dtype=float)
    COSI_BASE = np.asarray(fixed_geometry["cos_i"], dtype=float)
    DEPTH_BASE = RORS_BASE**2
    print(
        "Using white-light geometry handoff "
        f"({wl_geometry_handoff['estimator']}, "
        f"fingerprint={wl_geometry_handoff['artifact_fingerprint_sha256'][:12]}) "
        "for spectroscopic fitting."
    )
    if analysis_stage == 'whitelight':
        print("White-light-only analysis complete.")
        return None

    spec_good_mask = (~wl_mad_mask if len(wl_mad_mask) == len(data.time)
                      else np.ones(len(data.time), dtype=bool))
    wl_time_good = data.wl_time[~wl_mad_mask] if len(wl_mad_mask) == len(data.wl_time) else data.wl_time

    key_lr, key_hr, key_map_lr, key_mcmc_lr, key_map_hr, key_mcmc_hr, key_prior_pred = jax.random.split(key_master, 7)
    need_lowres_analysis = interpolate_trend or interpolate_ld or need_lowres
    
    trend_fixed_hr = None
    ld_fixed_hr = None
    best_poly_coeffs_c, best_poly_coeffs_v = None, None
    best_poly_coeffs_u1, best_poly_coeffs_u2 = None, None
    spot_trend, spot_trend2, jump_trend, exp_trend = None, None, None, None
    fixed_tau_spectro = None
    if spectro_fixed_timescale_trends and 'explinear' in detrending_type:
        if 'tau' not in bestfit_params_wl_df.columns:
            raise ValueError("Fixed-timescale spectroscopy requires white-light tau.")
        fixed_tau_spectro = float(bestfit_params_wl_df['tau'].values[0])
        if not np.isfinite(fixed_tau_spectro) or fixed_tau_spectro <= 0.0:
            raise ValueError("White-light tau must be finite and positive.")
        exp_trend = np.exp(
            -(np.asarray(wl_time_good) - np.min(np.asarray(wl_time_good)))
            / fixed_tau_spectro
        )
    if 'spot' in detrending_type and '2spot' not in detrending_type:
        if {'spot_amp', 'spot_mu', 'spot_sigma'}.issubset(bestfit_params_wl_df.columns):
            spot_amp = bestfit_params_wl_df['spot_amp'].values[0]
            spot_mu = bestfit_params_wl_df['spot_mu'].values[0]
            spot_sigma = bestfit_params_wl_df['spot_sigma'].values[0]
            if not np.isnan(spot_amp) and not np.isnan(spot_mu) and not np.isnan(spot_sigma):
                spot_trend = spot_crossing(wl_time_good, spot_amp, spot_mu, np.abs(spot_sigma))
    if '2spot' in detrending_type:
        if {'spot_amp', 'spot_mu', 'spot_sigma'}.issubset(bestfit_params_wl_df.columns):
            spot_amp = bestfit_params_wl_df['spot_amp'].values[0]
            spot_mu = bestfit_params_wl_df['spot_mu'].values[0]
            spot_sigma = bestfit_params_wl_df['spot_sigma'].values[0]
            if not np.isnan(spot_amp) and not np.isnan(spot_mu) and not np.isnan(spot_sigma):
                spot_trend = spot_crossing(wl_time_good, spot_amp, spot_mu, np.abs(spot_sigma))
        if {'spot_amp2', 'spot_mu2', 'spot_sigma2'}.issubset(bestfit_params_wl_df.columns):
            spot_amp2 = bestfit_params_wl_df['spot_amp2'].values[0]
            spot_mu2 = bestfit_params_wl_df['spot_mu2'].values[0]
            spot_sigma2 = bestfit_params_wl_df['spot_sigma2'].values[0]
            if not np.isnan(spot_amp2) and not np.isnan(spot_mu2) and not np.isnan(spot_sigma2):
                spot_trend2 = spot_crossing(wl_time_good, spot_amp2, spot_mu2, np.abs(spot_sigma2))
    if 'linear_discontinuity' in detrending_type:
        if {'t_jump', 'jump'}.issubset(bestfit_params_wl_df.columns):
            t_jump = bestfit_params_wl_df['t_jump'].values[0]
            jump = bestfit_params_wl_df['jump'].values[0]
            if not np.isnan(t_jump) and not np.isnan(jump):
                jump_trend = jump * _soft_step_np(np.array(wl_time_good), t_jump)
    valid = None
    if need_lowres_analysis:
        # Compute the exact LD inputs before deciding whether a previous LR fit
        # is reusable. This binds cache validity to the coefficients actually
        # consumed by the likelihood, including changes to external LD grids.
        if ld_prior_mode == 'informed':
            U_mu_lr, U_sigma_lr = get_or_build_power2_ld_prior(
                stellar_cfg,
                data.wavelengths_lr,
                data.wavelengths_err_lr,
                instrument,
                order=order if instrument == 'NIRISS/SOSS' else None,
                output_dir=output_dir,
                cache_label=lr_bin_str,
            )
        else:
            if instrument in [
                'NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM',
                'NIRSPEC/G140H', 'NIRSPEC/G235H', 'MIRI/LRS',
            ]:
                U_mu_lr = get_limb_darkening(
                    sld,
                    data.wavelengths_lr,
                    data.wavelengths_err_lr,
                    instrument,
                    ld_profile=ld_profile,
                    ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None),
                )
            elif instrument == 'NIRISS/SOSS':
                U_mu_lr = get_limb_darkening(
                    sld,
                    data.wavelengths_lr,
                    data.wavelengths_err_lr,
                    instrument,
                    order=order,
                    ld_profile=ld_profile,
                    ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None),
                )
            U_sigma_lr = None
            sing_model_c_lr = None
            if ld_prior_mode == 'sing':
                sing_model_c_lr = jnp.asarray(U_mu_lr)
                U_mu_lr, U_sigma_lr = build_sing_ld_prior(U_mu_lr, flags, stellar_cfg)

        lr_mask_path = (
            f"{output_dir}/{lr_artifact_stem}_spectroscopic_outlier_mask.npy"
        )
        lr_params_path = f"{output_dir}/{lr_artifact_stem}_bestfit_params.csv"
        poly_coeffs_path = f"{output_dir}/{lr_artifact_stem}_poly_coeffs.npz"
        lr_limb_csv_path = f"{output_dir}/{lr_artifact_stem}_limb_spectra.csv"
        lr_limb_samples_path = (
            f"{output_dir}/{lr_artifact_stem}_limb_posterior_samples.npz"
        )
        lr_manifest_path = (
            f"{output_dir}/{lr_artifact_stem}_artifacts.manifest.json"
        )
        lr_artifact_fingerprint = _science_artifact_fingerprint(
            "low_resolution",
            {
                "config": cfg,
                "time": data.time,
                "flux": data.flux_lr,
                "flux_err": data.flux_err_lr,
                "wavelength": data.wavelengths_lr,
                "wavelength_err": data.wavelengths_err_lr,
                "white_light_mask": wl_mad_mask,
                "white_light_fit": bestfit_params_wl_df.to_dict(orient="list"),
                "white_light_geometry_handoff_fingerprint": (
                    wl_geometry_handoff["artifact_fingerprint_sha256"]
                ),
                "transit_engine": transit_engine,
                "ld_profile": ld_profile,
                "ld_prior_mode": ld_prior_mode,
                "ld_coefficients": U_mu_lr,
                "ld_uncertainties": U_sigma_lr,
                "detrending_type": detrending_type,
                "harmonica_spectro_parameterization": (
                    harmonica_spectro_parameterization
                ),
                "spectroscopic_sigma": spectroscopic_sigma,
            },
        )
        required_lr_limb_products_exist = bool(
            transit_engine != 'harmonica'
            or not HARMONICA_ODD_HARMONICS
            or (
                os.path.exists(lr_limb_csv_path)
                and os.path.exists(lr_limb_samples_path)
            )
        )
        lr_cache_valid = bool(
            required_lr_limb_products_exist
            and _science_artifact_manifest_matches(
                lr_manifest_path, lr_artifact_fingerprint
            )
        )
        if (
            os.getenv("FIT_JWST_DUMP_SAMPLER_INPUTS")
            and os.getenv("FIT_JWST_DUMP_FORCE_LOWRES", "0") == "1"
        ):
            lr_cache_valid = False
            print(
                "Sampler-input dump requested low-resolution cache bypass; "
                "cached files remain untouched."
            )
        if (
            os.path.exists(lr_mask_path)
            and os.path.exists(lr_params_path)
            and lr_cache_valid
        ):
            if interpolate_trend or interpolate_ld:
                if os.path.exists(poly_coeffs_path):
                    print(
                        f"Reusing low-res results and polynomial fits from "
                        f"{lr_bin_str}; skipping low-res fit."
                    )
                    time_mask = np.load(lr_mask_path)
                    valid = ~time_mask
                    poly_data = np.load(poly_coeffs_path)
                    if interpolate_trend:
                        best_poly_coeffs_c = poly_data["coeffs_c"]
                        if "coeffs_v" in poly_data:
                            best_poly_coeffs_v = poly_data["coeffs_v"]
                    if interpolate_ld:
                        best_poly_coeffs_u1 = poly_data["coeffs_u1"]
                        best_poly_coeffs_u2 = poly_data["coeffs_u2"]
                else:
                    print(
                        "Low-res files found, but polynomial fits are missing; "
                        "running low-res to build polynomials."
                    )
            else:
                print(
                    f"Reusing low-res results from {lr_bin_str}; "
                    "skipping low-res fit."
                )
                time_mask = np.load(lr_mask_path)
                valid = ~time_mask
        elif os.path.exists(lr_mask_path) or os.path.exists(lr_params_path):
            print(
                "Existing low-resolution products are incomplete, unverified, "
                "or do not match the current data/LD target; recomputing them."
            )

    if need_lowres_analysis and valid is None:
        print(f"\n--- Running Low-Resolution Analysis (Binned to {lr_bin_str}) ---")
        time_lr = jnp.array(data.time[spec_good_mask])
        flux_lr = jnp.array(data.flux_lr[:, spec_good_mask])
        flux_err_lr = jnp.array(data.flux_err_lr[:, spec_good_mask])
        num_lcs_lr = int(data.flux_err_lr.shape[0])

        if 'gp' in detrending_type:
            gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            gp_trend_raw = gp_df['gp_trend'].values
            if len(gp_trend_raw) == len(wl_mad_mask):
                gp_trend_raw = gp_trend_raw[~wl_mad_mask]
            gp_trend = jnp.array(_align_trend_to_time(gp_trend_raw, wl_time_good, np.array(time_lr)))
        else:
            gp_trend = None
        detrend_type_multiwave = _spectro_detrend_type(
            detrending_type, spectro_fixed_timescale_trends
        )

        print(f"Low-res: {num_lcs_lr} light curves.")
        DEPTHS_BASE_LR = jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))

        init_params_lr = {
            "u": U_mu_lr,
            "rors": jnp.tile(RORS_BASE, (num_lcs_lr, 1))
        }
        if ld_prior_mode == 'sing':
            init_params_lr.pop('u', None)
            init_params_lr['limb_l'] = jnp.asarray(U_mu_lr)[:, 0]
            init_params_lr['limb_delta'] = jnp.asarray(U_mu_lr)[:, 1]
        if ld_profile == 'power2' and ld_prior_mode not in {'fixed', 'interpolated'}:
            init_params_lr['c1'] = jnp.asarray(U_mu_lr)[:, 0]
            init_params_lr['c2'] = jnp.asarray(U_mu_lr)[:, 1]
        if transit_engine == 'harmonica':
            if harmonica_spectro_parameterization == 'delta_r':
                init_val = (
                    bestfit_params_wl_df['a1'].values[0]
                    if 'a1' in bestfit_params_wl_df.columns
                    else HARMONICA_INIT_ODD_COEFF
                )
                init_delta_r = jnp.atleast_1d(
                    jnp.asarray(_harmonica_coeff_array_to_delta_r(init_val))
                )
                init_params_lr['delta_r'] = jnp.tile(
                    init_delta_r[None, :],
                    (num_lcs_lr, 1),
                )
            elif harmonica_spectro_parameterization == 'half_area':
                init_val = (
                    bestfit_params_wl_df['a1'].values
                    if 'a1' in bestfit_params_wl_df.columns
                    else np.full_like(RORS_BASE, HARMONICA_INIT_ODD_COEFF)
                )
                init_area_radius, init_q = (
                    _harmonica_coefficients_to_half_area_init(
                        RORS_BASE, init_val
                    )
                )
                init_params_lr['rors'] = jnp.tile(
                    jnp.atleast_1d(jnp.asarray(init_area_radius))[None, :],
                    (num_lcs_lr, 1),
                )
                init_params_lr['q'] = jnp.tile(
                    jnp.atleast_1d(jnp.asarray(init_q))[None, :],
                    (num_lcs_lr, 1),
                )
            else:
                for harmonic_name in HARMONICA_ODD_HARMONICS:
                    init_val = bestfit_params_wl_df[harmonic_name].values[0] if harmonic_name in bestfit_params_wl_df.columns else HARMONICA_INIT_ODD_COEFF
                    init_frac = _harmonica_coeff_array_to_frac(init_val, RORS_BASE)
                    init_params_lr[_harmonica_frac_site(harmonic_name)] = jnp.tile(
                        jnp.asarray(init_frac)[None, :],
                        (num_lcs_lr, 1),
                    )
        if detrend_type_multiwave != 'none':
            init_params_lr['c'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['c'].values[0])
            if 'v' in bestfit_params_wl_df.columns:
                 init_params_lr['v'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['v'].values[0])

        if 'explinear' in detrend_type_multiwave:
            init_params_lr['A'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['A'].values[0])
            if 'explinear_spectroscopic' not in detrend_type_multiwave:
                init_params_lr['log_tau'] = jnp.full(
                    num_lcs_lr,
                    jnp.log(bestfit_params_wl_df['tau'].values[0]),
                )
        
        lr_trend_mode = (
            'gaussian_marginalized'
            if trend_inference == 'gaussian_marginalized'
            else 'free'
        )
        lr_ld_mode = ld_prior_mode
        
        lr_transit_window_indices = None
        if (
            transit_engine == 'jaxoplanet'
            and transit_window_optimization == 'auto'
            and param_method == 'duration'
        ):
            lr_transit_window_indices = build_transit_window_indices(
                np.asarray(time_lr),
                np.asarray(PERIOD_FIXED),
                np.asarray(T0_BASE),
                np.asarray(DURATION_BASE),
            )
            print(
                "Jaxoplanet transit window (low-res): "
                f"{len(lr_transit_window_indices)}/{time_lr.size} active cadences."
            )

        lr_model_builder_kwargs = {
            'detrend_type': detrend_type_multiwave,
            'ld_mode': lr_ld_mode,
            'trend_mode': lr_trend_mode,
            'n_planets': n_planets,
            **(
                {'transit_window_indices': lr_transit_window_indices}
                if transit_engine == 'jaxoplanet' else {}
            ),
            **_engine_spectro_kw,
        }
        lr_model_for_run = _build_spectroscopic_model(
            create_vectorized_model,
            **lr_model_builder_kwargs,
        )

        model_run_args_lr = {
            'mu_duration': DURATION_BASE,
            'mu_t0': T0_BASE,
            'mu_b': B_BASE,
            'mu_depths': DEPTHS_BASE_LR,
            'PERIOD': PERIOD_FIXED,
        }
        if transit_engine == 'jaxoplanet':
            model_run_args_lr['precomputed_yerr_per_lc'] = jnp.nanmedian(
                flux_err_lr, axis=1
            )
        if transit_engine == 'harmonica':
            model_run_args_lr['mu_cos_i'] = COSI_BASE
            model_run_args_lr['harmonica_a_rs'] = A_RS_BASE
            model_run_args_lr['harmonica_ecc'] = HARMONICA_ECC
            model_run_args_lr['harmonica_omega'] = HARMONICA_OMEGA
        elif param_method == 'a_rs':
            model_run_args_lr['mu_a_rs'] = A_RS_BASE
            model_run_args_lr['mu_ecc'] = HARMONICA_ECC
            model_run_args_lr['mu_omega'] = HARMONICA_OMEGA

        if lr_ld_mode == 'fixed':
            model_run_args_lr['ld_fixed'] = U_mu_lr
        elif lr_ld_mode in {'widegaussian', 'informed', 'sing'}:
            model_run_args_lr['mu_u_ld'] = U_mu_lr
            if ((ld_profile == 'power2' and lr_ld_mode == 'informed') or lr_ld_mode == 'sing') and U_sigma_lr is not None:
                model_run_args_lr['sigma_u_ld'] = U_sigma_lr
        elif lr_ld_mode == 'uniform':
            pass
        else:
            raise ValueError(f"Unknown ld_prior mode: {lr_ld_mode}")
        
        if 'gp_spectroscopic' in detrend_type_multiwave:
            model_run_args_lr['gp_trend'] = gp_trend
            init_params_lr['A_gp'] = jnp.ones(num_lcs_lr)
        if '2spot_spectroscopic' in detrend_type_multiwave:
            if spot_trend is None or spot_trend2 is None:
                raise ValueError("2spot_spectroscopic requires WL spot and spot2 trends.")
            if len(spot_trend) != len(time_lr):
                spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_lr))
            if len(spot_trend2) != len(time_lr):
                spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_lr))
            model_run_args_lr['spot_trend'] = spot_trend
            model_run_args_lr['spot_trend2'] = spot_trend2
            init_params_lr['A_spot'] = jnp.ones(num_lcs_lr)
            init_params_lr['A_spot2'] = jnp.ones(num_lcs_lr)
        elif _has_single_spot_spectroscopic(detrend_type_multiwave):
            if spot_trend is None:
                raise ValueError(f"{detrend_type_multiwave} requires WL spot to build spot_trend.")
            if spot_trend is not None and len(spot_trend) != len(time_lr):
                spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_lr))
            model_run_args_lr['spot_trend'] = spot_trend
            init_params_lr['A_spot'] = jnp.ones(num_lcs_lr)
        if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
            if jump_trend is None:
                raise ValueError("linear_discontinuity_spectroscopic requires WL linear_discontinuity to build jump_trend.")
            if jump_trend is not None and len(jump_trend) != len(time_lr):
                jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_lr))
            model_run_args_lr['jump_trend'] = jump_trend
            init_params_lr['A_jump'] = jnp.ones(num_lcs_lr)
        if 'explinear_spectroscopic' in detrend_type_multiwave:
            exp_trend_lr = _align_trend_to_time(
                exp_trend, wl_time_good, np.asarray(time_lr)
            )
            model_run_args_lr['exp_trend'] = exp_trend_lr
            model_run_args_lr['fixed_tau'] = fixed_tau_spectro

        trend_names_lr = None
        if lr_trend_mode == 'gaussian_marginalized':
            (
                model_run_args_lr['trend_prior_mean'],
                model_run_args_lr['trend_prior_scale'],
                trend_names_lr,
            ) = _build_gaussian_trend_prior(
                detrend_type_multiwave,
                num_lcs_lr,
                init_params_lr,
                flags,
            )
            # These coefficients are integrated out exactly.  Keep only
            # genuinely nonlinear trend sites (for example log_tau).
            for trend_name in trend_names_lr:
                init_params_lr.pop(trend_name, None)

        # Sing et al. step 1: calibrate a single gray stellar offset from a
        # separate free-quadratic fit to the existing coarse channel set.
        if (
            ld_prior_mode == 'sing'
            and str(flags.get('ld_sing_offset', 'fit')).strip().lower() == 'fit'
            and not flags.get('ld_sing_offset_path')
        ):
            gray_fingerprint_inputs = {
                'kind': 'sing_gray_offset_v2_physical_ld',
                'time': np.asarray(time_lr),
                'flux': np.asarray(flux_lr),
                'flux_err': np.asarray(flux_err_lr),
                'wavelength': np.asarray(data.wavelengths_lr),
                'wavelength_err': np.asarray(data.wavelengths_err_lr),
                'model_c': np.asarray(sing_model_c_lr),
                'mu_min': float(stellar_cfg.get('ld_mu_min', 0.2)),
                'sampler': 'independent_nuts_laplace_metric_ta099_depth10',
                'geometry_handoff': wl_geometry_handoff['artifact_fingerprint_sha256'],
            }
            gray_digest = _science_artifact_fingerprint(
                'sing_gray_offset', gray_fingerprint_inputs
            )
            gray_artifact_path = os.path.join(
                output_dir,
                f"{instrument_full_str}_{lr_bin_str}_sing_gray_offset_{gray_digest[:12]}.json",
            )
            fitted_offsets = None
            if os.path.exists(gray_artifact_path):
                try:
                    with open(gray_artifact_path) as handle:
                        cached_gray = json.load(handle)
                    if cached_gray.get('fingerprint_sha256') == hashlib.sha256(
                        json.dumps(gray_fingerprint_inputs, sort_keys=True, default=str).encode()
                    ).hexdigest():
                        fitted_offsets = cached_gray
                        print(f"[Sing LD] reusing fitted gray offsets from {gray_artifact_path}", flush=True)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    fitted_offsets = None
            if fitted_offsets is None:
                print("[Sing LD] running broad physical (l, delta) coarse calibration pass", flush=True)
                gray_builder_kwargs = dict(lr_model_builder_kwargs)
                gray_builder_kwargs['ld_mode'] = 'sing_free'
                gray_model = _build_spectroscopic_model(
                    create_vectorized_model, **gray_builder_kwargs
                )
                gray_init = dict(init_params_lr)
                gray_init.pop('limb_l', None)
                gray_init.pop('limb_delta', None)
                gray_init.pop('u', None)
                gray_l_init, gray_delta_init = quadratic_to_sing(
                    np.asarray(sing_model_c_lr)[:, 0],
                    np.asarray(sing_model_c_lr)[:, 1],
                )
                gray_init['limb_l'] = jnp.asarray(gray_l_init)
                gray_init['limb_delta'] = jnp.asarray(gray_delta_init)
                gray_args = dict(model_run_args_lr)
                gray_args.pop('mu_u_ld', None)
                gray_args.pop('sigma_u_ld', None)
                gray_prefix = _harmonica_checkpoint_prefix(
                    f"{instrument_full_str}_{lr_bin_str}_sing_gray_free_{gray_digest[:12]}",
                    transit_engine,
                    harmonica_spectro_parameterization,
                )
                gray_mcmc = {
                    'num_warmup': int(flags.get('ld_sing_calibration_warmup', 1000)),
                    'num_samples': int(flags.get('ld_sing_calibration_samples', 1000)),
                }
                gray_nuts_kwargs = dict(jaxoplanet_lr_nuts_kwargs)
                gray_nuts_kwargs.update(
                    mass_matrix='laplace',
                    laplace_hessian_method='finite_difference',
                    laplace_warmup=1000,
                    laplace_target_accept=0.99,
                    laplace_max_tree_depth=10,
                    target_accept_prob=0.99,
                    max_tree_depth=10,
                )
                try:
                    gray_samples = _run_sampling_stage(
                        gray_model,
                        jax.random.fold_in(key_mcmc_lr, 91027),
                        time_lr, flux_err_lr, flux_lr, gray_init,
                        nuts_kwargs=gray_nuts_kwargs,
                        mcmc_kwargs=gray_mcmc,
                        use_chunked=True,
                        chunk_size=(vmap_chunk_size_lr or num_lcs_lr),
                        chunk_mode='serial',
                        output_dir=output_dir,
                        checkpoint_prefix=gray_prefix,
                        sampler_backend='independent_nuts',
                        laplace_is_kwargs=None,
                        channel_varying_kwargs=(JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS if transit_engine == 'jaxoplanet' else HARMONICA_CHANNEL_VARYING_MODEL_KWARGS),
                        checkpoint_signature={
                            'stage': 'sing_gray_offset_calibration',
                            'ld_profile': 'quadratic',
                            'ld_mode': 'sing_free',
                            'fingerprint': gray_digest,
                        },
                        **gray_args,
                    )
                    gray_diag_paths = sorted(glob.glob(os.path.join(
                        output_dir, 'chunks', f"{gray_prefix}*_chunk_*.diagnostics.json"
                    )))
                    gray_ok, gray_diagnostics = _assess_sing_gray_fit(
                        gray_samples,
                        gray_diag_paths,
                        min_ess=float(flags.get('ld_sing_calibration_min_ess', 100.0)),
                    )
                    if not gray_ok:
                        raise RuntimeError(f"gray calibration diagnostics failed: {gray_diagnostics}")
                    fitted_c = np.column_stack((
                        np.nanmedian(np.asarray(gray_samples['c1'], dtype=float), axis=0),
                        np.nanmedian(np.asarray(gray_samples['c2'], dtype=float), axis=0),
                    ))
                    fitted_c_sigma = np.column_stack((
                        np.nanstd(np.asarray(gray_samples['c1'], dtype=float), axis=0, ddof=1),
                        np.nanstd(np.asarray(gray_samples['c2'], dtype=float), axis=0, ddof=1),
                    ))
                    fitted_offsets = estimate_gray_offset(
                        fitted_c, np.asarray(sing_model_c_lr), fitted_c_sigma
                    )
                    fitted_offsets['diagnostics'] = gray_diagnostics
                    fitted_offsets['tabulated_l'] = SING_TABULATED_OFFSET['l']
                    fitted_offsets['tabulated_delta'] = SING_TABULATED_OFFSET['delta']
                    write_offset_artifact(
                        gray_artifact_path, fitted_offsets, gray_fingerprint_inputs
                    )
                except Exception as error:
                    print(
                        f"[Sing LD] WARNING: free gray calibration unavailable or failed ({error}); "
                        "falling back to tabulated Stagger offsets.",
                        flush=True,
                    )
                    fitted_offsets = None
            if fitted_offsets is not None:
                print(
                    "[Sing LD] gray offsets fitted vs tabulated: "
                    f"delta_l={float(fitted_offsets['l']):+.6f} +/- {float(fitted_offsets['l_sigma']):.6f} "
                    f"vs {SING_TABULATED_OFFSET['l']:+.6f}; "
                    f"delta_delta={float(fitted_offsets['delta']):+.6f} +/- {float(fitted_offsets['delta_sigma']):.6f} "
                    f"vs {SING_TABULATED_OFFSET['delta']:+.6f}",
                    flush=True,
                )
                flags['ld_sing_offset_path'] = gray_artifact_path
                U_mu_lr, U_sigma_lr = build_sing_ld_prior(
                    sing_model_c_lr, flags, stellar_cfg,
                    offsets_override=fitted_offsets,
                )
                model_run_args_lr['mu_u_ld'] = U_mu_lr
                model_run_args_lr['sigma_u_ld'] = U_sigma_lr
                init_params_lr['limb_l'] = jnp.asarray(U_mu_lr)[:, 0]
                init_params_lr['limb_delta'] = jnp.asarray(U_mu_lr)[:, 1]

        lr_stage_sampler_kwargs = (
            harmonica_lr_nuts_kwargs
            if transit_engine == 'harmonica'
            else jaxoplanet_lr_nuts_kwargs
        )
        vmap_chunk_size_lr = _resolve_stage_vmap_width(
            flags,
            'lowres',
            vmap_chunk_size_lr,
            sampler_backend=spectro_sampler,
            trend_inference=(
                'gaussian_marginalized'
                if lr_trend_mode == 'gaussian_marginalized'
                else 'sampled_uniform'
            ),
            num_cadences=int(time_lr.size),
            active_transit_cadences=(
                int(len(lr_transit_window_indices))
                if lr_transit_window_indices is not None
                else int(time_lr.size)
            ),
            mcmc_kwargs=lowres_mcmc_kwargs,
            sampler_kwargs=lr_stage_sampler_kwargs,
            transit_engine=transit_engine,
            ld_profile=ld_profile,
            ld_mode=lr_ld_mode,
            detrend_type=detrend_type_multiwave,
            param_method=param_method,
            n_planets=n_planets,
            transit_window=transit_window_optimization,
        )

        lr_channel_batch_plan = _resolve_stage_channel_batch_plan(
            flags,
            'lowres',
            num_lcs_lr,
        )
        if (
            lr_channel_batch_plan is not None
            and vmap_chunk_size_lr is not None
            and lr_channel_batch_plan.nominal_width != vmap_chunk_size_lr
        ):
            raise ValueError(
                "The lowres batch plan nominal width "
                f"({lr_channel_batch_plan.nominal_width}) does not match the "
                f"configured/measured resident width ({vmap_chunk_size_lr})."
            )

        samples_lr = _run_sampling_stage(
            lr_model_for_run,
            key_mcmc_lr,
            time_lr,
            flux_err_lr,
            flux_lr,
            init_params_lr,
            nuts_kwargs=lr_stage_sampler_kwargs,
            mcmc_kwargs=lowres_mcmc_kwargs,
            use_chunked=(
                spectro_sampling_mode == 'independent'
                or vmap_chunk_size_lr is not None
                or spectro_sampler in {
                    'independent_nuts', 'independent_hmc', 'laplace_is'
                }
                or lr_channel_batch_plan is not None
            ),
            chunk_size=vmap_chunk_size_lr,
            chunk_mode=chunk_mode,
            parallel_job_count=chunk_parallel_job_count,
            parallel_job_index=chunk_parallel_job_index,
            output_dir=output_dir,
            checkpoint_prefix=_harmonica_checkpoint_prefix(
                f"{instrument_full_str}_{lr_bin_str}",
                transit_engine,
                harmonica_spectro_parameterization,
            ),
            checkpoint_signature={
                'stage': 'low_resolution',
                'transit_engine': transit_engine,
                'ld_profile': ld_profile,
                'ld_mode': lr_ld_mode,
                'trend_inference': trend_inference,
                'detrend_type': detrend_type_multiwave,
                'param_method': param_method,
                'whitelight_geometry_estimator': whitelight_geometry_estimator,
                'whitelight_geometry_handoff_fingerprint': (
                    wl_geometry_handoff['artifact_fingerprint_sha256']
                ),
                'jaxoplanet_kernel': jaxoplanet_kernel,
                'transit_window_optimization': transit_window_optimization,
                'harmonica_max_order': max_harmonic_order,
                'harmonica_spectro_parameterization': harmonica_spectro_parameterization,
                'harmonica_spectro_fit_jitter': harmonica_spectro_fit_jitter,
                'harmonica_spectro_odd_frac_sigma': harmonica_spectro_odd_frac_sigma,
                'laplace_is_kwargs': (
                    lowres_laplace_is_kwargs
                    if spectro_sampler == 'laplace_is' else None
                ),
            },
            channel_batch_plan=lr_channel_batch_plan,
            gradient_diagnostic_mode=spectro_gradient_diagnostic,
            gradient_diagnostic_strict=spectro_gradient_diagnostic_strict,
            sampler_backend=spectro_sampler,
            laplace_is_kwargs=(
                lowres_laplace_is_kwargs
                if spectro_sampler == 'laplace_is' else None
            ),
            channel_varying_kwargs=(
                JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS
                if transit_engine == 'jaxoplanet'
                else HARMONICA_CHANNEL_VARYING_MODEL_KWARGS
            ),
            spectro_min_depth_ess=float(flags.get('spectro_min_depth_ess', 400)),
            spectro_max_divergences=int(flags.get('spectro_max_divergences', 0)),
            compile_box=bool(flags.get('compile_box', False)),
            dump_metadata={
                'config_path': os.path.abspath(args.config),
                'stage_kind': 'low_resolution',
                'stage_label': (
                    f"{instrument_full_str}_{lr_bin_str}_low_resolution"
                ),
                'instrument': instrument,
                'instrument_label': instrument_full_str,
                'output_dir': os.path.abspath(output_dir),
                'wavelength': np.asarray(data.wavelengths_lr),
                'wavelength_err': np.asarray(data.wavelengths_err_lr),
                'active_window_cadences': (
                    int(len(lr_transit_window_indices))
                    if lr_transit_window_indices is not None
                    else int(time_lr.size)
                ),
                'transit_engine': transit_engine,
            },
            **model_run_args_lr,
        )
        if samples_lr is None:
            print("\nLow-resolution chunk job complete. Exiting before final aggregation.")
            return
        if trend_names_lr is not None:
            samples_lr = materialize_marginalized_trend_samples(
                samples_lr,
                trend_names_lr,
                key_map_lr,
            )

        if ld_profile == 'quadratic' and 'u1' in samples_lr and 'u2' in samples_lr:
            ld_u_lr = np.stack(
                (np.array(samples_lr['u1']), np.array(samples_lr['u2'])),
                axis=-1,
            )
        elif 'u' in samples_lr:
            ld_u_lr = np.array(samples_lr["u"])
        elif ld_profile == 'power2' and 'c1' in samples_lr:
            c1_med_lr = jnp.nanmedian(samples_lr['c1'], axis=0)
            c2_med_lr = jnp.nanmedian(samples_lr['c2'], axis=0)
            ld_u_lr = jax.vmap(compute_u_from_c)(c1_med_lr, c2_med_lr)
            ld_u_lr = np.array(ld_u_lr)

        if detrend_type_multiwave != 'none':
            trend_c_lr = np.array(samples_lr["c"])
            if 'v' in samples_lr: trend_v_lr = np.array(samples_lr["v"])
        if 'explinear' in detrend_type_multiwave:
            trend_A_lr = np.array(samples_lr["A"])
            trend_tau_lr = np.array(samples_lr["tau"])

        map_params_lr = {
            "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
            "rors": jnp.nanmedian(
                _harmonica_model_radius_samples(
                    samples_lr, harmonica_spectro_parameterization
                )
                if transit_engine == 'harmonica'
                else samples_lr["rors"],
                axis=0,
            ),
            "period": PERIOD_FIXED,
        }
        
        if ld_profile == 'power2':
            c1_med = jnp.nanmedian(samples_lr['c1'], axis=0)
            c2_med = jnp.nanmedian(samples_lr['c2'], axis=0)
            map_params_lr['c1'] = c1_med
            map_params_lr['c2'] = c2_med
            map_params_lr['u'] = jax.vmap(compute_u_from_c)(c1_med, c2_med)
            if transit_engine == 'harmonica':
                map_params_lr['c_ld'] = c1_med
                map_params_lr['alpha_ld'] = c2_med
        else:
            quadratic_u_med = _posterior_quadratic_ld_median(
                samples_lr, U_mu_lr
            )
            map_params_lr['u'] = quadratic_u_med
            if transit_engine == 'harmonica':
                map_params_lr['u1_ld'] = quadratic_u_med[..., 0]
                map_params_lr['u2_ld'] = quadratic_u_med[..., 1]

        if transit_engine == 'harmonica':
            map_params_lr = _augment_harmonica_params(
                map_params_lr,
                a_rs=A_RS_BASE,
                ecc=HARMONICA_ECC,
                omega=HARMONICA_OMEGA,
                a1=jnp.nanmedian(samples_lr["a1"], axis=0) if "a1" in samples_lr else None,
                a3=jnp.nanmedian(samples_lr["a3"], axis=0) if "a3" in samples_lr else None,
                a5=jnp.nanmedian(samples_lr["a5"], axis=0) if "a5" in samples_lr else None,
            )
        elif param_method == 'a_rs':
            map_params_lr.update({
                "a_rs": A_RS_BASE,
                "ecc": HARMONICA_ECC,
                "omega": HARMONICA_OMEGA,
            })

        map_params_lr.update({k: jnp.nanmedian(samples_lr[k], axis=0) for k in TREND_PARAMS if k in samples_lr})
        if transit_engine == 'jaxoplanet':
            map_params_lr = _attach_jaxoplanet_eval_metadata(
                map_params_lr,
                time_lr,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
                transit_window_optimization=transit_window_optimization,
            )

        selected_kernel = resolve_detrend_kernel(detrend_type_multiwave)
        in_axes_map = {'rors': 0}
        in_axes_map.update({
            name: 0 for name in ('u', 'c1', 'c2') if name in map_params_lr
        })
        if transit_engine == 'harmonica':
            if ld_profile == 'power2':
                in_axes_map.update({'c_ld': 0, 'alpha_ld': 0})
            else:
                in_axes_map.update({'u1_ld': 0, 'u2_ld': 0})
            in_axes_map.update({name: 0 for name in HARMONICA_ODD_HARMONICS if name in map_params_lr})
        in_axes_map.update({k: 0 for k in TREND_PARAMS if k in map_params_lr})
        
        final_in_axes = {k: in_axes_map.get(k, None) for k in map_params_lr.keys()}

        # Evaluate channels sequentially to prevent Out-Of-Memory (OOM) errors.
        @jax.jit
        def eval_channel_lr(channel_params, t_val, *extra_args):
            if transit_engine == 'jaxoplanet':
                channel_params = {
                    **channel_params,
                    "_jaxoplanet_kernel": jaxoplanet_kernel,
                    "_ld_profile": ld_profile,
                }
            return selected_kernel(channel_params, t_val, *extra_args)

        model_all_list = []
        num_lcs_lr = flux_lr.shape[0]
        for i in range(num_lcs_lr):
            channel_params = {}
            for k, v in map_params_lr.items():
                if k in _JAXOPLANET_STATIC_EVAL_KEYS:
                    continue
                if final_in_axes[k] == 0:
                    channel_params[k] = v[i]
                else:
                    channel_params[k] = v
            
            if 'gp_spectroscopic' in detrend_type_multiwave:
                ch_model = eval_channel_lr(channel_params, time_lr, gp_trend)
            elif '2spot_spectroscopic' in detrend_type_multiwave:
                ch_model = eval_channel_lr(channel_params, time_lr, spot_trend, spot_trend2)
            elif _has_single_spot_spectroscopic(detrend_type_multiwave):
                if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
                    ch_model = eval_channel_lr(channel_params, time_lr, spot_trend, jump_trend)
                else:
                    ch_model = eval_channel_lr(channel_params, time_lr, spot_trend)
            elif 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
                ch_model = eval_channel_lr(channel_params, time_lr, jump_trend)
            elif 'explinear_spectroscopic' in detrend_type_multiwave:
                ch_model = eval_channel_lr(channel_params, time_lr, exp_trend_lr)
            else:
                ch_model = eval_channel_lr(channel_params, time_lr)
                
            model_all_list.append(ch_model)
            
        model_all = jnp.stack(model_all_list, axis=0)

        residuals = flux_lr - model_all
        _lr_dt_sec = float(np.nanmedian(np.diff(np.array(time_lr)))) * 86400.0
        plot_noise_binning_robust(
            residuals, _lr_dt_sec,
            f"{output_dir}/25_{lr_artifact_stem}_noisebin.png",
            title=f"Low-Res Noise Binning ({lr_bin_str})",
        )
        save_noise_binning_data(
            residuals,
            f"{output_dir}/25_{lr_artifact_stem}_noisebin.csv",
        )

        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)
        point_mask = np.abs(residuals - medians) > spectroscopic_sigma * sigmas
        time_mask = np.any(point_mask, axis=0)
        valid = ~time_mask
        _atomic_save_npy(lr_mask_path, time_mask)
        gp_trend_lr = gp_trend
        spot_trend_lr = spot_trend
        spot_trend2_lr = spot_trend2
        jump_trend_lr = jump_trend
        exp_trend_lr_save = (
            exp_trend_lr if 'explinear_spectroscopic' in detrend_type_multiwave else None
        )
        time_lr = time_lr[valid]
        flux_lr = flux_lr[:, valid]
        flux_err_lr = flux_err_lr[:, valid]
        if gp_trend_lr is not None: gp_trend_lr = gp_trend_lr[valid]
        if spot_trend_lr is not None: spot_trend_lr = spot_trend_lr[valid]
        if spot_trend2_lr is not None: spot_trend2_lr = spot_trend2_lr[valid]
        if jump_trend_lr is not None: jump_trend_lr = jump_trend_lr[valid]
        if exp_trend_lr_save is not None: exp_trend_lr_save = exp_trend_lr_save[valid]
        if transit_engine == 'jaxoplanet':
            map_params_lr = _attach_jaxoplanet_eval_metadata(
                map_params_lr,
                time_lr,
                jaxoplanet_kernel=jaxoplanet_kernel,
                ld_profile=ld_profile,
                transit_window_optimization=transit_window_optimization,
            )
        
        print("Plotting low-resolution fits and residuals...")
        median_total_error_lr = np.nanmedian(samples_lr['total_error'], axis=0)
        plot_wavelength_offset_summary(time_lr, flux_lr, median_total_error_lr, data.wavelengths_lr,
                                     map_params_lr, {
                                         "period": PERIOD_FIXED,
                                         "transit_engine": transit_engine,
                                         "param_method": param_method,
                                     },
                                     f"{output_dir}/22_{lr_artifact_stem}_summary.png",
                                     detrend_type=detrend_type_multiwave, gp_trend=gp_trend_lr, spot_trend=spot_trend_lr, spot_trend2=spot_trend2_lr, jump_trend=jump_trend_lr, exp_trend=exp_trend_lr_save)

        poly_orders = [1, 2, 3, 4]
        wl_lr = np.array(data.wavelengths_lr)

        if detrending_type != 'none':
            print("Fitting polynomials to trend coefficients...")
            best_poly_coeffs_c, best_order_c, _ = fit_polynomial(wl_lr, trend_c_lr, poly_orders)
            if 'v' in samples_lr:
                best_poly_coeffs_v, best_order_v, _ = fit_polynomial(wl_lr, trend_v_lr, poly_orders)
                plot_poly_fit(wl_lr, trend_v_lr, best_poly_coeffs_v, best_order_v, "Wavelength", "v", "Trend Slope v", f"{output_dir}/2opt_v.png")

        if interpolate_ld:
            print("Fitting polynomials to limb darkening coefficients...")
            if ld_profile == 'power2':
                c1_lr = np.array(samples_lr['c1'])
                c2_lr = np.array(samples_lr['c2'])
                best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, c1_lr, poly_orders)
                best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, c2_lr, poly_orders)
                plot_poly_fit(wl_lr, c1_lr, best_poly_coeffs_u1, best_order_u1, "Wavelength", "c1", "Limb Darkening c1", f"{output_dir}/2opt_c1.png")
                plot_poly_fit(wl_lr, c2_lr, best_poly_coeffs_u2, best_order_u2, "Wavelength", "c2", "Limb Darkening c2", f"{output_dir}/2opt_c2.png")
            else:
                best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 0], poly_orders)
                best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 1], poly_orders)
                plot_poly_fit(wl_lr, ld_u_lr[:, :, 0], best_poly_coeffs_u1, best_order_u1, "Wavelength", "u1", "Limb Darkening u1", f"{output_dir}/2opt_u1.png")
                plot_poly_fit(wl_lr, ld_u_lr[:, :, 1], best_poly_coeffs_u2, best_order_u2, "Wavelength", "u2", "Limb Darkening u2", f"{output_dir}/2opt_u2.png")

        poly_save = {}
        if best_poly_coeffs_c is not None:
            poly_save["coeffs_c"] = best_poly_coeffs_c
        if best_poly_coeffs_v is not None:
            poly_save["coeffs_v"] = best_poly_coeffs_v
        if best_poly_coeffs_u1 is not None:
            poly_save["coeffs_u1"] = best_poly_coeffs_u1
        if best_poly_coeffs_u2 is not None:
            poly_save["coeffs_u2"] = best_poly_coeffs_u2
        if poly_save:
            _atomic_savez(poly_coeffs_path, **poly_save)

        plot_transmission_spectrum(wl_lr, samples_lr["rors"], f"{output_dir}/24_{lr_artifact_stem}_spectrum")
        save_results(wl_lr, data.wavelengths_err_lr, samples_lr, f"{output_dir}/{lr_artifact_stem}.csv")
        save_detailed_fit_results(time_lr, flux_lr, flux_err_lr, data.wavelengths_lr, data.wavelengths_err_lr, samples_lr, map_params_lr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{lr_artifact_stem}", median_total_error_lr, gp_trend=gp_trend_lr, spot_trend=spot_trend_lr, jump_trend=jump_trend_lr)
        if transit_engine == 'harmonica' and _has_harmonica_odd_samples(samples_lr):
            save_harmonica_limb_products(
                wavelengths=wl_lr,
                wavelength_err=data.wavelengths_err_lr,
                rors_samples=_harmonica_model_radius_samples(
                    samples_lr, harmonica_spectro_parameterization
                ),
                harmonic_samples=_harmonica_sample_payload(samples_lr),
                csv_path=lr_limb_csv_path,
                limb_spectrum_path=f"{output_dir}/26_{lr_artifact_stem}_limb_spectra.png",
                transmission_strings_path=f"{output_dir}/27_{lr_artifact_stem}_transmission_strings.png",
                posterior_strings_path=f"{output_dir}/28_{lr_artifact_stem}_transmission_string_posterior.png",
                posterior_samples_path=lr_limb_samples_path,
                title_prefix=f"{planet_str} - {lr_bin_str}",
            )
        _write_science_artifact_manifest(
            lr_manifest_path, "low_resolution", lr_artifact_fingerprint
        )

    if analysis_stage == 'prep':
        print("\nPrep stage complete. Skipping high-resolution analysis.")
        return

    print(f"\n--- Running High-Resolution Analysis (Binned to {hr_bin_str}) ---")
    time_hr = jnp.array(data.time[spec_good_mask])
    flux_hr = jnp.array(data.flux_hr[:, spec_good_mask])
    flux_err_hr = jnp.array(data.flux_err_hr[:, spec_good_mask])

    if 'gp' in detrending_type:
        gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
        gp_trend_raw = gp_df['gp_trend'].values
        if len(gp_trend_raw) == len(wl_mad_mask):
            gp_trend_raw = gp_trend_raw[~wl_mad_mask]
        gp_trend = jnp.array(_align_trend_to_time(gp_trend_raw, wl_time_good, np.array(time_hr)))
    else:
        gp_trend = None
    detrend_type_multiwave = _spectro_detrend_type(
        detrending_type, spectro_fixed_timescale_trends
    )

    if valid is not None:
        if spot_trend is not None and len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        if spot_trend2 is not None and len(spot_trend2) != len(time_hr):
            spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_hr))
        if jump_trend is not None and len(jump_trend) != len(time_hr):
            jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_hr))
        if gp_trend is not None and len(gp_trend) != len(time_hr):
            gp_trend = _align_trend_to_time(gp_trend, wl_time_good, np.array(time_hr))

        time_hr = time_hr[valid]
        flux_hr = flux_hr[:, valid]
        flux_err_hr = flux_err_hr[:, valid]
        if gp_trend is not None: gp_trend = gp_trend[valid]
        if spot_trend is not None: spot_trend = spot_trend[valid]
        if spot_trend2 is not None: spot_trend2 = spot_trend2[valid]
        if jump_trend is not None: jump_trend = jump_trend[valid]

    num_lcs_hr = flux_err_hr.shape[0]
    DEPTHS_BASE_HR = jnp.tile(DEPTH_BASE, (num_lcs_hr, 1))
    
    hr_ld_mode = ld_prior_mode
    if flags.get('interpolate_ld', False): hr_ld_mode = 'interpolated'
    if flags.get('interpolate_trend', False):
        hr_trend_mode = 'fixed'
    elif trend_inference == 'gaussian_marginalized':
        hr_trend_mode = 'gaussian_marginalized'
    else:
        hr_trend_mode = 'free'

    model_run_args_hr = {}
    wl_hr = np.array(data.wavelengths_hr)

    if hr_ld_mode == 'interpolated':
        u1_interp_hr = np.polyval(best_poly_coeffs_u1, wl_hr)
        u2_interp_hr = np.polyval(best_poly_coeffs_u2, wl_hr)
        
        if ld_profile == 'power2':
            if transit_engine == 'harmonica':
                ld_interpolated_hr = jnp.asarray(
                    np.column_stack((u1_interp_hr, u2_interp_hr))
                )
            else:
                ld_interpolated_hr = jax.vmap(compute_u_from_c)(
                    jnp.array(u1_interp_hr), jnp.array(u2_interp_hr)
                )
        else:
            ld_interpolated_hr = jnp.array(np.column_stack((u1_interp_hr, u2_interp_hr)))
            
        model_run_args_hr['ld_interpolated'] = ld_interpolated_hr
    elif hr_ld_mode in {'fixed', 'widegaussian', 'informed', 'sing', 'uniform'}:
        if hr_custom_ld_path:
            print(f"Using custom high-resolution LD curve from {hr_custom_ld_path} with smoothing window={hr_custom_ld_smooth_window}")
            U_mu_hr_init = _load_custom_power2_ld_curve(
                hr_custom_ld_path,
                wl_hr,
                smooth_window=hr_custom_ld_smooth_window,
            )
            U_sigma_hr_init = None
            applied_ld_df = pd.DataFrame({
                'wavelength': np.asarray(wl_hr, dtype=float),
                'c1': np.asarray(U_mu_hr_init[:, 0], dtype=float),
                'c2': np.asarray(U_mu_hr_init[:, 1], dtype=float),
            })
            applied_ld_df.to_csv(f"{output_dir}/{instrument_full_str}_{hr_bin_str}_applied_custom_ld.csv", index=False)
        elif ld_prior_mode == 'informed':
            U_mu_hr_init, U_sigma_hr_init = get_or_build_power2_ld_prior(
                stellar_cfg, wl_hr, data.wavelengths_err_hr, instrument,
                order=order if instrument == 'NIRISS/SOSS' else None,
                output_dir=output_dir,
                cache_label=hr_bin_str,
            )
        else:
            if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H', 'MIRI/LRS']:
                U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
            elif instrument == 'NIRISS/SOSS':
                U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order, ld_profile=ld_profile, ld_mu_min=(stellar_cfg.get('ld_mu_min', 0.2) if ld_prior_mode == 'sing' else None))
            U_sigma_hr_init = None
            if ld_prior_mode == 'sing':
                U_mu_hr_init, U_sigma_hr_init = build_sing_ld_prior(U_mu_hr_init, flags, stellar_cfg)
        if hr_ld_mode == 'fixed':
            model_run_args_hr['ld_fixed'] = U_mu_hr_init
        elif hr_ld_mode in {'widegaussian', 'informed', 'sing'}:
            model_run_args_hr['mu_u_ld'] = U_mu_hr_init
            if ((ld_profile == 'power2' and hr_ld_mode == 'informed') or hr_ld_mode == 'sing') and U_sigma_hr_init is not None:
                model_run_args_hr['sigma_u_ld'] = U_sigma_hr_init
        elif hr_ld_mode == 'uniform':
            pass
        else:
            raise ValueError(f"Unknown ld_prior mode: {hr_ld_mode}")

    if hr_trend_mode == 'fixed':
        c_interp_hr = np.polyval(best_poly_coeffs_c, wl_hr)
        v_interp_hr = np.polyval(best_poly_coeffs_v, wl_hr)
        trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr))
        model_run_args_hr['trend_fixed'] = jnp.array(trend_fixed_hr)

    model_run_args_hr['mu_duration'] = DURATION_BASE
    model_run_args_hr['mu_t0'] = T0_BASE
    model_run_args_hr['mu_b'] = B_BASE
    model_run_args_hr['mu_depths'] = DEPTHS_BASE_HR
    model_run_args_hr['PERIOD'] = PERIOD_FIXED
    if transit_engine == 'jaxoplanet':
        model_run_args_hr['precomputed_yerr_per_lc'] = jnp.nanmedian(
            flux_err_hr, axis=1
        )
    if transit_engine == 'harmonica':
        model_run_args_hr['mu_cos_i'] = COSI_BASE
        model_run_args_hr['harmonica_a_rs'] = A_RS_BASE
        model_run_args_hr['harmonica_ecc'] = HARMONICA_ECC
        model_run_args_hr['harmonica_omega'] = HARMONICA_OMEGA
    elif param_method == 'a_rs':
        model_run_args_hr['mu_a_rs'] = A_RS_BASE
        model_run_args_hr['mu_ecc'] = HARMONICA_ECC
        model_run_args_hr['mu_omega'] = HARMONICA_OMEGA

    init_params_hr = { "rors": jnp.tile(RORS_BASE, (num_lcs_hr, 1)), "u": U_mu_hr_init if hr_ld_mode!='interpolated' else ld_interpolated_hr }
    if hr_ld_mode == 'sing':
        init_params_hr.pop('u', None)
        init_params_hr['limb_l'] = jnp.asarray(U_mu_hr_init)[:, 0]
        init_params_hr['limb_delta'] = jnp.asarray(U_mu_hr_init)[:, 1]
    if ld_profile == 'power2' and hr_ld_mode not in {'fixed', 'interpolated'}:
        init_params_hr['c1'] = jnp.asarray(U_mu_hr_init)[:, 0]
        init_params_hr['c2'] = jnp.asarray(U_mu_hr_init)[:, 1]
    if transit_engine == 'harmonica':
        if harmonica_spectro_parameterization == 'delta_r':
            init_val = (
                bestfit_params_wl_df['a1'].values[0]
                if 'a1' in bestfit_params_wl_df.columns
                else HARMONICA_INIT_ODD_COEFF
            )
            init_delta_r = jnp.atleast_1d(
                jnp.asarray(_harmonica_coeff_array_to_delta_r(init_val))
            )
            init_params_hr['delta_r'] = jnp.tile(
                init_delta_r[None, :],
                (num_lcs_hr, 1),
            )
        elif harmonica_spectro_parameterization == 'half_area':
            init_val = (
                bestfit_params_wl_df['a1'].values
                if 'a1' in bestfit_params_wl_df.columns
                else np.full_like(RORS_BASE, HARMONICA_INIT_ODD_COEFF)
            )
            init_area_radius, init_q = (
                _harmonica_coefficients_to_half_area_init(
                    RORS_BASE, init_val
                )
            )
            init_params_hr['rors'] = jnp.tile(
                jnp.atleast_1d(jnp.asarray(init_area_radius))[None, :],
                (num_lcs_hr, 1),
            )
            init_params_hr['q'] = jnp.tile(
                jnp.atleast_1d(jnp.asarray(init_q))[None, :],
                (num_lcs_hr, 1),
            )
        else:
            for harmonic_name in HARMONICA_ODD_HARMONICS:
                init_val = bestfit_params_wl_df[harmonic_name].values[0] if harmonic_name in bestfit_params_wl_df.columns else HARMONICA_INIT_ODD_COEFF
                init_frac = _harmonica_coeff_array_to_frac(init_val, RORS_BASE)
                init_params_hr[_harmonica_frac_site(harmonic_name)] = jnp.tile(
                    jnp.asarray(init_frac)[None, :],
                    (num_lcs_hr, 1),
                )
    if hr_trend_mode in {'free', 'gaussian_marginalized'}:
        if detrend_type_multiwave != 'none':
            if best_poly_coeffs_c is not None and np.ndim(best_poly_coeffs_c) > 0:
                init_params_hr["c"] = np.polyval(best_poly_coeffs_c, wl_hr)
            else:
                init_params_hr["c"] = jnp.full(num_lcs_hr, bestfit_params_wl_df['c'].values[0])
            if 'v' in bestfit_params_wl_df.columns:
                 if best_poly_coeffs_v is not None and np.ndim(best_poly_coeffs_v) > 0:
                     init_params_hr["v"] = np.polyval(best_poly_coeffs_v, wl_hr)
                 else:
                     init_params_hr["v"] = jnp.full(num_lcs_hr, bestfit_params_wl_df['v'].values[0])
        if 'explinear' in detrend_type_multiwave:
            init_params_hr['A'] = jnp.full(
                num_lcs_hr, bestfit_params_wl_df['A'].values[0]
            )
            if 'explinear_spectroscopic' not in detrend_type_multiwave:
                init_params_hr['log_tau'] = jnp.full(
                    num_lcs_hr,
                    jnp.log(bestfit_params_wl_df['tau'].values[0]),
                )

    hr_transit_window_indices = None
    if (
        transit_engine == 'jaxoplanet'
        and transit_window_optimization == 'auto'
        and param_method == 'duration'
    ):
        hr_transit_window_indices = build_transit_window_indices(
            np.asarray(time_hr),
            np.asarray(PERIOD_FIXED),
            np.asarray(T0_BASE),
            np.asarray(DURATION_BASE),
        )
        print(
            "Jaxoplanet transit window (high-res): "
            f"{len(hr_transit_window_indices)}/{time_hr.size} active cadences."
        )

    hr_model_builder_kwargs = {
        'detrend_type': detrend_type_multiwave,
        'ld_mode': hr_ld_mode,
        'trend_mode': hr_trend_mode,
        'n_planets': n_planets,
        **(
            {'transit_window_indices': hr_transit_window_indices}
            if transit_engine == 'jaxoplanet' else {}
        ),
        **_engine_spectro_kw,
    }
    hr_model_for_run = _build_spectroscopic_model(
        create_vectorized_model,
        **hr_model_builder_kwargs,
    )
    
    if 'gp_spectroscopic' in detrend_type_multiwave:
        model_run_args_hr['gp_trend'] = gp_trend
        init_params_hr['A_gp'] = jnp.ones(num_lcs_hr)
    if '2spot_spectroscopic' in detrend_type_multiwave:
        if spot_trend is None or spot_trend2 is None:
            raise ValueError("2spot_spectroscopic requires WL spot and spot2 trends.")
        if len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        if len(spot_trend2) != len(time_hr):
            spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_hr))
        model_run_args_hr['spot_trend'] = spot_trend
        model_run_args_hr['spot_trend2'] = spot_trend2
        init_params_hr['A_spot'] = jnp.ones(num_lcs_hr)
        init_params_hr['A_spot2'] = jnp.ones(num_lcs_hr)
    elif _has_single_spot_spectroscopic(detrend_type_multiwave):
        if spot_trend is None:
            raise ValueError(f"{detrend_type_multiwave} requires WL spot to build spot_trend.")
        if spot_trend is not None and len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        model_run_args_hr['spot_trend'] = spot_trend
        init_params_hr['A_spot'] = jnp.ones(num_lcs_hr)
    if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
        if jump_trend is None:
            raise ValueError("linear_discontinuity_spectroscopic requires WL linear_discontinuity to build jump_trend.")
        if jump_trend is not None and len(jump_trend) != len(time_hr):
            jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_hr))
        model_run_args_hr['jump_trend'] = jump_trend
        init_params_hr['A_jump'] = jnp.ones(num_lcs_hr)
    if 'explinear_spectroscopic' in detrend_type_multiwave:
        exp_trend_hr = _align_trend_to_time(
            exp_trend, wl_time_good, np.asarray(time_hr)
        )
        model_run_args_hr['exp_trend'] = exp_trend_hr
        model_run_args_hr['fixed_tau'] = fixed_tau_spectro

    trend_names_hr = None
    if hr_trend_mode == 'gaussian_marginalized':
        (
            model_run_args_hr['trend_prior_mean'],
            model_run_args_hr['trend_prior_scale'],
            trend_names_hr,
        ) = _build_gaussian_trend_prior(
            detrend_type_multiwave,
            int(num_lcs_hr),
            init_params_hr,
            flags,
        )
        for trend_name in trend_names_hr:
            init_params_hr.pop(trend_name, None)

    hr_stage_sampler_kwargs = (
        harmonica_hr_nuts_kwargs
        if transit_engine == 'harmonica'
        else jaxoplanet_hr_nuts_kwargs
    )
    vmap_chunk_size_hr = _resolve_stage_vmap_width(
        flags,
        'highres',
        vmap_chunk_size_hr,
        sampler_backend=spectro_sampler,
        trend_inference=(
            'fixed'
            if hr_trend_mode == 'fixed'
            else (
                'gaussian_marginalized'
                if hr_trend_mode == 'gaussian_marginalized'
                else 'sampled_uniform'
            )
        ),
        num_cadences=int(time_hr.size),
        active_transit_cadences=(
            int(len(hr_transit_window_indices))
            if hr_transit_window_indices is not None
            else int(time_hr.size)
        ),
        mcmc_kwargs=highres_mcmc_kwargs,
        sampler_kwargs=hr_stage_sampler_kwargs,
        transit_engine=transit_engine,
        ld_profile=ld_profile,
        ld_mode=hr_ld_mode,
        detrend_type=detrend_type_multiwave,
        param_method=param_method,
        n_planets=n_planets,
        transit_window=transit_window_optimization,
    )

    hr_channel_batch_plan = _resolve_stage_channel_batch_plan(
        flags,
        'highres',
        int(num_lcs_hr),
    )
    if (
        hr_channel_batch_plan is not None
        and vmap_chunk_size_hr is not None
        and hr_channel_batch_plan.nominal_width != vmap_chunk_size_hr
    ):
        raise ValueError(
            "The highres batch plan nominal width "
            f"({hr_channel_batch_plan.nominal_width}) does not match the "
            f"configured/measured resident width ({vmap_chunk_size_hr})."
        )

    if (
        chunk_mode != 'serial'
        and vmap_chunk_size_hr is None
        and hr_channel_batch_plan is None
    ):
        raise ValueError(
            "flags.chunk_mode requires flags.vmap_chunk, a measured width "
            "selection, or a channel batch plan."
        )

    samples_hr = _run_sampling_stage(
        hr_model_for_run,
        key_mcmc_hr,
        time_hr,
        flux_err_hr,
        flux_hr,
        init_params_hr,
        nuts_kwargs=hr_stage_sampler_kwargs,
        mcmc_kwargs=highres_mcmc_kwargs,
        use_chunked=(
            spectro_sampling_mode == 'independent'
            or vmap_chunk_size_hr is not None
            or spectro_sampler in {
                'independent_nuts', 'independent_hmc', 'laplace_is'
            }
            or hr_channel_batch_plan is not None
        ),
        chunk_size=vmap_chunk_size_hr,
        chunk_mode=chunk_mode,
        parallel_job_count=chunk_parallel_job_count,
        parallel_job_index=chunk_parallel_job_index,
        output_dir=output_dir,
        checkpoint_prefix=_harmonica_checkpoint_prefix(
            f"{instrument_full_str}_{hr_bin_str}",
            transit_engine,
            harmonica_spectro_parameterization,
        ),
        checkpoint_signature={
            'stage': 'high_resolution',
            'transit_engine': transit_engine,
            'ld_profile': ld_profile,
            'ld_mode': hr_ld_mode,
            'trend_inference': trend_inference,
            'detrend_type': detrend_type_multiwave,
            'param_method': param_method,
            'whitelight_geometry_estimator': whitelight_geometry_estimator,
            'whitelight_geometry_handoff_fingerprint': (
                wl_geometry_handoff['artifact_fingerprint_sha256']
            ),
            'jaxoplanet_kernel': jaxoplanet_kernel,
            'transit_window_optimization': transit_window_optimization,
            'harmonica_max_order': max_harmonic_order,
            'harmonica_spectro_parameterization': harmonica_spectro_parameterization,
            'harmonica_spectro_fit_jitter': harmonica_spectro_fit_jitter,
            'harmonica_spectro_odd_frac_sigma': harmonica_spectro_odd_frac_sigma,
            'laplace_is_kwargs': (
                highres_laplace_is_kwargs
                if spectro_sampler == 'laplace_is' else None
            ),
        },
        channel_batch_plan=hr_channel_batch_plan,
        gradient_diagnostic_mode=spectro_gradient_diagnostic,
        gradient_diagnostic_strict=spectro_gradient_diagnostic_strict,
        sampler_backend=spectro_sampler,
        laplace_is_kwargs=(
            highres_laplace_is_kwargs
            if spectro_sampler == 'laplace_is' else None
        ),
        channel_varying_kwargs=(
            JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS
            if transit_engine == 'jaxoplanet'
            else HARMONICA_CHANNEL_VARYING_MODEL_KWARGS
        ),
        spectro_min_depth_ess=float(flags.get('spectro_min_depth_ess', 400)),
        spectro_max_divergences=int(flags.get('spectro_max_divergences', 0)),
        compile_box=bool(flags.get('compile_box', False)),
        dump_metadata={
            'config_path': os.path.abspath(args.config),
            'stage_kind': 'high_resolution',
            'stage_label': (
                f"{instrument_full_str}_{hr_bin_str}_high_resolution"
            ),
            'instrument': instrument,
            'instrument_label': instrument_full_str,
            'output_dir': os.path.abspath(output_dir),
            'wavelength': np.asarray(data.wavelengths_hr),
            'wavelength_err': np.asarray(data.wavelengths_err_hr),
            'active_window_cadences': (
                int(len(hr_transit_window_indices))
                if hr_transit_window_indices is not None
                else int(time_hr.size)
            ),
            'transit_engine': transit_engine,
        },
        **model_run_args_hr
    )
    if samples_hr is None:
        print("\nHigh-resolution chunk job complete. Exiting before final aggregation.")
        return
    if trend_names_hr is not None:
        samples_hr = materialize_marginalized_trend_samples(
            samples_hr,
            trend_names_hr,
            key_map_hr,
        )

    map_params_hr = {
        "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
        "rors": jnp.nanmedian(
            _harmonica_model_radius_samples(
                samples_hr, harmonica_spectro_parameterization
            )
            if transit_engine == 'harmonica'
            else samples_hr["rors"],
            axis=0,
        ),
        "period": PERIOD_FIXED
    }
    if ld_profile == 'quadratic':
        quadratic_hr_coeffs = (
            ld_interpolated_hr
            if hr_ld_mode == 'interpolated' else U_mu_hr_init
        )
        quadratic_u_med_hr = _posterior_quadratic_ld_median(
            samples_hr, quadratic_hr_coeffs
        )
        map_params_hr["u"] = quadratic_u_med_hr
        if transit_engine == 'harmonica':
            map_params_hr['u1_ld'] = quadratic_u_med_hr[..., 0]
            map_params_hr['u2_ld'] = quadratic_u_med_hr[..., 1]
    elif ld_profile == 'power2':
        if 'c1' in samples_hr:
            c1_med_hr = jnp.nanmedian(samples_hr['c1'], axis=0)
            c2_med_hr = jnp.nanmedian(samples_hr['c2'], axis=0)
            map_params_hr['c1'] = c1_med_hr
            map_params_hr['c2'] = c2_med_hr
            map_params_hr['u'] = jax.vmap(compute_u_from_c)(c1_med_hr, c2_med_hr)
            if transit_engine == 'harmonica':
                map_params_hr['c_ld'] = c1_med_hr
                map_params_hr['alpha_ld'] = c2_med_hr
        else:
             if 'u' in samples_hr:
                 map_params_hr["u"] = jnp.nanmedian(np.array(samples_hr["u"]), axis=0)
    if transit_engine == 'harmonica':
        map_params_hr = _augment_harmonica_params(
            map_params_hr,
            a_rs=A_RS_BASE,
            ecc=HARMONICA_ECC,
            omega=HARMONICA_OMEGA,
            a1=jnp.nanmedian(samples_hr["a1"], axis=0) if "a1" in samples_hr else None,
            a3=jnp.nanmedian(samples_hr["a3"], axis=0) if "a3" in samples_hr else None,
            a5=jnp.nanmedian(samples_hr["a5"], axis=0) if "a5" in samples_hr else None,
        )
    elif param_method == 'a_rs':
        map_params_hr.update({
            "a_rs": A_RS_BASE,
            "ecc": HARMONICA_ECC,
            "omega": HARMONICA_OMEGA,
        })
    map_params_hr.update({k: jnp.nanmedian(samples_hr[k], axis=0) for k in TREND_PARAMS if k in samples_hr})
    if transit_engine == 'jaxoplanet':
        map_params_hr = _attach_jaxoplanet_eval_metadata(
            map_params_hr,
            time_hr,
            jaxoplanet_kernel=jaxoplanet_kernel,
            ld_profile=ld_profile,
            transit_window_optimization=transit_window_optimization,
        )

    in_axes_map_hr = {"rors": 0}
    in_axes_map_hr.update({
        name: 0 for name in ('u', 'c1', 'c2') if name in map_params_hr
    })
    if transit_engine == 'harmonica':
        if ld_profile == 'power2':
            in_axes_map_hr.update({'c_ld': 0, 'alpha_ld': 0})
        else:
            in_axes_map_hr.update({'u1_ld': 0, 'u2_ld': 0})
        in_axes_map_hr.update({name: 0 for name in HARMONICA_ODD_HARMONICS if name in map_params_hr})
    in_axes_map_hr.update({k: 0 for k in TREND_PARAMS if k in map_params_hr})
    
    final_in_axes_hr = {k: in_axes_map_hr.get(k, None) for k in map_params_hr.keys()}
    selected_kernel_hr = resolve_detrend_kernel(detrend_type_multiwave)
    
    # Evaluate channels sequentially to prevent XLA Out-Of-Memory (OOM) on large channel/time dimensions.
    # By running sequentially, the peak memory usage is flat w.r.t. the number of channels.
    @jax.jit
    def eval_channel_hr(channel_params, t_val, *extra_args):
        if transit_engine == 'jaxoplanet':
            channel_params = {
                **channel_params,
                "_jaxoplanet_kernel": jaxoplanet_kernel,
                "_ld_profile": ld_profile,
            }
        return selected_kernel_hr(channel_params, t_val, *extra_args)

    model_all_hr_list = []
    num_lcs_hr = flux_err_hr.shape[0]
    for i in range(num_lcs_hr):
        channel_params = {}
        for k, v in map_params_hr.items():
            if k in _JAXOPLANET_STATIC_EVAL_KEYS:
                continue
            if final_in_axes_hr[k] == 0:
                channel_params[k] = v[i]
            else:
                channel_params[k] = v
        
        if 'gp_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel_hr(channel_params, time_hr, gp_trend)
        elif '2spot_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel_hr(channel_params, time_hr, spot_trend, spot_trend2)
        elif _has_single_spot_spectroscopic(detrend_type_multiwave):
            if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
                ch_model = eval_channel_hr(channel_params, time_hr, spot_trend, jump_trend)
            else:
                ch_model = eval_channel_hr(channel_params, time_hr, spot_trend)
        elif 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel_hr(channel_params, time_hr, jump_trend)
        elif 'explinear_spectroscopic' in detrend_type_multiwave:
            ch_model = eval_channel_hr(channel_params, time_hr, exp_trend_hr)
        else:
            ch_model = eval_channel_hr(channel_params, time_hr)
            
        model_all_hr_list.append(ch_model)
        
    model_all_hr = jnp.stack(model_all_hr_list, axis=0)

    residuals_hr = np.array(flux_hr - model_all_hr)
    _hr_dt_sec = float(np.nanmedian(np.diff(np.array(time_hr)))) * 86400.0
    plot_noise_binning_robust(
        residuals_hr, _hr_dt_sec,
        f"{output_dir}/36_{hr_artifact_stem}_noisebin.png",
        title=f"High-Res Noise Binning ({hr_bin_str})",
    )
    save_noise_binning_data(
        residuals_hr,
        f"{output_dir}/36_{hr_artifact_stem}_noisebin.csv",
    )

    median_total_error_hr = np.nanmedian(samples_hr['total_error'], axis=0)
    plot_wavelength_offset_summary(time_hr, flux_hr, median_total_error_hr, data.wavelengths_hr,
                                    map_params_hr, {
                                        "period": PERIOD_FIXED,
                                        "transit_engine": transit_engine,
                                        "param_method": param_method,
                                    },
                                    f"{output_dir}/34_{hr_artifact_stem}_summary.png",
                                  detrend_type=detrend_type_multiwave, gp_trend=gp_trend, spot_trend=spot_trend, spot_trend2=spot_trend2, jump_trend=jump_trend, exp_trend=(exp_trend_hr if 'explinear_spectroscopic' in detrend_type_multiwave else None))

    plot_transmission_spectrum(wl_hr, samples_hr["rors"], f"{output_dir}/31_{hr_artifact_stem}_spectrum")
    save_results(wl_hr, data.wavelengths_err_hr, samples_hr,  f"{output_dir}/{hr_artifact_stem}.csv")
    save_detailed_fit_results(time_hr, flux_hr, flux_err_hr, data.wavelengths_hr, data.wavelengths_err_hr, samples_hr, map_params_hr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{hr_artifact_stem}", median_total_error_hr, gp_trend=gp_trend, spot_trend=spot_trend, jump_trend=jump_trend)

    if transit_engine == 'harmonica' and _has_harmonica_odd_samples(samples_hr):
        save_harmonica_limb_products(
            wavelengths=wl_hr,
            wavelength_err=data.wavelengths_err_hr,
            rors_samples=_harmonica_model_radius_samples(
                samples_hr, harmonica_spectro_parameterization
            ),
            harmonic_samples=_harmonica_sample_payload(samples_hr),
            csv_path=f"{output_dir}/{hr_artifact_stem}_limb_spectra.csv",
            limb_spectrum_path=f"{output_dir}/35_{hr_artifact_stem}_limb_spectra.png",
            transmission_strings_path=f"{output_dir}/36_{hr_artifact_stem}_transmission_strings.png",
            posterior_strings_path=f"{output_dir}/37_{hr_artifact_stem}_transmission_string_posterior.png",
            posterior_samples_path=f"{output_dir}/{hr_artifact_stem}_limb_posterior_samples.npz",
            title_prefix=f"{planet_str} - {hr_bin_str}",
        )

    print("\nAnalysis complete!")

if __name__ == "__main__":
    try:
        main()
    except SamplerInputsDumpExit as error:
        print(
            "Requested sampler-input dump is complete; exiting before "
            f"high-resolution sampling ({error}).",
            flush=True,
        )
