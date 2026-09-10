"""Limb Darkening helpers."""

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
from .instruments import ld_mode_and_bounds
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
from .config import _resolve_ld_prior_cache_options
from .artifacts import (_atomic_dataframe_csv, _directory_metadata_identity,
                        _file_content_identity, _update_checkpoint_hash)
from .outputs import compute_aic

def _power2_ld_initial_sites(coefficients, parameterization):
    """Return initial values in the actual latent coordinates of the LD model.

    The starting coefficients are first moved strictly inside the power-2
    prior support (c1 in (0, 1), c2 in (0.001, 1)); native-resolution prior
    means on strong stellar lines can sit outside it.
    """
    coefficients = _clip_ld_initial_values(coefficients, 'power2', 'power2 sites')
    if parameterization == 'coefficients':
        return {'c1': coefficients[..., 0], 'c2': coefficients[..., 1]}
    if parameterization == 'decorrelated':
        return {'ld_decorrelated': Power2MaxtedTransform()(coefficients)}
    raise ValueError(f"Unknown power-2 LD parameterization: {parameterization}")


def _power2_ld_optimization_sites(parameterization):
    """Return only latent sample sites, never deterministic LD coefficients."""
    if parameterization == 'coefficients':
        return ['c1', 'c2']
    if parameterization == 'decorrelated':
        return ['ld_decorrelated']
    raise ValueError(f"Unknown power-2 LD parameterization: {parameterization}")


def _clip_ld_initial_values(coefficients, ld_profile, stage_label='ld'):
    """Move limb-darkening *initial values* strictly inside the prior support.

    Narrow (native-resolution) bins on strong stellar lines can give power-2
    prior means with c1 > 1, which the truncated prior handles but which make
    ``init_to_value`` start outside the support and abort with "Cannot find
    valid initial parameters".  Only the starting point is changed; the prior
    and the posterior are untouched.
    """
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    eps = 1e-3
    if ld_profile == 'power2':
        low = jnp.asarray([eps, 0.001 + eps])
    else:
        low = jnp.asarray([eps, eps])
    high = 1.0 - eps
    clipped = jnp.clip(coefficients, low, high)
    n_moved = int(jnp.sum(jnp.any(clipped != coefficients, axis=-1)))
    if n_moved:
        print(
            f"[LD init {stage_label}] {n_moved} channel(s) had prior means outside "
            "the physical support; their starting values were clipped inside it.",
            flush=True,
        )
    return clipped


def _quadratic_uniform_initial_sites(coefficients, basis):
    """Initialize the selected quadratic-uniform prior coordinates."""
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    if basis == 'coefficients':
        return {'u': coefficients}
    if basis == 'uplus_uminus':
        return {'ld_uplus_uminus': jnp.stack(
            (coefficients[..., 0] + coefficients[..., 1],
             coefficients[..., 0] - coefficients[..., 1]), axis=-1)}
    raise ValueError(f"Unknown quadratic uniform LD basis: {basis}")


def _validate_whitelight_optimized_start(
    solution,
    *,
    ld_profile,
    ld_parameterization,
    ld_prior_mode,
    quadratic_uniform_bounds=(0.0, 1.0),
    n_planets=1,
):
    """Validate physical LD and transit geometry before using an optimizer result."""
    reasons = []
    physical_site_names = {
        'c1', 'c2', 'u', 'b', 'rors', 'depths', 'duration', 'a_rs',
        'cos_i', 'inc', 'error',
    }
    physical_site_prefixes = (
        'b_', 'rors_', 'depths_', 'duration_', 'a_rs_', 'cos_i_', 'inc_',
    )
    for name, value in solution.items():
        if name in physical_site_names or name.startswith(physical_site_prefixes):
            array = np.asarray(jax.device_get(value), dtype=float)
            if not np.all(np.isfinite(array)):
                reasons.append(f'deterministic physical site {name} is non-finite')
    physical_ld = None
    if ld_profile == 'power2':
        if 'ld_decorrelated' in solution:
            physical_ld = Power2MaxtedTransform().inv(
                jnp.asarray(solution['ld_decorrelated'])
            )
        elif 'c1' in solution and 'c2' in solution:
            physical_ld = jnp.stack((solution['c1'], solution['c2']))
        if physical_ld is not None:
            values = np.asarray(jax.device_get(physical_ld), dtype=float)
            low = np.asarray(
                [0.0, 0.0 if ld_prior_mode == 'uniform' else 0.001]
            )
            if not np.all(np.isfinite(values)):
                reasons.append('power-2 coefficients are non-finite')
            elif np.any(values < low) or np.any(values > 1.0):
                reasons.append(
                    f'power-2 coefficients {values.tolist()} are outside '
                    f'[{low.tolist()}, [1.0, 1.0]]'
                )
    elif ld_profile == 'quadratic' and 'u' in solution:
        values = np.asarray(jax.device_get(solution['u']), dtype=float)
        quadratic_low, quadratic_high = (
            quadratic_uniform_bounds
            if ld_prior_mode == 'uniform'
            else (0.0, 1.0)
        )
        if not np.all(np.isfinite(values)):
            reasons.append('quadratic coefficients are non-finite')
        elif np.any(values < quadratic_low) or np.any(values > quadratic_high):
            reasons.append(
                f'quadratic coefficients {values.tolist()} are outside '
                f'[{quadratic_low}, {quadratic_high}]'
            )

    rors_min = float(np.sqrt(1.0e-6))
    rors_max = float(np.sqrt(0.5))
    for planet_index in range(int(n_planets)):
        rors = solution.get(f'rors_{planet_index}')
        if rors is None and int(n_planets) == 1:
            rors = solution.get('rors')
        raw_b = solution.get(f'_b_{planet_index}')
        b = solution.get(f'b_{planet_index}')
        if b is None and raw_b is not None:
            b = jnp.abs(raw_b)
        if b is None and int(n_planets) == 1:
            b = solution.get('b')
        if rors is None or b is None:
            reasons.append(f'planet {planet_index} geometry sites are missing')
            continue
        rors_value = float(np.asarray(jax.device_get(rors)).reshape(-1)[0])
        b_value = float(np.asarray(jax.device_get(b)).reshape(-1)[0])
        if not np.isfinite(rors_value) or not (rors_min < rors_value < rors_max):
            reasons.append(
                f'planet {planet_index} rprs={rors_value} is outside '
                f'({rors_min}, {rors_max})'
            )
        if not np.isfinite(b_value) or not (0.0 <= b_value < 1.0 + rors_value):
            reasons.append(
                f'planet {planet_index} b={b_value} violates '
                f'0 <= b < 1+rprs={1.0 + rors_value}'
            )
    return not reasons, reasons


def _get_ld_mode_bounds(instrument, order=None):
    """ExoTiC-LD mode and wavelength bounds (angstroms) for a Koala instrument."""
    return ld_mode_and_bounds(instrument, order)


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


def _compute_ld_coeffs_with_retry(sld, ld_profile, range_min, range_max, mode,
                                  ld_mu_min, return_sigmas, wl_min, wl_max,
                                  max_doublings=4):
    """Fit one channel's limb-darkening law, widening the bin if the fit fails.

    Native-resolution bins can be only a few stellar-grid samples wide, and
    exotic_ld's curve_fit occasionally fails to converge on such a bin.  Each
    retry doubles the bin width about its centre (clipped to the instrument
    range), which only smooths the stellar intensity profile locally.
    """
    mu_kwargs = {"mu_min": float(ld_mu_min)} if (ld_profile == 'quadratic' and ld_mu_min is not None) else {}
    if ld_profile == 'quadratic':
        fn = sld.compute_quadratic_ld_coeffs
    elif ld_profile == 'power2':
        fn = sld.compute_power2_ld_coeffs
    else:
        raise ValueError(f"Unknown ld_profile: {ld_profile}")
    centre = 0.5 * (range_min + range_max)
    half = 0.5 * (range_max - range_min)
    last_err = None
    for attempt in range(max_doublings + 1):
        lo = max(wl_min, centre - half)
        hi = min(wl_max, centre + half)
        try:
            return fn(wavelength_range=[lo, hi], mode=mode, return_sigmas=return_sigmas, **mu_kwargs)
        except RuntimeError as err:
            last_err = err
            half *= 2.0
    raise RuntimeError(
        f"Limb-darkening fit failed for {range_min:.1f}-{range_max:.1f} A even after "
        f"{max_doublings} bin doublings: {last_err}"
    )


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
            result = _compute_ld_coeffs_with_retry(
                sld, ld_profile, range_min, range_max, mode,
                ld_mu_min, return_sigmas, wl_min, wl_max,
            )
            if return_sigmas:
                coeffs, sigmas = result
                U_mu.append(coeffs)
                U_sig.append(sigmas)
            else:
                U_mu.append(result)
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
    """Return per-channel (l, delta) centers/scales for the Sing prior."""
    coeff = np.asarray(quadratic_coefficients, dtype=float)
    scalar = coeff.ndim == 1
    coeff = np.atleast_2d(coeff)
    model_l, model_delta = quadratic_to_sing(coeff[:, 0], coeff[:, 1])
    offsets = dict(SING_TABULATED_OFFSET)
    source = 'tabulated'
    if offsets_override is not None:
        offsets = {
            'l': float(offsets_override['l']),
            'delta': float(offsets_override['delta']),
        }
        source = 'newly-fitted gray calibration'
    if offsets_override is None:
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
    """Validate divergence/finite gates and report per-channel LD ESS."""
    if 'limb_l' not in samples or 'limb_delta' not in samples:
        return False, {'reason': "physical free-LD fit did not return limb_l and limb_delta"}
    physical = np.stack(
        (np.asarray(samples['limb_l'], dtype=float),
         np.asarray(samples['limb_delta'], dtype=float)), axis=-1,
    )
    if physical.ndim != 3 or physical.shape[-1] != 2 or not np.all(np.isfinite(physical)):
        return False, {'reason': f"invalid free-LD posterior shape/values: {physical.shape}"}
    try:
        ess_matrix = np.asarray([
            [float(az.ess(physical[None, :, channel, coefficient], method='bulk'))
             for coefficient in range(physical.shape[2])]
            for channel in range(physical.shape[1])
        ])
        min_bulk_ess = float(np.nanmin(ess_matrix))
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
        'bulk_ess_l_per_channel': ess_matrix[:, 0].tolist(),
        'bulk_ess_delta_per_channel': ess_matrix[:, 1].tolist(),
        'channels_below_ess_warning': np.flatnonzero(
            np.any(ess_matrix < float(min_ess), axis=1)
        ).astype(int).tolist(),
        'num_divergences': int(num_divergences),
        'missing_diagnostics': missing_diagnostics,
    }
    passed = (not missing_diagnostics and num_divergences == 0
              and np.all(np.isfinite(ess_matrix)))
    return passed, result


def get_or_build_power2_ld_prior(stellar_cfg, wavelengths, wavelength_err, instrument, order=None, output_dir='.', cache_label='ld'):
    phase_started = time.perf_counter()
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

    cache_options = _resolve_ld_prior_cache_options(stellar_cfg)
    cache_enabled = cache_options["enabled"]
    cache_dir = cache_options["cache_dir"]
    if cache_enabled:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            probe = tempfile.NamedTemporaryFile(dir=cache_dir, delete=True)
            probe.close()
        except OSError as error:
            print(
                f"[LD prior] Cache directory {cache_dir} is unavailable "
                f"({error}); continuing without a persistent cache.",
                flush=True,
            )
            cache_enabled = False
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

    if cache_enabled and os.path.exists(cache_path):
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
    # Exact reformulation: integrate each unique stellar-grid node once per
    # bin, blend with exotic_ld's trilinear weights, and fit every
    # (combination, bin) power-2 law in one batched JAX solve.  See
    # models/ld_prior_fast.py.  Any (combination, bin) whose batched fit
    # did not converge is recomputed through the legacy per-bin path.
    from models.ld_prior_fast import build_power2_grid
    if has_vector_err:
        ranges = []
        for wl_i, err_i in zip(wavelengths_np, wavelength_err_np):
            intended_min = (wl_i - err_i) * 1e4
            intended_max = (wl_i + err_i) * 1e4
            ranges.append(_clip_ld_range(intended_min, intended_max, wl_min, wl_max))
    else:
        ranges = [_clip_ld_range(np.min(wavelengths_np) * 1e4,
                                 np.max(wavelengths_np) * 1e4, wl_min, wl_max)]
    fast_started = time.perf_counter()
    mu_grid, grad_norm = build_power2_grid(
        [(mh_i, teff_i, logg_i) for (mh_i, teff_i, logg_i, _) in combos],
        ranges, mode, ld_prior_model, ld_data_path,
        interpolate_type=ld_interpolate_type,
    )
    bad = ~(np.all(np.isfinite(mu_grid), axis=-1) & (grad_norm < 1e-6))
    print(
        f"[LD prior fast] {mu_grid.shape[0]} combinations x {mu_grid.shape[1]} bins "
        f"in {time.perf_counter() - fast_started:.1f} s; "
        f"{int(bad.sum())} fit(s) sent to the legacy path",
        flush=True,
    )
    for ci, bi in zip(*np.nonzero(bad)):
        mh_i, teff_i, logg_i, _ = combos[ci]
        sld_grid = StellarLimbDarkening(
            M_H=mh_i, Teff=teff_i, logg=logg_i, ld_model=ld_prior_model,
            ld_data_path=ld_data_path, interpolate_type=ld_interpolate_type,
            verbose=0,
        )
        lo, hi = ranges[bi]
        mu_grid[ci, bi] = np.asarray(_compute_ld_coeffs_with_retry(
            sld_grid, 'power2', lo, hi, mode, None, False, wl_min, wl_max,
        ), dtype=float)
    for ci, (_, _, _, weight) in enumerate(combos):
        successful_mu.append(mu_grid[ci])
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
        if cache_enabled:
            try:
                _atomic_dataframe_csv(out_df, cache_path, index=False)
            except OSError as error:
                cache_enabled = False
                print(
                    f"[LD prior] Could not write cache {cache_path} ({error}); "
                    "continuing with the newly computed prior.",
                    flush=True,
                )
        if cache_enabled:
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
        if cache_enabled:
            try:
                _atomic_dataframe_csv(out_df, cache_path, index=False)
            except OSError as error:
                print(
                    f"[LD prior] Could not write cache {cache_path} ({error}); "
                    "continuing with the newly computed prior.",
                    flush=True,
                )
            else:
                print(f"[LD prior] Saved power2 grid cache to {cache_path}", flush=True)
        return jnp.array([c1_mean[0], c2_mean[0]]), jnp.array([c1_tot[0], c2_tot[0]])


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
