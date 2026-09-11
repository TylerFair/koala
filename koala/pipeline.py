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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
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
from koala.exclusions import resolve_exclusions, has_cut_phase_directive
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




# This is the authoritative configuration surface for ``flags``.  Keep every
# accepted key in exactly one tier: PUBLIC values describe the dataset/science
# model, ADVANCED values are occasionally useful user controls, and INTERNAL
# values are compatibility-preserving implementation controls.  The expanded
# families cover keys assembled dynamically by the stage resolvers below.















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
    ParameterSpec, parse_planet_parameter_specs, planet_parameter_centers,
    describe_planet_parameter_specs,
)
from .data import _pad_spectro_cadences_exact, jax_bin_lightcurve
from .artifacts import (
    _update_checkpoint_hash, _science_artifact_fingerprint,
    _science_artifact_manifest_matches, _write_science_artifact_manifest,
    _file_content_identity, _optional_file_content_identity,
    _directory_metadata_identity, _atomic_save_npy, _atomic_savez,
    _atomic_savez_compressed, _atomic_dataframe_csv,
    ArtifactSet, load_or_compute_artifact_set,
)















from .outputs import (
    _param_at, _poly_trend_np, _soft_step_np, _trend_from_params_np,
    plot_noise_binning_robust, _noise_binning_stats, save_noise_binning_data,
    save_whitelight_timeseries, _save_mcmc_diagnostics, compute_aic,
    get_asym_errors, compute_u_from_c, fit_polynomial, plot_poly_fit,
    save_results, save_detailed_fit_results, get_robust_sigma,
    calculate_beta_metrics, run_beta_monte_carlo,
)
from .geometry import (
    _geometry_chain_quality, _continue_mcmc_until_geometry_gate,
    _posterior_num_draws, _geometry_from_white_light_medians,
    _selected_geometry_primitives, _whitelight_geometry_handoff_fingerprint,
    _write_whitelight_geometry_handoff, _load_whitelight_geometry_handoff,
)
from .sampling import (
    _DUMPED_FIRST_HIGHRES_STAGE, SamplerInputsDumpExit, _build_numpyro_mcmc,
    get_samples, _slice_by_channel, _take_by_channel, _callable_identity,
    _tree_to_numpy_for_pickle, _build_spectroscopic_model,
    _sampler_model_builder_spec, _sampler_initial_potential, _safe_stage_label,
    _maybe_dump_sampler_inputs, _chunk_checkpoint_fingerprint,
    _sampling_workload_fingerprint, _annotate_sampler_diagnostics,
    _write_or_validate_checkpoint_manifest, _chunk_ranges_for,
    _chunk_checkpoint_path, _SamplerSamples, _load_chunk_samples,
    _concatenate_chunk_samples, _spectro_failed_lanes,
    _spectro_sampler_swap_order, _resolve_parallel_chunk_job,
    get_samples_chunked, _run_sampling_stage,
)
from .instruments import normalize_instrument, resolve_detector, detector_label
from .limb_darkening import (
    _power2_ld_initial_sites, _power2_ld_optimization_sites,
    _clip_ld_initial_values, _quadratic_uniform_initial_sites,
    _validate_whitelight_optimized_start, _get_ld_mode_bounds, _clip_ld_range,
    _gaussian_pdf, _build_axis, _combine_weighted_means_and_sigmas,
    _compute_ld_coeffs_with_retry, get_limb_darkening, build_sing_ld_prior,
    _assess_sing_gray_fit, get_or_build_power2_ld_prior,
    _posterior_quadratic_ld_median,
)
from .harmonica_products import (
    _HARMONICA_LIMB_LOGGER, _LEGACY_LIMB_SCHEMA_WARNING_EMITTED,
    _coerce_wavelength_axis, _harmonica_frac_site, _harmonica_coeff_to_frac,
    _harmonica_coeff_array_to_frac, _harmonica_coeff_array_to_delta_r,
    _harmonica_coefficients_to_half_area_init,
    _validate_harmonica_spectro_parameterization,
    _harmonica_model_radius_samples, _harmonica_checkpoint_prefix,
    _harmonica_spectro_artifact_stem, _coerce_harmonica_sample_matrix,
    _extract_harmonica_limb_samples, _sum_harmonica_odd_samples,
    _harmonica_limb_product_samples, _harmonica_percentile_summary,
    _harmonica_r_vector_from_values, _harmonica_sample_payload,
    _has_harmonica_odd_samples, _augment_harmonica_params,
    _harmonica_cosi_from_b, _add_rp_summary_from_depth,
    load_harmonica_limb_dataframe, build_harmonica_limb_dataframe,
    _save_harmonica_limb_posterior_samples, save_harmonica_limb_products,
)
from .surface import (
    _attach_jaxoplanet_eval_metadata, _attach_surface_eval_metadata,
    _seed_surface_spectroscopic_init, _phase_curve_flux_init_sites,
    _prepare_fixed_surface_basis, _surface_basis_on_time_mask,
    _select_transit_eval_params,
)

from .spectroscopy import run_low_resolution_stage, run_high_resolution_stage

from .white_light import run_white_light_stage


_GPU_BACKEND_ALIASES = {'cuda': 'gpu', 'rocm': 'gpu', 'gpu': 'gpu'}


def _jax_backend_available(platform):
    """Return True when JAX can initialise ``platform`` on this machine."""
    try:
        return len(jax.devices(platform)) > 0
    except Exception:
        # jax raises RuntimeError when the requested plugin is missing or
        # fails to initialise (e.g. a CPU-only wheel on a CUDA machine).
        return False


def _concrete_gpu_backend():
    """Return the concrete GPU backend name JAX exposes, or None.

    Newer JAX releases reject the ``'gpu'`` alias in ``jax_platform_name``
    and require ``'cuda'`` or ``'rocm'``; older ones accept either.
    """
    for name in ('cuda', 'rocm'):
        if _jax_backend_available(name):
            return name
    if _jax_backend_available('gpu'):
        return 'gpu'
    return None


def _set_jax_platform(platform):
    """Set the JAX default platform, tolerating already-initialised JAX."""
    try:
        jax.config.update('jax_platform_name', platform)
    except Exception as exc:
        warnings.warn(
            f"Could not set jax_platform_name={platform!r}: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
    try:
        numpyro.set_platform(platform)
    except Exception:
        # numpyro.set_platform only repeats the jax.config.update above.
        pass


def resolve_host_device(requested):
    """Pick the JAX/numpyro platform for ``host_device``.

    ``requested`` is ``'gpu'``, ``'cpu'`` or ``'auto'``. A GPU is used
    only when JAX can actually initialise one; otherwise the run falls
    back to CPU with a warning instead of raising. This covers the common
    case where CUDA is installed but the CPU-only ``jax`` wheel is.
    Returns ``'gpu'`` or ``'cpu'``.
    """
    requested = (requested or 'auto').lower()
    if requested not in {'gpu', 'cpu', 'auto', 'cuda', 'rocm'}:
        raise ValueError(
            f"host_device must be 'gpu', 'cpu' or 'auto', got {requested!r}"
        )
    if requested in {'cuda', 'rocm'}:
        requested = 'gpu'
    want_gpu = requested in {'gpu', 'auto'}
    concrete = _concrete_gpu_backend() if want_gpu else None
    if concrete is not None:
        platform = 'gpu'
    else:
        if requested == 'gpu':
            warnings.warn(
                "host_device='gpu' was requested but JAX cannot initialise a "
                "GPU backend; falling back to CPU. If this machine has a "
                "CUDA GPU, install the CUDA build of JAX "
                "(pip install -U 'jax[cuda12]') and check that JAX_PLATFORMS "
                "is not set to 'cpu'.",
                RuntimeWarning,
                stacklevel=2,
            )
        platform = 'cpu'
        concrete = 'cpu'
    _set_jax_platform(concrete)
    # jax.default_backend() reports the plugin name ('cuda' or 'rocm'),
    # never the 'gpu' alias, so compare after mapping it back.
    actual = _GPU_BACKEND_ALIASES.get(jax.default_backend(), jax.default_backend())
    if actual != platform:
        raise RuntimeError(
            f"Requested JAX backend {platform!r}, but JAX selected "
            f"{actual!r}. Available devices: {jax.devices()}. This usually "
            "means JAX was already initialised on another backend before "
            "koala ran; set JAX_PLATFORMS before starting Python."
        )
    return platform

def _align_trend_to_time(trend, trend_time, target_time):
    trend = np.asarray(trend)
    target_time = np.asarray(target_time)
    trend_time = np.asarray(trend_time)
    if len(trend) == len(target_time):
        return trend
    return np.interp(target_time, trend_time, trend)










    

























































































































































# Full catalogue of possible odd harmonic names (used by helper functions).
# The active subset is set per-run via the config's harmonica_max_order.
from models.harmonica.core import (
    _ALL_ODD_COEFF_SPECS,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    harmonica_half_area_area_radius_and_q,
)






































































def _resolve_need_lowres(flags, low_resolution_bins):
    """Decide whether the low-resolution bridge stage runs.

    The stage needs a coarse grid (``resolution.low`` or ``pixels.low``).
    Without one it is skipped. An explicit ``flags.need_lowres: true`` with no
    grid is a configuration error rather than a silent no-op.
    """
    requested = flags.get('need_lowres', True)
    if low_resolution_bins is not None:
        return bool(requested)
    if 'need_lowres' in flags and requested:
        raise ValueError(
            "flags.need_lowres is true but no low-resolution grid is set; "
            "add resolution.low (or pixels.low) or remove flags.need_lowres."
        )
    print("No low-resolution grid configured; the low-resolution stage is skipped.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    return run(cfg, config_path=args.config)


def run(cfg, config_path=None):
    flags = cfg.get('flags', {})
    _validate_flag_keys(flags)
    instrument = normalize_instrument(cfg['instrument'])
    cfg['instrument'] = instrument
    nrs, order = resolve_detector(instrument, cfg)

    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']
    if 'period' not in planet_cfg:
        raise KeyError("'planet.period' is required.")
    planet_parameter_specs = parse_planet_parameter_specs(planet_cfg)
    from models.jaxoplanet.config import parse_surface_config
    _early_engine = str(flags.get('transit_engine', 'jaxoplanet')).strip().lower()
    _early_model = str(flags.get('light_curve_model', 'transit')).strip().lower()
    _early_param_method = (
        'a_rs' if _early_engine == 'harmonica' or _early_model != 'transit'
        or stellar_cfg.get('spots') else 'duration'
    )
    surface_config = parse_surface_config(
        flags, planet_cfg, stellar_cfg,
        len(planet_parameter_specs['period']),
        parameter_specs=planet_parameter_specs,
        param_method=_early_param_method,
    )
    for line in describe_planet_parameter_specs(planet_parameter_specs):
        print(line)
    _surface_spec_names = [k for k in surface_config if k.endswith('_spec')]
    if _surface_spec_names:
        _surface_specs = {
            name[:-5]: surface_config[name] for name in _surface_spec_names
        }
        for line in describe_planet_parameter_specs(
            _surface_specs, title="Planet surface parameters (model units)"
        ):
            print(line)
    is_prism = instrument == 'NIRSPEC/PRISM'
    is_explinear = 'explinear' in str(flags.get('detrending_type', 'linear'))
    flags.setdefault('spectro_sampler', 'independent_nuts')
    whitelight_trend_parameterization = _resolve_whitelight_trend_parameterization(
        flags.get('detrending_type', 'linear')
    )
    whitelight_two_spot_ordering = 'legacy'
    whitelight_mass_matrix = _resolve_whitelight_mass_matrix(
        flags.get('detrending_type', 'linear')
    )
    spectro_min_depth_ess = float(flags.get('spectro_min_depth_ess', 400))
    spectro_max_divergences = int(flags.get('spectro_max_divergences', 0))
    compilation_cache_dir = _resolve_compile_cache_options(flags)["cache_dir"]
    if compilation_cache_dir is not None:
        try:
            os.makedirs(compilation_cache_dir, exist_ok=True)
            probe = tempfile.NamedTemporaryFile(
                dir=compilation_cache_dir, delete=True
            )
            probe.close()
        except OSError as error:
            print(
                "Persistent JAX cache unavailable at "
                f"{compilation_cache_dir} ({error}); continuing without it."
            )
        else:
            jax.config.update('jax_compilation_cache_dir', compilation_cache_dir)
            jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)
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
    exclude_times, exclude_integrations = resolve_exclusions(cfg)

    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    input_file = cfg.get('input_file', cfg.get('fits_file'))
    if input_file is None:
        raise KeyError("'input_file' (or the older 'fits_file') is required.")
    fits_file = os.path.join(input_dir, input_file)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    plots_mode = str(flags.get('plots', 'full')).strip().lower()
    if plots_mode not in {'full', 'minimal'}:
        raise ValueError("flags.plots must be one of {'full', 'minimal'}.")
    if plots_mode == 'minimal':
        print("Minimal plots enabled: diagnostic noise/poly/light-curve grids skipped.")

    detrending_type = flags.get('detrending_type', 'linear')
    need_lowres = _resolve_need_lowres(flags, low_resolution_bins)
    spot_amp = flags.get('spot_amp', 0.0)
    spot_mu = flags.get('spot_center', 0.0)
    spot_sigma = flags.get('spot_width', 0.0)
    spot_amp2 = flags.get('spot_amp2', flags.get('spot_amp_2', 0.0))
    spot_mu2 = flags.get('spot_center2', flags.get('spot_center_2', 0.0))
    spot_sigma2 = flags.get('spot_width2', flags.get('spot_width_2', 0.0))
    t_jump_guess = flags.get('t_jump_guess', None)
    jump_guess = flags.get('jump_guess', 0.0)
    step_width_mode = 'free'
    step_width_days = _JUMP_WIDTH_DAYS
    whitelight_geometry_estimator = 'posterior_median'
    transit_engine = flags.get('transit_engine', 'jaxoplanet')
    jaxoplanet_kernel = 'auto'
    spectro_fixed_timescale_trends = True
    if (spectro_fixed_timescale_trends and 'explinear' in detrending_type
            and transit_engine != 'jaxoplanet'):
        raise ValueError(
            "flags.spectro_fixed_timescale_trends currently supports only "
            "flags.transit_engine='jaxoplanet'."
        )
    spectro_sampler = str(flags.get('spectro_sampler', 'independent_nuts')).lower()
    jitter_prior = 'lognormal'
    jitter_prior_scale = 2.0
    jitter_prior_center = 0.5
    if spectro_sampler not in {
        'joint_nuts', 'independent_nuts', 'independent_hmc'
    }:
        raise ValueError(
            "flags.spectro_sampler must be one of "
            "{'joint_nuts', 'independent_nuts', 'independent_hmc'}. "
            f"Received '{spectro_sampler}'."
        )
    if spectro_sampler in {'independent_nuts', 'independent_hmc'} and transit_engine not in {
        'jaxoplanet', 'harmonica'
    }:
        raise ValueError(
            f"flags.spectro_sampler='{spectro_sampler}' does not support "
            f"flags.transit_engine={transit_engine!r}."
        )
    transit_window_optimization = 'auto'
    spectro_cadence_reduction = str(
        flags.get('spectro_cadence_reduction', 'auto')
    ).strip().lower()
    if spectro_cadence_reduction not in {'auto', 'off'}:
        raise ValueError(
            "flags.spectro_cadence_reduction must be 'auto' or 'off'."
        )
    spectro_transit_grid = str(
        flags.get('spectro_transit_grid', 'auto')
    ).strip().lower()
    if spectro_transit_grid not in {'auto', 'off'}:
        raise ValueError(
            "flags.spectro_transit_grid must be 'auto' or 'off'."
        )
    spectro_transit_grid_nodes = int(
        flags.get('spectro_transit_grid_nodes', 769)
    )
    if spectro_transit_grid_nodes < 13:
        raise ValueError(
            "flags.spectro_transit_grid_nodes must be at least 13."
        )
    vmap_chunk = flags.get('spectro_chunk_size', flags.get('vmap_chunk', False))
    vmap_chunk_size = None
    if isinstance(vmap_chunk, str) and vmap_chunk.lower() == 'auto':
        vmap_chunk_size = 'auto'
    elif isinstance(vmap_chunk, (int, float)) and not isinstance(vmap_chunk, bool):
        vmap_chunk_size = int(vmap_chunk)
    elif vmap_chunk is True:
        vmap_chunk_size = 50
    elif spectro_sampler in {'independent_nuts', 'independent_hmc'}:
        vmap_chunk_size = 40
        print(
            f"flags.spectro_sampler='{spectro_sampler}' without "
            "flags.vmap_chunk; defaulting to 40 resident GPU lanes."
        )
    if vmap_chunk_size is not None and vmap_chunk_size != 'auto' and vmap_chunk_size < 1:
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
    whitelight_laplace_options = _resolve_whitelight_laplace_options(
        is_prism, whitelight_mass_matrix
    )
    lowres_mcmc_kwargs = _resolve_stage_mcmc_kwargs(flags, 'lowres')
    highres_mcmc_kwargs = _resolve_stage_mcmc_kwargs(flags, 'highres')
    ld_profile = validate_ld_profile(flags.get('ld_profile', 'quadratic'))
    ld_prior_mode = _resolve_ld_prior_mode(flags, stellar_cfg, ld_profile)
    spectro_ld_parameterization = _resolve_ld_parameterization(
        ld_prior_mode, ld_profile
    )
    whitelight_ld_parameterization = spectro_ld_parameterization
    ld_uniform_basis = str(
        flags.get('ld_uniform_basis', 'uplus_uminus')
    ).strip().lower()
    if ld_uniform_basis not in {'uplus_uminus', 'coefficients'}:
        raise ValueError(
            "flags.ld_uniform_basis must be 'uplus_uminus' or 'coefficients'."
        )
    ld_uniform_coefficient_bounds = np.asarray([0.0, 1.0])
    quadratic_uniform_physical_bounds = (
        (-1.5, 2.0)
        if ld_uniform_basis == 'uplus_uminus'
        else tuple(ld_uniform_coefficient_bounds)
    )
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
        flags, 'harmonica_wl', default_dense_mass=True,
        is_prism=is_prism, is_explinear=is_explinear,
    )
    harmonica_lr_nuts_kwargs = _resolve_harmonica_stage_nuts_kwargs(
        flags, 'harmonica_lr', default_dense_mass=True,
        is_prism=is_prism, is_explinear=is_explinear,
    )
    harmonica_hr_nuts_kwargs = _resolve_harmonica_stage_nuts_kwargs(
        flags, 'harmonica_hr', default_dense_mass=False,
        is_prism=is_prism, is_explinear=is_explinear,
    )

    # Surface and Harmonica models use a/Rs; ordinary JAXoplanet transits use duration.
    uses_surface_model = bool(
        surface_config['model'] != 'transit' or surface_config['spots']
    )
    param_method = 'a_rs' if transit_engine == 'harmonica' or uses_surface_model else 'duration'
    if uses_surface_model:
        if transit_engine != 'jaxoplanet':
            raise ValueError(
                "flags.light_curve_model and stellar.spots are supported only "
                "with flags.transit_engine='jaxoplanet'."
            )
        transit_window_optimization = 'off'
        if has_cut_phase_directive(exclude_times):
            raise ValueError(
                "cut_phase_to_transit cannot be used for eclipse, phase-curve, "
                "or rotating stellar-spot fits because it removes the signal."
            )
    if transit_engine == 'harmonica':
        from models.harmonica import create_whitelight_model, create_vectorized_model, NUTS_KWARGS, derive_geometry
        _engine_wl_kw = {
            'max_harmonic_order': max_harmonic_order,
            'param_method': param_method,
            'ld_profile': ld_profile,
            'trend_parameterization': whitelight_trend_parameterization,
            'two_spot_ordering': whitelight_two_spot_ordering,
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
            'ld_parameterization': whitelight_ld_parameterization,
            'ld_uniform_basis': ld_uniform_basis,
            'ld_uniform_coefficient_bounds': tuple(ld_uniform_coefficient_bounds),
            'trend_parameterization': whitelight_trend_parameterization,
            'two_spot_ordering': whitelight_two_spot_ordering,
            'surface_config': surface_config,
        }
        _engine_spectro_kw = {
            'jitter_prior': jitter_prior,
            'jitter_prior_scale': jitter_prior_scale,
            'jitter_prior_center': jitter_prior_center,
            'ld_profile': ld_profile,
            'param_method': param_method,
            'transit_window': transit_window_optimization,
            'cadence_reduction': spectro_cadence_reduction,
            'transit_grid': spectro_transit_grid,
            'transit_grid_nodes': spectro_transit_grid_nodes,
            'ld_parameterization': spectro_ld_parameterization,
            'ld_uniform_basis': ld_uniform_basis,
            'ld_uniform_coefficient_bounds': tuple(ld_uniform_coefficient_bounds),
            'surface_config': surface_config,
        }
        if trend_inference == 'gaussian_marginalized' and 'gp' in detrending_type:
            raise ValueError(
                "flags.trend_inference='gaussian_marginalized' does not yet "
                "support GP detrending."
            )
        independent_backend = spectro_sampler in {
            'independent_nuts', 'independent_hmc'
        }
        hmc_backend = spectro_sampler == 'independent_hmc'
        jaxoplanet_lr_nuts_kwargs = _resolve_jaxoplanet_spectro_nuts_kwargs(
            'lowres',
            NUTS_KWARGS,
            independent=independent_backend,
            hmc=hmc_backend,
            is_prism=is_prism,
            is_explinear=is_explinear,
        )
        jaxoplanet_hr_nuts_kwargs = _resolve_jaxoplanet_spectro_nuts_kwargs(
            'highres',
            NUTS_KWARGS,
            independent=independent_backend,
            hmc=hmc_backend,
            is_prism=is_prism,
            is_explinear=is_explinear,
        )

    HARMONICA_ODD_HARMONICS = tuple(
        name for name, _ in harmonica_odd_coeff_specs(max_harmonic_order)
    )

    host_device = cfg.get('host_device', 'gpu').lower()
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
    host_device = resolve_host_device(host_device)
    print(f"JAX backend verified: {host_device}; devices={jax.devices()}")
    master_seed = int(os.getenv("FIT_JWST_SEED", flags.get("random_seed", 555)))
    key_master = jax.random.PRNGKey(master_seed)
    print(f"Master random seed: {master_seed}")

    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    # Representative (fixed value / prior centre) orbital inputs.  The
    # white-light model samples whichever of these the specification frees.
    def _planet_centers(key, default=None):
        return jnp.asarray(
            planet_parameter_centers(planet_parameter_specs, key, default),
            dtype=jnp.float64,
        )

    periods = _planet_centers('period')
    n_planets = len(periods)
    period_is_free = any(spec.free for spec in planet_parameter_specs['period'])
    if transit_engine == 'harmonica' and n_planets != 1:
        raise NotImplementedError(
            "Harmonica production fitting and limb-product export currently "
            "support exactly one planet; refusing to silently export only "
            "planet_index=0."
        )
    if transit_engine == 'harmonica' and period_is_free:
        raise NotImplementedError(
            "A free planet.period is supported only with "
            "flags.transit_engine='jaxoplanet'; fix the period for Harmonica."
        )
    for key in ('t0', 'b', 'rprs'):
        if key not in planet_parameter_specs:
            raise KeyError(f"'planet.{key}' is required.")
    _ecc_tmp = _planet_centers('ecc', 0.0)
    _omega_tmp = _planet_centers('omega', 0.0)
    if 'duration' in planet_parameter_specs:
        durations = _planet_centers('duration')
    elif 'a_rs' in planet_parameter_specs:
        durations = harmonica_duration_from_geometry(
            periods, _planet_centers('a_rs'), _planet_centers('b'),
            _planet_centers('rprs'), ecc=_ecc_tmp, omega=_omega_tmp,
        )
        print(f"Computed duration prior from a_rs geometry: {durations}")
        # The data preparation stage uses duration only to identify an
        # out-of-transit normalization window.
        planet_cfg['duration'] = np.asarray(durations, dtype=float).tolist()
    else:
        raise KeyError(
            "'planet.duration' is required unless 'planet.a_rs' is provided."
        )
    t0s = _planet_centers('t0')
    bs = _planet_centers('b')
    rors = _planet_centers('rprs')
    depths = rors**2

    PERIOD_FIXED = periods
    PRIOR_DUR = durations
    PRIOR_T0 = t0s
    PRIOR_B = bs
    PRIOR_RPRS = rors
    PRIOR_DEPTH = depths
    # Centre values used by the data stage for the in-/out-of-transit masks
    # (every epoch t0 + n * period inside the series is masked).
    transit_ephemeris = {
        'period': np.asarray(PERIOD_FIXED, dtype=float).tolist(),
        't0': np.asarray(PRIOR_T0, dtype=float).tolist(),
        'duration': np.asarray(PRIOR_DUR, dtype=float).tolist(),
    }

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
    HARMONICA_ECC = _ecc_tmp
    HARMONICA_OMEGA = _omega_tmp
    if 'a_rs' in planet_parameter_specs:
        HARMONICA_A_RS = _planet_centers('a_rs')
    else:
        HARMONICA_A_RS = harmonica_a_rs_from_duration(
            periods, PRIOR_DUR, PRIOR_B, PRIOR_RPRS,
            ecc=HARMONICA_ECC, omega=HARMONICA_OMEGA,
        )
    HARMONICA_COS_I = _harmonica_cosi_from_b(PRIOR_B, HARMONICA_A_RS, HARMONICA_ECC, HARMONICA_OMEGA)
    HARMONICA_INC = jnp.arccos(HARMONICA_COS_I)
    if jnp.any((HARMONICA_ECC < 0.0) | (HARMONICA_ECC >= 1.0)):
        raise ValueError("`planet.ecc` must satisfy 0 <= ecc < 1.")

    explicit_ld = stellar_cfg.get('ld_coefficients')
    if explicit_ld is not None:
        explicit_ld = np.asarray(explicit_ld, dtype=float)
        if explicit_ld.shape != (2,) or not np.all(np.isfinite(explicit_ld)):
            raise ValueError("stellar.ld_coefficients must contain two finite values.")
        if ld_prior_mode != 'fixed':
            raise ValueError(
                "stellar.ld_coefficients is an explicit fixed profile and requires "
                "flags.ld_prior='fixed'."
            )
        sld = None
    else:
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

    def _explicit_ld_grid(wavelengths=None):
        if wavelengths is None:
            return jnp.asarray(explicit_ld, dtype=jnp.float64)
        return jnp.broadcast_to(
            jnp.asarray(explicit_ld, dtype=jnp.float64),
            (len(np.atleast_1d(wavelengths)), 2),
        )

    mini_instrument = detector_label(instrument, nrs=nrs, order=order)

    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    lr_label = 'no' if low_resolution_bins is None else str(low_resolution_bins)
    if bins == resolution:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{lr_label}LR_{high_resolution_bins}HR.pkl'
    elif bins == pixels:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{lr_label}pix_{high_resolution_bins}pix.pkl'

    if low_resolution_bins is None:
        lr_bin_str = 'nolr'
    else:
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
            "planet_specs": {
                key: [spec.__dict__ for spec in specs]
                for key, specs in planet_parameter_specs.items()
            },
            "transit_ephemeris": transit_ephemeris,
            "wavelength_filter": cfg.get("wavelength_filter", {}),
            "wavelength_masks": cfg.get("wavelength_masks"),
            "exclude_times": [list(pair) for pair in exclude_times],
            "exclude_integrations": [list(pair) for pair in exclude_integrations],
            "reference_grids": reference_grid_identities,
        },
    )
    spectro_artifact_set = ArtifactSet(
        stage="spectro_data",
        manifest_path=spectro_data_manifest,
        fingerprint=spectro_data_fingerprint,
        required_paths=(spectro_data_file,),
    )

    def _compute_spectro_data():
        return process_spectroscopy_data(
            instrument, input_dir, output_dir, planet_str, cfg, fits_file,
            transit_ephemeris=transit_ephemeris,
            exclude_times=exclude_times,
            exclude_integrations=exclude_integrations,
        )

    def _save_spectro_data(data_to_save):
        temporary_data = (
            f"{spectro_data_file}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        data_to_save.save(temporary_data)
        os.replace(temporary_data, spectro_data_file)

    data = load_or_compute_artifact_set(
        spectro_artifact_set,
        lambda: SpectroData.load(spectro_data_file),
        _compute_spectro_data,
        _save_spectro_data,
        load_errors=(
            OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError,
        ),
        on_loaded=lambda: print("Reusing fingerprinted spectroscopy data cache."),
        on_load_error=lambda: print(
            "Spectroscopy data cache is unreadable; rebuilding atomically."
        ),
    )
    
    print("Data loaded.")
    print(f"Data length: {data.time.shape}")
    




    wl_mad_mask, bestfit_params_wl_df, wl_geometry_handoff = run_white_light_stage(
        HARMONICA_A_RS=locals().get('HARMONICA_A_RS'),
        HARMONICA_ECC=locals().get('HARMONICA_ECC'),
        HARMONICA_ODD_HARMONICS=locals().get('HARMONICA_ODD_HARMONICS'),
        HARMONICA_OMEGA=locals().get('HARMONICA_OMEGA'),
        NUTS_KWARGS=locals().get('NUTS_KWARGS'),
        PERIOD_FIXED=locals().get('PERIOD_FIXED'),
        PRIOR_B=locals().get('PRIOR_B'),
        PRIOR_DUR=locals().get('PRIOR_DUR'),
        PRIOR_RPRS=locals().get('PRIOR_RPRS'),
        PRIOR_T0=locals().get('PRIOR_T0'),
        _engine_wl_kw=locals().get('_engine_wl_kw'),
        _explicit_ld_grid=locals().get('_explicit_ld_grid'),
        cfg=locals().get('cfg'),
        create_whitelight_model=locals().get('create_whitelight_model'),
        data=locals().get('data'),
        derive_geometry=locals().get('derive_geometry'),
        detrending_type=locals().get('detrending_type'),
        explicit_ld=locals().get('explicit_ld'),
        harmonica_wl_nuts_kwargs=locals().get('harmonica_wl_nuts_kwargs'),
        instrument=locals().get('instrument'),
        instrument_full_str=locals().get('instrument_full_str'),
        jaxoplanet_kernel=locals().get('jaxoplanet_kernel'),
        jump_guess=locals().get('jump_guess'),
        key_master=locals().get('key_master'),
        ld_prior_mode=locals().get('ld_prior_mode'),
        ld_profile=locals().get('ld_profile'),
        ld_uniform_basis=locals().get('ld_uniform_basis'),
        n_planets=locals().get('n_planets'),
        order=locals().get('order'),
        output_dir=locals().get('output_dir'),
        param_method=locals().get('param_method'),
        planet_parameter_specs=locals().get('planet_parameter_specs'),
        planet_str=locals().get('planet_str'),
        prepare_laplace_metric=locals().get('prepare_laplace_metric'),
        quadratic_uniform_physical_bounds=locals().get('quadratic_uniform_physical_bounds'),
        sld=locals().get('sld'),
        spot_amp=locals().get('spot_amp'),
        spot_amp2=locals().get('spot_amp2'),
        spot_mu=locals().get('spot_mu'),
        spot_mu2=locals().get('spot_mu2'),
        spot_sigma=locals().get('spot_sigma'),
        spot_sigma2=locals().get('spot_sigma2'),
        stellar_cfg=locals().get('stellar_cfg'),
        step_width_days=locals().get('step_width_days'),
        step_width_mode=locals().get('step_width_mode'),
        surface_config=locals().get('surface_config'),
        t_jump_guess=locals().get('t_jump_guess'),
        transit_engine=locals().get('transit_engine'),
        uses_surface_model=locals().get('uses_surface_model'),
        whitelight_geometry_estimator=locals().get('whitelight_geometry_estimator'),
        whitelight_laplace_options=locals().get('whitelight_laplace_options'),
        whitelight_ld_parameterization=locals().get('whitelight_ld_parameterization'),
        whitelight_mass_matrix=locals().get('whitelight_mass_matrix'),
        whitelight_mcmc_kwargs=locals().get('whitelight_mcmc_kwargs'),
        whitelight_sigma=locals().get('whitelight_sigma'),
        whitelight_trend_parameterization=locals().get('whitelight_trend_parameterization'),
        whitelight_two_spot_ordering=locals().get('whitelight_two_spot_ordering'),
    )
    if wl_geometry_handoff is None:
        raise RuntimeError(
            "White-light fitting completed without a valid fixed-geometry "
            "handoff artifact."
        )
    fixed_geometry = wl_geometry_handoff["geometry"]
    # A free white-light period is fixed at its posterior median from here on,
    # exactly like t0, b, and duration.
    PERIOD_FIXED = jnp.asarray(fixed_geometry["period"], dtype=jnp.float64)
    if period_is_free:
        print(
            "Spectroscopic stages use the white-light posterior median period "
            f"{np.asarray(PERIOD_FIXED, dtype=float).tolist()}."
        )
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
    need_lowres_analysis = need_lowres
    
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
        if {'t_jump', 'jump', 'width'}.issubset(bestfit_params_wl_df.columns):
            t_jump = bestfit_params_wl_df['t_jump'].values[0]
            jump = bestfit_params_wl_df['jump'].values[0]
            width = bestfit_params_wl_df['width'].values[0]
            if not np.isnan(t_jump) and not np.isnan(jump) and not np.isnan(width):
                jump_trend = jump * _soft_step_np(
                    np.array(wl_time_good), t_jump, width
                )
    valid = None
    if uses_surface_model:
        from .surface import _spectroscopic_surface_config
        surface_config = _spectroscopic_surface_config(
            surface_config, planet_parameter_specs,
        )
        _engine_spectro_kw = {**_engine_spectro_kw, 'surface_config': surface_config}

    _stage_result = run_low_resolution_stage(
        A_RS_BASE=locals().get('A_RS_BASE'),
        B_BASE=locals().get('B_BASE'),
        COSI_BASE=locals().get('COSI_BASE'),
        DEPTH_BASE=locals().get('DEPTH_BASE'),
        DURATION_BASE=locals().get('DURATION_BASE'),
        HARMONICA_ECC=locals().get('HARMONICA_ECC'),
        HARMONICA_ODD_HARMONICS=locals().get('HARMONICA_ODD_HARMONICS'),
        HARMONICA_OMEGA=locals().get('HARMONICA_OMEGA'),
        PERIOD_FIXED=locals().get('PERIOD_FIXED'),
        RORS_BASE=locals().get('RORS_BASE'),
        T0_BASE=locals().get('T0_BASE'),
        _align_trend_to_time=_align_trend_to_time,
        _engine_spectro_kw=locals().get('_engine_spectro_kw'),
        _explicit_ld_grid=locals().get('_explicit_ld_grid'),
        bestfit_params_wl_df=locals().get('bestfit_params_wl_df'),
        build_transit_window_indices=locals().get('build_transit_window_indices'),
        cfg=locals().get('cfg'),
        config_path=locals().get('config_path'),
        chunk_mode=locals().get('chunk_mode'),
        chunk_parallel_job_count=locals().get('chunk_parallel_job_count'),
        chunk_parallel_job_index=locals().get('chunk_parallel_job_index'),
        create_vectorized_model=locals().get('create_vectorized_model'),
        data=locals().get('data'),
        detrending_type=locals().get('detrending_type'),
        exp_trend=locals().get('exp_trend'),
        explicit_ld=locals().get('explicit_ld'),
        fixed_tau_spectro=locals().get('fixed_tau_spectro'),
        flags=locals().get('flags'),
        harmonica_lr_nuts_kwargs=locals().get('harmonica_lr_nuts_kwargs'),
        harmonica_spectro_fit_jitter=locals().get('harmonica_spectro_fit_jitter'),
        harmonica_spectro_odd_frac_sigma=locals().get('harmonica_spectro_odd_frac_sigma'),
        harmonica_spectro_parameterization=locals().get('harmonica_spectro_parameterization'),
        instrument=locals().get('instrument'),
        instrument_full_str=locals().get('instrument_full_str'),
        jaxoplanet_kernel=locals().get('jaxoplanet_kernel'),
        jaxoplanet_lr_nuts_kwargs=locals().get('jaxoplanet_lr_nuts_kwargs'),
        jump_trend=locals().get('jump_trend'),
        key_map_lr=locals().get('key_map_lr'),
        key_mcmc_lr=locals().get('key_mcmc_lr'),
        ld_prior_mode=locals().get('ld_prior_mode'),
        ld_profile=locals().get('ld_profile'),
        ld_uniform_basis=locals().get('ld_uniform_basis'),
        lowres_mcmc_kwargs=locals().get('lowres_mcmc_kwargs'),
        lr_artifact_stem=locals().get('lr_artifact_stem'),
        lr_bin_str=locals().get('lr_bin_str'),
        max_harmonic_order=locals().get('max_harmonic_order'),
        n_planets=locals().get('n_planets'),
        need_lowres_analysis=locals().get('need_lowres_analysis'),
        order=locals().get('order'),
        output_dir=locals().get('output_dir'),
        param_method=locals().get('param_method'),
        planet_str=locals().get('planet_str'),
        plots_mode=locals().get('plots_mode'),
        sld=locals().get('sld'),
        spec_good_mask=locals().get('spec_good_mask'),
        spectro_fixed_timescale_trends=locals().get('spectro_fixed_timescale_trends'),
        spectro_ld_parameterization=locals().get('spectro_ld_parameterization'),
        spectro_max_divergences=locals().get('spectro_max_divergences'),
        spectro_min_depth_ess=locals().get('spectro_min_depth_ess'),
        spectro_sampler=locals().get('spectro_sampler'),
        spectroscopic_sigma=locals().get('spectroscopic_sigma'),
        spot_trend=locals().get('spot_trend'),
        spot_trend2=locals().get('spot_trend2'),
        stellar_cfg=locals().get('stellar_cfg'),
        surface_config=locals().get('surface_config'),
        transit_engine=locals().get('transit_engine'),
        transit_window_optimization=locals().get('transit_window_optimization'),
        trend_inference=locals().get('trend_inference'),
        uses_surface_model=locals().get('uses_surface_model'),
        valid=locals().get('valid'),
        vmap_chunk_size_lr=locals().get('vmap_chunk_size_lr'),
        whitelight_geometry_estimator=locals().get('whitelight_geometry_estimator'),
        wl_geometry_handoff=locals().get('wl_geometry_handoff'),
        wl_mad_mask=locals().get('wl_mad_mask'),
        wl_time_good=locals().get('wl_time_good'),
    )
    if _stage_result is None:
        return
    valid, spot_trend, spot_trend2, jump_trend = _stage_result

    if analysis_stage == 'prep':
        print("\nPrep stage complete. Skipping high-resolution analysis.")
        return


    if run_high_resolution_stage(
        A_RS_BASE=locals().get('A_RS_BASE'),
        B_BASE=locals().get('B_BASE'),
        COSI_BASE=locals().get('COSI_BASE'),
        DEPTH_BASE=locals().get('DEPTH_BASE'),
        DURATION_BASE=locals().get('DURATION_BASE'),
        HARMONICA_ECC=locals().get('HARMONICA_ECC'),
        HARMONICA_ODD_HARMONICS=locals().get('HARMONICA_ODD_HARMONICS'),
        HARMONICA_OMEGA=locals().get('HARMONICA_OMEGA'),
        PERIOD_FIXED=locals().get('PERIOD_FIXED'),
        RORS_BASE=locals().get('RORS_BASE'),
        T0_BASE=locals().get('T0_BASE'),
        _align_trend_to_time=_align_trend_to_time,
        _engine_spectro_kw=locals().get('_engine_spectro_kw'),
        _explicit_ld_grid=locals().get('_explicit_ld_grid'),
        bestfit_params_wl_df=locals().get('bestfit_params_wl_df'),
        build_transit_window_indices=locals().get('build_transit_window_indices'),
        config_path=locals().get('config_path'),
        chunk_mode=locals().get('chunk_mode'),
        chunk_parallel_job_count=locals().get('chunk_parallel_job_count'),
        chunk_parallel_job_index=locals().get('chunk_parallel_job_index'),
        create_vectorized_model=locals().get('create_vectorized_model'),
        data=locals().get('data'),
        detrending_type=locals().get('detrending_type'),
        exp_trend=locals().get('exp_trend'),
        explicit_ld=locals().get('explicit_ld'),
        fixed_tau_spectro=locals().get('fixed_tau_spectro'),
        flags=locals().get('flags'),
        harmonica_hr_nuts_kwargs=locals().get('harmonica_hr_nuts_kwargs'),
        harmonica_spectro_fit_jitter=locals().get('harmonica_spectro_fit_jitter'),
        harmonica_spectro_odd_frac_sigma=locals().get('harmonica_spectro_odd_frac_sigma'),
        harmonica_spectro_parameterization=locals().get('harmonica_spectro_parameterization'),
        highres_mcmc_kwargs=locals().get('highres_mcmc_kwargs'),
        hr_artifact_stem=locals().get('hr_artifact_stem'),
        hr_bin_str=locals().get('hr_bin_str'),
        instrument=locals().get('instrument'),
        instrument_full_str=locals().get('instrument_full_str'),
        jaxoplanet_hr_nuts_kwargs=locals().get('jaxoplanet_hr_nuts_kwargs'),
        jaxoplanet_kernel=locals().get('jaxoplanet_kernel'),
        jump_trend=locals().get('jump_trend'),
        key_map_hr=locals().get('key_map_hr'),
        key_mcmc_hr=locals().get('key_mcmc_hr'),
        ld_prior_mode=locals().get('ld_prior_mode'),
        ld_profile=locals().get('ld_profile'),
        ld_uniform_basis=locals().get('ld_uniform_basis'),
        max_harmonic_order=locals().get('max_harmonic_order'),
        n_planets=locals().get('n_planets'),
        order=locals().get('order'),
        output_dir=locals().get('output_dir'),
        param_method=locals().get('param_method'),
        planet_str=locals().get('planet_str'),
        plots_mode=locals().get('plots_mode'),
        sld=locals().get('sld'),
        spec_good_mask=locals().get('spec_good_mask'),
        spectro_fixed_timescale_trends=locals().get('spectro_fixed_timescale_trends'),
        spectro_ld_parameterization=locals().get('spectro_ld_parameterization'),
        spectro_max_divergences=locals().get('spectro_max_divergences'),
        spectro_min_depth_ess=locals().get('spectro_min_depth_ess'),
        spectro_sampler=locals().get('spectro_sampler'),
        spot_trend=locals().get('spot_trend'),
        spot_trend2=locals().get('spot_trend2'),
        stellar_cfg=locals().get('stellar_cfg'),
        surface_config=locals().get('surface_config'),
        transit_engine=locals().get('transit_engine'),
        transit_window_optimization=locals().get('transit_window_optimization'),
        trend_inference=locals().get('trend_inference'),
        uses_surface_model=locals().get('uses_surface_model'),
        valid=locals().get('valid'),
        vmap_chunk_size_hr=locals().get('vmap_chunk_size_hr'),
        whitelight_geometry_estimator=locals().get('whitelight_geometry_estimator'),
        wl_geometry_handoff=locals().get('wl_geometry_handoff'),
        wl_mad_mask=locals().get('wl_mad_mask'),
        wl_time_good=locals().get('wl_time_good'),
    ) is None:
        return
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
