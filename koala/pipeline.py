from .output_paths import OutputPaths, state_path
from koala.console import debug, info, configure, koala_version, finish_run, grid_description, reading, joint_display, notice_once
from .paths import DEFAULT_LD_DATA_PATH

from types import SimpleNamespace
import os
import sys
import glob
import pickle
import uuid
import time
import tempfile
import warnings


import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.7
from .limb_darkening import CachedStellarLimbDarkening as StellarLimbDarkening
import argparse
from .datacube import (
    SpectroData, process_spectroscopy_data,
)
from koala.exclusions import resolve_exclusions, has_cut_phase_directive
from .host_devices import REQUESTED_ENV as CPU_HOST_DEVICES_ENV, verify_cpu_host_devices
from models.common import compute_transit_model_auto
from models.harmonica.core import (
    harmonica_a_rs_from_duration,
    harmonica_duration_from_geometry,
    harmonica_odd_coeff_specs,
)
from models.trends import spot_crossing
from models.gp import compute_lc_linear_gp_mean, compute_lc_explinear_gp_mean
from models.limb_darkening_config import validate_ld_profile, validate_gaussian_ld_width, validate_uniform_ld_bounds, GAUSSIAN_LD_WIDTH
from models.model_building import kipping_u_from_q


from .constants import (
    HARMONICA_LIMB_PRODUCT_CONVENTION, HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION,
    SCIENCE_ARTIFACT_SCHEMA_VERSION, ADVANCED_FLAGS, FLAG_TIERS, HARMONICA_ODD_HARMONICS,
    INTERNAL_FLAGS, KNOWN_FLAGS, PUBLIC_FLAGS, SPECTRO_DATA_TARGET_REVISION, _JUMP_WIDTH_DAYS,
)
from .config import (
    UnknownFlagWarning, _validate_flag_keys, _resolve_ld_prior_mode, _resolve_ld_configuration,
    _resolve_ld_parameterization, load_config, _resolve_stage_vmap_width,
    _resolve_stage_mcmc_kwargs, _resolve_whitelight_laplace_options,
    _resolve_whitelight_mass_matrix, _resolve_whitelight_trend_parameterization,
    _resolve_spectro_fixed_timescale_trends,
    _resolve_compile_cache_options, _resolve_ld_prior_cache_options,
    _resolve_harmonica_stage_nuts_kwargs, _resolve_jaxoplanet_spectro_nuts_kwargs,
    _resolve_spectro_mass_matrix,
    parse_planet_parameter_specs, planet_parameter_centers, check_ephemeris_covers_data,
    describe_planet_parameter_specs, _spectro_detrend_type, parse_trend_priors,
)
from .artifacts import (
    _science_artifact_fingerprint, _file_content_identity, _optional_file_content_identity,
    ArtifactSet, load_or_compute_artifact_set, _science_artifact_manifest_matches,
    _write_science_artifact_manifest, _directory_metadata_identity,
)


from .outputs import (
    _soft_step_np, _save_mcmc_diagnostics,
)
from .geometry import (
    _geometry_chain_quality, _continue_mcmc_until_geometry_gate,
    _write_whitelight_geometry_handoff, _load_whitelight_geometry_handoff,
)
from .sampling import (
    _concatenate_chunk_samples,
    _build_numpyro_mcmc, get_samples, _chunk_checkpoint_fingerprint,
    _annotate_sampler_diagnostics, _load_chunk_samples, _spectro_failed_lanes,
    _spectro_sampler_swap_order, get_samples_chunked, _slice_by_channel,
    _write_or_validate_checkpoint_manifest, _run_sampling_stage, _build_spectroscopic_model,
)
from .instruments import (
    normalize_instrument, resolve_detector, detector_label,
)
from .limb_darkening import (
    _quadratic_uniform_initial_sites, get_or_build_power2_ld_prior, _power2_ld_initial_sites,
    _power2_ld_optimization_sites, _validate_whitelight_optimized_start, _build_axis,
    _posterior_quadratic_ld_median, resolve_ld_model, is_stagger, ld_model_metallicity, STAGGER_MAX_FEH,
)
from .harmonica_products import (
    _LEGACY_LIMB_SCHEMA_WARNING_EMITTED, _validate_harmonica_spectro_parameterization,
    _harmonica_spectro_artifact_stem, _harmonica_cosi_from_b,
    _harmonica_coefficients_to_half_area_init, _harmonica_model_radius_samples,
    _harmonica_checkpoint_prefix, _harmonica_limb_product_samples,
    load_harmonica_limb_dataframe, build_harmonica_limb_dataframe,
    _save_harmonica_limb_posterior_samples, save_harmonica_limb_products,
)

from .spectroscopy import (
    run_low_resolution_stage, run_high_resolution_stage,
)

from .white_light import (
    run_white_light_stage,
)


from .surface import (
    _attach_jaxoplanet_eval_metadata, _attach_surface_eval_metadata,
    _seed_surface_spectroscopic_init, _surface_basis_on_time_mask, _select_transit_eval_params,
)


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
            f'Could not set jax_platform_name={platform!r}: {exc}', RuntimeWarning,
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
        raise ValueError(f"host_device must be 'gpu', 'cpu' or 'auto', got {requested!r}")
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

def resolve_input_path(cfg, planet_name):
    base = cfg.get('path', '.')
    filename = cfg.get('input_file', cfg.get('fits_file'))
    if filename is None:
        raise KeyError("'input_file' (or the older 'fits_file') is required.")
    original = os.path.join(base, cfg.get('input_dir', planet_name + '_NIRSPEC'), filename)
    direct = os.path.join(base, filename)
    if 'input_dir' not in cfg and not os.path.exists(original) and os.path.exists(direct):
        return direct
    return original


def _align_trend_to_time(trend, trend_time, target_time):
    trend = np.asarray(trend)
    target_time = np.asarray(target_time)
    trend_time = np.asarray(trend_time)
    if len(trend) == len(target_time):
        return trend
    return np.interp(target_time, trend_time, trend)


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
    debug("No low-resolution grid configured; the low-resolution stage is skipped.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed diagnostics')
    args = parser.parse_args()

    cfg = load_config(args.config)
    return run(cfg, config_path=args.config, verbose=args.verbose)


def run(cfg, config_path=None, execution_context=None, *, prepare_only=False,
        white_light_posterior=None, verbose=False, prepared_white_light=None):
    started = time.perf_counter()
    configure(verbose)
    presentation = joint_display.get()
    detail = debug if white_light_posterior is not None or presentation is not None else info
    reading_detail = info if presentation is not None and presentation.get('reading') else detail
    if "joint" in cfg:
        from .joint import run_joint
        return run_joint(cfg, config_path=config_path, verbose=verbose)
    from models.spectro_trends import resolve_spectral_trend_defaults
    flags = resolve_spectral_trend_defaults(cfg.get('flags', {}))
    cfg['flags'] = flags
    _validate_flag_keys(flags)
    instrument = normalize_instrument(cfg['instrument'])
    cfg['instrument'] = instrument
    nrs, order = resolve_detector(instrument, cfg)

    planet_cfg = cfg['planet']
    configured_planet_keys = frozenset(planet_cfg)
    stellar_cfg = cfg['stellar']
    if 'period' not in planet_cfg:
        raise KeyError("'planet.period' is required.")
    planet_parameter_specs = parse_planet_parameter_specs(planet_cfg)
    trend_priors = parse_trend_priors(cfg.get('trend_priors'))
    if trend_priors:
        for line in describe_planet_parameter_specs(
            {key: (spec,) for key, spec in trend_priors.items()},
            title="White-light trend priors (trend_priors)",
        ):
            debug(line)
    from models.jaxoplanet.config import parse_surface_config
    _early_engine = str(flags.get('transit_engine', 'jaxoplanet')).strip().lower()
    _early_model = str(flags.get('light_curve_model', 'transit')).strip().lower()
    _early_param_method = (
        'duration' if 'duration' in planet_parameter_specs else 'a_rs'
    )
    surface_config = parse_surface_config(
        flags, planet_cfg, stellar_cfg, len(planet_parameter_specs['period']),
        parameter_specs=planet_parameter_specs, param_method=_early_param_method,
    )
    for line in describe_planet_parameter_specs(planet_parameter_specs):
        debug(line)
    _surface_spec_names = [k for k in surface_config if k.endswith('_spec')]
    if _surface_spec_names:
        _surface_specs = {
            name[:-5]: surface_config[name] for name in _surface_spec_names
        }
        for line in describe_planet_parameter_specs(
            _surface_specs, title="Planet surface parameters (model units)"
        ):
            debug(line)
    is_prism = instrument == 'NIRSPEC/PRISM'
    is_explinear = 'explinear' in str(flags.get('detrending_type', 'linear'))
    flags.setdefault('spectro_sampler', 'independent_nuts')
    whitelight_trend_parameterization = _resolve_whitelight_trend_parameterization(
        flags.get('detrending_type', 'linear')
    )
    whitelight_two_spot_ordering = 'legacy'
    whitelight_mass_matrix = _resolve_whitelight_mass_matrix(
        flags.get('detrending_type', 'linear'), flags.get('whitelight_mass_matrix'),
    )
    spectro_min_depth_ess = float(flags.get('spectro_min_depth_ess', 400))
    spectro_max_divergences = int(flags.get('spectro_max_divergences', 0))
    compilation_cache_dir = _resolve_compile_cache_options(flags)["cache_dir"]
    if compilation_cache_dir is not None:
        try:
            os.makedirs(compilation_cache_dir, exist_ok=True)
            probe = tempfile.NamedTemporaryFile(dir=compilation_cache_dir, delete=True)
            probe.close()
        except OSError as error:
            info('The compilation cache is unavailable; continuing without it.')
            debug(
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
    fits_file = resolve_input_path(cfg, planet_str)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    configure(verbose, output_dir, config_path)
    plots_mode = str(flags.get('plots', 'full')).strip().lower()
    if presentation is not None and presentation.get('plots') is not None:
        plots_mode = str(presentation['plots']).strip().lower()
    if plots_mode not in {'full', 'minimal'}:
        raise ValueError("flags.plots must be one of {'full', 'minimal'}.")
    if plots_mode == 'minimal':
        debug("Minimal plots enabled: diagnostic noise/poly/light-curve grids skipped.")

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
    spectro_fixed_timescale_trends = _resolve_spectro_fixed_timescale_trends(
        flags, detrending_type, transit_engine)
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
    spectro_cadence_reduction = str(flags.get('spectro_cadence_reduction', 'auto')).strip().lower()
    if spectro_cadence_reduction not in {'auto', 'off'}:
        raise ValueError("flags.spectro_cadence_reduction must be 'auto' or 'off'.")
    spectro_transit_grid = str(flags.get('spectro_transit_grid', 'off')).strip().lower()
    if spectro_transit_grid not in {'auto', 'off'}:
        raise ValueError("flags.spectro_transit_grid must be 'auto' or 'off'.")
    spectro_transit_grid_nodes = int(flags.get('spectro_transit_grid_nodes', 769))
    if spectro_transit_grid_nodes < 13:
        raise ValueError('flags.spectro_transit_grid_nodes must be at least 13.')
    vmap_chunk = flags.get('spectro_chunk_size', flags.get('vmap_chunk', False))
    vmap_chunk_size = None
    if isinstance(vmap_chunk, str) and vmap_chunk.lower() == 'auto':
        vmap_chunk_size = 'auto'
    elif isinstance(vmap_chunk, (int, float)) and not isinstance(vmap_chunk, bool):
        vmap_chunk_size = int(vmap_chunk)
    elif vmap_chunk is True:
        vmap_chunk_size = 50
    elif spectro_sampler in {'independent_nuts', 'independent_hmc'}:
        # With CPU host devices, one channel per device: on native HATS-6
        # PRISM (24 devices) two lanes per device cost 33 s per channel
        # against 27 s for one, because a device's lanes advance in lockstep.
        host_devices = int(os.environ.get(CPU_HOST_DEVICES_ENV, 0))
        vmap_chunk_size = (
            'auto' if transit_engine == 'jaxoplanet'
            and str(cfg.get('host_device', 'auto')).lower() == 'gpu' else host_devices or 40
        )
        debug(
            f"flags.spectro_sampler='{spectro_sampler}' without "
            f"flags.vmap_chunk; using resident width {vmap_chunk_size}."
        )
    if vmap_chunk_size is not None and vmap_chunk_size != 'auto' and vmap_chunk_size < 1:
        raise ValueError("flags.vmap_chunk must resolve to an integer >= 1.")
    # Measured widths are resolved only when their stage is actually about to
    # run, after the requested JAX backend and exact cadence/model workload are
    # known.  This avoids initializing a GPU for a CPU/prep-only job and keeps
    # irrelevant stage manifests from aborting the run.
    vmap_chunk_size_lr = vmap_chunk_size
    vmap_chunk_size_hr = vmap_chunk_size
    analysis_stage = str(flags.get('analysis_stage', 'all')).lower()
    chunk_mode = str(flags.get('chunk_mode', 'serial')).lower()
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
    trend_inference = str(flags.get('trend_inference', 'sampled_uniform')).lower()
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
    ld_profile, ld_prior_mode, ld_uniform_basis = _resolve_ld_configuration(flags, stellar_cfg)
    gaussian_ld_width = validate_gaussian_ld_width(flags.get('ld_gaussian_sigma', GAUSSIAN_LD_WIDTH))
    spectro_ld_parameterization = _resolve_ld_parameterization(
        ld_prior_mode, ld_profile, flags.get('ld_parameterization'),
    )
    whitelight_ld_parameterization = spectro_ld_parameterization
    ld_uniform_coefficient_bounds = validate_uniform_ld_bounds(
        flags.get('ld_uniform_coefficient_bounds', (0.0, 1.0)))
    if ('ld_uniform_coefficient_bounds' in flags and ld_prior_mode != 'fixed_u1'
            and (ld_profile != 'quadratic' or ld_uniform_basis != 'coefficients')):
        raise ValueError('flags.ld_uniform_coefficient_bounds requires quadratic coefficients or fixed_u1.')
    quadratic_uniform_physical_bounds = (
        (-1.5, 2.0)
        if ld_uniform_basis == 'uplus_uminus' and ld_prior_mode != 'fixed_u1'
        else (-1.0, 2.0) if ld_uniform_basis == 'kipping'
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
        and ld_prior_mode not in {'fixed', 'sing', 'uniform', 'fixed_u1'}
    ):
        raise ValueError(
            "Harmonica quadratic limb darkening supports fixed direct u1/u2 "
            "coefficients, uniform, fixed_u1, or flags.ld_prior: 'sing'."
        )
    if ld_prior_mode == 'uniform' and ld_profile != 'power2':
        debug(
            "[LD prior] flags.ld_prior='uniform' was requested for a non-power2 profile; "
            "the underlying model will sample the free LD coefficients uniformly within its native bounds.",
            flush=True,
        )
    max_harmonic_order = int(flags.get('harmonica_max_order', 1))
    harmonica_spectro_parameterization = (
        _validate_harmonica_spectro_parameterization(
            flags.get('harmonica_spectro_parameterization', 'delta_r'), max_harmonic_order,
        )
    )
    harmonica_spectro_fit_jitter = bool(flags.get('harmonica_spectro_fit_jitter', True))
    harmonica_spectro_odd_frac_sigma = float(flags.get('harmonica_spectro_odd_frac_sigma', 0.1))
    if (
        not np.isfinite(harmonica_spectro_odd_frac_sigma)
        or harmonica_spectro_odd_frac_sigma <= 0
    ):
        raise ValueError('flags.harmonica_spectro_odd_frac_sigma must be finite and > 0.')
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

    # Sample the configured WL parameter; every engine evaluates its derived orbit.
    uses_surface_model = bool(surface_config['model'] != 'transit' or surface_config['spots'])
    param_method = 'duration' if 'duration' in planet_parameter_specs else 'a_rs'
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
    jaxoplanet_lr_nuts_kwargs = jaxoplanet_hr_nuts_kwargs = None
    build_transit_window_indices = None
    if transit_engine == 'harmonica':
        from models.harmonica import create_whitelight_model, create_vectorized_model, NUTS_KWARGS, derive_geometry
        _engine_wl_kw = {
            'gaussian_ld_width': gaussian_ld_width,
            'max_harmonic_order': max_harmonic_order,
            'param_method': param_method,
            'ld_profile': ld_profile,
            'trend_parameterization': whitelight_trend_parameterization,
            'two_spot_ordering': whitelight_two_spot_ordering,
        }
        _engine_spectro_kw = {
            'gaussian_ld_width': gaussian_ld_width,
            'trend_basis': flags.get('spectro_trend_basis', 'elapsed_days'),
            'trend_bounds': flags.get('spectro_trend_bounds'),
            'max_harmonic_order': max_harmonic_order,
            'odd_parameterization': harmonica_spectro_parameterization,
            'fit_jitter': harmonica_spectro_fit_jitter,
            'odd_frac_sigma': harmonica_spectro_odd_frac_sigma,
            'ld_profile': ld_profile,
        }
        for options in (_engine_wl_kw, _engine_spectro_kw):
            options.update(ld_uniform_basis=ld_uniform_basis,
                           ld_uniform_coefficient_bounds=tuple(ld_uniform_coefficient_bounds))
        if whitelight_ld_parameterization == 'maxted_smooth':
            _engine_wl_kw['ld_parameterization'] = 'maxted_smooth'
        if spectro_ld_parameterization == 'maxted_smooth':
            _engine_spectro_kw['ld_parameterization'] = 'maxted_smooth'
    else:
        from models.jaxoplanet import (
            NUTS_KWARGS,
            build_transit_window_indices,
            create_vectorized_model,
            create_whitelight_model,
            derive_geometry,
        )
        _engine_wl_kw = {
            'gaussian_ld_width': gaussian_ld_width,
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
            'gaussian_ld_width': gaussian_ld_width,
            'jitter_prior': jitter_prior,
            'trend_basis': flags.get('spectro_trend_basis', 'elapsed_days'),
            'trend_bounds': flags.get('spectro_trend_bounds'),
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
        power2_evaluation = ('exact' if ld_profile == 'power2'
            and surface_config.get('model') == 'transit' and not surface_config.get('spots')
            and ld_prior_mode != 'interpolated' else 'polynomial')
        _engine_wl_kw['power2_evaluation'] = power2_evaluation
        _engine_spectro_kw['power2_evaluation'] = power2_evaluation
        if execution_context is not None:
            jaxoplanet_kernel = execution_context.kernel(ld_profile, param_method)
            _engine_wl_kw['jaxoplanet_kernel'] = jaxoplanet_kernel
            _engine_spectro_kw['jaxoplanet_kernel'] = jaxoplanet_kernel
            create_whitelight_model = execution_context.factory(create_whitelight_model)
            create_vectorized_model = execution_context.factory(create_vectorized_model)
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
            'lowres', NUTS_KWARGS, independent=independent_backend, hmc=hmc_backend,
            is_prism=is_prism, is_explinear=is_explinear,
            mass_matrix=_resolve_spectro_mass_matrix(flags),
        )
        jaxoplanet_hr_nuts_kwargs = _resolve_jaxoplanet_spectro_nuts_kwargs(
            'highres', NUTS_KWARGS, independent=independent_backend, hmc=hmc_backend,
            is_prism=is_prism, is_explinear=is_explinear,
            mass_matrix=_resolve_spectro_mass_matrix(flags),
        )

    if trend_priors:
        # Only configured fits pass the keyword, keeping default model caches unchanged.
        _engine_wl_kw['trend_priors'] = trend_priors
    HARMONICA_ODD_HARMONICS = tuple(
        name for name, _ in harmonica_odd_coeff_specs(max_harmonic_order)
    )

    host_device = cfg.get('host_device', 'auto').lower()
    # The old C++ FFI custom calls are CPU-only. N_c<=1 power-2 and standard
    # direct-u1/u2 quadratic transits use the pure-JAX path on GPU.
    if transit_engine == 'harmonica' and host_device != 'cpu':
        if max_harmonic_order > 1:
            info('This transit model requires the CPU; using the CPU for this run.')
            debug(
                f"Harmonica N_c>{1} requires CPU-only FFI custom calls; "
                f"overriding host_device='{host_device}' to 'cpu'."
            )
            host_device = 'cpu'
        else:
            debug(f'Using pure JAX harmonica GPU path (N_c=1 {ld_profile} LD).')
    host_device = resolve_host_device(host_device)
    verify_cpu_host_devices(jax, host_device)
    debug(f"JAX backend verified: {host_device}; devices={jax.devices()}")
    device_label = (f'CPU, {len(jax.devices("cpu"))} cores' if host_device == 'cpu'
                    else str(jax.devices()[0].device_kind))
    detector_description = detector_label(instrument, nrs=nrs, order=order).replace('order', 'order ')
    if presentation is None:
        info(f'koala {koala_version()} | {planet_str} | {instrument} {detector_description} | {device_label}')
        info('')
    elif presentation.get('reading'):
        if not presentation.get('header_printed'):
            info(f'koala {koala_version()} | {planet_str} | joint fit of {presentation["count"]} observations | {device_label}')
            presentation['header_printed'] = True
            info('')
        info(f'{instrument} {detector_description} ({os.path.basename(config_path)})')
    master_seed = int(flags.get("random_seed", 555))
    key_master = jax.random.PRNGKey(master_seed)
    debug(f"Master random seed: {master_seed}")

    whitelight_sigma = outlier_clip.get('whitelight_sigma', 5)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 0)

    # Representative (fixed value / prior centre) orbital inputs.  The
    # white-light model samples whichever of these the specification frees.
    def _planet_centers(key, default=None):
        return jnp.asarray(
            planet_parameter_centers(planet_parameter_specs, key, default), dtype=jnp.float64,
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
        debug(f"Computed duration prior from a_rs geometry: {durations}")
        # The data preparation stage uses duration only to identify an
        # out-of-transit normalization window.
        planet_cfg['duration'] = np.asarray(durations, dtype=float).tolist()
    else:
        raise KeyError("'planet.duration' is required unless 'planet.a_rs' is provided.")
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


    # Shared orbital geometry inputs used by harmonica and jaxoplanet a_rs parameterizations.
    HARMONICA_ECC = _ecc_tmp
    HARMONICA_OMEGA = _omega_tmp
    if 'a_rs' in planet_parameter_specs:
        HARMONICA_A_RS = _planet_centers('a_rs')
    else:
        HARMONICA_A_RS = harmonica_a_rs_from_duration(
            periods, PRIOR_DUR, PRIOR_B, PRIOR_RPRS, ecc=HARMONICA_ECC, omega=HARMONICA_OMEGA,
        )
    HARMONICA_COS_I = _harmonica_cosi_from_b(PRIOR_B, HARMONICA_A_RS, HARMONICA_ECC, HARMONICA_OMEGA)
    HARMONICA_INC = jnp.arccos(HARMONICA_COS_I)
    if jnp.any((HARMONICA_ECC < 0.0) | (HARMONICA_ECC >= 1.0)):
        raise ValueError("`planet.ecc` must satisfy 0 <= ecc < 1.")

    def _explicit_ld_grid(wavelengths=None):
        if wavelengths is None:
            return jnp.asarray(explicit_ld, dtype=jnp.float64)
        return jnp.broadcast_to(
            jnp.asarray(explicit_ld, dtype=jnp.float64), (len(np.atleast_1d(wavelengths)), 2),
        )

    mini_instrument = detector_label(instrument, nrs=nrs, order=order)

    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    lr_label = 'no' if low_resolution_bins is None else str(low_resolution_bins)
    if bins == resolution:
        spectro_data_file = state_path(output_dir, f'{instrument_full_str}_spectroscopy_data_{lr_label}LR_{high_resolution_bins}HR.pkl')
    elif bins == pixels:
        spectro_data_file = state_path(output_dir, f'{instrument_full_str}_spectroscopy_data_{lr_label}pix_{high_resolution_bins}pix.pkl')

    if low_resolution_bins is None:
        lr_bin_str = 'nolr'
    else:
        lr_bin_str = f'R{low_resolution_bins}' if bins == resolution else f'pix{low_resolution_bins}'
    hr_bin_str = f'R{high_resolution_bins}' if bins == resolution else f'pix{high_resolution_bins}'
    lr_artifact_stem = _harmonica_spectro_artifact_stem(
        f'{instrument_full_str}_{lr_bin_str}', transit_engine,
        harmonica_spectro_parameterization,
    )
    hr_artifact_stem = _harmonica_spectro_artifact_stem(
        f'{instrument_full_str}_{hr_bin_str}', transit_engine,
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
    if execution_context is not None:
        spectro_data_file = execution_context.data_path(spectro_data_fingerprint)
        spectro_data_manifest = f"{spectro_data_file}.manifest.json"
    spectro_artifact_set = ArtifactSet(
        stage='spectro_data', manifest_path=spectro_data_manifest,
        fingerprint=spectro_data_fingerprint, required_paths=(spectro_data_file,),
    )

    def _compute_spectro_data():
        return process_spectroscopy_data(
            instrument, input_dir, output_dir, planet_str, cfg, fits_file,
            transit_ephemeris=transit_ephemeris, exclude_times=exclude_times,
            exclude_integrations=exclude_integrations,
        )

    def _save_spectro_data(data_to_save):
        temporary_data = (
            f"{spectro_data_file}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        data_to_save.save(temporary_data)
        os.replace(temporary_data, spectro_data_file)

    def load_stage_data():
        return load_or_compute_artifact_set(
            spectro_artifact_set,
            lambda: SpectroData.load(spectro_data_file),
            _compute_spectro_data,
            _save_spectro_data,
            load_errors=(OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError),
            on_loaded=lambda: reading_detail("  Reusing the prepared spectra from a previous run"),
            on_load_error=lambda: (
                info("The prepared spectra could not be read; preparing them again."),
                debug("Spectroscopy data cache is unreadable; rebuilding atomically."),
            ),
        )
    with reading(fits_file, enabled=white_light_posterior is None):
        data = (load_stage_data() if execution_context is None else
                execution_context.load_data(spectro_artifact_set, load_stage_data))
        for stage_label, waves, label in (('Low resolution', data.wavelengths_lr, lr_bin_str),
                                         ('Spectroscopy', data.wavelengths_hr, hr_bin_str)):
            if len(waves):
                stage_suffix = ' (low resolution)' if stage_label == 'Low resolution' else ''
                reading_detail(f'  Binned to {len(waves)} channels {grid_description(label)}{stage_suffix}')

    if white_light_posterior is None:
        info('')
    check_ephemeris_covers_data(
        np.asarray(data.wl_time), planet_parameter_specs,
        str(flags.get('light_curve_model', 'transit')).strip().lower(),
    )
    for line in describe_planet_parameter_specs({
            key: entries for key, entries in planet_parameter_specs.items()
            if key in configured_planet_keys}):
        detail(line)
    ld_description = ld_prior_mode + ' priors'
    if ld_profile == 'quadratic' and ld_prior_mode == 'uniform' and ld_uniform_basis == 'uplus_uminus':
        ld_description = 'uniform priors on u1 + u2 and u1 - u2'
    detail(f'Limb darkening: {ld_profile} law with {ld_description}')
    if presentation is None:
        info('')
    explicit_ld = stellar_cfg.get('ld_coefficients')
    if (explicit_ld is None and ld_uniform_basis == 'kipping' and ld_prior_mode == 'uniform'
            and not all(k in stellar_cfg for k in ('teff', 'logg', 'feh'))):
        # Kipping-uniform needs no stellar grid; start from the centre of the (q1, q2) square.
        debug("[LD prior] Kipping q1, q2 ~ U(0, 1) without a stellar grid; starting at q = (0.5, 0.5).",
              flush=True)
        explicit_ld = np.asarray(kipping_u_from_q(np.array([0.5, 0.5])), dtype=float)
        sld = None
    elif explicit_ld is not None:
        explicit_ld = np.asarray(explicit_ld, dtype=float)
        if explicit_ld.shape != (2,) or not np.all(np.isfinite(explicit_ld)):
            raise ValueError("stellar.ld_coefficients must contain two finite values.")
        if ld_uniform_basis == 'kipping':
            if not np.all((explicit_ld >= 0.0) & (explicit_ld <= 1.0)):
                raise ValueError(
                    "With flags.ld_profile='kipping', stellar.ld_coefficients are [q1, q2] in [0, 1]."
                )
            explicit_ld = np.asarray(kipping_u_from_q(explicit_ld), dtype=float)
        if ld_prior_mode not in {'fixed', 'fixed_u1'}:
            raise ValueError(
                "stellar.ld_coefficients is an explicit fixed profile and requires "
                "flags.ld_prior='fixed' or 'fixed_u1'."
            )
        sld = None
    else:
        stellar_feh = stellar_cfg['feh']
        stellar_teff = stellar_cfg['teff']
        stellar_logg = stellar_cfg['logg']
        ld_model = resolve_ld_model(stellar_cfg)
        if stellar_cfg.get('ld_model') is None and not is_stagger(ld_model):
            notice_once('mps1', f'The star is outside the Stagger grid, so limb darkening uses MPS-ATLAS ({ld_model}).')
        elif is_stagger(ld_model) and float(stellar_feh) > STAGGER_MAX_FEH:
            notice_once('stagger-solar', 'Stagger has no models above solar metallicity, '
                                         'so limb darkening uses its [Fe/H] = 0 models.')
        ld_data_path = stellar_cfg.get('ld_data_path', DEFAULT_LD_DATA_PATH)
        ld_interpolate_type = stellar_cfg.get('ld_interpolate_type', 'trilinear')
        stellar_factory = (StellarLimbDarkening if execution_context is None else
                           lambda **kw: execution_context.stellar(StellarLimbDarkening, **kw))
        sld = stellar_factory(
            M_H=ld_model_metallicity(stellar_feh, ld_model), Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
            ld_data_path=ld_data_path, interpolate_type=ld_interpolate_type,
        )

    debug("Data loaded.")
    debug(f"Data length: {data.time.shape}")


    white_light_context = SimpleNamespace(
        execution_context=execution_context, joint_member_products=white_light_posterior is not None,
        HARMONICA_A_RS=HARMONICA_A_RS, HARMONICA_ECC=HARMONICA_ECC,
        HARMONICA_ODD_HARMONICS=HARMONICA_ODD_HARMONICS, HARMONICA_OMEGA=HARMONICA_OMEGA,
        NUTS_KWARGS=NUTS_KWARGS, PERIOD_FIXED=PERIOD_FIXED, PRIOR_B=PRIOR_B,
        PRIOR_DUR=PRIOR_DUR, PRIOR_RPRS=PRIOR_RPRS, PRIOR_T0=PRIOR_T0,
        _engine_wl_kw=_engine_wl_kw, _explicit_ld_grid=_explicit_ld_grid, cfg=cfg,
        create_whitelight_model=create_whitelight_model, data=data,
        derive_geometry=derive_geometry, detrending_type=detrending_type,
        explicit_ld=explicit_ld, harmonica_wl_nuts_kwargs=harmonica_wl_nuts_kwargs,
        instrument=instrument, instrument_full_str=instrument_full_str,
        jaxoplanet_kernel=jaxoplanet_kernel, jump_guess=jump_guess, key_master=key_master,
        ld_prior_mode=ld_prior_mode, ld_profile=ld_profile, ld_uniform_basis=ld_uniform_basis,
        n_planets=n_planets, order=order, output_dir=output_dir, param_method=param_method,
        plots_mode=plots_mode,
        planet_parameter_specs=planet_parameter_specs, planet_str=planet_str,
        trend_priors=trend_priors,
        quadratic_uniform_physical_bounds=quadratic_uniform_physical_bounds, sld=sld,
        spot_amp=spot_amp, spot_amp2=spot_amp2, spot_mu=spot_mu, spot_mu2=spot_mu2,
        spot_sigma=spot_sigma, spot_sigma2=spot_sigma2, stellar_cfg=stellar_cfg,
        step_width_days=step_width_days, step_width_mode=step_width_mode,
        surface_config=surface_config, t_jump_guess=t_jump_guess,
        transit_engine=transit_engine, uses_surface_model=uses_surface_model,
        whitelight_geometry_estimator=whitelight_geometry_estimator,
        whitelight_laplace_options=whitelight_laplace_options,
        whitelight_ld_parameterization=whitelight_ld_parameterization,
        whitelight_mass_matrix=whitelight_mass_matrix,
        whitelight_mcmc_kwargs=whitelight_mcmc_kwargs, whitelight_sigma=whitelight_sigma,
        whitelight_trend_parameterization=whitelight_trend_parameterization,
        whitelight_two_spot_ordering=whitelight_two_spot_ordering,
    )
    if prepare_only:
        return white_light_context
    prepare_hook = getattr(execution_context, 'prepare_white_light', None)
    if prepare_hook is not None:
        prepare_hook(white_light_context)
    wl_mad_mask, bestfit_params_wl_df, wl_geometry_handoff = (
        prepared_white_light if prepared_white_light is not None else run_white_light_stage(
            white_light_context, posterior=white_light_posterior))
    if presentation is None:
        info('')

    if wl_geometry_handoff is None:
        raise RuntimeError(
            'White-light fitting completed without a valid fixed-geometry handoff artifact.',
        )
    fixed_geometry = wl_geometry_handoff["geometry"]
    # A free white-light period is fixed at its posterior median from here on,
    # exactly like t0, b, and duration.
    PERIOD_FIXED = jnp.asarray(fixed_geometry["period"], dtype=jnp.float64)
    if period_is_free:
        debug(
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
    debug(
        "Using white-light geometry handoff "
        f"({wl_geometry_handoff['estimator']}, "
        f"fingerprint={wl_geometry_handoff['artifact_fingerprint_sha256'][:12]}) "
        "for spectroscopic fitting."
    )
    if analysis_stage == 'whitelight':
        finish_run(output_dir, started)
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
            -(np.asarray(wl_time_good) - np.min(np.asarray(wl_time_good))) / fixed_tau_spectro,
        )
    spot_templates = []
    for suffix in ('', '2'):
        template = None
        names = [f'spot_{name}{suffix}' for name in ('amp', 'mu', 'sigma')]
        if f'{suffix}spot' in detrending_type and set(names).issubset(bestfit_params_wl_df.columns):
            amplitude, center, width = (bestfit_params_wl_df[name].values[0] for name in names)
            if not any(np.isnan(value) for value in (amplitude, center, width)):
                template = spot_crossing(wl_time_good, amplitude, center, np.abs(width))
        spot_templates.append(template)
    spot_trend, spot_trend2 = spot_templates
    if 'linear_discontinuity' in detrending_type:
        if {'t_jump', 'jump', 'width'}.issubset(bestfit_params_wl_df.columns):
            t_jump = bestfit_params_wl_df['t_jump'].values[0]
            jump = bestfit_params_wl_df['jump'].values[0]
            width = bestfit_params_wl_df['width'].values[0]
            if not np.isnan(t_jump) and not np.isnan(jump) and not np.isnan(width):
                jump_trend = jump * _soft_step_np(np.array(wl_time_good), t_jump, width)
    valid = None
    if uses_surface_model:
        from .surface import _spectroscopic_surface_config
        surface_config = _spectroscopic_surface_config(surface_config, planet_parameter_specs)
        _engine_spectro_kw = {**_engine_spectro_kw, 'surface_config': surface_config}

    spectro_geometry = SimpleNamespace(
        A_RS_BASE=A_RS_BASE, B_BASE=B_BASE, COSI_BASE=COSI_BASE, DEPTH_BASE=DEPTH_BASE,
        DURATION_BASE=DURATION_BASE, HARMONICA_ECC=HARMONICA_ECC,
        HARMONICA_ODD_HARMONICS=HARMONICA_ODD_HARMONICS, HARMONICA_OMEGA=HARMONICA_OMEGA,
        PERIOD_FIXED=PERIOD_FIXED, RORS_BASE=RORS_BASE, T0_BASE=T0_BASE, n_planets=n_planets,
        param_method=param_method, whitelight_geometry_estimator=whitelight_geometry_estimator,
        wl_geometry_handoff=wl_geometry_handoff,
    )
    spectro_model = SimpleNamespace(
        _engine_spectro_kw=_engine_spectro_kw, create_vectorized_model=create_vectorized_model,
        detrending_type=detrending_type, fixed_tau_spectro=fixed_tau_spectro,
        harmonica_spectro_fit_jitter=harmonica_spectro_fit_jitter,
        harmonica_spectro_odd_frac_sigma=harmonica_spectro_odd_frac_sigma,
        harmonica_spectro_parameterization=harmonica_spectro_parameterization,
        jaxoplanet_kernel=jaxoplanet_kernel, ld_prior_mode=ld_prior_mode,
        ld_profile=ld_profile, ld_uniform_basis=ld_uniform_basis,
        spectro_fixed_timescale_trends=spectro_fixed_timescale_trends,
        spectro_ld_parameterization=spectro_ld_parameterization,
        spectro_max_divergences=spectro_max_divergences,
        spectro_min_depth_ess=spectro_min_depth_ess, surface_config=surface_config,
        transit_engine=transit_engine, transit_window_optimization=transit_window_optimization,
        trend_inference=trend_inference, uses_surface_model=uses_surface_model,
    )
    spectro_data = SimpleNamespace(
        execution_context=execution_context,
        _align_trend_to_time=_align_trend_to_time, _explicit_ld_grid=_explicit_ld_grid,
        bestfit_params_wl_df=bestfit_params_wl_df,
        build_transit_window_indices=build_transit_window_indices, chunk_mode=chunk_mode,
        chunk_parallel_job_count=chunk_parallel_job_count,
        chunk_parallel_job_index=chunk_parallel_job_index, dataset=data, exp_trend=exp_trend,
        explicit_ld=explicit_ld, flags=flags, instrument=instrument,
        instrument_full_str=instrument_full_str, max_harmonic_order=max_harmonic_order,
        order=order, output_dir=output_dir, planet_str=planet_str, plots_mode=plots_mode,
        sld=sld, spec_good_mask=spec_good_mask, stellar_cfg=stellar_cfg,
        wl_mad_mask=wl_mad_mask, wl_time_good=wl_time_good,
    )

    _stage_result = run_low_resolution_stage(
        spectro_geometry, spectro_model, spectro_data, spectro_sampler=spectro_sampler,
        cfg=cfg, harmonica_lr_nuts_kwargs=harmonica_lr_nuts_kwargs,
        jaxoplanet_lr_nuts_kwargs=jaxoplanet_lr_nuts_kwargs, jump_trend=jump_trend,
        key_map_lr=key_map_lr, key_mcmc_lr=key_mcmc_lr, lowres_mcmc_kwargs=lowres_mcmc_kwargs,
        lr_artifact_stem=lr_artifact_stem, lr_bin_str=lr_bin_str,
        need_lowres_analysis=need_lowres_analysis, spectroscopic_sigma=spectroscopic_sigma,
        spot_trend=spot_trend, spot_trend2=spot_trend2, valid=valid,
        vmap_chunk_size_lr=vmap_chunk_size_lr,
    )
    if _stage_result is None:
        return
    valid, spot_trend, spot_trend2, jump_trend = _stage_result

    if analysis_stage == 'prep':
        info("Preparation complete.")
        finish_run(output_dir, started)
        return


    if run_high_resolution_stage(
        spectro_geometry, spectro_model, spectro_data, spectro_sampler=spectro_sampler,
        harmonica_hr_nuts_kwargs=harmonica_hr_nuts_kwargs,
        highres_mcmc_kwargs=highres_mcmc_kwargs, hr_artifact_stem=hr_artifact_stem,
        hr_bin_str=hr_bin_str, jaxoplanet_hr_nuts_kwargs=jaxoplanet_hr_nuts_kwargs,
        jump_trend=jump_trend, key_map_hr=key_map_hr, key_mcmc_hr=key_mcmc_hr,
        spot_trend=spot_trend, spot_trend2=spot_trend2, valid=valid,
        vmap_chunk_size_hr=vmap_chunk_size_hr,
    ) is None:
        return
    finish_run(output_dir, started)

if __name__ == "__main__":
    main()
