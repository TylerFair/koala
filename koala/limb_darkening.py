"""Limb Darkening helpers."""

from .paths import DEFAULT_LD_DATA_PATH

from koala.console import debug, info, progress, stellar_output, grid_description, exotic_output

import os
import sys
import json
import hashlib
import time
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from exotic_ld import StellarLimbDarkening
from .instruments import ld_mode_and_bounds
import arviz as az
from models.ld_parameterization import (Power2MaxtedTransform, Power2MaxtedUniform, Power2SmoothTransform,
                                       clip_into_maxted_uniform_support, power2_smooth_transform)
from models.sing_ld import SING_TABULATED_OFFSET, SING_TABULATED_SCATTER, quadratic_to_sing
from .constants import POWER2_LD_CACHE_TARGET_REVISION
from .config import _resolve_ld_prior_cache_options
from .artifacts import (
    _update_checkpoint_hash,
    _file_content_identity,
    _directory_metadata_identity,
    _atomic_dataframe_csv,
)


class CachedStellarLimbDarkening(StellarLimbDarkening):
    """Reuse bandpass setup while retaining ExoTiC-LD's fitting procedure.

    The integration uses the same 500-point rule, interpolation, strict bin
    masks and narrow-bin fallback. Throughputs are cached per stellar object,
    rather than globally, so independent runs still read their input files.
    """

    def __init__(self, *args, **kwargs):
        with exotic_output(), stellar_output(kwargs.get('ld_model', 'stellar'), kwargs.get('Teff', 0),
                            kwargs.get('logg', 0), kwargs.get('M_H', 0)):
            super().__init__(*args, **kwargs)

    def _integrate_I_mu(self, wavelength_range, mode, custom_wavelengths,
                        custom_throughput):
        from models.ld_prior_fast import integrate_passband

        if mode == 'custom':
            sensitivity = (custom_wavelengths, custom_throughput)
        else:
            if not hasattr(self, '_koala_sensitivity_cache'):
                self._koala_sensitivity_cache = {}
            if mode not in self._koala_sensitivity_cache:
                self._koala_sensitivity_cache[mode] = self._read_sensitivity_data(mode)
            sensitivity = self._koala_sensitivity_cache[mode]
        intensity = integrate_passband(
            self.stellar_wavelengths, self.stellar_intensities,
            np.asarray(sensitivity[0]), np.asarray(sensitivity[1]),
            [wavelength_range],
        )[0]
        if intensity[0] == 0.:
            raise ValueError("Zero intensity in this passband, check your "
                             "wavelength range is correct and in angstroms.")
        self.I_mu = intensity / intensity[0]


def _power2_ld_initial_sites(coefficients, parameterization, mode='gaussian'):
    """Return initial values in the actual latent coordinates of the LD model.

    The starting coefficients are first moved strictly inside the power-2
    prior support (c1 in (0, 1), c2 in (0.001, 1)); native-resolution prior
    means on strong stellar lines can sit outside it.
    """
    if mode == 'uniform':
        # The uniform prior's support is the Maxted region (alpha > 1 allowed); clip only into it.
        coefficients = clip_into_maxted_uniform_support(coefficients)
    else:
        coefficients = _clip_ld_initial_values(coefficients, 'power2', 'power2 sites')
    if parameterization == 'coefficients':
        return {'c1': coefficients[..., 0], 'c2': coefficients[..., 1]}
    if parameterization == 'maxted_smooth':
        return {'ld_smooth': power2_smooth_transform(mode).inv(coefficients)}
    if parameterization == 'decorrelated':
        return {'ld_decorrelated': Power2MaxtedTransform()(coefficients)}
    raise ValueError(f"Unknown power-2 LD parameterization: {parameterization}")


def _power2_ld_optimization_sites(parameterization):
    """Return only latent sample sites, never deterministic LD coefficients."""
    if parameterization == 'coefficients':
        return ['c1', 'c2']
    if parameterization == 'maxted_smooth':
        return ['ld_smooth']
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
        info('Some limb-darkening starting values were outside their prior support; moving them inside it.')
        debug(
            f"[LD init {stage_label}] {n_moved} channel(s) had prior means outside "
            "the physical support; their starting values were clipped inside it.",
            flush=True,
        )
    return clipped


def _repair_gaussian_ld_start(solution, *, ld_profile, ld_prior_mode,
                              ld_uniform_basis='uplus_uminus'):
    """Repair only coefficient starts at the Gaussian prior's support boundary.

    Keep the stellar prior centres and widths unchanged. This also covers a
    constrained optimizer returning an endpoint whose inverse transform is
    infinite at the Laplace/sampler handoff.
    """
    if ld_prior_mode != 'gaussian':
        return solution
    if ld_profile == 'quadratic' and ld_uniform_basis != 'kipping' and 'u' in solution:
        return {**solution, 'u': _clip_ld_initial_values(solution['u'], 'quadratic', 'gaussian start')}
    if ld_profile == 'power2' and 'c1' in solution and 'c2' in solution:
        values = jnp.stack((jnp.asarray(solution['c1']), jnp.asarray(solution['c2'])), axis=-1)
        values = _clip_ld_initial_values(values, 'power2', 'gaussian start')
        return {**solution, 'c1': values[..., 0], 'c2': values[..., 1]}
    return solution


def _quadratic_uniform_initial_sites(coefficients, basis):
    """Initialize the selected quadratic-uniform prior coordinates."""
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    if basis == 'coefficients':
        return {'u': coefficients}
    if basis == 'kipping':
        from models.model_building import kipping_q_from_u
        return {'ld_q': jnp.clip(kipping_q_from_u(coefficients), 1e-4, 1.0 - 1e-4)}
    if basis == 'uplus_uminus':
        return {'ld_uplus_uminus': jnp.stack(
            (coefficients[..., 0] + coefficients[..., 1],
             coefficients[..., 0] - coefficients[..., 1]), axis=-1)}
    raise ValueError(f"Unknown quadratic uniform LD basis: {basis}")


def _validate_whitelight_optimized_start(
    solution, *, ld_profile, ld_parameterization, ld_prior_mode,
    quadratic_uniform_bounds=(0.0, 1.0), n_planets=1,
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
        if 'ld_smooth' in solution:
            physical_ld = power2_smooth_transform(ld_prior_mode)(jnp.asarray(solution['ld_smooth']))
        elif 'ld_decorrelated' in solution:
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
            elif ld_prior_mode == 'uniform':
                if not np.all(np.isfinite(np.asarray(Power2MaxtedUniform().log_prob(
                        jnp.asarray(values).reshape(-1, 2))))):
                    reasons.append(f'power-2 coefficients {values.tolist()} are outside the '
                                   'Maxted-uniform support (0 < h1 < 1, 0 < h2 < min(1, h1 + 0.5))')
            elif np.any(values < low) or np.any(values > 1.0):
                reasons.append(
                    f'power-2 coefficients {values.tolist()} are outside '
                    f'[{low.tolist()}, [1.0, 1.0]]'
                )
    elif ld_profile == 'quadratic' and 'u' in solution:
        values = np.asarray(jax.device_get(solution['u']), dtype=float)
        quadratic_low, quadratic_high = (
            quadratic_uniform_bounds
            if ld_prior_mode in {'uniform', 'fixed_u1'}
            else (-1.0, 2.0) if 'ld_q' in solution  # Kipping: u1 in [0, 2], u2 in [-1, 1]
            else (0.0, 1.0)
        )
        if not np.all(np.isfinite(values)):
            reasons.append('quadratic coefficients are non-finite')
        elif np.any((values[..., 1:] if ld_prior_mode == 'fixed_u1' else values) < quadratic_low) or np.any(
                (values[..., 1:] if ld_prior_mode == 'fixed_u1' else values) > quadratic_high):
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
                f'planet {planet_index} rprs={rors_value} is outside ({rors_min}, {rors_max})',
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


# Stellar-parameter grid points whose Gaussian weight is below this fraction of the
# peak are skipped. On the default 9-node, +/-5 sigma grid this keeps 305 of 729
# points and changes the propagated mean and scatter by less than 1e-4 of the scatter.
LD_PRIOR_MIN_RELATIVE_WEIGHT = 1e-6


def _stellar_grid_combos(teff, teff_sigma, logg, logg_sigma, feh, feh_sigma, n_grid, nsigma):
    """Gaussian-weighted ([Fe/H], Teff, log g, weight) grid points for the stellar LD prior."""
    combos = []
    for mh_i in _build_axis(feh, feh_sigma, n_grid, nsigma):
        for teff_i in _build_axis(teff, teff_sigma, n_grid, nsigma):
            for logg_i in _build_axis(logg, logg_sigma, n_grid, nsigma):
                weight = (
                    _gaussian_pdf(mh_i, feh, feh_sigma)
                    * _gaussian_pdf(teff_i, teff, teff_sigma)
                    * _gaussian_pdf(logg_i, logg, logg_sigma)
                )
                if weight >= LD_PRIOR_MIN_RELATIVE_WEIGHT:
                    combos.append((mh_i, teff_i, logg_i, weight))
    return combos


def _build_axis(mu, sigma, n_grid, nsigma):
    if sigma <= 0 or n_grid == 1:
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
            info(f'Limb-darkening fit at {centre / 1e4:.3f} µm failed; retrying with a wider stellar passband.')
            last_err = err
            half *= 2.0
    raise RuntimeError(
        f"Limb-darkening fit failed for {range_min:.1f}-{range_max:.1f} A even after "
        f"{max_doublings} bin doublings: {last_err}"
    )


@exotic_output()
def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None, ld_profile='quadratic', return_sigmas=False, ld_mu_min=None):
    mode, wl_min, wl_max = _get_ld_mode_bounds(instrument, order)

    wavelengths = np.asarray(wavelengths, dtype=float)
    has_vector_err = hasattr(wavelength_err, '__len__') and np.asarray(wavelength_err).ndim > 0
    wavelength_err = np.asarray(wavelength_err, dtype=float) if has_vector_err else float(wavelength_err)

    if has_vector_err:
        results = []
        for index, wavelength in enumerate(progress(wavelengths, desc='Limb darkening from stellar models')):
            center = wavelength * 1e4
            half_width = wavelength_err[index] * 1e4
            low, high = _clip_ld_range(center - half_width, center + half_width, wl_min, wl_max)
            results.append(_compute_ld_coeffs_with_retry(
                sld, ld_profile, low, high, mode, ld_mu_min, return_sigmas, wl_min, wl_max,
            ))
        if return_sigmas:
            return (jnp.array([result[0] for result in results]),
                    jnp.array([result[1] for result in results]))
        return jnp.array(results)
    low, high = _clip_ld_range(
        np.min(wavelengths) * 1e4, np.max(wavelengths) * 1e4, wl_min, wl_max,
    )
    if ld_profile == 'quadratic':
        function = sld.compute_quadratic_ld_coeffs
        options = {'mu_min': float(ld_mu_min)} if ld_mu_min is not None else {}
    elif ld_profile == 'power2':
        function = sld.compute_power2_ld_coeffs
        options = {}
    else:
        raise ValueError(f"Unknown ld_profile: {ld_profile}")
    result = function(
        wavelength_range=[low, high], mode=mode, return_sigmas=return_sigmas, **options,
    )
    return tuple(jnp.array(value) for value in result) if return_sigmas else jnp.array(result)


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
        info('No fitted limb-darkening offsets were supplied; using the tabulated values.')
        debug("[Sing LD] no fitted gray-offset artifact supplied; using tabulated fallback for this first pass.", flush=True)
    centers = np.column_stack((model_l + offsets['l'], model_delta + offsets['delta']))
    scales = np.broadcast_to(np.array([SING_TABULATED_SCATTER['l'], SING_TABULATED_SCATTER['delta']]), centers.shape).copy()
    debug(
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


def get_or_build_power2_ld_prior(*args, **kwargs):
    """Existing power-2 stellar-grid prior (unchanged density/cache contract)."""
    return _get_or_build_stellar_ld_prior(*args, **kwargs, ld_profile='power2')


def get_or_build_quadratic_ld_prior(*args, **kwargs):
    """Propagate the stellar grid through actual quadratic-law fits.

    Like the established power-2 mode, the inference density is a product of
    coefficient-space truncated normals with marginal stellar-grid scatter,
    floored by stellar.ld_prior_min_sigma. It does not claim a correlated
    mixture density. Reference callers must use this same declared density.
    """
    return _get_or_build_stellar_ld_prior(*args, **kwargs, ld_profile='quadratic')


@exotic_output()
def _get_or_build_stellar_ld_prior(stellar_cfg, wavelengths, wavelength_err, instrument, order=None, output_dir='.', cache_label='ld', ld_profile='power2'):
    if ld_profile not in {'quadratic', 'power2'}:
        raise ValueError('Stellar LD grid requires quadratic or power2.')
    coefficient_names = ('u1', 'u2') if ld_profile == 'quadratic' else ('c1', 'c2')
    ld_mu_min = stellar_cfg.get('ld_mu_min') if ld_profile == 'quadratic' else None
    ld_prior_model = stellar_cfg.get('ld_prior_model', stellar_cfg.get('ld_model', 'stagger'))
    ld_data_path = stellar_cfg.get('ld_data_path', DEFAULT_LD_DATA_PATH)
    ld_interpolate_type = stellar_cfg.get('ld_interpolate_type', 'trilinear')
    teff = float(stellar_cfg['teff'])
    logg = float(stellar_cfg['logg'])
    feh = float(stellar_cfg['feh'])
    teff_sigma = float(stellar_cfg.get('teff_sigma', 0.0))
    logg_sigma = float(stellar_cfg.get('logg_sigma', 0.0))
    feh_sigma = float(stellar_cfg.get('feh_sigma', 0.0))
    # Nine Gaussian-weighted nodes over +/-5 sigma reproduce the continuous mean and
    # scatter; five nodes over that span would understate the scatter by ~30%.
    n_grid = int(stellar_cfg.get('ld_prior_n_grid', 9))
    nsigma = float(stellar_cfg.get('ld_prior_nsigma', 5.0))
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
            info('The limb-darkening cache is unavailable; computing the prior for this run.')
            debug(
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
            "grid": (n_grid, nsigma, LD_PRIOR_MIN_RELATIVE_WEIGHT),
            "min_sigma": min_sigma,
            "cache_label": cache_label,
        },
    )
    if ld_profile == 'quadratic':
        _update_checkpoint_hash(digest, {'law': 'quadratic', 'mu_min': ld_mu_min,
            'prior_convention': 'independent-coefficient-truncated-normals-v1'})
    cache_name = f"{instrument.replace('/', '_')}_{cache_label}_{digest.hexdigest()[:12]}.csv"
    cache_path = os.path.join(cache_dir, cache_name)

    if cache_enabled and os.path.exists(cache_path):
        expected_rows = len(wavelengths_np) if has_vector_err else 1
        required_columns = {
            *(name + suffix for name in coefficient_names for suffix in ('_mean', '_sigma_star'))
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
                [df[f'{coefficient_names[0]}_mean'].to_numpy(float), df[f'{coefficient_names[1]}_mean'].to_numpy(float)]
            )
            sigma = np.column_stack(
                [
                    df[f'{coefficient_names[0]}_sigma_star'].to_numpy(float),
                    df[f'{coefficient_names[1]}_sigma_star'].to_numpy(float),
                ]
            )
            if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
                raise ValueError("contains non-finite coefficients or uncertainties")
        except (OSError, ValueError, TypeError, KeyError) as error:
            info('The cached limb-darkening prior could not be read; rebuilding it.')
            debug(
                f"[LD prior] Ignoring invalid cache {cache_path}: {error}; "
                "rebuilding atomically.",
                flush=True,
            )
        else:
            info(f'Reusing the cached stellar limb-darkening prior {grid_description(cache_label)}.')
            debug(
                f"[LD prior] Loading cached {ld_profile} grid for {instrument} "
                f"({cache_label}) from {cache_path}",
                flush=True,
            )
            sigma = np.maximum(sigma, min_sigma)
            if not has_vector_err:
                return jnp.array(mu[0]), jnp.array(sigma[0])
            return jnp.array(mu), jnp.array(sigma)

    combos = _stellar_grid_combos(teff, teff_sigma, logg, logg_sigma, feh, feh_sigma, n_grid, nsigma)

    debug(
        f"[LD prior] Building {ld_profile} grid for {instrument} ({cache_label}) with {len(combos)} stellar combinations using model={ld_prior_model}",
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
    if ld_profile == 'power2':
        fast_started = time.perf_counter()
        mu_grid, grad_norm = build_power2_grid(
            [(mh_i, teff_i, logg_i) for (mh_i, teff_i, logg_i, _) in combos], ranges, mode,
            ld_prior_model, ld_data_path, interpolate_type=ld_interpolate_type,
        )
        bad = ~(np.all(np.isfinite(mu_grid), axis=-1) & (grad_norm < 1e-6))
        debug(
            f"[LD prior fast] {mu_grid.shape[0]} combinations x {mu_grid.shape[1]} bins "
            f"in {time.perf_counter() - fast_started:.1f} s; "
            f"{int(bad.sum())} fit(s) sent to the legacy path",
            flush=True,
        )
        for ci, bi in zip(*np.nonzero(bad)):
            mh_i, teff_i, logg_i, _ = combos[ci]
            sld_grid = StellarLimbDarkening(
                M_H=mh_i, Teff=teff_i, logg=logg_i, ld_model=ld_prior_model,
                ld_data_path=ld_data_path, interpolate_type=ld_interpolate_type, verbose=0,
            )
            lo, hi = ranges[bi]
            mu_grid[ci, bi] = np.asarray(_compute_ld_coeffs_with_retry(
                sld_grid, 'power2', lo, hi, mode, None, False, wl_min, wl_max,
            ), dtype=float)
    else:
        # Fit the actual quadratic law at every declared stellar-grid point.
        # Reuse passband setup per star; never substitute power-2 coefficients.
        mu_grid = np.empty((len(combos), len(ranges), 2))
        for ci, (mh_i, teff_i, logg_i, _) in enumerate(progress(combos, desc='Limb-darkening priors from stellar models')):
            sld_grid = CachedStellarLimbDarkening(
                M_H=mh_i, Teff=teff_i, logg=logg_i, ld_model=ld_prior_model,
                ld_data_path=ld_data_path, interpolate_type=ld_interpolate_type, verbose=0)
            for bi, (lo, hi) in enumerate(ranges):
                mu_grid[ci, bi] = np.asarray(_compute_ld_coeffs_with_retry(
                    sld_grid, 'quadratic', lo, hi, mode, ld_mu_min, False, wl_min, wl_max), dtype=float)
        if not np.all(np.isfinite(mu_grid)):
            raise ValueError('Non-finite quadratic stellar-grid coefficients.')
    for ci, (_, _, _, weight) in enumerate(progress(combos, desc='Limb-darkening priors from stellar models')):
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

    columns = {
        'wavelength': wavelengths_np if has_vector_err else [float(np.mean(wavelengths_np))],
        'wavelength_err': wavelength_err_np if has_vector_err else [float(wavelength_err_np[0])],
    }
    for name, values in (
        ('c1_mean', c1_mean), ('c1_sigma_total', c1_tot),
        ('c1_sigma_fit', c1_fit), ('c1_sigma_star', c1_star),
        ('c2_mean', c2_mean), ('c2_sigma_total', c2_tot),
        ('c2_sigma_fit', c2_fit), ('c2_sigma_star', c2_star),
    ):
        if ld_profile == 'quadratic':
            name = name.replace('c1_', 'u1_').replace('c2_', 'u2_')
        columns[name] = values if has_vector_err else [float(values[0])]
    if cache_enabled:
        try:
            _atomic_dataframe_csv(pd.DataFrame(columns), cache_path, index=False)
        except OSError as error:
            info('The limb-darkening prior could not be cached; continuing with the computed prior.')
            debug(
                f"[LD prior] Could not write cache {cache_path} ({error}); "
                "continuing with the newly computed prior.", flush=True,
            )
    if has_vector_err:
        return jnp.array(np.column_stack([c1_mean, c2_mean])), jnp.array(np.column_stack([c1_tot, c2_tot]))
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
