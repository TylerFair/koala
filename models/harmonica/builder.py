import os

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from models.priors import sample_parameter
import numpy as np

from .core import (
    compute_transit_model_harmonica,
    compute_transit_model_harmonica_batched,
    harmonica_cos_i_from_geometry,
    harmonica_impact_param_from_cos_i,
    harmonica_duration_from_geometry,
    harmonica_duration_from_cos_i,
    harmonica_a_rs_from_duration,
    harmonica_odd_coeff_specs,
    HARMONICA_HALF_AREA_CONTRAST_FACTOR,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT,
    harmonica_half_area_coefficients_from_area_radius,
)
from ..common import apply_systematics
from ..detrend import _split_components
from ..limb_darkening_config import GAUSSIAN_LD_WIDTH
from ..trends import (
    _soft_step,
    resolve_whitelight_trend_parameterization,
    sample_ordered_spot_centers,
    sample_step_width,
    spot_crossing,
)

NUTS_KWARGS = {
    "dense_mass": True,
    "regularize_mass_matrix": True,
    "max_tree_depth": 8,
    "target_accept_prob": 0.8,
}


def _sample_harmonica_odd_coeffs(base_radius, odd_specs, frac_sigma=0.1):
    """Sample odd harmonics as fractions of the base radius."""
    radius = jnp.maximum(jnp.asarray(base_radius, dtype=jnp.float64), 1e-6)
    frac_scale = jnp.ones_like(radius, dtype=jnp.float64) * frac_sigma
    coeffs = {}
    for _name, _order in odd_specs:
        frac = numpyro.sample(f"{_name}_frac", dist.Normal(jnp.zeros_like(radius), frac_scale))
        coeffs[_name] = numpyro.deterministic(_name, frac * radius)
    return coeffs


def _sample_harmonica_delta_r(mean_radius, frac_sigma=0.1):
    """Sample the first odd harmonic via the limb radius difference delta_r."""
    radius = jnp.maximum(jnp.asarray(mean_radius, dtype=jnp.float64), 1e-6)
    delta_r_scale = 2.0 * frac_sigma * radius
    delta_r = numpyro.sample(
        "delta_r",
        dist.Normal(jnp.zeros_like(radius), delta_r_scale),
    )
    return {
        "a1": numpyro.deterministic("a1", 0.5 * delta_r),
    }


def _sample_harmonica_half_area(mean_radius, frac_sigma=0.1):
    """Sample a bounded physical contrast between two indexed half-areas.

    Locally, ``q ~= (4/pi) * (a1/a0)``. Scaling the truncated-normal prior by
    the same factor therefore preserves the small-asymmetry prior implied by
    the legacy fractional parameterization, while the support enforces a
    globally convex ``N_c=1`` limacon. This local scale match is not global
    prior equivalence: the radius prior is on total-area radius rather than
    ``a0``, and the truncated ``q`` support removes non-convex legacy tails.
    """
    area_radius = jnp.maximum(
        jnp.asarray(mean_radius, dtype=jnp.float64), 1e-6
    )
    q_scale = jnp.ones_like(area_radius) * (
        HARMONICA_HALF_AREA_CONTRAST_FACTOR * frac_sigma
    )
    q_bound = jnp.asarray(
        HARMONICA_HALF_AREA_CONVEX_Q_LIMIT, dtype=jnp.float64
    )
    q = numpyro.sample(
        "q",
        dist.TruncatedNormal(
            jnp.zeros_like(area_radius),
            q_scale,
            low=-q_bound,
            high=q_bound,
        ),
    )
    a0, a1 = harmonica_half_area_coefficients_from_area_radius(
        area_radius, q
    )
    a0 = numpyro.deterministic("a0", a0)
    a1 = numpyro.deterministic("a1", a1)
    total = area_radius**2
    depth_one = total * (1.0 - q)
    depth_two = total * (1.0 + q)
    numpyro.deterministic("depth_total_area", total)
    numpyro.deterministic("depth_one", depth_one)
    numpyro.deterministic("depth_two", depth_two)
    return {"a0": a0, "a1": a1}


def derive_geometry(wl_samples, period, ecc=0.0, omega=0.0):
    """Derive a consistent harmonica geometry bundle from any supported WL parameterization."""
    period = jnp.atleast_1d(jnp.asarray(period, dtype=jnp.float64))
    ecc = jnp.atleast_1d(jnp.asarray(ecc, dtype=jnp.float64))
    omega = jnp.atleast_1d(jnp.asarray(omega, dtype=jnp.float64))
    n_planets = period.shape[0]

    result = {}
    for i in range(n_planets):
        if f"a_rs_{i}" in wl_samples:
            a_rs_samples = wl_samples[f'a_rs_{i}']
        elif f"log_a_rs_{i}" in wl_samples:
            a_rs_samples = jnp.exp(wl_samples[f'log_a_rs_{i}'])
        else:
            a_rs_samples = None

        if f"b_{i}" in wl_samples:
            b_samples = wl_samples[f"b_{i}"]
        elif f"_b_{i}" in wl_samples:
            b_samples = jnp.abs(wl_samples[f"_b_{i}"])
        else:
            b_samples = None

        if f"cos_i_{i}" in wl_samples:
            cos_i_samples = wl_samples[f"cos_i_{i}"]
            inc_samples = jnp.arccos(jnp.clip(cos_i_samples, 0.0, 1.0 - 1e-9))
        elif f'inc_{i}' in wl_samples:
            inc_samples = wl_samples[f'inc_{i}']
            cos_i_samples = jnp.cos(inc_samples)
        else:
            inc_samples = None
            cos_i_samples = None

        if f"duration_{i}" in wl_samples:
            duration_samples = wl_samples[f"duration_{i}"]
        elif f"logD_{i}" in wl_samples:
            duration_samples = jnp.exp(wl_samples[f"logD_{i}"])
        else:
            duration_samples = None

        rors_samples = wl_samples[f'rors_{i}']

        if a_rs_samples is None and duration_samples is not None and b_samples is not None:
            a_rs_samples = harmonica_a_rs_from_duration(
                period[i], duration_samples, b_samples, rors_samples,
                ecc=ecc[i], omega=omega[i],
            )
        if b_samples is None and cos_i_samples is not None and a_rs_samples is not None:
            b_samples = harmonica_impact_param_from_cos_i(
                cos_i_samples, a_rs_samples, ecc=ecc[i], omega=omega[i],
            )
        if cos_i_samples is None and b_samples is not None and a_rs_samples is not None:
            cos_i_samples = harmonica_cos_i_from_geometry(
                b_samples, a_rs_samples, ecc=ecc[i], omega=omega[i],
            )
            inc_samples = jnp.arccos(jnp.clip(cos_i_samples, 0.0, 1.0 - 1e-9))
        if duration_samples is None and cos_i_samples is not None and a_rs_samples is not None:
            duration_samples = harmonica_duration_from_cos_i(
                period[i], a_rs_samples, cos_i_samples, rors_samples,
                ecc=ecc[i], omega=omega[i],
            )
        elif duration_samples is None and b_samples is not None and a_rs_samples is not None:
            duration_samples = harmonica_duration_from_geometry(
                period[i], a_rs_samples, b_samples, rors_samples,
                ecc=ecc[i], omega=omega[i],
            )

        if a_rs_samples is None or b_samples is None or cos_i_samples is None or duration_samples is None:
            continue

        result[f'a_rs_{i}'] = a_rs_samples
        if inc_samples is not None:
            result[f'inc_{i}'] = inc_samples
        result[f'cos_i_{i}'] = cos_i_samples
        result[f'b_{i}'] = b_samples
        result[f'duration_{i}'] = duration_samples

    return result


def create_whitelight_model(detrend_type='linear', n_planets=1, ld_mode='gaussian',
                            max_harmonic_order=1, param_method='duration',
                            ld_profile='power2', step_width_mode='free',
                            trend_parameterization='physical',
                            two_spot_ordering='legacy', gp_solver=None,
                            gp_assume_sorted=False):
    """Harmonica white-light model with power-2 or fixed quadratic LD.

    Geometry sites come from prior_params['parameter_priors'] (one
    koala.config.ParameterSpec per planet and parameter); param_method
    picks whether duration or a_rs is the sampled quantity.
    """
    if param_method not in ('duration', 'a_rs'):
        raise ValueError(f"Unknown param_method: {param_method}")
    ld_profile = str(ld_profile).lower()
    if ld_profile not in {'power2', 'quadratic'}:
        raise ValueError(f"Unsupported Harmonica ld_profile: {ld_profile}")
    if ld_profile == 'quadratic' and ld_mode != 'fixed':
        raise ValueError(
            "Harmonica quadratic limb darkening uses fixed direct u1/u2 "
            "coefficients; set ld_prior='fixed'."
        )
    odd_specs = harmonica_odd_coeff_specs(max_harmonic_order)
    detrend_components = _split_components(detrend_type)
    from ..gp import GP_HYPERPARAMETER_BOUNDS, resolve_gp_builder, resolve_gp_solver
    # Resolve once at factory time, and only when a GP is actually built, so
    # non-GP trends never touch the solver policy (or its install check).
    resolved_gp_solver = resolve_gp_solver(gp_solver) if 'gp' in detrend_type else None
    configured_trend_parameterization = trend_parameterization
    two_spot_ordering = str(two_spot_ordering).strip().lower()
    if two_spot_ordering not in {'legacy', 'ordered'}:
        raise ValueError("two_spot_ordering must be 'legacy' or 'ordered'.")

    print(f"Building harmonica whitelight model: detrend='{detrend_type}', "
          f"ld='{ld_mode}' ({ld_profile}), max_order={max_harmonic_order}, "
          f"for {n_planets} planets")

    def _whitelight_model(t, yerr, y=None, prior_params=None):
        trend_parameterization = resolve_whitelight_trend_parameterization(
            configured_trend_parameterization
        )
        cadence = jnp.median(jnp.diff(jnp.sort(jnp.asarray(t))))

        parameter_priors = prior_params.get('parameter_priors')
        if not parameter_priors:
            raise ValueError(
                "prior_params['parameter_priors'] is required: every planet "
                "parameter must carry a {value, prior, ...} specification."
            )
        harmonica_ecc = jnp.asarray([s.value for s in parameter_priors['ecc']], dtype=jnp.float64)
        harmonica_omega = jnp.asarray([s.value for s in parameter_priors['omega']], dtype=jnp.float64)

        durations, t0s, bs, cos_is, rorss, a_rss, incs = [], [], [], [], [], [], []

        def _site(name, i):
            return sample_parameter(parameter_priors[name][i], f"{name}_{i}")

        for i in range(n_planets):
            t0_i = _site('t0', i)
            t0s.append(t0_i)
            if 'eclipse_time' in parameter_priors:
                numpyro.deterministic(
                    f"eclipse_time_{i}", t0_i + 0.5 * prior_params['period'][i]
                )
            rors_i = sample_parameter(parameter_priors['rprs'][i], f"rors_{i}")
            numpyro.deterministic(f"depths_{i}", rors_i ** 2)
            rorss.append(rors_i)
            b_i = _site('b', i)

            if param_method == 'duration':
                duration_i = _site('duration', i)
                a_rs_i = numpyro.deterministic(
                    f"a_rs_{i}",
                    harmonica_a_rs_from_duration(
                        prior_params['period'][i], duration_i, b_i, rors_i,
                        ecc=harmonica_ecc[i], omega=harmonica_omega[i],
                    ),
                )
            else:  # param_method == 'a_rs'
                a_rs_i = _site('a_rs', i)
                duration_i = numpyro.deterministic(
                    f"duration_{i}",
                    harmonica_duration_from_geometry(
                        prior_params['period'][i], a_rs_i, b_i, rors_i,
                        ecc=harmonica_ecc[i], omega=harmonica_omega[i],
                    ),
                )
            cos_i_i = numpyro.deterministic(
                f"cos_i_{i}",
                harmonica_cos_i_from_geometry(
                    b_i, a_rs_i, ecc=harmonica_ecc[i], omega=harmonica_omega[i],
                ),
            )
            inc_i = numpyro.deterministic(
                f"inc_{i}", jnp.arccos(jnp.clip(cos_i_i, 0.0, 1.0 - 1e-9))
            )

            durations.append(duration_i)
            bs.append(b_i)
            cos_is.append(cos_i_i)
            a_rss.append(a_rs_i)
            incs.append(inc_i)

        u_prior = jnp.asarray(prior_params['u'], dtype=jnp.float64)
        if ld_profile == 'quadratic':
            u1 = numpyro.deterministic('u1', u_prior[0])
            u2 = numpyro.deterministic('u2', u_prior[1])
        else:
            if ld_mode in {'gaussian', 'stellarprior'}:
                if ld_mode == 'stellarprior':
                    if 'u_sigma' not in prior_params:
                        raise ValueError("ld_mode='stellarprior' requires prior_params['u_sigma'].")
                    u_sigma = jnp.asarray(prior_params['u_sigma'], dtype=jnp.float64)
                else:
                    u_sigma = jnp.full_like(u_prior, GAUSSIAN_LD_WIDTH)
                u_sigma = jnp.broadcast_to(u_sigma, u_prior.shape)
                u_sigma = jnp.clip(u_sigma, 1e-6, None)
                c1 = numpyro.sample('c1', dist.TruncatedNormal(u_prior[0], u_sigma[0], low=0.0, high=1.0))
                c2 = numpyro.sample('c2', dist.TruncatedNormal(u_prior[1], u_sigma[1], low=0.001, high=1.0))
            elif ld_mode == 'uniform':
                c1 = numpyro.sample('c1', dist.Uniform(0.0, 1.0))
                c2 = numpyro.sample('c2', dist.Uniform(0.0, 1.0))
            elif ld_mode == 'fixed':
                c1 = numpyro.deterministic('c1', u_prior[0])
                c2 = numpyro.deterministic('c2', u_prior[1])
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")

        odd_sampled = _sample_harmonica_odd_coeffs(jnp.min(jnp.asarray(rorss)), odd_specs)

        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter) ** 2 + yerr ** 2))

        params = {
            "period": prior_params['period'],
            "t0": jnp.array(t0s),
            "b": jnp.array(bs),
            "rors": jnp.array(rorss),
            "a_rs": jnp.array(a_rss),
            "cos_i": jnp.array(cos_is),
            "inc": jnp.array(incs),
            "ecc": harmonica_ecc,
            "omega": harmonica_omega,
            **{_name: jnp.array([odd_sampled[_name]] * n_planets) for _name, _ in odd_specs},
        }
        if ld_profile == 'power2':
            params.update({"c_ld": c1, "alpha_ld": c2})
        else:
            params.update({"u1_ld": u1, "u2_ld": u2})

        has_offset_term = not detrend_components.isdisjoint(
            {'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', '2spot', 'gp'}
        )
        if has_offset_term:
            params['c'] = numpyro.sample('c', dist.Uniform(0.9, 1.1))

        has_linear = not detrend_components.isdisjoint(
            {'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', '2spot'}
        )
        if has_linear:
            params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1))

        if not detrend_components.isdisjoint({'quadratic', 'cubic', 'quartic'}):
            params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1))
        if not detrend_components.isdisjoint({'cubic', 'quartic'}):
            params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1))
        if 'quartic' in detrend_components:
            params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1))

        if 'linear_discontinuity' in detrend_components:
            t_jump_guess = prior_params.get('t_jump_guess', 0.5 * (jnp.min(t) + jnp.max(t)))
            jump_guess = prior_params.get('jump_guess', 0.0)
            if trend_parameterization == 'cadence':
                t_jump_offset_cadences = numpyro.sample(
                    't_jump_offset_cadences', dist.Normal(0.0, 1e-2 / cadence)
                )
                params['t_jump'] = numpyro.deterministic(
                    't_jump', t_jump_guess + cadence * t_jump_offset_cadences
                )
            else:
                params['t_jump'] = numpyro.sample(
                    't_jump', dist.Normal(t_jump_guess, 1e-2)
                )
            params['jump'] = numpyro.sample('jump', dist.Normal(jump_guess, 0.01))
            params['width'] = sample_step_width(
                t, prior_params, step_width_mode, trend_parameterization
            )

        if 'explinear' in detrend_components:
            params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1))
            log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)))
            params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))

        if not detrend_components.isdisjoint({'spot', '2spot'}):
            params['spot_amp'] = numpyro.sample('spot_amp', dist.Uniform(0.0, 0.1))
            ordered_pair = (
                '2spot' in detrend_components and two_spot_ordering == 'ordered'
            )
            if trend_parameterization == 'cadence':
                if not ordered_pair:
                    spot_mu_offset_cadences = numpyro.sample(
                        'spot_mu_offset_cadences', dist.Normal(0.0, 0.01 / cadence)
                    )
                    params['spot_mu'] = numpyro.deterministic(
                        'spot_mu',
                        prior_params['spot_guess'] + cadence * spot_mu_offset_cadences,
                    )
                spot_sigma_cadences = numpyro.sample(
                    'spot_sigma_cadences',
                    dist.Uniform(1e-4 / cadence, 0.1 / cadence),
                )
                params['spot_sigma'] = numpyro.deterministic(
                    'spot_sigma', cadence * spot_sigma_cadences
                )
            else:
                if not ordered_pair:
                    params['spot_mu'] = numpyro.sample(
                        'spot_mu', dist.Normal(prior_params['spot_guess'], 0.01)
                    )
                params['spot_sigma'] = numpyro.sample(
                    'spot_sigma', dist.Uniform(1e-4, 0.1)
                )
        if '2spot' in detrend_components:
            spot_guess2 = prior_params.get('spot_guess2', prior_params['spot_guess'])
            params['spot_amp2'] = numpyro.sample('spot_amp2', dist.Uniform(0.0, 0.1))
            if trend_parameterization == 'cadence':
                spot_sigma2_cadences = numpyro.sample(
                    'spot_sigma2_cadences',
                    dist.Uniform(1e-4 / cadence, 0.1 / cadence),
                )
                if two_spot_ordering == 'legacy':
                    spot_mu2_offset_cadences = numpyro.sample(
                        'spot_mu2_offset_cadences', dist.Normal(0.0, 0.01 / cadence)
                    )
                    params['spot_mu2'] = numpyro.deterministic(
                        'spot_mu2', spot_guess2 + cadence * spot_mu2_offset_cadences
                    )
                params['spot_sigma2'] = numpyro.deterministic(
                    'spot_sigma2', cadence * spot_sigma2_cadences
                )
            else:
                if two_spot_ordering == 'legacy':
                    params['spot_mu2'] = numpyro.sample(
                        'spot_mu2', dist.Normal(spot_guess2, 0.01)
                    )
                params['spot_sigma2'] = numpyro.sample(
                    'spot_sigma2', dist.Uniform(1e-4, 0.1)
                )
            if two_spot_ordering == 'ordered':
                params['spot_mu'], params['spot_mu2'] = (
                    sample_ordered_spot_centers(
                        prior_params['spot_guess'],
                        spot_guess2,
                        cadence,
                        parameterization=trend_parameterization,
                    )
                )

        if 'gp' in detrend_components:
            params['GP_log_sigma'] = numpyro.sample('GP_log_sigma', dist.Uniform(*GP_HYPERPARAMETER_BOUNDS['GP_log_sigma']))
            params['GP_log_rho'] = numpyro.sample('GP_log_rho', dist.Uniform(*GP_HYPERPARAMETER_BOUNDS['GP_log_rho']))

        transit_signal = compute_transit_model_harmonica(params, t)
        t_norm = t - jnp.min(t)
        if detrend_type == 'none':
            lc_model = apply_systematics(transit_signal, 1.0)
        else:
            trend = params.get('c', 1.0) + params.get('v', 0.0) * t_norm
            if 'v2' in params:
                trend = trend + params['v2'] * t_norm ** 2
            if 'v3' in params:
                trend = trend + params['v3'] * t_norm ** 3
            if 'v4' in params:
                trend = trend + params['v4'] * t_norm ** 4
            if 'A' in params:
                trend = trend + params['A'] * jnp.exp(-t_norm / params['tau'])
            if 't_jump' in params:
                trend = trend + params['jump'] * _soft_step(
                    t, params['t_jump'], params['width']
                )
            if 'spot_amp' in params:
                trend = trend + spot_crossing(
                    t, params['spot_amp'], params['spot_mu'], params['spot_sigma']
                )
            if 'spot_amp2' in params:
                trend = trend + spot_crossing(
                    t, params['spot_amp2'], params['spot_mu2'], params['spot_sigma2']
                )
            lc_model = apply_systematics(transit_signal, trend)

        if 'gp' not in detrend_type:
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)

        if 'gp' in detrend_type:
            gp_builder = resolve_gp_builder(detrend_type)
            gp = gp_builder(
                params,
                t,
                error,
                gp_solver=resolved_gp_solver,
                assume_sorted=gp_assume_sorted,
            )
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)

    return _whitelight_model


def create_vectorized_model(detrend_type='linear', ld_mode='gaussian', trend_mode='free',
                            n_planets=1, max_harmonic_order=1,
                            odd_parameterization='fractional', fit_jitter=True,
                            odd_frac_sigma=0.1, ld_profile='power2'):
    """Harmonica spectroscopic model with WL-fixed geometry and per-channel asymmetry."""
    odd_specs = harmonica_odd_coeff_specs(max_harmonic_order)
    detrend_components = _split_components(detrend_type)
    ld_profile = str(ld_profile).lower()
    if ld_profile not in {'power2', 'quadratic'}:
        raise ValueError(f"Unsupported Harmonica ld_profile: {ld_profile}")
    if ld_profile == 'quadratic' and ld_mode not in {
        'fixed', 'interpolated', 'sing'
    }:
        raise ValueError(
            "Harmonica quadratic limb darkening uses fixed direct u1/u2 "
            "coefficients or the Sing (l, delta) parameterization."
        )

    unsupported_components = {'gp', 'spot', '2spot', 'linear_discontinuity'}
    active_unsupported = detrend_components & unsupported_components
    if active_unsupported:
        raise ValueError(
            "Harmonica vectorized detrending supports only "
            "spectroscopic trend families and polynomial/explinear components. "
            f"Unsupported components: {sorted(active_unsupported)}."
        )
    if trend_mode != 'free':
        raise ValueError("Harmonica vectorized detrending only supports trend_mode='free'.")
    if odd_parameterization not in {'fractional', 'delta_r', 'half_area'}:
        raise ValueError(
            "odd_parameterization must be one of "
            "{'fractional', 'delta_r', 'half_area'}."
        )
    if odd_parameterization in {'delta_r', 'half_area'} and max_harmonic_order != 1:
        raise ValueError(
            f"odd_parameterization='{odd_parameterization}' only supports "
            "max_harmonic_order=1."
        )

    print(f"Building harmonica vectorized model: detrend='{detrend_type}', "
          f"ld='{ld_mode}' ({ld_profile}), max_order={max_harmonic_order}, "
          f"odd_parameterization='{odd_parameterization}', fit_jitter={fit_jitter} "
          f"for {n_planets} planets")

    def _vectorized_model(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                          mu_cos_i=None, mu_depths=None, PERIOD=None,
                          harmonica_a_rs=None, harmonica_ecc=0.0, harmonica_omega=0.0,
                          mu_u_ld=None, sigma_u_ld=None, ld_fixed=None, gp_trend=None,
                          spot_trend=None, spot_trend2=None, jump_trend=None, **kwargs):

        num_lcs = jnp.atleast_2d(yerr).shape[0]
        t0s = mu_t0
        bs = mu_b
        cos_is = mu_cos_i

        if harmonica_a_rs is None:
            raise ValueError("harmonica_a_rs must be provided, got None")
        _harmonica_a_rs = harmonica_a_rs
        _harmonica_ecc = harmonica_ecc
        _harmonica_omega = harmonica_omega

        rors = numpyro.sample(
            "rors",
            dist.Uniform(jnp.sqrt(1e-5), jnp.sqrt(0.5)).expand([num_lcs, n_planets]),
        )
        depths = numpyro.deterministic("depths", rors ** 2)

        yerr_matrix = jnp.atleast_2d(jnp.asarray(yerr, dtype=jnp.float64))
        if yerr_matrix.shape[0] == 1 and num_lcs > 1:
            yerr_matrix = jnp.broadcast_to(
                yerr_matrix, (num_lcs, yerr_matrix.shape[1])
            )
        yerr_per_lc = jnp.nanmedian(yerr_matrix, axis=1)
        if fit_jitter:
            log_jitter = numpyro.sample(
                'log_jitter',
                dist.Uniform(jnp.log(1e-6), jnp.log(1)).expand([num_lcs]),
            )
            jitter = jnp.exp(log_jitter)
            total_error = numpyro.deterministic(
                'total_error',
                jnp.sqrt(jitter ** 2 + yerr_per_lc ** 2),
            )
            error_broadcast = jnp.sqrt(
                jitter[:, None] ** 2 + yerr_matrix ** 2
            )
        else:
            total_error = numpyro.deterministic('total_error', yerr_per_lc)
            error_broadcast = yerr_matrix

        if ld_profile == 'quadratic' and ld_mode == 'sing':
            if sigma_u_ld is None:
                raise ValueError("ld_mode='sing' requires (l, delta) sigma_u_ld.")
            sing_mu = jnp.asarray(mu_u_ld, dtype=jnp.float64)
            sing_sigma = jnp.asarray(sigma_u_ld, dtype=jnp.float64)
            limb_l = numpyro.sample(
                'limb_l',
                dist.TruncatedNormal(
                    sing_mu[:, 0], sing_sigma[:, 0], low=0.0, high=1.0
                ),
            )
            u_plus = 1.0 - limb_l
            limb_delta = numpyro.sample(
                'limb_delta',
                dist.TruncatedNormal(
                    sing_mu[:, 1], sing_sigma[:, 1],
                    low=-u_plus / 4.0, high=u_plus / 4.0,
                ),
            )
            u1 = numpyro.deterministic('c1', u_plus - 4.0 * limb_delta)
            u2 = numpyro.deterministic('c2', 4.0 * limb_delta)
            numpyro.deterministic('u', jnp.stack((u1, u2), axis=1))
        elif ld_profile == 'quadratic':
            fixed_values = ld_fixed if ld_mode == 'fixed' else kwargs.get('ld_interpolated')
            if fixed_values is None:
                raise ValueError(
                    "Fixed quadratic Harmonica coefficients were not provided."
                )
            fixed_values = jnp.asarray(fixed_values, dtype=jnp.float64)
            u1 = numpyro.deterministic('u1', fixed_values[:, 0])
            u2 = numpyro.deterministic('u2', fixed_values[:, 1])
        elif ld_mode in {'gaussian', 'stellarprior'}:
            if ld_mode == 'stellarprior' and sigma_u_ld is None:
                raise ValueError("ld_mode='stellarprior' requires sigma_u_ld.")
            if ld_mode == 'gaussian':
                sigma_u_ld = jnp.full_like(mu_u_ld, GAUSSIAN_LD_WIDTH)
            sigma_u_ld = jnp.asarray(sigma_u_ld, dtype=jnp.float64)
            sigma_u_ld = jnp.broadcast_to(sigma_u_ld, mu_u_ld.shape)
            sigma_u_ld = jnp.clip(sigma_u_ld, 1e-6, None)
            c1 = numpyro.sample('c1', dist.TruncatedNormal(mu_u_ld[:, 0], sigma_u_ld[:, 0], low=0.0, high=1.0))
            c2 = numpyro.sample('c2', dist.TruncatedNormal(mu_u_ld[:, 1], sigma_u_ld[:, 1], low=0.001, high=1.0))
        elif ld_mode == 'uniform':
            c1 = numpyro.sample('c1', dist.Uniform(0.0, 1.0).expand([num_lcs]))
            c2 = numpyro.sample('c2', dist.Uniform(0.0, 1.0).expand([num_lcs]))
        elif ld_mode == 'fixed':
            c1 = numpyro.deterministic('c1', ld_fixed[:, 0])
            c2 = numpyro.deterministic('c2', ld_fixed[:, 1])
        elif ld_mode == 'interpolated':
            interpolated = kwargs.get('ld_interpolated')
            if interpolated is None:
                raise ValueError(
                    "Interpolated power-2 Harmonica coefficients were not provided."
                )
            interpolated = jnp.asarray(interpolated, dtype=jnp.float64)
            c1 = numpyro.deterministic('c1', interpolated[:, 0])
            c2 = numpyro.deterministic('c2', interpolated[:, 1])
        else:
            raise ValueError(f"Unknown ld_mode: {ld_mode}")

        if odd_parameterization == 'delta_r':
            odd_sampled = _sample_harmonica_delta_r(rors, frac_sigma=odd_frac_sigma)
        elif odd_parameterization == 'half_area':
            odd_sampled = _sample_harmonica_half_area(
                rors, frac_sigma=odd_frac_sigma
            )
        else:
            odd_sampled = _sample_harmonica_odd_coeffs(
                rors,
                odd_specs,
                frac_sigma=odd_frac_sigma,
            )

        if cos_is is None and bs is not None:
            cos_is = harmonica_cos_i_from_geometry(
                bs, _harmonica_a_rs, ecc=_harmonica_ecc, omega=_harmonica_omega
            )
        elif cos_is is not None and bs is None:
            bs = harmonica_impact_param_from_cos_i(
                cos_is, _harmonica_a_rs, ecc=_harmonica_ecc, omega=_harmonica_omega
            )

        _harm_orbital = {
            "period": PERIOD, "t0": t0s, "b": bs, "cos_i": cos_is,
            "a_rs": _harmonica_a_rs, "ecc": _harmonica_ecc,
            "omega": _harmonica_omega,
        }

        has_offset_term = detrend_type != 'none'
        c_trend = (
            numpyro.sample('c', dist.Uniform(0.9, 1.1).expand([num_lcs]))
            if has_offset_term else jnp.ones(num_lcs)
        )

        poly_order = 0
        if 'quartic' in detrend_components:
            poly_order = 4
        elif 'cubic' in detrend_components:
            poly_order = 3
        elif 'quadratic' in detrend_components:
            poly_order = 2
        elif 'linear' in detrend_components or 'explinear' in detrend_components:
            poly_order = 1

        v_trend = (
            numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
            if poly_order >= 1 else jnp.zeros(num_lcs)
        )
        v2_trend = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs])) if poly_order >= 2 else None
        v3_trend = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs])) if poly_order >= 3 else None
        v4_trend = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs])) if poly_order >= 4 else None

        has_explinear = 'explinear' in detrend_components
        if has_explinear:
            A_trend = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
            log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
            tau_trend = numpyro.deterministic('tau', jnp.exp(log_tau))

        harmonica_params = {
            **_harm_orbital,
            "rors": odd_sampled.get("a0", rors),
        }
        if ld_profile == 'power2':
            harmonica_params.update({"c_ld": c1, "alpha_ld": c2})
        else:
            harmonica_params.update({"u1_ld": u1, "u2_ld": u2})
        harmonica_params.update({_name: odd_sampled[_name] for _name, _ in odd_specs})
        transit_sig = compute_transit_model_harmonica_batched(harmonica_params, t)

        t_norm = t - jnp.min(t)
        trend = c_trend[:, None] + v_trend[:, None] * t_norm[None, :]
        if poly_order >= 2:
            trend = trend + v2_trend[:, None] * (t_norm ** 2)[None, :]
        if poly_order >= 3:
            trend = trend + v3_trend[:, None] * (t_norm ** 3)[None, :]
        if poly_order >= 4:
            trend = trend + v4_trend[:, None] * (t_norm ** 4)[None, :]
        if has_explinear:
            trend = trend + A_trend[:, None] * jnp.exp(-t_norm[None, :] / tau_trend[:, None])
        if 'gp_spectroscopic' in detrend_components:
            if gp_trend is None:
                raise ValueError("gp_spectroscopic requires gp_trend.")
            A_gp = numpyro.sample('A_gp', dist.Uniform(0.5, 2).expand([num_lcs]))
            trend = trend + A_gp[:, None] * gp_trend
        if 'spot_spectroscopic' in detrend_components:
            if spot_trend is None:
                raise ValueError("spot_spectroscopic requires spot_trend.")
            A_spot = numpyro.sample('A_spot', dist.Uniform(0.5, 2).expand([num_lcs]))
            trend = trend + A_spot[:, None] * spot_trend
        if '2spot_spectroscopic' in detrend_components:
            if spot_trend is None or spot_trend2 is None:
                raise ValueError("2spot_spectroscopic requires spot_trend and spot_trend2.")
            A_spot = numpyro.sample('A_spot', dist.Uniform(0.5, 2).expand([num_lcs]))
            A_spot2 = numpyro.sample('A_spot2', dist.Uniform(0.5, 2).expand([num_lcs]))
            trend = trend + A_spot[:, None] * spot_trend + A_spot2[:, None] * spot_trend2
        if 'linear_discontinuity_spectroscopic' in detrend_components:
            if jump_trend is None:
                raise ValueError("linear_discontinuity_spectroscopic requires jump_trend.")
            A_jump = numpyro.sample('A_jump', dist.Uniform(0.5, 2).expand([num_lcs]))
            trend = trend + A_jump[:, None] * jump_trend

        y_model = apply_systematics(transit_sig, trend)
        numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model
