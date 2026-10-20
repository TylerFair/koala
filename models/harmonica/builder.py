
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .core import (
    compute_transit_model_harmonica, compute_transit_model_harmonica_batched,
    harmonica_cos_i_from_geometry, harmonica_impact_param_from_cos_i,
    harmonica_odd_coeff_specs, HARMONICA_HALF_AREA_CONTRAST_FACTOR,
    HARMONICA_HALF_AREA_CONVEX_Q_LIMIT, harmonica_half_area_coefficients_from_area_radius,
)
from ..model_building import (
    polynomial_order, sample_whitelight_trend, sample_sing_coefficients,
    sample_power2_coefficients, sample_quadratic_uniform, sample_quadratic_fixed_u1,
)
from ..model_building import posterior_geometry, sample_orbital_geometry
from ..limb_darkening_config import GAUSSIAN_LD_WIDTH, validate_gaussian_ld_width, validate_uniform_ld_bounds
from ..trend_marginal import normalized_polynomial_transform
from ..common import apply_systematics, safe_tau
from ..detrend import _split_components
from ..trends import _soft_step, resolve_whitelight_trend_parameterization, spot_crossing

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
        'delta_r', dist.Normal(jnp.zeros_like(radius), delta_r_scale),
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
        'q',
        dist.TruncatedNormal(jnp.zeros_like(area_radius), q_scale, low=-q_bound, high=q_bound),
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
    """Derive orbital geometry from the supported white-light parameterizations."""
    return posterior_geometry(wl_samples, period, ecc, omega, harmonica=True)


def create_whitelight_model(detrend_type='linear', n_planets=1, ld_mode='gaussian',
                            max_harmonic_order=1, param_method='duration',
                            ld_profile='power2', step_width_mode='free',
                            trend_parameterization='physical',
                            two_spot_ordering='legacy', gp_solver=None,
                            gp_assume_sorted=False, parameter_priors=None,
                            ld_parameterization='coefficients', trend_priors=None,
                            gaussian_ld_width=GAUSSIAN_LD_WIDTH,
                            ld_uniform_basis='coefficients', ld_uniform_coefficient_bounds=(0., 1.)):
    """Harmonica white-light model with power-2 or fixed quadratic LD.

    Geometry sites come from ``parameter_priors`` (one
    koala.config.ParameterSpec per planet and parameter); param_method
    picks whether duration or a_rs is the sampled quantity.
    """
    if param_method not in ('duration', 'a_rs'):
        raise ValueError(f"Unknown param_method: {param_method}")
    ld_profile = str(ld_profile).lower()
    if ld_profile not in {'power2', 'quadratic'}:
        raise ValueError(f"Unsupported Harmonica ld_profile: {ld_profile}")
    if ld_profile == 'quadratic' and ld_mode not in {'fixed', 'fixed_u1', 'uniform'}:
        raise ValueError(
            "Harmonica quadratic limb darkening uses fixed direct u1/u2 "
            "coefficients; set ld_prior='fixed'."
        )
    if ld_mode == 'fixed_u1' and ld_profile != 'quadratic':
        raise ValueError('fixed_u1 requires quadratic limb darkening.')
    uniform_low, uniform_high = validate_uniform_ld_bounds(ld_uniform_coefficient_bounds)
    if ld_uniform_basis not in {'coefficients', 'uplus_uminus', 'kipping'}:
        raise ValueError('Invalid quadratic uniform basis.')
    gaussian_ld_width = validate_gaussian_ld_width(gaussian_ld_width)
    if ld_parameterization not in {'coefficients', 'maxted_smooth'}:
        raise ValueError("Harmonica ld_parameterization must be coefficients or maxted_smooth.")
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


    def _whitelight_model(t, yerr, y=None, prior_params=None):
        trend_parameterization = resolve_whitelight_trend_parameterization(
            configured_trend_parameterization
        )

        geometry = sample_orbital_geometry(
            parameter_priors, n_planets, param_method, fixed_period=prior_params['period'],
        )

        u_prior = jnp.asarray(prior_params['u'], dtype=jnp.float64)
        if ld_profile == 'quadratic' and ld_mode in {'uniform', 'fixed_u1'}:
            u = (sample_quadratic_fixed_u1(u_prior, (), uniform_low, uniform_high)
                 if ld_mode == 'fixed_u1' else sample_quadratic_uniform((), ld_uniform_basis, uniform_low, uniform_high))
            u1, u2 = u[..., 0], u[..., 1]
            if ld_mode == 'uniform' and ld_uniform_basis == 'coefficients':
                numpyro.deterministic('u1', u1); numpyro.deterministic('u2', u2)
        elif ld_profile == 'quadratic':
            u1 = numpyro.deterministic('u1', u_prior[0])
            u2 = numpyro.deterministic('u2', u_prior[1])
        else:
            c1, c2 = sample_power2_coefficients(
                u_prior, prior_params.get('u_sigma'), ld_mode,
                sigma_name="prior_params['u_sigma']", gaussian_width=gaussian_ld_width,
                maxted_smooth=ld_parameterization == 'maxted_smooth',
            )

        odd_sampled = _sample_harmonica_odd_coeffs(jnp.min(jnp.asarray(geometry['rors'])), odd_specs)

        jitter_spec = (trend_priors or {}).get('jitter')
        if jitter_spec is None:
            log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
            jitter = jnp.exp(log_jitter)
        else:
            from models.priors import sample_parameter
            jitter = sample_parameter(jitter_spec, 'jitter')
        error = numpyro.deterministic('error', jnp.sqrt(jitter ** 2 + yerr ** 2))

        params = {name: geometry[name] for name in
                  ('t0', 'b', 'rors', 'a_rs', 'cos_i', 'inc', 'ecc', 'omega')}
        params['period'] = prior_params['period']
        params.update({name: jnp.array([odd_sampled[name]] * n_planets)
                       for name, _ in odd_specs})
        if ld_profile == 'power2':
            params.update({"c_ld": c1, "alpha_ld": c2})
        else:
            params.update({"u1_ld": u1, "u2_ld": u2})

        params.update(sample_whitelight_trend(
            t, prior_params, detrend_components, trend_parameterization,
            step_width_mode, two_spot_ordering, trend_priors=trend_priors,
        ))

        transit_signal = compute_transit_model_harmonica(params, t)
        t_norm = t - jnp.min(t)
        if detrend_type == 'none':
            lc_model = apply_systematics(transit_signal, 1.0)
        else:
            trend = params.get('c', 1.0) + params.get('v', 0.0) * t_norm
            for degree in range(2, 5):
                if f'v{degree}' in params:
                    trend = trend + params[f'v{degree}'] * t_norm ** degree
            if 'A' in params:
                trend = trend + params['A'] * jnp.exp(-t_norm / safe_tau(params['tau']))
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
                params, t, error, gp_solver=resolved_gp_solver, assume_sorted=gp_assume_sorted,
            )
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)

    return _whitelight_model


def create_vectorized_model(detrend_type='linear', ld_mode='gaussian', trend_mode='free',
                            n_planets=1, max_harmonic_order=1,
                            odd_parameterization='fractional', fit_jitter=True,
                            odd_frac_sigma=0.1, ld_profile='power2',
                            ld_parameterization='coefficients', trend_basis='elapsed_days',
                            trend_bounds=None, gaussian_ld_width=GAUSSIAN_LD_WIDTH,
                            ld_uniform_basis='coefficients', ld_uniform_coefficient_bounds=(0., 1.)):
    """Harmonica spectroscopic model with WL-fixed geometry and per-channel asymmetry."""
    if trend_mode!='free':
        raise ValueError('Harmonica spectral trends require trend_mode=free; use jaxoplanet for fixed or Gaussian-integrated coefficients.')
    if ld_mode == 'fixed_u1' and ld_profile != 'quadratic':
        raise ValueError('fixed_u1 requires quadratic limb darkening.')
    uniform_low, uniform_high = validate_uniform_ld_bounds(ld_uniform_coefficient_bounds)
    if ld_uniform_basis not in {'coefficients', 'uplus_uminus', 'kipping'}:
        raise ValueError('Invalid quadratic uniform basis.')
    gaussian_ld_width = validate_gaussian_ld_width(gaussian_ld_width)
    if ld_parameterization not in {'coefficients', 'maxted_smooth'}:
        raise ValueError("Harmonica ld_parameterization must be coefficients or maxted_smooth.")
    if trend_basis not in {'elapsed_days', 'batman_normalized'}:
        raise ValueError("trend_basis must be 'elapsed_days' or 'batman_normalized'.")
    if trend_basis == 'batman_normalized' and detrend_type=='none':
        raise ValueError('batman_normalized requires sampled polynomial coefficients.')
    from ..spectro_trends import validate_spectral_trend_bounds
    trend_bounds=validate_spectral_trend_bounds(trend_bounds)
    if 'log_tau' in (trend_bounds or {}) and 'explinear' not in _split_components(detrend_type):
        raise ValueError('log_tau bounds require a sampled-timescale explinear trend.')
    odd_specs = harmonica_odd_coeff_specs(max_harmonic_order)
    detrend_components = _split_components(detrend_type)
    if 'explinear_spectroscopic' in detrend_components:
        raise ValueError('Harmonica fixed-template exponential ramps are unsupported; use explinear with sampled tau or the jaxoplanet engine.')
    ld_profile = str(ld_profile).lower()
    if ld_profile not in {'power2', 'quadratic'}:
        raise ValueError(f"Unsupported Harmonica ld_profile: {ld_profile}")
    if ld_profile == 'quadratic' and ld_mode not in {
        'fixed', 'interpolated', 'sing', 'fixed_u1', 'uniform'
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
            "odd_parameterization must be one of {'fractional', 'delta_r', 'half_area'}.",
        )
    if odd_parameterization in {'delta_r', 'half_area'} and max_harmonic_order != 1:
        raise ValueError(
            f"odd_parameterization='{odd_parameterization}' only supports "
            "max_harmonic_order=1."
        )


    def _vectorized_model(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                          mu_cos_i=None, mu_depths=None, PERIOD=None,
                          harmonica_a_rs=None, harmonica_ecc=0.0, harmonica_omega=0.0,
                          mu_u_ld=None, sigma_u_ld=None, ld_fixed=None, gp_trend=None,
                          spot_trend=None, spot_trend2=None, jump_trend=None, likelihood_mask=None, **kwargs):

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
            'rors', dist.Uniform(jnp.sqrt(1e-05), jnp.sqrt(0.5)).expand([num_lcs, n_planets]),
        )
        depths = numpyro.deterministic("depths", rors ** 2)

        yerr_matrix = jnp.atleast_2d(jnp.asarray(yerr, dtype=jnp.float64))
        if yerr_matrix.shape[0] == 1 and num_lcs > 1:
            yerr_matrix = jnp.broadcast_to(
                yerr_matrix, (num_lcs, yerr_matrix.shape[1])
            )
        valid_errors = (yerr_matrix if likelihood_mask is None else
                        jnp.where(jnp.asarray(likelihood_mask, dtype=bool), yerr_matrix, jnp.nan))
        yerr_per_lc = jnp.nanmedian(valid_errors, axis=1)
        if fit_jitter:
            log_jitter = numpyro.sample(
                'log_jitter', dist.Uniform(jnp.log(1e-06), jnp.log(1)).expand([num_lcs]),
            )
            jitter = jnp.exp(log_jitter)
            total_error = numpyro.deterministic(
                'total_error', jnp.sqrt(jitter ** 2 + yerr_per_lc ** 2),
            )
            error_broadcast = jnp.sqrt(
                jitter[:, None] ** 2 + yerr_matrix ** 2
            )
        else:
            total_error = numpyro.deterministic('total_error', yerr_per_lc)
            error_broadcast = yerr_matrix

        if ld_profile == 'quadratic' and ld_mode == 'sing':
            u1, u2, _ = sample_sing_coefficients(mu_u_ld, sigma_u_ld)
        elif ld_profile == 'quadratic' and ld_mode in {'uniform', 'fixed_u1'}:
            u = (sample_quadratic_fixed_u1(ld_fixed, (num_lcs,), uniform_low, uniform_high)
                 if ld_mode == 'fixed_u1' else sample_quadratic_uniform((num_lcs,), ld_uniform_basis, uniform_low, uniform_high))
            u1, u2 = u[..., 0], u[..., 1]
            if ld_mode == 'uniform' and ld_uniform_basis == 'coefficients':
                numpyro.deterministic('u1', u1); numpyro.deterministic('u2', u2)
        elif ld_profile == 'quadratic':
            fixed_values = ld_fixed if ld_mode == 'fixed' else kwargs.get('ld_interpolated')
            if fixed_values is None:
                raise ValueError(
                    "Fixed quadratic Harmonica coefficients were not provided."
                )
            fixed_values = jnp.asarray(fixed_values, dtype=jnp.float64)
            u1 = numpyro.deterministic('u1', fixed_values[:, 0])
            u2 = numpyro.deterministic('u2', fixed_values[:, 1])
        elif ld_mode in {'gaussian', 'stellarprior', 'uniform', 'fixed'}:
            c1, c2 = sample_power2_coefficients(
                ld_fixed if ld_mode == 'fixed' else mu_u_ld, sigma_u_ld,
                ld_mode, shape=(num_lcs,),
                maxted_smooth=ld_parameterization == 'maxted_smooth', gaussian_width=gaussian_ld_width,
            )
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
                rors, odd_specs, frac_sigma=odd_frac_sigma,
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
        poly_order = polynomial_order(detrend_components, minimum=int(bool(
            detrend_components & {'spot_spectroscopic','2spot_spectroscopic','linear_discontinuity_spectroscopic'})))
        names = ['c'] + ['v' if degree == 1 else f'v{degree}'
                         for degree in range(1, poly_order + 1)]
        if trend_basis == 'batman_normalized':
            transform = normalized_polynomial_transform(t, num_lcs, poly_order, likelihood_mask)
            beta = jnp.stack([
                numpyro.sample('trend_' + name, dist.Uniform(
                    *(trend_bounds or {}).get(name, (.5, 1.5) if name == 'c' else (-1., 1.))
                ).expand([num_lcs])) for name in names], axis=-1)
            physical = jnp.einsum('...ij,...j->...i', transform, beta)
            values = {name: numpyro.deterministic(name, physical[:, index])
                      for index, name in enumerate(names)}
            c_trend = values['c']
            slopes = {degree: values[name] for degree, name in enumerate(names) if degree}
        else:
            c_trend = (numpyro.sample('c', dist.Uniform(
                *(trend_bounds or {}).get('c', (.9, 1.1))).expand([num_lcs]))
                if has_offset_term else jnp.ones(num_lcs))
            slopes = {}
            for degree, name in enumerate(names[1:], 1):
                slopes[degree] = numpyro.sample(name, dist.Uniform(
                    *(trend_bounds or {}).get(name, (-.1, .1))).expand([num_lcs]))
        v_trend = slopes.get(1, jnp.zeros(num_lcs))

        has_explinear = 'explinear' in detrend_components
        if has_explinear:
            A_trend = numpyro.sample('A', dist.Uniform(*(trend_bounds or {}).get('A',(-0.1,0.1))).expand([num_lcs]))
            log_tau = numpyro.sample('log_tau', dist.Uniform(*(trend_bounds or {}).get('log_tau',(jnp.log(1e-3),jnp.log(1e-1)))).expand([num_lcs]))
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
        for degree in range(2, poly_order + 1):
            trend = trend + slopes[degree][:, None] * (t_norm ** degree)[None, :]
        if has_explinear:
            trend = trend + A_trend[:, None] * jnp.exp(-t_norm[None, :] / tau_trend[:, None])
        if 'gp_spectroscopic' in detrend_components:
            if gp_trend is None:
                raise ValueError("gp_spectroscopic requires gp_trend.")
            A_gp = numpyro.sample('A_gp', dist.Uniform(*(trend_bounds or {}).get('A_gp',(0.5,2))).expand([num_lcs]))
            trend = trend + A_gp[:, None] * gp_trend
        if 'spot_spectroscopic' in detrend_components:
            if spot_trend is None:
                raise ValueError("spot_spectroscopic requires spot_trend.")
            A_spot = numpyro.sample('A_spot', dist.Uniform(*(trend_bounds or {}).get('A_spot',(0.5,2))).expand([num_lcs]))
            trend = trend + A_spot[:, None] * spot_trend
        if '2spot_spectroscopic' in detrend_components:
            if spot_trend is None or spot_trend2 is None:
                raise ValueError("2spot_spectroscopic requires spot_trend and spot_trend2.")
            A_spot = numpyro.sample('A_spot', dist.Uniform(*(trend_bounds or {}).get('A_spot',(0.5,2))).expand([num_lcs]))
            A_spot2 = numpyro.sample('A_spot2', dist.Uniform(*(trend_bounds or {}).get('A_spot2',(0.5,2))).expand([num_lcs]))
            trend = trend + A_spot[:, None] * spot_trend + A_spot2[:, None] * spot_trend2
        if 'linear_discontinuity_spectroscopic' in detrend_components:
            if jump_trend is None:
                raise ValueError("linear_discontinuity_spectroscopic requires jump_trend.")
            A_jump = numpyro.sample('A_jump', dist.Uniform(*(trend_bounds or {}).get('A_jump',(0.5,2))).expand([num_lcs]))
            trend = trend + A_jump[:, None] * jump_trend

        y_model = apply_systematics(transit_sig, trend)
        with numpyro.handlers.mask(mask=True if likelihood_mask is None else likelihood_mask):
            numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model
