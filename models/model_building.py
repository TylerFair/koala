"""Shared prior sites for the transit model builders."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .ld_parameterization import (Power2MaxtedTransform, Power2MaxtedUniform, Power2SmoothTransform,
                                  power2_smooth_transform)
from .limb_darkening_config import GAUSSIAN_LD_WIDTH, validate_gaussian_ld_width
from .gp import GP_HYPERPARAMETER_BOUNDS
from .trends import sample_ordered_spot_centers, sample_step_width


def polynomial_order(components, minimum=0):
    """Return the highest polynomial degree in a composite trend."""
    degrees = {'linear': 1, 'explinear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4}
    return max(minimum, max((degrees.get(name, 0) for name in components), default=0))


def _configured_trend_site(trend_priors, name, default):
    """Sample ``name`` from a configured ``trend_priors`` spec, else run ``default()``."""
    spec = (trend_priors or {}).get(name)
    if spec is None:
        return default()
    from .priors import sample_parameter
    return sample_parameter(spec, name)


def sample_whitelight_trend(t, prior, components, parameterization, step_width_mode,
                           two_spot_ordering='legacy', trend_priors=None):
    """Sample common systematics sites in their established RNG order.

    ``trend_priors`` optionally maps ``c``, ``v``, ``A`` and ``tau`` to a
    ``koala.config.ParameterSpec`` (the ``trend_priors`` config block). Keys
    that are absent keep the built-in priors and site names below.
    """
    def default_tau():
        log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)))
        return numpyro.deterministic('tau', jnp.exp(log_tau))

    cadence = jnp.median(jnp.diff(jnp.sort(jnp.asarray(t))))
    params = {}
    linear = {'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity',
              'explinear', 'spot', '2spot'}
    if components & (linear | {'constant', 'gp'}):
        params['c'] = _configured_trend_site(
            trend_priors, 'c', lambda: numpyro.sample('c', dist.Uniform(0.9, 1.1)))
    order = polynomial_order(components, minimum=int(bool(components & linear)))
    for degree in range(1, order + 1):
        name = 'v' if degree == 1 else f'v{degree}'
        params[name] = _configured_trend_site(
            trend_priors, name,
            lambda name=name: numpyro.sample(name, dist.Uniform(-0.1, 0.1)))

    def center(name, guess, width):
        if parameterization == 'cadence':
            offset = numpyro.sample(f'{name}_offset_cadences', dist.Normal(0.0, width / cadence))
            return numpyro.deterministic(name, guess + cadence * offset)
        return numpyro.sample(name, dist.Normal(guess, width))

    if 'linear_discontinuity' in components:
        guess = prior.get('t_jump_guess', 0.5 * (jnp.min(t) + jnp.max(t)))
        params['t_jump'] = center('t_jump', guess, 1e-2)
        params['jump'] = numpyro.sample('jump', dist.Normal(prior.get('jump_guess', 0.0), 0.01))
        params['width'] = sample_step_width(t, prior, step_width_mode, parameterization)
    if 'explinear' in components:
        params['A'] = _configured_trend_site(
            trend_priors, 'A', lambda: numpyro.sample('A', dist.Uniform(-0.1, 0.1)))

        # A uniform/gaussian tau is sampled at the site ``tau`` itself; the
        # trend kernels floor it (``models.trends.safe_tau``) so tau -> 0 stays finite.
        params['tau'] = _configured_trend_site(trend_priors, 'tau', default_tau)

    ordered = '2spot' in components and two_spot_ordering == 'ordered'
    spots = ('', '2') if '2spot' in components else ('',) if 'spot' in components else ()
    for suffix in spots:
        amp, mu, sigma = (f'spot_{name}{suffix}' for name in ('amp', 'mu', 'sigma'))
        guess = prior.get(f'spot_guess{suffix}', prior['spot_guess'])
        params[amp] = numpyro.sample(amp, dist.Uniform(0.0, 0.1))
        # The second cadence width historically precedes its center draw.
        if parameterization == 'cadence' and suffix:
            width = numpyro.sample(f'{sigma}_cadences', dist.Uniform(1e-4 / cadence, 0.1 / cadence))
        if not ordered:
            params[mu] = center(mu, guess, 0.01)
        if parameterization == 'cadence':
            if not suffix:
                width = numpyro.sample(f'{sigma}_cadences', dist.Uniform(1e-4 / cadence, 0.1 / cadence))
            params[sigma] = numpyro.deterministic(sigma, cadence * width)
        else:
            params[sigma] = numpyro.sample(sigma, dist.Uniform(1e-4, 0.1))
    if ordered:
        params['spot_mu'], params['spot_mu2'] = sample_ordered_spot_centers(
            prior['spot_guess'], prior.get('spot_guess2', prior['spot_guess']),
            cadence, parameterization=parameterization,
        )
    if 'gp' in components:
        for name in ('GP_log_sigma', 'GP_log_rho'):
            params[name] = numpyro.sample(name, dist.Uniform(*GP_HYPERPARAMETER_BOUNDS[name]))
    return params


def sample_sing_coefficients(mean, sigma):
    """Sample Sing coordinates with c1 >= 0, c1 + 2*c2 >= 0, c1+c2 <= 1."""
    if sigma is None:
        raise ValueError("ld_mode='sing' requires (l, delta) sigma_u_ld.")
    mean, sigma = jnp.asarray(mean, dtype=jnp.float64), jnp.asarray(sigma, dtype=jnp.float64)
    limb_l = numpyro.sample('limb_l', dist.TruncatedNormal(mean[:, 0], sigma[:, 0], low=0.0, high=1.0))
    u_plus = 1.0 - limb_l
    delta = numpyro.sample('limb_delta', dist.TruncatedNormal(
        mean[:, 1], sigma[:, 1], low=-u_plus / 4.0, high=u_plus / 4.0,
    ))
    c1 = numpyro.deterministic('c1', u_plus - 4.0 * delta)
    c2 = numpyro.deterministic('c2', 4.0 * delta)
    return c1, c2, numpyro.deterministic('u', jnp.stack((c1, c2), axis=1))


def _maxted_uniform_inside(coefficients):
    coefficients = jnp.asarray(coefficients)
    inside = jnp.isfinite(Power2MaxtedUniform().log_prob(jnp.nan_to_num(coefficients)))
    return inside & jnp.all(jnp.isfinite(coefficients), axis=-1)


def _enforce_maxted_uniform_support(coefficients):
    """Zero density outside the Maxted-uniform power-2 support, with safe model values there."""
    inside = _maxted_uniform_inside(coefficients)
    # exp(-1e100) is exactly zero in float64 yet keeps NumPyro's validated factor finite (see below).
    numpyro.factor("ld_maxted_uniform_support", jnp.where(inside, 0.0, -1.0e100))
    return jnp.where(inside[..., None], coefficients, jnp.asarray([0.5, 0.5], dtype=jnp.asarray(coefficients).dtype))


def _maxted_uniform_hard_wall(raw, prior, box_volume):
    """Turn a flat (c, alpha) box site of volume ``box_volume`` into the Maxted-uniform prior with hard walls."""
    inside = _maxted_uniform_inside(raw)
    safe = jnp.where(inside[..., None], raw, jnp.asarray([0.5, 0.5], dtype=jnp.asarray(raw).dtype))
    numpyro.factor("ld_maxted_uniform_prior",
                   jnp.where(inside, prior.log_prob(safe) + jnp.log(box_volume), -1.0e100))
    return safe


def _enforce_decorrelated_coefficient_support(coefficients, low, high):
    """Apply a numerically zero outside-support density and safe model values."""
    coefficients = jnp.asarray(coefficients)
    low = jnp.asarray(low, dtype=coefficients.dtype)
    high = jnp.asarray(high, dtype=coefficients.dtype)
    inside = jnp.all(
        jnp.isfinite(coefficients) & (coefficients >= low) & (coefficients <= high), axis=-1,
    )
    # Unit validates its log factor when NumPyro validation is enabled, so an
    # IEEE -inf raises before HMC can reject the point. exp(-1e100) is exactly
    # zero in float64 while remaining a valid finite Unit parameter.
    numpyro.factor(
        "ld_decorrelated_support", jnp.where(inside, 0.0, -1.0e100)
    )
    midpoint = 0.5 * (low + high)
    safe = jnp.nan_to_num(
        coefficients, nan=midpoint, posinf=high, neginf=low
    )
    return jnp.clip(safe, low, high)


def ld_prior_scale(mean, sigma, mode, *, clipped=True, sigma_name='sigma_u_ld',
                   gaussian_width=GAUSSIAN_LD_WIDTH):
    """Resolve the shared Gaussian width or the supplied stellar uncertainty."""
    if mode == 'stellarprior' and sigma is None:
        raise ValueError(f"ld_mode='stellarprior' requires {sigma_name}.")
    mean = jnp.asarray(mean, dtype=jnp.float64)
    scale = jnp.full_like(mean, validate_gaussian_ld_width(gaussian_width)) if mode == 'gaussian' else sigma
    if clipped:
        scale = jnp.asarray(scale, dtype=jnp.float64)
        scale = jnp.broadcast_to(scale, mean.shape)
        if mode != 'gaussian':
            scale = jnp.clip(scale, 1e-6, None)
    return scale


def sample_power2_coefficients(mean, sigma, mode, *, shape=(), decorrelated=False,
                               joint_uniform=False, maxted_smooth=False, sigma_name='sigma_u_ld',
                               gaussian_width=GAUSSIAN_LD_WIDTH):
    """Sample power-2 coefficients, retaining each builder's site structure."""
    if mode == 'fixed':
        coefficients = mean
    elif mode in {'gaussian', 'stellarprior', 'uniform'}:
        normal = mode != 'uniform'
        low = jnp.asarray([0.0, 0.001]) if normal else 0.0
        if normal:
            scale = ld_prior_scale(mean, sigma, mode, sigma_name=sigma_name, gaussian_width=gaussian_width)
            prior = dist.TruncatedNormal(mean, scale, low=low, high=1.0).to_event(1)
        else:
            # Uniform in the Maxted coordinates (h1, h2) = (I(1/2), I(1/2) - I(0)) on 0 < h1 < 1,
            # 0 < h2 < min(1, h1 + 0.5): c < 1.5 and any alpha > 0. A flat (c, alpha) box truncated alpha at 1 and
            # biased native depths (HATS-6 injection-recovery, 2026-10-07); see docs/tutorials/limb_darkening.md.
            prior = Power2MaxtedUniform(batch_shape=tuple(shape))
        if maxted_smooth and mode in {'gaussian', 'uniform'}:
            transform = power2_smooth_transform(mode)
            z = numpyro.sample('ld_smooth', dist.TransformedDistribution(prior, transform.inv))
            coefficients = transform(z)
        elif decorrelated and mode in {'gaussian', 'uniform'}:
            transform = Power2MaxtedTransform()
            h = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(prior, transform))
            if normal:
                coefficients = _enforce_decorrelated_coefficient_support(transform.inv(h), low, 1.0)
            else:
                coefficients = _enforce_maxted_uniform_support(transform.inv(h))
        elif not normal:
            # Direct (c, alpha) coordinates with hard walls; the prior is the same Maxted-uniform density
            # (alpha is capped at 40 here, which removes h2 < c * 2**-40 of the support).
            if joint_uniform:
                raw = numpyro.sample('ld_coefficients', dist.Uniform(
                    jnp.asarray([0.0, 0.0]), jnp.asarray([1.5, 40.0])).expand([*shape, 2]).to_event(1))
            else:
                raw = jnp.stack((numpyro.sample('c1', dist.Uniform(0.0, 1.5).expand(shape)),
                                 numpyro.sample('c2', dist.Uniform(0.0, 40.0).expand(shape))), axis=-1)
            coefficients = _maxted_uniform_hard_wall(raw, prior, 1.5 * 40.0)
            if not joint_uniform:
                return coefficients[..., 0], coefficients[..., 1]
        elif normal:
            c1 = numpyro.sample('c1', dist.TruncatedNormal(mean[..., 0], scale[..., 0], low=0.0, high=1.0))
            c2 = numpyro.sample('c2', dist.TruncatedNormal(mean[..., 1], scale[..., 1], low=0.001, high=1.0))
            return c1, c2
    else:
        raise ValueError(f"Unknown ld_mode: {mode}")
    return (numpyro.deterministic('c1', coefficients[..., 0]),
            numpyro.deterministic('c2', coefficients[..., 1]))


def sample_quadratic_fixed_u1(coefficients, shape=(), low=0.0, high=1.0):
    """Mukherjee-style quadratic prior: stellar u1 fixed, direct u2 uniform."""
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    u1 = numpyro.deterministic('u1', jnp.broadcast_to(coefficients[..., 0], tuple(shape)))
    u2 = numpyro.sample('u2', dist.Uniform(low, high).expand(tuple(shape)))
    return numpyro.deterministic('u', jnp.stack((u1, u2), axis=-1))


def sample_quadratic_uniform(shape, basis, low, high):
    """Sample direct coefficients, Kipping (q1, q2), or the sum/difference box prior."""
    if basis == 'kipping':
        return sample_kipping_coefficients(shape)
    if basis == 'coefficients':
        return numpyro.sample('u', dist.Uniform(low, high).expand([*shape, 2]).to_event(1))
    low, high = jnp.array([-1.0, -2.0]), jnp.array([2.0, 2.0])
    if shape:
        low, high = jnp.broadcast_to(low, (*shape, 2)), jnp.broadcast_to(high, (*shape, 2))
    values = numpyro.sample('ld_uplus_uminus', dist.Uniform(low, high).to_event(1))
    u1 = numpyro.deterministic('u1', 0.5 * (values[..., 0] + values[..., 1]))
    u2 = numpyro.deterministic('u2', 0.5 * (values[..., 0] - values[..., 1]))
    numpyro.deterministic('l', 1.0 - values[..., 0])
    numpyro.deterministic('delta', (values[..., 0] - values[..., 1]) / 8.0)
    return numpyro.deterministic('u', jnp.stack((u1, u2), axis=-1))


def kipping_u_from_q(q):
    """Kipping (2013) map: ``(q1, q2)`` in the unit square to quadratic ``(u1, u2)``."""
    q = jnp.asarray(q, dtype=jnp.float64)
    root = jnp.sqrt(q[..., 0])
    return jnp.stack((2.0 * root * q[..., 1], root * (1.0 - 2.0 * q[..., 1])), axis=-1)


def kipping_q_from_u(u):
    """Inverse Kipping map: ``q1 = (u1 + u2)**2``, ``q2 = u1 / (2 (u1 + u2))``."""
    u = jnp.asarray(u, dtype=jnp.float64)
    total = u[..., 0] + u[..., 1]
    safe = jnp.where(total == 0.0, 1.0, total)
    return jnp.stack((total ** 2, jnp.where(total == 0.0, 0.5, u[..., 0] / (2.0 * safe))), axis=-1)


def kipping_q_sigma(u, sigma_u):
    """Linearly propagate independent ``(u1, u2)`` widths to ``(q1, q2)`` widths."""
    u = jnp.asarray(u, dtype=jnp.float64)
    sigma_u = jnp.broadcast_to(jnp.asarray(sigma_u, dtype=jnp.float64), u.shape)
    u1, u2 = u[..., 0], u[..., 1]
    s1, s2 = sigma_u[..., 0], sigma_u[..., 1]
    total = u1 + u2
    total = jnp.where(jnp.abs(total) < 1e-6, 1e-6, total)
    sigma_q1 = 2.0 * jnp.abs(total) * jnp.sqrt(s1 ** 2 + s2 ** 2)
    sigma_q2 = jnp.sqrt((u2 * s1) ** 2 + (u1 * s2) ** 2) / (2.0 * total ** 2)
    return jnp.clip(jnp.stack((sigma_q1, sigma_q2), axis=-1), 1e-6, None)


def sample_kipping_coefficients(shape=(), mean_u=None, sigma_u=None):
    """Sample Kipping ``(q1, q2)`` and return the quadratic ``u``.

    With ``mean_u=None`` the prior is ``q1, q2 ~ U(0, 1)`` (uniform over the physical
    quadratic triangle). Otherwise it is a normal in ``q`` truncated to the unit square,
    centred on the converted mean with linearly propagated widths.
    """
    if mean_u is None:
        prior = dist.Uniform(0.0, 1.0).expand([*shape, 2]).to_event(1)
    else:
        mean_u = jnp.asarray(mean_u, dtype=jnp.float64)
        eps = 1e-6
        loc = jnp.clip(kipping_q_from_u(mean_u), eps, 1.0 - eps)
        prior = dist.TruncatedNormal(
            loc, kipping_q_sigma(mean_u, sigma_u), low=0.0, high=1.0).to_event(1)
    q = numpyro.sample('ld_q', prior)
    numpyro.deterministic('q1', q[..., 0])
    numpyro.deterministic('q2', q[..., 1])
    u = kipping_u_from_q(q)
    numpyro.deterministic('u1', u[..., 0])
    numpyro.deterministic('u2', u[..., 1])
    return numpyro.deterministic('u', u)


def posterior_geometry(samples, period, ecc=0.0, omega=0.0, *, harmonica=False):
    """Reconstruct orbital geometry while retaining each engine's conventions."""
    from .harmonica.core import (
        harmonica_a_rs_from_duration, harmonica_cos_i_from_geometry,
        harmonica_impact_param_from_cos_i, harmonica_duration_from_geometry,
        harmonica_duration_from_cos_i,
    )
    period, ecc, omega = (jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64))
                          for x in (period, ecc, omega))
    result = {}
    for i in range(len(period)):
        def value(name, latent=None, transform=None):
            if f'{name}_{i}' in samples:
                return samples[f'{name}_{i}']
            if latent and f'{latent}_{i}' in samples:
                return transform(samples[f'{latent}_{i}'])
            return None

        p = period[i] if harmonica else samples.get(f'period_{i}', period[i])
        orbit = dict(ecc=ecc[i], omega=omega[i])
        a_rs = value('a_rs', 'log_a_rs', jnp.exp)
        b = value('b', '_b', jnp.abs)
        duration = value('duration', 'logD', jnp.exp)
        cos_i, inc = value('cos_i'), value('inc')
        cos_limit = 1.0 - 1e-9 if harmonica else 1.0
        if cos_i is not None:
            inc = jnp.arccos(jnp.clip(cos_i, 0.0, cos_limit))
        elif inc is not None:
            cos_i = jnp.cos(inc)
        radius = samples[f'rors_{i}']
        if a_rs is None and duration is not None and b is not None:
            a_rs = harmonica_a_rs_from_duration(p, duration, b, radius, **orbit)
        if b is None and cos_i is not None and a_rs is not None:
            b = harmonica_impact_param_from_cos_i(cos_i, a_rs, **orbit)
        if cos_i is None and b is not None and a_rs is not None:
            cos_i = harmonica_cos_i_from_geometry(b, a_rs, **orbit)
            inc = jnp.arccos(jnp.clip(cos_i, 0.0, cos_limit))
        if duration is None and a_rs is not None:
            if harmonica and cos_i is not None:
                duration = harmonica_duration_from_cos_i(p, a_rs, cos_i, radius, **orbit)
            elif b is not None:
                duration = harmonica_duration_from_geometry(p, a_rs, b, radius, **orbit)
        if any(x is None for x in (a_rs, b, cos_i, duration)):
            continue
        values = dict(duration=duration, b=b, a_rs=a_rs, cos_i=cos_i, inc=inc)
        order = ('a_rs', 'inc', 'cos_i', 'b', 'duration') if harmonica else values
        result.update((f'{name}_{i}', values[name]) for name in order)
    return result


def sample_orbital_geometry(priors, n_planets, method, *, fixed_period=None,
                           geometry_fixed=False):
    """Sample orbital sites in order, deriving the complementary geometry."""
    from .priors import sample_parameter
    from .harmonica.core import (
        harmonica_a_rs_from_duration, harmonica_duration_from_geometry,
        harmonica_cos_i_from_geometry,
    )
    if not priors:
        raise ValueError("parameter_priors is required: every planet parameter must "
                         "carry a {value, prior, ...} specification (see koala.config).")
    eccentricity = jnp.asarray([spec.value for spec in priors['ecc']], dtype=jnp.float64)
    omega = jnp.asarray([spec.value for spec in priors['omega']], dtype=jnp.float64)
    # Static per planet: a configured e = 0 never needs the eccentric contact solver.
    circular = [spec.value == 0 for spec in priors['ecc']]
    result = {name: [] for name in ('period', 't0', 'rors', 'b', 'duration', 'a_rs', 'cos_i', 'inc')}
    if geometry_fixed:
        numpyro.deterministic('_geometry_fixed', jnp.asarray(True))
    for i in range(n_planets):
        def site(name):
            return sample_parameter(priors[name][i], f'{name}_{i}')

        period = site('period') if fixed_period is None else fixed_period[i]
        t0 = site('t0')
        if 'eclipse_time' in priors:
            numpyro.deterministic(f'eclipse_time_{i}', t0 + 0.5 * period)
        radius = sample_parameter(priors['rprs'][i], f'rors_{i}')
        numpyro.deterministic(f'depths_{i}', radius ** 2)
        b = site('b')
        orbit = dict(ecc=eccentricity[i], omega=omega[i])
        if method == 'duration':
            duration = site('duration')
            a_rs = numpyro.deterministic(f'a_rs_{i}', harmonica_a_rs_from_duration(
                period, duration, b, radius, **orbit, circular=circular[i],
            ))
        else:
            a_rs = site('a_rs')
            duration = numpyro.deterministic(f'duration_{i}', harmonica_duration_from_geometry(
                period, a_rs, b, radius, **orbit, circular=circular[i],
            ))
        cos_i = numpyro.deterministic(f'cos_i_{i}', harmonica_cos_i_from_geometry(b, a_rs, **orbit))
        limit = 1.0 if fixed_period is None else 1.0 - 1e-9
        inc = numpyro.deterministic(f'inc_{i}', jnp.arccos(jnp.clip(cos_i, 0.0, limit)))
        values = (period, t0, radius, b, duration, a_rs, cos_i, inc)
        for name, value in zip(result, values):
            result[name].append(value)
    return {**{name: jnp.array(value) for name, value in result.items()},
            'ecc': eccentricity, 'omega': omega}
