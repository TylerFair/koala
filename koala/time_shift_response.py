"""Boundary integrals and conditional fit response for two semicircular limbs.

No finite differences or injected-light-curve fits are used in production.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4)
def _arc_rule(order):
    x, w = np.polynomial.legendre.leggauss(order)
    x = (x + 1) / 2
    # Fourth-order endpoint map resolves fractional powers at the stellar limb.
    den = x**4 + (1-x)**4
    return x**4 / den, 2*w*x**3*(1-x)**3 / den**2


def projected_position(time, period, t0, a_rs, b, ecc=0., omega=0.):
    """Koala's inferior-conjunction Keplerian orbit, with x increasing at transit.

    Angles are radians; the arbitrary sky rotation does not affect the integral.
    These are the Cartesian components underlying core.orbit_separation.
    """
    import jax.numpy as jnp
    from jaxoplanet.core.kepler import kepler
    ecc = jnp.asarray(ecc, dtype=jnp.float64)
    f0 = .5*jnp.pi - omega
    e0 = 2*jnp.arctan2(jnp.sqrt(1-ecc)*jnp.sin(f0/2),
                       jnp.sqrt(1+ecc)*jnp.cos(f0/2))
    mean = e0-ecc*jnp.sin(e0)+2*jnp.pi*(time-t0)/period
    sf, cf = kepler(mean, ecc)
    sa = sf*jnp.cos(omega)+cf*jnp.sin(omega)
    ca = cf*jnp.cos(omega)-sf*jnp.sin(omega)
    radius = a_rs*(1-ecc**2)/(1+ecc*cf)
    cosi = b*(1+ecc*jnp.sin(omega))/(a_rs*(1-ecc**2))
    return jnp.stack((-radius*ca, radius*sa*cosi), axis=-1), sa > 0


def half_radius_responses(position, velocity, radius, coefficients, *,
                          law='quadratic', foreground=True, order=96):
    """Return (time, leading/trailing) derivatives of normalized stellar flux.

    Integrate only boundary arcs on the star, splitting at BOTH the stellar
    intersections and the moving half-disk division. No pixelization is used.
    Polynomial laws use I(mu)=1-sum(u_n*(1-mu)**(n+1)); power2 uses (c,alpha).
    """
    p, v = np.asarray(position, float), np.asarray(velocity, float)
    r = float(radius)
    u = np.asarray(coefficients, float)
    if law == 'power2':
        norm = np.pi*(1-u[0]+2*u[0]/(u[1]+2))
    else:
        n = np.arange(1, len(u)+1)
        norm = np.pi*(1-np.sum(2*u/((n+1)*(n+2))))
    if not np.isfinite(norm) or norm <= 0 or r <= 0:
        raise ValueError('Half-disk response requires positive radius and stellar flux.')
    z = np.linalg.norm(p, axis=-1)
    alpha = np.arctan2(-p[:, 1], -p[:, 0])
    beta = np.arctan2(v[:, 1], v[:, 0])
    safe_z = np.where(z > 0, z, 1.)
    width = np.arccos(np.clip((z*z+r*r-1)/(2*safe_z*r), -1, 1))
    width = np.where(z+r <= 1, np.pi, width)
    width = np.where((z >= 1+r) | (r >= 1+z), 0., width)
    # Rotate alpha into the leading half's local frame.
    center = (alpha-beta+np.pi) % (2*np.pi)-np.pi
    fraction, weights = _arc_rule(order)
    result = np.zeros((len(p), 2))
    for h in range(2):
        lo_half, hi_half = (-np.pi/2, np.pi/2) if h == 0 else (np.pi/2, 3*np.pi/2)
        for turn in (-1, 0, 1):
            lo = np.maximum(center-width+turn*2*np.pi, lo_half)
            hi = np.minimum(center+width+turn*2*np.pi, hi_half)
            span = np.maximum(hi-lo, 0.)
            angle = beta[:, None]+lo[:, None]+span[:, None]*fraction
            x = p[:, 0, None]+r*np.cos(angle)
            y = p[:, 1, None]+r*np.sin(angle)
            mu = np.sqrt(np.maximum(0., 1-x*x-y*y))
            if law == 'power2':
                intensity = 1-u[0]*(1-mu**u[1])
            else:
                intensity = 1-sum(c*(1-mu)**(n+1) for n,c in enumerate(u))
            result[:, h] -= r/norm*span*(intensity@weights)
    return result*np.broadcast_to(np.asarray(foreground), (len(p),))[:, None]


def solve_response(jacobian, half_responses, errors, prior_precision=None):
    """Solve the weighted normal equations in scaled physical coordinates.

    Scaling changes conditioning only; rank loss raises instead of silently
    regularizing an unidentified timing measurement.
    """
    j = np.asarray(jacobian, float)/np.asarray(errors, float)[:, None]
    y = np.asarray(half_responses, float)/np.asarray(errors, float)[:, None]
    p = np.zeros((j.shape[1], j.shape[1])) if prior_precision is None else np.asarray(prior_precision, float)
    scale = np.sqrt(np.sum(j*j, axis=0)+np.diag(p))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('Unidentified or nonfinite coefficient response.')
    scaled = j/scale
    normal = scaled.T@scaled + p/scale[:, None]/scale[None, :]
    return np.linalg.solve(normal, scaled.T@y)/scale[:, None]


_REPLAY_CACHE = {}


def fitted_bin_response(model, time, errors, samples, model_kwargs, *, kernel='auto',
                        ld_profile='quadratic', power2_exact=True,
                        detrend_type='none', marginalized=False):
    """Replay one fitted lane at its sample-site medians, including its priors.

    ``samples`` and channel-varying kwargs retain a length-one channel axis.
    Jitter is substituted but is excluded from the response coordinates.
    Marginalized Gaussian baseline coefficients are explicitly restored; this
    gives the same Schur complement as their analytic Gaussian integral.
    """
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from numpyro import handlers
    from models.jaxoplanet.core import compute_transit_model
    from models.detrend import _prepare_power2_poly
    from models.trend_marginal import (build_marginalized_trend_design,
                                       normalized_linear_trend_transform)

    t = jnp.asarray(time)
    err = jnp.atleast_2d(jnp.asarray(errors))
    kw = dict(model_kwargs)
    normalized = kw.pop('_response_normalized_trend', False)
    valid = np.isfinite(np.asarray(err)) & (np.asarray(err) > 0)
    if kw.get('likelihood_mask') is not None:
        valid &= np.broadcast_to(np.asarray(kw['likelihood_mask'], bool), valid.shape)
    kw['likelihood_mask'] = jnp.asarray(valid)
    err = jnp.where(valid, err, 1.)
    medians = {k: jnp.asarray(np.median(v, axis=0)) for k,v in samples.items()}

    def replay(values, t, err, kw):
        return handlers.trace(handlers.substitute(handlers.seed(model, 0), data=values)).get_trace(
            t, err, y=jnp.ones_like(err), **kw)

    initial = replay(medians, t, err, kw)
    power2_projection = _prepare_power2_poly() if 'u' not in initial else None
    names = [k for k,s in initial.items() if s['type'] == 'sample'
             and not s.get('is_observed', False) and k != 'log_jitter']
    fixed = {'log_jitter': medians['log_jitter']}
    coordinates = {k: medians[k] for k in names}
    if marginalized:
        _, beta_names = build_marginalized_trend_design(
            detrend_type, t, 1, tau=initial.get('tau', {}).get('value'),
            **{k: kw.get(k) for k in ('spot_trend','spot_trend2','jump_trend','exp_trend')})
        coordinates['_response_beta'] = jnp.stack([medians[n] for n in beta_names], axis=-1)
    theta, unravel = ravel_pytree(coordinates)

    def evaluate(x, t, err, kw, fixed):
        coords = unravel(x)
        values = {**fixed, **{k:v for k,v in coords.items() if k != '_response_beta'}}
        tr = replay(values, t, err, kw)
        if 'u' in tr:
            coefficients = tr['u']['value'][0]
        else:
            # The power-2 builder records c1/c2, but does not record its
            # polynomial projection as a deterministic u site.
            mu, projection = power2_projection
            c, alpha = tr['c1']['value'][0], tr['c2']['value'][0]
            coefficients = projection @ (c*(1-mu**alpha))
        transit_params = dict(
            period=kw['PERIOD'], t0=tr['t0']['value'][0],
            a_rs=kw['mu_a_rs'], b=kw['mu_b'], rors=tr['rors']['value'][0],
            ecc=kw.get('mu_ecc', 0.), omega=kw.get('mu_omega', 0.),
            u=coefficients, _jaxoplanet_kernel=kernel, _ld_profile=ld_profile)
        if ld_profile == 'power2' and power2_exact:
            transit_params.update(c1=tr['c1']['value'][0], c2=tr['c2']['value'][0], _power2_exact=True)
        transit = compute_transit_model(transit_params, t)
        if marginalized:
            dynamic_design, _ = build_marginalized_trend_design(
                detrend_type, t, 1, tau=tr.get('tau', {}).get('value'),
                **{k:kw.get(k) for k in ('spot_trend','spot_trend2','jump_trend','exp_trend')})
            trend = (dynamic_design @ coords['_response_beta'][..., None])[0, :, 0]
            flux = (1+transit)*trend
        else:
            observation = tr['obs']['fn']
            while not hasattr(observation, 'loc'):
                observation = observation.base_dist
            flux = observation.loc[0]
            trend = flux/(1+transit)
        numeric_params = {k:v for k,v in transit_params.items() if not k.startswith('_')}
        return flux, (numeric_params, trend)

    # Priors are evaluated in the same constrained sample coordinates as J.
    # Uniform densities have zero curvature away from hard support walls.
    def negative_log_prior(x, t, err, kw, fixed):
        coords = unravel(x)
        tr = replay({**fixed, **{k:v for k,v in coords.items() if k != '_response_beta'}}, t, err, kw)
        result = 0.
        for k,s in tr.items():
            if s['type'] == 'sample' and k not in ('obs','obs_marginalized','log_jitter'):
                result -= jnp.sum(s['fn'].log_prob(s['value']))
        if marginalized:
            beta = coords['_response_beta']
            # Gaussian trend priors may use normalized rather than elapsed-day coordinates.
            transform = normalized_linear_trend_transform(t, 1, beta_names, kw['likelihood_mask'])
            if normalized:
                beta = jnp.linalg.solve(transform, beta[..., None])[..., 0]
            mean = kw.get('trend_prior_mean', jnp.zeros_like(beta).at[:, 0].set(1.))
            scale = kw.get('trend_prior_scale', jnp.full_like(beta, .1))
            result += .5*jnp.sum(((beta-mean)/scale)**2)
        return result

    # Cache a compiled evaluator per model/layout. All bin-specific inputs are
    # dynamic arguments: neither masks nor errors nor LD get captured as constants.
    cache_key = (model, tuple((k, v.shape) for k,v in sorted(coordinates.items())),
                 kernel, ld_profile, power2_exact, detrend_type, marginalized, normalized)
    if cache_key not in _REPLAY_CACHE:
        if len(_REPLAY_CACHE) >= 8:
            _REPLAY_CACHE.pop(next(iter(_REPLAY_CACHE)))
        _REPLAY_CACHE[cache_key] = (
            jax.jit(evaluate),
            jax.jit(jax.jacfwd(lambda *args: evaluate(*args)[0])),
            jax.jit(jax.hessian(negative_log_prior)))
    primal, derivative, prior = _REPLAY_CACHE[cache_key]
    args = (theta, t, err, kw, fixed)
    _, (params, trend) = primal(*args)
    jac = np.asarray(derivative(*args))
    precision = np.asarray(prior(*args))
    # Recover exact flat-coordinate indices without depending on pytree key order.
    indices = unravel(jnp.arange(theta.size, dtype=float))
    keep = valid[0]
    sigma = np.sqrt(np.asarray(errors)[0]**2 + np.exp(float(medians['log_jitter'][0]))**2)
    rows = []
    for planet, radius in enumerate(np.asarray(params['rors'])):
        orbit = [float(np.ravel(params[k])[planet if np.size(params[k]) > 1 else 0])
                 for k in ('period','t0','a_rs','b','ecc','omega')]
        def position(tt):
            return projected_position(tt, *orbit)[0]
        p, v = jax.jvp(position, (t,), (jnp.ones_like(t),))
        foreground = projected_position(t, *orbit)[1]
        if ld_profile == 'power2' and power2_exact:
            coefficients, law = [params['c1'], params['c2']], 'power2'
        else:
            coefficients, law = params['u'], 'polynomial'
        halves = half_radius_responses(p, v, radius, coefficients,
                                       law=law, foreground=foreground)*np.asarray(trend)[:, None]
        ri = int(np.ravel(indices['rors'])[planet])
        ti = int(np.ravel(indices['dt0'])[planet])
        parity = np.linalg.norm(halves[keep].sum(1)-jac[keep, ri])/np.linalg.norm(jac[keep, ri])
        response = solve_response(jac[keep], halves[keep], sigma[keep], precision)
        left, right = response[ti]*86400
        dl, dr = 2*radius*response[ri]
        rows.append(dict(time_shift_coefficient=float(right),
                         d_dt0_dRL_s_per_Rstar=float(left), d_dt0_dRT_s_per_Rstar=float(right),
                         d_depth_dRL=float(dl), d_depth_dRT=float(dr),
                         antisymmetry_check=float(abs(left+right)/abs(right)),
                         radius_response_relative_error=float(parity), n_retained=int(keep.sum())))
    return rows


def spectroscopic_responses(fit, samples, errors, *, ld_profile='quadratic'):
    """Calculate each channel using its original sampler inputs, never binning time."""
    import time
    from .constants import JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS
    rows, elapsed = [], []
    for channel in range(np.shape(errors)[0]):
        start = time.perf_counter()
        kw = dict(fit.time_shift_model_kwargs)
        for key in (*JAXOPLANET_CHANNEL_VARYING_MODEL_KWARGS, 'likelihood_mask'):
            value = kw.get(key)
            if value is not None and np.ndim(value) > 0:
                if key != 'likelihood_mask' or np.ndim(value) > 1:
                    kw[key] = value[channel:channel+1]
        kw['_response_normalized_trend'] = fit.builder_kwargs.get('trend_basis') == 'batman_normalized'
        lane = {k:np.asarray(v)[:, channel:channel+1] for k,v in samples.items()
                if np.ndim(v) >= 2 and np.shape(v)[1] == np.shape(errors)[0]}
        rows.append(fitted_bin_response(
            getattr(fit, 'response_model', None) or fit.model, fit.time, np.asarray(errors)[channel:channel+1], lane, kw,
            kernel=fit.builder_kwargs.get('jaxoplanet_kernel', 'auto'), ld_profile=ld_profile,
            power2_exact=(fit.builder_kwargs.get('power2_evaluation', 'auto') != 'polynomial'
                          and getattr(fit, 'ld_mode', None) != 'interpolated'),
            detrend_type=fit.detrend_type, marginalized=fit.trend_mode == 'gaussian_marginalized'))
        elapsed.append(time.perf_counter()-start)
    return rows, elapsed
