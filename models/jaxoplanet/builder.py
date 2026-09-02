import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np

from ..common import get_I_power2
from ..linear_marginalization import marginalized_log_likelihood_and_conditional
from ..ld_parameterization import (
    Power2LinearTransform,
    Power2MaxtedTransform,
    QuadraticKippingTransform,
    gaussian_to_truncated_normal,
    gaussian_to_uniform,
)
from ..trend_marginal import build_marginalized_trend_design
from ..detrend import (
    COMPUTE_KERNELS,
    resolve_detrend_kernel,
    _split_components,
    _prepare_power2_poly,
)
from ..gp import (
    build_gp, build_gp_linear, build_gp_quadratic, build_gp_cubic,
    build_gp_quartic, build_gp_explinear,
)
from ..trends import spot_crossing

from ..harmonica.core import (
    harmonica_a_rs_from_duration,
    harmonica_cos_i_from_geometry,
    harmonica_duration_from_geometry,
    harmonica_impact_param_from_cos_i,
)
from .core import (
    build_transit_phase_offsets,
    compute_transit_model,
    resolve_jaxoplanet_kernel,
)

NUTS_KWARGS = {
    "dense_mass": False,
    "regularize_mass_matrix": True,
    "target_accept_prob": 0.8,
}


def _resolve_builder_kernel(jaxoplanet_kernel, ld_profile, param_method, *, ld_mode=None):
    """Validate model-level kernel constraints and return the exact route."""

    requested = str(jaxoplanet_kernel).lower()
    if requested == "native_power2":
        if ld_profile != "power2":
            raise ValueError(
                "jaxoplanet_kernel='native_power2' requires ld_profile='power2'."
            )
        if param_method != "duration":
            raise ValueError(
                "jaxoplanet_kernel='native_power2' supports only "
                "param_method='duration'; a_rs/Keplerian geometry is unsupported."
            )
        if ld_mode == "interpolated":
            raise ValueError(
                "jaxoplanet_kernel='native_power2' does not support interpolated "
                "polynomial limb-darkening coefficients; provide direct Power-2 "
                "c1/c2 coefficients instead."
            )
    degree = 12 if ld_profile == "power2" else 2 if ld_profile == "quadratic" else None
    return resolve_jaxoplanet_kernel(
        requested,
        ld_profile=ld_profile,
        degree=degree,
        keplerian=param_method == "a_rs",
    )


def derive_geometry(wl_samples, period, ecc=0.0, omega=0.0):
    """Derive a consistent geometry bundle from any supported jaxoplanet WL parameterization."""
    period = jnp.atleast_1d(jnp.asarray(period, dtype=jnp.float64))
    ecc = jnp.atleast_1d(jnp.asarray(ecc, dtype=jnp.float64))
    omega = jnp.atleast_1d(jnp.asarray(omega, dtype=jnp.float64))
    n_planets = period.shape[0]

    result = {}
    for i in range(n_planets):
        if f"a_rs_{i}" in wl_samples:
            a_rs_samples = wl_samples[f"a_rs_{i}"]
        elif f"log_a_rs_{i}" in wl_samples:
            a_rs_samples = jnp.exp(wl_samples[f"log_a_rs_{i}"])
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
            inc_samples = jnp.arccos(jnp.clip(cos_i_samples, 0.0, 1.0))
        elif f"inc_{i}" in wl_samples:
            inc_samples = wl_samples[f"inc_{i}"]
            cos_i_samples = jnp.cos(inc_samples)
        else:
            cos_i_samples = None
            inc_samples = None

        if f"duration_{i}" in wl_samples:
            dur_samples = wl_samples[f"duration_{i}"]
        elif f"logD_{i}" in wl_samples:
            dur_samples = jnp.exp(wl_samples[f"logD_{i}"])
        else:
            dur_samples = None

        rors_samples = wl_samples[f"rors_{i}"]

        if a_rs_samples is None and dur_samples is not None and b_samples is not None:
            a_rs_samples = harmonica_a_rs_from_duration(
                period[i], dur_samples, b_samples, rors_samples,
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
            inc_samples = jnp.arccos(jnp.clip(cos_i_samples, 0.0, 1.0))
        if dur_samples is None and a_rs_samples is not None and b_samples is not None:
            dur_samples = harmonica_duration_from_geometry(
                period[i], a_rs_samples, b_samples, rors_samples,
                ecc=ecc[i], omega=omega[i],
            )

        if a_rs_samples is None or b_samples is None or cos_i_samples is None or dur_samples is None:
            continue

        result[f'duration_{i}'] = dur_samples
        result[f'b_{i}'] = b_samples
        result[f'a_rs_{i}'] = a_rs_samples
        result[f'cos_i_{i}'] = cos_i_samples
        result[f'inc_{i}'] = inc_samples

    return result


def create_whitelight_model(detrend_type='linear', n_planets=1, ld_profile='quadratic',
                            ld_mode='free', param_method='duration',
                            jaxoplanet_kernel='auto',
                            ld_parameterization='coefficients'):
    """Jaxoplanet white-light model with duration- or a_rs-based geometry."""
    if param_method not in ('duration', 'a_rs'):
        raise ValueError(f"Unknown param_method: {param_method}")
    if ld_parameterization not in {'coefficients', 'decorrelated', 'decorrelated_linear', 'latent_gaussian'}:
        raise ValueError("Unknown ld_parameterization.")
    power2_transform = (
        Power2LinearTransform()
        if ld_parameterization == 'decorrelated_linear'
        else Power2MaxtedTransform()
    )
    selected_kernel = _resolve_builder_kernel(
        jaxoplanet_kernel, ld_profile, param_method, ld_mode=ld_mode
    )
    detrend_components = _split_components(detrend_type)
    if ld_profile == "power2" and selected_kernel != "native_power2":
        MUS, P = _prepare_power2_poly()

    print(f"Building jaxoplanet whitelight model: detrend='{detrend_type}', "
          f"ld='{ld_mode}', ld_profile='{ld_profile}', "
          f"param_method='{param_method}', kernel='{selected_kernel}' "
          f"for {n_planets} planets")

    def _whitelight_model(t, yerr, y=None, prior_params=None):
        def _prior_array(name, default):
            arr = jnp.atleast_1d(jnp.asarray(prior_params.get(name, default), dtype=jnp.float64))
            if arr.size == 1 and n_planets > 1:
                return jnp.repeat(arr, n_planets)
            return arr

        eccs = _prior_array("ecc", 0.0)
        omegas = _prior_array("omega", 0.0)
        a_rs_prior_min = _prior_array("a_rs_prior_min", 2.0)
        a_rs_prior_max = _prior_array("a_rs_prior_max", 100.0)

        durations, t0s, bs, rorss, a_rss, cos_is, incs = [], [], [], [], [], [], []

        for i in range(n_planets):
            t0s.append(numpyro.sample(f"t0_{i}", dist.Uniform(jnp.min(t), jnp.max(t))))
            rors_i = numpyro.sample(f"rors_{i}", dist.Uniform(jnp.sqrt(1e-6), jnp.sqrt(0.5)))
            numpyro.deterministic(f"depths_{i}", rors_i ** 2)
            rorss.append(rors_i)

            _b = numpyro.sample(f"_b_{i}", dist.Uniform(-2.0, 2.0))
            b_i = numpyro.deterministic(f'b_{i}', jnp.abs(_b))
            bs.append(b_i)

            if param_method == 'duration':
                logD = numpyro.sample(f"logD_{i}", dist.Uniform(jnp.log(0.0007), jnp.log(1)))
                duration_i = numpyro.deterministic(f"duration_{i}", jnp.exp(logD))
                a_rs_i = numpyro.deterministic(
                    f"a_rs_{i}",
                    harmonica_a_rs_from_duration(
                        prior_params['period'][i], duration_i, b_i, rors_i,
                        ecc=eccs[i], omega=omegas[i],
                    ),
                )
            else:
                log_a_rs = numpyro.sample(
                    f"log_a_rs_{i}",
                    dist.Uniform(
                        jnp.log(jnp.maximum(a_rs_prior_min[i], 1e-6)),
                        jnp.log(jnp.maximum(a_rs_prior_max[i], a_rs_prior_min[i] + 1e-6)),
                    ),
                )
                a_rs_i = numpyro.deterministic(f"a_rs_{i}", jnp.exp(log_a_rs))
                duration_i = numpyro.deterministic(
                    f"duration_{i}",
                    harmonica_duration_from_geometry(
                        prior_params['period'][i], a_rs_i, b_i, rors_i,
                        ecc=eccs[i], omega=omegas[i],
                    ),
                )

            cos_i_i = numpyro.deterministic(
                f"cos_i_{i}",
                harmonica_cos_i_from_geometry(
                    b_i, a_rs_i, ecc=eccs[i], omega=omegas[i],
                ),
            )
            inc_i = numpyro.deterministic(
                f"inc_{i}", jnp.arccos(jnp.clip(cos_i_i, 0.0, 1.0))
            )

            durations.append(duration_i)
            a_rss.append(a_rs_i)
            cos_is.append(cos_i_i)
            incs.append(inc_i)

        if ld_profile == 'quadratic':
            if ld_mode in {'free', 'widegaussian', 'informed'}:
                u_prior = jnp.asarray(prior_params['u'], dtype=jnp.float64)
                if ld_mode == 'informed':
                    if 'u_sigma' not in prior_params:
                        raise ValueError("ld_mode='informed' requires prior_params['u_sigma'].")
                    u_sigma = jnp.asarray(prior_params['u_sigma'], dtype=jnp.float64)
                else:
                    u_sigma = jnp.asarray(prior_params.get('u_sigma', jnp.array([0.2, 0.2])), dtype=jnp.float64)
                u_sigma = jnp.broadcast_to(u_sigma, u_prior.shape)
                u_sigma = jnp.clip(u_sigma, 1e-6, None)
                coefficient_prior = dist.TruncatedNormal(
                    loc=u_prior, scale=u_sigma, low=0.0, high=1.0
                ).to_event(1)
                if ld_parameterization == 'latent_gaussian' and ld_mode in {'free', 'widegaussian'}:
                    z = numpyro.sample('ld_latent', dist.Normal(0.0, 1.0).expand([2]).to_event(1))
                    u = numpyro.deterministic(
                        'u', gaussian_to_truncated_normal(z, u_prior, u_sigma, 0.0, 1.0)
                    )
                elif ld_parameterization in {'decorrelated', 'decorrelated_linear'} and ld_mode in {'free', 'widegaussian'}:
                    q = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, QuadraticKippingTransform()))
                    u = numpyro.deterministic('u', QuadraticKippingTransform().inv(q))
                else:
                    u = numpyro.sample("u", coefficient_prior)
            elif ld_mode == 'uniform':
                coefficient_prior = dist.Uniform(0.0, 1.0).expand([2]).to_event(1)
                if ld_parameterization == 'latent_gaussian':
                    z = numpyro.sample('ld_latent', dist.Normal(0.0, 1.0).expand([2]).to_event(1))
                    u = numpyro.deterministic('u', gaussian_to_uniform(z, 0.0, 1.0))
                elif ld_parameterization in {'decorrelated', 'decorrelated_linear'}:
                    q = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, QuadraticKippingTransform()))
                    u = numpyro.deterministic('u', QuadraticKippingTransform().inv(q))
                else:
                    u = numpyro.sample("u", coefficient_prior)
            elif ld_mode == 'fixed':
                u = numpyro.deterministic("u", jnp.asarray(prior_params['u'], dtype=jnp.float64))
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")
        elif ld_profile == 'power2':
            u_prior = jnp.asarray(prior_params['u'], dtype=jnp.float64)
            if ld_mode in {'free', 'widegaussian', 'informed'}:
                if ld_mode == 'informed':
                    if 'u_sigma' not in prior_params:
                        raise ValueError("ld_mode='informed' requires prior_params['u_sigma'].")
                    u_sigma = jnp.asarray(prior_params['u_sigma'], dtype=jnp.float64)
                else:
                    u_sigma = jnp.asarray(prior_params.get('u_sigma', jnp.array([0.2, 0.2])), dtype=jnp.float64)
                u_sigma = jnp.broadcast_to(u_sigma, u_prior.shape)
                u_sigma = jnp.clip(u_sigma, 1e-6, None)
                coefficient_prior = dist.TruncatedNormal(
                    u_prior, u_sigma, low=jnp.asarray([0.0, 0.001]), high=1.0
                ).to_event(1)
                if ld_parameterization == 'latent_gaussian' and ld_mode in {'free', 'widegaussian'}:
                    z = numpyro.sample('ld_latent', dist.Normal(0.0, 1.0).expand([2]).to_event(1))
                    coefficients = gaussian_to_truncated_normal(
                        z, u_prior, u_sigma, jnp.asarray([0.0, 0.001]), 1.0
                    )
                    c1 = numpyro.deterministic('c1', coefficients[0])
                    c2 = numpyro.deterministic('c2', coefficients[1])
                elif ld_parameterization in {'decorrelated', 'decorrelated_linear'} and ld_mode in {'free', 'widegaussian'}:
                    h = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, power2_transform))
                    coefficients = power2_transform.inv(h)
                    c1 = numpyro.deterministic('c1', coefficients[0])
                    c2 = numpyro.deterministic('c2', coefficients[1])
                else:
                    c1 = numpyro.sample('c1', dist.TruncatedNormal(
                        u_prior[0], u_sigma[0], low=0.0, high=1.0))
                    c2 = numpyro.sample('c2', dist.TruncatedNormal(
                        u_prior[1], u_sigma[1], low=0.001, high=1.0))
            elif ld_mode == 'uniform':
                coefficient_prior = dist.Uniform(0.0, 1.0).expand([2]).to_event(1)
                if ld_parameterization == 'latent_gaussian':
                    z = numpyro.sample('ld_latent', dist.Normal(0.0, 1.0).expand([2]).to_event(1))
                    coefficients = gaussian_to_uniform(z, 0.0, 1.0)
                    c1 = numpyro.deterministic('c1', coefficients[0])
                    c2 = numpyro.deterministic('c2', coefficients[1])
                elif ld_parameterization in {'decorrelated', 'decorrelated_linear'}:
                    h = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, power2_transform))
                    coefficients = power2_transform.inv(h)
                    c1 = numpyro.deterministic('c1', coefficients[0])
                    c2 = numpyro.deterministic('c2', coefficients[1])
                else:
                    c1 = numpyro.sample('c1', dist.Uniform(0.0, 1.0))
                    c2 = numpyro.sample('c2', dist.Uniform(0.0, 1.0))
            elif ld_mode == 'fixed':
                c1 = numpyro.deterministic('c1', u_prior[0])
                c2 = numpyro.deterministic('c2', u_prior[1])
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")
            if selected_kernel != "native_power2":
                prof = get_I_power2(c1, c2, MUS)
                u = P @ (1.0 - prof)
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")

        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter) ** 2 + yerr ** 2))

        params = {
            "period": prior_params['period'],
            "duration": jnp.array(durations),
            "t0": jnp.array(t0s),
            "b": jnp.array(bs),
            "rors": jnp.array(rorss),
            "_jaxoplanet_kernel": jaxoplanet_kernel,
            "_ld_profile": ld_profile,
        }
        if selected_kernel == "native_power2":
            params["c1"] = c1
            params["c2"] = c2
        else:
            params["u"] = u
        if param_method == 'a_rs':
            params["a_rs"] = jnp.array(a_rss)
            params["ecc"] = eccs
            params["omega"] = omegas

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
            params['t_jump'] = numpyro.sample('t_jump', dist.Normal(t_jump_guess, 1e-2))
            params['jump'] = numpyro.sample('jump', dist.Normal(jump_guess, 0.01))

        if 'explinear' in detrend_components:
            params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1))
            log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)))
            params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))

        if not detrend_components.isdisjoint({'spot', '2spot'}):
            params['spot_amp'] = numpyro.sample('spot_amp', dist.Uniform(0.0, 0.1))
            params['spot_mu'] = numpyro.sample('spot_mu', dist.Normal(prior_params['spot_guess'], 0.01))
            params['spot_sigma'] = numpyro.sample('spot_sigma', dist.Uniform(1e-4, 0.1))
        if '2spot' in detrend_components:
            spot_guess2 = prior_params.get('spot_guess2', prior_params['spot_guess'])
            params['spot_amp2'] = numpyro.sample('spot_amp2', dist.Uniform(0.0, 0.1))
            params['spot_mu2'] = numpyro.sample('spot_mu2', dist.Normal(spot_guess2, 0.01))
            params['spot_sigma2'] = numpyro.sample('spot_sigma2', dist.Uniform(1e-4, 0.1))

        if 'gp' in detrend_components:
            params['GP_log_sigma'] = numpyro.sample('GP_log_sigma', dist.Uniform(jnp.log(1e-5), jnp.log(1e3)))
            params['GP_log_rho'] = numpyro.sample('GP_log_rho', dist.Uniform(jnp.log(0.007), jnp.log(0.3)))

        if 'gp' in detrend_type:
            if detrend_type == 'gp':
                gp_builder = build_gp
            elif detrend_type == 'linear+gp':
                gp_builder = build_gp_linear
            elif detrend_type == 'quadratic+gp':
                gp_builder = build_gp_quadratic
            elif detrend_type == 'cubic+gp':
                gp_builder = build_gp_cubic
            elif detrend_type == 'quartic+gp':
                gp_builder = build_gp_quartic
            elif detrend_type == 'explinear+gp':
                gp_builder = build_gp_explinear
            else:
                raise ValueError(f"Unknown GP detrend_type: {detrend_type}")
            gp = gp_builder(params, t, error)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            try:
                lc_model = resolve_detrend_kernel(detrend_type)(params, t)
                numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
            except KeyError:
                raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model


def create_vectorized_model(detrend_type='linear', ld_mode='free', trend_mode='free',
                            n_planets=1, ld_profile='quadratic',
                            param_method='duration', transit_window='off',
                            transit_window_indices=None,
                            jaxoplanet_kernel='auto',
                            jitter_prior='log_uniform',
                            jitter_prior_scale=2.0,
                            jitter_prior_center=0.5,
                            ld_parameterization='coefficients'):
    """Jaxoplanet spectroscopic model with WL-fixed duration- or a_rs-based geometry.

    ``jitter_prior`` selects the prior on the per-channel white-noise jitter:

    * ``'log_uniform'`` (default, historical): ``log_jitter ~ Uniform(log 1e-6, 0)``.
      When the jitter is not identified by the data this leaves a flat plateau
      in log-jitter down to the prior floor, which is strongly non-Gaussian.
    * ``'lognormal'``: ``log_jitter ~ Normal(log(jitter_prior_center * median(yerr)),
      jitter_prior_scale)``.  The posterior is likelihood-dominated whenever the
      jitter is identified (posterior widths ~0.1-0.4 in log versus a prior
      width of 2.0 e-folds) and reduces to a smooth Gaussian tail instead of a
      plateau when it is not.  The site name, shape, and downstream
      ``total_error`` deterministic are unchanged.
    """
    if param_method not in {'duration', 'a_rs'}:
        raise ValueError(f"Unknown param_method: {param_method}")
    if ld_parameterization not in {'coefficients', 'decorrelated', 'decorrelated_linear', 'latent_gaussian'}:
        raise ValueError("Unknown ld_parameterization.")
    power2_transform = (
        Power2LinearTransform()
        if ld_parameterization == 'decorrelated_linear'
        else Power2MaxtedTransform()
    )
    if jitter_prior not in {'log_uniform', 'lognormal'}:
        raise ValueError(f"Unknown jitter_prior: {jitter_prior}")
    jitter_prior_scale = float(jitter_prior_scale)
    jitter_prior_center = float(jitter_prior_center)
    if jitter_prior_scale <= 0.0 or jitter_prior_center <= 0.0:
        raise ValueError("jitter_prior_scale and jitter_prior_center must be > 0.")
    if trend_mode not in {'free', 'fixed', 'gaussian_marginalized'}:
        raise ValueError(f"Unknown trend_mode: {trend_mode}")
    selected_kernel = _resolve_builder_kernel(
        jaxoplanet_kernel, ld_profile, param_method, ld_mode=ld_mode
    )
    if transit_window not in {'auto', 'off'}:
        raise ValueError("transit_window must be either 'auto' or 'off'.")
    use_transit_window = (
        transit_window == 'auto'
        and transit_window_indices is not None
        and param_method == 'duration'
    )
    if transit_window_indices is not None:
        transit_window_indices = jnp.asarray(
            transit_window_indices, dtype=jnp.int32
        )
        if transit_window_indices.ndim != 1:
            raise ValueError("transit_window_indices must be one-dimensional.")
    detrend_components = _split_components(detrend_type)
    unsupported_non_spectroscopic = {'spot', '2spot', 'linear_discontinuity'}
    if detrend_components & unsupported_non_spectroscopic:
        unsupported = ", ".join(sorted(detrend_components & unsupported_non_spectroscopic))
        raise ValueError(
            "Vectorized model does not support non-spectroscopic spot/jump trends "
            f"({unsupported}). Use the corresponding *_spectroscopic trend family instead."
        )
    compute_lc_kernel = resolve_detrend_kernel(detrend_type)
    if ld_profile == "power2" and selected_kernel != "native_power2":
        MUS_LD, P_LD = _prepare_power2_poly()

    print(f"Building jaxoplanet vectorized model: detrend='{detrend_type}', "
          f"ld='{ld_mode}', trend='{trend_mode}', "
          f"param_method='{param_method}', kernel='{selected_kernel}', "
          f"jitter_prior='{jitter_prior}' "
          f"for {n_planets} planets")

    def _vectorized_model(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                          mu_depths=None, PERIOD=None, trend_fixed=None,
                          mu_a_rs=None, mu_ecc=0.0, mu_omega=0.0,
                          ld_interpolated=None, ld_fixed=None,
                          mu_u_ld=None, sigma_u_ld=None, gp_trend=None, spot_trend=None,
                          spot_trend2=None, jump_trend=None,
                          exp_trend=None, fixed_tau=None,
                          precomputed_yerr_per_lc=None,
                          trend_prior_mean=None, trend_prior_scale=None,
                          likelihood_mask=None):

        num_lcs = jnp.atleast_2d(yerr).shape[0]
        t0s = mu_t0
        bs = mu_b

        if param_method == 'duration':
            orbital_params = {"duration": mu_duration}
        else:
            if mu_a_rs is None:
                raise ValueError("mu_a_rs must be provided when param_method='a_rs'.")
            orbital_params = {
                "a_rs": jnp.asarray(mu_a_rs, dtype=jnp.float64),
                "ecc": jnp.asarray(mu_ecc, dtype=jnp.float64),
                "omega": jnp.asarray(mu_omega, dtype=jnp.float64),
            }

        rors = numpyro.sample(
            "rors",
            dist.Uniform(jnp.sqrt(1e-5), jnp.sqrt(0.5)).expand([num_lcs, n_planets]),
        )
        depths = numpyro.deterministic("depths", rors ** 2)

        yerr_matrix = jnp.atleast_2d(
            jnp.asarray(yerr, dtype=jnp.float64)
        )
        if yerr_matrix.shape[0] == 1 and num_lcs > 1:
            yerr_matrix = jnp.broadcast_to(
                yerr_matrix, (num_lcs, yerr_matrix.shape[1])
            )
        if precomputed_yerr_per_lc is None:
            yerr_per_lc = jnp.nanmedian(yerr_matrix, axis=1)
        else:
            yerr_per_lc = jnp.asarray(
                precomputed_yerr_per_lc, dtype=jnp.float64
            )
        if jitter_prior == 'lognormal':
            log_jitter = numpyro.sample(
                'log_jitter',
                dist.Normal(
                    jnp.log(jitter_prior_center * yerr_per_lc),
                    jnp.full((num_lcs,), jitter_prior_scale, dtype=jnp.float64),
                ),
            )
        else:
            log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-6), jnp.log(1)).expand([num_lcs]))
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter ** 2 + yerr_per_lc ** 2))
        # ``total_error`` is a compact per-channel posterior summary.  The
        # likelihood itself must retain the reported time-dependent errors;
        # replacing them with their median changes the statistical weighting
        # whenever a JWST integration has a different uncertainty.
        error_broadcast = jnp.sqrt(
            jitter[:, None] ** 2 + yerr_matrix ** 2
        )

        if ld_mode in {'free', 'widegaussian', 'informed', 'sing', 'sing_free'}:
            if ld_profile == 'quadratic':
                if ld_mode in {'sing', 'sing_free'}:
                    if ld_mode == 'sing' and sigma_u_ld is None:
                        raise ValueError("ld_mode='sing' requires (l, delta) sigma_u_ld.")
                    if ld_mode == 'sing_free':
                        limb_l = numpyro.sample(
                            'limb_l', dist.Uniform(0.0, 1.0).expand([num_lcs])
                        )
                    else:
                        sing_mu = jnp.asarray(mu_u_ld, dtype=jnp.float64)
                        sing_sigma = jnp.asarray(sigma_u_ld, dtype=jnp.float64)
                        limb_l = numpyro.sample(
                            'limb_l',
                            dist.TruncatedNormal(sing_mu[:, 0], sing_sigma[:, 0], low=0.0, high=1.0),
                        )
                    u_plus = 1.0 - limb_l
                    # c1 >= 0, c1 + 2*c2 >= 0, and c1+c2 <= 1.
                    if ld_mode == 'sing_free':
                        limb_delta = numpyro.sample(
                            'limb_delta', dist.Uniform(-u_plus / 4.0, u_plus / 4.0)
                        )
                    else:
                        limb_delta = numpyro.sample(
                            'limb_delta',
                            dist.TruncatedNormal(
                                sing_mu[:, 1], sing_sigma[:, 1],
                                low=-u_plus / 4.0, high=u_plus / 4.0,
                            ),
                        )
                    c1 = numpyro.deterministic('c1', u_plus - 4.0 * limb_delta)
                    c2 = numpyro.deterministic('c2', 4.0 * limb_delta)
                    u = numpyro.deterministic('u', jnp.stack((c1, c2), axis=1))
                else:
                    if ld_mode == 'informed' and sigma_u_ld is None:
                        raise ValueError("ld_mode='informed' requires sigma_u_ld.")
                    u_scale = sigma_u_ld if sigma_u_ld is not None else 0.2
                    u_prior_dist = dist.TruncatedNormal(
                        loc=mu_u_ld, scale=u_scale, low=0.0, high=1.0
                    ).to_event(1)
                    if ld_parameterization == 'latent_gaussian' and ld_mode in {'free', 'widegaussian'}:
                        z = numpyro.sample(
                            'ld_latent',
                            dist.Normal(0.0, 1.0).expand([num_lcs, 2]).to_event(1),
                        )
                        u = numpyro.deterministic(
                            'u',
                            gaussian_to_truncated_normal(
                                z, mu_u_ld, u_scale, 0.0, 1.0
                            ),
                        )
                    elif ld_parameterization in {'decorrelated', 'decorrelated_linear'} and ld_mode in {'free', 'widegaussian'}:
                        q = numpyro.sample(
                            'ld_decorrelated',
                            dist.TransformedDistribution(
                                u_prior_dist, QuadraticKippingTransform()
                            ),
                        )
                        u = numpyro.deterministic(
                            'u', QuadraticKippingTransform().inv(q)
                        )
                    else:
                        u = numpyro.sample('u', u_prior_dist)
            elif ld_profile == 'power2':
                if ld_mode == 'informed' and sigma_u_ld is None:
                    raise ValueError("ld_mode='informed' requires sigma_u_ld.")
                if sigma_u_ld is None:
                    sigma_u_ld = jnp.full_like(mu_u_ld, 0.2)
                sigma_u_ld = jnp.asarray(sigma_u_ld, dtype=jnp.float64)
                sigma_u_ld = jnp.broadcast_to(sigma_u_ld, mu_u_ld.shape)
                sigma_u_ld = jnp.clip(sigma_u_ld, 1e-6, None)
                coefficient_prior = dist.TruncatedNormal(
                    mu_u_ld, sigma_u_ld,
                    low=jnp.asarray([0.0, 0.001]), high=1.0,
                ).to_event(1)
                if ld_parameterization == 'latent_gaussian' and ld_mode in {'free', 'widegaussian'}:
                    z = numpyro.sample(
                        'ld_latent',
                        dist.Normal(0.0, 1.0).expand([num_lcs, 2]).to_event(1),
                    )
                    coefficients = gaussian_to_truncated_normal(
                        z,
                        mu_u_ld,
                        sigma_u_ld,
                        jnp.asarray([0.0, 0.001]),
                        1.0,
                    )
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                elif ld_parameterization in {'decorrelated', 'decorrelated_linear'} and ld_mode in {'free', 'widegaussian'}:
                    h = numpyro.sample(
                        'ld_decorrelated',
                        dist.TransformedDistribution(
                            coefficient_prior, power2_transform
                        ),
                    )
                    coefficients = power2_transform.inv(h)
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                else:
                    c1 = numpyro.sample('c1', dist.TruncatedNormal(
                        mu_u_ld[:, 0], sigma_u_ld[:, 0], low=0.0, high=1.0))
                    c2 = numpyro.sample('c2', dist.TruncatedNormal(
                        mu_u_ld[:, 1], sigma_u_ld[:, 1], low=0.001, high=1.0))
                if selected_kernel != "native_power2":
                    profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
                    u = (P_LD @ (1.0 - profs).T).T
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        elif ld_mode == 'uniform':
            if ld_profile == 'quadratic':
                coefficient_prior = dist.Uniform(0.0, 1.0).expand([num_lcs, 2]).to_event(1)
                if ld_parameterization == 'latent_gaussian':
                    z = numpyro.sample(
                        'ld_latent',
                        dist.Normal(0.0, 1.0).expand([num_lcs, 2]).to_event(1),
                    )
                    u = numpyro.deterministic(
                        'u', gaussian_to_uniform(z, 0.0, 1.0)
                    )
                elif ld_parameterization in {'decorrelated', 'decorrelated_linear'}:
                    q = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, QuadraticKippingTransform()))
                    u = numpyro.deterministic('u', QuadraticKippingTransform().inv(q))
                else:
                    u = numpyro.sample('u', coefficient_prior)
            elif ld_profile == 'power2':
                coefficient_prior = dist.Uniform(0.0, 1.0).expand([num_lcs, 2]).to_event(1)
                if ld_parameterization == 'latent_gaussian':
                    z = numpyro.sample(
                        'ld_latent',
                        dist.Normal(0.0, 1.0).expand([num_lcs, 2]).to_event(1),
                    )
                    coefficients = gaussian_to_uniform(z, 0.0, 1.0)
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                elif ld_parameterization in {'decorrelated', 'decorrelated_linear'}:
                    h = numpyro.sample('ld_decorrelated', dist.TransformedDistribution(
                        coefficient_prior, power2_transform))
                    coefficients = power2_transform.inv(h)
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                else:
                    coefficients = numpyro.sample('ld_coefficients', coefficient_prior)
                    c1 = numpyro.deterministic('c1', coefficients[:, 0])
                    c2 = numpyro.deterministic('c2', coefficients[:, 1])
                if selected_kernel != "native_power2":
                    profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
                    u = (P_LD @ (1.0 - profs).T).T
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        elif ld_mode == 'fixed':
            if ld_profile == 'quadratic':
                u = numpyro.deterministic('u', ld_fixed)
            elif ld_profile == 'power2':
                c1_mu = numpyro.deterministic('c1', ld_fixed[:, 0])
                c2_mu = numpyro.deterministic('c2', ld_fixed[:, 1])
                c1, c2 = c1_mu, c2_mu
                if selected_kernel != "native_power2":
                    profs = get_I_power2(c1_mu[:, None], c2_mu[:, None], MUS_LD[None, :])
                    u = (P_LD @ (1.0 - profs).T).T
        elif ld_mode == 'interpolated':
            u = numpyro.deterministic('u', ld_interpolated)
        else:
            raise ValueError(f"Unknown ld_mode: {ld_mode}")

        params = {
            "period": PERIOD, "t0": t0s, "b": bs,
            "rors": rors,
            "_jaxoplanet_kernel": jaxoplanet_kernel,
            "_ld_profile": ld_profile,
            **orbital_params,
        }
        in_axes = {
            "period": None, "t0": None, "b": None,
            "rors": 0,
            "_jaxoplanet_kernel": None,
            "_ld_profile": None,
        }
        if selected_kernel == "native_power2":
            params.update({"c1": c1, "c2": c2})
            in_axes.update({"c1": 0, "c2": 0})
        else:
            params["u"] = u
            in_axes["u"] = 0
        if param_method == 'duration':
            in_axes["duration"] = None
            phase_offsets, phase_mask = build_transit_phase_offsets(
                t, PERIOD, t0s, mu_duration
            )
            params["_transit_phase_offsets"] = phase_offsets
            params["_transit_phase_mask"] = phase_mask
            in_axes["_transit_phase_offsets"] = None
            in_axes["_transit_phase_mask"] = None
        else:
            in_axes["a_rs"] = None
            in_axes["ecc"] = None
            in_axes["omega"] = None
        if use_transit_window:
            params["_transit_window_indices"] = transit_window_indices
            in_axes["_transit_window_indices"] = None

        if trend_mode == 'gaussian_marginalized':
            if y is None:
                raise ValueError(
                    "Gaussian trend marginalization requires observed y values."
                )
            tau = None
            if 'explinear_spectroscopic' in detrend_components:
                if exp_trend is None or fixed_tau is None:
                    raise ValueError("explinear_spectroscopic requires exp_trend and fixed_tau.")
                tau = numpyro.deterministic(
                    'tau', jnp.full((num_lcs,), jnp.asarray(fixed_tau, dtype=jnp.float64))
                )
            elif 'explinear' in detrend_components:
                log_tau = numpyro.sample(
                    'log_tau',
                    dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]),
                )
                tau = numpyro.deterministic('tau', jnp.exp(log_tau))
            transit_model = jax.vmap(
                compute_transit_model,
                in_axes=(in_axes, None),
            )(params, t)
            design, coefficient_names = build_marginalized_trend_design(
                detrend_type,
                t,
                num_lcs,
                tau=tau,
                spot_trend=spot_trend,
                spot_trend2=spot_trend2,
                jump_trend=jump_trend,
                exp_trend=exp_trend,
            )
            num_coefficients = len(coefficient_names)
            if trend_prior_mean is None:
                default_mean = jnp.zeros((num_coefficients,), dtype=jnp.float64)
                default_mean = default_mean.at[0].set(1.0)
                trend_prior_mean = jnp.broadcast_to(
                    default_mean, (num_lcs, num_coefficients)
                )
            else:
                trend_prior_mean = jnp.asarray(
                    trend_prior_mean, dtype=jnp.float64
                )
            if trend_prior_scale is None:
                trend_prior_scale = jnp.full(
                    (num_lcs, num_coefficients), 0.1, dtype=jnp.float64
                )
            else:
                trend_prior_scale = jnp.asarray(
                    trend_prior_scale, dtype=jnp.float64
                )
            log_likelihood, conditional = (
                marginalized_log_likelihood_and_conditional(
                    jnp.asarray(y, dtype=jnp.float64) - transit_model,
                    design,
                    error_broadcast,
                    trend_prior_mean,
                    prior_scale=trend_prior_scale,
                )
            )
            numpyro.deterministic('trend_beta_mean', conditional.mean)
            numpyro.deterministic('trend_beta_factor', conditional.factor)
            numpyro.factor('obs_marginalized', jnp.sum(log_likelihood))
            return

        if detrend_type != 'none':
            if trend_mode == 'free':
                params['c'] = numpyro.sample('c', dist.Uniform(0.9, 1.1).expand([num_lcs]))
                params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes.update({'c': 0, 'v': 0})

                poly_order = 1
                if 'quartic' in detrend_type:
                    poly_order = 4
                elif 'cubic' in detrend_type:
                    poly_order = 3
                elif 'quadratic' in detrend_type:
                    poly_order = 2

                if poly_order >= 2:
                    params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v2'] = 0
                if poly_order >= 3:
                    params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v3'] = 0
                if poly_order >= 4:
                    params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v4'] = 0

                if 'explinear_spectroscopic' in detrend_components:
                    if exp_trend is None or fixed_tau is None:
                        raise ValueError("explinear_spectroscopic requires exp_trend and fixed_tau.")
                    params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    params['tau'] = numpyro.deterministic(
                        'tau', jnp.full((num_lcs,), jnp.asarray(fixed_tau, dtype=jnp.float64))
                    )
                    in_axes.update({'A': 0, 'tau': 0})
                elif 'explinear' in detrend_components:
                    params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
                    params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
                    in_axes.update({'A': 0, 'tau': 0})

            elif trend_mode == 'fixed':
                trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
                params['c'] = numpyro.deterministic('c', trend_temp[:, 0])
                params['v'] = numpyro.deterministic('v', trend_temp[:, 1])
                in_axes.update({'c': 0, 'v': 0})

                poly_order = 1
                if 'quartic' in detrend_type:
                    poly_order = 4
                elif 'cubic' in detrend_type:
                    poly_order = 3
                elif 'quadratic' in detrend_type:
                    poly_order = 2

                if poly_order >= 2:
                    params['v2'] = numpyro.deterministic('v2', trend_temp[:, 2])
                    in_axes['v2'] = 0
                if poly_order >= 3:
                    params['v3'] = numpyro.deterministic('v3', trend_temp[:, 3])
                    in_axes['v3'] = 0
                if poly_order >= 4:
                    params['v4'] = numpyro.deterministic('v4', trend_temp[:, 4])
                    in_axes['v4'] = 0
                if 'linear_discontinuity' in detrend_components:
                    params['t_jump'] = numpyro.deterministic('t_jump', trend_temp[:, 2])
                    params['jump'] = numpyro.deterministic('jump', trend_temp[:, 3])
                    in_axes.update({'t_jump': 0, 'jump': 0})
                elif 'explinear' in detrend_components:
                    params['A'] = numpyro.deterministic('A', trend_temp[:, 2])
                    params['tau'] = numpyro.deterministic('tau', trend_temp[:, 3])
                    in_axes.update({'A': 0, 'tau': 0})
                elif 'spot' in detrend_components and '2spot' not in detrend_components:
                    params['spot_amp'] = numpyro.deterministic('spot_amp', trend_temp[:, 2])
                    params['spot_mu'] = numpyro.deterministic('spot_mu', trend_temp[:, 3])
                    params['spot_sigma'] = numpyro.deterministic('spot_sigma', trend_temp[:, 4])
                    in_axes.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})
            else:
                raise ValueError(f"Unknown trend_mode: {trend_mode}")

        if 'explinear_spectroscopic' in detrend_components:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(
                params, t, exp_trend
            )
        elif 'gp_spectroscopic' in detrend_components:
            params['A_gp'] = numpyro.sample('A_gp', dist.Uniform(0.5, 2).expand([num_lcs]))
            in_axes['A_gp'] = 0

            if 'linear' in detrend_components and 'v' not in params:
                params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v'] = 0

            poly_order = 1
            if 'quartic' in detrend_components:
                poly_order = 4
            elif 'cubic' in detrend_components:
                poly_order = 3
            elif 'quadratic' in detrend_components:
                poly_order = 2

            if poly_order >= 2 and 'v2' not in params:
                params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v2'] = 0
            if poly_order >= 3 and 'v3' not in params:
                params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v3'] = 0
            if poly_order >= 4 and 'v4' not in params:
                params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                in_axes['v4'] = 0

            if 'explinear' in detrend_components and 'A' not in params:
                params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
                params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
                in_axes.update({'A': 0, 'tau': 0})

            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, gp_trend)

        elif '2spot_spectroscopic' in detrend_components:
            params['A_spot'] = numpyro.sample('A_spot', dist.Uniform(0.5, 2).expand([num_lcs]))
            params['A_spot2'] = numpyro.sample('A_spot2', dist.Uniform(0.5, 2).expand([num_lcs]))
            in_axes['A_spot'] = 0
            in_axes['A_spot2'] = 0
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, spot_trend2)
        elif 'spot_spectroscopic' in detrend_components:
            params['A_spot'] = numpyro.sample('A_spot', dist.Uniform(0.5, 2).expand([num_lcs]))
            in_axes['A_spot'] = 0
            if 'linear_discontinuity_spectroscopic' in detrend_components:
                params['A_jump'] = numpyro.sample('A_jump', dist.Uniform(0.5, 2).expand([num_lcs]))
                in_axes['A_jump'] = 0
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, jump_trend)
            else:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, spot_trend)
        elif 'linear_discontinuity_spectroscopic' in detrend_components:
            params['A_jump'] = numpyro.sample('A_jump', dist.Uniform(0.5, 2).expand([num_lcs]))
            in_axes['A_jump'] = 0
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, jump_trend)
        else:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None))(params, t)

        if likelihood_mask is None:
            numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)
        else:
            with numpyro.handlers.mask(mask=jnp.asarray(likelihood_mask, dtype=bool)):
                numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model
