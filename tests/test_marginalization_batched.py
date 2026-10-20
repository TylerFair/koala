"""Scientific target parity of augmented, independently batched variants."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
import pytest

from koala.config import ParameterSpec
from koala.marginalization.batched import create_batched_model
from models.jaxoplanet.builder import create_vectorized_model, create_whitelight_model

jax.config.update("jax_enable_x64", True)


def priors():
    fixed = lambda name, value: (ParameterSpec(name, "fixed", value),)
    return dict(period=fixed("period", 3.), t0=fixed("t0", 0.),
        duration=fixed("duration", .1), b=fixed("b", .71),
        rprs=(ParameterSpec("rprs", "uniform", .1, low=.01, high=.5),),
        ecc=fixed("ecc", 0.), omega=fixed("omega", 0.))

def trace(model, values, t, error, y, kwargs):
    return handlers.trace(handlers.seed(handlers.substitute(model, data=values), 0))\
        .get_trace(t, error, y=y, **kwargs)

def logp(tr):
    return sum(jnp.sum(site["fn"].log_prob(site["value"])) for site in tr.values()
               if site.get("type") == "sample")


@pytest.mark.parametrize("stage", ["white", "highres"])
@pytest.mark.parametrize("law,basis,parameterization", [("quadratic", "coefficients", 'coefficients'),
                                       ("quadratic", "uplus_uminus", 'coefficients'),
                                       ("power2", "coefficients", 'coefficients'),
                                       ("power2", "coefficients", 'maxted_smooth')])
def test_all_four_candidate_targets_flux_density_and_gradients(stage, law, basis, parameterization):
    t = jnp.linspace(-.08, .08, 47)
    errors = jnp.full((1, len(t)), .001)
    y = jnp.ones_like(errors)
    fixed_ld = jnp.array([[.4, .2]])
    kwargs = dict(ld_fixed=fixed_ld, PERIOD=jnp.array([[3.]]),
        rors_reference=jnp.array([.1]), mu_duration=jnp.array([[.1]]), mu_t0=jnp.array([[0.]]), mu_b=jnp.array([[.71]]))
    augmented = create_batched_model(law=law, stage=stage,
        parameter_priors=priors() if stage == "white" else None, ld_uniform_basis=basis,
        power2_parameterization=parameterization, power2_quadrature_order=64)
    coordinates = jnp.array([.5, .1]) if basis == "uplus_uminus" else jnp.array([.3, .2])
    coefficients = (jnp.array([(coordinates[0]+coordinates[1])/2,
                               (coordinates[0]-coordinates[1])/2])
                    if basis == "uplus_uminus" else coordinates)
    for quadratic in (False, True):
        for free in (False, True):
            trend = "quadratic" if quadratic else "linear"
            options = dict(ld_profile=law, ld_mode="uniform" if free else "fixed",
                detrend_type=trend, ld_uniform_basis=basis,
                jaxoplanet_kernel="quadratic_contact_safe" if law == "quadratic" else "auto")
            native = (create_whitelight_model(parameter_priors=priors(), **options)
                      if stage == "white" else create_vectorized_model(**options))
            def densities(p):
                radius, jitter, c, v, v2, a, b = p
                coord = jnp.array([a, b])
                if parameterization == 'maxted_smooth':
                    from models.ld_parameterization import Power2MaxtedUniformTransform
                    transform = Power2MaxtedUniformTransform()
                    latent = transform.inv(coord)
                else:
                    latent = coord
                ld = (jnp.array([(a+b)/2, (a-b)/2]) if basis == "uplus_uminus" else coord)
                aug_values = dict(ld_coordinates=latent[None, :],
                    trend_coordinates=jnp.array([[c, v, v2]]),
                    log_jitter=jnp.array([jitter]))
                aug_values["rors_0" if stage == "white" else "rors"] = (
                    jnp.array([radius]) if stage == "white" else jnp.array([[radius]]))
                aug_trace = trace(augmented, aug_values, t, errors, y,
                    dict(kwargs, quadratic_trend=jnp.array([quadratic]), free_ld=jnp.array([free])))
                native_values = dict(c=c, v=v, log_jitter=jitter)
                if quadratic:
                    native_values["v2"] = v2
                if stage == "white":
                    native_values["rors_0"] = radius
                    native_kwargs = dict(prior_params=dict(u=fixed_ld[0]))
                    native_error, native_y = errors[0], y[0]
                    if free:
                        if law == "power2":
                            native_values.update(c1=a, c2=b)
                        elif basis == "uplus_uminus":
                            native_values["ld_uplus_uminus"] = coord
                        else:
                            native_values["u"] = coord
                else:
                    native_values = {k: jnp.atleast_1d(vv) for k, vv in native_values.items()}
                    native_values["rors"] = jnp.array([[radius]])
                    native_kwargs = {k: val if k == 'ld_fixed' else val[0]
                                     for k, val in kwargs.items()}
                    native_error, native_y = errors, y
                    if free:
                        name = ("ld_coefficients" if law == "power2" else
                                "ld_uplus_uminus" if basis == "uplus_uminus" else "u")
                        native_values[name] = coord[None, :]
                native_trace = trace(native, native_values, t, native_error, native_y, native_kwargs)
                ghost = (0. if quadratic else dist.Uniform(-.1, .1).log_prob(v2))
                if free and parameterization == 'maxted_smooth':
                    ghost += transform.log_abs_det_jacobian(latent, coord)
                if not free:
                    lo = jnp.array([-1., -2.]) if basis == "uplus_uminus" else jnp.zeros(2)
                    hi = (jnp.array([2., 2.]) if basis == "uplus_uminus" else
                          jnp.array([1.5, 40.]) if law == "power2" else jnp.ones(2))
                    ghost += (jnp.sum(dist.Normal(0., 1.).log_prob(latent))
                              if parameterization == 'maxted_smooth' else
                              jnp.sum(dist.Uniform(lo, hi).log_prob(coord)))
                return logp(aug_trace)-ghost, logp(native_trace), aug_trace["obs"]["fn"].loc, native_trace["obs"]["fn"].loc
            point = jnp.array([.1, jnp.log(.0001), 1., .003, .002,
                               coordinates[0], coordinates[1]])
            augmented_logp, native_logp, aug_flux, native_flux = densities(point)
            np.testing.assert_allclose(aug_flux.reshape(-1), native_flux.reshape(-1), atol=2e-12, rtol=0)
            np.testing.assert_allclose(augmented_logp, native_logp, atol=2e-9, rtol=0)
            gradient = jax.jit(jax.jacrev(lambda p: jnp.array(densities(p)[:2])))(point)
            np.testing.assert_allclose(gradient[0], gradient[1], atol=2e-7, rtol=2e-6)

def test_variant_geometry_is_lane_specific_and_not_shared():
    model = create_batched_model(law="quadratic")
    t = jnp.linspace(-.06, .06, 31)
    values = dict(ld_coordinates=jnp.array([[.5, .1], [.5, .1]]),
        trend_coordinates=jnp.array([[1., 0., 0.], [1., 0., 0.]]),
        rors=jnp.array([[.1], [.1]]), log_jitter=jnp.full(2, -9.))
    tr = trace(model, values, t, jnp.full((2, len(t)), .001), jnp.ones((2, len(t))),
        dict(quadratic_trend=jnp.array([False, True]), free_ld=jnp.array([False, False]),
             ld_fixed=jnp.array([[.3, .2], [.3, .2]]), PERIOD=jnp.full((2, 1), 3.),
             rors_reference=jnp.array([.1]), mu_duration=jnp.full((2, 1), .1), mu_t0=jnp.array([[0.], [.01]]),
             mu_b=jnp.array([[.2], [1.03]])))
    flux = np.asarray(tr["obs"]["fn"].loc)
    assert flux.shape == (2, len(t))
    assert not np.allclose(flux[0], flux[1])


@pytest.mark.parametrize('law', ['quadratic', 'power2'])
def test_full_target_geometry_gradients_across_radius_and_contact_regimes(law):
    """Check the likelihood and priors as well as the physical flux kernel."""
    t = jnp.asarray([-.08, -.0500000001, -.05, -.0499999999, -.025,
                     0., .025, .0499999999, .05, .0500000001, .08])
    errors = jnp.full((1, len(t)), .001)
    y = jnp.ones_like(errors)
    ld = jnp.array([[.4, .2]])
    augmented = create_batched_model(law=law, power2_quadrature_order=64)
    native = create_vectorized_model(ld_profile=law, ld_mode='fixed',
        detrend_type='quadratic', jaxoplanet_kernel=(
            'quadratic_contact_safe' if law == 'quadratic' else 'auto'))

    def targets(point):
        radius, impact, duration, epoch = point
        geometry = dict(PERIOD=jnp.array([[3.]]), mu_b=impact.reshape(1, 1),
                        rors_reference=jnp.array([.1]), mu_duration=duration.reshape(1, 1), mu_t0=epoch.reshape(1, 1))
        aug_values = dict(rors=radius.reshape(1, 1), log_jitter=jnp.array([-9.]),
            ld_coordinates=jnp.array([[.5, .1]]),
            trend_coordinates=jnp.array([[1., .003, .002]]))
        native_values = dict(rors=radius.reshape(1, 1), log_jitter=jnp.array([-9.]),
                             c=jnp.array([1.]), v=jnp.array([.003]), v2=jnp.array([.002]))
        aug = trace(augmented, aug_values, t, errors, y,
            dict(geometry, ld_fixed=ld, quadratic_trend=jnp.array([True]), free_ld=jnp.array([False])))
        original = trace(native, native_values, t, errors, y,
            dict(ld_fixed=ld, **{key:value[0] for key,value in geometry.items()}))
        lower = jnp.array([-1., -2.]) if law == 'quadratic' else jnp.zeros(2)
        upper = jnp.array([2., 2.]) if law == 'quadratic' else jnp.array([1.5, 40.])
        ghost = dist.Uniform(lower, upper).log_prob(aug_values['ld_coordinates']).sum()
        return jnp.stack((logp(aug)-ghost, logp(original))), jnp.stack(
            (aug['obs']['fn'].loc, original['obs']['fn'].loc))

    points = []
    for radius in (.0032, .11, .707):
        impacts = (0., .17, .49, .93, 1-radius, 1-radius-1e-10,
                   1-radius+1e-10, 1+radius, 1+radius-1e-10, 1+radius+1e-10, 1.8)
        points.extend([radius, impact, .1, 0.] for impact in impacts)
    points = jnp.asarray(points)
    density, flux = jax.jit(jax.vmap(targets))(points)
    gradients = jax.jit(jax.vmap(jax.jacrev(lambda p: targets(p)[0])))(points)
    assert np.isfinite(density).all()
    assert np.isfinite(flux).all()
    assert np.isfinite(gradients).all()
    np.testing.assert_allclose(flux[:, 0], flux[:, 1], atol=2e-12, rtol=0)
    np.testing.assert_allclose(density[:, 0], density[:, 1], atol=2e-8, rtol=0)
    np.testing.assert_allclose(gradients[:, 0], gradients[:, 1], atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('law', ['quadratic', 'power2'])
def test_conservative_window_preserves_full_likelihood_and_gradients(law):
    from koala.marginalization.batched_fitting import _transit_window
    t = jnp.linspace(-.13, .13, 81)
    medians = dict(period_0=np.array([3., 3.]), t0_0=np.array([0., .013]),
                   duration_0=np.array([.1, .11]),
                   b_0=np.array([0., .91]), rors_0=np.array([.1, .1]))
    window = _transit_window(t, medians)
    assert 0 < len(window) < len(t)
    kwargs = dict(quadratic_trend=jnp.array([False, True]), free_ld=jnp.array([True, False]),
        ld_fixed=jnp.array([[.4, .2], [.4, .2]]), PERIOD=jnp.full((2, 1), 3.),
        rors_reference=jnp.array([.1]), mu_duration=jnp.array([[.1], [.11]]), mu_t0=jnp.array([[0.], [.013]]),
        mu_b=jnp.array([[0.], [.91]]))
    full = create_batched_model(law=law)
    bounded = create_batched_model(law=law, transit_window_indices=window)
    errors, y = jnp.full((2, len(t)), .001), jnp.ones((2, len(t)))
    def density(radius):
        values = dict(ld_coordinates=jnp.array([[.5, .1], [.5, .1]]),
            trend_coordinates=jnp.array([[1., .003, 0.], [.99, -.002, .004]]),
            rors=radius[:, None], log_jitter=jnp.full(2, -9.))
        return jnp.array([logp(trace(model, values, t, errors, y, kwargs))
                          for model in (full, bounded)])
    for radius in (jnp.array([.01, .03]), jnp.array([.1, .4]), jnp.array([.7, .707])):
        scores = jax.jit(density)(radius)
        np.testing.assert_allclose(scores[0], scores[1], atol=2e-8, rtol=0)
        gradient = jax.jit(jax.jacrev(density))(radius)
        np.testing.assert_allclose(gradient[0], gradient[1], atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('law', ['quadratic', 'power2'])
def test_white_prior_envelope_preserves_geometry_and_contact_gradients(law):
    from koala.marginalization.batched_fitting import _white_transit_window
    specs = priors()
    specs['t0'] = (ParameterSpec('t0', 'uniform', 0., low=-.013, high=.017),)
    specs['duration'] = (ParameterSpec('duration', 'uniform', .1, low=.04, high=.11),)
    specs['b'] = (ParameterSpec('b', 'uniform', .71, low=0., high=1.5),)
    # Three separate orbital events, exact envelope endpoints, and OOT data.
    t = jnp.asarray(np.unique(np.concatenate((np.linspace(-3.2, 3.2, 151),
                   np.array([-.068, .072, 3.-.068, -3.+.072])))))
    window = _white_transit_window(t, specs)
    assert 0 < len(window) < len(t)
    full = create_batched_model(law=law, stage='white', parameter_priors=specs)
    bounded = create_batched_model(law=law, stage='white', parameter_priors=specs,
                                   transit_window_indices=window)
    errors, y = jnp.full((1, len(t)), .001), jnp.ones((1, len(t)))
    kwargs = dict(quadratic_trend=jnp.array([True]), free_ld=jnp.array([True]),
                  ld_fixed=jnp.array([[.4, .2]]))
    def density(p):
        radius, impact, duration, epoch = p
        values = dict(ld_coordinates=jnp.array([[.5, .1]]),
            trend_coordinates=jnp.array([[1., .003, .002]]),
            rors_0=jnp.array([radius]), b_0=jnp.array([impact]),
            duration_0=jnp.array([duration]), t0_0=jnp.array([epoch]),
            log_jitter=jnp.array([-9.]))
        return jnp.stack([logp(trace(model, values, t, errors, y, kwargs))
                          for model in (full, bounded)])
    evaluate, differentiate = jax.jit(density), jax.jit(jax.jacrev(density))
    for epoch in (-.013, .017, 0.):
        for impact in (0., .71, 1.1, 1.3):
            point = jnp.array([.1, impact, .11, epoch])
            scores = evaluate(point)
            np.testing.assert_allclose(scores[0], scores[1], atol=2e-8, rtol=0)
            gradients = differentiate(point)
            np.testing.assert_allclose(gradients[0], gradients[1], atol=2e-7, rtol=2e-6)

def test_white_prior_envelope_unbounded_and_variable_period_fallbacks():
    from koala.marginalization.batched_fitting import _white_transit_window
    specs = priors()
    t = np.linspace(-.2, .2, 101)
    specs['t0'] = (ParameterSpec('t0', 'gaussian', 0., sigma=.02),)
    assert _white_transit_window(t, specs) is None
    specs = priors()
    specs['period'] = (ParameterSpec('period', 'uniform', 3., low=2., high=4.),)
    assert _white_transit_window(t, specs) is None
