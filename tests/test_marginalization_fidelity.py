"""Broad geometry and contact gates for batching the physical transit laws."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from models.jaxoplanet.quadratic_safe import light_curve as quadratic_flux
from models.jaxoplanet.power2_exact import light_curve as power2_flux
from koala.marginalization.weights import hierarchical_weights, pseudo_bma_plus_weights

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("law", ["quadratic", "power2"])
def test_flux_forward_reverse_gradients_and_nested_vmap_at_contacts(law):
    # Separation is the relevant geometric coordinate of disk intersection;
    # sampling every contact regime also covers arbitrary impact/phase angles.
    rng = np.random.default_rng(371)
    radii = [.001, .01, .1, .499999, .5, .9, 1., 1.2, 2.]
    parameters = []
    for radius in radii:
        separations = [0., radius, abs(1-radius), 1+radius]
        separations += [max(0, contact+epsilon) for contact in (abs(1-radius), 1+radius)
                        for epsilon in (-1e-7, -1e-10, 1e-10, 1e-7)]
        separations += list(rng.uniform(0, 1+radius, 12))
        for separation in separations:
            parameters.append([separation, radius, .6, .4 if law == "quadratic" else .01])
    parameters = jnp.asarray(parameters)
    def evaluate(p):
        z, r, first, second = p
        if law == "quadratic":
            return quadratic_flux(jnp.array([first, second]), z, r)
        return power2_flux(first, second, z, r)
    reference = jax.jit(lambda p: (evaluate(p), jax.jacfwd(evaluate)(p)))
    expected = [reference(p) for p in parameters]
    values, gradients = jax.jit(jax.vmap(lambda p: (evaluate(p), jax.grad(evaluate)(p))))(parameters)
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(gradients))
    np.testing.assert_allclose(values, np.array([v for v, _ in expected]), atol=2e-12, rtol=0)
    # Use the physical gradient fidelity gate already applied to power-2
    # quadrature/grid routes. Near tangencies AD accumulation orders differ.
    np.testing.assert_allclose(gradients, np.array([g for _, g in expected]), atol=2e-7, rtol=2e-6)
    nested = jax.jit(jax.vmap(jax.vmap(jax.value_and_grad(evaluate))))(parameters.reshape(3, -1, 4))
    np.testing.assert_allclose(nested[0].reshape(-1), values, atol=2e-12, rtol=0)
    np.testing.assert_allclose(nested[1].reshape(-1, 4), gradients, atol=2e-7, rtol=2e-6)


def test_streaming_bootstrap_matches_explicit_dirichlet_construction():
    from scipy.special import softmax
    elpd = np.random.default_rng(74).normal(size=(3, 79))
    raw = np.random.default_rng(17).exponential(size=(1000, 79))
    expected = softmax(raw/raw.sum(axis=1, keepdims=True) @ elpd.T * 79, axis=1).mean(axis=0)
    actual = pseudo_bma_plus_weights(elpd, n_boot=1000, rng=17, point_batch_size=79)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    # Compare two equally accurate Monte Carlo estimates, not a 10x larger
    # run against a noisier 1000-bootstrap reference.
    raw_large = np.random.default_rng(91).exponential(size=(10000, 79))
    large_expected = softmax(raw_large/raw_large.sum(axis=1, keepdims=True) @ elpd.T * 79, axis=1).mean(axis=0)
    blocked = pseudo_bma_plus_weights(elpd, n_boot=10000, rng=17, point_batch_size=7)
    np.testing.assert_allclose(blocked, large_expected, atol=.015)


def test_bayesian_hierarchical_samples_and_diagnostics():
    scores = np.zeros((2, 2, 30))  # symmetric uninformative predictive models
    weights, samples, diagnostics = hierarchical_weights(scores, warmup=300, draws=400,
        chains=2, seed=84, min_ess=20, max_rhat=1.1)
    assert samples.shape == (800, 2, 2)
    np.testing.assert_allclose(samples.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(weights, .5, atol=.12)
    assert diagnostics["divergences"] == 0
    with pytest.raises(ValueError, match="convergence gates"):
        hierarchical_weights(scores, chains=1)
    with pytest.raises(ValueError, match='target_accept'):
        hierarchical_weights(scores, target_accept=1.)


def _quadratic_numerical_flux(z, r, u1, u2):
    from scipy.integrate import quad
    if z >= 1+r or r == 0:
        return 0.
    if r >= 1+z:
        return -1.
    total = np.pi*(1-u1/3-u2/6)
    def intensity(rho):
        mu = np.sqrt(max(0., 1-rho*rho))
        return 1-u1*(1-mu)-u2*(1-mu)**2
    inner_radius = min(1., max(0., r-z))
    inner = quad(lambda rho: 2*np.pi*rho*intensity(rho), 0., inner_radius,
                 epsabs=2e-13, epsrel=2e-12)[0]
    if z == 0:
        return -inner/total
    def annulus(rho):
        angle = np.arccos(np.clip((rho*rho+z*z-r*r)/(2*rho*z), -1., 1.))
        return 2*angle*rho*intensity(rho)
    lo, hi = abs(z-r), min(1., z+r)
    partial = quad(annulus, lo, hi, epsabs=2e-13, epsrel=2e-12, limit=300)[0]
    return -(inner+partial)/total


def test_quadratic_independent_integral_and_gradient_reference():
    rng = np.random.default_rng(286)
    rows = []
    for radius in (.001, .1, .5, .9, 1., 1.2, 2.):
        z = [0., abs(1-radius), 1+radius]
        z += list(rng.uniform(0, 1+radius, 8))
        if radius == 1.:
            z += [1e-10, 1e-8, 1e-6, 1e-5]
        rows += [[separation, radius, .3, .2] for separation in z]
    points = np.asarray(rows)
    evaluate = lambda p: quadratic_flux(p[2:], p[0], p[1])
    actual = jax.jit(jax.vmap(evaluate))(points)
    expected = [_quadratic_numerical_flux(*point) for point in points]
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)
    # Central differences away from non-differentiable topology boundaries;
    # exact contacts have finite limiting derivatives checked separately.
    interior = np.column_stack((rng.uniform(.2, 1., 12), rng.uniform(.02, .4, 12),
                                 rng.uniform(.1, .4, 12), rng.uniform(.1, .3, 12)))
    gradients = jax.jit(jax.vmap(jax.grad(evaluate)))(interior)
    numerical = np.empty_like(interior)
    step = 2e-5
    for index, point in enumerate(interior):
        for coordinate in range(4):
            delta = np.eye(4)[coordinate]*step
            numerical[index, coordinate] = (
                _quadratic_numerical_flux(*(point-2*delta))
                -8*_quadratic_numerical_flux(*(point-delta))
                +8*_quadratic_numerical_flux(*(point+delta))
                -_quadratic_numerical_flux(*(point+2*delta)))/(12*step)
    np.testing.assert_allclose(gradients, numerical, atol=2e-7, rtol=2e-6)


def test_quadratic_coefficient_prior_extrema_and_zero_radius():
    rows = []
    for plus, minus in ((-1., -2.), (-1., 2.), (2., -2.), (2., 2.), (.5, .1)):
        first, second = (plus+minus)/2, (plus-minus)/2
        for radius in (0., .001, .1, .5, .9, 1.2, 2.):
            for separation in (0., radius, abs(1-radius), .31*(1+radius), .81*(1+radius), 1+radius):
                rows.append([separation, radius, first, second])
    points = np.asarray(rows)
    evaluate = lambda p:quadratic_flux(p[2:], p[0], p[1])
    values, gradient = jax.jit(jax.vmap(jax.value_and_grad(evaluate)))(points)
    expected = [_quadratic_numerical_flux(*p) for p in points]
    np.testing.assert_allclose(values, expected, atol=2e-9, rtol=0)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize('law', ['quadratic', 'power2'])
def test_full_orbital_gradients_at_central_and_grazing_contacts(law):
    from models.jaxoplanet.core import compute_transit_model
    t = jnp.array([-.1, -.05, -.05+1e-10, 0., .05-1e-10, .05, .1])
    rows = []
    for radius in (.001, .1, .5, 1., 1.2):
        for impact in (0., 1e-10, .23, .71, abs(1-radius), 1+radius-1e-10, 1+radius, 1+radius+1e-10):
            rows.append([radius, impact, .1, 0.])
    def curve(p):
        r, b, duration, t0 = p
        params = dict(period=jnp.array([3.]), duration=jnp.array([duration]),
            t0=jnp.array([t0]), b=jnp.array([b]), rors=jnp.array([r]), _ld_profile=law)
        if law == 'quadratic':
            params.update(u=jnp.array([.3, .2]), _jaxoplanet_kernel='quadratic_contact_safe')
        else:
            params.update(u=jnp.zeros(12), c1=.6, c2=.1, _power2_exact=True)
        return compute_transit_model(params, t)
    scalar = jax.jit(lambda p: (curve(p), jax.jacfwd(curve)(p)))
    expected = [scalar(jnp.asarray(p)) for p in rows]
    flux, gradient = jax.jit(jax.vmap(lambda p: (curve(p), jax.jacrev(curve)(p))))(jnp.asarray(rows))
    assert np.all(np.isfinite(flux))
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(flux, np.stack([v for v, _ in expected]), atol=2e-12, rtol=0)
    np.testing.assert_allclose(gradient, np.stack([g for _, g in expected]), atol=2e-7, rtol=2e-6)
