"""Gaussian trend marginalization: exactness against numerical integration and padded cadences."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from numpyro.infer.util import log_density

from models.jaxoplanet import builder, build_transit_window_indices
from models.harmonica.core import harmonica_a_rs_from_duration, harmonica_duration_from_geometry

jax.config.update("jax_enable_x64", True)

BOX = np.array([[0.9, 1.1], [-0.1, 0.1], [-0.1, 0.1]])


def _fixture(num_cadences=401, seed=3):
    rng = np.random.default_rng(seed)
    time = np.linspace(-.3, .3, num_cadences)
    # Define one shared physical orbit before varying either channel radius.
    # The time window covers contacts at the widest supported radius.
    axis = harmonica_a_rs_from_duration(2., .1, .3, .1)
    widest = harmonica_duration_from_geometry(2., axis, .3, np.sqrt(.5))
    window = build_transit_window_indices(time, [2.], [0.], np.atleast_1d(widest))
    error = np.array([.0003, .0006])[:, None] * np.where(np.arange(num_cadences) % 3 == 0, 1.0, 1.2)[None, :]
    trend = 1.0 + .002 * (time - time.min()) - .004 * (time - time.min()) ** 2
    y = trend[None, :] * (1.0 - .01 * (np.abs(time) < .05))[None, :] + error * rng.standard_normal(error.shape)
    common = dict(
        y=y, mu_a_rs=jnp.atleast_1d(axis), mu_duration=jnp.array([.1]), mu_t0=jnp.array([0.]), mu_b=jnp.array([.3]),
        PERIOD=jnp.array([2.]), ld_fixed=jnp.array([[.4, .2], [.3, .1]]),
        precomputed_yerr_per_lc=np.median(error, axis=1),
    )
    return time, window, y, error, common


def _model(trend_mode, window):
    return builder.create_vectorized_model(
        detrend_type='quadratic', ld_mode='fixed', ld_profile='quadratic', trend_mode=trend_mode,
        transit_window='auto', transit_window_indices=window, cadence_reduction='off',
        transit_grid='off', transit_grid_non_grazing=True,
    )


def test_gaussian_marginal_matches_numerical_integration():
    """Integrate the sampled model's likelihood times the Gaussian trend prior on a dense grid."""
    time, window, y, error, common = _fixture()
    rors = jnp.array([[.1], [.12]])
    log_jitter = jnp.log(jnp.array([.001, .002]))
    prior_mean = np.array([[1.0, 0.0, 0.0]] * 2)
    prior_scale = np.full((2, 3), 0.1)
    model = _model('gaussian_marginalized', window)
    kwargs = dict(common, trend_prior_mean=prior_mean, trend_prior_scale=prior_scale)
    marginal, trace = log_density(model, (time, error), kwargs, dict(rors=rors, log_jitter=log_jitter))
    mean = np.asarray(trace['trend_beta_mean']['value'])
    factor = np.asarray(trace['trend_beta_factor']['value'])
    sampled = _model('free', window)

    def joint(beta):
        # Sampled-trend log density with its flat box prior swapped for the Gaussian prior.
        value, _ = log_density(sampled, (time, error), common,
                               dict(rors=rors, log_jitter=log_jitter, c=beta[:, 0], v=beta[:, 1], v2=beta[:, 2]))
        flat = -jnp.sum(jnp.log(BOX[:, 1] - BOX[:, 0])) * 2
        return value - flat + jnp.sum(norm.logpdf(beta, prior_mean, prior_scale))

    nodes = np.linspace(-8.0, 8.0, 41)
    step = nodes[1] - nodes[0]
    grid = np.stack(np.meshgrid(nodes, nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 3)
    evaluate = jax.jit(jax.vmap(joint))
    total = -float(joint(jnp.asarray(mean)))
    for lane in range(2):
        betas = np.broadcast_to(mean, (grid.shape[0], 2, 3)).copy()
        betas[:, lane] = mean[lane] + grid @ factor[lane].T
        values = np.asarray(evaluate(jnp.asarray(betas)))
        peak = values.max()
        total += (peak + np.log(np.sum(np.exp(values - peak)) * step ** 3)
                  + np.log(abs(np.linalg.det(factor[lane]))))
    np.testing.assert_allclose(total, float(marginal), rtol=0, atol=1e-8)


def test_gaussian_marginal_ignores_masked_padding():
    time, window, y, error, common = _fixture()
    values = dict(rors=jnp.array([[.1], [.12]]), log_jitter=jnp.log(jnp.array([.001, .002])))
    model = _model('gaussian_marginalized', window)
    reference, reference_trace = log_density(model, (time, error), common, values)
    pad = 37
    padded_time = np.concatenate([time, time[-1] + (time[1] - time[0]) * np.arange(1, pad + 1)])
    padded = dict(common, y=np.pad(y, ((0, 0), (0, pad)), constant_values=1.0),
                  likelihood_mask=np.arange(padded_time.size) < time.size)
    padded_error = np.pad(error, ((0, 0), (0, pad)), constant_values=1.0)
    value, padded_trace = log_density(model, (padded_time, padded_error), padded, values)
    np.testing.assert_allclose(float(value), float(reference), rtol=0, atol=1e-8)
    # Padding must leave both the nonlinear target gradients and the conditional
    # coefficient distribution unchanged, not only its scalar normalization.
    reference_gradient = jax.grad(
        lambda parameters: log_density(model, (time, error), common, parameters)[0]
    )(values)
    padded_gradient = jax.grad(
        lambda parameters: log_density(model, (padded_time, padded_error), padded, parameters)[0]
    )(values)
    for name in values:
        np.testing.assert_allclose(padded_gradient[name], reference_gradient[name],
                                   atol=1e-7, rtol=1e-6)
    for name in ('trend_beta_mean', 'trend_beta_factor'):
        np.testing.assert_allclose(padded_trace[name]['value'], reference_trace[name]['value'],
                                   atol=1e-10, rtol=1e-9)
