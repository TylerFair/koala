"""Regression for XLA folding squared gathered errors into the wrong layout."""
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model

from models.cadence_reduction import build_linear_oot_statistics
from models.jaxoplanet import builder

jax.config.update("jax_enable_x64", True)


def test_jitted_multichannel_reduced_likelihood_preserves_error_lanes(monkeypatch):
    # Exercise the production branch with a small CI fixture. Unequal errors
    # in both axes are essential: identical lanes hide the layout defect.
    monkeypatch.setattr(builder, '_AUTO_CADENCE_REDUCTION_THRESHOLD', 20)
    time = np.linspace(-.3, .3, 65)
    window = np.flatnonzero(np.abs(time) < .051)
    y = 1.0 + np.array([.001, .002])[:, None] * np.cos(np.arange(65))[None, :]
    error = np.array([.003, .006])[:, None] * (1.0 + .1 * np.arange(65)[None, :] / 65)
    reference = np.array([[1.001], [.999]])
    common = dict(
        y=y, rors_reference=jnp.array([.1]), mu_duration=jnp.array([.1]), mu_t0=jnp.array([0.]),
        mu_b=jnp.array([.3]), PERIOD=jnp.array([2.]),
        ld_fixed=jnp.array([[.4, .2], [.3, .1]]),
        precomputed_yerr_per_lc=np.median(error, axis=1),
        **build_linear_oot_statistics(time, y, error, window, np.ones((65, 1)), reference),
    )
    initial = dict(rors=jnp.array([[.1], [.12]]), c=reference[:, 0],
                   log_jitter=jnp.log(jnp.array([.001, .002])))
    results = []
    for reduction in ['off', 'auto']:
        model = builder.create_vectorized_model(
            detrend_type='constant', ld_mode='fixed', ld_profile='quadratic',
            trend_mode='free', transit_window='auto', transit_window_indices=window,
            cadence_reduction=reduction, transit_grid='auto',
            transit_grid_non_grazing=True, transit_grid_nodes=65,
        )
        info = initialize_model(
            jax.random.PRNGKey(1), model, model_args=(time, error),
            model_kwargs=common, init_strategy=init_to_value(values=initial),
            validate_grad=False,
        )
        results.append(jax.jit(jax.value_and_grad(info.potential_fn))(info.param_info.z))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-12, atol=1e-10)
    for name in results[0][1]:
        np.testing.assert_allclose(results[1][1][name], results[0][1][name], rtol=1e-10, atol=1e-9)
