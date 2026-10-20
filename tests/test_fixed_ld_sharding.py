"""Fixed-LD Laplace line searches must compile with multiple CPU devices."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _worker():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from models.independent_nuts import build_independent_nuts_runner
    from models.jaxoplanet.builder import create_vectorized_model

    assert len(jax.devices('cpu')) == 2
    t = jnp.linspace(-.09, .09, 15)
    observed = jnp.ones((2, 15)) - .02 * jnp.exp(-(t / .04)**8)
    errors = jnp.full_like(observed, .001)
    coefficients = jnp.tile(jnp.array([.3, .2]), (2, 1))
    kwargs = dict(
        mu_a_rs=jnp.array([11.]), mu_duration=jnp.array([.12]),
        mu_t0=jnp.array([0.]), mu_b=jnp.array([.4]), PERIOD=jnp.array([4.]),
        ld_fixed=coefficients, likelihood_mask=jnp.ones(15, dtype=bool),
        transit_separation=jnp.array([
            [1.4, 1.2, 1.05, .9, .7, .5, .4, .5, .7, .9, 1.05, 1.2, 1.4, 1.5, 1.6]
        ]),
    )
    model = create_vectorized_model(
        ld_mode='fixed', trend_basis='batman_normalized',
        trend_bounds={'c': (-10., 10.), 'v': (-10., 10.)},
        transit_window='auto', transit_window_indices=jnp.arange(1, 12),
    )
    initial = dict(rors=jnp.full((2, 1), .145), log_jitter=jnp.full(2, -8.),
                   trend_c=jnp.ones(2), trend_v=jnp.zeros(2))
    results = []
    for devices in (2, 1):
        runner = build_independent_nuts_runner(
            model, lane_width=2, lane_devices=devices,
            channel_varying_kwargs=('ld_fixed',), mass_matrix='laplace',
            num_samples=8, laplace_warmup=8, laplace_max_tree_depth=3,
            laplace_start_at_map=True,
            nuts_kwargs=dict(laplace_fuse_program=False,
                             laplace_hessian_method='finite_difference',
                             laplace_map_iterations=3),
            mcmc_kwargs=dict(num_chains=4, progress_bar=False),
        )
        samples, diagnostics = runner.run_raw(
            jax.random.PRNGKey(71), t, errors, observed, initial, kwargs,
        )
        assert (runner._mesh is not None) == (devices == 2)
        assert samples['rors'].shape == (32, 2, 1)
        assert 'u' not in runner._continuation[0].z
        for value in samples.values():
            assert np.isfinite(value).all()
        np.testing.assert_array_equal(samples['u'],
                                      np.broadcast_to(coefficients, (32, 2, 2)))
        results.append((samples, diagnostics))

    # The same keys and starts must give the same posterior in each lane,
    # including the deterministic outputs and Laplace/adaptation diagnostics.
    for sharded, single in zip(*results):
        assert sharded.keys() == single.keys()
        for name in sharded:
            np.testing.assert_allclose(sharded[name], single[name],
                                       rtol=1e-9, atol=1e-10, equal_nan=True,
                                       err_msg=name)


def test_fixed_ld_sharded_laplace_matches_single_device():
    # Host-device count is fixed at backend startup. Isolate the compile so
    # the original XLA SIGFPE becomes a test failure instead of killing pytest.
    environment = dict(os.environ, JAX_PLATFORMS='cpu', JAX_ENABLE_X64='1',
                       XLA_FLAGS='--xla_force_host_platform_device_count=2')
    environment['PYTHONPATH'] = os.pathsep.join(
        filter(None, (str(ROOT), environment.get('PYTHONPATH'))))
    subprocess.run([sys.executable, '-X', 'faulthandler', str(Path(__file__).resolve())],
                   cwd=ROOT, env=environment, check=True, timeout=600)


if __name__ == '__main__':
    _worker()
