"""Split compilation must retain the standalone Laplace MAP and target."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from models.independent_nuts import prepare_laplace_metric


def _model(t, yerr, y=None):
    q = numpyro.sample('q', dist.Beta(2., 3.))
    scale = numpyro.sample('scale', dist.LogNormal(0., .7))
    numpyro.sample('obs', dist.Normal(q + (.2 + .1*t)*scale, yerr), obs=y)


@pytest.mark.parametrize('method', ['exact', 'finite_difference'])
@pytest.mark.parametrize('sequential', [False, True])
@pytest.mark.parametrize('scaling', ['raw', 'correlation'])
def test_split_map_retains_fused_target_and_metric(method, sequential, scaling):
    t = jnp.linspace(-1., 1., 9)
    kwargs = dict(model_kwargs={'y': .73+.12*t}, hessian_method=method,
                  sequential_evaluations=sequential, metric_scaling=scaling,
                  gradient_tolerance=1e-6, decrement_tolerance=1e-6)
    preparations = [prepare_laplace_metric(
        _model, jax.random.PRNGKey(71), {'q': .4, 'scale': 1.2},
        t, jnp.full(t.shape, .2), map_execution=mode, **kwargs,
    ) for mode in ('fused', 'split')]
    fused, split = preparations
    for reference, candidate in zip(jax.tree.leaves(fused.unconstrained_map),
                                    jax.tree.leaves(split.unconstrained_map)):
        np.testing.assert_allclose(candidate, reference, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(split.inverse_mass_matrix, fused.inverse_mass_matrix,
                               rtol=2e-6, atol=1e-8)
    # Both returned potentials include the original constrained-support
    # Jacobians. Compiling differently must not replace that target.
    for point in (fused.unconstrained_map, split.unconstrained_map):
        np.testing.assert_allclose(split.potential_fn(point), fused.potential_fn(point),
                                   rtol=0, atol=1e-10)
    assert float(split.gradient_norm) < 1e-4
    assert float(split.newton_decrement) < 1e-4
    assert 0 < int(split.iterations) <= 200


def test_map_execution_rejects_unknown_mode():
    with pytest.raises(ValueError, match='map_execution'):
        prepare_laplace_metric(_model, jax.random.PRNGKey(1), {},
                               map_execution='unknown')


@pytest.mark.parametrize('method', ['exact', 'finite_difference'])
@pytest.mark.parametrize('surface', [False, True])
def test_white_caller_preserves_metric_and_target_with_staged_execution(monkeypatch, method, surface):
    """Exercise the production caller through real metric preparation."""
    from types import SimpleNamespace
    import models.independent_nuts as engine
    from koala.white_light import _sample_white_light

    t = jnp.linspace(-1., 1., 9)
    error = jnp.full(t.shape, .2)
    observed = .73 + .12*t
    initial = {'q': .4, 'scale': 1.2}
    key = jax.random.PRNGKey(71)
    def model(time, uncertainty, y=None, prior_params=None):
        return _model(time, uncertainty, y=y)

    class Prepared(Exception):
        def __init__(self, result, controls):
            self.result, self.controls = result, controls

    original = engine.prepare_laplace_metric
    def prepare_and_stop(*args, **kwargs):
        # Stop after the actual production boundary; MCMC is covered by the
        # independent posterior tests and the paired native acceptance fit.
        raise Prepared(original(*args, **kwargs), kwargs)
    monkeypatch.setattr(engine, 'prepare_laplace_metric', prepare_and_stop)
    context = SimpleNamespace(
        cfg={'flags': {}}, transit_engine='harmonica',
        harmonica_wl_nuts_kwargs={}, key_master=key,
        whitelight_laplace_options=dict(mass_matrix='laplace', hessian_method=method,
                                        trust_radius=5.),
        data=SimpleNamespace(wl_time=t, wl_flux_err=error, wl_flux=observed),
        uses_surface_model=surface, ld_profile='quadratic',
    )
    with pytest.raises(Prepared) as caught:
        _sample_white_light(context=context, hyper_params_wl={}, science_sites=('q',),
                            soln=initial, whitelight_mcmc_kwargs={},
                            whitelight_model_for_run=model)
    staged, controls = caught.value.result, caught.value.controls
    assert controls['map_execution'] == 'split'
    assert controls['eigenvalue_floor'] == 1e-12
    fused_controls = {**controls, 'map_execution': 'fused'}
    reference = original(model, key, initial, t, error, **fused_controls)
    np.testing.assert_allclose(staged.inverse_mass_matrix, reference.inverse_mass_matrix,
                               rtol=2e-6, atol=1e-8)
    for actual, expected in zip(jax.tree.leaves(staged.unconstrained_map),
                                jax.tree.leaves(reference.unconstrained_map)):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(staged.potential_fn(staged.unconstrained_map),
                               reference.potential_fn(staged.unconstrained_map),
                               atol=1e-10, rtol=0)
