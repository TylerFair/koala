"""Abstract priming preserves actual independent chains and continuation."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from models.independent_nuts import build_independent_nuts_runner, get_samples_independent
from koala.marginalization.priming import prime_laplace_runner


def model(t, yerr, y=None):
    value = numpyro.sample('value', dist.Normal(0., 1.).expand((len(yerr),)))
    numpyro.sample('obs', dist.Normal(value[:, None]+t, yerr), obs=y)


def test_abstract_priming_preserves_samples_metrics_and_reuse():
    nuts = dict(mass_matrix='laplace', laplace_warmup=20, laplace_max_tree_depth=4,
                laplace_start_at_map=True, laplace_map_iterations=10,
                laplace_hessian_method='finite_difference')
    counts = dict(num_chains=2, num_samples=30, num_warmup=20, progress_bar=False)
    def runner():
        return build_independent_nuts_runner(model, nuts_kwargs=nuts, mcmc_kwargs=counts, lane_width=2)
    ordinary, primed = runner(), prime_laplace_runner(runner())
    t = jnp.linspace(0., 1., 9)
    errors = jnp.full((2, len(t)), .3)
    y = jnp.tile(t, (2, 1))+.4
    arguments = (model, jax.random.PRNGKey(871), t, errors, y, {'value':jnp.zeros(2)})
    first, diagnostic = get_samples_independent(*arguments, _runner=ordinary,
        nuts_kwargs=nuts, mcmc_kwargs=counts, return_diagnostics=True)
    second, second_diagnostic = get_samples_independent(*arguments, _runner=primed,
        nuts_kwargs=nuts, mcmc_kwargs=counts, return_diagnostics=True)
    np.testing.assert_allclose(first['value'], second['value'], atol=2e-12, rtol=0)
    np.testing.assert_allclose(diagnostic.step_size_by_chain, second_diagnostic.step_size_by_chain, atol=2e-12, rtol=0)
    assert primed.program_build_count == ordinary.program_build_count == 1
    assert len(primed._grid_priming_timings) == 1
    assert primed._grid_priming_timings[0]['disposable_sampling_iterations'] == 0
    # Dynamic new observations reuse the programs, with new MAP/adaptation.
    get_samples_independent(model, jax.random.PRNGKey(872), t, errors, y-.7,
        {'value':jnp.zeros(2)}, _runner=primed, nuts_kwargs=nuts, mcmc_kwargs=counts)
    assert len(primed._grid_priming_timings) == 1 and primed.program_build_count == 1
    more, raw = primed.continue_raw()
    assert more['value'].shape == first['value'].shape
    assert raw['num_chains'] == 2
