import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS
import pytest

jax.config.update("jax_enable_x64", True)

import models.gp as kgp
from models.independent_nuts import prepare_laplace_metric
from models.jaxoplanet.builder import create_whitelight_model


SOLVERS = ["serial"] + (["parallel"] if kgp.gp_parallel_supported() else [])


def _case(n=200):
    t = jnp.linspace(0.88, 1.18, n)
    yerr = jnp.full(n, 7.5e-4)
    prior = {"period": jnp.array([3.0]), "u": jnp.array([0.2, 0.1])}
    init = {
        "t0_0": jnp.array(1.0),
        "rors_0": jnp.array(0.095),
        "_b_0": jnp.array(0.25),
        "logD_0": jnp.log(jnp.array(0.12)),
        "log_jitter": jnp.log(jnp.array(4e-4)),
        "c": jnp.array(0.995),
        "v": jnp.array(0.012),
        "GP_log_sigma": jnp.log(jnp.array(7e-4)),
        "GP_log_rho": jnp.log(jnp.array(0.06)),
    }
    return t, yerr, prior, init


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("hessian_method", ["finite_difference", "exact"])
def test_gp_laplace_metric_and_nuts(solver, hessian_method):
    t, yerr, prior, init = _case()
    # Generate a deterministic, mildly correlated target from the same transit
    # and linear mean used by the real white-light factory.
    physical = {
        "period": prior["period"], "duration": jnp.array([0.12]),
        "t0": jnp.array([1.0]), "b": jnp.array([0.25]),
        "rors": jnp.array([0.095]), "u": prior["u"],
        "c": init["c"], "v": init["v"],
        "GP_log_sigma": init["GP_log_sigma"],
        "GP_log_rho": init["GP_log_rho"],
    }
    y = kgp.compute_lc_linear_gp_mean(physical, t, t_ref=jnp.min(t))
    y = y + 3e-4 * jnp.sin(jnp.linspace(0.0, 5.0, t.size))
    model = create_whitelight_model(
        detrend_type="linear+gp", ld_mode="fixed", ld_profile="quadratic",
        gp_solver=solver, gp_assume_sorted=True,
    )
    preparation = prepare_laplace_metric(
        model,
        jax.random.PRNGKey(81),
        init,
        t,
        yerr,
        model_kwargs={"y": y, "prior_params": prior},
        hessian_method=hessian_method,
        max_iterations=20,
        trust_radius=2.0,
        eigenvalue_floor=1e-10,
    )
    metric = np.asarray(preparation.inverse_mass_matrix)
    eigenvalues = np.linalg.eigvalsh(metric)
    assert np.all(np.isfinite(metric))
    assert np.all(eigenvalues > 0)

    kernel = NUTS(
        model,
        dense_mass=True,
        inverse_mass_matrix=preparation.inverse_mass_matrix,
        target_accept_prob=0.9,
        max_tree_depth=8,
    )
    mcmc = MCMC(
        kernel, num_warmup=200, num_samples=200, num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        jax.random.PRNGKey(82),
        t,
        yerr,
        y=y,
        prior_params=prior,
        init_params=preparation.unconstrained_map,
        extra_fields=("diverging",),
    )
    samples = mcmc.get_samples(group_by_chain=True)
    assert all(np.all(np.isfinite(np.asarray(value))) for value in samples.values())
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    assert divergences <= 4
    ess_values = [
        np.min(np.asarray(effective_sample_size(value)))
        for name, value in samples.items()
        if name in init and np.asarray(value).dtype.kind == "f"
    ]
    minimum_ess = float(np.min(ess_values))
    print(
        f"GP Laplace-NUTS solver={solver} method={hessian_method} "
        f"metric_eigs=[{eigenvalues[0]:.6e}, {eigenvalues[-1]:.6e}] "
        f"divergences={divergences}/200 min_ess={minimum_ess:.1f}"
    )
