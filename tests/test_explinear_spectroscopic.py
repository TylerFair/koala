import jax
import jax.numpy as jnp
from numpyro import handlers

from fit_jwst import _spectro_detrend_type
from models.jaxoplanet.builder import create_vectorized_model
from models.trends import compute_lc_explinear, compute_lc_explinear_spectroscopic


jax.config.update("jax_enable_x64", True)


def _model_kwargs(t, tau):
    return dict(
        t=t,
        yerr=jnp.full((2, t.size), 1.0e-4),
        y=jnp.ones((2, t.size)),
        mu_duration=jnp.array([0.08]),
        mu_t0=jnp.array([0.05]),
        mu_b=jnp.array([0.3]),
        mu_depths=jnp.full((2, 1), 0.01),
        PERIOD=jnp.array([3.0]),
        ld_fixed=jnp.tile(jnp.array([[0.4, 0.2]]), (2, 1)),
        exp_trend=jnp.exp(-(t - jnp.min(t)) / tau),
        fixed_tau=tau,
    )


def test_fixed_timescale_model_trace_has_constant_tau_and_no_log_tau():
    tau = 0.023
    model = create_vectorized_model(
        detrend_type="explinear_spectroscopic", ld_mode="fixed"
    )
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(2))).get_trace(
        **_model_kwargs(jnp.linspace(0.0, 0.12, 31), tau)
    )
    assert "log_tau" not in trace
    assert trace["tau"]["value"].shape == (2,)
    assert jnp.all(trace["tau"]["value"] == tau)
    assert trace["A"]["fn"].support.lower_bound == -0.1
    assert trace["A"]["fn"].support.upper_bound == 0.1


def test_fixed_template_equals_free_tau_kernel_at_same_tau():
    t = jnp.linspace(1.0, 1.15, 47)
    tau = 0.017
    params = dict(
        period=jnp.array([3.0]), duration=jnp.array([0.08]),
        t0=jnp.array([1.07]), b=jnp.array([0.2]), rors=jnp.array([0.1]),
        u=jnp.array([0.4, 0.2]), c=1.001, v=-0.003, A=0.008, tau=tau,
    )
    exp_trend = jnp.exp(-(t - jnp.min(t)) / tau)
    assert jnp.allclose(
        compute_lc_explinear(params, t),
        compute_lc_explinear_spectroscopic(params, t, exp_trend),
        rtol=0.0, atol=1.0e-14,
    )


def test_pipeline_mapping_is_opt_in():
    assert _spectro_detrend_type("explinear") == "explinear"
    assert _spectro_detrend_type("explinear", True) == "explinear_spectroscopic"
    assert _spectro_detrend_type("linear", True) == "linear"
