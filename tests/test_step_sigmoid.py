import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers

import fit_jwst
from models.trends import _soft_step, sample_step_width


def test_sigmoid_matches_sharp_step_away_from_transition():
    t = jnp.asarray([-1.0e-4, -1.0e-6, 1.0e-6, 1.0e-4])
    actual = np.asarray(_soft_step(t, 0.0, 1.0e-10))
    np.testing.assert_allclose(actual, np.asarray([0.0, 0.0, 1.0, 1.0]))


def test_free_step_width_prior_and_sites():
    t = jnp.arange(12, dtype=jnp.float64) * 2.0e-4

    def model():
        sample_step_width(t, {"step_width_mode": "free"})

    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(3))).get_trace()
    assert trace["log_width"]["type"] == "sample"
    assert trace["width"]["type"] == "deterministic"
    assert trace["width_minutes"]["type"] == "deterministic"
    width = float(trace["width"]["value"])
    assert 0.5 * 2.0e-4 <= width <= 30.0 / (24.0 * 60.0)
    assert float(trace["width_minutes"]["value"]) == width * 1440.0


def test_free_width_prior_rejects_boundary_as_unconstrained_start():
    t = jnp.arange(12, dtype=jnp.float64) * 2.0e-4
    lower = np.log(0.5 * 2.0e-4)

    def model():
        sample_step_width(t, {"step_width_mode": "free"})

    seeded = handlers.seed(model, jax.random.PRNGKey(8))
    substituted = handlers.substitute(seeded, data={"log_width": lower})
    trace = handlers.trace(substituted).get_trace()
    assert np.isclose(float(trace["width"]["value"]), 0.5 * 2.0e-4)


class _FakeMCMC:
    def __init__(self):
        self.run_calls = 0

    def get_samples(self, group_by_chain=False):
        assert group_by_chain
        return {"depth": np.zeros((1, 20)), "b": np.zeros((1, 20))}

    def get_extra_fields(self, group_by_chain=False):
        assert group_by_chain
        return {"diverging": np.zeros((1, 20), dtype=bool)}

    def run(self, *args, **kwargs):
        self.run_calls += 1


def test_whitelight_gate_failfast_skips_extension_blocks(monkeypatch):
    monkeypatch.setattr(
        fit_jwst,
        "_geometry_chain_quality",
        lambda grouped: ({"depth": 12.0, "b": 30.0}, {}),
    )
    mcmc = _FakeMCMC()
    _, _, _, diagnostics = fit_jwst._continue_mcmc_until_geometry_gate(
        mcmc,
        jax.random.PRNGKey(5),
        (),
        {},
        min_ess=400,
        max_extra_blocks=3,
        failfast_ess=50,
    )
    assert mcmc.run_calls == 0
    assert diagnostics["whitelight_failfast_triggered"] is True
    assert diagnostics["whitelight_extra_blocks"] == 0
    assert diagnostics["whitelight_quality_gate_passed"] is False
