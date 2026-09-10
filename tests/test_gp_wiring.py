import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.handlers import seed, substitute, trace
import pytest

jax.config.update("jax_enable_x64", True)

import models.gp as kgp
from models.harmonica.builder import create_whitelight_model as harmonica_factory
from models.jaxoplanet.builder import create_whitelight_model as jaxoplanet_factory
from planet_specs import default_planet_specs


SPECS = default_planet_specs(period=3.0, t0=1.0, b=0.2, rprs=0.1, duration=0.12)


def _jaxoplanet(**kwargs):
    kwargs.setdefault("parameter_priors", SPECS)
    return jaxoplanet_factory(**kwargs)


_jaxoplanet.__name__ = "jaxoplanet_factory"
FACTORIES = (_jaxoplanet, harmonica_factory)
GP_TYPES = tuple(kgp.GP_BUILDERS)


def _prior():
    return {"period": jnp.array([3.0]), "u": jnp.array([0.2, 0.1]),
            "parameter_priors": SPECS}


def _substitutions(detrend_type):
    values = {
        "t0_0": jnp.array(1.0),
        "rors_0": jnp.array(0.1),
        "b_0": jnp.array(0.2),
        "logD_0": jnp.log(jnp.array(0.12)),
        "log_jitter": jnp.log(jnp.array(1e-3)),
        "c": jnp.array(0.96),
        "v": jnp.array(0.08),
        "v2": jnp.array(-0.02),
        "v3": jnp.array(0.006),
        "v4": jnp.array(-0.002),
        "A": jnp.array(0.03),
        "log_tau": jnp.log(jnp.array(0.25)),
        "GP_log_sigma": jnp.log(jnp.array(0.01)),
        "GP_log_rho": jnp.log(jnp.array(0.08)),
    }
    return values


@pytest.mark.parametrize("factory", FACTORIES)
def test_builder_forwards_resolved_solver_and_sortedness(monkeypatch, factory):
    calls = []

    class FakeGP:
        def numpyro_dist(self):
            return dist.Normal(jnp.zeros(8), jnp.ones(8)).to_event(1)

    def fake_builder(params, t, error, *, gp_solver, assume_sorted, **kwargs):
        calls.append((gp_solver, assume_sorted, np.asarray(t).shape))
        return FakeGP()

    monkeypatch.setitem(kgp.GP_BUILDERS, "linear+gp", fake_builder)
    model = factory(
        detrend_type="linear+gp",
        ld_mode="fixed",
        ld_profile="quadratic",
        gp_solver="serial",
        gp_assume_sorted=True,
    )
    t = jnp.linspace(0.9, 1.1, 8)
    seeded = seed(substitute(model, data=_substitutions("linear+gp")), 4)
    trace(seeded).get_trace(t, jnp.full_like(t, 1e-3), y=jnp.ones_like(t),
                            prior_params=_prior())
    assert calls == [("serial", True, (8,))]


def test_solver_environment_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv(kgp.GP_SOLVER_ENV, "parallel")
    monkeypatch.setattr(kgp, "gp_parallel_supported", lambda: True)
    assert kgp.resolve_gp_solver(None, device="cpu") == "parallel"
    assert kgp.resolve_gp_solver("serial", device="gpu") == "serial"


def test_factory_resolves_environment_once(monkeypatch):
    monkeypatch.setenv(kgp.GP_SOLVER_ENV, "serial")
    model = jaxoplanet_factory(
        detrend_type="linear+gp", ld_mode="fixed", ld_profile="quadratic",
        gp_assume_sorted=True, parameter_priors=SPECS,
    )
    monkeypatch.setenv(kgp.GP_SOLVER_ENV, "not-a-solver")
    t = jnp.linspace(0.9, 1.1, 8)
    seeded = seed(substitute(model, data=_substitutions("linear+gp")), 6)
    sites = trace(seeded).get_trace(
        t, jnp.full_like(t, 1e-3), y=jnp.ones_like(t), prior_params=_prior()
    )
    assert "obs" in sites


@pytest.mark.parametrize("factory", FACTORIES)
def test_legacy_solver_failure_happens_at_factory_time(monkeypatch, factory):
    monkeypatch.setattr(kgp, "gp_parallel_supported", lambda: False)
    factory(detrend_type="linear+gp", gp_solver="serial")
    with pytest.raises(kgp.GPSolverUnavailableError, match="Python >= 3.10"):
        factory(detrend_type="linear+gp", gp_solver="parallel")


@pytest.mark.parametrize("factory", FACTORIES)
def test_non_gp_trends_never_consult_the_solver_policy(monkeypatch, factory):
    """A plain trend must build even where parallel tinygp is unavailable."""
    monkeypatch.setattr(kgp, "gp_parallel_supported", lambda: False)
    monkeypatch.setenv(kgp.GP_SOLVER_ENV, "parallel")
    factory(detrend_type="linear")
    factory(detrend_type="linear", gp_solver="parallel")


def test_validate_times_rejects_unsorted_without_reordering():
    original = np.array([1.0, 0.9, 1.1])
    with pytest.raises(ValueError, match="nondecreasing"):
        kgp.validate_gp_times(original)
    np.testing.assert_array_equal(original, [1.0, 0.9, 1.1])


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("detrend_type", GP_TYPES)
def test_all_white_light_gp_variants_have_nonconstant_model_mean(factory,
                                                                  detrend_type):
    model = factory(
        detrend_type=detrend_type,
        ld_mode="fixed",
        ld_profile="quadratic",
        gp_solver="serial",
        gp_assume_sorted=True,
    )
    t = jnp.linspace(0.9, 1.25, 24)
    seeded = seed(substitute(model, data=_substitutions(detrend_type)), 5)
    sites = trace(seeded).get_trace(
        t, jnp.full_like(t, 1e-3), y=jnp.ones_like(t), prior_params=_prior()
    )
    mean = np.asarray(sites["obs"]["fn"].loc)
    assert mean.shape == (24,)
    assert np.ptp(mean) > 1e-5
