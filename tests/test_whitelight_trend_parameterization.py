import jax
import jax.numpy as jnp
from numpyro.handlers import seed, trace

from models.harmonica.builder import (
    create_whitelight_model as create_harmonica_whitelight_model,
)
from models.jaxoplanet.builder import create_whitelight_model
from planet_specs import default_planet_specs

SPECS = default_planet_specs(period=3.0, t0=1.05, b=0.2, rprs=0.1, duration=0.12)


def _trace_default(model_factory, detrend_type):
    kwargs = dict(detrend_type=detrend_type, ld_mode="fixed", ld_profile="quadratic")
    kwargs["parameter_priors"] = SPECS
    model = model_factory(**kwargs)
    time = jnp.linspace(1.0, 1.1, 64)
    prior = {
        "period": jnp.array([3.0]),
        "u": jnp.array([0.2, 0.1]),
        "spot_guess": 1.04,
        "spot_guess2": 1.06,
        "t_jump_guess": 1.05,
        "jump_guess": 0.0,
    }
    return trace(seed(model, jax.random.PRNGKey(2))).get_trace(
        time,
        jnp.full_like(time, 1e-3),
        y=jnp.ones_like(time),
        prior_params=prior,
    )


def test_jaxoplanet_default_trends_use_physical_coordinates():
    spot_sites = _trace_default(create_whitelight_model, "2spot")
    step_sites = _trace_default(create_whitelight_model, "linear_discontinuity")

    for name in ("spot_mu", "spot_sigma", "spot_mu2", "spot_sigma2"):
        assert spot_sites[name]["type"] == "sample"
    for name in ("t_jump", "log_width"):
        assert step_sites[name]["type"] == "sample"
    assert not any(name.endswith("cadences") for name in spot_sites)
    assert not any(name.endswith("cadences") for name in step_sites)


def test_harmonica_default_trends_use_physical_coordinates():
    spot_sites = _trace_default(create_harmonica_whitelight_model, "2spot")
    step_sites = _trace_default(
        create_harmonica_whitelight_model, "linear_discontinuity"
    )

    for name in ("spot_mu", "spot_sigma", "spot_mu2", "spot_sigma2"):
        assert spot_sites[name]["type"] == "sample"
    for name in ("t_jump", "log_width"):
        assert step_sites[name]["type"] == "sample"
    assert not any(name.endswith("cadences") for name in spot_sites)
    assert not any(name.endswith("cadences") for name in step_sites)
