"""A GP on top of spot-crossing and step trends: the white-light GP mean is the non-GP trend kernel, and the channel
kernel is the non-GP channel kernel plus the GP template."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.handlers import seed, substitute, trace

jax.config.update("jax_enable_x64", True)

import models.gp as kgp
import models.trends as trends
from models.detrend import resolve_detrend_kernel
from models.jaxoplanet.builder import create_vectorized_model, create_whitelight_model
from planet_specs import default_planet_specs

SPECS = default_planet_specs(period=3.0, t0=1.0, b=0.2, rprs=0.1, duration=0.12)
T = jnp.linspace(0.85, 1.15, 301)
PARAMS = dict(period=jnp.array([3.0]), t0=jnp.array([1.0]), rors=jnp.array([0.1]), b=jnp.array([0.2]),
              duration=jnp.array([0.12]), u=jnp.array([0.3, 0.2]), c=jnp.array(0.998), v=jnp.array(0.004),
              spot_amp=jnp.array(4e-4), spot_mu=jnp.array(1.01), spot_sigma=jnp.array(0.006),
              spot_amp2=jnp.array(2e-4), spot_mu2=jnp.array(0.97), spot_sigma2=jnp.array(0.004),
              jump=jnp.array(-3e-4), t_jump=jnp.array(1.08), width=jnp.array(2e-3))

WHITE = {"spot+gp": trends.compute_lc_spot, "2spot+gp": trends.compute_lc_2spot,
         "linear_discontinuity+gp": trends.compute_lc_linear_discontinuity,
         "spot+linear_discontinuity+gp": trends.compute_lc_spot_linear_discontinuity}


@pytest.mark.parametrize("name", sorted(WHITE))
def test_white_light_gp_mean_is_the_trend_kernel(name):
    gp_mean = kgp.resolve_gp_mean_function(name)(PARAMS, T, t_ref=jnp.min(T))
    np.testing.assert_allclose(np.asarray(gp_mean), np.asarray(WHITE[name](PARAMS, T)), rtol=0, atol=1e-15)


def test_white_light_gp_mean_honours_the_reference_time():
    """Conditioning at a subset of times must keep the full light curve's time origin (models.gp t_ref)."""
    mean = kgp.resolve_gp_mean_function("spot+gp")
    full = mean(PARAMS, T, t_ref=jnp.min(T))
    np.testing.assert_allclose(np.asarray(mean(PARAMS, T[100:], t_ref=jnp.min(T))), np.asarray(full[100:]),
                               rtol=0, atol=1e-15)


def test_component_order_does_not_matter():
    assert kgp.resolve_gp_mean_function("gp+spot") is kgp.resolve_gp_mean_function("spot+gp")
    assert kgp.resolve_gp_builder("linear_discontinuity+gp+spot") is kgp.GP_BUILDERS["spot+linear_discontinuity+gp"]
    with pytest.raises(ValueError):
        kgp.resolve_gp_mean_function("spot+bogus+gp")


CHANNEL = {"spot_spectroscopic+gp_spectroscopic": ("spot_spectroscopic", ("spot_trend",)),
           "2spot_spectroscopic+gp_spectroscopic": ("2spot_spectroscopic", ("spot_trend", "spot_trend2")),
           "linear_discontinuity_spectroscopic+gp_spectroscopic": ("linear_discontinuity_spectroscopic",
                                                                     ("jump_trend",)),
           "spot_spectroscopic+linear_discontinuity_spectroscopic+gp_spectroscopic": (
               "spot_spectroscopic+linear_discontinuity_spectroscopic", ("spot_trend", "jump_trend"))}
TEMPLATES = dict(spot_trend=trends.spot_crossing(T, 4e-4, 1.01, 0.006),
                 spot_trend2=trends.spot_crossing(T, 2e-4, 0.97, 0.004),
                 jump_trend=-3e-4 * trends._soft_step(T, 1.08, 2e-3),
                 gp_trend=2e-4 * jnp.sin(40.0 * T))
AMPLITUDES = dict(A_spot=jnp.array(1.3), A_spot2=jnp.array(0.7), A_jump=jnp.array(1.1))


@pytest.mark.parametrize("name", sorted(CHANNEL))
def test_channel_kernel_is_the_event_kernel_plus_the_gp_template(name):
    base, used = CHANNEL[name]
    kernel = resolve_detrend_kernel(name)
    p = dict(PARAMS, **AMPLITUDES)
    args = {k: TEMPLATES[k] for k in used}
    reference = resolve_detrend_kernel(base)(p, T, *args.values())
    without = kernel(dict(p, A_gp=jnp.array(0.0)), T, TEMPLATES["gp_trend"], **args)
    np.testing.assert_allclose(np.asarray(without), np.asarray(reference), rtol=0, atol=1e-15)
    # the GP template enters the trend like any other template: transit x (trend + A_gp * gp)
    transit = jnp.reshape(trends.compute_transit_model_auto(p, T), T.shape)
    with_gp = kernel(dict(p, A_gp=jnp.array(1.4)), T, TEMPLATES["gp_trend"], **args)
    added = trends.apply_systematics(transit, 1.4 * TEMPLATES["gp_trend"]) - trends.apply_systematics(transit, 0.0)
    np.testing.assert_allclose(np.asarray(with_gp - without), np.asarray(added), rtol=0, atol=1e-15)


@pytest.mark.parametrize("name", sorted(CHANNEL))
def test_vectorized_model_uses_every_template(name):
    """The channel model of a GP-on-events trend uses the GP template and every event template (none dropped)."""
    from numpyro.infer.util import log_density
    _, used = CHANNEL[name]
    model = create_vectorized_model(detrend_type=name, ld_mode="fixed", trend_mode="free")
    n = 2
    base = dict(rors_reference=jnp.array([.1]), mu_duration=jnp.array([.12]), mu_t0=jnp.array([1.0]),
                mu_b=jnp.array([.2]), PERIOD=jnp.array([3.0]), ld_fixed=jnp.tile(jnp.array([.3, .2]), (n, 1)),
                precomputed_yerr_per_lc=jnp.full(n, 1e-4), y=jnp.ones((n, T.size)),
                gp_trend=TEMPLATES["gp_trend"], **{k: TEMPLATES[k] for k in used})
    values = dict(rors=jnp.full((n, 1), .1), log_jitter=jnp.full(n, -12.), c=jnp.full(n, .998), v=jnp.full(n, .004),
                  A_gp=jnp.full(n, 1.4), **{k: jnp.full(n, float(v)) for k, v in AMPLITUDES.items()})

    def loc(kw, vals):
        _, tr = log_density(model, (T, jnp.full((n, T.size), 1e-4)), kw, vals)
        return np.asarray(tr["obs"]["fn"].loc)

    full = loc(base, values)
    assert np.ptp(full - loc(base, dict(values, A_gp=jnp.zeros(n)))) > 1e-6
    for k in used:
        assert np.ptp(full - loc(dict(base, **{k: jnp.zeros_like(T)}), values)) > 1e-6, k
    # and the model mean is the kernel itself, channel by channel
    p = dict(PARAMS, c=jnp.array(.998), v=jnp.array(.004), A_gp=jnp.array(1.4), **AMPLITUDES)
    kernel = resolve_detrend_kernel(name)(p, T, TEMPLATES["gp_trend"], **{k: TEMPLATES[k] for k in used})
    np.testing.assert_allclose(full[0], np.asarray(kernel), rtol=0, atol=2e-12)


@pytest.mark.parametrize("name", sorted(WHITE))
def test_white_light_model_builds_with_the_gp(name):
    model = create_whitelight_model(detrend_type=name, ld_mode="fixed", ld_profile="quadratic", gp_solver="serial",
                                    gp_assume_sorted=True, parameter_priors=SPECS)
    t = jnp.linspace(0.9, 1.1, 64)
    sites = trace(seed(model, 3)).get_trace(t, jnp.full_like(t, 1e-3), y=jnp.ones_like(t),
                                            prior_params={"period": jnp.array([3.0]), "u": jnp.array([0.2, 0.1]),
                                                          "spot_guess": 1.0, "spot_guess2": 0.98,
                                                          "t_jump_guess": 1.05})
    assert "GP_log_sigma" in sites and "obs" in sites
    for site in {"spot+gp": ["spot_amp"], "2spot+gp": ["spot_amp", "spot_amp2"],
                 "linear_discontinuity+gp": ["jump"], "spot+linear_discontinuity+gp": ["spot_amp", "jump"]}[name]:
        assert site in sites, site
