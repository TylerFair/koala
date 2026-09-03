import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
from numpyro.handlers import seed, trace

from models.independent_nuts import _synchronize_cadence_init_values
from models.harmonica.builder import create_whitelight_model as create_harmonica_whitelight_model
from models.jaxoplanet.builder import create_whitelight_model
from models.trends import sample_step_width


def _sites(monkeypatch, detrend_type, parameterization="cadence"):
    if parameterization is None:
        monkeypatch.delenv("JWSTJAXFIT_WL_TREND_PARAMETERIZATION", raising=False)
    else:
        monkeypatch.setenv(
            "JWSTJAXFIT_WL_TREND_PARAMETERIZATION", parameterization
        )
    model = create_whitelight_model(detrend_type=detrend_type, ld_mode="fixed")
    t = jnp.linspace(1.0, 1.1, 64)
    prior = {
        "period": jnp.array([3.0]),
        "u": jnp.array([0.2, 0.1]),
        "spot_guess": 1.04,
        "spot_guess2": 1.06,
        "t_jump_guess": 1.05,
        "jump_guess": 0.0,
    }
    return trace(seed(model, jax.random.PRNGKey(2))).get_trace(
        t, jnp.full_like(t, 1e-3), y=jnp.ones_like(t), prior_params=prior
    )


def _configured_sites(monkeypatch, parameterization, ordering="ordered"):
    monkeypatch.delenv("JWSTJAXFIT_WL_TREND_PARAMETERIZATION", raising=False)
    model = create_whitelight_model(
        detrend_type="2spot",
        ld_mode="fixed",
        trend_parameterization=parameterization,
        two_spot_ordering=ordering,
    )
    t = jnp.linspace(1.0, 1.1, 64)
    prior = {
        "period": jnp.array([3.0]),
        "u": jnp.array([0.2, 0.1]),
        "spot_guess": 1.04,
        "spot_guess2": 1.06,
    }
    return trace(seed(model, jax.random.PRNGKey(12))).get_trace(
        t, jnp.full_like(t, 1e-3), y=jnp.ones_like(t), prior_params=prior
    )


def test_cadence_spot_exposes_physical_deterministics(monkeypatch):
    sites = _sites(monkeypatch, "2spot")
    for name in (
        "spot_mu_offset_cadences",
        "spot_sigma_cadences",
        "spot_mu2_offset_cadences",
        "spot_sigma2_cadences",
    ):
        assert sites[name]["type"] == "sample"
    for name in ("spot_mu", "spot_sigma", "spot_mu2", "spot_sigma2"):
        assert sites[name]["type"] == "deterministic"
    assert sites["spot_sigma"]["value"] > 0
    assert sites["spot_sigma2"]["value"] > 0


def test_cadence_step_exposes_normalized_latents(monkeypatch):
    sites = _sites(monkeypatch, "linear_discontinuity")
    for name in ("t_jump_offset_cadences", "log_width_unit_logit"):
        assert sites[name]["type"] == "sample"
    for name in (
        "t_jump",
        "log_width_cadences",
        "log_width",
        "width",
        "width_minutes",
    ):
        assert sites[name]["type"] == "deterministic"
    assert sites["width"]["value"] > 0


def test_harmonica_cadence_spot_uses_normalized_latents(monkeypatch):
    monkeypatch.setenv("JWSTJAXFIT_WL_TREND_PARAMETERIZATION", "cadence")
    model = create_harmonica_whitelight_model(
        detrend_type="2spot", ld_mode="fixed", ld_profile="quadratic"
    )
    t = jnp.linspace(1.0, 1.1, 64)
    sites = trace(seed(model, jax.random.PRNGKey(3))).get_trace(
        t,
        jnp.full_like(t, 1e-3),
        y=jnp.ones_like(t),
        prior_params={
            "period": jnp.array([3.0]),
            "u": jnp.array([0.2, 0.1]),
            "spot_guess": 1.04,
            "spot_guess2": 1.06,
        },
    )
    for name in (
        "spot_mu_offset_cadences",
        "spot_sigma_cadences",
        "spot_mu2_offset_cadences",
        "spot_sigma2_cadences",
    ):
        assert sites[name]["type"] == "sample"
    for name in ("spot_mu", "spot_sigma", "spot_mu2", "spot_sigma2"):
        assert sites[name]["type"] == "deterministic"


def test_harmonica_cadence_step_uses_normalized_latents(monkeypatch):
    monkeypatch.setenv("JWSTJAXFIT_WL_TREND_PARAMETERIZATION", "cadence")
    model = create_harmonica_whitelight_model(
        detrend_type="linear_discontinuity",
        ld_mode="fixed",
        ld_profile="quadratic",
    )
    t = jnp.linspace(1.0, 1.1, 64)
    sites = trace(seed(model, jax.random.PRNGKey(4))).get_trace(
        t,
        jnp.full_like(t, 1e-3),
        y=jnp.ones_like(t),
        prior_params={
            "period": jnp.array([3.0]),
            "u": jnp.array([0.2, 0.1]),
            "t_jump_guess": 1.05,
            "jump_guess": 0.0,
        },
    )
    for name in ("t_jump_offset_cadences", "log_width_unit_logit"):
        assert sites[name]["type"] == "sample"
    for name in ("t_jump", "log_width_cadences", "log_width", "width"):
        assert sites[name]["type"] == "deterministic"


def test_real_config_parameterization_works_without_environment(monkeypatch):
    sites = _configured_sites(monkeypatch, "cadence", ordering="legacy")
    assert sites["spot_mu_offset_cadences"]["type"] == "sample"
    assert sites["spot_mu"]["type"] == "deterministic"


def test_ordered_two_spot_trace_has_one_canonical_labeling(monkeypatch):
    sites = _configured_sites(monkeypatch, "cadence")
    for name in (
        "spot_center_midpoint_offset_cadences",
        "log_spot_center_separation_cadences",
    ):
        assert sites[name]["type"] == "sample"
    for name in ("spot_mu", "spot_mu2", "spot_center_separation"):
        assert sites[name]["type"] == "deterministic"
    assert sites["spot_mu"]["value"] < sites["spot_mu2"]["value"]
    assert "spot_mu_offset_cadences" not in sites
    assert "spot_mu2_offset_cadences" not in sites


def test_harmonica_ordered_two_spot_uses_same_canonical_sites(monkeypatch):
    monkeypatch.delenv("JWSTJAXFIT_WL_TREND_PARAMETERIZATION", raising=False)
    model = create_harmonica_whitelight_model(
        detrend_type="2spot",
        ld_mode="fixed",
        ld_profile="quadratic",
        trend_parameterization="cadence",
        two_spot_ordering="ordered",
    )
    t = jnp.linspace(1.0, 1.1, 64)
    sites = trace(seed(model, jax.random.PRNGKey(13))).get_trace(
        t,
        jnp.full_like(t, 1e-3),
        y=jnp.ones_like(t),
        prior_params={
            "period": jnp.array([3.0]),
            "u": jnp.array([0.2, 0.1]),
            "spot_guess": 1.04,
            "spot_guess2": 1.06,
        },
    )
    assert sites["spot_mu"]["value"] < sites["spot_mu2"]["value"]
    assert sites["ordered_spot_center_prior"]["type"] == "sample"


def test_ordered_two_spot_density_sums_both_labels_and_jacobian(monkeypatch):
    cadence = jnp.median(jnp.diff(jnp.linspace(1.0, 1.1, 64)))
    first_prior = dist.Normal(1.04, 0.01)
    second_prior = dist.Normal(1.06, 0.01)
    for parameterization in ("physical", "cadence"):
        sites = _configured_sites(monkeypatch, parameterization)
        mu1 = sites["spot_mu"]["value"]
        mu2 = sites["spot_mu2"]["value"]
        separation = sites["spot_center_separation"]["value"]
        target = logsumexp(jnp.stack((
            first_prior.log_prob(mu1) + second_prior.log_prob(mu2),
            first_prior.log_prob(mu2) + second_prior.log_prob(mu1),
        )))
        if parameterization == "physical":
            latent_names = (
                "spot_center_midpoint",
                "log_spot_center_separation",
            )
            log_jacobian = jnp.log(separation)
        else:
            latent_names = (
                "spot_center_midpoint_offset_cadences",
                "log_spot_center_separation_cadences",
            )
            log_jacobian = jnp.log(cadence) + jnp.log(separation)
        base = sum(
            sites[name]["fn"].log_prob(sites[name]["value"])
            for name in latent_names
        )
        actual = base + sites["ordered_spot_center_prior"]["fn"].log_factor
        assert jnp.allclose(actual, target + log_jacobian, atol=1e-12)


def test_ordered_physical_and_cadence_draws_have_identical_model(monkeypatch):
    physical = _configured_sites(monkeypatch, "physical")
    cadence = _configured_sites(monkeypatch, "cadence")
    for name in ("spot_mu", "spot_mu2", "spot_sigma", "spot_sigma2"):
        assert jnp.allclose(
            physical[name]["value"], cadence[name]["value"], atol=1e-12
        )
    assert jnp.allclose(
        physical["obs"]["fn"].loc,
        cadence["obs"]["fn"].loc,
        atol=1e-12,
    )


def test_physical_default_has_no_cadence_latents(monkeypatch):
    spot_sites = _sites(monkeypatch, "2spot", None)
    step_sites = _sites(monkeypatch, "linear_discontinuity", None)
    for name in (
        "spot_mu",
        "spot_sigma",
        "spot_mu2",
        "spot_sigma2",
    ):
        assert spot_sites[name]["type"] == "sample"
    for name in (
        "t_jump",
        "log_width",
    ):
        assert step_sites[name]["type"] == "sample"
    assert not any(name.endswith("cadences") for name in spot_sites)
    assert not any(name.endswith("cadences") for name in step_sites)


def test_cadence_and_physical_prior_draws_map_to_same_model(monkeypatch):
    physical = _sites(monkeypatch, "spot+linear_discontinuity", "physical")
    cadence = _sites(monkeypatch, "spot+linear_discontinuity", "cadence")
    for name in (
        "spot_mu",
        "spot_sigma",
        "t_jump",
        "log_width",
        "width",
    ):
        assert jnp.allclose(
            physical[name]["value"], cadence[name]["value"], rtol=1e-12, atol=1e-12
        )
    assert jnp.allclose(
        physical["obs"]["fn"].loc,
        cadence["obs"]["fn"].loc,
        rtol=1e-12,
        atol=1e-12,
    )


def test_cadence_priors_induce_original_physical_priors(monkeypatch):
    spot_sites = _sites(monkeypatch, "2spot")
    step_sites = _sites(monkeypatch, "linear_discontinuity")
    cadence = float(jnp.median(jnp.diff(jnp.linspace(1.0, 1.1, 64))))

    assert float(spot_sites["spot_mu_offset_cadences"]["fn"].loc) == 0.0
    assert jnp.isclose(
        spot_sites["spot_mu_offset_cadences"]["fn"].scale * cadence, 0.01
    )
    assert jnp.isclose(
        spot_sites["spot_sigma_cadences"]["fn"].low * cadence, 1e-4
    )
    assert jnp.isclose(
        spot_sites["spot_sigma_cadences"]["fn"].high * cadence, 0.1
    )
    assert jnp.isclose(
        step_sites["t_jump_offset_cadences"]["fn"].scale * cadence, 1e-2
    )
    assert isinstance(step_sites["log_width_unit_logit"]["fn"], dist.Logistic)

    mu_offset = jnp.asarray(0.25)
    mu = 1.04 + cadence * mu_offset
    assert jnp.isclose(
        spot_sites["spot_mu_offset_cadences"]["fn"].log_prob(mu_offset),
        dist.Normal(1.04, 0.01).log_prob(mu) + jnp.log(cadence),
    )
    sigma_cadences = jnp.asarray(2.0)
    sigma = cadence * sigma_cadences
    assert jnp.isclose(
        spot_sites["spot_sigma_cadences"]["fn"].log_prob(sigma_cadences),
        dist.Uniform(1e-4, 0.1).log_prob(sigma) + jnp.log(cadence),
    )
    log_width_low = jnp.log(0.5)
    log_width_high = jnp.log(30.0 / (24.0 * 60.0) / cadence)
    log_width_unit_logit = jnp.asarray(-0.75)
    fraction = jax.nn.sigmoid(log_width_unit_logit)
    log_width_cadences = log_width_low + (
        log_width_high - log_width_low
    ) * fraction
    physical_log_width = log_width_cadences + jnp.log(cadence)
    assert jnp.isclose(
        step_sites["log_width_unit_logit"]["fn"].log_prob(
            log_width_unit_logit
        ),
        dist.Uniform(
            jnp.log(0.5 * cadence),
            jnp.log(30.0 / (24.0 * 60.0)),
        ).log_prob(physical_log_width)
        + jnp.log(log_width_high - log_width_low)
        + jnp.log(fraction)
        + jnp.log1p(-fraction),
    )


def test_laplace_start_uses_repaired_physical_step_width():
    time = jnp.linspace(10.0, 10.1, 101)
    cadence = jnp.median(jnp.diff(time))
    repaired_log_width = jnp.log(4.0 * cadence)
    synchronized = _synchronize_cadence_init_values(
        {
            "log_width": repaired_log_width,
            "log_width_cadences": jnp.log(0.5),
            "log_width_unit_logit": -100.0,
        },
        (time,),
    )
    assert jnp.isclose(synchronized["log_width_cadences"], jnp.log(4.0))
    low = jnp.log(0.5)
    high = jnp.log((30.0 / (24.0 * 60.0)) / cadence)
    expected_fraction = (jnp.log(4.0) - low) / (high - low)
    assert jnp.isclose(
        jax.nn.sigmoid(synchronized["log_width_unit_logit"]),
        expected_fraction,
    )

    physical = {"log_width": repaired_log_width}
    assert _synchronize_cadence_init_values(physical, (time,)) is physical


def test_cadence_step_width_uses_finite_unconstrained_start(monkeypatch):
    monkeypatch.setenv("JWSTJAXFIT_WL_TREND_PARAMETERIZATION", "cadence")
    time = jnp.arange(12, dtype=jnp.float64) * 2.0e-4

    def model():
        sample_step_width(time, {"step_width_mode": "free"})

    info = initialize_model(
        jax.random.PRNGKey(9),
        model,
        init_strategy=init_to_value(
            values={"log_width_unit_logit": -100.0}
        ),
    )
    assert jnp.isfinite(info.param_info.z["log_width_unit_logit"])
    assert jnp.isclose(info.param_info.z["log_width_unit_logit"], -100.0)
