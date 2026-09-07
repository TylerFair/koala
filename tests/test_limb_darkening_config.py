import pytest

from models.limb_darkening_config import (
    GAUSSIAN_LD_WIDTH,
    LD_PRIORS,
    LD_PROFILES,
    resolve_ld_prior,
    validate_ld_prior,
    validate_ld_profile,
)


def test_public_limb_darkening_choices_are_canonical_and_small():
    assert LD_PROFILES == ("quadratic", "power2")
    assert LD_PRIORS == ("uniform", "gaussian", "sing", "stellarprior", "fixed")
    assert GAUSSIAN_LD_WIDTH == 0.2


@pytest.mark.parametrize(
    "alias",
    [
        "free",
        "widegaussian",
        "informed",
        "stellar",
        "stellarinformed",
        "stellar-informed",
        "singfree",
        "sing_free",
        "singuniform",
        "sing_uniform",
    ],
)
def test_old_ld_prior_aliases_are_rejected(alias):
    with pytest.raises(ValueError, match="flags.ld_prior must be one of"):
        validate_ld_prior(alias, ld_profile="quadratic")


def test_ld_values_are_exact_not_case_or_whitespace_aliases():
    with pytest.raises(ValueError, match="flags.ld_profile must be one of"):
        validate_ld_profile("Power2")
    with pytest.raises(ValueError, match="flags.ld_prior must be one of"):
        validate_ld_prior(" gaussian ", ld_profile="power2")


def test_profile_specific_priors_are_checked():
    with pytest.raises(ValueError, match="'sing'.*'quadratic'"):
        validate_ld_prior("sing", ld_profile="power2")
    with pytest.raises(ValueError, match="'stellarprior'.*'power2'"):
        validate_ld_prior("stellarprior", ld_profile="quadratic")


def test_omitted_prior_resolves_only_to_canonical_names():
    assert resolve_ld_prior(
        None, ld_profile="power2", has_stellar_uncertainties=True
    ) == "stellarprior"
    assert resolve_ld_prior(None, ld_profile="power2") == "gaussian"
    assert resolve_ld_prior(None, ld_profile="quadratic") == "gaussian"


@pytest.mark.parametrize(
    "alias", ["free", "widegaussian", "informed", "stellar", "sing_free"]
)
def test_public_config_parser_rejects_old_prior_aliases(alias):
    import fit_jwst

    with pytest.raises(ValueError, match="flags.ld_prior must be one of"):
        fit_jwst._resolve_ld_prior_mode(
            {"ld_prior": alias},
            {"teff_sigma": 1.0, "logg_sigma": 0.1, "feh_sigma": 0.1},
            "power2",
        )


@pytest.mark.parametrize("profile", ["quadratic", "power2"])
def test_gaussian_prior_uses_width_point_two_for_both_profiles(profile):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numpyro import handlers

    from models.jaxoplanet import create_vectorized_model

    model = create_vectorized_model(
        ld_mode="gaussian",
        ld_profile=profile,
        ld_parameterization="coefficients",
        transit_window="off",
    )
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(4))).get_trace(
        jnp.linspace(-0.03, 0.03, 11),
        jnp.full((2, 11), 1e-3),
        y=jnp.ones((2, 11)),
        mu_duration=jnp.array([0.06]),
        mu_t0=jnp.array([0.0]),
        mu_b=jnp.array([0.3]),
        mu_depths=jnp.full((2, 1), 0.01),
        PERIOD=jnp.array([3.0]),
        mu_u_ld=jnp.full((2, 2), 0.4),
        sigma_u_ld=jnp.full((2, 2), 0.03),
    )
    def normal_scale(site):
        distribution = trace[site]["fn"]
        while not hasattr(distribution, "scale"):
            distribution = distribution.base_dist
        return np.asarray(distribution.scale)

    scales = (
        normal_scale("u")
        if profile == "quadratic"
        else np.column_stack((normal_scale("c1"), normal_scale("c2")))
    )
    np.testing.assert_allclose(scales, GAUSSIAN_LD_WIDTH)
