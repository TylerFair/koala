import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed, trace

from fit_jwst import _resolve_harmonica_stage_nuts_kwargs
from models.harmonica.builder import create_vectorized_model


def test_harmonica_resolves_laplace_independent_nuts_options():
    flags = {
        "spectro_sampler": "independent_nuts",
        "spectro_mass_matrix": "laplace",
        "spectro_laplace_hessian_method": "finite_difference",
        "spectro_laplace_target_accept": 0.99,
    }
    options = _resolve_harmonica_stage_nuts_kwargs(
        flags, "harmonica_lr", default_dense_mass=True
    )
    assert options["mass_matrix"] == "laplace"
    assert options["laplace_hessian_method"] == "finite_difference"
    assert options["laplace_target_accept"] == 0.99


def test_harmonica_quadratic_sing_sites_flow_to_limb_coefficients(monkeypatch):
    captured = {}

    def fake_transit(params, t):
        captured.update(params)
        return jnp.zeros((1, t.size), dtype=jnp.float64)

    monkeypatch.setattr(
        "models.harmonica.builder.compute_transit_model_harmonica_batched",
        fake_transit,
    )
    model = create_vectorized_model(
        ld_mode="sing", ld_profile="quadratic", fit_jitter=False
    )
    values = trace(seed(model, jax.random.PRNGKey(1))).get_trace(
        jnp.linspace(0.0, 0.1, 8),
        jnp.full((1, 8), 1e-4),
        y=jnp.ones((1, 8)),
        mu_t0=jnp.array([0.05]),
        mu_b=jnp.array([0.2]),
        mu_cos_i=jnp.array([0.02]),
        PERIOD=jnp.array([3.0]),
        harmonica_a_rs=jnp.array([10.0]),
        mu_u_ld=jnp.array([[0.45, 0.03]]),
        sigma_u_ld=jnp.array([[0.02, 0.01]]),
    )
    assert "limb_l" in values and "limb_delta" in values
    assert jnp.allclose(captured["u1_ld"], values["c1"]["value"])
    assert jnp.allclose(captured["u2_ld"], values["c2"]["value"])
