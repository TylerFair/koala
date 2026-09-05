import numpy as np
import pytest


pytest.importorskip("jaxoplanet")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from models.jaxoplanet.surface import compute_surface_model
from models.jaxoplanet.surface_basis import (
    compute_emission_basis_model,
    compute_spot_basis_model,
    prepare_emission_light_curve_basis,
    prepare_spot_light_curve_basis,
)


jax.config.update("jax_enable_x64", True)


SPOTS = (
    {"latitude": 0.15, "longitude": -0.3, "radius": 0.24, "contrast": 0.3},
    {"latitude": -0.25, "longitude": 1.1, "radius": 0.18, "contrast": 0.2},
)


def _params(epoch):
    return {
        "_surface_model": "transit",
        "period": 1.0,
        "t0": epoch,
        "a_rs": 7.0,
        "b": 0.25,
        "rors": 0.11,
        "ecc": 0.0,
        "omega": 0.0,
        "u": jnp.array([0.3, 0.2]),
        "stellar_rotation_period": 3.7,
    }


@pytest.mark.parametrize("epoch", [0.0, 60_000.123])
def test_basis_matches_native_at_two_contrasts_and_absolute_epochs(epoch):
    params = _params(epoch)
    time = epoch + jnp.linspace(-0.04, 0.04, 25)
    basis = prepare_spot_light_curve_basis(
        params, time, spots=SPOTS, spot_degree=4, order=10
    )
    contrast_cases = jnp.array([[0.2, 0.45], [0.65, 0.1]])
    basis_curves = jax.vmap(
        lambda contrasts: compute_spot_basis_model(
            {"stellar_spot_contrast": contrasts}, basis
        )
    )(contrast_cases)

    def native(contrasts):
        return compute_surface_model(
            {**params, "stellar_spot_contrast": contrasts},
            time,
            model="transit",
            spots=SPOTS,
            spot_degree=4,
            order=10,
        )

    native_curves = jax.jit(jax.vmap(native))(contrast_cases)
    np.testing.assert_allclose(basis_curves, native_curves, rtol=0, atol=2e-12)


def test_basis_contrast_gradient_matches_native_starry_gradient():
    params = _params(60_000.123)
    time = params["t0"] + jnp.linspace(-0.035, 0.035, 19)
    basis = prepare_spot_light_curve_basis(
        params, time, spots=SPOTS, spot_degree=4, order=10
    )
    weights = jnp.linspace(0.5, 1.5, time.size)

    def basis_objective(contrasts):
        signal = compute_spot_basis_model(
            {"stellar_spot_contrast": contrasts}, basis
        )
        return jnp.dot(weights, signal)

    def native_objective(contrasts):
        signal = compute_surface_model(
            {**params, "stellar_spot_contrast": contrasts},
            time,
            model="transit",
            spots=SPOTS,
            spot_degree=4,
            order=10,
        )
        return jnp.dot(weights, signal)

    contrasts = jnp.array([0.35, 0.25])
    basis_gradient = jax.grad(basis_objective)(contrasts)
    native_gradient = jax.jit(jax.grad(native_objective))(contrasts)
    np.testing.assert_allclose(basis_gradient, native_gradient, rtol=2e-11, atol=2e-12)


def test_basis_is_a_pytree_and_supports_batched_channel_contrasts():
    params = _params(0.0)
    time = jnp.linspace(-0.02, 0.02, 11)
    basis = prepare_spot_light_curve_basis(
        params, time, spots=SPOTS, spot_degree=3, order=10
    )
    contrasts = jnp.array([[0.2, 0.3], [0.4, 0.1], [0.0, 0.0]])
    result = jax.jit(
        lambda values, prepared: compute_spot_basis_model(
            {"stellar_spot_contrast": values}, prepared
        )
    )(contrasts, basis)

    assert result.shape == (3, time.size)
    np.testing.assert_allclose(result[-1], basis.baseline, rtol=0, atol=0)


def test_basis_requires_fixed_host_geometry_and_matching_contrasts():
    params = _params(0.0)
    time = jnp.linspace(-0.01, 0.01, 5)
    basis = prepare_spot_light_curve_basis(
        params, time, spots=SPOTS, spot_degree=3, order=10
    )
    with pytest.raises(ValueError, match="one value per prepared spot"):
        compute_spot_basis_model({"stellar_spot_contrast": jnp.array([0.2])}, basis)

    with pytest.raises(ValueError, match="fixed before JAX tracing"):
        jax.jit(
            lambda radius: prepare_spot_light_curve_basis(
                {**params, "rors": radius},
                time,
                spots=SPOTS,
                spot_degree=3,
                order=10,
            ).baseline
        )(0.1)


def test_eclipse_basis_matches_native_flux_and_depth_gradient():
    params = _params(60_000.123)
    params = {**params, "ecc": 0.22, "omega": 0.45}
    time = params["t0"] + jnp.linspace(0.3, 0.8, 31)
    basis = prepare_emission_light_curve_basis(
        params, time, model="eclipse", order=10
    )

    def basis_curve(depth):
        return compute_emission_basis_model({"eclipse_depth": depth}, basis)

    def native_curve(depth):
        return compute_surface_model(
            {**params, "eclipse_depth": depth},
            time,
            model="eclipse",
            order=10,
        )

    for depth in (2.5e-4, 1.1e-3):
        np.testing.assert_allclose(
            basis_curve(depth), native_curve(depth), rtol=0, atol=3e-12
        )
    weights = jnp.linspace(0.7, 1.3, time.size)
    basis_gradient = jax.grad(lambda depth: jnp.dot(weights, basis_curve(depth)))(
        8e-4
    )
    native_gradient = jax.grad(lambda depth: jnp.dot(weights, native_curve(depth)))(
        8e-4
    )
    np.testing.assert_allclose(basis_gradient, native_gradient, rtol=0, atol=3e-12)


def test_phase_basis_matches_native_eccentric_curve_and_all_gradients():
    params = _params(60_000.123)
    params = {**params, "ecc": 0.27, "omega": -0.35}
    time = params["t0"] + jnp.linspace(-0.08, 1.08, 39)
    basis = prepare_emission_light_curve_basis(
        params, time, model="phase_curve", order=10
    )

    def basis_curve(values):
        day, night, offset = values
        return compute_emission_basis_model(
            {
                "dayside_flux": day,
                "nightside_flux": night,
                "hotspot_offset": offset,
            },
            basis,
        )

    def native_curve(values):
        day, night, offset = values
        return compute_surface_model(
            {
                **params,
                "dayside_flux": day,
                "nightside_flux": night,
                "hotspot_offset": offset,
            },
            time,
            model="phase_curve",
            order=10,
        )

    cases = (
        jnp.array([1.0e-3, 2.5e-4, 0.31]),
        jnp.array([6.0e-4, 3.0e-4, -0.72]),
    )
    for values in cases:
        np.testing.assert_allclose(
            basis_curve(values), native_curve(values), rtol=0, atol=4e-12
        )

    weights = jnp.linspace(0.5, 1.5, time.size)
    values = cases[0]
    basis_gradient = jax.jacrev(
        lambda point: jnp.dot(weights, basis_curve(point))
    )(values)
    native_gradient = jax.jacrev(
        lambda point: jnp.dot(weights, native_curve(point))
    )(values)
    np.testing.assert_allclose(
        basis_gradient, native_gradient, rtol=2e-9, atol=3e-11
    )
