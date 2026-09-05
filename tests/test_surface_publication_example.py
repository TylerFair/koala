"""Physical normalization of the tutorial map (independent disk quadrature)."""
import numpy as np
import pytest

from tools.surface_publication_example import thermal_map


def test_posterior_lightcurves_match_production_baseline_scaling():
    from tools.surface_publication_example import posterior_lightcurves
    from models.jaxoplanet.surface_basis import EmissionLightCurveBasis
    from models.trends import compute_lc_linear
    import jax.numpy as jnp

    t = jnp.array([0., .5, 1.])
    basis = EmissionLightCurveBasis(baseline=jnp.zeros(3), uniform=jnp.array([1., 0., 1.]),
                                    dipole_cos=None, dipole_sin=None)
    samples = {"eclipse_depth": np.array([[[.001]], [[.002]]]),
               "c": np.array([[.95], [1.05]]), "v": np.array([[.001], [-.001]])}
    curves = posterior_lightcurves("eclipse", samples, basis, t)
    for draw in range(2):
        params = {key: value[draw, 0] for key, value in samples.items()}
        params.update(_surface_model="eclipse", _surface_basis=basis)
        np.testing.assert_allclose(curves[draw, 0], compute_lc_linear(params, t), atol=1e-14)


@pytest.mark.parametrize("view", [0., .3, np.pi, 2.4])
def test_map_disk_integral_matches_phase_extrema(view):
    # Gauss–Legendre quadrature on the visible hemisphere, with projected
    # area mu*dOmega; independent of the analytic map-to-phase conversion.
    nodes, weights = np.polynomial.legendre.leggauss(100)
    latitude = nodes[:, None] * np.pi / 2
    longitude = view + nodes[None, :] * np.pi / 2
    mu = np.cos(latitude) * np.cos(longitude - view)
    intensity = thermal_map(1200., 400., .3, longitude, latitude)
    integral = np.sum(intensity * mu * np.cos(latitude) * weights[:, None] * weights[None, :]) * (np.pi / 2)**2 / np.pi
    expected = 800. + 400. * np.cos(view - .3)
    np.testing.assert_allclose(integral, expected, rtol=1e-12)


def test_map_is_nonnegative_at_physical_boundary_and_uniform_for_equal_fluxes():
    longitude, latitude = np.meshgrid(np.linspace(-np.pi, np.pi, 101), np.linspace(-np.pi/2, np.pi/2, 51))
    assert thermal_map(1000., 200., 0., longitude, latitude).min() >= 0
    np.testing.assert_allclose(thermal_map(400., 400., .5, longitude, latitude), 400.)
