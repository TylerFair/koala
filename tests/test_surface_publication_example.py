"""Production baseline scaling of the surface tutorial's posterior light curves."""
import numpy as np


def test_posterior_lightcurves_match_production_baseline_scaling():
    from tools.surface_publication_example import posterior_lightcurves
    from models.jaxoplanet.surface_basis import EmissionLightCurveBasis
    from models.trends import compute_lc_linear
    import jax.numpy as jnp

    t = jnp.array([0., .5, 1.])
    basis = EmissionLightCurveBasis(baseline=jnp.zeros(3), uniform=jnp.array([1., 0., 1.]),
                                    phase_cos=None, phase_sin=None)
    samples = {"eclipse_depth": np.array([[[.001]], [[.002]]]),
               "c": np.array([[.95], [1.05]]), "v": np.array([[.001], [-.001]])}
    curves = posterior_lightcurves("eclipse", samples, basis, t)
    for draw in range(2):
        params = {key: value[draw, 0] for key, value in samples.items()}
        params.update(_surface_model="eclipse", _surface_basis=basis)
        np.testing.assert_allclose(curves[draw, 0], compute_lc_linear(params, t), atol=1e-14)
