"""Projected power-2 must never silently become a production surface law."""
import jax.numpy as jnp
import pytest
from models.configuration_support import validate_power2_support
from models.jaxoplanet.builder import create_whitelight_model, create_vectorized_model
from models.jaxoplanet.core import compute_transit_model
from models.jaxoplanet.config import parse_surface_config


@pytest.mark.parametrize('factory', [create_whitelight_model, create_vectorized_model])
@pytest.mark.parametrize('model,spots', [('eclipse', ()), ('phase_curve', ()),
                                         ('transit', ({'contrast': .5},)),
                                         ('transit', ({'contrast': .5}, {'contrast': .6}))])
def test_power2_surface_classes_raise_with_exact_alternatives(factory, model, spots):
    with pytest.raises(ValueError, match='Use quadratic limb darkening'):
        factory(ld_profile='power2', surface_config={'model': model, 'spots': spots})


def test_public_config_rejects_power2_surface_before_setup():
    with pytest.raises(ValueError, match='degree-12 surface projection'):
        parse_surface_config({'ld_profile': 'power2', 'light_curve_model': 'eclipse'}, {}, {}, 1)


def test_surface_basis_cannot_bypass_power2_or_prepared_geometry_guards():
    for basis in [None, object()]:
        with pytest.raises(ValueError, match='degree-12 surface projection'):
            compute_transit_model(dict(_ld_profile='power2', _surface_model='eclipse',
                                       _surface_basis=basis), jnp.array([0.]))
    with pytest.raises(ValueError, match='Prepared transit separations'):
        compute_transit_model(dict(_surface_basis=object(), _transit_separation=jnp.ones((1, 1))),
                              jnp.array([0.]))


@pytest.mark.parametrize('factory', [create_whitelight_model, create_vectorized_model])
def test_explicit_polynomial_power2_transit_is_rejected(factory):
    with pytest.raises(ValueError, match="power2_evaluation='exact'"):
        factory(ld_profile='power2', power2_evaluation='polynomial')


def test_exact_unspotted_power2_and_quadratic_surface_policy_remain_available():
    for trend in ['constant', 'linear', 'quadratic', 'cubic', 'quartic', 'explinear', 'gp']:
        # Trend choice cannot turn an exact transit into a projected law.
        validate_power2_support('power2')
    for model in ['transit', 'eclipse', 'phase_curve']:
        validate_power2_support('quadratic', model=model, spots=({'contrast': .5},))
