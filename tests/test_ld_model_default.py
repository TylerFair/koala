"""Stagger by default, MPS-ATLAS for stars it doesn't cover, solar Stagger models for metal-rich stars."""
import pytest

from koala.limb_darkening import ld_model_metallicity, resolve_ld_model


@pytest.mark.parametrize('teff, logg, feh, expected', [
    (5509, 4.22, 0.04, 'stagger'),   # WASP-39: metal-rich, so Stagger's solar models
    (4870, 4.57, 0.10, 'stagger'),
    (3300, 4.90, 0.00, 'mps1'),      # an M dwarf is cooler than any Stagger model
    (7500, 4.20, 0.00, 'mps1'),      # and so is the other end
    (5500, 4.40, -3.5, 'mps1'),      # more metal-poor than the grid
])
def test_default_grid_follows_stagger_coverage(teff, logg, feh, expected):
    assert resolve_ld_model({'teff': teff, 'logg': logg, 'feh': feh}) == expected


def test_explicit_grid_is_kept_and_only_stagger_caps_metallicity():
    assert resolve_ld_model({'teff': 3300, 'logg': 4.9, 'feh': 0.0, 'ld_model': 'stagger'}) == 'stagger'
    assert resolve_ld_model({'teff': 5509, 'logg': 4.22, 'feh': 0.04, 'ld_model': 'mps2'}) == 'mps2'
    assert ld_model_metallicity(0.3, 'stagger') == 0.0
    assert ld_model_metallicity(-0.5, 'stagger') == -0.5
    assert ld_model_metallicity(0.3, 'mps1') == 0.3
