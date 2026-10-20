"""Cached bandpass setup must preserve the independent ExoTiC-LD result."""
import numpy as np
import pytest
from exotic_ld import StellarLimbDarkening
from koala.limb_darkening import CachedStellarLimbDarkening


def stellar(cls):
    sld = object.__new__(cls)
    sld.verbose = 0
    sld.mus = np.linspace(1., 0., 12)
    sld.stellar_wavelengths = np.linspace(6000., 53000., 1200)
    c = .6 + .05*np.sin(sld.stellar_wavelengths/3000.)
    sld.stellar_intensities = 1. - c[:,None]*(1.-sld.mus[None,:]**.65)
    return sld


@pytest.mark.parametrize('bounds', [(10000., 24000.), (15000., 15001.), (6000., 8000.), (49000., 53000.)])
@pytest.mark.parametrize('profile', ['quadratic', 'power2'])
def test_cached_bandpass_matches_original_law_fit(bounds, profile):
    reference = stellar(StellarLimbDarkening)
    actual = stellar(CachedStellarLimbDarkening)
    waves = np.linspace(6000., 53000., 75)
    throughput = np.exp(-((waves-28000.)/16000.)**2)
    options = dict(wavelength_range=bounds, mode='custom',
                   custom_wavelengths=waves, custom_throughput=throughput,
                   return_sigmas=True)
    expected = getattr(reference, 'compute_'+profile+'_ld_coeffs')(**options)
    result = getattr(actual, 'compute_'+profile+'_ld_coeffs')(**options)
    np.testing.assert_allclose(actual.I_mu, reference.I_mu, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result, expected, atol=2e-8, rtol=1e-6)


def test_instrument_throughput_read_once_per_object():
    actual = stellar(CachedStellarLimbDarkening)
    calls = []
    def read(mode):
        calls.append(mode)
        return np.linspace(6000.,53000.,50),np.ones(50)
    actual._read_sensitivity_data = read
    for bounds in [(10000.,20000.),(20000.,30000.)]:
        actual.compute_quadratic_ld_coeffs(wavelength_range=bounds, mode='test')
    assert calls == ['test']


@pytest.mark.parametrize('bounds', [(1.,100.), (100000.,200000.)])
def test_disjoint_passband_still_rejected(bounds):
    actual = stellar(CachedStellarLimbDarkening)
    with pytest.raises(ValueError, match='no overlap'):
        actual.compute_quadratic_ld_coeffs(wavelength_range=bounds,mode='custom',
            custom_wavelengths=np.linspace(6000.,53000.,50),custom_throughput=np.ones(50))
