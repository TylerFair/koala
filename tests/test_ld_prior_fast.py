"""The fast power-2 prior builder must reproduce exotic_ld's per-call numbers."""

import os

import numpy as np
import pytest

exotic_ld = pytest.importorskip("exotic_ld")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LD_DATA = os.path.join(REPO, "exotic_ld_data")
NODE = os.path.join(LD_DATA, "stagger", "MH0.0", "teff5000", "logg4.5", "stagger_spectra.dat")

pytestmark = pytest.mark.skipif(
    not os.path.exists(NODE), reason="local Stagger grid not available"
)


def _ranges(n=6):
    centres = np.linspace(3.0e4, 4.9e4, n)
    return np.column_stack([centres - 40.0, centres + 40.0])


def test_node_weights_sum_to_one_and_match_interpolation_mode():
    from models.ld_prior_fast import stellar_node_weights

    w = stellar_node_weights(0.05, 5210.0, 4.3, "stagger", LD_DATA, "trilinear")
    assert abs(sum(v[3] for v in w.values()) - 1.0) < 1e-12
    nearest = stellar_node_weights(0.05, 5210.0, 4.3, "stagger", LD_DATA, "nearest")
    assert len(nearest) == 1 and list(nearest.values())[0][3] == 1.0


def test_batched_grid_matches_exotic_ld():
    from exotic_ld import StellarLimbDarkening
    from models.ld_prior_fast import build_power2_grid

    combos = [(0.0, 5000.0, 4.5), (0.05, 5210.0, 4.3), (-0.3, 4750.0, 4.6)]
    ranges = _ranges()
    coeffs, grad_norm = build_power2_grid(
        combos, ranges, "JWST_NIRSpec_G395H", "stagger", LD_DATA, log=lambda *a: None
    )
    assert coeffs.shape == (len(combos), len(ranges), 2)
    assert np.all(np.isfinite(coeffs))
    assert grad_norm.max() < 1e-8

    for ci, (mh, te, lg) in enumerate(combos):
        sld = StellarLimbDarkening(
            M_H=mh, Teff=te, logg=lg, ld_model="stagger", ld_data_path=LD_DATA,
            interpolate_type="trilinear", verbose=0,
        )
        for bi, (a, b) in enumerate(ranges):
            ref = sld.compute_power2_ld_coeffs(
                wavelength_range=[a, b], mode="JWST_NIRSpec_G395H", return_sigmas=False
            )
            np.testing.assert_allclose(coeffs[ci, bi], ref, atol=5e-7, rtol=0)


def test_integration_matches_exotic_ld_I_mu():
    from exotic_ld import StellarLimbDarkening
    from models.ld_prior_fast import _sensitivity, integrate_passband

    sld = StellarLimbDarkening(
        M_H=0.0, Teff=5000.0, logg=4.5, ld_model="stagger", ld_data_path=LD_DATA,
        interpolate_type="nearest", verbose=0,
    )
    s_wvs, s_thp = _sensitivity(LD_DATA, "JWST_NIRSpec_G395H", "")
    ranges = _ranges(3)
    ours = integrate_passband(sld.stellar_wavelengths, sld.stellar_intensities, s_wvs, s_thp, ranges)
    ours /= ours[:, :1]
    for bi, (a, b) in enumerate(ranges):
        sld._integrate_I_mu([a, b], "JWST_NIRSpec_G395H", None, None)
        np.testing.assert_allclose(ours[bi], sld.I_mu, rtol=1e-12, atol=1e-14)
