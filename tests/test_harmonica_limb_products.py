import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

from fit_jwst import (
    HARMONICA_LIMB_PRODUCT_CONVENTION,
    HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION,
    _harmonica_limb_product_samples,
    _save_harmonica_limb_posterior_samples,
    build_harmonica_limb_dataframe,
)


def test_first_order_limb_products_are_half_area_equivalent_depths():
    a0 = np.array([[0.10, 0.20], [0.11, 0.19]])
    a1 = np.array([[0.01, -0.02], [0.015, -0.01]])

    products = _harmonica_limb_product_samples(a0, {"a1": a1})

    total = a0**2 + 0.5 * a1**2
    contrast = (4.0 / np.pi) * a0 * a1
    np.testing.assert_allclose(products["depth_evening"], total + contrast)
    np.testing.assert_allclose(products["depth_morning"], total - contrast)
    np.testing.assert_allclose(products["depth_total_area"], total)
    np.testing.assert_allclose(products["depth_leading_endpoint"], (a0 + a1) ** 2)
    np.testing.assert_allclose(products["depth_trailing_endpoint"], (a0 - a1) ** 2)
    np.testing.assert_allclose(products["asymmetry_coefficient"], a1)
    np.testing.assert_allclose(products["endpoint_delta_r"], 2.0 * a1)


def test_higher_odd_order_formula_matches_direct_half_area_integration():
    a0 = np.array([[0.10]])
    coefficients = {
        "a1": np.array([[0.008]]),
        "a3": np.array([[-0.003]]),
        "a5": np.array([[0.0015]]),
    }
    products = _harmonica_limb_product_samples(a0, coefficients)

    nodes, weights = np.polynomial.legendre.leggauss(256)

    def equivalent_half_depth(theta_midpoint):
        theta = theta_midpoint + 0.5 * np.pi * nodes
        radius = np.full_like(theta, a0.item())
        for name, coefficient in coefficients.items():
            radius += coefficient.item() * np.cos(int(name[1:]) * theta)
        # The interval transform contributes pi/2, and division by pi turns
        # each half-area into its equivalent circular transit depth.
        return 0.5 * np.sum(weights * radius**2)

    expected_leading = equivalent_half_depth(0.0)
    expected_trailing = equivalent_half_depth(np.pi)
    np.testing.assert_allclose(
        products["depth_evening"].item(), expected_leading, rtol=2e-14
    )
    np.testing.assert_allclose(
        products["depth_morning"].item(), expected_trailing, rtol=2e-14
    )
    np.testing.assert_allclose(
        products["depth_total_area"].item(),
        0.5 * (expected_leading + expected_trailing),
        rtol=2e-14,
    )


def test_limb_dataframe_schema_exposes_half_area_and_endpoint_products():
    a0_one_draw = np.array([0.10, 0.12])
    a1_one_draw = np.array([0.01, -0.02])
    a0 = np.repeat(a0_one_draw[None, :], 5, axis=0)
    a1 = np.repeat(a1_one_draw[None, :], 5, axis=0)

    limb_df, _, _ = build_harmonica_limb_dataframe(
        wavelengths=np.array([1.1, 1.2]),
        wavelength_err=0.01,
        rors_samples=a0,
        harmonic_samples={"a1": a1},
    )

    total = a0_one_draw**2 + 0.5 * a1_one_draw**2
    contrast = (4.0 / np.pi) * a0_one_draw * a1_one_draw
    assert (limb_df["limb_product_schema_version"] == 2).all()
    assert HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION == 2
    assert (
        limb_df["limb_product_convention"] == HARMONICA_LIMB_PRODUCT_CONVENTION
    ).all()
    np.testing.assert_allclose(
        limb_df["depth_evening_median"], (total + contrast) * 1e6
    )
    np.testing.assert_allclose(
        limb_df["depth_morning_median"], (total - contrast) * 1e6
    )
    np.testing.assert_allclose(limb_df["depth_total_area_median"], total * 1e6)
    np.testing.assert_allclose(
        limb_df["depth_leading_endpoint_median"], (a0_one_draw + a1_one_draw) ** 2 * 1e6
    )
    np.testing.assert_allclose(
        limb_df["depth_trailing_endpoint_median"], (a0_one_draw - a1_one_draw) ** 2 * 1e6
    )
    np.testing.assert_allclose(limb_df["asymmetry_coefficient_median"], a1_one_draw)
    np.testing.assert_allclose(limb_df["endpoint_delta_r_median"], 2.0 * a1_one_draw)
    np.testing.assert_allclose(limb_df["depth_morning_err_lo"], 0.0)
    np.testing.assert_allclose(limb_df["depth_morning_err_hi"], 0.0)


def test_joint_limb_posterior_archive_preserves_drawwise_covariance(tmp_path):
    wavelength = np.array([1.1, 1.2])
    a0 = np.array([[0.10, 0.12], [0.11, 0.11], [0.12, 0.10]])
    a1 = np.array([[0.010, -0.005], [0.0, 0.0], [-0.010, 0.005]])
    path = tmp_path / "limb_posterior_samples.npz"

    _save_harmonica_limb_posterior_samples(
        path,
        wavelength,
        0.01,
        a0,
        {"a1": a1},
    )

    with np.load(path, allow_pickle=False) as saved:
        assert saved["sample_axes"].item() == "draw,wavelength"
        assert saved["depth_units"].item() == "fractional_stellar_area"
        assert saved["limb_product_schema_version"].item() == 2
        np.testing.assert_array_equal(saved["a0"], a0)
        np.testing.assert_array_equal(saved["a1"], a1)
        expected = _harmonica_limb_product_samples(a0, {"a1": a1})
        np.testing.assert_allclose(saved["depth_evening"], expected["depth_evening"])
        np.testing.assert_allclose(saved["depth_morning"], expected["depth_morning"])
        covariance = np.cov(
            saved["depth_evening"][:, 0], saved["depth_morning"][:, 0]
        )[0, 1]
        expected_covariance = np.cov(
            expected["depth_evening"][:, 0], expected["depth_morning"][:, 0]
        )[0, 1]
        np.testing.assert_allclose(covariance, expected_covariance)
