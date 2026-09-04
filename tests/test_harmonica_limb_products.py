import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pandas as pd

import fit_jwst

from fit_jwst import (
    HARMONICA_LIMB_PRODUCT_CONVENTION,
    HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION,
    _harmonica_limb_product_samples,
    _save_harmonica_limb_posterior_samples,
    build_harmonica_limb_dataframe,
    load_harmonica_limb_dataframe,
)


def test_first_order_limb_products_are_half_area_equivalent_depths():
    a0 = np.array([[0.10, 0.20], [0.11, 0.19]])
    a1 = np.array([[0.01, -0.02], [0.015, -0.01]])

    products = _harmonica_limb_product_samples(a0, {"a1": a1})

    total = a0**2 + 0.5 * a1**2
    contrast = (4.0 / np.pi) * a0 * a1
    np.testing.assert_allclose(products["depth_two"], total + contrast)
    np.testing.assert_allclose(products["depth_one"], total - contrast)
    np.testing.assert_allclose(products["rp_two"], np.sqrt(total + contrast))
    np.testing.assert_allclose(products["rp_one"], np.sqrt(total - contrast))
    np.testing.assert_allclose(products["depth_total_area"], total)
    np.testing.assert_allclose(products["depth_two_endpoint"], (a0 + a1) ** 2)
    np.testing.assert_allclose(products["depth_one_endpoint"], (a0 - a1) ** 2)
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

    expected_two = equivalent_half_depth(0.0)
    expected_one = equivalent_half_depth(np.pi)
    np.testing.assert_allclose(
        products["depth_two"].item(), expected_two, rtol=2e-14
    )
    np.testing.assert_allclose(
        products["depth_one"].item(), expected_one, rtol=2e-14
    )
    np.testing.assert_allclose(
        products["depth_total_area"].item(),
        0.5 * (expected_two + expected_one),
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
    assert (limb_df["limb_product_schema_version"] == 3).all()
    assert HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION == 3
    assert (
        limb_df["limb_product_convention"] == HARMONICA_LIMB_PRODUCT_CONVENTION
    ).all()
    np.testing.assert_allclose(
        limb_df["depth_two_median"], (total + contrast) * 1e6
    )
    np.testing.assert_allclose(
        limb_df["depth_one_median"], (total - contrast) * 1e6
    )
    np.testing.assert_allclose(limb_df["depth_total_area_median"], total * 1e6)
    np.testing.assert_allclose(
        limb_df["depth_two_endpoint_median"], (a0_one_draw + a1_one_draw) ** 2 * 1e6
    )
    np.testing.assert_allclose(
        limb_df["depth_one_endpoint_median"], (a0_one_draw - a1_one_draw) ** 2 * 1e6
    )
    np.testing.assert_allclose(limb_df["asymmetry_coefficient_median"], a1_one_draw)
    np.testing.assert_allclose(limb_df["endpoint_delta_r_median"], 2.0 * a1_one_draw)
    np.testing.assert_allclose(limb_df["depth_one_err_lo"], 0.0)
    np.testing.assert_allclose(limb_df["depth_one_err_hi"], 0.0)
    np.testing.assert_allclose(limb_df["rp_one_median"], np.sqrt(total - contrast))
    np.testing.assert_allclose(limb_df["rp_two_median"], np.sqrt(total + contrast))


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
        assert saved["limb_product_schema_version"].item() == 3
        np.testing.assert_array_equal(saved["a0"], a0)
        np.testing.assert_array_equal(saved["a1"], a1)
        expected = _harmonica_limb_product_samples(a0, {"a1": a1})
        np.testing.assert_allclose(saved["depth_two"], expected["depth_two"])
        np.testing.assert_allclose(saved["depth_one"], expected["depth_one"])
        covariance = np.cov(
            saved["depth_two"][:, 0], saved["depth_one"][:, 0]
        )[0, 1]
        expected_covariance = np.cov(
            expected["depth_two"][:, 0], expected["depth_one"][:, 0]
        )[0, 1]
        np.testing.assert_allclose(covariance, expected_covariance)


def test_schema_v3_csv_round_trips_with_neutral_indices(tmp_path):
    draws = np.linspace(0.095, 0.105, 12)[:, None]
    limb_df, _, _ = build_harmonica_limb_dataframe(
        wavelengths=np.array([1.25]),
        wavelength_err=np.array([0.03]),
        rors_samples=draws,
        harmonic_samples={"a1": np.full_like(draws, 0.002)},
    )
    path = tmp_path / "limb_v3.csv"
    limb_df.to_csv(path, index=False)

    loaded = load_harmonica_limb_dataframe(path)

    assert loaded["limb_product_schema_version"].unique().tolist() == [3]
    assert "depth_morning_median" not in loaded
    assert "depth_evening_median" not in loaded
    pd.testing.assert_frame_equal(loaded, limb_df, check_exact=False, rtol=1e-13)


def test_schema_v2_reader_uses_documented_index_mapping_once(tmp_path, caplog):
    legacy = pd.DataFrame(
        {
            "wavelength": [1.1],
            "wavelength_err": [0.02],
            "limb_product_schema_version": [2],
            "depth_morning_median": [10100.0],
            "depth_morning_err_lo": [80.0],
            "depth_morning_err_hi": [90.0],
            "depth_evening_median": [10300.0],
            "depth_evening_err_lo": [100.0],
            "depth_evening_err_hi": [110.0],
            "depth_total_area_median": [10200.0],
            "depth_total_area_err_lo": [60.0],
            "depth_total_area_err_hi": [70.0],
        }
    )
    path = tmp_path / "limb_v2.csv"
    legacy.to_csv(path, index=False)
    original = path.read_bytes()
    fit_jwst._LEGACY_LIMB_SCHEMA_WARNING_EMITTED = False

    with caplog.at_level("WARNING"):
        loaded_first = load_harmonica_limb_dataframe(path)
        loaded_second = load_harmonica_limb_dataframe(path)

    np.testing.assert_array_equal(
        loaded_first["depth_one_median"], legacy["depth_morning_median"]
    )
    np.testing.assert_array_equal(
        loaded_first["depth_two_median"], legacy["depth_evening_median"]
    )
    assert "rp_one_median" in loaded_first
    assert "rp_two_median" in loaded_second
    assert sum("legacy Harmonica limb-product schema v2" in record.message
               for record in caplog.records) == 1
    assert path.read_bytes() == original
