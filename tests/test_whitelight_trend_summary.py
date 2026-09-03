import numpy as np
import xarray as xr

from tools.diag_whitelight.summarize_trend_matrix import (
    parity,
    top_abs_correlations,
    two_spot_label_diagnostics,
)


def _run(median_shift=0.0, sigma=1.0):
    return {
        "geometry": {
            name: {"median": median_shift, "sigma": sigma}
            for name in ("t0_0", "b_0", "duration_0", "rors_0")
        }
    }


def test_geometry_parity_applies_shift_and_width_gates():
    result = parity(_run(median_shift=0.1, sigma=1.1), _run())
    assert result["passed"]
    assert result["rows"]["rors_0"]["sigma_ratio"] == 1.1

    assert not parity(_run(median_shift=0.14), _run())["passed"]
    assert not parity(_run(sigma=1.3), _run())["passed"]


def test_top_correlations_are_ranked_by_absolute_value():
    x = np.arange(8.0)
    posterior = xr.Dataset(
        {
            "c": (("chain", "draw"), x.reshape(1, -1)),
            "v": (("chain", "draw"), (-x).reshape(1, -1)),
            "jump": (("chain", "draw"), np.array([0, 1, 0, 1, 0, 1, 0, 1]).reshape(1, -1)),
        }
    )
    rows = top_abs_correlations(posterior)
    assert rows[0]["left"] == "c"
    assert rows[0]["right"] == "v"
    assert rows[0]["correlation"] == -1.0
    assert abs(rows[0]["correlation"]) >= abs(rows[-1]["correlation"])


def test_two_spot_label_diagnostics_count_mirrored_draws():
    posterior = xr.Dataset(
        {
            "spot_mu": (("chain", "draw"), np.array([[1.0, 2.0, 4.0]])),
            "spot_mu2": (("chain", "draw"), np.array([[2.0, 1.0, 5.0]])),
        }
    )
    result = two_spot_label_diagnostics(posterior)
    assert result["num_draws"] == 3
    assert result["fraction_spot_mu_less_than_spot_mu2"] == 2.0 / 3.0
    assert result["num_order_violations"] == 1
