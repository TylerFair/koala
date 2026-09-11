"""The white-light corner plot shows the sampled scalar sites only."""

import numpy as np
import pytest

pytest.importorskip("corner")
import matplotlib
matplotlib.use("Agg")

from plotting import _corner_columns, plot_whitelight_corner


def _samples(n=400):
    rng = np.random.default_rng(1)
    return {
        "t0_0": 59787.05 + rng.normal(0, 1e-4, n),
        "rors_0": 0.145 + rng.normal(0, 1e-3, n),
        "b": 0.45 + rng.normal(0, 0.02, n),
        "duration": 0.117 + rng.normal(0, 1e-3, n),
        "u": np.column_stack([0.3 + rng.normal(0, 0.05, n), 0.2 + rng.normal(0, 0.05, n)]),
        "c": 1.0 + rng.normal(0, 1e-4, n),
        "v": rng.normal(0, 1e-3, n),
        "error": 3e-4 + rng.normal(0, 1e-5, n),
        "width_minutes": np.full(n, 5.0),      # derived duplicate: excluded
        "total_error": rng.normal(size=(n, 3)),  # per-channel site: excluded
        "_transit_phase_mask": np.ones(n),     # bookkeeping: excluded
        "constant": np.full(n, 2.0),           # degenerate: dropped
        # Latent / derived sites that must not appear:
        "cos_i_0": rng.normal(0.04, 0.001, n),
        "delta": rng.normal(0.04, 0.001, n),
        "log_jitter": rng.normal(-8, 0.1, n),
        "ld_uplus_uminus1": rng.normal(0.4, 0.05, n),
        "depths_0": rng.normal(0.021, 1e-4, n),
        "inc_0": rng.normal(1.53, 0.001, n),
        "a_rs": rng.normal(11.4, 0.1, n),      # duration present: dropped
    }


def test_corner_columns_select_and_order_scalar_sites():
    columns, labels = _corner_columns(_samples())
    assert len(columns) == len(labels) == 9
    assert labels[0].startswith("$t_0$")
    assert "$-$ 59787 [d]" in labels[0]
    assert labels[1].startswith(r"$R_p/R_\star$")
    assert r"$u_1$" in labels and r"$u_2$" in labels
    assert all(np.isfinite(c).all() and np.std(c) > 0 for c in columns)
    assert abs(np.median(columns[0]) - 0.05) < 1e-3


def test_plot_whitelight_corner_writes_png(tmp_path):
    out = tmp_path / "13_test_whitelight_corner.png"
    path = plot_whitelight_corner(_samples(), str(out), instrument_label="TEST_NIRISS_SOSS_order1")
    assert path == str(out)
    assert out.stat().st_size > 10_000


def test_plot_whitelight_corner_skips_degenerate_input(tmp_path):
    out = tmp_path / "corner.png"
    assert plot_whitelight_corner({"c": np.ones(50)}, str(out)) is None
    assert not out.exists()


def test_corner_excludes_fixed_nonbinary_period():
    samples = _samples(n=1000)
    samples['period'] = np.full(1000, 2.6162644)
    columns, labels = _corner_columns(samples)
    assert len(columns) == 9
    assert labels[0].startswith('$t_0$')
