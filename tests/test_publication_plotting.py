import builtins
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting
from plotting import (
    plot_harmonica_limb_spectra,
    plot_harmonica_transmission_posterior,
    plot_harmonica_transmission_strings,
    plot_map_fits,
    plot_map_residuals,
    plot_noise_binning_from_csv,
    plot_spectrum_precision,
    plot_transmission_spectrum,
    plot_wavelength_offset_summary,
    plot_whitelight_curve,
    plot_whitelight_residuals,
    plot_whitelight_summary,
)


def _assert_png(path):
    assert path.is_file()
    assert path.stat().st_size > 1_000


def test_style_module_degrades_without_optional_packages(monkeypatch):
    real_import = builtins.__import__

    def import_without_extras(name, *args, **kwargs):
        if name == "scienceplots" or name.startswith("cmcrameri"):
            raise ImportError(f"blocked optional package {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_extras)
    module_path = Path(plotting.__file__).with_name("plotting_style.py")
    spec = importlib.util.spec_from_file_location("plotting_style_no_extras", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.apply_publication_style()
    assert module.scienceplots is None
    assert module.cmc is None
    assert module.accent_color_for_label("NIRISS_SOSS_order1") == "mediumorchid"
    assert len(module.model_palette_colors(3)) == 3
    assert plt.rcParams["axes.labelsize"] == 18.0


def test_white_light_plot_entry_points_write_files(tmp_path):
    time = np.linspace(100.0, 100.12, 35)
    model = 1.0 - 0.01 * np.exp(-0.5 * ((time - 100.06) / 0.018) ** 2)
    flux = model + 7e-5 * np.sin(np.linspace(0.0, 5.0, time.size))
    error = np.full(time.size, 9e-5)
    outlier = np.zeros(time.size, dtype=bool)
    outlier[3] = True

    curve_path = tmp_path / "white_curve.png"
    residual_path = tmp_path / "white_residual.png"
    summary_path = tmp_path / "white_summary.png"
    curve_figure = plot_whitelight_curve(
        time,
        flux,
        float(error[0]),
        model,
        curve_path,
        instrument_label="NIRISS_SOSS_order1",
        t0_reference=100.06,
        outlier_mask=outlier,
    )
    plot_whitelight_residuals(
        time,
        flux - model,
        error,
        residual_path,
        instrument_label="NIRISS_SOSS_order1",
        t0_reference=100.06,
        outlier_mask=outlier,
    )
    plot_whitelight_summary(
        {
            "period": np.array([3.5]),
            "duration": np.array([0.11]),
            "t0": np.array([100.06]),
            "b": np.array([0.42]),
            "rors": np.array([0.10]),
            "depths": np.array([0.01]),
        },
        summary_path,
        instrument_label="NIRISS_SOSS_order1",
    )
    for path in (curve_path, residual_path, summary_path):
        _assert_png(path)
    data_lines = [
        line
        for line in curve_figure.axes[0].lines
        if line.get_marker() == "o" and line.get_linestyle() == "None"
    ]
    assert data_lines, "white-light data markers must remain visible"
    assert len(curve_figure.axes[0].texts) == 1


def test_spectrum_and_noise_entry_points_write_files(tmp_path):
    wavelength = np.linspace(1.0, 2.8, 12)
    depth = 10200.0 + 180.0 * np.sin(wavelength * 3.0)
    precision = np.linspace(55.0, 95.0, wavelength.size)
    spectrum_path = tmp_path / "spectrum_precision.png"
    plot_spectrum_precision(
        wavelength,
        depth,
        precision,
        spectrum_path,
        instrument_label="NIRISS_SOSS_order1",
    )
    _assert_png(spectrum_path)

    rng = np.random.default_rng(42)
    rors = np.sqrt(
        np.maximum(
            rng.normal(depth / 1e6, precision / 1e6, size=(40, wavelength.size)),
            1e-8,
        )
    )
    plot_transmission_spectrum(wavelength, rors, str(tmp_path / "spectrum"))
    _assert_png(tmp_path / "spectrum_00.png")

    bins = np.array([1, 2, 4, 8, 16])
    noise = pd.DataFrame(
        {
            "bin_size_points": bins,
            "measured_rms": 180.0 / np.sqrt(bins) * 1.08,
            "measured_rms_p16": 180.0 / np.sqrt(bins) * 0.95,
            "measured_rms_p84": 180.0 / np.sqrt(bins) * 1.20,
            "expected_white_rms": 180.0 / np.sqrt(bins),
        }
    )
    noise_path = tmp_path / "noise.png"
    plot_noise_binning_from_csv(
        noise, noise_path, instrument_label="NIRSpec_G395M_NRS1"
    )
    _assert_png(noise_path)


def test_map_plot_entry_points_write_files(tmp_path, monkeypatch):
    time = np.linspace(-0.06, 0.06, 25)
    wavelength = np.linspace(2.9, 5.0, 4)
    transit = -0.01 * np.exp(-0.5 * (time / 0.018) ** 2)
    map_params = {
        "t0": np.array([0.0]),
        "b": np.array([0.4]),
        "rors": np.full((4, 1), 0.10),
        "c": np.ones(4),
        "v": np.zeros(4),
    }
    transit_params = {"period": np.array([3.2])}
    data = np.tile(1.0 + transit, (4, 1))
    data += np.linspace(-5e-5, 5e-5, time.size)[None, :]
    monkeypatch.setattr(
        plotting,
        "_single_curve_transit_signal",
        lambda t, params, transit_params, index: transit,
    )

    fits_path = tmp_path / "map_fits.png"
    residuals_path = tmp_path / "map_residuals.png"
    summary_path = tmp_path / "offset_summary.png"
    plot_map_fits(
        time,
        data,
        np.full(4, 8e-5),
        wavelength,
        map_params,
        transit_params,
        fits_path,
        ncols=2,
    )
    plot_map_residuals(
        time,
        data,
        np.full(4, 8e-5),
        wavelength,
        map_params,
        transit_params,
        residuals_path,
        ncols=2,
    )
    plot_wavelength_offset_summary(
        time,
        data,
        np.full(4, 8e-5),
        wavelength,
        map_params,
        transit_params,
        summary_path,
    )
    for path in (fits_path, residuals_path, summary_path):
        _assert_png(path)


def _limb_dataframe():
    wavelength = np.linspace(0.9, 2.8, 10)
    base = 10300.0 + 120.0 * np.sin(wavelength * 2.5)
    frame = {
        "wavelength": wavelength,
        "wavelength_err": np.full(wavelength.size, 0.035),
    }
    for name, values in (
        ("depth_total_area", base),
        ("depth_one", base - 90.0),
        ("depth_two", base + 90.0),
    ):
        frame[f"{name}_median"] = values
        frame[f"{name}_err_lo"] = np.full(wavelength.size, 75.0)
        frame[f"{name}_err_hi"] = np.full(wavelength.size, 85.0)
    return pd.DataFrame(frame)


def test_harmonica_plot_entry_points_write_files(tmp_path):
    limb_path = tmp_path / "limbs.png"
    limb_figure = plot_harmonica_limb_spectra(
        _limb_dataframe(), limb_path, instrument_label="NIRISS_SOSS_order1"
    )
    _assert_png(limb_path)
    assert len(limb_figure.axes) == 3
    total_axis, one_axis, two_axis = limb_figure.axes
    assert one_axis.get_title() == "Terminator One"
    assert two_axis.get_title() == "Terminator Two"
    assert [axis.texts[0].get_text() for axis in limb_figure.axes] == ["a", "b", "c"]
    assert one_axis.get_shared_y_axes().joined(one_axis, two_axis)
    assert not any(label.get_visible() for label in two_axis.get_yticklabels())

    theta = np.linspace(-np.pi, np.pi, 120)
    curves = np.vstack(
        [0.10 + amplitude * np.cos(theta) for amplitude in (0.001, 0.002, 0.003)]
    )
    strings_path = tmp_path / "strings.png"
    posterior_path = tmp_path / "strings_posterior.png"
    plot_harmonica_transmission_strings(
        theta,
        curves,
        0.10,
        strings_path,
        instrument_label="NIRISS_SOSS_order1",
    )
    plot_harmonica_transmission_posterior(
        theta,
        curves,
        np.median(curves, axis=0),
        0.10,
        posterior_path,
        instrument_label="NIRISS_SOSS_order1",
    )
    _assert_png(strings_path)
    _assert_png(posterior_path)
