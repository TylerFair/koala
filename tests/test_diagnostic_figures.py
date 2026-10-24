"""Presentation preserves the native arrays and puts useful labels outside the data."""
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from koala import diagnostic_figures as figures


def test_white_light_panels_titles_and_display_bins(tmp_path):
    time = 60000 + np.linspace(-.1, .1, 101)
    transit = 1-.01*np.exp(-((time-60000)/.03)**2)
    trend = 1+.001*(time-time[0])
    flux = transit*trend + .0001*np.sin(np.arange(len(time)))
    error = np.full(len(time), .0002)
    mask = np.zeros(len(time), bool); mask[4] = True
    before = [a.copy() for a in (time, flux, error, transit, trend)]
    fig = figures.white_light(time, flux, error, transit*trend, transit, tmp_path/'white.png',
                              label='test_NIRISS_SOSS_order1', t0=60000., outlier=mask)
    assert len(fig.axes) == 3
    assert 'White light' in fig.axes[0].get_title()
    assert not fig.axes[0].texts
    assert any(line.get_marker() == 'x' for line in fig.axes[0].lines)
    assert fig.axes[0].get_ylim()[1] > max(flux)
    np.testing.assert_allclose(fig.axes[-1].lines[-1].get_xdata(), (time-60000)*24)
    for actual, expected in zip((time, flux, error, transit, trend), before):
        np.testing.assert_array_equal(actual, expected)


def test_residual_map_robust_symmetric_scale_and_multi_planet_spectrum(tmp_path):
    time = np.linspace(0, 1, 21)
    residual = np.random.default_rng(4).normal(0, .0001, (3, len(time)))
    residual[0, 0] = 1.
    original = residual.copy()
    fig = figures.residual_map(time, [1., 1.5, 2.], residual, tmp_path/'residual.png',
                              label='test_NIRISS_SOSS_order1', t0=.5)
    lower, upper = fig.axes[0].collections[0].get_clim()
    assert lower == -upper and upper < 1e4
    np.testing.assert_array_equal(residual, original)
    samples = np.random.default_rng(9).normal(.1, .001, (100, 3, 2))
    spectrum = figures.spectrum([1., 1.5, 2.], samples, tmp_path/'spectrum.png',
                               label='test_NIRISS_SOSS_order1')
    assert len(spectrum.axes) == 2
    assert {path.name for path in tmp_path.iterdir()} == {'residual.png', 'spectrum.png'}


def test_expected_mid_eclipse_is_shifted_for_display_only():
    assert figures.event_epoch([60000.], [4.], 'eclipse') == 60002.
    assert figures.event_epoch([60000.], [4.], 'transit') == 60000.


def test_emission_waterfall_separates_curves_and_keeps_wavelength_labels_outside(tmp_path):
    time = np.linspace(-.1, .1, 101)
    signal = 1+.001*(1-np.exp(-(time/.02)**4))
    model = np.tile(signal, (2, 1))
    flux = model+np.random.default_rng(12).normal(0, .0001, model.shape)
    fig = figures.channel_diagnostics(time, flux, model, [3., 4.], np.ones_like(model),
        tmp_path/'emission_channels.png', label='test_MIRI_LRS', t0=0., event='eclipse')
    curves = [line.get_ydata() for line in fig.axes[0].lines if line.get_color() == 'k']
    assert len(curves) == 2 and min(curves[0]) > max(curves[1])
    assert all(text.get_position()[0] > 1 for text in fig.axes[0].texts)
    assert fig.axes[0].get_ylim()[1] > flux.max()
    assert 'eclipse' in fig.axes[0].get_xlabel()
    # The figure has been saved and closed, which detaches its canvas in Matplotlib 3.11.
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    for ax in fig.axes:
        bounds = ax.xaxis.label.get_window_extent(canvas.get_renderer())
        assert 0 <= bounds.x0 < bounds.x1 <= fig.bbox.x1
