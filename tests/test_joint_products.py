"""The combined depth and timing figure of a joint run."""
import numpy as np
import pandas as pd

from koala.joint_products import collect_joint_spectra, plot_joint_spectra


def _member(directory, name, wavelengths):
    member = directory / name
    member.mkdir()
    pd.DataFrame({
        'wavelength': wavelengths, 'wavelength_err': np.full(len(wavelengths), .01),
        'planet': 0, 'depth': np.full(len(wavelengths), .02), 'depth_err': np.full(len(wavelengths), 5e-5),
        'dt0_s': np.linspace(-3, 3, len(wavelengths)), 't0_err_s': np.full(len(wavelengths), 4.),
    }).to_csv(member / f'{name}_Rreference_time_shift.csv', index=False)


def test_joint_spectrum_stacks_datasets_and_writes_figure(tmp_path):
    _member(tmp_path, 'soss_order1', [1.2, 0.9, 2.5])
    _member(tmp_path, 'g395h_nrs1', [3.1, 3.4])
    (tmp_path / 'joint').mkdir()
    table = collect_joint_spectra(tmp_path)
    assert list(table.wavelength) == sorted(table.wavelength)
    assert set(table.dataset) == {'soss_order1', 'g395h_nrs1'}
    figure = plot_joint_spectra(tmp_path)
    assert figure.exists() and (tmp_path / 'joint' / 'joint_spectrum.pdf').exists()
    saved = pd.read_csv(tmp_path / 'joint' / 'joint_spectrum.csv')
    np.testing.assert_allclose(saved.depth_ppm, 2e4)
    assert len(saved) == 5


def test_joint_spectrum_needs_two_datasets(tmp_path):
    _member(tmp_path, 'soss_order1', [1.0, 1.5])
    assert plot_joint_spectra(tmp_path) is None
    assert not (tmp_path / 'joint' / 'joint_spectrum.png').exists()


def test_long_wavelength_range_uses_log_axis_and_marks_clipped_points(tmp_path):
    _member(tmp_path, 'soss_order1', list(np.linspace(.9, 2.8, 60)))
    _member(tmp_path, 'miri', [6., 8., 11.5])
    path = tmp_path / 'miri' / 'miri_Rreference_time_shift.csv'
    t = pd.read_csv(path); t.loc[2, 'dt0_s'] = -900.; t.to_csv(path, index=False)
    assert plot_joint_spectra(tmp_path).exists()
