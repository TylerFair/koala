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


def test_single_dataset_gets_its_own_figure_but_no_joint_one(tmp_path):
    _member(tmp_path, 'soss_order1', [1.0, 1.5])
    figure = plot_joint_spectra(tmp_path)
    assert figure == tmp_path / 'joint' / 'datasets' / 'soss_order1.png' and figure.exists()
    assert (tmp_path / 'soss_order1' / 'soss_order1_Rreference_depth_and_t0.png').exists()
    assert not (tmp_path / 'joint' / 'joint_spectrum.png').exists()


def test_miri_beyond_ten_microns_is_kept_in_the_table_but_not_drawn(tmp_path):
    _member(tmp_path, 'soss_order1', [1.0, 1.5, 2.0])
    _member(tmp_path, 'miri', [6.0, 9.0, 11.0])
    for path in (tmp_path / 'miri').glob('*.csv'):
        path.rename(path.with_name('WASP-193_MIRI_LRS_pix6_time_shift.csv'))
    plot_joint_spectra(tmp_path)
    table = pd.read_csv(tmp_path / 'joint' / 'joint_spectrum.csv')
    assert len(table) == 6
    assert list(table.loc[table.instrument == 'MIRI', 'shown']) == [True, True, False]


def test_display_binning_reaches_target_resolution(tmp_path):
    from koala.joint_products import collect_joint_spectra, display_bin
    edges = np.geomspace(3.0, 5.0, 301)
    member = tmp_path / 'g395h_nrs1'
    member.mkdir()
    pd.DataFrame({'wavelength': 0.5 * (edges[1:] + edges[:-1]), 'wavelength_err': 0.5 * np.diff(edges),
                  'wave_lo_um': edges[:-1], 'wave_hi_um': edges[1:], 'planet': 0,
                  'depth': .02, 'depth_err': 1e-4, 'dt0_s': 0., 't0_err_s': 5.}).to_csv(
        member / 'WASP-107_NIRSPEC_G395H_nrs1_Rreference_time_shift.csv', index=False)
    binned = display_bin(collect_joint_spectra(tmp_path))
    resolution = binned.wavelength / (binned.wave_hi_um - binned.wave_lo_um)
    assert abs(np.median(resolution) - 100) < 15
    np.testing.assert_allclose(binned.t0_err_s, 5 / np.sqrt(binned.channels))


def test_long_wavelength_range_uses_log_axis_and_marks_clipped_points(tmp_path):
    _member(tmp_path, 'soss_order1', list(np.linspace(.9, 2.8, 60)))
    _member(tmp_path, 'miri', [6., 8., 11.5])
    path = tmp_path / 'miri' / 'miri_Rreference_time_shift.csv'
    t = pd.read_csv(path); t.loc[2, 'dt0_s'] = -900.; t.to_csv(path, index=False)
    assert plot_joint_spectra(tmp_path).exists()
