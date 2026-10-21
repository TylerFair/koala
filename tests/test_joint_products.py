"""The combined depth and timing figure of a joint run."""
import shutil

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


def test_miri_axis_is_linear_to_five_microns_and_compressed_beyond():
    from koala.joint_products import _hybrid_forward, _hybrid_inverse
    wavelength = np.array([0.6, 2.0, 4.99, 5.0, 6.0, 10.0])
    position = _hybrid_forward(wavelength)
    np.testing.assert_allclose(position[:4], wavelength[:4])
    np.testing.assert_allclose(_hybrid_inverse(position), wavelength)
    # MIRI/LRS takes about a fifth of a 0.6-10 um axis.
    assert 0.15 < (_hybrid_forward(10.0) - 5.0) / (_hybrid_forward(10.0) - 0.6) < 0.25


def _limb_table(dt0_s, k=0.12, coefficient=2500.0):
    """Channels with the analytic half-disk responses and tiny errors."""
    n = len(dt0_s)
    return pd.DataFrame({
        'dataset': 'g395h_nrs1', 'instrument': 'NIRSpec', 'label': 'NIRSpec G395H nrs1',
        'wavelength': np.linspace(3, 5, n), 'wavelength_err': np.full(n, .01), 'planet': 0,
        'depth': np.full(n, k**2), 'depth_err': np.full(n, 1e-9),
        'dt0_s': np.asarray(dt0_s, float), 't0_err_s': np.full(n, 1e-6), 'corr_depth_t0': 0.,
        'd_depth_dRL': k, 'd_depth_dRT': k, 'd_dt0_dRL_s_per_Rstar': -coefficient,
        'd_dt0_dRT_s_per_Rstar': coefficient,
    })


def test_limb_depths_recover_the_two_halves():
    from koala.joint_products import limb_spectra
    k, coefficient, half_difference = 0.12, 2500.0, 0.002
    # A planet with R_T = k + e and R_L = k - e has the same depth to first order and a later transit.
    limbs = limb_spectra(_limb_table([0., 2 * coefficient * half_difference, -2 * coefficient * half_difference]))
    np.testing.assert_allclose(limbs.depth_evening_ppm, 1e6 * np.array([k, k + half_difference, k - half_difference])**2)
    np.testing.assert_allclose(limbs.depth_morning_ppm, 1e6 * np.array([k, k - half_difference, k + half_difference])**2)
    np.testing.assert_allclose(limbs.radius_evening_minus_morning_Rstar, [0, 2 * half_difference, -2 * half_difference],
                               atol=1e-9)


def test_visit_offset_moves_the_limbs_apart_uniformly():
    from koala.joint_products import limb_spectra
    table = _limb_table([0., 5., 10.])
    shifted = limb_spectra(table, offsets={'g395h_nrs1': 5.})
    np.testing.assert_allclose(shifted.radius_evening_minus_morning_Rstar,
                               limb_spectra(table).radius_evening_minus_morning_Rstar - 5. / 2500., atol=1e-9)


def test_limb_figure_is_written_only_with_radius_responses(tmp_path):
    _member(tmp_path, 'soss_order1', [1.0, 1.5])
    plot_joint_spectra(tmp_path)
    assert not (tmp_path / 'joint' / 'limb_spectra.png').exists()
    member = tmp_path / 'g395h_nrs1'
    member.mkdir()
    _limb_table(np.linspace(-20, 20, 40)).drop(columns=['dataset', 'instrument', 'label']).to_csv(
        member / 'WASP-107_NIRSPEC_G395H_nrs1_Rreference_time_shift.csv', index=False)
    plot_joint_spectra(tmp_path)
    for stem in ('limb_spectra', 'limb_spectra_binned'):
        assert (tmp_path / 'joint' / f'{stem}.png').exists()
    saved = pd.read_csv(tmp_path / 'joint' / 'limb_spectra.csv')
    assert saved.depth_evening_ppm.notna().sum() == 40 and saved.depth_morning_ppm.isna().sum() == 2


def test_limb_errors_are_anticorrelated_when_timing_dominates():
    from koala.joint_products import limb_spectra
    table = _limb_table([0., 0.]).assign(depth_err=1e-5, t0_err_s=20.)
    corr = limb_spectra(table).corr_morning_evening
    assert np.all(corr < -0.9)


def _visit(name, start, stop, epoch, depth=.02, dt0=0., depth_err=1e-4, t0_err_s=4., n=40):
    """A NIRSpec dataset of one visit; channels at R ~ 200 between start and stop (um)."""
    edges = np.geomspace(start, stop, n + 1)
    return pd.DataFrame({
        'dataset': name, 'instrument': 'NIRSpec', 'label': name, 'wavelength': 0.5 * (edges[1:] + edges[:-1]),
        'wavelength_err': 0.5 * np.diff(edges), 'wave_lo_um': edges[:-1], 'wave_hi_um': edges[1:], 'planet': 0,
        'depth': depth, 'depth_err': depth_err, 'dt0_s': dt0, 't0_err_s': t0_err_s,
        't0_bjd': epoch + dt0 / 86400.0, 'corr_depth_t0': 0.,
        'd_depth_dRL': .12, 'd_depth_dRT': .12, 'd_dt0_dRL_s_per_Rstar': -2500., 'd_dt0_dRT_s_per_Rstar': 2500.,
    })


def test_ivw_lines_up_visits_before_averaging_them():
    from koala.joint_products import ivw_combine
    # The second visit reads 200 ppm deeper and 6 s later everywhere; the spectra are otherwise flat.
    table = pd.concat([_visit('a', 1.0, 2.0, 2460000.0, n=140),
                       _visit('b', 1.5, 2.5, 2460100.0, depth=.0202, dt0=6., n=100)], ignore_index=True)
    combined, offsets = ivw_combine(table)
    offsets = offsets.set_index('visit')
    np.testing.assert_allclose(offsets.depth_offset_ppm['b'] - offsets.depth_offset_ppm['a'], 200., atol=1e-6)
    np.testing.assert_allclose(offsets.t0_offset_s['b'] - offsets.t0_offset_s['a'], 6., atol=1e-6)
    np.testing.assert_allclose(offsets.depth_overlap_chi2, 0., atol=1e-12)
    # No step where the coverage changes, and the zero sits between the two visits.
    np.testing.assert_allclose(combined.depth, combined.depth.iloc[0], atol=1e-12)
    np.testing.assert_allclose(combined.dt0_s, combined.dt0_s.iloc[0], atol=1e-9)
    assert 0. < combined.dt0_s.iloc[0] < 6.
    # Inverse-variance errors from all channels of both visits in a bin.
    np.testing.assert_allclose(combined.t0_err_s, 4. / np.sqrt(combined.channels))
    assert set(combined.visits) == {1, 2}
    resolution = combined.wavelength / (combined.wave_hi_um - combined.wave_lo_um)
    assert abs(np.median(resolution) - 100) < 2


def test_ivw_limb_asymmetry_tightens_with_more_visits():
    from koala.joint_products import ivw_combine, limb_spectra
    one, _ = ivw_combine(pd.concat([_visit('a', 3., 4., 2460000.), _visit('b', 4.2, 5., 2460100.)]))
    assert one is None
    four = pd.concat([_visit(name, 3., 4., 2460000. + 100 * k, t0_err_s=20.) for k, name in enumerate('abcd')])
    combined, _ = ivw_combine(four)
    single, _ = ivw_combine(pd.concat([four[four.dataset == 'a'],
                                       four[four.dataset == 'b'].assign(t0_err_s=2e4, depth_err=1.)]))
    ratio = (limb_spectra(combined).radius_evening_minus_morning_err_Rstar /
             limb_spectra(single).radius_evening_minus_morning_err_Rstar)
    np.testing.assert_allclose(np.median(ratio), 0.5, rtol=0.05)


def test_ivw_figures_are_written_only_for_overlapping_visits(tmp_path):
    for name, (start, stop), epoch in (('g235h_nrs2', (2.3, 3.1), 2460000.), ('prism_nrs1', (2.0, 5.0), 2460100.)):
        member = tmp_path / name
        member.mkdir()
        _visit(name, start, stop, epoch).drop(columns=['dataset', 'instrument', 'label']).to_csv(
            member / f'WASP-39_NIRSPEC_{name}_Rreference_time_shift.csv', index=False)
    plot_joint_spectra(tmp_path)
    for stem in ('joint_spectrum_ivw', 'limb_spectra_ivw'):
        assert (tmp_path / 'joint' / f'{stem}.png').exists()
    assert len(pd.read_csv(tmp_path / 'joint' / 'ivw_offsets.csv')) == 2
    # Without shared wavelengths there is nothing to combine.
    shutil.rmtree(tmp_path / 'prism_nrs1')
    shutil.rmtree(tmp_path / 'joint')
    # Two detectors of one visit that touch at their ends do not count as an overlap.
    (tmp_path / 'g235h_nrs1').mkdir()
    _visit('g235h_nrs1', 1.7, 2.31, 2460000.).drop(columns=['dataset', 'instrument', 'label']).to_csv(
        tmp_path / 'g235h_nrs1' / 'WASP-39_NIRSPEC_g235h_nrs1_Rreference_time_shift.csv', index=False)
    (tmp_path / 'g395h_nrs1').mkdir()
    _visit('g395h_nrs1', 3.2, 3.7, 2460100.).drop(columns=['dataset', 'instrument', 'label']).to_csv(
        tmp_path / 'g395h_nrs1' / 'WASP-39_NIRSPEC_g395h_nrs1_Rreference_time_shift.csv', index=False)
    plot_joint_spectra(tmp_path)
    assert not (tmp_path / 'joint' / 'joint_spectrum_ivw.png').exists()
