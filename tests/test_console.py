"""Console presentation must remain independent of inference settings."""
import io
import logging
import os

from koala import console


def test_verbosity_and_complete_appended_log(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv('KOALA_VERBOSE', raising=False)
    console.configure(False, tmp_path, 'example.yaml')
    console.debug('checkpoint details')
    console.info('Reading spectra')
    assert capsys.readouterr().out == 'Reading spectra\n'
    console.configure(True, tmp_path, 'second.yaml')
    console.debug('sampler details')
    assert 'sampler details' in capsys.readouterr().out
    log = (tmp_path / 'koala.log').read_text()
    assert all(text in log for text in ('example.yaml', 'second.yaml', 'checkpoint details', 'sampler details'))
    monkeypatch.setenv('KOALA_VERBOSE', '1')
    console.configure()
    console.debug('environment verbosity')
    assert 'environment verbosity' in capsys.readouterr().out


def test_non_tty_progress(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr('sys.stdout', stream)
    monkeypatch.delenv('TQDM_MININTERVAL', raising=False)
    console.configure()
    assert os.environ['TQDM_MININTERVAL'] == '30'
    with console.progress(total=2) as bar:
        assert bar.mininterval == 30
        bar.update(2)
    monkeypatch.setenv('TQDM_MININTERVAL', '60')
    console.configure()
    assert os.environ['TQDM_MININTERVAL'] == '60'


def test_channel_batches_include_restored_batches(monkeypatch):
    updates = []
    class Bar:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n): updates.append(n)
    monkeypatch.setattr(console, 'progress', lambda **kw: Bar())
    restored = {0, 2}
    computed = []
    for index in console.channel_batches([0, 1, 2], [(0, 4), (4, 8), (8, 10)]):
        if index in restored:
            continue
        computed.append(index)
    assert computed == [1]
    assert updates == [4, 4, 2]


def test_verdicts(capsys):
    console.configure()
    console.white_light_verdict({'whitelight_quality_gate_passed': True,
        'whitelight_geometry_rhat': {'b': 1.002},
        'whitelight_geometry_bulk_ess': {'b': 1277}, 'whitelight_divergences': 0})
    console.spectroscopy_verdict([1.1, 1.2], [])
    console.spectroscopy_verdict([1.1, 1.2], [1])
    output = capsys.readouterr().out
    assert 'Converged: R-hat <= 1.002' in output
    assert '1277, no divergences' in output
    assert 'All 2 channels passed' in output
    assert '1 of 2 channels did not pass' in output
    assert '1.200 µm' in output


def test_runtime_progress_override_does_not_mutate_options():
    options = {'progress_bar': True, 'num_samples': 200}
    assert console.sampling_options(options) == options
    with console.quiet_sampling():
        assert console.sampling_options(options) == {'progress_bar': False, 'num_samples': 200}
    assert options == {'progress_bar': True, 'num_samples': 200}


def test_input_path_fallback_and_explicit_directory(tmp_path):
    from koala.pipeline import resolve_input_path
    (tmp_path / 'spectra.fits').touch()
    cfg = {'path': str(tmp_path), 'input_file': 'spectra.fits'}
    assert resolve_input_path(cfg, 'planet') == str(tmp_path / 'spectra.fits')
    original = tmp_path / 'planet_NIRSPEC' / 'spectra.fits'
    original.parent.mkdir()
    original.touch()
    assert resolve_input_path(cfg, 'planet') == str(original)
    cfg['input_dir'] = 'explicit'
    assert resolve_input_path(cfg, 'planet') == str(tmp_path / 'explicit' / 'spectra.fits')


def test_omitted_host_device_is_auto(monkeypatch):
    from koala import host_devices
    from koala.pipeline import resolve_host_device
    monkeypatch.setattr(host_devices.glob, 'glob', lambda pattern: [])
    monkeypatch.setattr(host_devices.os.path, 'exists', lambda path: False)
    assert host_devices._runs_on_cpu({})
    import koala.pipeline as pipeline
    monkeypatch.setattr(pipeline, '_concrete_gpu_backend', lambda: None)
    monkeypatch.setattr(pipeline, '_set_jax_platform', lambda platform: None)
    monkeypatch.setattr(pipeline.jax, 'default_backend', lambda: 'cpu')
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        assert resolve_host_device(None) == 'cpu'
    assert not caught


def test_sampling_bar_advances_after_computation_and_checkpoint_load(tmp_path, monkeypatch):
    import jax
    import numpy as np
    from models.mcmc_diagnostics import PosteriorSamples
    from koala import sampling
    updates, calls, fingerprints = [], [], []
    class Bar:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, count): updates.append(count)
    monkeypatch.setattr(console, 'progress', lambda **kw: Bar())
    def sample(backend, cache, model, key, t, errors, observations, initial, **kwargs):
        calls.append(observations.shape[0])
        return PosteriorSamples({'x': np.zeros((2, observations.shape[0]))}, num_chains=1)
    monkeypatch.setattr(sampling, '_sample_independent_chunk', sample)
    original = sampling._chunk_checkpoint_fingerprint
    def fingerprint(*args, **kwargs):
        assert kwargs['mcmc_kwargs']['progress_bar'] is True
        result = original(*args, **kwargs)
        fingerprints.append(result)
        return result
    monkeypatch.setattr(sampling, '_chunk_checkpoint_fingerprint', fingerprint)
    def model(t, errors, y=None): pass
    kwargs = dict(model=model, key=jax.random.PRNGKey(1), t=np.arange(4.),
        yerr=np.ones((5, 4)), indiv_y=np.zeros((5, 4)), init_params={}, chunk_size=3,
        sampler_backend='independent_nuts', output_dir=str(tmp_path), checkpoint_prefix='test',
        mcmc_kwargs={'num_samples': 2, 'num_chains': 1, 'progress_bar': True})
    first = sampling.get_samples_chunked(**kwargs)
    second = sampling.get_samples_chunked(**kwargs)
    np.testing.assert_array_equal(first['x'], second['x'])
    assert calls == [3, 2]
    assert updates == [3, 2, 3, 2]
    assert fingerprints[0] == fingerprints[1]


def test_stellar_notice_once_and_real_warnings_retained(tmp_path, capsys, monkeypatch):
    import warnings
    console.configure(False, tmp_path, 'example.yaml')
    for _ in range(2):
        with console.stellar_output('stagger', 5509, 4.22, .04):
            print('Warning: insufficient model coverage. Falling back to nearest point.')
    assert capsys.readouterr().out.count('using the nearest grid point') == 1
    forwarded = []
    monkeypatch.setattr(console, '_original_showwarning', lambda message, *a, **kw: forwarded.append(str(message)))
    console._showwarning('layout engine that is incompatible with subplots_adjust', UserWarning, 'plot.py', 1)
    assert 'incompatible' not in capsys.readouterr().err
    console._showwarning('real problem', RuntimeWarning, 'fit.py', 1)
    assert forwarded == ['real problem']
    log = (tmp_path / 'koala.log').read_text()
    assert 'insufficient model coverage' in log
    assert 'incompatible' in log
    assert 'real problem' in log


def test_progress_display_preserves_sampler_options_and_rng():
    import jax
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    from koala.sampling import _build_numpyro_mcmc
    def model():
        numpyro.sample('x', dist.Normal(0., 1.))
    options = dict(num_warmup=20, num_samples=20, num_chains=1, progress_bar=True)
    visible, _, visible_options = _build_numpyro_mcmc(model, {}, {}, options)
    with console.quiet_sampling():
        quiet, _, quiet_options = _build_numpyro_mcmc(model, {}, {}, options)
    assert visible_options['progress_bar'] is True
    assert quiet_options['progress_bar'] is False
    assert {k: v for k, v in visible_options.items() if k != 'progress_bar'} == {
        k: v for k, v in quiet_options.items() if k != 'progress_bar'}
    visible.run(jax.random.PRNGKey(99))
    quiet.run(jax.random.PRNGKey(99))
    np.testing.assert_array_equal(visible.get_samples()['x'], quiet.get_samples()['x'])


def test_uncertainties_units_grid_labels_and_elapsed_time():
    assert console.parameter_summary('t0', 59787.056748, .000014, 'BMJD_TDB') == (
        '  t0               59787.056748 ± 0.000014 BMJD_TDB')
    assert console.parameter_summary('duration', .119604, .000085, 'd').endswith(
        '0.119604 ± 0.000085 d')
    assert console.parameter_summary('rprs', .14591, .00018).endswith('0.14591 ± 0.00018')
    assert console.parameter_summary('b', .451, .0048).endswith('0.4510 ± 0.0048')
    assert console.parameter_summary('x', 1.5, .0999).endswith('1.50 ± 0.10')
    assert console.parameter_summary('depth', 18523., 230., 'ppm').endswith('18520 ± 230 ppm')
    assert console.elapsed_time(9) == '9 s'
    assert console.elapsed_time(843) == '14 min 3 s'
    assert console.elapsed_time(3729) == '1 h 02 min'
    assert console.grid_description('R10') == 'at R = 10'
    assert console.grid_description('Rnative') == 'at native resolution'
    assert console.grid_description('Rreference') == 'on the reference grid'
    assert console.grid_description('pix8') == 'in bins of 8 columns'


def test_sampling_history_counts_unique_channels_and_verdict(capsys):
    # Channels 0 and 2 need extensions; channel 2 later needs another sampler.
    attempts = [
        {'num_chains': 4, 'science_ess_per_channel': {'depths': [200, 800, 300]},
         'science_rhat_per_channel': {'depths': [1., 1., 1.]},
         'science_tail_ess_per_channel': {'depths': [800, 800, 800]}},
        {'sampling_extension_blocks': 1, 'science_ess_per_channel': {'depths': [800, 800, 300]}},
        {'sampling_extension_blocks': 2, 'science_ess_per_channel': {'depths': [800, 800, 300]}},
        {'input_local_lanes': [2], 'science_ess_per_channel': {'depths': [800]}},
    ]
    assert console.sampling_history_notes(attempts, 3, 400, 0) == (2, 1)
    console.configure()
    console.spectroscopy_verdict([1., 1.1, 1.2], [], extra_draws=2, refitted=1)
    assert 'All 3 channels passed the convergence checks! (2 needed extra draws, 1 was refitted with a second sampler)' in capsys.readouterr().out


def test_arviz_single_chain_noise_goes_only_to_file_even_when_verbose(tmp_path, capsys):
    import arviz
    console.configure(True, tmp_path, 'example.yaml')
    message = 'Shape validation failed: input_shape: (1, 137), minimum_shape: (chains=2, draws=4)'
    arviz._log.warning(message)
    captured = capsys.readouterr()
    assert 'Shape validation failed' not in captured.out + captured.err
    assert message in (tmp_path / 'koala.log').read_text()
    record = logging.makeLogRecord({'msg': 'A real ArviZ problem'})
    assert console._ArvizFilter().filter(record)


def test_default_stellar_cache_honours_xdg_and_home(monkeypatch):
    from koala import paths
    import importlib
    with monkeypatch.context() as patch:
        patch.setenv('XDG_CACHE_HOME', '/tmp/koala-cache-test')
        importlib.reload(paths)
        assert paths.DEFAULT_LD_DATA_PATH == '/tmp/koala-cache-test/koala/exotic_ld_data'
        patch.delenv('XDG_CACHE_HOME')
        importlib.reload(paths)
        from pathlib import Path
        assert paths.DEFAULT_LD_DATA_PATH == str(Path.home() / '.cache/koala/exotic_ld_data')
    importlib.reload(paths)


def test_joint_parent_log_retains_member_details(tmp_path, capsys):
    console.configure(False, tmp_path / 'joint', 'joint.yaml')
    with console.parent_log():
        console.configure(False, tmp_path / 'member', 'member.yaml')
        console.debug('member checkpoint details')
        console.info('Member spectroscopy')
    assert 'member checkpoint details' in (tmp_path / 'joint/koala.log').read_text()
    assert 'member checkpoint details' in (tmp_path / 'member/koala.log').read_text()
    assert 'member checkpoint details' not in capsys.readouterr().out


def test_stellar_download_bars_are_logged_and_real_stderr_is_retained(tmp_path, capsys):
    import sys
    console.configure(True, tmp_path, 'example.yaml')
    with console.exotic_output():
        sys.stderr.write('\rDownloading https://example.org/model.dat: 100%|###| 100/100\n')
        sys.stderr.write('A real stellar-model warning\n')
    captured = capsys.readouterr()
    assert 'model.dat' not in captured.out + captured.err
    assert 'Downloaded stellar limb-darkening data.' in captured.out
    assert 'A real stellar-model warning' in captured.err
    assert 'model.dat' in (tmp_path / 'koala.log').read_text()


def test_new_run_keeps_early_details_in_its_own_log(tmp_path):
    console.configure(False, tmp_path / 'first', 'first.yaml')
    console.debug('first run detail')
    console.configure(False)
    console.debug('early second run detail')
    console.configure(False, tmp_path / 'second', 'second.yaml')
    first = (tmp_path / 'first/koala.log').read_text()
    second = (tmp_path / 'second/koala.log').read_text()
    assert 'early second run detail' not in first
    assert 'early second run detail' in second


def test_root_logging_does_not_duplicate_or_reveal_debug_details(capsys):
    stream = io.StringIO()
    root_handler = logging.StreamHandler(stream)
    logging.getLogger().addHandler(root_handler)
    try:
        console.configure(False)
        console.debug('internal checkpoint detail')
        console.info('Reading spectra')
        assert capsys.readouterr().out == 'Reading spectra\n'
        assert stream.getvalue() == ''
    finally:
        logging.getLogger().removeHandler(root_handler)


def test_refactor_notice_is_file_only_before_configuration(tmp_path, capsys):
    console.configure()
    console._showwarning('\nArviZ is undergoing a major refactor: details', FutureWarning, 'arviz/__init__.py', 1)
    console.configure(True, tmp_path)
    assert 'major refactor' not in capsys.readouterr().out
    assert 'major refactor' in (tmp_path / 'koala.log').read_text()


def test_residual_wording_and_joint_collection(capsys):
    console.configure()
    console.residual_summary(109., 1, 5)
    console.residual_summary(189., 0, 4)
    output = capsys.readouterr().out
    assert 'Residual scatter 109 ppm; 1 integration clipped at 5σ' in output
    assert 'Residual scatter 189 ppm; no integrations clipped' in output
    state = {'collect_residuals': True}
    with console.joint_presentation(state):
        console.residual_summary(109., 1, 4, 'order 1')
        console.finish_run('member', 0)
    assert capsys.readouterr().out == ''
    assert state['residuals'] == [(109., 1, 4, 'order 1')]


def test_early_import_warning_routing(tmp_path):
    import subprocess
    import sys
    script = """
import warnings
import koala
from koala.console import configure
warnings.warn('ArviZ is undergoing a major refactor: import notice', FutureWarning)
warnings.warn('Another future change', FutureWarning)
warnings.warn('ArviZ is undergoing a major refactor: genuine user warning', UserWarning)
configure(output_dir=__import__('sys').argv[1])
"""
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True, text=True, check=True)
    assert 'import notice' not in result.stdout + result.stderr
    assert 'Another future change' in result.stderr
    assert 'genuine user warning' in result.stderr
    assert 'import notice' in (tmp_path / 'koala.log').read_text()


def test_joint_spectroscopy_heading_precedes_coefficients(monkeypatch, capsys):
    from types import SimpleNamespace
    from koala import spectroscopy
    console.configure()
    def coefficients(*args):
        console.info('coefficients here')
        return 'unchanged coefficients'
    monkeypatch.setattr(spectroscopy, '_spectroscopic_ld_uncached', coefficients)
    inputs = SimpleNamespace(instrument='NIRISS/SOSS', order=2)
    with console.joint_presentation({}):
        result = spectroscopy._spectroscopic_ld(None, inputs, [.65, .75], None, 'R10')
    assert result == 'unchanged coefficients'
    output = capsys.readouterr().out
    assert output.index('NIRISS/SOSS order 2: spectroscopy at R = 10, 2 channels') < output.index('coefficients here')
    assert 'orbit fixed to the joint white-light posterior' in output


def test_stellar_download_notice_is_once_per_run_across_contexts(tmp_path, capsys):
    import sys
    console.configure()
    console.configure(False, tmp_path, 'example.yaml')
    for model in ('one', 'two'):
        with console.exotic_output():
            sys.stderr.write(f'Downloading {model}: 100%\n')
    output = capsys.readouterr().out
    assert output.count('Downloaded stellar limb-darkening data.') == 1
    log = (tmp_path/'koala.log').read_text()
    assert 'Downloading one' in log and 'Downloading two' in log
