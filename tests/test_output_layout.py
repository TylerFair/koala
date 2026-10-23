"""The real pipeline writes a small, predictable public output directory."""
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from koala.output_paths import OutputPaths


def test_paths_share_science_and_state_conventions(tmp_path):
    paths = OutputPaths(tmp_path, 'target_NIRISS_SOSS_order1')
    assert Path(paths.table('R100', 'spectrum')).name == 'target_NIRISS_SOSS_order1_R100_spectrum.csv'
    assert Path(paths.figure('whitelight')).relative_to(tmp_path) == Path('figures/whitelight.png')
    assert paths.chunks_dir == tmp_path / 'state' / 'chunks'
    assert Path(OutputPaths(tmp_path).table('', 'joint_params')).parent == tmp_path
    parsed, stage = OutputPaths.from_spectrum(paths.table('Rreference', 'spectrum'))
    assert parsed == paths and stage == 'Rreference'


@pytest.fixture(scope='module')
def small_fit(tmp_path_factory):
    """A short native-cadence transit, with real Laplace-initialized sampling."""
    from models.jaxoplanet.core import compute_transit_model
    directory = tmp_path_factory.mktemp('output_fit')
    time = np.linspace(59999.9, 60000.1, 81)
    parameters = {key: jnp.asarray([value]) for key, value in
                  dict(period=3., t0=60000., duration=.08, b=.3, rors=.1).items()}
    parameters['u'] = jnp.asarray([.3, .2])
    model = 1 + np.asarray(compute_transit_model(parameters, jnp.asarray(time), ld_profile='quadratic'))
    wavelength = np.linspace(1., 2., 15)
    error = np.full((len(time), len(wavelength)), .001)
    flux = model[:, None] + np.random.default_rng(44).normal(size=error.shape)*error
    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(a) for a in
            (wavelength, np.full(15, .03), flux, error)]
    hdus += [fits.ImageHDU(np.zeros(1)) for _ in range(4)]
    hdus.append(fits.ImageHDU(time))
    filename = directory / 'spectra.fits'
    fits.HDUList(hdus).writeto(filename)
    planet = {'name': 'test', **{key: {'value': value, 'prior': 'fixed'}
              for key, value in dict(period=3., t0=60000., duration=.08, b=.3).items()},
              'rprs': {'value': .1, 'prior': 'uniform', 'low': .05, 'high': .15}}
    planet['t0'] = {'value': 60000., 'prior': 'uniform', 'low': 59999.998, 'high': 60000.002}
    return directory, dict(planet=planet, stellar={'ld_coefficients': [.3, .2]},
        instrument='NIRISS/SOSS', order=1, input_file=str(filename), host_device='cpu',
        resolution={'high': 'native'}, flags={'ld_prior': 'fixed', 'detrending_type': 'none',
            'whitelight_num_warmup': 200, 'whitelight_num_samples': 400,
            'highres_num_warmup': 100, 'highres_num_samples': 300})


@pytest.mark.parametrize('plots', ['full', 'minimal'])
def test_end_to_end_files_and_minimal_figures(small_fit, plots):
    from koala.pipeline import run
    directory, config = small_fit
    output = directory / plots
    config = deepcopy(config)
    config['output_dir'] = str(output)
    config['flags']['plots'] = plots
    run(config)
    prefix = 'test_NIRISS_SOSS_order1'
    assert {p.name for p in output.iterdir()} == {
        'koala.log', 'figures', 'state', f'{prefix}_Rnative_spectrum.csv',
        f'{prefix}_Rnative_params.csv', f'{prefix}_Rnative_lightcurves.csv',
        f'{prefix}_whitelight_params.csv', f'{prefix}_whitelight_lightcurve.csv'}
    minimal = {'whitelight.png', 'Rnative_spectrum.png'}
    full = minimal | {'whitelight_start.png', 'whitelight_corner.png', 'Rnative_lightcurves.png',
                      'Rnative_residual_map.png', 'Rnative_limb_darkening.png', 'Rnative_noise.png'}
    assert {p.name for p in (output/'figures').iterdir()} == (full if plots == 'full' else minimal)
    assert list((output/'state'/'chunks').glob('*.pkl'))
    assert not list(output.rglob('*.pdf'))
    spectrum = pd.read_csv(output/f'{prefix}_Rnative_spectrum.csv')
    curves = pd.read_csv(output/f'{prefix}_Rnative_lightcurves.csv')
    assert [f'flux_{i:03d}' for i in range(len(spectrum))] == [c for c in curves if c.startswith('flux_') and not c.startswith('flux_err_')]
    # Reopening the same run must leave its checkpoint contents untouched.
    checkpoints = {p: p.read_bytes() for p in (output/'state'/'chunks').glob('*.pkl')}
    run(deepcopy(config))
    assert checkpoints == {p: p.read_bytes() for p in checkpoints}
    assert 'Reusing the white-light fit' in (output/'koala.log').read_text()
