"""Container archives survive deletion of the historical import facade."""
import pickle
import sys
import types

import numpy as np
import pytest

from koala.datacube import SpectroData


@pytest.mark.parametrize('legacy', [False, True], ids=['canonical', 'legacy-class'])
def test_container_load_preserves_values_without_global_alias(tmp_path, monkeypatch, legacy):
    values = dict(wl_time=np.array([60000., 60000.01]),
                  wl_flux=np.array([1., .99]),
                  wl_flux_err=np.array([.001, .002]), instrument='NIRISS/SOSS')
    path = tmp_path / 'observation.pkl'
    if legacy:
        module = types.ModuleType('createdatacube')
        module.SpectroData = type('SpectroData', (), {'__module__': module.__name__})
        archived = module.SpectroData()
        archived.__dict__.update(values)
        with monkeypatch.context() as context:
            context.setitem(sys.modules, module.__name__, module)
            with path.open('wb') as stream:
                pickle.dump(archived, stream)
    else:
        SpectroData(**values).save(path)

    previous = sys.modules.get('createdatacube')
    loaded = SpectroData.load(path)
    assert type(loaded) is SpectroData
    assert loaded.instrument == values['instrument']
    for name in ('wl_time', 'wl_flux', 'wl_flux_err'):
        np.testing.assert_array_equal(getattr(loaded, name), values[name])
    assert sys.modules.get('createdatacube') is previous
