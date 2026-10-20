"""Resident likelihood/PSIS slabs match disk streaming, including padding."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
import pytest

from koala.marginalization.likelihood import Stage,CheckpointPosterior,CompiledLikelihood
from koala.marginalization import scoring

jax.config.update('jax_enable_x64',True)


def model(t,yerr,y=None,likelihood_mask=None):
    location = numpyro.sample('location',dist.Normal(jnp.zeros(len(yerr)),1.))
    with handlers.mask(mask=True if likelihood_mask is None else likelihood_mask):
        numpyro.sample('obs',dist.Normal(location[:,None]+jnp.zeros_like(t),yerr),obs=y)


@pytest.mark.parametrize('checkpoint',[False,True])
def test_device_psis_matches_disk_streaming_without_likelihood_disk_writes(tmp_path,monkeypatch,checkpoint):
    stage = Stage(model,np.arange(7.),np.ones((3,7)),np.zeros((3,7)),
        dict(likelihood_mask=np.array([True,True,False,True,True,False,True])),dict(wavelength=np.arange(3.)))
    samples = dict(location=np.random.default_rng(23).normal(0.,.15,(103,3)),
                   rors=np.full((103,3,1),.1))
    if checkpoint:
        paths = []
        for begin,end in [(0,2),(2,3)]:
            path = tmp_path/f'demo_chunk_{begin}_{end}.npz'
            np.savez(path,**{name:value[:,begin:end] for name,value in samples.items()})
            paths.append(path)
        posterior = CheckpointPosterior(paths,stage)
    else:
        posterior = samples
    monkeypatch.setattr(scoring,'_device_loglik_buffer',lambda *args:None)
    expected = scoring.score_variant(stage,posterior,draw_batch_size=16,point_batch_size=4,scratch=tmp_path)
    monkeypatch.setattr(scoring,'_device_loglik_buffer',lambda draws,times:jnp.empty((draws,1,times),dtype=jnp.float64))
    monkeypatch.setattr(np.lib.format,'open_memmap',lambda *args,**kwargs:pytest.fail('Resident scoring wrote a full likelihood slab to disk.'))
    actual = scoring.score_variant(stage,posterior,draw_batch_size=16,point_batch_size=4,scratch=tmp_path)
    np.testing.assert_allclose(actual[0],expected[0],atol=2e-14,rtol=0)
    np.testing.assert_allclose(actual[1],expected[1],atol=2e-12,rtol=0)
    if checkpoint:
        np.testing.assert_array_equal(posterior.fill_depth(),samples['rors'][...,0]**2)


def test_device_likelihood_rejects_nonfinite_posterior_values():
    stage = Stage(model,np.arange(7.),np.ones((1,7)),np.zeros((1,7)),{},dict(wavelength=[3.]))
    samples = dict(location=np.full((32,1),np.nan))
    with pytest.raises(ValueError,match='Nonfinite pointwise likelihood'):
        list(CompiledLikelihood(stage).batches(samples,device=True))
