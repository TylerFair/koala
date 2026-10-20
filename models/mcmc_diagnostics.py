"""Chain-aware posterior containers and standard rank-normalized diagnostics."""
from __future__ import annotations

import jax
import numpy as np
import arviz as az


@jax.tree_util.register_pytree_node_class
class PosteriorSamples(dict):
    """Pooled draws with a recoverable, interleaved independent-chain axis.

    The established array layout is [draw, channel, ...]. Multi-chain draws
    interleave chains within each iteration, so reshaping reconstructs every
    chain. Diagnose or thin the grouped arrays to preserve chain identity.
    """
    def __init__(self, *args, num_chains=1, sampler_used=None, sampled_sites=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_chains = int(num_chains)
        self.sampled_sites = None if sampled_sites is None else tuple(sampled_sites)
        self.sampler_used = None if sampler_used is None else list(sampler_used)
        if self.num_chains < 1:
            raise ValueError('num_chains must be positive.')

    def group_by_chain(self):
        return {name: value.reshape((-1, self.num_chains) + value.shape[1:]).swapaxes(0, 1)
                for name, value in self.items()}

    def tree_flatten(self):
        keys = tuple(self)
        provenance = None if self.sampler_used is None else tuple(self.sampler_used)
        return tuple(self[key] for key in keys), (keys, self.num_chains, provenance, self.sampled_sites)

    @classmethod
    def tree_unflatten(cls, metadata, values):
        keys, chains, provenance, sampled_sites = metadata
        return cls(zip(keys, values), num_chains=chains, sampler_used=provenance, sampled_sites=sampled_sites)


def pool_chains(grouped):
    """Pool [chain, draw, ...] by iteration, retaining chain identity."""
    return {name: value.swapaxes(0, 1).reshape((-1,) + value.shape[2:])
            for name, value in grouped.items()}


def rank_diagnostics(grouped):
    """Return ArviZ bulk/tail ESS and rank/folded split R-hat per element.

    No chain averaging or channel pooling occurs before these diagnostics.
    A constant or non-finite sampled parameter fails with non-finite metrics.
    """
    arrays = {name: np.asarray(jax.device_get(value)) for name, value in grouped.items()}
    if not arrays:
        return {'bulk_ess': {}, 'tail_ess': {}, 'rhat': {}}
    data = az.from_dict(posterior=arrays)
    return {
        'bulk_ess': {name: np.asarray(value) for name, value in az.ess(data, method='bulk').data_vars.items()},
        'tail_ess': {name: np.asarray(value) for name, value in az.ess(data, method='tail').data_vars.items()},
        'rhat': {name: np.asarray(value) for name, value in az.rhat(data, method='rank').data_vars.items()},
    }
