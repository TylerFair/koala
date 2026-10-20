"""Sweep-local caches and compatible starts, without cross-model adaptation."""
from __future__ import annotations

import copy
import gc
from functools import wraps
import hashlib
from pathlib import Path
import pickle

import numpy as np


def _resident_mb():
    try:
        import os
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/2**20
    except (OSError, ValueError, IndexError):
        return 0.


def _cache_budget_mb():
    """Bound compilation growth against the tightest parent cgroup limit."""
    budget = 4096.
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, group = line.split(':', 2)
            if 'memory' in controllers.split(','):
                directory = Path('/sys/fs/cgroup/memory')/group.lstrip('/')
                while directory != Path('/sys/fs/cgroup/memory').parent:
                    limit = directory/'memory.limit_in_bytes'
                    if limit.exists():
                        value = int(limit.read_text())
                        if 0 < value < 2**60:
                            budget = min(budget, .3*value/2**20)
                    directory = directory.parent
                break
    except (OSError, ValueError):
        pass
    return max(256., budget)


def _key(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


class SweepContext:
    """One grid's preparation, stellar, model and shape-specialized runner caches.

    Cached data are copied at the stage boundary, preserving the ordinary
    pipeline's ownership contract. Starts contain common physical/transit
    sites only. Every fit still recomputes its MAP, metric and adaptation.
    """

    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self._data, self._models, self._stellar, self._coefficients = {}, {}, {}, {}
        self._runners, self._starts = {}, {}
        self._science_identity = None
        self.stats = dict(data_hits=0, model_hits=0, stellar_hits=0, ld_hits=0,
                          warm_started_sites=0, runner_cache_partitions=0,
                          compilation_evictions=[])

    def science_identity(self):
        if self._science_identity is None:
            from .scoring import code_identity
            self._science_identity = code_identity()
        return self._science_identity

    def data_path(self, fingerprint):
        directory = self.workspace / "prepared_data"
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / f"{fingerprint}.pkl")

    def load_data(self, artifact, loader):
        key = artifact.fingerprint
        if key in self._data and artifact.is_reusable():
            self.stats["data_hits"] += 1
            print("Reusing sweep preparation in memory.")
        else:
            self._data[key] = loader()
        # JAX arrays are immutable: retain their device buffers while copying
        # the container and mutable host arrays owned by each pipeline stage.
        import jax
        memo = {id(value): value for value in vars(self._data[key]).values()
                if isinstance(value, jax.Array)}
        return copy.deepcopy(self._data[key], memo)

    def factory(self, builder):
        @wraps(builder)
        def cached(**kwargs):
            key = (builder, _key(kwargs))
            if key in self._models:
                self.stats["model_hits"] += 1
            else:
                self._models[key] = builder(**kwargs)
            return self._models[key]
        return cached

    def stellar(self, builder, **kwargs):
        # Include atmosphere file metadata to avoid reuse if the grid changes.
        from koala.artifacts import _directory_metadata_identity
        key = _key((kwargs, _directory_metadata_identity(kwargs["ld_data_path"])))
        if key in self._stellar:
            self.stats["stellar_hits"] += 1
        else:
            self._stellar[key] = builder(**kwargs)
        return self._stellar[key]

    def coefficients(self, key, compute):
        key = _key(key)
        if key in self._coefficients:
            self.stats["ld_hits"] += 1
        else:
            self._coefficients[key] = compute()
        return copy.deepcopy(self._coefficients[key])

    def runner_caches(self, model, nuts_kwargs, mcmc_kwargs):
        key = (model, _key((nuts_kwargs, mcmc_kwargs)))
        if key not in self._runners:
            self._runners[key] = ({}, {})
            self.stats["runner_cache_partitions"] += 1
        return self._runners[key]

    def finish_target(self, budget_mb=None):
        """Evict completed numerical programs when their resident cache is large.

        Prepared data, stellar objects, LD coefficients, compatible starts and
        persistent XLA files remain available. This runs only inside a grid,
        after a complete target; it never resets an active chain.
        """
        budget = _cache_budget_mb() if budget_mb is None else float(budget_mb)
        if not np.isfinite(budget) or budget < 0:
            raise ValueError('Grid compilation-cache budget must be finite and nonnegative.')
        if budget == 0:
            return False
        gc.collect()
        before = _resident_mb()
        if before <= budget:
            return False
        self._models.clear()
        self._runners.clear()
        import jax
        jax.clear_caches()
        gc.collect()
        event = dict(budget_mb=budget, rss_before_mb=before, rss_after_mb=_resident_mb())
        self.stats['compilation_evictions'].append(event)
        print(f'Grid compilation-cache eviction: {before:.0f} MB resident (budget {budget:.0f} MB).', flush=True)
        return True

    def warm_start(self, stage, initial, *, law, parameter_specs=None, site_bounds=None):
        """Copy common sites and same-law LD starts, never previous priors/metrics."""
        values = dict(initial)
        previous = self._starts.get(stage, {})
        common = {"rors", "c", "v", "v2", "v3", "v4", "log_jitter", "trend_coordinates"}
        supports = dict(site_bounds or {})
        if parameter_specs:
            from models.priors import latent_init_site
            for parameter, specs in parameter_specs.items():
                for index, spec in enumerate(specs):
                    site = latent_init_site(spec, f"{'rors' if parameter == 'rprs' else parameter}_{index}")
                    if site is not None:
                        lower, upper = spec.low, spec.high
                        if spec.prior == 'log_uniform':
                            lower, upper = np.log(lower), np.log(upper)
                        supports[site[0]] = (lower, upper)
        for name, value in previous.get("values", {}).items():
            shared_geometry = stage == "white" and name.endswith("_0") and any(
                name.startswith(stem) for stem in ("rors", "log_rors", "period", "log_period", "logD",
                                                    "t0", "log_t0", "b", "log_b", "_b", "duration"))
            same_ld = previous.get("law") == law and name in {"u", "c1", "c2", "ld_smooth", "ld_uplus_uminus", "ld_coefficients", "ld_q"}
            if name in values and (name in common or shared_geometry or same_ld):
                candidate = np.asarray(value)
                lower, upper = supports.get(name, (None, None))
                supported = ((lower is None or np.all(candidate > lower)) and
                             (upper is None or np.all(candidate < upper)))
                if supported and candidate.shape == np.shape(values[name]) and np.all(np.isfinite(candidate)):
                    values[name] = candidate.copy()
                    self.stats["warm_started_sites"] += 1
        return values

    def remember(self, stage, samples, *, law):
        # Posterior medians of actual sampled sites preserve the prior support.
        sites = getattr(samples, "sampled_sites", None) or tuple(samples)
        values = {key: np.median(np.asarray(samples[key]), axis=0) for key in sites if key in samples}
        self._starts[stage] = dict(law=law, values=values)

    @staticmethod
    def kernel(law, param_method):
        return "quadratic_contact_safe" if law == "quadratic" and param_method == "duration" else "auto"

    @staticmethod
    def ld_site_bounds(law, prior, basis, coefficient_bounds=(0., 1.)):
        if law == 'power2' and prior in {'uniform', 'gaussian', 'stellarprior'}:
            lower = np.array([0., 0. if prior == 'uniform' else .001])
            return dict(c1=(0., 1.), c2=(lower[1], 1.), ld_coefficients=(lower, 1.))
        if law == 'quadratic' and prior == 'fixed_u1':
            return {'u2': tuple(coefficient_bounds)}
        if law == 'quadratic' and basis == 'kipping' and prior in {'uniform', 'gaussian', 'stellarprior'}:
            return {'ld_q': (0., 1.)}
        if law == 'quadratic' and prior in {'uniform', 'sing'}:
            return {'u': tuple(coefficient_bounds)} if basis == 'coefficients' else {
                'ld_uplus_uminus': (np.array([-1., -2.]), np.array([2., 2.]))}
        return {}
