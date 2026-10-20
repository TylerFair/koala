"""Opt-in two-program grid fitting with independently adapted model lanes."""
from __future__ import annotations

import copy
from collections import OrderedDict
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .batched import create_batched_model, BATCHED_CHANNEL_KWARGS
from .fitting import SweepContext, _key
from .grid import resolve_path
from .scoring import code_identity, _file_digest, _save_npz


class _Prepared(Exception):
    def __init__(self, context):
        self.context = context


class _Preparation(SweepContext):
    def prepare_white_light(self, context):
        raise _Prepared(context)


class _RepeatedChannels:
    """Index repeated variants without materializing variant × channel × time."""
    def __init__(self, values, repeats):
        self.values = np.asarray(values)
        if self.values.ndim != 2 or repeats < 1:
            raise ValueError('Repeated observations need [channel, time] data and positive repeats.')
        self.shape = (len(self.values)*repeats, self.values.shape[1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, selection):
        if isinstance(selection, slice):
            begin, end, step = selection.indices(len(self))
            if step == 1 and begin < end and begin//len(self.values) == (end-1)//len(self.values):
                return self.values[begin%len(self.values):(end-1)%len(self.values)+1]
            positions = np.arange(begin, end, step)
        else:
            positions = np.asarray(selection)
        return self.values[positions%len(self.values)]


def _validate(context):
    flags = context.cfg.get('flags', {})
    if (context.n_planets != 1 or context.param_method != 'duration'
            or context.uses_surface_model or context.transit_engine != 'jaxoplanet'
            or context.detrending_type not in {'linear', 'quadratic', 'linear_discontinuity'}
            or context.ld_profile not in {'quadratic', 'power2'}
            or context.ld_prior_mode not in {'fixed', 'uniform'}
            or flags.get('spectro_joint_geometry', False)
            or flags.get('trend_inference', 'sampled_uniform') != 'sampled_uniform'
            or (context.whitelight_trend_parameterization != 'physical' and not (
                context.detrending_type == 'linear_discontinuity' and
                context.whitelight_trend_parameterization == 'cadence'))):
        raise ValueError('Batched fitting currently requires one duration-based unspotted transit, '
                         'sampled linear/quadratic/sigmoid-step trends and fixed/uniform LD. Use ordinary grid fitting for other targets.')
    from models.spectro_trends import validate_spectral_trend_bounds
    bounds=validate_spectral_trend_bounds(flags.get('spectro_trend_bounds'))
    if bounds and set(bounds)-{'c','v','v2','A_jump'}:
        raise ValueError('Batched spectral bounds support c/v/v2/A_jump; use ordinary grid fitting for other priors.')
    if getattr(context,'trend_priors',None):
        raise ValueError('Batched white fitting does not support custom trend_priors; use ordinary grid fitting to retain those priors.')
    if context.whitelight_mcmc_kwargs['num_chains'] < 2:
        raise ValueError('Batched grid convergence requires at least two independent chains.')
    if flags.get('grid_prime_samplers') is not None and not isinstance(flags['grid_prime_samplers'], bool):
        raise ValueError('flags.grid_prime_samplers must be Boolean.')


def _coefficients(context, sweep):
    from koala.limb_darkening import get_limb_darkening
    data = context.data
    if context.explicit_ld is not None:
        return np.asarray(context._explicit_ld_grid()), np.asarray(context._explicit_ld_grid(data.wavelengths_hr))
    def compute(waves, widths):
        return get_limb_darkening(context.sld, waves, widths, context.instrument,
                                 order=context.order, ld_profile=context.ld_profile)
    def cached(waves, widths):
        return np.asarray(sweep.coefficients((id(context.sld), waves, widths,
            context.instrument, context.order, context.ld_profile), lambda: compute(waves, widths)))
    return cached(data.wavelengths_unbinned, 0.), cached(data.wavelengths_hr, data.wavelengths_err_hr)


def _initial_ld(coefficients, law, basis, *, parameterization='coefficients', free=None):
    value = np.asarray(coefficients).copy()
    if law == 'quadratic' and basis == 'uplus_uminus':
        value = np.column_stack((value[:, 0]+value[:, 1], value[:, 0]-value[:, 1]))
        return np.clip(value, [-.999, -1.999], [1.999, 1.999])
    if law == 'power2':
        from models.ld_parameterization import clip_into_maxted_uniform_support
        value = np.asarray(clip_into_maxted_uniform_support(jnp.asarray(value)))
    else:
        value = np.clip(value, .001, .999)
    if law == 'power2' and parameterization == 'maxted_smooth':
        from models.ld_parameterization import Power2MaxtedUniformTransform
        value = np.asarray(Power2MaxtedUniformTransform().inv(jnp.asarray(value)))
        if free is not None:
            value = np.where(np.asarray(free)[:, None], value, 0.)
    return value


def _highres_ld_start(white, white_fixed, highres_fixed, channels, law, basis, free, parameterization):
    """Transfer the fitted white-light LD offset to each stellar bandpass."""
    physical = (np.asarray(white['u']) if law == 'quadratic' else
                np.stack((np.asarray(white['c1']), np.asarray(white['c2'])), axis=-1))
    correction = np.median(physical, axis=0)-np.asarray(white_fixed)
    coefficients = np.asarray(highres_fixed)+np.repeat(correction, channels, axis=0)
    return _initial_ld(coefficients, law, basis, parameterization=parameterization, free=free)


def _transit_window(t, medians):
    """Union of all fixed-geometry transit masks, including exact contacts."""
    from models.jaxoplanet.core import build_transit_phase_offsets
    active = np.zeros(len(t), dtype=bool)
    from models.harmonica.core import harmonica_a_rs_from_duration, harmonica_duration_from_geometry
    axes = harmonica_a_rs_from_duration(medians['period_0'], medians['duration_0'],
                                      medians['b_0'], medians['rors_0'])
    durations = harmonica_duration_from_geometry(medians['period_0'], axes, medians['b_0'], np.sqrt(.5))
    for period, epoch, duration in zip(medians['period_0'], medians['t0_0'], durations):
        offsets, _ = build_transit_phase_offsets(jnp.asarray(t), jnp.atleast_1d(period),
                                                jnp.atleast_1d(epoch), jnp.atleast_1d(duration))
        tolerance = 32*np.finfo(float).eps*max(1., abs(float(epoch)))
        active |= np.abs(np.asarray(offsets)[0]) <= .5*duration+tolerance
    return tuple(np.flatnonzero(active).tolist())


def _white_transit_window(t, parameter_priors):
    """Prove an epoch/duration envelope over the complete white-light prior.

    Fixed positive period is required. Unbounded or variable-period priors
    retain direct evaluation of every cadence. The envelope includes all
    periodic events, all impact parameters/radii, and endpoint roundoff.
    """
    if parameter_priors.get('ecc', ()) and parameter_priors['ecc'][0].value != 0:
        return None  # Exact eccentric contacts are asymmetric around conjunction.
    period = parameter_priors['period'][0]
    epoch = parameter_priors['t0'][0]
    duration = parameter_priors['duration'][0]
    if not period.fixed or not np.isfinite(period.value) or period.value <= 0:
        return None
    lower, upper = (epoch.value, epoch.value) if epoch.fixed else (epoch.low, epoch.high)
    longest = duration.value if duration.fixed else duration.high
    if any(value is None or not np.isfinite(value) for value in (lower, upper, longest)):
        return None
    t = np.asarray(t)
    midpoint = lower+(upper-lower)/2
    half_width = (upper-lower)/2+max(0., longest)/2
    if half_width >= period.value/2:
        return None
    offsets = np.mod(t-midpoint+period.value/2, period.value)-period.value/2
    tolerance = 64*np.finfo(float).eps*max(1., abs(midpoint), abs(period.value),
                                         float(np.max(np.abs(t))))
    return tuple(np.flatnonzero(np.abs(offsets) <= half_width+tolerance).tolist())


def _lane_kwargs(kwargs, selection):
    return {name: value[selection] if name in BATCHED_CHANNEL_KWARGS else value
            for name, value in kwargs.items()}


def _quality(samples, diagnostics, *, min_ess=400.):
    from models.mcmc_diagnostics import rank_diagnostics
    grouped = samples.group_by_chain()
    sites = samples.sampled_sites or tuple(samples)
    metrics = rank_diagnostics({name: grouped[name] for name in sites})
    bulk = {name: np.asarray(value) for name, value in metrics['bulk_ess'].items()}
    tail = {name: np.asarray(value) for name, value in metrics['tail_ess'].items()}
    rhat = {name: np.asarray(value) for name, value in metrics['rhat'].items()}
    report = dict(min_bulk_ess=float(min(np.min(v) for v in bulk.values())),
        min_tail_ess=float(min(np.min(v) for v in tail.values())),
        max_rhat=float(max(np.max(v) for v in rhat.values())),
        divergences=int(np.asarray(diagnostics.diverging).sum()),
        min_ess_required=float(min_ess), max_rhat_required=1.01,
        mean_num_steps=float(np.asarray(diagnostics.num_steps).mean()))
    report['passed'] = bool(report['divergences'] == 0 and
        np.isfinite(report['max_rhat']) and report['max_rhat'] < 1.01 and
        np.isfinite(report['min_bulk_ess']) and report['min_bulk_ess'] >= min_ess and
        np.isfinite(report['min_tail_ess']) and report['min_tail_ess'] >= min_ess)
    lanes = next(iter(samples.values())).shape[1]
    lane_ess = np.min(np.stack([v.reshape(lanes, -1).min(axis=1) for v in bulk.values()]), axis=0)
    lane_tail = np.min(np.stack([v.reshape(lanes, -1).min(axis=1) for v in tail.values()]), axis=0)
    lane_rhat = np.max(np.stack([v.reshape(lanes, -1).max(axis=1) for v in rhat.values()]), axis=0)
    lane_divergences = np.asarray(diagnostics.diverging).sum(axis=0)
    report['failed_lanes'] = np.flatnonzero((~np.isfinite(lane_ess)) | (lane_ess < min_ess)
        | (~np.isfinite(lane_tail)) | (lane_tail < min_ess)
        | (~np.isfinite(lane_rhat)) | (lane_rhat >= 1.01) | (lane_divergences > 0)).tolist()
    return report


def _extend_to_quality(runner, samples, raw, *, min_ess, max_blocks=3):
    """Continue current chains, thinning each actual longer history evenly."""
    from models.independent_nuts import IndependentNUTSDiagnostics
    from models.mcmc_diagnostics import PosteriorSamples, pool_chains
    quality = _quality(samples, IndependentNUTSDiagnostics(**raw), min_ess=min_ess)
    original_draws = len(next(iter(samples.values()))) // samples.num_chains
    history, raw_history = samples.group_by_chain(), dict(raw)
    for block in range(max_blocks):
        if quality['passed'] or quality['divergences']:
            return samples, raw, quality, block
        more, extra = runner.continue_raw()
        grouped = PosteriorSamples(more, num_chains=samples.num_chains).group_by_chain()
        history = {name: np.concatenate((value, grouped[name]), axis=1)
                   for name, value in history.items()}
        positions = np.linspace(0, next(iter(history.values())).shape[1]-1, original_draws).astype(int)
        samples = PosteriorSamples(pool_chains({name: value[:, positions] for name, value in history.items()}),
            num_chains=samples.num_chains, sampled_sites=samples.sampled_sites)
        for name in ('num_steps', 'accept_prob', 'diverging'):
            raw_history[name] = np.concatenate((raw_history[name], np.asarray(extra[name])), axis=0)
        raw = dict(extra)
        for name in ('num_steps', 'accept_prob', 'diverging'):
            value = raw_history[name]
            grouped_value = value.reshape((-1, samples.num_chains)+value.shape[1:]).swapaxes(0, 1)
            raw[name] = pool_chains({name: grouped_value[:, positions]})[name]
        quality = _quality(samples, IndependentNUTSDiagnostics(**raw), min_ess=min_ess)
        # Thinning cannot erase evidence of a divergent transition. Include
        # all continued draws when deciding whether this target needs a refit.
        history_divergences = np.asarray(raw_history['diverging']).sum(axis=0)
        quality['divergences'] = int(history_divergences.sum())
        if quality['divergences']:
            quality['passed'] = False
            quality['failed_lanes'] = sorted(set(quality['failed_lanes']) |
                                            set(np.flatnonzero(history_divergences).tolist()))
    return samples, raw, quality, max_blocks


def _sample(model, t, errors, observations, initial, kwargs, options, cache, fingerprint, *, stream=False):
    from models.independent_nuts import (get_samples_independent, build_independent_nuts_runner,
                                        IndependentNUTSDiagnostics)
    from models.mcmc_diagnostics import PosteriorSamples
    from .posterior import PosteriorStore
    store = PosteriorStore(cache.with_suffix(''), fingerprint, lanes=len(errors),
        chains=options['chains'], draws=options['draws']) if stream else None
    meta = cache.with_suffix('.json')
    if stream and store.complete() and meta.exists():
        report = json.loads(meta.read_text())
        if report.get('fingerprint') == fingerprint and report['quality']['passed']:
            return store, dict(report, cache_reused=True)
    if not stream and cache.exists() and meta.exists():
        report = json.loads(meta.read_text())
        if report.get('fingerprint') == fingerprint and report['quality']['passed']:
            with np.load(cache, allow_pickle=False) as saved:
                samples = PosteriorSamples({name: np.asarray(v) for name, v in saved.items()},
                    num_chains=report['chains'], sampled_sites=report['sampled_sites'])
            return samples, dict(report, cache_reused=True)
    started = time.perf_counter()
    chains, draws, warmup, seed, min_ess = (options[name] for name in
        ('chains', 'draws', 'warmup', 'seed', 'min_ess'))
    # Each lane receives fresh dispersed starts, MAP/metric and adaptation.
    mcmc_options = dict(num_chains=chains, num_samples=draws, num_warmup=warmup,
                        progress_bar=bool(options.get('progress', True)))
    nuts_options = dict(dense_mass=True, mass_matrix='laplace', target_accept_prob=.95,
            max_tree_depth=8, laplace_target_accept=.95, laplace_max_tree_depth=8,
            laplace_warmup=200, laplace_start_at_map=True,
            laplace_hessian_method='finite_difference', laplace_fd_batch_size=1,
            laplace_fuse_program=False)
    width = min(len(errors), options.get('width', 40))
    stats = jax.devices()[0].memory_stats() or {}
    memory = int(stats.get('bytes_limit', 0))
    if memory:
        estimate = 6000*len(t)*chains + 128*draws*chains
        width = min(width, max(1, int((.6*memory-500000000)//estimate)))
    # Balance the final batch while retaining one static executable width.
    batches = (len(errors)+width-1)//width
    width = (len(errors)+batches-1)//batches
    runner = build_independent_nuts_runner(model, nuts_kwargs=nuts_options,
        mcmc_kwargs=mcmc_options, lane_width=width,
        channel_varying_kwargs=tuple(name for name in kwargs if name in BATCHED_CHANNEL_KWARGS))
    if options.get('prime', False):
        from .priming import prime_laplace_runner
        prime_laplace_runner(runner)
    blocks, qualities = [], []
    retries = []
    retry_runners = OrderedDict()
    for begin in range(0, len(errors), width):
        end = min(len(errors), begin+width)
        accepted = store.accepted(begin, end) if stream else None
        if accepted is not None:
            qualities.append(accepted['quality'])
            print(f'Reusing accepted posterior lanes {begin}:{end}', flush=True)
            continue
        print(f'Batched sampler lanes {begin}:{end}/{len(errors)} (width {width})', flush=True)
        block, diagnostic = get_samples_independent(model,
            jax.random.fold_in(jax.random.PRNGKey(seed), begin), jnp.asarray(t),
            jnp.asarray(errors[begin:end]), jnp.asarray(observations[begin:end]),
            {name: value[begin:end] for name, value in initial.items()},
            lane_width=width, channel_varying_kwargs=runner.channel_varying_kwargs, mcmc_kwargs=mcmc_options,
            nuts_kwargs=nuts_options, return_diagnostics=True, _runner=runner,
            **_lane_kwargs(kwargs, slice(begin, end)))
        block, raw, quality, extensions = _extend_to_quality(runner, block,
            dict(runner._last_pooled_raw), min_ess=min_ess)
        print(f'Batched lanes {begin}:{end} quality: {quality}; continuation blocks: {extensions}', flush=True)
        if not quality['passed']:
            failed = np.asarray(quality['failed_lanes'], dtype=int)
            print(f'Retrying failed batched lanes {(begin+failed).tolist()} with fresh metrics.', flush=True)
            fallback_initial = {name: np.median(np.asarray(block[name])[:, failed], axis=0)
                                for name in block.sampled_sites}
            adaptive = dict(dense_mass=True, mass_matrix='adaptive', target_accept_prob=.99, max_tree_depth=10)
            plans = [('adaptive', adaptive)]
            if (quality['divergences'] and quality['min_bulk_ess'] >= min_ess
                    and np.isfinite(quality['max_rhat']) and quality['max_rhat'] <= 1.01):
                # Well-mixed lanes with a rare divergence already have useful
                # curvature. Recompute it at their posterior median and use
                # smaller steps before paying for full dense-mass adaptation.
                strict = dict(nuts_options, target_accept_prob=.99, max_tree_depth=10,
                              laplace_target_accept=.99, laplace_max_tree_depth=10)
                plans.insert(0, ('laplace_retuned', strict))
            attempts = []
            for attempt, (strategy, retry_options) in enumerate(plans):
                retry_key = (strategy,len(failed))
                reused_runner = retry_key in retry_runners
                if reused_runner:
                    fallback_runner = retry_runners.pop(retry_key)
                else:
                    fallback_runner = build_independent_nuts_runner(model, nuts_kwargs=retry_options,
                        mcmc_kwargs=mcmc_options, lane_width=len(failed),
                        channel_varying_kwargs=runner.channel_varying_kwargs)
                    if options.get('prime', False) and strategy == 'laplace_retuned':
                        prime_laplace_runner(fallback_runner)
                retry_runners[retry_key] = fallback_runner
                while len(retry_runners) > 4:
                    retry_runners.popitem(last=False)
                fallback, fallback_diag = get_samples_independent(model,
                    jax.random.fold_in(jax.random.PRNGKey(seed), 193000+begin+attempt*1000000), jnp.asarray(t),
                    jnp.asarray(errors[begin:end][failed]), jnp.asarray(observations[begin:end][failed]),
                    fallback_initial, lane_width=len(failed), channel_varying_kwargs=runner.channel_varying_kwargs,
                    mcmc_kwargs=mcmc_options, nuts_kwargs=retry_options, return_diagnostics=True,
                    _runner=fallback_runner, **_lane_kwargs(_lane_kwargs(kwargs, slice(begin, end)), failed))
                fallback, fallback_raw, fallback_quality, fallback_extensions = _extend_to_quality(
                    fallback_runner, fallback, dict(fallback_runner._last_pooled_raw), min_ess=min_ess)
                print(f'{strategy} retry quality: {fallback_quality}', flush=True)
                attempts.append(dict(strategy=strategy, quality=fallback_quality, extensions=fallback_extensions,
                                     runner_reused=reused_runner,program_builds=fallback_runner.program_build_count))
                if fallback_quality['passed']:
                    break
            retries.append(dict(lanes=(begin+failed).tolist(), initial_quality=quality,
                                attempts=attempts, accepted=fallback_quality['passed']))
            if fallback_quality['passed']:
                merged = {name: np.asarray(value).copy() for name, value in block.items()}
                for name in merged:
                    merged[name][:, failed] = np.asarray(fallback[name])
                block = PosteriorSamples(merged, num_chains=chains, sampled_sites=block.sampled_sites)
                for name in ('num_steps', 'accept_prob', 'diverging'):
                    raw[name] = np.asarray(raw[name]).copy()
                    raw[name][:, failed] = np.asarray(fallback_raw[name])
                quality = _quality(block, IndependentNUTSDiagnostics(**raw), min_ess=min_ess)
        qualities.append(quality)
        if not quality['passed']:
            meta.write_text(json.dumps(dict(fingerprint=fingerprint, quality=quality,
                failed_lanes=[begin, end], seconds=time.perf_counter()-started), indent=2))
            raise RuntimeError(f'Batched posterior failed convergence: {quality}; diagnostics saved at {meta}.')
        if stream:
            store.save(begin, end, block, quality)
            del block
        else:
            blocks.append(block)
    samples = store if stream else PosteriorSamples({name: np.concatenate(
        [np.asarray(block[name]) for block in blocks], axis=1) for name in blocks[0]},
        num_chains=chains, sampled_sites=blocks[0].sampled_sites)
    quality = dict(min_bulk_ess=min(q['min_bulk_ess'] for q in qualities),
        min_tail_ess=min(q['min_tail_ess'] for q in qualities), max_rhat=max(q['max_rhat'] for q in qualities),
        divergences=sum(q['divergences'] for q in qualities), passed=all(q['passed'] for q in qualities),
        min_ess_required=float(min_ess), max_rhat_required=1.01,
        mean_num_steps=float(np.mean([q['mean_num_steps'] for q in qualities])))
    report = dict(fingerprint=fingerprint, quality=quality, chains=chains,
        sampled_sites=list(samples.sampled_sites or samples), seconds=time.perf_counter()-started,
        cache_reused=False, resident_width=width, chunks=len(qualities),
        sampler_program_builds=runner.program_build_count, retries=retries,
        priming=getattr(runner, '_grid_priming_timings', []), storage='chunks' if stream else 'pooled archive')
    meta.write_text(json.dumps(report, indent=2))
    if not quality['passed']:
        raise RuntimeError(f'Batched posterior failed convergence: {quality}; diagnostics saved at {meta}.')
    if not stream:
        _save_npz(cache, **{name: np.asarray(v) for name, v in samples.items()})
    return samples, report



def _spectral_trend_group(context):
    """Distinct declared spectral priors must never share a static target."""
    from models.spectro_trends import validate_spectral_trend_bounds
    flags=context.cfg.get('flags',{})
    return (flags.get('spectro_trend_basis','elapsed_days'),
            json.dumps(validate_spectral_trend_bounds(flags.get('spectro_trend_bounds')),sort_keys=True))


def run_batched_grid(manifest):
    """Fit compatible variants in reusable law and trend-family programs.

    Returns a separate manifest for scoring. Outputs do not overwrite the
    ordinary grid artifacts. Unsupported targets fail before sampling.
    """
    from koala.pipeline import run
    from koala.config import _resolve_stage_mcmc_kwargs
    from models.priors import latent_init_site
    manifest = Path(manifest).resolve()
    root = manifest.parent
    spec = yaml.safe_load(manifest.read_text())
    sweep = _Preparation(root)
    started = time.perf_counter()
    contexts, coefficients = [], []
    for variant in spec['variants']:
        path = resolve_path(manifest, variant['config'])
        cfg = yaml.safe_load(path.read_text())
        cfg['output_dir'] = str(root / 'batched_preparation' / variant['name'])
        try:
            run(cfg, config_path=str(path), execution_context=sweep)
        except _Prepared as prepared:
            context = prepared.context
        else:
            raise RuntimeError('Pipeline preparation did not return a fitting context.')
        _validate(context)
        contexts.append(context)
        coefficients.append(_coefficients(context, sweep))
    first = contexts[0]
    # One group only has one set of shared parameter priors and observations.
    for context in contexts[1:]:
        if _key(context.planet_parameter_specs) != _key(first.planet_parameter_specs):
            raise ValueError('Batched variants must share the planet parameter priors.')
        for name in ('time', 'wl_flux', 'wl_flux_err', 'flux_hr', 'flux_err_hr', 'wavelengths_hr'):
            if not np.array_equal(getattr(context.data, name), getattr(first.data, name)):
                raise ValueError(f'Batched variant {name} observations differ.')
    source = Path(first.cfg['path']) / first.cfg.get('input_dir', '') / first.cfg.get('input_file', first.cfg.get('fits_file', ''))
    identity = dict(code=code_identity(), source=_file_digest(source),
                    configs=[_file_digest(resolve_path(manifest, v['config'])) for v in spec['variants']])
    reports = dict(preparation_seconds=time.perf_counter()-started, groups=[], cache=sweep.stats)
    groups = {}
    for index, context in enumerate(contexts):
        groups.setdefault((context.ld_profile, context.ld_uniform_basis,
                           tuple(context.quadratic_uniform_physical_bounds), context.detrending_type == 'linear_discontinuity',
                           *_spectral_trend_group(context)), []).append(index)
    output_rows = copy.deepcopy(spec['variants'])
    cache_dir = root / 'batched_posteriors'
    cache_dir.mkdir(exist_ok=True)
    polynomial_starts = {}
    for (law, basis, ld_bounds, step_trend, trend_basis, bounds_signature), indices in groups.items():
        group_tag = f'{law}_{basis}'+('_linear_discontinuity' if step_trend else '')
        if tuple(ld_bounds) != tuple(first.quadratic_uniform_physical_bounds):
            group_tag += '_' + _key(ld_bounds)[:10]
        trend_bounds=json.loads(bounds_signature)
        if trend_basis!='elapsed_days' or trend_bounds is not None:
            group_tag+='_'+_key((trend_basis,trend_bounds))[:10]
        print(f'Batched group: {group_tag} ({len(indices)} variants)', flush=True)
        members = [contexts[i] for i in indices]
        parameterization = 'maxted_smooth' if law == 'power2' else 'coefficients'
        count = len(members)
        data = first.data
        channels = len(data.wavelengths_hr)
        def options(stage):
            stages = [_resolve_stage_mcmc_kwargs(c.cfg['flags'], stage) for c in members]
            if any(value != stages[0] for value in stages[1:]):
                raise ValueError('Members of a batched group need the same sampler counts.')
            value = stages[0]
            if value['num_chains'] < 2:
                raise ValueError('At least two independent chains are required.')
            width = first.cfg['flags'].get('vmap_chunk')
            if width is None:
                width = 4 if jax.devices()[0].platform == 'cpu' else 40
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError('flags.vmap_chunk must be a positive integer.')
            return dict(chains=value['num_chains'], draws=value['num_samples'],
                warmup=value['num_warmup'], seed=first.cfg['flags'].get('random_seed', 0),
                min_ess=max(400., max(c.cfg['flags'].get('spectro_min_depth_ess', 400.) for c in members)),
                width=width,
                prime=bool(first.cfg['flags'].get('grid_prime_samplers',
                                                 jax.devices()[0].platform != 'cpu')))
        variant_flags = dict(quadratic_trend=np.array([c.detrending_type == 'quadratic' for c in members]),
                     free_ld=np.array([c.ld_prior_mode == 'uniform' for c in members]))
        white_ld = np.stack([coefficients[i][0] for i in indices])
        # The 64-node route passes the complete radius/contact audit for
        # alpha in [0,10]; retain the conservative default beyond that range.
        order = 64 if law == 'power2' and max(white_ld[:, 1].max(),
            max(coefficients[i][1][:, 1].max() for i in indices)) <= 10 else 128
        white_kwargs = dict(variant_flags, ld_fixed=white_ld)
        initial = dict(ld_coordinates=_initial_ld(white_ld, law, basis,
            parameterization=parameterization, free=variant_flags['free_ld']),
            trend_coordinates=np.tile([1., 0., 0.], (count, 1)),
            log_jitter=np.full(count, np.log(np.clip(float(data.wl_flux_err)/2, 1.01e-5, .0099))))
        if step_trend:
            from models.trends import _MAX_JUMP_WIDTH_DAYS
            cadence = float(np.median(np.diff(np.sort(data.wl_time))))
            epochs = np.array([c.t_jump_guess if c.t_jump_guess is not None else
                .5*(data.wl_time.min()+data.wl_time.max()) for c in members])
            amplitudes = np.array([c.jump_guess for c in members])
            white_kwargs.update(t_jump_guess=epochs,jump_guess=amplitudes)
            log_width = np.log(np.array([np.clip(c.step_width_days,.51*cadence,
                .99*_MAX_JUMP_WIDTH_DAYS) for c in members]))
            initial['jump'] = amplitudes
            if members[0].whitelight_trend_parameterization == 'cadence':
                initial['t_jump_offset_cadences'] = np.zeros(count)
                fraction = (log_width-np.log(.5*cadence))/np.log(_MAX_JUMP_WIDTH_DAYS/(.5*cadence))
                initial['log_width_unit_logit'] = np.log(fraction/(1-fraction))
            else:
                initial.update(t_jump=epochs,log_width=log_width)
        for name, specs in first.planet_parameter_specs.items():
            if name in ('period', 't0', 'b', 'duration', 'rprs'):
                site = latent_init_site(specs[0], 'rors_0' if name == 'rprs' else f'{name}_0')
                if site is not None:
                    initial[site[0]] = np.full(count, site[1])
        initial = sweep.warm_start('white', initial, law=law,
                                   parameter_specs=first.planet_parameter_specs)
        if step_trend:
            for member,context in enumerate(members):
                previous = polynomial_starts.get((law,context.ld_prior_mode),{})
                for site,value in previous.items():
                    if site in initial and np.shape(value) == np.shape(initial[site][member]):
                        initial[site][member] = value
                        sweep.stats['warm_started_sites'] += 1
        white_window = _white_transit_window(data.wl_time, first.planet_parameter_specs)
        white_window_options = {}
        if white_window and len(white_window) < len(data.wl_time):
            white_window_options['transit_window_indices'] = white_window
            print(f'Batched white-light prior envelope: {len(white_window)}/{len(data.wl_time)} active cadences', flush=True)
        white_model = create_batched_model(law=law, stage='white',
            parameter_priors=first.planet_parameter_specs, ld_uniform_basis=basis,
            ld_bounds=ld_bounds, power2_parameterization=parameterization,
            power2_quadrature_order=128, step_trend=step_trend,
            step_parameterization=members[0].whitelight_trend_parameterization, **white_window_options)
        fingerprint = _key((identity, indices, 'white', options('whitelight')))
        white, white_report = _sample(white_model, data.wl_time,
            np.full((count, len(data.wl_time)), data.wl_flux_err),
            np.tile(data.wl_flux, (count, 1)), initial, white_kwargs, options('whitelight'),
            cache_dir / f'{group_tag}_white.npz', fingerprint)
        sweep.remember('white', white, law=law)
        if not step_trend:
            starts = {site:np.median(np.asarray(white[site]),axis=0) for site in initial}
            for member,context in enumerate(members):
                polynomial_starts.setdefault((law,context.ld_prior_mode),
                    {site:value[member].copy() for site,value in starts.items()})
        medians = {name: np.median(np.asarray(white[name]), axis=0) for name in
                   ('period_0', 'duration_0', 't0_0', 'b_0', 'rors_0')}
        hr_ld = np.concatenate([coefficients[i][1] for i in indices])
        hr_kwargs = {name: np.repeat(value, channels) for name, value in variant_flags.items()}
        hr_kwargs['ld_fixed'] = hr_ld
        for name, site in (('PERIOD', 'period_0'), ('mu_duration', 'duration_0'),
                           ('mu_t0', 't0_0'), ('mu_b', 'b_0')):
            hr_kwargs[name] = np.repeat(medians[site], channels)[:, None]
        from models.harmonica.core import harmonica_a_rs_from_duration
        hr_kwargs['mu_a_rs'] = np.repeat(np.asarray(harmonica_a_rs_from_duration(
            medians['period_0'], medians['duration_0'], medians['b_0'], medians['rors_0'],
            ecc=first.planet_parameter_specs['ecc'][0].value,
            omega=first.planet_parameter_specs['omega'][0].value)), channels)[:, None]
        hr_kwargs['mu_ecc'] = first.planet_parameter_specs['ecc'][0].value
        hr_kwargs['mu_omega'] = first.planet_parameter_specs['omega'][0].value
        hr_init = dict(ld_coordinates=_highres_ld_start(white, white_ld, hr_ld,
            channels, law, basis, hr_kwargs['free_ld'], parameterization),
            trend_coordinates=np.repeat(np.median(np.asarray(white['trend_coordinates']), axis=0),
                                         channels, axis=0),
            log_jitter=np.tile(np.log(.5*np.median(np.asarray(data.flux_err_hr), axis=1)), count),
            rors=np.repeat(medians['rors_0'], channels)[:, None])
        if step_trend:
            epoch,jump,width = [np.median(np.asarray(white[name]),axis=0) for name in ('t_jump','jump','width')]
            templates = jump[:,None]*np.asarray(jax.nn.sigmoid(
                (jnp.asarray(data.time)[None,:]-epoch[:,None])/width[:,None]))
            hr_kwargs.update(jump_template_bank=templates,step_index=np.repeat(np.arange(count),channels))
            hr_init['A_jump'] = np.ones(count*channels)
        builder_kwargs = dict(law=law, stage='highres', ld_uniform_basis=basis,
                              ld_bounds=ld_bounds,
                              power2_parameterization=parameterization, power2_quadrature_order=order,
                              step_trend=step_trend,trend_basis=trend_basis,trend_bounds=trend_bounds)
        if trend_basis=='batman_normalized':
            from models.spectro_trends import normalized_trend_initial_values
            physical={name:np.repeat(np.median(np.asarray(white[name]),axis=0),channels)
                      for name in ('c','v','v2')}
            normalized=normalized_trend_initial_values(data.time,physical)
            hr_init['trend_coordinates']=np.stack([normalized['trend_'+name] for name in ('c','v','v2')],axis=-1)
        window = (_transit_window(data.time, medians)
                  if first.planet_parameter_specs['ecc'][0].value == 0 else tuple(range(len(data.time))))
        if window and len(window) < len(data.time):
            builder_kwargs['transit_window_indices'] = window
            print(f'Batched transit window: {len(window)}/{len(data.time)} active cadences', flush=True)
        table_report = None
        hr_model = create_batched_model(**builder_kwargs)
        fingerprint = _key((identity, indices, 'highres', medians, options('highres')))
        highres, hr_report = _sample(hr_model, data.time,
            _RepeatedChannels(data.flux_err_hr, count), _RepeatedChannels(data.flux_hr, count),
            hr_init, hr_kwargs, options('highres'), cache_dir / f'{group_tag}_highres.npz', fingerprint,
            stream=True)
        reports['groups'].append(dict(law=law, basis=basis, step_trend=step_trend,variants=[spec['variants'][i]['name'] for i in indices],
                                     white=white_report, highres=hr_report, table=table_report))
        for member, index in enumerate(indices):
            variant = output_rows[index]
            name = variant['name']
            begin, end = member*channels, (member+1)*channels
            fit_dir, input_dir = root / 'batched_fits' / name, root / 'batched_stage_inputs' / name
            (fit_dir / 'chunks').mkdir(parents=True, exist_ok=True)
            input_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(schema_version=1, model_builder=dict(
                identity='koala.marginalization.batched.create_batched_model', kwargs=builder_kwargs),
                t=np.asarray(data.time), yerr=np.asarray(data.flux_err_hr), indiv_y=np.asarray(data.flux_hr),
                model_kwargs=_lane_kwargs(hr_kwargs, slice(begin, end)),
                meta=dict(stage_kind='high_resolution', stage_label=name+'_high_resolution',
                          wavelength=np.asarray(data.wavelengths_hr)))
            with (input_dir / f'{name}_high_resolution_inputs.pkl').open('wb') as stream:
                pickle.dump(payload, stream, protocol=5)
            for suffix in ('pkl', 'npz'):
                for old in (fit_dir/'chunks').glob(f'{name}_chunk_*.{suffix}'):
                    old.unlink()
            highres.export(begin, end, fit_dir/'chunks', name)
            _save_npz(fit_dir / 'white_light_posterior.npz',
                      **{key: np.asarray(value)[:, member:member+1] for key, value in white.items()})
            (fit_dir / 'diagnostics.json').write_text(json.dumps(dict(white=white_report, highres=hr_report), indent=2))
            variant.update(fit_dir=str(fit_dir.relative_to(root)), stage_inputs=str(input_dir.relative_to(root)))
        del white, highres, white_model, hr_model
        sweep.finish_target(first.cfg['flags'].get('grid_cache_memory_mb'))
    result = root / 'analysis_batched.yaml'
    result.write_text(yaml.safe_dump(dict(spec, variants=output_rows, execution='independent batched variants'), sort_keys=False))
    reports['total_seconds'] = time.perf_counter()-started
    (root / 'batched_fit_timings.json').write_text(json.dumps(reports, indent=2))
    return result
