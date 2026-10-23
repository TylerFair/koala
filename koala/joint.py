"""Joint white light with ordinary per-dataset products and spectroscopy."""

from koala.console import debug, info
from .output_paths import OutputPaths

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.handlers import Messenger, scope, seed, trace
from models.priors import sample_parameter, latent_init_site
from .config import load_config, parse_planet_parameter_specs, parse_parameter_spec
from .artifacts import _science_artifact_fingerprint
from .readers import TIME_SYSTEM
from .white_light import (
    prepare_white_light_ld, _initialize_white_light, _whitelight_geometry_sites,
    _sample_white_light, run_white_light_stage,
)


def load_joint_members(cfg, config_path):
    """Resolve member paths and reject ambiguous orbital/configuration choices."""
    block = cfg['joint']
    if not isinstance(block, dict):
        raise ValueError('joint must be a mapping.')
    unknown = set(block) - {'members', 'apply_to', 'shared', 'visits'}
    if unknown:
        raise ValueError(f'Unknown joint keys: {sorted(unknown)}')
    shared = block.get('shared', ['a_rs', 'b'])
    if (not isinstance(shared, list) or len(shared) != 2
            or any(not isinstance(key, str) for key in shared)
            or set(shared) != {'a_rs', 'b'}):
        raise ValueError('joint.shared must contain exactly [a_rs, b].')
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    output = Path(cfg.get('output_dir', 'results/joint'))
    if not output.is_absolute():
        output = base / output
    members = block.get('members')
    apply_to = block.get('apply_to', [])
    if not isinstance(members, list) or not members:
        raise ValueError('joint.members must be a nonempty list of config paths.')
    if not isinstance(apply_to, list):
        raise ValueError('joint.apply_to must be a list of config paths.')
    seen_paths, seen_names, result = set(), set(), []
    period = None
    for filename in members + apply_to:
        if not isinstance(filename, str):
            raise ValueError('joint members and apply_to entries must be config paths.')
        path = (base / filename).resolve()
        name = path.stem
        if name == 'joint':
            raise ValueError('The member file stem joint is reserved for joint outputs.')
        if path in seen_paths or name in seen_names:
            raise ValueError(f'Joint member listed twice or duplicate file stem: {filename}')
        seen_paths.add(path)
        seen_names.add(name)
        member = deepcopy(load_config(path))
        if 'joint' in member:
            raise ValueError(f'{name}: nested joint configurations are not supported.')
        specs = parse_planet_parameter_specs(member['planet'])
        if len(specs['period']) != 1 or not specs['period'][0].fixed:
            raise ValueError(f'{name}: joint white light requires one planet with a fixed period.')
        if specs['ecc'][0].value != 0:
            raise ValueError(f'{name}: joint white light requires circular orbits.')
        p = specs['period'][0].value
        if period is not None and p != period:
            raise ValueError(f'{name}: period mismatch ({p} versus {period}).')
        period = p
        flags = member.setdefault('flags', {})
        if flags.get('spectro_joint_geometry', False):
            raise ValueError(f'{name}: spectro_joint_geometry must be false for a joint white-light run.')
        if len(result) < len(members) and (
            flags.get('transit_engine', 'jaxoplanet') != 'jaxoplanet'
            or flags.get('light_curve_model', 'transit') != 'transit'
        ):
            raise ValueError(f'{name}: joint white light currently requires jaxoplanet transits.')
        if flags.get('whitelight_reuse_samples'):
            raise ValueError(f'{name}: whitelight_reuse_samples cannot define a joint posterior.')
        member['host_device'] = cfg.get('host_device', 'auto')
        member['output_dir'] = str(output / name)
        result.append((name, member, path))
    first = result[0][1]['planet']
    first_specs = parse_planet_parameter_specs(first)
    overrides = cfg.get('planet', {})
    if not isinstance(overrides, dict):
        raise ValueError('The joint planet block must be a mapping of a_rs and b priors.')
    if set(overrides) - {'a_rs', 'b'}:
        raise ValueError('The joint planet block accepts only a_rs and b.')
    shared_specs = {}
    for key in ('a_rs', 'b'):
        raw = overrides.get(key)
        if raw is None and key in first_specs:
            raw = {field: value for field, value in vars(first_specs[key][0]).items()
                   if field != 'name' and value is not None}
        if raw is None and key == 'a_rs':
            duration = first_specs['duration'][0]
            if not duration.fixed:
                raise ValueError('A duration prior cannot be shared: supply an explicit joint planet.a_rs prior.')
            # A fixed duration supplies only the starting/fixed orbit, at the
            # first member radius. Durations of other bands remain derived.
            p = period
            b = first_specs['b'][0].value
            k = first_specs['rprs'][0].value
            value = np.sqrt(b*b + ((1+k)**2-b*b) / np.sin(np.pi*duration.value/p)**2)
            raw = {'value': float(value), 'prior': 'fixed'}
        shared_specs[key] = parse_parameter_spec(key, raw)
    # All member builders use a radius-independent orbit and the same shared
    # specifications; their geometry sites are injected by compose_joint_model.
    for _, member, _ in result:
        member['planet'].pop('duration', None)
        for key, spec in shared_specs.items():
            member['planet'][key] = {field: value for field, value in vars(spec).items()
                                     if field != 'name' and value is not None}
    return result[:len(members)], result[len(members):], shared_specs, output


def group_visits(names, ranges, visits='auto'):
    """Connected overlapping time ranges form visits, retaining member order."""
    if visits == 'auto':
        groups = []
        for i in sorted(range(len(names)), key=lambda i: ranges[i][0]):
            if groups and ranges[i][0] <= max(ranges[j][1] for j in groups[-1]):
                groups[-1].append(i)
            else:
                groups.append([i])
        return sorted([sorted(group) for group in groups], key=lambda group: group[0])
    if not isinstance(visits, list) or not visits or any(not isinstance(g, list) or not g for g in visits):
        raise ValueError('joint.visits must be auto or nonempty lists of member names.')
    flat = [name for group in visits for name in group]
    if (len(flat) != len(names) or any(not isinstance(name, str) for name in flat)
            or len(set(flat)) != len(flat) or set(flat) != set(names)):
        raise ValueError('joint.visits must cover every member exactly once, using file stems.')
    return [[names.index(name) for name in group] for group in visits]


class _InjectSites(Messenger):
    """Replace local prior sites by deterministic shared values, without density."""
    def __init__(self, values):
        super().__init__()
        self.values = values

    def process_message(self, msg):
        if msg['type'] in {'sample', 'deterministic'} and msg['name'] in self.values:
            msg['type'] = 'deterministic'
            msg['value'] = self.values[msg['name']]
            msg.pop('fn', None)
            msg['kwargs'] = {}


def _latent_value(spec, site, value):
    latent = latent_init_site(spec, site)
    if latent is None:
        return site, value
    return latent[0], jnp.log(value) if spec.prior == 'log_uniform' else value


def compose_joint_model(contexts, pieces, visits, shared_specs):
    """Compose unchanged member likelihoods; each shared prior occurs once."""
    visit_of = {i: v for v, group in enumerate(visits) for i in group}

    def model(t, yerr, y=None, prior_params=None):
        shared = {key: sample_parameter(spec, f'{key}_0')
                  for key, spec in shared_specs.items()}
        epochs, spots = {}, {}
        for v, group in enumerate(visits):
            with scope(prefix=f'visit{v}'):
                spec = contexts[group[0]].planet_parameter_specs['t0'][0]
                epochs[v] = sample_parameter(spec, 't0_0')
                for suffix in ('', '2'):
                    owner = next((i for i in group if ('2spot' if suffix else 'spot')
                                  in contexts[i].detrending_type), None)
                    if owner is not None:
                        prior = pieces[owner][0]
                        guess = prior.get(f'spot_guess{suffix}', prior.get('spot_guess'))
                        cadence = jnp.median(jnp.diff(jnp.sort(t[owner])))
                        if contexts[owner].whitelight_trend_parameterization == 'cadence':
                            offset = numpyro.sample(f'spot_mu{suffix}_offset_cadences',
                                numpyro.distributions.Normal(0., .01 / cadence))
                            width = numpyro.sample(f'spot_sigma{suffix}_cadences',
                                numpyro.distributions.Uniform(1e-4 / cadence, .1 / cadence))
                            spots[v, f'spot_mu{suffix}'] = numpyro.deterministic(
                                f'spot_mu{suffix}', guess + cadence * offset)
                            spots[v, f'spot_sigma{suffix}'] = numpyro.deterministic(
                                f'spot_sigma{suffix}', cadence * width)
                        else:
                            spots[v, f'spot_mu{suffix}'] = numpyro.sample(
                                f'spot_mu{suffix}', numpyro.distributions.Normal(guess, .01))
                            spots[v, f'spot_sigma{suffix}'] = numpyro.sample(
                                f'spot_sigma{suffix}', numpyro.distributions.Uniform(1e-4, .1))
        for i, (context, piece) in enumerate(zip(contexts, pieces)):
            prior, _, _, member_model = piece
            v = visit_of[i]
            values = dict(_latent_value(shared_specs[key], f'{key}_0', value)
                          for key, value in shared.items())
            # Fixed sites and deterministic physical aliases are injected too.
            values.update({f'{key}_0': value for key, value in shared.items()})
            spec = context.planet_parameter_specs['t0'][0]
            values.update([_latent_value(spec, 't0_0', epochs[v])])
            values['t0_0'] = epochs[v]
            cadence = jnp.median(jnp.diff(jnp.sort(t[i])))
            for (visit, name), value in spots.items():
                if visit != v:
                    continue
                if context.whitelight_trend_parameterization == 'cadence':
                    if 'mu' in name:
                        suffix = '2' if name.endswith('2') else ''
                        guess = prior.get(f'spot_guess{suffix}', prior.get('spot_guess'))
                        values[name + '_offset_cadences'] = (value - guess) / cadence
                    else:
                        values[name + '_cadences'] = value / cadence
                values[name] = value
            with scope(prefix=f'member{i}'), _InjectSites(values):
                member_model(t[i], yerr[i], y=None if y is None else y[i], prior_params=prior)
    return model


def run_joint(cfg, config_path=None, verbose=False):
    """Fit all white lights, write their posterior products, then spectroscopy."""
    from .pipeline import run
    from .console import (configure, parent_log, finish_run, parameter_summary,
                          joint_presentation, clipping_description)
    from .config import describe_planet_parameter_specs
    from models.mcmc_diagnostics import PosteriorSamples
    started = time.perf_counter()
    members, apply_to, shared_specs, output = load_joint_members(cfg, config_path)
    configure(verbose, output, config_path)
    with parent_log():
        plots = str(cfg.get('flags', {}).get('plots', 'full')).strip().lower()
        presentation = {'count': len(members), 'reading': True, 'plots': cfg.get('flags', {}).get('plots')}
        if cfg['joint'].get('visits', 'auto') != 'auto':
            group_visits([name for name, _, _ in members], [], cfg['joint']['visits'])
        with joint_presentation(presentation):
            contexts = [run(deepcopy(member), config_path=path, prepare_only=True, verbose=verbose) for _, member, path in members]
        for line in describe_planet_parameter_specs({
                key: entries for key, entries in contexts[0].planet_parameter_specs.items()
                if key in members[0][1]['planet']}):
            info(line)
        first = contexts[0]
        ld_description = first.ld_prior_mode + ' priors'
        if first.ld_profile == 'quadratic' and first.ld_prior_mode == 'uniform' and first.ld_uniform_basis == 'uplus_uminus':
            ld_description = 'uniform priors on u1 + u2 and u1 - u2'
        info(f'Limb darkening: {first.ld_profile} law with {ld_description}')
        info('')
        presentation['reading'] = False
        visits = group_visits([name for name, _, _ in members],
                              [(float(c.data.wl_time.min()), float(c.data.wl_time.max())) for c in contexts],
                              cfg['joint'].get('visits', 'auto'))
        pieces = []
        for context in contexts:
            center, width = prepare_white_light_ld(context)
            kw = dict(context._engine_wl_kw, parameter_priors=context.planet_parameter_specs,
                      gp_solver=context.cfg.get('gp_solver'), gp_assume_sorted=True)
            pieces.append(_initialize_white_light(
                context=context, U_mu_wl=center, U_sigma_wl=width, _engine_wl_kw=kw,
                geometry_sites=_whitelight_geometry_sites(context.planet_parameter_specs, 1, 'a_rs')))
        model = compose_joint_model(contexts, pieces, visits, shared_specs)
        times = tuple(c.data.wl_time for c in contexts)
        errors = tuple(c.data.wl_flux_err for c in contexts)
        fluxes = tuple(c.data.wl_flux for c in contexts)
        # Trace once at the configured starts to identify exactly the free sites.
        starts = {}
        for key, spec in shared_specs.items():
            init = latent_init_site(spec, f'{key}_0')
            if init:
                starts[init[0]] = init[1]
        for v, group in enumerate(visits):
            init = latent_init_site(contexts[group[0]].planet_parameter_specs['t0'][0], 't0_0')
            if init:
                starts[f'visit{v}/{init[0]}'] = init[1]
            for suffix in ('', '2'):
                owner = next((i for i in group if ('2spot' if suffix else 'spot') in contexts[i].detrending_type), None)
                if owner is not None:
                    c = contexts[owner]
                    if c.whitelight_trend_parameterization == 'cadence':
                        cadence = jnp.median(jnp.diff(jnp.sort(c.data.wl_time)))
                        starts[f'visit{v}/spot_mu{suffix}_offset_cadences'] = 0.
                        starts[f'visit{v}/spot_sigma{suffix}_cadences'] = jnp.clip(getattr(c, f'spot_sigma{suffix}'), 1.01e-4, .0999) / cadence
                    else:
                        starts[f'visit{v}/spot_mu{suffix}'] = getattr(c, f'spot_mu{suffix}')
                        starts[f'visit{v}/spot_sigma{suffix}'] = jnp.clip(getattr(c, f'spot_sigma{suffix}'), 1.01e-4, .0999)
        for i, piece in enumerate(pieces):
            starts.update({f'member{i}/{k}': (jnp.clip(value, 1e-5, .09999)
                               if k in {'spot_amp', 'spot_amp2'} else value)
                           for k, value in piece[1].items()})
        with numpyro.handlers.substitute(data=starts):
            sites = trace(seed(model, contexts[0].key_master)).get_trace(times, errors, y=fluxes)
        starts = {name: site['value'] for name, site in sites.items()
                  if site['type'] == 'sample' and not site['is_observed']}
        joint_dir = Path(OutputPaths(output).state_dir)
        joint_dir.mkdir(parents=True, exist_ok=True)
        context = SimpleNamespace(**vars(contexts[0]))
        context.output_dir = str(output)
        context.cfg = {'flags': {}}
        context.data = SimpleNamespace(wl_time=times, wl_flux_err=errors, wl_flux=fluxes)
        context.ld_profile = 'joint'
        context.uses_surface_model = any(c.uses_surface_model or (
            c.ld_profile == 'power2' and c.data.wl_time.size > 5000) for c in contexts)
        context.whitelight_laplace_options = dict(context.whitelight_laplace_options, mass_matrix='laplace')
        # The power-2 stellar-prior jitter metric correction, per member.
        context.jitter_metric_sites = joint_jitter_metric_sites(contexts)
        context.console_sampling_heading = (
            f'Joint white light: {len(members)} observations sharing a_rs and b, '
            f'4 chains of {context.whitelight_mcmc_kwargs.get("num_samples", 1000)} draws')
        samples = _sample_white_light(
            context=context, hyper_params_wl={}, science_sites={name: name for name in starts},
            soln=starts, whitelight_mcmc_kwargs=dict(context.whitelight_mcmc_kwargs, num_chains=4),
            whitelight_model_for_run=model)
        for key, spec in shared_specs.items():
            name = f'{key}_0'
            if spec.free and name in samples:
                values = np.asarray(samples[name])
                info(parameter_summary(key, float(np.median(values)), float(np.std(values))))
        for visit_index, group in enumerate(visits):
            name = f'visit{visit_index}/t0_0'
            if contexts[group[0]].planet_parameter_specs['t0'][0].free and name in samples:
                values = np.asarray(samples[name])
                label = 't0' if len(visits) == 1 else f't0 visit {visit_index + 1}'
                info(parameter_summary(label, float(np.median(values)), float(np.std(values)), TIME_SYSTEM))
        for member_index, member_context in enumerate(contexts):
            name = f'member{member_index}/rors_0'
            if member_context.planet_parameter_specs['rprs'][0].free and name in samples:
                values = np.asarray(samples[name])
                order = getattr(member_context, 'order', None)
                label = f'rprs order {order}' if order is not None else f'rprs observation {member_index + 1}'
                info(parameter_summary(label, float(np.median(values)), float(np.std(values))))
        with np.load(joint_dir / 'whitelight_samples_by_chain.npz') as saved:
            grouped = dict(saved)
        fingerprint = _science_artifact_fingerprint('joint_white_light', {
            'config': cfg, 'members': [c.cfg for c in contexts],
            'time': times, 'flux': fluxes, 'error': errors, 'pieces': [p[0] for p in pieces],
        })
        _write_joint_products(output, samples, members, visits, plots=plots)
        products = []
        presentation['collect_residuals'] = True
        for i, (_, member, path) in enumerate(members):
            prefix = f'member{i}/'
            flat = PosteriorSamples({name[len(prefix):]: value for name, value in samples.items()
                                     if name.startswith(prefix)}, num_chains=4)
            member_grouped = {name[len(prefix):]: value for name, value in grouped.items()
                              if name.startswith(prefix)}
            posterior = {'samples': flat, 'grouped': member_grouped, 'fingerprint': fingerprint,
                         'diagnostics_path': str(joint_dir / 'whitelight_mcmc_diagnostics.json')}
            # Write each member's ordinary white-light products once. Presentation
            # collects their existing residual statistics before spectroscopy starts.
            member_context = contexts[i]
            member_context.joint_member_products = True
            with joint_presentation(presentation):
                configure(verbose, member_context.output_dir, path)
                prepared = run_white_light_stage(member_context, posterior=posterior)
            products.append((member, path, posterior, prepared))
        residuals = presentation.get('residuals', [])
        if residuals:
            scatter = ', '.join(f'{value:.0f} ppm ({label})' for value, _, _, label in residuals)
            # Keep differing clipping thresholds explicit rather than hiding them.
            thresholds = {sigma for _, _, sigma, _ in residuals}
            if len(thresholds) == 1:
                clipped = clipping_description(sum(n for _, n, _, _ in residuals), residuals[0][2])
            else:
                clipped = '; '.join(f'{clipping_description(n, sigma)} ({label})'
                                    for _, n, sigma, label in residuals)
            info(f'  Residual scatter {scatter}; {clipped}')
        presentation['collect_residuals'] = False
        for member, path, posterior, prepared in products:
            with joint_presentation(presentation):
                run(member, config_path=path, verbose=verbose, white_light_posterior=posterior,
                    prepared_white_light=prepared)
        for _, member, path in apply_to:
            presentation['reading'] = True
            for key in shared_specs:
                member['planet'][key] = {'value': float(np.median(samples[f'{key}_0'])), 'prior': 'fixed'}
            with joint_presentation(presentation):
                run(member, config_path=path, verbose=verbose)
        try:
            from .joint_products import plot_joint_spectra
            plot_joint_spectra(output, plots=plots)
        except Exception as error:  # the member spectra are already written; this figure is a summary
            info('The joint spectrum figure could not be written; see koala.log.')
            debug(f'WARNING: the joint spectrum figure was not written: {error}')
        finish_run(output, started)


def joint_jitter_metric_sites(contexts):
    """Scoped jitter sites that get the single-dataset power-2 metric correction."""
    return [f'member{i}/log_jitter' for i, c in enumerate(contexts)
            if c.ld_profile == 'power2' and getattr(c, 'ld_prior_mode', None) == 'stellarprior'
            and not (getattr(c, 'trend_priors', None) or {}).get('jitter')]


def _write_joint_products(directory, samples, members, visits, *, plots='full'):
    import pandas as pd
    rows = []
    for name, values in samples.items():
        values = np.asarray(values)
        if values.ndim == 1:
            columns = [(name, values)]
        elif values.ndim == 2 and values.shape[1] <= 4:
            columns = [(f'{name}[{j}]', values[:, j]) for j in range(values.shape[1])]
        else:
            continue  # Per-integration error arrays belong in member products.
        for label, column in columns:
            low, median, high = np.quantile(column, [.16, .5, .84])
            rows.append(dict(parameter=label, median=median, lower_68=low, upper_68=high))
    pd.DataFrame(rows).to_csv(OutputPaths(directory).joint_table('params'), index=False)
    if plots == 'minimal':
        return
    import corner
    import matplotlib.pyplot as plt
    from .plotting_style import apply_publication_style, accent_color_for_label
    apply_publication_style()
    columns, labels = [], []
    selected = [('a_rs_0', 'a/R*'), ('b_0', 'b')]
    selected += [(f'visit{v}/t0_0', f'Visit {v} t0 [d]') for v in range(len(visits))]
    for name, label in selected:
        values = np.asarray(samples[name])
        if np.ptp(values) > 0:
            if '/t0_' in name:
                values = (values - np.median(values)) * 86400
                label = label.replace('[d]', 'offset [s]')
            columns.append(values)
            labels.append(label)
    if columns:
        fig = corner.corner(np.column_stack(columns), labels=labels,
                            quantiles=[.16, .5, .84], show_titles=True,
                            color=accent_color_for_label('joint'), max_n_ticks=4,
                            label_kwargs={'fontsize': 10}, title_kwargs={'fontsize': 10})
        target = members[0][1].get('planet', {}).get('name', '')
        fig.suptitle(f'{target} | Joint white-light posterior', fontsize=16, y=1.02)
        fig.savefig(OutputPaths(directory).figure('joint', 'corner'), dpi=250, bbox_inches='tight')
        plt.close(fig)
    debug('Joint white light shares a_rs and b with a fixed circular period.')
    for i, (name, _, _) in enumerate(members):
        debug(f'Observation {i + 1}: {name}')
    for v, group in enumerate(visits):
        debug(f'Visit {v}: {", ".join(members[i][0] for i in group)}')
