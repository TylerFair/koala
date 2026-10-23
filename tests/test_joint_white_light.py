"""Exact targets for composing ordinary white-light member models."""
from types import SimpleNamespace
import numpy as np
import pytest
import jax
import numpyro
import jax.numpy as jnp
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density
from koala.joint import compose_joint_model, group_visits, load_joint_members
from models.jaxoplanet.builder import create_whitelight_model
from planet_specs import default_planet_specs


def make_piece(trend='linear', profile='quadratic', mode='gaussian', cadence=False):
    specs = default_planet_specs(param_method='a_rs')
    context = SimpleNamespace(planet_parameter_specs=specs, detrending_type=trend,
                             whitelight_trend_parameterization='cadence' if cadence else 'physical')
    prior = dict(u=jnp.array([.3, .2]) if profile == 'quadratic' else jnp.array([.6, .7]),
                 u_sigma=jnp.array([.04, .08]), spot_guess=1., spot_guess2=1.03)
    model = create_whitelight_model(detrend_type=trend, ld_profile=profile,
        ld_mode=mode, param_method='a_rs', parameter_priors=specs,
        trend_parameterization=context.whitelight_trend_parameterization)
    return context, (prior, {}, mode, model)


def evaluate_case(pieces):
    contexts = [p[0] for p in pieces]
    pieces = [p[1] for p in pieces]
    shared = {key: contexts[0].planet_parameter_specs[key][0] for key in ('a_rs', 'b')}
    joint = compose_joint_model(contexts, pieces, [list(range(len(pieces)))], shared)
    times = tuple(jnp.linspace(.9, 1.1, 39 + i*2) for i in range(len(pieces)))
    errors = tuple(jnp.full_like(t, .001) for t in times)
    flux = tuple(jnp.ones_like(t) for t in times)
    for draw in range(4):
        sites = trace(seed(joint, jax.random.PRNGKey(draw))).get_trace(times, errors, y=flux)
        for i in range(1, len(pieces)):
            for key in ('a_rs_0', 'b_0', 't0_0', 'spot_mu', 'spot_sigma', 'spot_mu2', 'spot_sigma2'):
                if f'member0/{key}' in sites and f'member{i}/{key}' in sites:
                    np.testing.assert_allclose(sites[f'member0/{key}']['value'], sites[f'member{i}/{key}']['value'], rtol=1e-14)
        values = {name: site['value'] for name, site in sites.items()
                  if site['type'] == 'sample' and not site['is_observed']}
        joint_density, _ = log_density(joint, (times, errors), dict(y=flux), values)
        singles, duplicate = 0., 0.
        for i, piece in enumerate(pieces):
            local = {name.split('/', 1)[1]: site['value'] for name, site in sites.items()
                     if name.startswith(f'member{i}/') and site['type'] in {'sample', 'deterministic'}}
            density, single_sites = log_density(piece[3], (times[i], errors[i]),
                dict(y=flux[i], prior_params=piece[0]), local)
            singles += density
            if i:
                for name in ('log_a_rs_0', '_b_0', 't0_0', 'spot_mu', 'spot_sigma', 'spot_mu2', 'spot_sigma2',
                             'spot_mu_offset_cadences', 'spot_sigma_cadences',
                             'spot_mu2_offset_cadences', 'spot_sigma2_cadences'):
                    if name in single_sites and single_sites[name]['type'] == 'sample':
                        site = single_sites[name]
                        duplicate += jnp.sum(site['fn'].log_prob(site['value']))
        np.testing.assert_allclose(joint_density, singles-duplicate, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('trend,cadence', [('linear', False), ('spot', False), ('spot', True), ('2spot', False)])
def test_one_member_exact_density(trend, cadence):
    evaluate_case([make_piece(trend, cadence=cadence)])


def test_two_member_exact_density_mixed_ld():
    evaluate_case([make_piece(), make_piece(profile='power2', mode='stellarprior')])


def test_shared_spot_density():
    evaluate_case([make_piece('2spot'), make_piece('2spot')])
    evaluate_case([make_piece('spot', cadence=True), make_piece('spot')])


def test_visit_grouping():
    assert group_visits(['a', 'b', 'c'], [(0, 2), (4, 5), (1, 3)]) == [[0, 2], [1]]
    assert group_visits(['a', 'b', 'c'], [(0, 2), (4, 5), (1, 3)], [['a', 'b'], ['c']]) == [[0, 1], [2]]
    for groups in ([['a']], [['a', 'b'], ['b', 'c']], [['a', 'b', 'x']]):
        with pytest.raises(ValueError, match='cover every member'):
            group_visits(['a', 'b', 'c'], [], groups)


def test_validation(tmp_path):
    import yaml
    planet = dict(name='test', period=dict(value=3., prior='fixed'),
        t0=dict(value=1., prior='fixed'), a_rs=dict(value=10., prior='fixed'),
        b=dict(value=.2, prior='fixed'), rprs=dict(value=.1, prior='fixed'))
    def write(name, p):
        (tmp_path / name).write_text(yaml.safe_dump(dict(planet=p)))
    write('a.yaml', planet)
    write('b.yaml', planet)
    def load(block):
        return load_joint_members(dict(joint=block, host_device='cpu'), tmp_path/'joint.yaml')
    members, _, _, _ = load(dict(members=['a.yaml', 'b.yaml']))
    assert members[0][1]['host_device'] == 'cpu'
    with pytest.raises(ValueError, match='twice'):
        load(dict(members=['a.yaml', './a.yaml']))
    with pytest.raises(ValueError, match='shared'):
        load(dict(members=['a.yaml'], shared=['duration', 'b']))
    write('b.yaml', dict(planet, period=dict(value=4., prior='fixed')))
    with pytest.raises(ValueError, match='period mismatch'):
        load(dict(members=['a.yaml', 'b.yaml']))
    write('b.yaml', dict(planet, ecc=dict(value=.1, prior='fixed')))
    with pytest.raises(ValueError, match='circular'):
        load(dict(members=['b.yaml']))
    p = dict(planet)
    p.pop('a_rs')
    p['duration'] = dict(value=.1, prior='uniform', low=.05, high=.2)
    write('b.yaml', p)
    with pytest.raises(ValueError, match='explicit joint planet.a_rs'):
        load(dict(members=['b.yaml']))
    p['duration']['prior'] = 'fixed'
    p['duration'].pop('low')
    p['duration'].pop('high')
    write('b.yaml', p)
    converted, _, specs, _ = load(dict(members=['b.yaml']))
    assert specs['a_rs'].fixed and specs['a_rs'].value > 1.
    assert 'duration' not in converted[0][1]['planet']


@pytest.mark.parametrize('trend', ['linear', 'spot'])
def test_one_member_same_laplace_map(trend):
    from models.independent_nuts import prepare_laplace_metric
    context, piece = make_piece(trend, mode='fixed', cadence=trend == 'spot')
    shared = {key: context.planet_parameter_specs[key][0] for key in ('a_rs', 'b')}
    joint = compose_joint_model([context], [piece], [[0]], shared)
    t = jnp.linspace(.9, 1.1, 45)
    err = jnp.full_like(t, .0004)
    starts = dict(log_a_rs_0=jnp.log(10.), _b_0=.2, t0_0=1., rors_0=.1,
                  c=1., v=0., log_jitter=jnp.log(.0001),
                  spot_amp=.002, spot_mu=1., spot_sigma=.01)
    if trend == 'spot':
        starts.update(spot_mu_offset_cadences=0.,
                      spot_sigma_cadences=.01 / jnp.median(jnp.diff(t)))
    original = trace(seed(piece[3], jax.random.PRNGKey(0)))
    with numpyro.handlers.substitute(data=starts):
        sites = original.get_trace(t, err, y=jnp.ones_like(t), prior_params=piece[0])
    y = sites['obs']['fn'].loc
    joint_starts = {}
    for name, value in starts.items():
        if name in ('log_a_rs_0', '_b_0'):
            joint_starts[name] = value
        elif name == 't0_0' or name.startswith(('spot_mu', 'spot_sigma')):
            joint_starts['visit0/'+name] = value
        else:
            joint_starts['member0/'+name] = value
    kwargs = dict(max_iterations=40, map_execution='split', eigenvalue_floor=1e-12)
    single = prepare_laplace_metric(piece[3], jax.random.PRNGKey(1), starts, t, err,
        model_kwargs=dict(y=y, prior_params=piece[0]), **kwargs)
    combined = prepare_laplace_metric(joint, jax.random.PRNGKey(1), joint_starts, (t,), (err,),
        model_kwargs=dict(y=(y,)), **kwargs)
    for name, value in single.unconstrained_map.items():
        target = name if name in ('log_a_rs_0', '_b_0') else ('visit0/' if name == 't0_0' or name.startswith(('spot_mu', 'spot_sigma')) else 'member0/') + name
        np.testing.assert_allclose(value, combined.unconstrained_map[target], rtol=1e-6, atol=1e-6)


def test_distinct_visits_have_independent_epoch_priors():
    contexts, pieces = zip(make_piece(), make_piece())
    shared = {key: contexts[0].planet_parameter_specs[key][0] for key in ('a_rs', 'b')}
    model = compose_joint_model(contexts, pieces, [[0], [1]], shared)
    times = (jnp.linspace(.9, 1.1, 31),) * 2
    errors = (jnp.full_like(times[0], .001),) * 2
    flux = (jnp.ones_like(times[0]),) * 2
    sites = trace(seed(model, jax.random.PRNGKey(5))).get_trace(times, errors, y=flux)
    assert sites['visit0/t0_0']['type'] == sites['visit1/t0_0']['type'] == 'sample'
    assert sites['visit0/t0_0']['value'] != sites['visit1/t0_0']['value']
    for i in range(2):
        assert sites[f'member{i}/t0_0']['value'] == sites[f'visit{i}/t0_0']['value']


def test_stellarprior_preparation_uses_member_band(monkeypatch, tmp_path):
    import koala.white_light as white
    calls = []
    def build(stellar, wavelengths, resolution, instrument, **kwargs):
        calls.append((instrument, kwargs['order'], np.asarray(wavelengths)))
        return jnp.array([.6, .7]), jnp.array([.03, .06])
    monkeypatch.setattr(white, 'get_or_build_power2_ld_prior', build)
    for order, band in [(1, [1., 2.]), (2, [.6, 1.])]:
        context = SimpleNamespace(explicit_ld=None, ld_prior_mode='stellarprior', ld_profile='power2',
            stellar_cfg={'teff':5000}, instrument='NIRISS/SOSS', order=order,
            data=SimpleNamespace(wavelengths_unbinned=jnp.array(band)), output_dir=str(tmp_path))
        center, width = white.prepare_white_light_ld(context)
        np.testing.assert_allclose(width, [.03, .06])
    assert calls[0][:2] == ('NIRISS/SOSS', 1)
    assert calls[1][:2] == ('NIRISS/SOSS', 2)
    assert not np.array_equal(calls[0][2], calls[1][2])


def test_orchestration_preserves_configs_and_hands_off_slices(monkeypatch, tmp_path):
    """Preparation mutates configs; products must still get clean member configs."""
    import yaml
    import koala.joint as joint
    import koala.pipeline as pipeline
    from koala.config import parse_planet_parameter_specs
    from models.mcmc_diagnostics import PosteriorSamples
    planet = dict(name='test', period=dict(value=3., prior='fixed'),
        t0=dict(value=1., prior='uniform', low=.95, high=1.05),
        a_rs=dict(value=10., prior='log_uniform', low=2., high=100.),
        b=dict(value=.2, prior='uniform', low=0., high=1.5),
        rprs=dict(value=.1, prior='uniform', low=.01, high=.5))
    for name in ('a', 'b', 'c'):
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(dict(planet=planet)))
    completed = []
    def run(cfg, config_path=None, *, prepare_only=False, white_light_posterior=None, verbose=False, prepared_white_light=None):
        from koala.console import joint_display
        presentation = joint_display.get()
        assert presentation is not None
        assert presentation['reading'] == (prepare_only or white_light_posterior is None)
        specs = parse_planet_parameter_specs(cfg['planet'])
        if not prepare_only:
            if white_light_posterior is not None:
                assert prepared_white_light == ('mask', 'parameters', 'geometry')
            completed.append((cfg, white_light_posterior))
            return
        context, piece = make_piece(mode='fixed')
        context.planet_parameter_specs = specs
        t = jnp.linspace(.9, 1.1, 31)
        context.data = SimpleNamespace(wl_time=t, wl_flux_err=jnp.full_like(t, .001), wl_flux=jnp.ones_like(t))
        context._engine_wl_kw = {}
        context.cfg = cfg
        context.key_master = jax.random.PRNGKey(0)
        context.ld_profile = 'quadratic'
        context.ld_prior_mode = 'uniform'
        context.ld_uniform_basis = 'uplus_uminus'
        context.output_dir = cfg['output_dir']
        context.uses_surface_model = False
        context.whitelight_laplace_options = {}
        context.whitelight_mcmc_kwargs = {}
        context.piece = piece
        cfg['planet']['duration'] = [.1]  # What ordinary preparation does.
        return context
    monkeypatch.setattr(pipeline, 'run', run)
    monkeypatch.setattr(joint, 'prepare_white_light_ld', lambda context: (jnp.array([.3,.2]), None))
    monkeypatch.setattr(joint, '_initialize_white_light', lambda **kw: kw['context'].piece)
    monkeypatch.setattr(joint, '_write_joint_products', lambda *args, **kwargs: None)
    monkeypatch.setattr(joint, 'run_white_light_stage', lambda context, posterior: ('mask', 'parameters', 'geometry'))
    def sample(**kwargs):
        assert kwargs['whitelight_mcmc_kwargs']['num_chains'] == 4
        c = kwargs['context']
        sites = trace(seed(kwargs['whitelight_model_for_run'], jax.random.PRNGKey(10))).get_trace(
            c.data.wl_time, c.data.wl_flux_err, y=c.data.wl_flux)
        grouped = {name: np.broadcast_to(np.asarray(site['value']), (4, 10)+np.shape(site['value']))
                   for name, site in sites.items() if site['type'] in {'sample', 'deterministic'} and name.split('/')[-1] != 'obs'}
        np.savez_compressed(tmp_path/'out/state/whitelight_samples_by_chain.npz', **grouped)
        return PosteriorSamples({name: value.reshape((40,)+value.shape[2:]) for name, value in grouped.items()}, num_chains=4)
    monkeypatch.setattr(joint, '_sample_white_light', sample)
    joint.run_joint(dict(joint=dict(members=['a.yaml','b.yaml'], apply_to=['c.yaml']),
                         output_dir='out', host_device='cpu'), tmp_path/'joint.yaml')
    assert len(completed) == 3
    for cfg, posterior in completed[:2]:
        assert 'duration' not in cfg['planet']
        assert posterior is not None
        np.testing.assert_array_equal(posterior['samples']['a_rs_0'], completed[0][1]['samples']['a_rs_0'])
        np.testing.assert_array_equal(posterior['samples']['t0_0'], completed[0][1]['samples']['t0_0'])
    applied, posterior = completed[2]
    assert posterior is None
    for key in ('a_rs', 'b'):
        assert applied['planet'][key]['prior'] == 'fixed'
        assert applied['planet'][key]['value'] == np.median(completed[0][1]['samples'][key+'_0'])
    assert applied['planet']['t0']['prior'] == 'uniform'
