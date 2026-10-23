import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

import koala.sampling as koala_sampling
import models.independent_nuts as independent_nuts
from models.independent_hmc import (
    build_independent_hmc_runner,
    get_samples_independent_hmc,
)
from models.independent_nuts import (
    build_independent_nuts_runner,
    get_samples_independent,
)


def _dynamic_normal_model(t, yerr, y=None, prior_loc=None):
    x = numpyro.sample("x", dist.Normal(prior_loc, 0.7))
    prediction = x[:, None] + jnp.zeros_like(t)
    numpyro.deterministic("twice_x", 2.0 * x)
    numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)


@pytest.mark.parametrize('backend', ['nuts', 'hmc'])
@pytest.mark.parametrize('mass_matrix', ['adaptive', 'laplace'])
@pytest.mark.parametrize('synchronize_segments', [True, False])
@pytest.mark.parametrize('show_progress', [True, False])
def test_continuation_matches_uninterrupted_independent_chains(backend, mass_matrix, synchronize_segments, show_progress):
    builder=build_independent_nuts_runner if backend=='nuts' else build_independent_hmc_runner
    sampler=get_samples_independent if backend=='nuts' else get_samples_independent_hmc
    options=dict(nuts_kwargs={"synchronize_segments":synchronize_segments,"progress_segment":50},num_warmup=30,num_samples=100,lane_width=3,
        channel_varying_kwargs=('prior_loc',),mass_matrix=mass_matrix,laplace_warmup=30,
        mcmc_kwargs=dict(num_chains=4,progress_bar=show_progress))
    if backend=='hmc':options['num_steps']=5
    t=jnp.linspace(-1.,1.,4);errors=jnp.full((2,4),.25)
    observations=jnp.array([[1.3]*4,[-.8]*4]);loc=jnp.array([1.,-1.])
    initial={'x':loc};key=jax.random.PRNGKey(121)
    runner=builder(_dynamic_normal_model,**options)
    first=sampler(_dynamic_normal_model,key,t,errors,observations,initial,
                  prior_loc=loc,_runner=runner,**options)
    before=jax.tree.map(np.asarray,runner._continuation[0])
    extra,raw=runner.continue_raw()
    combined={name:jnp.concatenate((first[name],extra[name])) for name in first}
    whole=sampler(_dynamic_normal_model,key,t,errors,observations,initial,
                  prior_loc=loc,**{**options,'num_samples':200})
    np.testing.assert_array_equal(combined['x'],whole['x'])
    np.testing.assert_array_equal(combined['twice_x'],whole['twice_x'])
    np.testing.assert_array_equal(runner._continuation[0].i,before.i+100)
    np.testing.assert_array_equal(runner._continuation[0].adapt_state.step_size,
                                  before.adapt_state.step_size)
    assert runner.program_build_count==1
    assert raw['num_chains']==4
    with pytest.raises(ValueError, match='identical shapes'):
        runner.run_raw(key, t, errors, observations[:, :-1], initial, {'prior_loc':loc})
    with pytest.raises(RuntimeError, match='completed initial run'):
        runner.continue_raw()


@pytest.mark.parametrize("backend", ["nuts", "hmc"])
@pytest.mark.parametrize("chains", [1, 4])
def test_reused_independent_runner_has_no_stale_data_or_initial_state(backend, chains):
    t = jnp.linspace(-1.0, 1.0, 4)
    yerr = jnp.full((2, 4), 0.25)
    first_y = jnp.asarray([[-2.2] * 4, [-1.8] * 4])
    second_y = jnp.asarray([[2.3] * 4, [1.7] * 4])
    first_loc = jnp.asarray([-2.0, -2.0])
    second_loc = jnp.asarray([2.0, 2.0])
    first_init = {"x": jnp.asarray([-2.4, -1.6])}
    second_init = {"x": jnp.asarray([2.4, 1.6])}
    common = {
        "num_warmup": 8,
        "num_samples": 10,
        "lane_width": 3,
        "channel_varying_kwargs": ("prior_loc",),
        "mcmc_kwargs": {"num_chains": chains, "progress_bar": False},
    }
    if backend == "nuts":
        sampler = get_samples_independent
        builder = build_independent_nuts_runner
        common["max_tree_depth"] = 5
    else:
        sampler = get_samples_independent_hmc
        builder = build_independent_hmc_runner
        common["num_steps"] = 5

    runner = builder(_dynamic_normal_model, **common)
    sampler(
        _dynamic_normal_model,
        jax.random.PRNGKey(20),
        t,
        yerr,
        first_y,
        first_init,
        prior_loc=first_loc,
        _runner=runner,
        **common,
    )
    reused = sampler(
        _dynamic_normal_model,
        jax.random.PRNGKey(21),
        t,
        yerr,
        second_y,
        second_init,
        prior_loc=second_loc,
        _runner=runner,
        **common,
    )
    fresh = sampler(
        _dynamic_normal_model,
        jax.random.PRNGKey(21),
        t,
        yerr,
        second_y,
        second_init,
        prior_loc=second_loc,
        **common,
    )

    assert runner.program_build_count == 1
    assert reused.num_chains == chains
    assert reused.group_by_chain()["x"].shape == (chains, 10, 2)
    np.testing.assert_array_equal(np.asarray(reused["x"]), np.asarray(fresh["x"]))
    np.testing.assert_array_equal(
        np.asarray(reused["twice_x"]), np.asarray(fresh["twice_x"])
    )
    assert float(np.asarray(reused["x"]).mean()) > 1.0


def test_fit_chunk_router_reuses_one_independent_runner_per_padded_width(
    monkeypatch,
):
    built = []
    calls = []

    def fake_builder(model, *, lane_width, **kwargs):
        runner = object()
        built.append((model, lane_width, runner, kwargs))
        return runner

    def fake_sampler(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        *,
        lane_width,
        _runner,
        **kwargs,
    ):
        del model, key, t, yerr, kwargs
        calls.append((lane_width, _runner, np.asarray(init_params["x"])))
        return {"x": jnp.asarray(indiv_y[:, 0])[None, :]}

    monkeypatch.setattr(
        independent_nuts, "build_independent_nuts_runner", fake_builder
    )
    monkeypatch.setattr(
        independent_nuts, "get_samples_independent", fake_sampler
    )
    samples = koala_sampling.get_samples_chunked(
        _dynamic_normal_model,
        jax.random.PRNGKey(3),
        jnp.arange(3.0),
        jnp.ones((5, 3)),
        jnp.broadcast_to(jnp.arange(5.0)[:, None], (5, 3)),
        {"x": jnp.arange(5.0)},
        chunk_size=2,
        sampler_backend="independent_nuts",
        channel_varying_kwargs=("prior_loc",),
        prior_loc=jnp.arange(5.0),
        mcmc_kwargs={"num_warmup": 1, "num_samples": 1},
    )

    assert len(built) == 1
    assert built[0][1] == 2
    assert [width for width, _, _ in calls] == [2, 2, 2]
    assert all(runner is built[0][2] for _, runner, _ in calls)
    np.testing.assert_array_equal(
        np.asarray(samples["x"]), np.arange(5.0)[None, :]
    )


def test_fit_chunk_router_shrinks_width_for_a_single_partial_chunk(
    monkeypatch, koala_caplog
):
    built = []
    calls = []

    def fake_builder(model, *, lane_width, **kwargs):
        runner = object()
        built.append((model, lane_width, runner, kwargs))
        return runner

    def fake_sampler(
        model,
        key,
        t,
        yerr,
        indiv_y,
        init_params,
        *,
        lane_width,
        _runner,
        **kwargs,
    ):
        del model, key, t, yerr, init_params, kwargs
        calls.append((lane_width, _runner))
        return {"x": jnp.asarray(indiv_y[:, 0])[None, :]}

    monkeypatch.setattr(
        independent_nuts, "build_independent_nuts_runner", fake_builder
    )
    monkeypatch.setattr(
        independent_nuts, "get_samples_independent", fake_sampler
    )
    samples = koala_sampling.get_samples_chunked(
        _dynamic_normal_model,
        jax.random.PRNGKey(3),
        jnp.arange(3.0),
        jnp.ones((5, 3)),
        jnp.broadcast_to(jnp.arange(5.0)[:, None], (5, 3)),
        {"x": jnp.arange(5.0)},
        chunk_size=40,
        sampler_backend="independent_nuts",
        channel_varying_kwargs=("prior_loc",),
        prior_loc=jnp.arange(5.0),
        mcmc_kwargs={"num_warmup": 1, "num_samples": 1},
    )

    assert len(built) == 1
    assert built[0][1] == 5
    assert calls == [(5, built[0][2])]
    assert "resident width of 5" in koala_caplog.text
    np.testing.assert_array_equal(
        np.asarray(samples["x"]), np.arange(5.0)[None, :]
    )


def test_fixed_support_trace_reuse_keeps_new_priors_data_and_initial_values(monkeypatch):
    original = independent_nuts.initialize_model
    calls = []
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(independent_nuts,'initialize_model',counted)
    def model(*args, **kwargs):
        return _dynamic_normal_model(*args, **kwargs)
    model._koala_fixed_initialization_supports = True
    common=dict(num_warmup=8,num_samples=10,lane_width=2,max_tree_depth=5,
                channel_varying_kwargs=('prior_loc',),mcmc_kwargs={'progress_bar':False})
    t=jnp.arange(4.)
    errors=jnp.full((2,4),.25)
    runner=build_independent_nuts_runner(model,**common)
    for seed,loc in [(20,-2.),(21,2.)]:
        y=jnp.full((2,4),loc)
        init={'x':jnp.full((2,),loc+.1)}
        reused=get_samples_independent(model,jax.random.PRNGKey(seed),t,errors,y,init,
            prior_loc=jnp.full((2,),loc),_runner=runner,**common)
    assert len(calls)==1
    fresh=get_samples_independent(model,jax.random.PRNGKey(21),t,errors,y,init,
        prior_loc=jnp.full((2,),2.),**common)
    np.testing.assert_array_equal(reused['x'],fresh['x'])
    np.testing.assert_array_equal(reused['twice_x'],fresh['twice_x'])
    # Missing starts retain NumPyro's newly seeded initialization semantics.
    get_samples_independent(model,jax.random.PRNGKey(22),t,errors,y,{},
        prior_loc=jnp.full((2,),2.),_runner=runner,**common)
    assert len(calls)==3


@pytest.mark.slow
@pytest.mark.parametrize('mode,parameterization',[('gaussian','coefficients'),('uniform','maxted_smooth')])
def test_production_fixed_support_reuse_matches_fresh_seeded_draws(mode,parameterization):
    from models.jaxoplanet.builder import create_vectorized_model
    model=create_vectorized_model(ld_profile='power2',ld_mode=mode,
        ld_parameterization=parameterization)
    assert model._koala_fixed_initialization_supports
    common=dict(num_warmup=6,num_samples=8,lane_width=2,max_tree_depth=4,
        channel_varying_kwargs=('ld_fixed','mu_u_ld'),mcmc_kwargs={'progress_bar':False})
    runner=build_independent_nuts_runner(model,**common)
    t=jnp.linspace(-.08,.08,21)
    errors=jnp.full((2,21),.003)
    shared=dict(rors_reference=jnp.array([.1]), mu_duration=jnp.array([.12]),mu_t0=jnp.array([0.]),mu_b=jnp.array([.45]),PERIOD=jnp.array([3.]))
    for seed,centre in [(20,.14),(21,.16)]:
        coeff=jnp.tile(jnp.array([.6,.6]),(2,1))
        init=dict(rors=jnp.full((2,1),centre),c=jnp.ones(2),v=jnp.zeros(2),
                  log_jitter=jnp.full((2,),jnp.log(.001)),c1=coeff[:,0],c2=coeff[:,1])
        y=jnp.ones_like(errors)-centre**2*jnp.exp(-(t/.045)**8)
        kwargs=dict(shared,ld_fixed=coeff,mu_u_ld=coeff+.01*(seed-20))
        reused=get_samples_independent(model,jax.random.PRNGKey(seed),t,errors,y,init,
            _runner=runner,**common,**kwargs)
    fresh=get_samples_independent(model,jax.random.PRNGKey(21),t,errors,y,init,**common,**kwargs)
    for name in ['depths','c1','c2']:
        np.testing.assert_array_equal(reused[name],fresh[name])
    assert not create_vectorized_model(ld_profile='quadratic',ld_mode='sing')._koala_fixed_initialization_supports


def test_split_laplace_profile_matches_combined_preparation():
    options=dict(num_warmup=20,num_samples=20,lane_width=2,
        channel_varying_kwargs=('prior_loc',),mass_matrix='laplace',laplace_warmup=20,
        mcmc_kwargs=dict(num_chains=4,progress_bar=False))
    runner=build_independent_nuts_runner(_dynamic_normal_model,**options)
    original=runner._build_laplace_programs;checked=[]
    def build(*args,**kwargs):
        result=original(*args,**kwargs);combined=runner._init_program
        def compare(*values):
            expected=combined(*values)
            states,diagnostics=runner._laplace_map_program(*values)
            states=runner._laplace_warmup_program(states,values[5],values[1],values[2],values[3],values[6])
            for actual,reference in zip(jax.tree.leaves((states,diagnostics)),jax.tree.leaves(expected)):
                np.testing.assert_allclose(actual,reference,atol=1e-10,rtol=1e-10)
            checked.append(True)
            return expected
        runner._init_program=compare
        return result
    runner._build_laplace_programs=build
    loc=jnp.array([1.,-1.]);samples=get_samples_independent(
        _dynamic_normal_model,jax.random.PRNGKey(121),jnp.arange(4.),
        jnp.full((2,4),.25),jnp.array([[1.3]*4,[-.8]*4]),{'x':loc},
        prior_loc=loc,_runner=runner,**options)
    assert checked == [True]
    assert samples.num_chains == 4
