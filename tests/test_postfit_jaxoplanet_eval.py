import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from koala.surface import (
    _attach_jaxoplanet_eval_metadata, _select_transit_eval_params,
)
from koala.limb_darkening import (
    _posterior_quadratic_ld_median,
)
from models.jaxoplanet.core import build_transit_window_indices
from koala.plotting import _single_curve_transit_signal


def test_quadratic_postfit_prefers_sampled_joint_u_over_legacy_sites_and_prior():
    sampled_u = jnp.asarray(
        [
            [[0.10, 0.20], [0.30, 0.40]],
            [[0.20, 0.30], [0.40, 0.50]],
            [[0.90, 0.80], [0.70, 0.60]],
        ],
        dtype=jnp.float64,
    )
    samples = {
        "u": sampled_u,
        "u1": jnp.full((3, 2), 0.99),
        "u2": jnp.full((3, 2), 0.98),
    }
    fallback = jnp.full((2, 2), 0.01)

    actual = _posterior_quadratic_ld_median(samples, fallback)

    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(jnp.nanmedian(sampled_u, axis=0))
    )


def test_quadratic_postfit_supports_harmonica_split_sites_and_partial_fallback():
    samples = {
        "u1": jnp.asarray([[0.20, 0.30], [0.40, 0.50], [0.30, 0.40]])
    }
    fallback = jnp.asarray([[0.11, 0.12], [0.13, 0.14]])

    actual = _posterior_quadratic_ld_median(samples, fallback)

    np.testing.assert_allclose(np.asarray(actual[:, 0]), [0.30, 0.40])
    np.testing.assert_allclose(np.asarray(actual[:, 1]), [0.12, 0.14])


def test_quadratic_postfit_rejects_nonquadratic_joint_site():
    with pytest.raises(ValueError, match="final dimension 2"):
        _posterior_quadratic_ld_median(
            {"u": jnp.ones((4, 3, 12))}, jnp.ones((3, 2))
        )


def _duration_map_params():
    return {
        "period": jnp.asarray([2.75]),
        "duration": jnp.asarray([0.12]),
        "rors_reference": jnp.asarray([0.1]),
        "t0": jnp.asarray([0.0]),
        "b": jnp.asarray([0.31]),
        "rors": jnp.asarray([[0.096], [0.104]]),
        "u": jnp.asarray([[0.27, 0.18], [0.34, 0.12]]),
    }


def test_postfit_metadata_is_rebuilt_for_the_current_clipped_time_axis():
    full_time = jnp.linspace(-0.20, 0.20, 101)
    clipped_time = full_time[7:-9]
    params = _attach_jaxoplanet_eval_metadata(
        _duration_map_params(),
        full_time,
        jaxoplanet_kernel="auto",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )

    rebuilt = _attach_jaxoplanet_eval_metadata(
        params,
        clipped_time,
        jaxoplanet_kernel="auto",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )

    assert not {'_transit_phase_offsets','_transit_phase_mask','_transit_window_indices'} & set(rebuilt)
    assert rebuilt['a_rs'].shape == (1,)  # One orbit shared by both channel radii.
    np.testing.assert_array_equal(rebuilt['a_rs'],params['a_rs'])
    from models.jaxoplanet.core import compute_transit_model
    for index in (0,1):
        lane=dict(rebuilt,rors=rebuilt['rors'][index],u=rebuilt['u'][index])
        full=compute_transit_model(lane,full_time)
        clipped=compute_transit_model(lane,clipped_time)
        np.testing.assert_allclose(clipped,full[7:-9],atol=1e-12,rtol=0)


def test_select_eval_params_preserves_shared_physical_orbit():
    times = jnp.linspace(-0.08, 0.08, 33)
    raw = {**_duration_map_params(), "a_rs": jnp.asarray([10.4])}

    selected = _select_transit_eval_params(
        raw,
        transit_engine="jaxoplanet",
        param_method="duration",
        t=times,
        jaxoplanet_kernel="auto",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )

    np.testing.assert_array_equal(selected["a_rs"],raw["a_rs"])
    assert selected["_jaxoplanet_kernel"] == "auto"
    assert selected["_ld_profile"] == "quadratic"
    assert not {"_transit_phase_offsets","_transit_window_indices"} & set(selected)


def test_plotting_forwards_postfit_jaxoplanet_metadata_unchanged():
    times = jnp.linspace(-0.08, 0.08, 33)
    map_params = _attach_jaxoplanet_eval_metadata(
        _duration_map_params(),
        times,
        jaxoplanet_kernel="auto",
        ld_profile="quadratic",
        transit_window_optimization="auto",
    )
    expected = jnp.zeros_like(times)

    with patch("koala.plotting.compute_transit_model_auto", return_value=expected) as model:
        actual = _single_curve_transit_signal(
            np.asarray(times),
            map_params,
            {"period": np.asarray([2.75]), "transit_engine": "jaxoplanet", "param_method": "duration"},
            1,
        )

    forwarded = model.call_args.args[0]
    assert forwarded["_jaxoplanet_kernel"] == "auto"
    assert forwarded["_ld_profile"] == "quadratic"
    np.testing.assert_array_equal(forwarded['a_rs'],map_params['a_rs'])
    assert not {'_transit_phase_offsets','_transit_window_indices'} & set(forwarded)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))



def test_power2_plotting_keeps_native_coefficients_and_shared_orbit():
    from models.jaxoplanet.core import compute_transit_model
    t=jnp.linspace(-.08,.08,33)
    raw=dict(_duration_map_params(),u=jnp.zeros((2,12)),
             c1=jnp.array([.6,.8]),c2=jnp.array([.1,.6]))
    params=_attach_jaxoplanet_eval_metadata(raw,t,jaxoplanet_kernel='auto',ld_profile='power2')
    for index in (0,1):
        expected=compute_transit_model(dict(params,rors=params['rors'][index],u=params['u'][index],
            c1=params['c1'][index],c2=params['c2'][index]),t)
        actual=_single_curve_transit_signal(t,params,dict(period=params['period'],
            transit_engine='jaxoplanet',param_method='duration'),index)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
