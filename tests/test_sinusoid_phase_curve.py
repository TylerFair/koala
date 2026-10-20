"""Sinusoidal phase curve: batman parity, gradients, configuration, outputs.

The model is ``F - 1 = transit(t) + L(phi) E(t)`` with
``L(phi) = f_d + C1 (cos phi - 1) + D1 sin phi``, ``phi = 2 pi (t - t0 - P/2)/P``
and ``E`` the visible fraction of a uniform planet disk.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from models.jaxoplanet.config import parse_surface_config
from models.jaxoplanet.surface import (
    compute_surface_model,
    phase_curve_derived_quantities,
)
from models.jaxoplanet.surface_basis import (
    compute_emission_basis_model,
    prepare_emission_light_curve_basis,
)

PERIOD, T0, A_RS, B, RP = 1.7, 0.3, 5.5, 0.35, 0.11
U = (0.35, 0.2)
# Paper-like values: the nightside f_d - 2 C1 = 10 ppm violates any 5:1 cap.
FD, C1, D1 = 116e-6, 53e-6, -20e-6


def _time():
    # Transit plus two secondary eclipses, dense (about 13 s cadence).
    return np.linspace(T0 - 0.2, T0 + 2.0 * PERIOD, 20001)


def _batman_reference(t):
    batman = pytest.importorskip("batman")
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.a = T0, PERIOD, RP, A_RS
    params.inc = np.degrees(np.arccos(B / A_RS))
    params.ecc, params.w = 0.0, 90.0
    params.u, params.limb_dark = list(U), "quadratic"
    params.fp, params.t_secondary = 1.0e-3, T0 + 0.5 * PERIOD
    transit = batman.TransitModel(params, t).light_curve(params) - 1.0
    secondary = batman.TransitModel(params, t, transittype="secondary").light_curve(params)
    # batman normalises the secondary curve to 1 in full eclipse and 1 + fp
    # out of eclipse, so this is the visible fraction of the planet disk.
    visible = (secondary - 1.0) / params.fp
    assert visible.max() == pytest.approx(1.0) and visible.min() == pytest.approx(0.0)
    phi = 2.0 * np.pi * (t - params.t_secondary) / PERIOD
    planet = FD + C1 * (np.cos(phi) - 1.0) + D1 * np.sin(phi)
    return transit, visible, transit + planet * visible


def _params(**overrides):
    params = dict(
        period=jnp.array([PERIOD]), t0=jnp.array([T0]), a_rs=jnp.array([A_RS]),
        b=jnp.array([B]), rors=jnp.array([RP]), u=jnp.array(U),
        eclipse_depth=jnp.array([FD]), phase_c1=jnp.array([C1]),
        phase_d1=jnp.array([D1]),
    )
    params.update(overrides)
    return params


def test_native_phase_curve_matches_batman():
    t = _time()
    transit, visible, reference = _batman_reference(t)
    model = np.asarray(compute_surface_model(_params(), jnp.asarray(t), model="phase_curve"))
    max_abs = np.max(np.abs(model - reference))
    # Relative to the transit depth (rp^2) the agreement is better than 1e-6.
    assert max_abs / RP ** 2 < 1e-6, f"max abs diff {max_abs * 1e6:.4g} ppm"
    # The planet term alone (isolating E(t) and the sinusoid) is far tighter.
    # Subtract the same kernel's transit (a zero-depth eclipse), not the
    # starry ``transit`` model, which differs from the analytic kernel at 5e-9.
    planet_model = model - np.asarray(compute_surface_model(
        _params(eclipse_depth=jnp.array([0.0])), jnp.asarray(t), model="eclipse"
    ))
    assert np.max(np.abs(planet_model - (reference - transit))) < 1e-12


def test_fixed_basis_matches_batman_and_native():
    t = _time()
    transit, visible, reference = _batman_reference(t)
    basis = prepare_emission_light_curve_basis(
        dict(period=PERIOD, t0=T0, a_rs=A_RS, b=B, rors=RP, u=np.array(U)),
        t, model="phase_curve",
    )
    np.testing.assert_allclose(np.asarray(basis.uniform), visible, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(basis.baseline), transit, rtol=0, atol=1e-6 * RP ** 2)
    fast = np.asarray(compute_emission_basis_model(_params(), basis))
    native = np.asarray(compute_surface_model(_params(), jnp.asarray(t), model="phase_curve"))
    np.testing.assert_allclose(fast, native, rtol=0, atol=1e-13)
    np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-6 * RP ** 2)


def test_eclipse_depth_is_flux_at_eclipse_and_nightside_at_transit():
    t = jnp.asarray([T0 + 0.5 * PERIOD - 0.25, T0 + 0.5 * PERIOD + 0.25])
    params = _params(phase_c1=jnp.array([0.0]), phase_d1=jnp.array([0.0]))
    # Out of eclipse with C1 = D1 = 0 the planet adds exactly f_d.
    np.testing.assert_allclose(
        np.asarray(compute_surface_model(params, t, model="phase_curve")), FD, atol=1e-14,
    )
    derived = phase_curve_derived_quantities(FD, C1, D1)
    assert float(derived["nightside_flux"]) == pytest.approx(FD - 2 * C1)
    assert float(derived["phase_amplitude"]) == pytest.approx(np.hypot(C1, D1))
    # D1 < 0: the maximum precedes eclipse, a positive (eastward) offset.
    assert float(derived["phase_offset_deg"]) == pytest.approx(np.degrees(np.arctan2(-D1, C1)))
    assert float(derived["phase_offset_deg"]) > 0.0


def test_phase_curve_is_differentiable_in_geometry():
    t = jnp.asarray(_time()[::40])

    def loss(t0, a_rs, b, rors):
        model = compute_surface_model(
            _params(t0=jnp.atleast_1d(t0), a_rs=jnp.atleast_1d(a_rs),
                    b=jnp.atleast_1d(b), rors=jnp.atleast_1d(rors)),
            t, model="phase_curve",
        )
        return jnp.sum(model ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3))(T0, A_RS, B, RP)
    assert all(np.isfinite(float(g)) and float(g) != 0.0 for g in grads)
    step = 1e-6
    numeric = (loss(T0, A_RS, B, RP + step) - loss(T0, A_RS, B, RP - step)) / (2 * step)
    assert float(grads[3]) == pytest.approx(float(numeric), rel=1e-5)


def _fixed(value):
    return {"value": value, "prior": "fixed"}


def _uniform(value, low, high):
    return {"value": value, "prior": "uniform", "low": low, "high": high}


def test_phase_config_parses_paper_priors_without_ratio_check():
    result = parse_surface_config(
        {"light_curve_model": "phase_curve"},
        {"eclipse_depth_ppm": _uniform(116., 0., 1000.),
         "phase_c1_ppm": _uniform(53., -1000., 1000.),
         "phase_d1_ppm": _uniform(-20., -1000., 1000.)},
        {}, 1,
    )
    np.testing.assert_allclose(result["eclipse_depth"], [116e-6])
    np.testing.assert_allclose(result["phase_c1"], [53e-6])
    np.testing.assert_allclose(result["phase_d1"], [-20e-6])
    assert result["phase_c1_spec"][0].low == pytest.approx(-1e-3)
    # A fixed nightside far below a fifth of the dayside is accepted.
    parse_surface_config(
        {"light_curve_model": "phase_curve"},
        {"eclipse_depth_ppm": _fixed(500.), "phase_c1_ppm": _fixed(249.),
         "phase_d1_ppm": _fixed(0.)},
        {}, 1,
    )


@pytest.mark.parametrize("old_key", [
    "dayside_flux_ppm", "nightside_flux_ppm", "hotspot_offset_deg",
])
def test_removed_phase_map_keys_name_their_replacements(old_key):
    planet = {"eclipse_depth_ppm": _fixed(100.), "phase_c1_ppm": _fixed(10.),
              "phase_d1_ppm": _fixed(0.), old_key: _fixed(1.)}
    with pytest.raises(ValueError, match=rf"{old_key}.*removed.*phase_c1_ppm"):
        parse_surface_config({"light_curve_model": "phase_curve"}, planet, {}, 1)


def test_sampled_sites_and_derived_outputs():
    import numpyro
    from models.jaxoplanet.builder import _sample_surface_parameters

    config = parse_surface_config(
        {"light_curve_model": "phase_curve"},
        {"eclipse_depth_ppm": _uniform(116., 0., 1000.),
         "phase_c1_ppm": _uniform(53., -1000., 1000.),
         "phase_d1_ppm": {"value": -20., "prior": "gaussian", "sigma": 30.}},
        {}, 1,
    )
    with numpyro.handlers.trace() as trace, numpyro.handlers.seed(rng_seed=1):
        values = _sample_surface_parameters(config, 1)
    for site in ("_eclipse_depth_0", "_phase_c1_0", "_phase_d1_0"):
        assert trace[site]["type"] == "sample"
    depth, c1, d1 = (float(values[name][0]) for name in ("eclipse_depth", "phase_c1", "phase_d1"))
    assert float(trace["nightside_flux_ppm"]["value"][0]) == pytest.approx((depth - 2 * c1) * 1e6)
    assert float(trace["phase_amplitude_ppm"]["value"][0]) == pytest.approx(np.hypot(c1, d1) * 1e6)
    assert float(trace["phase_offset_deg"]["value"][0]) == pytest.approx(
        np.degrees(np.arctan2(-d1, c1)))


def test_emission_csv_reports_coefficients_and_derived(tmp_path):
    import pandas as pd
    from koala.surface_outputs import save_surface_results

    rng = np.random.default_rng(0)
    samples = {
        "eclipse_depth": FD + 1e-6 * rng.normal(size=(400, 2, 1)),
        "phase_c1": C1 + 1e-6 * rng.normal(size=(400, 2, 1)),
        "phase_d1": D1 + 1e-6 * rng.normal(size=(400, 2, 1)),
    }
    save_surface_results(np.array([3.0, 4.0]), np.array([0.5, 0.5]), samples,
                         tmp_path / "spec.csv")
    frame = pd.read_csv(tmp_path / "spec_emission.csv")
    for column in ("eclipse_depth_ppm", "phase_c1_ppm", "phase_d1_ppm",
                   "nightside_flux_ppm", "phase_amplitude_ppm", "phase_offset_deg"):
        assert column in frame
    np.testing.assert_allclose(frame["nightside_flux_ppm"], (FD - 2 * C1) * 1e6, atol=0.5)


def test_eclipse_time_scan_ignores_transits_in_a_full_orbit():
    from koala.config import ParameterSpec
    from koala.surface import scan_eclipse_time

    t = np.linspace(T0 - 0.1, T0 + 1.05 * PERIOD, 1500)
    true_offset = 0.01  # the true epoch is 14.4 min after the configured start
    model = np.asarray(compute_surface_model(
        _params(t0=jnp.array([T0 + true_offset])), jnp.asarray(t), model="phase_curve"))
    rng = np.random.default_rng(4)
    flux = (1.0 + model) * (1.0 + 2e-4 * (t - t[0])) + rng.normal(0.0, 60e-6, t.size)
    t0_spec = ParameterSpec("t0", "uniform", T0, low=T0 - 0.03, high=T0 + 0.03)
    depth_spec = ParameterSpec("eclipse_depth", "uniform", 1e-4, low=0.0, high=1e-3)
    found_t0, depth, delta_chi2 = scan_eclipse_time(
        {"model": "phase_curve"}, t, flux, 60e-6, t0_spec, depth_spec, "linear",
        period=PERIOD, a_rs=A_RS, b=B, rors=RP, ecc=0.0, omega=0.0)
    assert abs(found_t0 - (T0 + true_offset)) * 1440 < 3.0
    assert abs(depth - FD) < 25e-6
    assert delta_chi2 > 25.0


def test_closed_form_emission_path_matches_native_starry():
    """Unspotted quadratic-LD eclipse/phase curves skip starry; check parity."""
    import jax.numpy as jnp
    import numpy as np
    from models.jaxoplanet.surface import compute_surface_model

    base = dict(period=0.3711281, t0=60337.6185, a_rs=5.0, b=0.22, rors=0.0412,
                ecc=0.0, omega=0.0, u=jnp.array([0.17, 0.27]), eclipse_depth=116e-6,
                phase_c1=53e-6, phase_d1=-5e-6)
    t = jnp.linspace(60337.28, 60337.84, 6000)
    for extra in ({}, {"b": 0.95}, {"b": 1.02}, {"ecc": 0.1, "omega": 0.7}):
        params = {**base, **extra}
        # A zero third coefficient keeps the law but forces the native starry path.
        native_params = {**params, "u": jnp.array([0.17, 0.27, 0.0])}
        for model in ("eclipse", "phase_curve"):
            fast = np.asarray(compute_surface_model(params, t, model=model))
            native = np.asarray(compute_surface_model(native_params, t, model=model))
            assert np.max(np.abs(fast - native)) < 2e-9, (extra, model)
