"""Independent photometric checks; external fitting code is not vendored here.

Reference conventions reviewed in ideasrule/sparta/emcee_methods.py, including
875273800194b57e86c4145647a90906e863eb24 (the limited phase driver's version).
The sinusoidal phase curve is checked against batman's transit and uniform
secondary eclipse combined with an independent first-harmonic formula.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

pytest.importorskip("jaxoplanet")
batman = pytest.importorskip("batman")
import jax
import jax.numpy as jnp
from numpyro import handlers

from koala.config import parse_planet_parameter_specs
from models.jaxoplanet.builder import create_vectorized_model, create_whitelight_model
from models.jaxoplanet.config import parse_surface_config
from models.jaxoplanet.surface import compute_surface_model
from models.jaxoplanet.surface_basis import prepare_emission_light_curve_basis

jax.config.update("jax_enable_x64", True)


def parameters():
    return dict(period=1.7, t0=60000.123, a_rs=8., b=.3, rors=.0318,
                ecc=0., omega=.4, u=np.array([.3, .2]),
                eclipse_depth=100e-6, phase_c1=40e-6, phase_d1=-15e-6)


def batman_geometry(p):
    bp = batman.TransitParams()
    bp.t0, bp.per, bp.rp, bp.a = (p[k] for k in ("t0", "period", "rors", "a_rs"))
    bp.ecc, bp.w = p["ecc"], np.rad2deg(p["omega"])
    cos_i = p["b"] / p["a_rs"] * (1 + bp.ecc*np.sin(p["omega"])) / (1-bp.ecc**2)
    bp.inc = np.rad2deg(np.arccos(cos_i))
    bp.limb_dark, bp.u, bp.fp = "quadratic", list(p["u"]), 1.
    primary = batman.TransitModel(bp, np.array([bp.t0]))
    bp.t_secondary = primary.get_t_secondary(bp)
    return bp


def batman_components(p, t):
    bp = batman_geometry(p)
    star = batman.TransitModel(bp, t).light_curve(bp)
    visible = batman.TransitModel(bp, t, transittype="secondary").light_curve(bp) - 1.
    return star, visible


def harmonic_flux(p, t):
    """L = f_d + C1 (cos phi - 1) + D1 sin phi, phi = 0 at t0 + P/2 (mean motion)."""
    phase = 2*np.pi*(t - p["t0"] - p["period"]/2)/p["period"]
    return (p["eclipse_depth"] + p["phase_c1"]*(np.cos(phase)-1)
            + p["phase_d1"]*np.sin(phase))


@pytest.mark.parametrize("radius,impact,ecc,omega", [
    (.01, 0., 0., 0.), (.0318, .7, .3, .8),
    (.0318, 1.015, 0., 0.), (.12, .4, .45, -.7),
])
def test_full_orbit_uniform_emission_matches_batman(radius, impact, ecc, omega):
    p = {**parameters(), "rors": radius, "b": impact, "ecc": ecc, "omega": omega}
    midpoint = batman_geometry(p).t_secondary
    # Dense local grids include ingress and egress of small/grazing planets.
    t = np.sort(np.r_[p["t0"] + p["period"]*np.linspace(-.1, 1.1, 401),
                      p["t0"] + np.linspace(-.08, .08, 301),
                      midpoint + np.linspace(-.08, .08, 301)])
    star, visible = batman_components(p, t)
    expected = star + p["eclipse_depth"]*visible
    actual = compute_surface_model(p, t, model="eclipse") + 1.
    # The two quadratic stellar-transit evaluators agree within 0.05 ppm on
    # these grids. Keep emission-only comparisons much tighter; increasing
    # starry's quadrature order does not remove the small transit difference.
    np.testing.assert_allclose(actual, expected, rtol=0., atol=5e-8)
    np.testing.assert_allclose(np.asarray(actual)[star == 1.], expected[star == 1.],
                               rtol=0., atol=5e-10)


@pytest.mark.parametrize("ecc,omega", [(0., 0.), (0., .8), (.25, .4)])
def test_sinusoid_phase_curve_matches_batman_including_partial_eclipse(ecc, omega):
    p = {**parameters(), "ecc": ecc, "omega": omega}
    midpoint = batman_geometry(p).t_secondary
    t = np.sort(np.r_[p["t0"] + p["period"]*np.linspace(-.05, 1.05, 1001),
                      midpoint + np.linspace(-.06, .06, 301)])
    star, visible = batman_components(p, t)
    expected = star + harmonic_flux(p, t)*visible
    actual = np.asarray(compute_surface_model(p, t, model="phase_curve")) + 1.
    assert np.count_nonzero(visible == 0.) > 0
    assert np.count_nonzero((visible > 0.) & (visible < 1.)) > 0
    assert np.count_nonzero(star < 1.) > 0
    np.testing.assert_allclose(actual, expected, rtol=0., atol=5e-8)
    emission_only = star == 1.
    np.testing.assert_allclose(actual[emission_only], expected[emission_only], rtol=0., atol=5e-12)


def configuration(p, model_name):
    def fixed(value):
        return {"value": float(value), "prior": "fixed"}
    planet = {key: fixed(p[key]) for key in ("period", "t0", "a_rs", "b", "ecc", "omega")}
    planet["rprs"] = fixed(p["rors"])
    names = ("eclipse_depth",) if model_name == "eclipse" else ("eclipse_depth", "phase_c1", "phase_d1")
    for name in names:
        planet[name+"_ppm"] = fixed(p[name]*1e6)
    specs = parse_planet_parameter_specs(planet)
    config = parse_surface_config({"light_curve_model": model_name}, planet, {}, 1,
                                  parameter_specs=specs, param_method="a_rs")
    return specs, config
@pytest.mark.parametrize("model_name", ["eclipse", "phase_curve"])
@pytest.mark.parametrize("use_basis", [False, True])
def test_white_and_spectral_likelihood_baseline_and_heteroscedastic_errors(model_name, use_basis):
    p = parameters()
    t = p["t0"] + p["period"]*np.linspace(.36, .64, 201)
    specs, config = configuration(p, model_name)
    basis = prepare_emission_light_curve_basis(p, t, model=model_name) if use_basis else None
    white = create_whitelight_model(ld_mode="fixed", param_method="a_rs",
        surface_config=config, surface_basis=basis, parameter_priors=specs)
    spectral = create_vectorized_model(ld_mode="fixed", param_method="a_rs",
        surface_config=config, surface_basis=basis)
    # Convert Fstar*(1+m*(t-mean(t))) to Koala's c+v*(t-min(t)).
    fstar, slope, jitter = 1.0002, .0013, 17e-6
    c = fstar*(1+slope*(t.min()-t.mean()))
    v = fstar*slope
    error = np.linspace(12e-6, 80e-6, t.size)
    error[t.size//3] = 1e3  # Reader bad-pixel sentinel must retain negligible weight.
    star, visible = batman_components(p, t)
    reference = star + (p["eclipse_depth"] if model_name == "eclipse" else harmonic_flux(p, t))*visible
    selected = np.ones(t.size, dtype=bool)  # uniform-disk physics holds at every cadence
    mean = reference*fstar*(1+slope*(t-t.mean()))
    y = mean + 8e-6*np.sin(np.arange(t.size))
    values = {"c": c, "v": v, "log_jitter": np.log(jitter)}
    wt = handlers.trace(handlers.seed(handlers.substitute(white, data=values), 5)).get_trace(
        t, error, y=y, prior_params={"u": p["u"]})
    st = handlers.trace(handlers.seed(handlers.substitute(spectral, data={
        key: jnp.array([value, value]) for key, value in values.items()}), 5)).get_trace(
        t, np.stack([error, 2*error]), y=np.stack([y, y]),
        PERIOD=jnp.array([p["period"]]), mu_t0=jnp.array([p["t0"]]),
        mu_b=jnp.array([p["b"]]), mu_depths=jnp.array([p["rors"]**2]),
        mu_a_rs=jnp.array([p["a_rs"]]), mu_ecc=jnp.array([p["ecc"]]),
        mu_omega=jnp.array([p["omega"]]), ld_fixed=np.stack([p["u"], p["u"]]))
    for obs, factor in [(wt["obs"]["fn"], 1.), (st["obs"]["fn"], np.array([[1.], [2.]]))]:
        expected_error = np.sqrt((error*factor)**2+jitter**2)
        np.testing.assert_allclose(obs.scale, expected_error, rtol=2e-14)
        np.testing.assert_allclose(np.asarray(obs.loc)[..., selected],
            np.broadcast_to(mean, obs.loc.shape)[..., selected], rtol=0., atol=2e-9)
        expected_logp = norm.logpdf(y, loc=obs.loc, scale=expected_error)
        np.testing.assert_allclose(obs.log_prob(y), expected_logp, atol=2e-12)
    np.testing.assert_allclose(st["obs"]["fn"].loc[0], wt["obs"]["fn"].loc, atol=2e-12, rtol=0.)


@pytest.mark.parametrize("model_name", ["eclipse", "phase_curve"])
def test_emission_gp_matches_independent_dense_matern_likelihood(model_name):
    p = parameters()
    t = p["t0"] + p["period"]*np.linspace(.3, .7, 61)
    specs, config = configuration(p, model_name)
    model = create_whitelight_model(detrend_type="linear+gp", ld_mode="fixed",
        param_method="a_rs", surface_config=config, parameter_priors=specs, gp_solver="serial")
    values = {"c": 1.0001, "v": .0003, "log_jitter": np.log(20e-6),
              "GP_log_sigma": np.log(35e-6), "GP_log_rho": np.log(.04)}
    mean = (1 + np.asarray(compute_surface_model(p, t, model=model_name)))*(values["c"]+values["v"]*(t-t.min()))
    error = np.linspace(15e-6, 40e-6, t.size)
    y = mean + 25e-6*np.cos(np.arange(t.size)/5)
    trace = handlers.trace(handlers.seed(handlers.substitute(model, data=values), 3)).get_trace(
        t, error, y=y, prior_params={"u": p["u"]})
    distance = np.sqrt(3)*np.abs(t[:, None]-t[None, :])/.04
    covariance = (35e-6)**2*(1+distance)*np.exp(-distance)
    covariance += np.diag(error**2+(20e-6)**2)
    expected = multivariate_normal.logpdf(y, mean=mean, cov=covariance)
    np.testing.assert_allclose(trace["obs"]["fn"].log_prob(y), expected, rtol=0., atol=2e-8)



def test_external_first_harmonic_injection_recovers_coefficients():
    p = parameters()
    t = p["t0"] + p["period"]*np.linspace(-.05, 1.05, 1201)
    star, visible = batman_components(p, t)
    selected = star == 1.
    t, star, visible = t[selected], star[selected], visible[selected]
    basis = prepare_emission_light_curve_basis(p, t, model="phase_curve")
    injected = star + harmonic_flux(p, t)*visible
    # f_d, C1, D1 are linear coordinates. Fit them with an unknown
    # multiplicative baseline: star*c plus three emission columns.
    design = np.column_stack([1+np.asarray(basis.baseline), basis.uniform,
                              basis.phase_cos, basis.phase_sin])
    coefficients = np.linalg.lstsq(design, 1.0003*injected, rcond=None)[0]
    np.testing.assert_allclose(coefficients[1:]/coefficients[0],
                               [p["eclipse_depth"], p["phase_c1"], p["phase_d1"]],
                               atol=2e-10, rtol=0.)
