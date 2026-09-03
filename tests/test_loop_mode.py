import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro

from models.jaxoplanet.builder import (
    LD_MAP_COEFFICIENTS,
    LD_MAP_POWER2_MAXTED,
    LD_MAP_QUADRATIC_SUMDIFF,
    LD_MAP_SING,
    _ld_variant_data_form,
)
from models.ld_parameterization import Power2MaxtedTransform


def _data_log_density(latent, center, scale, low, high, code, law):
    _, correction = _ld_variant_data_form(
        latent, center, scale, low, high, code, law
    )
    return dist.Normal(0.0, 1.0).log_prob(latent).sum(-1) + correction


def test_data_form_coefficient_and_maxted_potentials_are_exact():
    center = jnp.array([0.42, 0.58])
    scale = jnp.array([0.18, 0.21])
    low = jnp.array([0.0, 0.001])
    high = jnp.ones(2)
    coefficient = jnp.array([0.36, 0.64])
    expected = dist.TruncatedNormal(
        center, scale, low=low, high=high
    ).log_prob(coefficient).sum()
    actual = _data_log_density(
        coefficient, center, scale, low, high,
        LD_MAP_COEFFICIENTS, "power2",
    )
    assert jnp.allclose(actual, expected, rtol=0.0, atol=1e-12)

    transform = Power2MaxtedTransform()
    h = transform(coefficient)
    expected_h = expected - transform.log_abs_det_jacobian(coefficient, h)
    actual_h = _data_log_density(
        h, center, scale, low, high,
        LD_MAP_POWER2_MAXTED, "power2",
    )
    assert jnp.allclose(actual_h, expected_h, rtol=0.0, atol=1e-12)


def test_data_form_uniform_sumdiff_and_sing_potentials_are_exact():
    sumdiff = jnp.array([0.7, -0.2])
    low = jnp.array([-1.0, -2.0])
    high = jnp.array([2.0, 2.0])
    negative_scale = -jnp.ones(2)
    physical, correction = _ld_variant_data_form(
        sumdiff, jnp.zeros(2), negative_scale, low, high,
        LD_MAP_QUADRATIC_SUMDIFF, "quadratic",
    )
    assert jnp.allclose(physical, jnp.array([0.25, 0.45]))
    total = dist.Normal(0.0, 1.0).log_prob(sumdiff).sum() + correction
    assert jnp.allclose(total, dist.Uniform(low, high).log_prob(sumdiff).sum())

    sing = jnp.array([0.62, 0.03])
    center = jnp.array([0.6, 0.02])
    scale = jnp.array([0.08, 0.04])
    sing_low = jnp.array([0.0, -1.0])
    sing_high = jnp.array([1.0, 1.0])
    total = _data_log_density(
        sing, center, scale, sing_low, sing_high,
        LD_MAP_SING, "quadratic",
    )
    uplus = 1.0 - sing[0]
    expected = dist.TruncatedNormal(
        center[0], scale[0], low=0.0, high=1.0
    ).log_prob(sing[0]) + dist.TruncatedNormal(
        center[1], scale[1], low=-uplus / 4.0, high=uplus / 4.0
    ).log_prob(sing[1])
    assert jnp.allclose(total, expected, rtol=0.0, atol=1e-12)


def test_data_form_fixed_coefficients_ignore_auxiliary_latent():
    center = jnp.array([0.31, 0.47])
    first, first_correction = _ld_variant_data_form(
        jnp.array([-2.0, 0.5]), center, jnp.zeros(2), jnp.zeros(2),
        jnp.ones(2), LD_MAP_COEFFICIENTS, "quadratic",
    )
    second, second_correction = _ld_variant_data_form(
        jnp.array([1.2, -0.7]), center, jnp.zeros(2), jnp.zeros(2),
        jnp.ones(2), LD_MAP_COEFFICIENTS, "quadratic",
    )
    assert jnp.array_equal(first, center)
    assert jnp.array_equal(second, center)
    assert first_correction == 0.0
    assert second_correction == 0.0


def test_loop_order_reuses_uniform_quadratic_for_sing_offset():
    from tools.loop_fit import LOOP_VARIANTS

    names = [variant.name for variant in LOOP_VARIANTS]
    assert names.index("uniform_quadratic") < names.index("sing_quadratic")
    sing = next(variant for variant in LOOP_VARIANTS if variant.name == "sing_quadratic")
    assert sing.offset_source == "uniform_quadratic"


def test_loop_spectrum_schema_matches_pipeline_columns():
    from tools.loop_fit import SPECTRUM_COLUMNS

    assert SPECTRUM_COLUMNS == (
        "wavelength", "wavelength_err", "depth00", "depth_err00",
        "depth_ppm00", "depth_err_ppm00", "sampler_used",
    )


def test_three_data_variants_build_one_independent_program():
    from models.independent_nuts import build_independent_nuts_runner

    def model(t, yerr, y=None, ld_center=None, ld_map_code=0):
        latent = numpyro.sample(
            "ld_variant_latent",
            dist.Normal(0.0, 1.0).expand([1, 2]).to_event(1),
        )
        prediction = latent[:, :1] + 0.0 * ld_center[:, :1]
        numpyro.sample("obs", dist.Normal(prediction, yerr), obs=y)

    runner = build_independent_nuts_runner(
        model, lane_width=1, channel_varying_kwargs=("ld_center",),
        num_warmup=1, num_samples=2,
    )
    common = dict(
        t=jnp.zeros(1), yerr=jnp.ones((1, 1)), indiv_y=jnp.zeros((1, 1)),
        init_params={"ld_variant_latent": jnp.zeros((1, 2))},
    )
    for code in range(3):
        runner.run_raw(
            jax.random.PRNGKey(900 + code), **common,
            model_kwargs={"ld_center": jnp.full((1, 2), code / 10.0),
                          "ld_map_code": jnp.asarray(code)},
        )
    assert runner.program_build_count == 1
