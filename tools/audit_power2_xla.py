"""Isolated XLA cost/timing audit for the fixed-geometry Power-2 path."""

import json
import re
import statistics
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from jaxoplanet.core.limb_dark import light_curve as stock_light_curve
from models.common import get_I_power2
from models.detrend import _prepare_power2_poly
from models.jaxoplanet.limb_dark_fused import light_curve as fused_light_curve
from models.jaxoplanet.limb_dark_streamed import light_curve as streamed_light_curve
from tools.prototype_power2_circle_quadrature import (
    make_power2_flux,
    make_power2_flux_unified,
)


BATCH = 64
N_TIME = 256
REPEATS = 50
DURATION = 0.12
IMPACT = 0.35
DT = jnp.linspace(-0.5 * DURATION, 0.5 * DURATION, N_TIME)
MASK = jnp.abs(DT) < 0.5 * DURATION
THETA = jnp.column_stack(
    (
        jnp.linspace(0.075, 0.14, BATCH),
        jnp.linspace(0.1, 0.9, BATCH),
        jnp.linspace(0.05, 1.0, BATCH),
    )
)
POLYS = {
    degree: _prepare_power2_poly(degree=degree, n_mu=300)
    for degree in (2, 4, 6, 8, 10, 12)
}
NATIVE_POWER2 = {order: make_power2_flux(order) for order in (12, 16, 24)}
NATIVE_POWER2_UNIFIED = {
    order: make_power2_flux_unified(order) for order in (12, 16, 24)
}


def projection(theta, degree=12):
    mus, matrix = POLYS[degree]
    profile = get_I_power2(theta[:, 1, None], theta[:, 2, None], mus[None, :])
    return (matrix @ (1.0 - profile).T).T


def separation(rors):
    speed = 2.0 * jnp.sqrt(
        jnp.maximum(0.0, (1.0 + rors) ** 2 - IMPACT**2)
    ) / DURATION
    return jnp.sqrt((speed[:, None] * DT[None, :]) ** 2 + IMPACT**2)


def forward(theta, kernel, degree=12):
    u = projection(theta, degree=degree)
    z = separation(theta[:, 0])
    flux = jax.vmap(lambda one_u, one_z, r: kernel(one_u, one_z, r, order=10))(
        u, z, theta[:, 0]
    )
    return jnp.where(MASK[None, :], flux, 0.0)


def forward_preprojected(packed, kernel):
    rors = packed[:, 0]
    u = packed[:, 1:]
    z = separation(rors)
    flux = jax.vmap(lambda one_u, one_z, r: kernel(one_u, one_z, r, order=10))(
        u, z, rors
    )
    return jnp.where(MASK[None, :], flux, 0.0)


def forward_native(theta, order, *, unified=False):
    """Evaluate the direct Power-2 Green-contour prototype as flux deltas."""
    z = separation(theta[:, 0])
    evaluators = NATIVE_POWER2_UNIFIED if unified else NATIVE_POWER2
    evaluator = evaluators[order]
    flux = jax.vmap(
        lambda one_z, r, c, alpha: jax.vmap(
            lambda one_value: evaluator(one_value, r, c, alpha)
        )(one_z)
    )(z, theta[:, 0], theta[:, 1], theta[:, 2])
    return jnp.where(MASK[None, :], flux - 1.0, 0.0)


def value_grad(fun):
    def objective(arg):
        flux = fun(arg)
        return 0.5 * jnp.sum((flux / 1.0e-4) ** 2)
    return jax.value_and_grad(objective)


def native_point_gradients(theta, order):
    """Per-cadence derivatives (z, r, c, alpha) for quadrature convergence."""
    z = separation(theta[:, 0])
    shape = z.shape
    r = jnp.broadcast_to(theta[:, 0, None], shape)
    c = jnp.broadcast_to(theta[:, 1, None], shape)
    alpha = jnp.broadcast_to(theta[:, 2, None], shape)
    evaluator = NATIVE_POWER2_UNIFIED[order]
    gradients = jax.vmap(
        jax.grad(evaluator, argnums=(0, 1, 2, 3))
    )(z.reshape(-1), r.reshape(-1), c.reshape(-1), alpha.reshape(-1))
    return jnp.stack(gradients, axis=-1).reshape(shape + (4,))


def ready(value):
    for leaf in jax.tree.leaves(value):
        leaf.block_until_ready()


def profile(name, fun, arg):
    start = time.perf_counter()
    compiled = jax.jit(fun).lower(arg).compile()
    compile_seconds = time.perf_counter() - start
    ready(compiled(arg))
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        ready(compiled(arg))
        timings.append(time.perf_counter() - start)
    raw_analysis = compiled.cost_analysis()
    analysis = {
        key: raw_analysis[key]
        for key in ("flops", "transcendentals", "bytes accessed")
        if key in raw_analysis
    }
    hlo = compiled.as_text().lower()
    memory = compiled.memory_analysis()
    memory_analysis = {
        key: int(getattr(memory, key))
        for key in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
        )
        if getattr(memory, key, None) is not None
    }
    needles = {
        "atan2": r"atan2",
        "cosine": r"cosine",
        "sine": r"sine",
        "sqrt": r"sqrt",
        "power": r"power",
        "exponential": r"exponential",
        "log": r"\blog\b",
        "fusion": r"fusion",
        "reduce": r"reduce",
        "dot": r"\bdot\(",
    }
    return {
        "name": name,
        "compile_seconds": compile_seconds,
        "median_ms": 1e3 * statistics.median(timings),
        "p05_ms": 1e3 * float(np.percentile(timings, 5)),
        "p95_ms": 1e3 * float(np.percentile(timings, 95)),
        "cost_analysis": {key: float(value) for key, value in analysis.items()},
        "memory_analysis": memory_analysis,
        "optimized_hlo_bytes": len(hlo.encode("utf-8")),
        "hlo_token_counts": {
            key: len(re.findall(pattern, hlo)) for key, pattern in needles.items()
        },
    }


results = []
results.append(profile("projection_value_grad", value_grad(projection), THETA))
results.append(profile("separation_value_grad", value_grad(lambda x: separation(x[:, 0])), THETA))

preprojected = jnp.concatenate((THETA[:, :1], projection(THETA)), axis=1)
for name, kernel in (
    ("stock", stock_light_curve),
    ("streamed", streamed_light_curve),
    ("fused", fused_light_curve),
):
    results.append(
        profile(
            f"{name}_full_power2_forward",
            lambda x, k=kernel: forward(x, k),
            THETA,
        )
    )
    results.append(
        profile(
            f"{name}_occultation_value_grad_preprojected",
            value_grad(lambda x, k=kernel: forward_preprojected(x, k)),
            preprojected,
        )
    )
    results.append(
        profile(
            f"{name}_full_power2_value_grad",
            value_grad(lambda x, k=kernel: forward(x, k)),
            THETA,
        )
    )

for order in (12, 16):
    results.append(
        profile(
            f"native_power2_order{order}_value_grad",
            value_grad(lambda x, o=order: forward_native(x, o)),
            THETA,
        )
    )
    results.append(
        profile(
            f"native_power2_unified_order{order}_value_grad",
            value_grad(
                lambda x, o=order: forward_native(x, o, unified=True)
            ),
            THETA,
        )
    )
    results.append(
        profile(
            f"native_power2_unified_order{order}_forward",
            lambda x, o=order: forward_native(x, o, unified=True),
            THETA,
        )
    )
    if order == 16:
        rematerialized = jax.checkpoint(
            lambda x: forward_native(x, 16, unified=True)
        )
        results.append(
            profile(
                "native_power2_unified_order16_remat_value_grad",
                value_grad(rematerialized),
                THETA,
            )
        )

reference = jax.jit(lambda x: forward(x, stock_light_curve, degree=12))(THETA)
ready(reference)
degree_accuracy = []
for degree in (2, 4, 6, 8, 10):
    candidate = jax.jit(lambda x, d=degree: forward(x, stock_light_curve, degree=d))(THETA)
    ready(candidate)
    error = np.asarray(candidate - reference)
    degree_accuracy.append(
        {
            "degree": degree,
            "max_abs_error_ppm": float(np.max(np.abs(error)) * 1e6),
            "p99_abs_error_ppm": float(np.percentile(np.abs(error), 99) * 1e6),
            "rms_error_ppm": float(np.sqrt(np.mean(error**2)) * 1e6),
        }
    )


native_reference = jax.jit(lambda x: forward_native(x, 24))(THETA)
ready(native_reference)
unified_reference = jax.jit(
    lambda x: forward_native(x, 24, unified=True)
)(THETA)
ready(unified_reference)
gradient_reference = jax.jit(
    lambda x: native_point_gradients(x, 24)
)(THETA)
ready(gradient_reference)
native_accuracy = []
for order in (12, 16):
    candidate = jax.jit(lambda x, o=order: forward_native(x, o))(THETA)
    ready(candidate)
    unified_candidate = jax.jit(
        lambda x, o=order: forward_native(x, o, unified=True)
    )(THETA)
    ready(unified_candidate)
    gradient_candidate = jax.jit(
        lambda x, o=order: native_point_gradients(x, o)
    )(THETA)
    ready(gradient_candidate)
    native_error = np.asarray(candidate - native_reference)
    unified_error = np.asarray(unified_candidate - unified_reference)
    implementation_difference = np.asarray(unified_candidate - candidate)
    polynomial_error = np.asarray(candidate - reference)
    gradient_error = np.asarray(gradient_candidate - gradient_reference)
    gradient_reference_np = np.asarray(gradient_reference)
    active = np.broadcast_to(np.asarray(MASK)[None, :, None], gradient_error.shape)
    gradient_components = []
    for index, parameter in enumerate(("z", "rors", "c", "alpha")):
        component_active = active[..., index]
        difference = gradient_error[..., index][component_active]
        reference_component = gradient_reference_np[..., index][component_active]
        gradient_components.append(
            {
                "parameter": parameter,
                "max_abs_difference": float(np.max(np.abs(difference))),
                "p99_abs_difference": float(
                    np.percentile(np.abs(difference), 99)
                ),
                "relative_l2": float(
                    np.linalg.norm(difference)
                    / max(np.linalg.norm(reference_component), 1.0e-300)
                ),
            }
        )
    native_accuracy.append(
        {
            "order": order,
            "vs_native_order24": {
                "max_abs_error_ppm": float(np.max(np.abs(native_error)) * 1e6),
                "p99_abs_error_ppm": float(
                    np.percentile(np.abs(native_error), 99) * 1e6
                ),
                "rms_error_ppm": float(np.sqrt(np.mean(native_error**2)) * 1e6),
            },
            "unified_vs_unified_order24": {
                "max_abs_error_ppm": float(
                    np.max(np.abs(unified_error)) * 1e6
                ),
                "p99_abs_error_ppm": float(
                    np.percentile(np.abs(unified_error), 99) * 1e6
                ),
                "rms_error_ppm": float(
                    np.sqrt(np.mean(unified_error**2)) * 1e6
                ),
            },
            "unified_vs_branched_same_order": {
                "max_abs_difference_ppm": float(
                    np.max(np.abs(implementation_difference)) * 1e6
                ),
                "p99_abs_difference_ppm": float(
                    np.percentile(np.abs(implementation_difference), 99) * 1e6
                ),
                "rms_difference_ppm": float(
                    np.sqrt(np.mean(implementation_difference**2)) * 1e6
                ),
            },
            "unified_gradient_convergence_vs_order24": gradient_components,
            "vs_production_degree12": {
                "max_abs_difference_ppm": float(
                    np.max(np.abs(polynomial_error)) * 1e6
                ),
                "p99_abs_difference_ppm": float(
                    np.percentile(np.abs(polynomial_error), 99) * 1e6
                ),
                "rms_difference_ppm": float(
                    np.sqrt(np.mean(polynomial_error**2)) * 1e6
                ),
            },
        }
    )

print(json.dumps({
    "backend": jax.default_backend(),
    "device": str(jax.devices()[0]),
    "batch": BATCH,
    "n_time": N_TIME,
    "degree_accuracy_vs_12": degree_accuracy,
    "native_accuracy": native_accuracy,
    "profiles": results,
}, indent=2, sort_keys=True))
