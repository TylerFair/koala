"""Pre-NUTS checks for the differentiated NumPyro potential.

The diagnostic works in NumPyro's unconstrained sampling coordinates, exactly
where NUTS consumes the gradient.  In addition to checking finiteness, it
compares autodiff directional derivatives with symmetric finite differences
along deterministic directions.  It is diagnostic only: callers decide
whether a failed check should stop a fit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model


def _gradient_site_norms(gradient_tree):
    if isinstance(gradient_tree, dict):
        items = sorted(gradient_tree.items())
    else:
        items = [
            (f"leaf_{index}", value)
            for index, value in enumerate(jax.tree_util.tree_leaves(gradient_tree))
        ]
    result = {}
    for name, value in items:
        host = np.ravel(np.asarray(jax.device_get(value), dtype=np.float64))
        result[str(name)] = {
            "size": int(host.size),
            "l2": float(np.linalg.norm(host)),
            "max_abs": float(np.max(np.abs(host), initial=0.0)),
            "all_finite": bool(np.all(np.isfinite(host))),
        }
    return result


def diagnose_gradient_quality(
    model,
    init_params,
    t,
    yerr,
    y,
    prior_params=None,
    *,
    model_kwargs=None,
    num_directions=3,
    relative_tolerance=5.0e-4,
    seed=1729,
):
    """Evaluate potential-gradient finiteness and directional consistency.

    Parameters use the same call signature as the white-light model in
    ``fit_jwst.py``.  The finite-difference scale is proportional to the RMS
    unconstrained coordinate scale and to ``eps**(1/3)``, the standard
    roundoff/truncation compromise for a symmetric first derivative.
    """
    call_kwargs = {"y": y}
    if model_kwargs is not None:
        call_kwargs.update(dict(model_kwargs))
    if prior_params is not None:
        call_kwargs["prior_params"] = prior_params

    model_info = initialize_model(
        jax.random.PRNGKey(seed),
        model,
        init_strategy=init_to_value(values=init_params),
        dynamic_args=False,
        model_args=(t, yerr),
        model_kwargs=call_kwargs,
        validate_grad=False,
    )
    position = model_info.param_info.z
    flat_position, unravel = ravel_pytree(position)

    def flat_potential(flat_value):
        return model_info.potential_fn(unravel(flat_value))

    compiled_value_grad = jax.jit(jax.value_and_grad(flat_potential))
    potential, flat_gradient = compiled_value_grad(flat_position)
    potential, flat_gradient = jax.device_get((potential, flat_gradient))
    position_host = np.asarray(jax.device_get(flat_position), dtype=np.float64)
    gradient_host = np.asarray(flat_gradient, dtype=np.float64)

    finite = bool(
        np.isfinite(np.asarray(potential)).all()
        and np.all(np.isfinite(position_host))
        and np.all(np.isfinite(gradient_host))
    )
    count = int(position_host.size)
    # Identity-transformed absolute-time parameters can be ~60,000 BJD while
    # varying scientifically on ~1e-3 day scales.  A fixed float64 step is
    # still hundreds of thousands of ULPs at that origin, and avoids letting
    # the arbitrary BJD zero point inflate the scientific displacement.
    step = float(np.cbrt(np.finfo(np.float64).eps))

    rng = np.random.default_rng(seed)
    comparisons = []
    for index in range(max(0, int(num_directions))):
        direction = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        direction /= max(np.linalg.norm(direction), np.finfo(np.float64).tiny)
        direction_device = jnp.asarray(direction, dtype=flat_position.dtype)
        plus, _ = compiled_value_grad(flat_position + step * direction_device)
        minus, _ = compiled_value_grad(flat_position - step * direction_device)
        plus, minus = jax.device_get((plus, minus))
        finite_difference = float((plus - minus) / (2.0 * step))
        autodiff = float(np.vdot(gradient_host, direction))
        scale = max(1.0, abs(finite_difference), abs(autodiff))
        relative_error = abs(autodiff - finite_difference) / scale
        comparison_finite = bool(
            np.isfinite(autodiff)
            and np.isfinite(finite_difference)
            and np.isfinite(relative_error)
        )
        finite = finite and comparison_finite
        comparisons.append(
            {
                "index": index,
                "autodiff": autodiff,
                "finite_difference": finite_difference,
                "relative_error": float(relative_error),
                "finite": comparison_finite,
            }
        )

    max_directional_error = float(
        max((item["relative_error"] for item in comparisons), default=0.0)
    )
    gradient_tree = unravel(jnp.asarray(flat_gradient))
    passed = bool(finite and max_directional_error <= relative_tolerance)
    return {
        "passed": passed,
        "all_finite": finite,
        "potential": float(np.asarray(potential)),
        "parameter_count": count,
        "gradient_l2": float(np.linalg.norm(gradient_host)),
        "gradient_max_abs": float(
            np.max(np.abs(gradient_host), initial=0.0)
        ),
        "finite_difference_step": step,
        "relative_tolerance": float(relative_tolerance),
        "max_directional_relative_error": max_directional_error,
        "directions": comparisons,
        "sites": _gradient_site_norms(gradient_tree),
    }


__all__ = ["diagnose_gradient_quality"]
