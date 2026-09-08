"""Gaussian-process utilities used by the white-light models."""

from __future__ import annotations

import inspect
import os
import warnings
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import tinygp

from .common import compute_transit_model_auto


GP_MODEL_REVISION = "koala-gp-2026-09-08-tref-v1"
GP_SOLVER_ENV = "KOALA_GP_SOLVER"
GP_SOLVER_CHOICES = ("auto", "serial", "parallel")
# Uniform prior supports of the white-light GP hyperparameters (shared by both
# transit-engine builders).  The white-light stage uses them to keep the
# optimizer's start point strictly inside the support before NumPyro
# initialization.
GP_HYPERPARAMETER_BOUNDS = {
    "GP_log_sigma": (float(np.log(1e-5)), float(np.log(1e3))),
    "GP_log_rho": (float(np.log(0.007)), float(np.log(0.3))),
}
_PARALLEL_INSTALL = (
    'pip install "tinygp @ git+https://github.com/TylerFair/tinygp.git@'
    'koala-parallel-v1"'
)


class GPSolverUnavailableError(RuntimeError):
    """Raised when the requested tinygp solver is not installed."""


def gp_parallel_supported() -> bool:
    """Return whether the installed tinygp exposes the parallel QSM solver."""
    try:
        parameters = inspect.signature(
            tinygp.solvers.QuasisepSolver.__init__
        ).parameters
    except (TypeError, ValueError):
        return False
    return "parallel" in parameters


def _parallel_unavailable_message() -> str:
    return (
        "The parallel tinygp QuasisepSolver is unavailable. It requires "
        "Python >= 3.10 and the Koala-pinned parallel tinygp revision "
        "(upstream dfm/tinygp main); install it with:\n"
        f"{_PARALLEL_INSTALL}"
    )


def resolve_gp_solver(solver=None, *, device=None) -> str:
    """Resolve a configured GP solver mode to ``serial`` or ``parallel``."""
    requested = os.environ.get(GP_SOLVER_ENV, "auto") if solver is None else solver
    if not isinstance(requested, str):
        raise ValueError(
            f"GP solver must be one of {GP_SOLVER_CHOICES}; got {requested!r}"
        )
    requested = requested.lower()
    if requested not in GP_SOLVER_CHOICES:
        raise ValueError(
            f"Unknown GP solver {requested!r}; expected one of {GP_SOLVER_CHOICES}"
        )

    supported = gp_parallel_supported()
    if requested == "parallel":
        if not supported:
            raise GPSolverUnavailableError(_parallel_unavailable_message())
        return "parallel"
    if requested == "serial":
        return "serial"

    backend = jax.default_backend() if device is None else device
    if not isinstance(backend, str):
        backend = getattr(backend, "platform", str(backend))
    wants_parallel = backend.lower() in {"gpu", "tpu"}
    if wants_parallel and not supported:
        warnings.warn(
            "Parallel tinygp is unavailable; falling back to the serial GP solver.",
            RuntimeWarning,
            stacklevel=2,
        )
    return "parallel" if wants_parallel and supported else "serial"


def gp_solver_kwargs(solver, *, assume_sorted=False) -> dict:
    """Build keyword arguments forwarded by ``tinygp.GaussianProcess``."""
    if solver not in ("serial", "parallel"):
        raise ValueError("solver must already be resolved to 'serial' or 'parallel'")
    if solver == "parallel" and not gp_parallel_supported():
        raise GPSolverUnavailableError(_parallel_unavailable_message())
    kwargs = {
        "solver": tinygp.solvers.QuasisepSolver,
        "assume_sorted": bool(assume_sorted),
    }
    if gp_parallel_supported():
        kwargs["parallel"] = solver == "parallel"
    return kwargs


def validate_gp_times(t) -> np.ndarray:
    """Validate white-light GP coordinates on the host."""
    values = np.asarray(t, dtype=float)
    if values.ndim != 1:
        raise ValueError("GP times must be a 1-D array")
    if values.size == 0:
        raise ValueError("GP times must be nonempty")
    if not np.all(np.isfinite(values)):
        raise ValueError("GP times must contain only finite values")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("GP times must be sorted in nondecreasing order")
    return values


def _time_offset(t, t_ref):
    return t - (jnp.min(t) if t_ref is None else t_ref)


def _transit_signal(params, t):
    """Evaluate either transit engine while preserving the coordinate shape."""
    return jnp.reshape(compute_transit_model_auto(params, t), jnp.shape(t))


# --- GP MEAN FUNCTIONS ---
def compute_lc_gp_mean(params, t, t_ref=None):
    """The mean function for the simple GP model is transit plus a constant."""
    return _transit_signal(params, t) + params["c"]


def compute_lc_linear_gp_mean(params, t, t_ref=None):
    """The mean function for the linear + GP model."""
    lc_transit = _transit_signal(params, t)
    return lc_transit + params["c"] + params["v"] * _time_offset(t, t_ref)


def compute_lc_quadratic_gp_mean(params, t, t_ref=None):
    """The mean function for the quadratic + GP model."""
    lc_transit = _transit_signal(params, t)
    t_norm = _time_offset(t, t_ref)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2
    return lc_transit + trend


def compute_lc_cubic_gp_mean(params, t, t_ref=None):
    """The mean function for the cubic + GP model."""
    lc_transit = _transit_signal(params, t)
    t_norm = _time_offset(t, t_ref)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
    )
    return lc_transit + trend


def compute_lc_quartic_gp_mean(params, t, t_ref=None):
    """The mean function for the quartic + GP model."""
    lc_transit = _transit_signal(params, t)
    t_norm = _time_offset(t, t_ref)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
        + params["v4"] * t_norm**4
    )
    return lc_transit + trend


def compute_lc_explinear_gp_mean(params, t, t_ref=None):
    """The mean function for the exponential-linear + GP model."""
    lc_transit = _transit_signal(params, t)
    t_norm = _time_offset(t, t_ref)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["A"] * jnp.exp(-t_norm / params["tau"])
    )
    return lc_transit + trend


GP_MEAN_FUNCTIONS = {
    "gp": compute_lc_gp_mean,
    "linear+gp": compute_lc_linear_gp_mean,
    "quadratic+gp": compute_lc_quadratic_gp_mean,
    "cubic+gp": compute_lc_cubic_gp_mean,
    "quartic+gp": compute_lc_quartic_gp_mean,
    "explinear+gp": compute_lc_explinear_gp_mean,
}


def resolve_gp_mean_function(detrend_type):
    """Return the public mean function for a GP detrending mode."""
    try:
        return GP_MEAN_FUNCTIONS[detrend_type]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown GP detrend type {detrend_type!r}") from None


# --- SPECTROSCOPIC GP FUNCTIONS ---
def compute_lc_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    return lc_transit + params["c"] + params["A_gp"] * gp_trend


def compute_lc_linear_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend + params["A_gp"] * gp_trend


def compute_lc_quadratic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2
    return lc_transit + trend + params["A_gp"] * gp_trend


def compute_lc_cubic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
    )
    return lc_transit + trend + params["A_gp"] * gp_trend


def compute_lc_quartic_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["v2"] * t_norm**2
        + params["v3"] * t_norm**3
        + params["v4"] * t_norm**4
    )
    return lc_transit + trend + params["A_gp"] * gp_trend


def compute_lc_explinear_gp_spectroscopic(params, t, gp_trend):
    lc_transit = compute_transit_model_auto(params, t)
    t_norm = t - jnp.min(t)
    trend = (
        params["c"]
        + params["v"] * t_norm
        + params["A"] * jnp.exp(-t_norm / params["tau"])
    )
    return lc_transit + trend + params["A_gp"] * gp_trend


# --- GP BUILDERS AND TRAINING-POINT PREDICTION ---
def build_gp_model(
    params, t, error, *, mean_function, gp_solver=None,
    assume_sorted=False, t_ref=None,
):
    """Build the shared exact quasiseparable Matern-3/2 GP."""
    resolved_solver = resolve_gp_solver(gp_solver)
    if t_ref is None:
        t_ref = jnp.min(t)
    kernel = tinygp.kernels.quasisep.Matern32(
        scale=jnp.exp(params["GP_log_rho"]),
        sigma=jnp.exp(params["GP_log_sigma"]),
    )
    bound_mean = partial(mean_function, params, t_ref=t_ref)
    mean_value = mean_function(params, t, t_ref=t_ref)
    return tinygp.GaussianProcess(
        kernel, t, diag=error**2, mean=bound_mean, mean_value=mean_value,
        **gp_solver_kwargs(resolved_solver, assume_sorted=assume_sorted),
    )


def build_gp(params, t, error, *, gp_solver=None, assume_sorted=False, t_ref=None):
    return build_gp_model(
        params, t, error, mean_function=compute_lc_gp_mean,
        gp_solver=gp_solver, assume_sorted=assume_sorted, t_ref=t_ref,
    )


def build_gp_linear(
    params, t, error, *, gp_solver=None, assume_sorted=False, t_ref=None
):
    return build_gp_model(
        params, t, error, mean_function=compute_lc_linear_gp_mean,
        gp_solver=gp_solver, assume_sorted=assume_sorted, t_ref=t_ref,
    )


def build_gp_quadratic(
    params, t, error, *, gp_solver=None, assume_sorted=False, t_ref=None
):
    return build_gp_model(
        params, t, error, mean_function=compute_lc_quadratic_gp_mean,
        gp_solver=gp_solver, assume_sorted=assume_sorted, t_ref=t_ref,
    )


def build_gp_cubic(
    params, t, error, *, gp_solver=None, assume_sorted=False, t_ref=None
):
    return build_gp_model(
        params, t, error, mean_function=compute_lc_cubic_gp_mean,
        gp_solver=gp_solver, assume_sorted=assume_sorted, t_ref=t_ref,
    )


def build_gp_quartic(
    params, t, error, *, gp_solver=None, assume_sorted=False, t_ref=None
):
    return build_gp_model(
        params, t, error, mean_function=compute_lc_quartic_gp_mean,
        gp_solver=gp_solver, assume_sorted=assume_sorted, t_ref=t_ref,
    )


def build_gp_explinear(
    params, t, error, *, gp_solver=None, assume_sorted=False, t_ref=None
):
    return build_gp_model(
        params, t, error, mean_function=compute_lc_explinear_gp_mean,
        gp_solver=gp_solver, assume_sorted=assume_sorted, t_ref=t_ref,
    )


GP_BUILDERS = {
    "gp": build_gp,
    "linear+gp": build_gp_linear,
    "quadratic+gp": build_gp_quadratic,
    "cubic+gp": build_gp_cubic,
    "quartic+gp": build_gp_quartic,
    "explinear+gp": build_gp_explinear,
}


def resolve_gp_builder(detrend_type):
    """Return the public builder for a GP detrending mode."""
    try:
        return GP_BUILDERS[detrend_type]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown GP detrend type {detrend_type!r}") from None


def gp_inverse_diagonal(gp):
    """Return the diagonal of the observation covariance inverse in O(N)."""
    inv = gp.solver.matrix.inv
    try:
        supports_parallel = "parallel" in inspect.signature(inv).parameters
    except (TypeError, ValueError):
        supports_parallel = False
    if supports_parallel:
        inverse = inv(parallel=gp.solver.parallel)
    else:
        inverse = inv()
    return inverse.diag.d


def predict_gp_training_points(
    gp, y, *, diag=None, include_mean=True, return_var=True
):
    """Condition a quasiseparable GP at its own training coordinates in O(N)."""
    residual = y - gp.loc
    alpha = gp.solver.solve_triangular(residual)
    alpha = gp.solver.solve_triangular(alpha, transpose=True)
    noise_diag = jnp.broadcast_to(gp.noise.diag, jnp.shape(alpha))
    mean_value = y - noise_diag * alpha
    if not include_mean:
        mean_value = mean_value - gp.loc
    if not return_var:
        return mean_value

    if diag is None:
        diag_pred = jnp.sqrt(jnp.finfo(jnp.asarray(y).dtype).eps)
    else:
        diag_pred = jnp.asarray(diag, dtype=jnp.asarray(y).dtype)
    inverse_diag = gp_inverse_diagonal(gp)
    variance = diag_pred + noise_diag * (1.0 - noise_diag * inverse_diag)
    return mean_value, jnp.maximum(variance, 0.0)


@eqx.filter_jit
def _compute_gp_training_prediction_jit(
    params, t, error, y, *, detrend_type, gp_solver, assume_sorted,
    diag, include_mean,
):
    builder = resolve_gp_builder(detrend_type)
    gp = builder(
        params, t, error, gp_solver=gp_solver, assume_sorted=assume_sorted
    )
    return predict_gp_training_points(
        gp, y, diag=diag, include_mean=include_mean, return_var=True
    )


def compute_gp_training_prediction(
    params, t, error, y, *, detrend_type, gp_solver=None,
    assume_sorted=False, diag=None, include_mean=True,
):
    """Build and condition a GP in one filtered JIT-compiled operation."""
    resolved_solver = resolve_gp_solver(gp_solver)
    return _compute_gp_training_prediction_jit(
        params, t, error, y, detrend_type=detrend_type,
        gp_solver=resolved_solver, assume_sorted=assume_sorted,
        diag=diag, include_mean=include_mean,
    )
