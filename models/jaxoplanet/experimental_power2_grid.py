"""Experimental tensor-grid emulator for fixed-geometry power-2 transits.

This module is deliberately not wired into the production fitting path.  It
approximates the existing degree-12 jaxoplanet power-2 implementation only in
the narrow spectroscopic case where the cadence grid and transit geometry are
fixed and ``(rors, c, alpha)`` vary.  A grid is unusable through the checked
API until it has passed pointwise flux *and gradient* validation against the
exact implementation.

The interpolation is multilinear.  It is therefore cheap and differentiable
almost everywhere, but its derivatives are discontinuous at grid knots.  The
validation report records this approximation error; it is not a mathematical
certificate between the sampled validation points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..common import get_I_power2
from ..detrend import _prepare_power2_poly
from .core import (
    _compute_transit_model_duration,
    build_transit_phase_offsets,
    compute_transit_model,
    resolve_jaxoplanet_kernel,
)


Array = jax.Array


class Power2GridError(RuntimeError):
    """Base class for experimental grid failures."""


class UnvalidatedPower2GridError(Power2GridError):
    """Raised when an unvalidated grid is used through the checked API."""


class Power2GridOutOfDomainError(Power2GridError):
    """Raised instead of extrapolating beyond the grid domain."""


class Power2GridValidationError(Power2GridError):
    """Raised when a candidate grid fails its requested validation limits."""

    def __init__(self, report: "Power2GridValidationReport"):
        self.report = report
        super().__init__(
            "Power-2 grid validation failed: "
            f"max flux error={report.max_abs_error_ppm:.6g} ppm "
            f"(limit {report.flux_tolerance_ppm:.6g} ppm), "
            "max gradient error="
            f"{report.max_gradient_abs_error_ppm_per_unit:.6g} ppm/unit "
            f"(limit {report.gradient_abs_tolerance_ppm_per_unit:.6g}), "
            "max relative gradient L2="
            f"{report.max_gradient_relative_l2:.6g} "
            f"(limit {report.gradient_relative_tolerance:.6g})."
        )


def _require_x64() -> None:
    if not bool(jax.config.x64_enabled):
        raise RuntimeError(
            "The experimental power-2 grid requires JAX 64-bit mode. "
            "Set JAX_ENABLE_X64=1 before importing/running the fitting code; "
            "the emulator intentionally has no float32 mode."
        )


def _as_strict_axis(name: str, values: Sequence[float], *, minimum=None,
                    maximum=None) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with at least two knots.")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} knots must all be finite.")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} knots must be strictly increasing and unique.")
    if minimum is not None and np.any(axis < minimum):
        raise ValueError(f"{name} knots must be >= {minimum}.")
    if maximum is not None and np.any(axis > maximum):
        raise ValueError(f"{name} knots must be <= {maximum}.")
    return axis


def _hash_array(digest, array) -> None:
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def exact_power2_implementation_fingerprint(kernel: str = "stock") -> str:
    """Hash exact-model source and package versions used to validate artifacts."""
    resolved = resolve_jaxoplanet_kernel(
        kernel, ld_profile="power2", degree=12, keplerian=False
    )
    digest = hashlib.sha256()
    digest.update(b"power2-exact-implementation-v1")
    versions = {"jax": jax.__version__, "numpy": np.__version__}
    try:
        versions["jaxoplanet"] = importlib.metadata.version("jaxoplanet")
    except importlib.metadata.PackageNotFoundError:
        versions["jaxoplanet"] = "unknown"
    digest.update(json.dumps(versions, sort_keys=True).encode("utf-8"))
    exact_objects = (
        get_I_power2,
        _prepare_power2_poly,
        build_transit_phase_offsets,
        _compute_transit_model_duration,
        compute_transit_model,
    )
    for function in exact_objects:
        digest.update(inspect.getsource(function).encode("utf-8"))
    if resolved == "streamed":
        from . import limb_dark_streamed

        digest.update(inspect.getsource(limb_dark_streamed).encode("utf-8"))
    elif resolved == "fused":
        from . import limb_dark_fused

        digest.update(inspect.getsource(limb_dark_fused).encode("utf-8"))
    else:
        from jaxoplanet.core import limb_dark

        digest.update(inspect.getsource(limb_dark).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedPower2TransitGeometry:
    """Single-planet duration-based geometry held fixed by an emulator grid."""

    period: float
    duration: float
    t0: float
    b: float

    def __post_init__(self):
        values = np.asarray(
            [self.period, self.duration, self.t0, self.b], dtype=np.float64
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Fixed transit geometry values must all be finite.")
        if self.period <= 0.0:
            raise ValueError("period must be positive.")
        if self.duration <= 0.0 or self.duration >= self.period:
            raise ValueError("duration must be positive and shorter than period.")
        if self.b < 0.0:
            raise ValueError("impact parameter b must be non-negative.")


@dataclass(frozen=True)
class Power2GridValidationReport:
    """Pointwise comparison of a grid with the exact jaxoplanet model."""

    grid_fingerprint: str
    exact_kernel: str
    exact_implementation_fingerprint: str
    validation_backend: str
    seed: int
    samples_per_cell: int
    n_flux_points: int
    n_gradient_points: int
    flux_tolerance_ppm: float
    gradient_abs_tolerance_ppm_per_unit: float
    gradient_relative_tolerance: float
    max_abs_error_ppm: float
    p99_abs_error_ppm: float
    rms_error_ppm: float
    worst_parameters: tuple[float, float, float]
    max_gradient_abs_error_ppm_per_unit: float
    gradient_abs_error_ppm_per_unit: tuple[float, float, float]
    max_gradient_relative_l2: float
    gradient_relative_l2: tuple[float, float, float]
    worst_gradient_parameters: tuple[float, float, float]
    worst_gradient_time_index: int
    worst_gradient_time: float
    worst_gradient_parameter: str
    passed: bool

    def to_json_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Power2TransitGrid:
    """A fixed-cadence, fixed-geometry ``(rors, c, alpha)`` flux grid."""

    times: Array
    geometry: FixedPower2TransitGeometry
    rors_knots: Array
    c_knots: Array
    alpha_knots: Array
    flux_grid: Array
    exact_kernel: str = "stock"
    validation: Optional[Power2GridValidationReport] = None
    exact_implementation_fingerprint: str = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self):
        times = np.asarray(self.times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(times)):
            raise ValueError("times must all be finite.")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("times must be strictly increasing and unique.")

        rors = _as_strict_axis("rors", self.rors_knots, minimum=np.finfo(float).tiny)
        c_axis = _as_strict_axis("c", self.c_knots, minimum=0.0, maximum=1.0)
        alpha = _as_strict_axis(
            "alpha", self.alpha_knots, minimum=np.finfo(float).tiny
        )
        values = np.asarray(self.flux_grid, dtype=np.float64)
        expected_shape = (rors.size, c_axis.size, alpha.size, times.size)
        if values.shape != expected_shape:
            raise ValueError(
                f"flux_grid has shape {values.shape}, expected {expected_shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("flux_grid must contain only finite values.")

        resolved = resolve_jaxoplanet_kernel(
            self.exact_kernel, ld_profile="power2", degree=12, keplerian=False
        )
        if resolved not in {"stock", "streamed", "fused"}:
            raise ValueError(
                "The grid supports only the stock/streamed/fused degree-12 "
                "power-2 duration-based kernels."
            )

        object.__setattr__(self, "times", jnp.asarray(times, dtype=jnp.float64))
        object.__setattr__(self, "rors_knots", jnp.asarray(rors, dtype=jnp.float64))
        object.__setattr__(self, "c_knots", jnp.asarray(c_axis, dtype=jnp.float64))
        object.__setattr__(self, "alpha_knots", jnp.asarray(alpha, dtype=jnp.float64))
        object.__setattr__(self, "flux_grid", jnp.asarray(values, dtype=jnp.float64))

        digest = hashlib.sha256()
        digest.update(b"power2-transit-grid-v1")
        digest.update(json.dumps(asdict(self.geometry), sort_keys=True).encode("utf-8"))
        digest.update(str(self.exact_kernel).encode("ascii"))
        implementation_fingerprint = exact_power2_implementation_fingerprint(
            self.exact_kernel
        )
        digest.update(implementation_fingerprint.encode("ascii"))
        for array in (times, rors, c_axis, alpha, values):
            _hash_array(digest, array)
        fingerprint = digest.hexdigest()
        object.__setattr__(
            self,
            "exact_implementation_fingerprint",
            implementation_fingerprint,
        )
        object.__setattr__(self, "fingerprint", fingerprint)

        if self.validation is not None:
            if self.validation.grid_fingerprint != fingerprint:
                raise ValueError(
                    "Validation report fingerprint does not match this grid artifact."
                )
            if not self.validation.passed:
                raise ValueError("A failed validation report cannot be attached to a grid.")
            if (
                self.validation.exact_implementation_fingerprint
                != implementation_fingerprint
            ):
                raise ValueError(
                    "Validation report was produced by a different exact-model "
                    "implementation."
                )

    @property
    def estimated_bytes(self) -> int:
        return int(np.prod(self.flux_grid.shape, dtype=np.int64) * 8)

    @property
    def is_validated(self) -> bool:
        return self.validation is not None and self.validation.passed

    def with_validation(
        self, report: Power2GridValidationReport
    ) -> "Power2TransitGrid":
        """Attach a passing report only when it identifies this exact artifact."""
        if report.grid_fingerprint != self.fingerprint:
            raise ValueError("Validation report belongs to a different grid artifact.")
        if not report.passed:
            raise Power2GridValidationError(report)
        return replace(self, validation=report)

    @staticmethod
    def _axis_interval(axis: Array, value: Array):
        value = jnp.asarray(value, dtype=jnp.float64)
        index = jnp.searchsorted(axis, value, side="right") - 1
        index = jnp.clip(index, 0, axis.shape[0] - 2)
        low = axis[index]
        high = axis[index + 1]
        fraction = (value - low) / (high - low)
        valid = (
            jnp.isfinite(value)
            & (value >= axis[0])
            & (value <= axis[-1])
        )
        return index, fraction, valid

    def _interpolate_raw(self, rors, c, alpha):
        ir, fr, valid_r = self._axis_interval(self.rors_knots, rors)
        ic, fc, valid_c = self._axis_interval(self.c_knots, c)
        ia, fa, valid_a = self._axis_interval(self.alpha_knots, alpha)

        # Interpolate alpha, then c, then radius ratio.  Each corner is a full
        # cadence vector, so no large one-hot tensor is constructed.
        g = self.flux_grid
        a000 = g[ir, ic, ia]
        a001 = g[ir, ic, ia + 1]
        a010 = g[ir, ic + 1, ia]
        a011 = g[ir, ic + 1, ia + 1]
        a100 = g[ir + 1, ic, ia]
        a101 = g[ir + 1, ic, ia + 1]
        a110 = g[ir + 1, ic + 1, ia]
        a111 = g[ir + 1, ic + 1, ia + 1]

        c00 = a000 + fa * (a001 - a000)
        c01 = a010 + fa * (a011 - a010)
        c10 = a100 + fa * (a101 - a100)
        c11 = a110 + fa * (a111 - a110)
        r0 = c00 + fc * (c01 - c00)
        r1 = c10 + fc * (c11 - c10)
        flux = r0 + fr * (r1 - r0)
        return flux, valid_r & valid_c & valid_a

    def evaluate_with_status(
        self, rors, c, alpha, *, allow_unvalidated: bool = False
    ):
        """Return ``(flux, valid)`` without ever silently extrapolating.

        This method can be JIT compiled.  Outside the domain, or when the grid
        lacks a passing validation report and ``allow_unvalidated`` is false,
        every flux value is NaN and ``valid`` is false.  The explicit escape
        hatch exists so that :func:`validate_power2_grid` can evaluate a new
        candidate; production callers should never set it.
        """
        flux, domain_valid = self._interpolate_raw(rors, c, alpha)
        approved = bool(allow_unvalidated or self.is_validated)
        valid = domain_valid & jnp.asarray(approved)
        return jnp.where(valid, flux, jnp.nan), valid

    def evaluate_checked(
        self, rors, c, alpha, *, allow_unvalidated: bool = False
    ) -> Array:
        """Host-side strict evaluator that raises on invalid use.

        Use :meth:`evaluate_with_status` inside JIT-compiled code because a
        Python exception cannot be raised conditionally from an XLA program.
        """
        concrete = []
        for name, value in (("rors", rors), ("c", c), ("alpha", alpha)):
            if isinstance(value, jax.core.Tracer):
                raise TypeError(
                    "evaluate_checked cannot receive JAX tracers; use "
                    "evaluate_with_status and propagate its validity flag."
                )
            array = np.asarray(value)
            if array.ndim != 0:
                raise ValueError(f"{name} must be scalar; vmap scalar evaluations.")
            concrete.append(float(array))

        axes = (
            ("rors", np.asarray(self.rors_knots)),
            ("c", np.asarray(self.c_knots)),
            ("alpha", np.asarray(self.alpha_knots)),
        )
        for value, (name, axis) in zip(concrete, axes):
            if not np.isfinite(value) or value < axis[0] or value > axis[-1]:
                raise Power2GridOutOfDomainError(
                    f"{name}={value!r} is outside [{axis[0]}, {axis[-1]}]; "
                    "the experimental grid refuses to extrapolate."
                )
        if not allow_unvalidated and not self.is_validated:
            raise UnvalidatedPower2GridError(
                "This grid has no passing exact-model validation report."
            )
        flux, valid = self.evaluate_with_status(
            *concrete, allow_unvalidated=allow_unvalidated
        )
        if not bool(np.asarray(valid)):
            raise Power2GridError("Grid evaluation failed its validity check.")
        return flux


def make_exact_power2_evaluator(
    times: Sequence[float],
    geometry: FixedPower2TransitGeometry,
    *,
    kernel: str = "stock",
) -> Callable[[Array], Array]:
    """Return the existing degree-12 jaxoplanet power-2 model for ``theta``.

    ``theta`` is ``[rors, c, alpha]``.  Shared phase offsets are precomputed,
    matching the fixed-geometry spectroscopic optimization already available
    in :mod:`models.jaxoplanet.core`.
    """
    _require_x64()
    times_array = np.asarray(times, dtype=np.float64)
    if times_array.ndim != 1 or times_array.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(times_array)):
        raise ValueError("times must all be finite.")
    if times_array.size > 1 and not np.all(np.diff(times_array) > 0.0):
        raise ValueError("times must be strictly increasing and unique.")
    resolved = resolve_jaxoplanet_kernel(
        kernel, ld_profile="power2", degree=12, keplerian=False
    )
    if resolved not in {"stock", "streamed", "fused"}:
        raise ValueError("Unsupported exact power-2 kernel.")

    t = jnp.asarray(times_array, dtype=jnp.float64)
    mus, projection = _prepare_power2_poly(degree=12, n_mu=300)
    periods = jnp.asarray([geometry.period], dtype=jnp.float64)
    durations = jnp.asarray([geometry.duration], dtype=jnp.float64)
    t0s = jnp.asarray([geometry.t0], dtype=jnp.float64)
    bs = jnp.asarray([geometry.b], dtype=jnp.float64)
    phase_offsets, phase_mask = build_transit_phase_offsets(
        t, periods, t0s, durations
    )

    def exact(theta: Array) -> Array:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        rors, c_value, alpha = theta[0], theta[1], theta[2]
        profile = get_I_power2(c_value, alpha, mus)
        u = projection @ (1.0 - profile)
        params = {
            "period": periods,
            "duration": durations,
            "t0": t0s,
            "b": bs,
            "rors": jnp.asarray([rors], dtype=jnp.float64),
            "u": u,
            "_jaxoplanet_kernel": kernel,
            "_ld_profile": "power2",
            "_transit_phase_offsets": phase_offsets,
            "_transit_phase_mask": phase_mask,
        }
        return compute_transit_model(params, t)

    return exact


def build_power2_transit_grid(
    times: Sequence[float],
    geometry: FixedPower2TransitGeometry,
    rors_knots: Sequence[float],
    c_knots: Sequence[float],
    alpha_knots: Sequence[float],
    *,
    exact_kernel: str = "stock",
    build_batch_size: int = 128,
    max_grid_bytes: int = 2 * 1024**3,
) -> Power2TransitGrid:
    """Evaluate the exact model on a Cartesian grid in bounded batches."""
    _require_x64()
    times_array = np.asarray(times, dtype=np.float64)
    rors = _as_strict_axis("rors", rors_knots, minimum=np.finfo(float).tiny)
    c_axis = _as_strict_axis("c", c_knots, minimum=0.0, maximum=1.0)
    alpha = _as_strict_axis(
        "alpha", alpha_knots, minimum=np.finfo(float).tiny
    )
    if geometry.b >= 1.0 + float(rors[-1]):
        raise ValueError(
            "The complete radius-ratio domain is non-transiting for the fixed b."
        )
    if build_batch_size <= 0:
        raise ValueError("build_batch_size must be positive.")
    if max_grid_bytes <= 0:
        raise ValueError("max_grid_bytes must be positive.")
    estimated_bytes = int(
        rors.size * c_axis.size * alpha.size * times_array.size * 8
    )
    if estimated_bytes > max_grid_bytes:
        raise MemoryError(
            "Refusing to construct a power-2 grid requiring approximately "
            f"{estimated_bytes / 1024**3:.3f} GiB; max_grid_bytes permits "
            f"{max_grid_bytes / 1024**3:.3f} GiB."
        )

    rr, cc, aa = np.meshgrid(rors, c_axis, alpha, indexing="ij")
    theta = np.stack([rr.ravel(), cc.ravel(), aa.ravel()], axis=-1)
    exact = make_exact_power2_evaluator(
        times_array, geometry, kernel=exact_kernel
    )
    effective_batch = min(int(build_batch_size), theta.shape[0])
    batched_exact = jax.jit(jax.vmap(exact))
    values = np.empty((theta.shape[0], times_array.size), dtype=np.float64)
    for start in range(0, theta.shape[0], effective_batch):
        stop = min(start + effective_batch, theta.shape[0])
        chunk = theta[start:stop]
        if chunk.shape[0] < effective_batch:
            padding = np.repeat(chunk[-1:, :], effective_batch - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)
        evaluated = jax.block_until_ready(
            batched_exact(jnp.asarray(chunk, dtype=jnp.float64))
        )
        values[start:stop] = np.asarray(evaluated)[: stop - start]

    shaped = values.reshape(
        rors.size, c_axis.size, alpha.size, times_array.size
    )
    return Power2TransitGrid(
        times=times_array,
        geometry=geometry,
        rors_knots=rors,
        c_knots=c_axis,
        alpha_knots=alpha,
        flux_grid=shaped,
        exact_kernel=exact_kernel,
    )


def stratified_power2_validation_points(
    grid: Power2TransitGrid,
    *,
    samples_per_cell: int = 1,
    seed: int = 0,
    max_points: int = 200_000,
) -> np.ndarray:
    """Return cell-stratified interior points (center plus random points)."""
    if samples_per_cell <= 0:
        raise ValueError("samples_per_cell must be positive.")
    axes = [
        np.asarray(grid.rors_knots),
        np.asarray(grid.c_knots),
        np.asarray(grid.alpha_knots),
    ]
    cell_shape = tuple(axis.size - 1 for axis in axes)
    n_cells = int(np.prod(cell_shape, dtype=np.int64))
    n_points = n_cells * int(samples_per_cell)
    if n_points > max_points:
        raise ValueError(
            f"Validation requests {n_points} points, exceeding max_points={max_points}."
        )
    rng = np.random.default_rng(seed)
    points = np.empty((n_points, 3), dtype=np.float64)
    cursor = 0
    for index in np.ndindex(cell_shape):
        lows = np.asarray([axes[d][index[d]] for d in range(3)])
        highs = np.asarray([axes[d][index[d] + 1] for d in range(3)])
        fractions = rng.uniform(size=(samples_per_cell, 3))
        fractions[0, :] = 0.5
        points[cursor:cursor + samples_per_cell] = (
            lows + fractions * (highs - lows)
        )
        cursor += samples_per_cell
    return points


def _evaluate_in_fixed_batches(function, theta: np.ndarray, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    effective_batch = min(int(batch_size), theta.shape[0])
    batched = jax.jit(jax.vmap(function))
    outputs = []
    for start in range(0, theta.shape[0], effective_batch):
        stop = min(start + effective_batch, theta.shape[0])
        chunk = theta[start:stop]
        if chunk.shape[0] < effective_batch:
            padding = np.repeat(chunk[-1:, :], effective_batch - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)
        result = jax.block_until_ready(
            batched(jnp.asarray(chunk, dtype=jnp.float64))
        )
        outputs.append(np.asarray(result)[: stop - start])
    return np.concatenate(outputs, axis=0)


def validate_power2_grid(
    grid: Power2TransitGrid,
    *,
    samples_per_cell: int = 1,
    seed: int = 0,
    flux_tolerance_ppm: float = 1.0,
    gradient_abs_tolerance_ppm_per_unit: float = 25.0,
    gradient_relative_tolerance: float = 0.02,
    gradient_points: int = 32,
    validation_batch_size: int = 128,
    max_validation_points: int = 200_000,
) -> Power2GridValidationReport:
    """Compare an emulator with exact fluxes and parameter Jacobians.

    Flux validation samples every interpolation cell.  The first point in each
    cell is its center and additional points are deterministic pseudorandom
    strata.  Gradient checks use an evenly spread subset of those points.
    Passing this test establishes only sampled error bounds, not an analytic
    guarantee over the continuous domain.
    """
    _require_x64()
    tolerances = np.asarray(
        [
            flux_tolerance_ppm,
            gradient_abs_tolerance_ppm_per_unit,
            gradient_relative_tolerance,
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("Validation tolerances must be finite and non-negative.")
    if gradient_points <= 0:
        raise ValueError("gradient_points must be positive.")

    points = stratified_power2_validation_points(
        grid,
        samples_per_cell=samples_per_cell,
        seed=seed,
        max_points=max_validation_points,
    )
    exact = make_exact_power2_evaluator(
        np.asarray(grid.times), grid.geometry, kernel=grid.exact_kernel
    )

    def approximate(theta):
        return grid._interpolate_raw(theta[0], theta[1], theta[2])[0]

    exact_flux = _evaluate_in_fixed_batches(
        exact, points, validation_batch_size
    )
    approximate_flux = _evaluate_in_fixed_batches(
        approximate, points, validation_batch_size
    )
    abs_error_ppm = np.abs(approximate_flux - exact_flux) * 1.0e6
    per_point_max = np.max(abs_error_ppm, axis=1)
    worst_index = int(np.argmax(per_point_max))

    n_gradient = min(int(gradient_points), points.shape[0])
    gradient_indices = np.unique(
        np.linspace(0, points.shape[0] - 1, n_gradient, dtype=np.int64)
    )
    gradient_theta = points[gradient_indices]
    exact_jacobian = jax.jacrev(exact)
    approximate_jacobian = jax.jacrev(approximate)
    exact_grad = _evaluate_in_fixed_batches(
        exact_jacobian, gradient_theta, min(validation_batch_size, 16)
    )
    approximate_grad = _evaluate_in_fixed_batches(
        approximate_jacobian, gradient_theta, min(validation_batch_size, 16)
    )
    # Jacobian shape is (n_point, n_time, 3).  Each parameter has different
    # units, so both per-unit absolute errors and dimensionless L2 errors are
    # retained separately for rors, c, and alpha.
    gradient_error = approximate_grad - exact_grad
    worst_gradient_flat = int(np.argmax(np.abs(gradient_error)))
    worst_gradient_point, worst_gradient_time_index, worst_gradient_parameter = (
        np.unravel_index(worst_gradient_flat, gradient_error.shape)
    )
    gradient_abs_by_parameter = np.max(
        np.abs(gradient_error), axis=(0, 1)
    ) * 1.0e6
    exact_norm = np.linalg.norm(exact_grad, axis=(0, 1))
    error_norm = np.linalg.norm(gradient_error, axis=(0, 1))
    gradient_relative = error_norm / np.maximum(exact_norm, 1.0e-30)

    max_flux = float(np.max(abs_error_ppm))
    max_gradient_abs = float(np.max(gradient_abs_by_parameter))
    max_gradient_relative = float(np.max(gradient_relative))
    passed = bool(
        max_flux <= flux_tolerance_ppm
        and max_gradient_abs <= gradient_abs_tolerance_ppm_per_unit
        and max_gradient_relative <= gradient_relative_tolerance
    )
    return Power2GridValidationReport(
        grid_fingerprint=grid.fingerprint,
        exact_kernel=grid.exact_kernel,
        exact_implementation_fingerprint=(
            grid.exact_implementation_fingerprint
        ),
        validation_backend=jax.default_backend(),
        seed=int(seed),
        samples_per_cell=int(samples_per_cell),
        n_flux_points=int(points.shape[0]),
        n_gradient_points=int(gradient_indices.size),
        flux_tolerance_ppm=float(flux_tolerance_ppm),
        gradient_abs_tolerance_ppm_per_unit=float(
            gradient_abs_tolerance_ppm_per_unit
        ),
        gradient_relative_tolerance=float(gradient_relative_tolerance),
        max_abs_error_ppm=max_flux,
        p99_abs_error_ppm=float(np.percentile(abs_error_ppm, 99.0)),
        rms_error_ppm=float(np.sqrt(np.mean(np.square(abs_error_ppm)))),
        worst_parameters=tuple(float(x) for x in points[worst_index]),
        max_gradient_abs_error_ppm_per_unit=max_gradient_abs,
        gradient_abs_error_ppm_per_unit=tuple(
            float(x) for x in gradient_abs_by_parameter
        ),
        max_gradient_relative_l2=max_gradient_relative,
        gradient_relative_l2=tuple(float(x) for x in gradient_relative),
        worst_gradient_parameters=tuple(
            float(x) for x in gradient_theta[worst_gradient_point]
        ),
        worst_gradient_time_index=int(worst_gradient_time_index),
        worst_gradient_time=float(np.asarray(grid.times)[worst_gradient_time_index]),
        worst_gradient_parameter=("rors", "c", "alpha")[worst_gradient_parameter],
        passed=passed,
    )


def build_validated_power2_transit_grid(*args, **kwargs) -> Power2TransitGrid:
    """Build, validate, and return a grid, raising if any limit is missed.

    Grid-construction keywords are passed directly.  Validation controls use a
    ``validation_`` prefix, for example ``validation_flux_tolerance_ppm``.
    """
    validation_names = {
        "validation_samples_per_cell": "samples_per_cell",
        "validation_seed": "seed",
        "validation_flux_tolerance_ppm": "flux_tolerance_ppm",
        "validation_gradient_abs_tolerance_ppm_per_unit": (
            "gradient_abs_tolerance_ppm_per_unit"
        ),
        "validation_gradient_relative_tolerance": "gradient_relative_tolerance",
        "validation_gradient_points": "gradient_points",
        "validation_batch_size": "validation_batch_size",
        "validation_max_points": "max_validation_points",
    }
    validation_kwargs = {}
    for public_name, internal_name in validation_names.items():
        if public_name in kwargs:
            validation_kwargs[internal_name] = kwargs.pop(public_name)
    grid = build_power2_transit_grid(*args, **kwargs)
    report = validate_power2_grid(grid, **validation_kwargs)
    if not report.passed:
        raise Power2GridValidationError(report)
    return grid.with_validation(report)


def save_power2_grid(path: str | Path, grid: Power2TransitGrid) -> None:
    """Save a grid artifact; only validated artifacts may be persisted."""
    if not grid.is_validated:
        raise UnvalidatedPower2GridError(
            "Refusing to persist a grid without a passing validation report."
        )
    metadata = {
        "format": "experimental-power2-grid-v1",
        "geometry": asdict(grid.geometry),
        "exact_kernel": grid.exact_kernel,
        "exact_implementation_fingerprint": (
            grid.exact_implementation_fingerprint
        ),
        "fingerprint": grid.fingerprint,
        "validation": grid.validation.to_json_dict(),
    }
    destination = Path(path)
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            times=np.asarray(grid.times),
            rors_knots=np.asarray(grid.rors_knots),
            c_knots=np.asarray(grid.c_knots),
            alpha_knots=np.asarray(grid.alpha_knots),
            flux_grid=np.asarray(grid.flux_grid),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


def load_power2_grid(path: str | Path) -> Power2TransitGrid:
    """Load and fingerprint-check a persisted validated grid artifact."""
    with np.load(Path(path), allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata.get("format") != "experimental-power2-grid-v1":
            raise ValueError("Unsupported power-2 grid artifact format.")
        report_data = dict(metadata["validation"])
        for tuple_name in (
            "worst_parameters",
            "gradient_abs_error_ppm_per_unit",
            "gradient_relative_l2",
            "worst_gradient_parameters",
        ):
            report_data[tuple_name] = tuple(report_data[tuple_name])
        report = Power2GridValidationReport(**report_data)
        grid = Power2TransitGrid(
            times=payload["times"],
            geometry=FixedPower2TransitGeometry(**metadata["geometry"]),
            rors_knots=payload["rors_knots"],
            c_knots=payload["c_knots"],
            alpha_knots=payload["alpha_knots"],
            flux_grid=payload["flux_grid"],
            exact_kernel=metadata["exact_kernel"],
            validation=report,
        )
    if grid.fingerprint != metadata["fingerprint"]:
        raise ValueError("Power-2 grid artifact fingerprint is corrupt or stale.")
    return grid
