"""Contact-aligned solution-vector emulator for power-2 transits.

This evaluation-only prototype factorizes the production jaxoplanet light
curve into

``flux = solution_vector(separation, rors) @ greens(c, alpha) - 1``.

Only the 13-component geometric solution vector is interpolated.  Limb
darkening is transformed exactly at runtime, eliminating the previous ``c``
and ``alpha`` grid dimensions.  The separation coordinate is split at the
analytic overlap boundaries so both non-smooth contacts are fixed grid lines:

* full overlap: ``q = separation / (1 - rors)`` (inner contact at ``q=1``),
* partial overlap: ``q = (separation - (1-rors)) / (2*rors)``
  (inner/outer contacts at ``q=0/1``),
* no overlap: exact zero signal.

The module is not connected to the production fitter or artifact cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.core.limb_dark import greens_basis_transform, solution_vector

from ..common import get_I_power2
from .experimental_power2_grid import (
    FixedPower2TransitGeometry,
    _evaluate_in_fixed_batches,
    _hash_array,
    _require_x64,
)


RORS_PRIOR_BOUNDS = (float(np.sqrt(1.0e-5)), float(np.sqrt(0.5)))
POWER2_C_PRIOR_BOUNDS = (0.0, 1.0)
POWER2_ALPHA_PRIOR_BOUNDS = (0.001, 1.0)

# Keep the least-squares basis as NumPy constants so constructing it never
# happens under a JAX transform.  This is algebraically identical to
# ``_prepare_power2_poly(degree=12, n_mu=300)`` used by the production model.
_POWER2_MUS = np.linspace(0.0, 1.0, 300, endpoint=True, dtype=np.float64)
_POWER2_DESIGN = np.vander(
    1.0 - _POWER2_MUS, N=13, increasing=True
)[:, 1:]
_POWER2_PROJECTION = np.linalg.pinv(_POWER2_DESIGN)


def make_contact_basis_axes(
    *,
    n_rors: int = 513,
    n_full: int = 257,
    n_partial: int = 257,
    rors_bounds=RORS_PRIOR_BOUNDS,
):
    """Return axes with quadratic contact clustering.

    Radius knots are uniform because the leading occulted area is quadratic
    in radius; 513 knots keep the linear-interpolation area error below about
    0.5 ppm across the complete current spectroscopic prior.  Full-overlap
    points cluster at inner contact, while partial-overlap points cluster at
    both contacts.
    """
    for name, value in (
        ("n_rors", n_rors), ("n_full", n_full), ("n_partial", n_partial)
    ):
        if int(value) < 2:
            raise ValueError(f"{name} must be at least two.")
    r_min, r_max = (float(rors_bounds[0]), float(rors_bounds[1]))
    if not (np.isfinite(r_min) and np.isfinite(r_max) and 0.0 < r_min < r_max < 1.0):
        raise ValueError("rors_bounds must satisfy 0 < lower < upper < 1.")
    rors = np.linspace(r_min, r_max, int(n_rors), dtype=np.float64)
    full_u = np.linspace(0.0, 1.0, int(n_full), dtype=np.float64)
    full_q = 1.0 - np.square(1.0 - full_u)
    partial_u = np.linspace(0.0, 1.0, int(n_partial), dtype=np.float64)
    partial_q = np.square(np.sin(0.5 * np.pi * partial_u))
    return rors, full_q, partial_q


def _validate_axis(name, values, lower, upper):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with >=2 points.")
    if not np.all(np.isfinite(result)) or not np.all(np.diff(result) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    if abs(result[0] - lower) > 8 * np.finfo(float).eps:
        raise ValueError(f"{name} must start at {lower}.")
    if abs(result[-1] - upper) > 8 * np.finfo(float).eps:
        raise ValueError(f"{name} must end at {upper}.")
    return result


@dataclass(frozen=True)
class Power2ContactBasisGrid:
    geometry: FixedPower2TransitGeometry
    rors_knots: jax.Array
    full_q_knots: jax.Array
    partial_q_knots: jax.Array
    full_solution_grid: jax.Array
    partial_solution_grid: jax.Array
    c_bounds: tuple[float, float] = POWER2_C_PRIOR_BOUNDS
    alpha_bounds: tuple[float, float] = POWER2_ALPHA_PRIOR_BOUNDS
    fingerprint: str = field(init=False)

    def __post_init__(self):
        rors = np.asarray(self.rors_knots, dtype=np.float64)
        if (
            rors.ndim != 1
            or rors.size < 2
            or not np.all(np.isfinite(rors))
            or not np.all(np.diff(rors) > 0.0)
            or rors[0] <= 0.0
            or rors[-1] >= 1.0
        ):
            raise ValueError("rors_knots must be finite, increasing, and inside (0,1).")
        full_q = _validate_axis("full_q_knots", self.full_q_knots, 0.0, 1.0)
        partial_q = _validate_axis(
            "partial_q_knots", self.partial_q_knots, 0.0, 1.0
        )
        full = np.asarray(self.full_solution_grid, dtype=np.float64)
        partial = np.asarray(self.partial_solution_grid, dtype=np.float64)
        expected_full = (rors.size, full_q.size, 13)
        expected_partial = (rors.size, partial_q.size, 13)
        if full.shape != expected_full:
            raise ValueError(
                f"full_solution_grid shape {full.shape}, expected {expected_full}."
            )
        if partial.shape != expected_partial:
            raise ValueError(
                "partial_solution_grid shape "
                f"{partial.shape}, expected {expected_partial}."
            )
        if not np.all(np.isfinite(full)) or not np.all(np.isfinite(partial)):
            raise ValueError("Solution grids must contain only finite values.")
        c_bounds = tuple(float(x) for x in self.c_bounds)
        alpha_bounds = tuple(float(x) for x in self.alpha_bounds)
        if not (0.0 <= c_bounds[0] < c_bounds[1] <= 1.0):
            raise ValueError("c_bounds must lie within [0,1].")
        if not (0.0 < alpha_bounds[0] < alpha_bounds[1] <= 1.0):
            raise ValueError("alpha_bounds must lie within (0,1].")

        object.__setattr__(self, "rors_knots", jnp.asarray(rors, jnp.float64))
        object.__setattr__(self, "full_q_knots", jnp.asarray(full_q, jnp.float64))
        object.__setattr__(self, "partial_q_knots", jnp.asarray(partial_q, jnp.float64))
        object.__setattr__(self, "full_solution_grid", jnp.asarray(full, jnp.float64))
        object.__setattr__(
            self, "partial_solution_grid", jnp.asarray(partial, jnp.float64)
        )
        object.__setattr__(self, "c_bounds", c_bounds)
        object.__setattr__(self, "alpha_bounds", alpha_bounds)

        digest = hashlib.sha256()
        digest.update(b"power2-contact-basis-v1")
        digest.update(json.dumps(asdict(self.geometry), sort_keys=True).encode())
        digest.update(json.dumps([c_bounds, alpha_bounds]).encode())
        for array in (rors, full_q, partial_q, full, partial):
            _hash_array(digest, array)
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @property
    def estimated_bytes(self):
        return int(
            (self.full_solution_grid.size + self.partial_solution_grid.size) * 8
        )

    @staticmethod
    def _axis_interval(axis, value):
        index = jnp.searchsorted(axis, value, side="right") - 1
        index = jnp.clip(index, 0, axis.shape[0] - 2)
        low = axis[index]
        high = axis[index + 1]
        return index, (value - low) / (high - low)

    def _interpolate_solution(self, grid, q_axis, rors, q):
        ir, fr = self._axis_interval(self.rors_knots, rors)
        iq, fq = self._axis_interval(q_axis, q)
        s00 = grid[ir, iq]
        s01 = grid[ir, iq + 1]
        s10 = grid[ir + 1, iq]
        s11 = grid[ir + 1, iq + 1]
        r0 = s00 + fq[..., None] * (s01 - s00)
        r1 = s10 + fq[..., None] * (s11 - s10)
        return r0 + fr * (r1 - r0)

    @staticmethod
    def _power2_green(c_value, alpha):
        mus = jnp.asarray(_POWER2_MUS, dtype=jnp.float64)
        projection = jnp.asarray(_POWER2_PROJECTION, dtype=jnp.float64)
        profile = get_I_power2(c_value, alpha, mus)
        u = projection @ (1.0 - profile)
        green = greens_basis_transform(u)
        return green / (jnp.pi * (green[0] + green[1] / 1.5))

    def solution_from_separation(self, rors, separation):
        """Interpolate solution vectors and return ``(solution, in_domain)``."""
        rors = jnp.asarray(rors, dtype=jnp.float64)
        separation = jnp.asarray(separation, dtype=jnp.float64)
        r_valid = (
            jnp.isfinite(rors)
            & (rors >= self.rors_knots[0])
            & (rors <= self.rors_knots[-1])
        )
        z_valid = jnp.isfinite(separation) & (separation >= 0.0)
        inner = 1.0 - rors
        outer = 1.0 + rors
        is_full = separation <= inner
        is_partial = (separation > inner) & (separation < outer)
        full_q = jnp.clip(separation / inner, 0.0, 1.0)
        partial_q = jnp.clip((separation - inner) / (2.0 * rors), 0.0, 1.0)
        full_solution = self._interpolate_solution(
            self.full_solution_grid, self.full_q_knots, rors, full_q
        )
        partial_solution = self._interpolate_solution(
            self.partial_solution_grid,
            self.partial_q_knots,
            rors,
            partial_q,
        )
        outside_solution = jnp.asarray(
            [jnp.pi, 2.0 * jnp.pi / 3.0] + [0.0] * 11,
            dtype=jnp.float64,
        )
        solution = jnp.where(
            is_full[..., None],
            full_solution,
            jnp.where(
                is_partial[..., None], partial_solution, outside_solution
            ),
        )
        valid = r_valid & z_valid
        return jnp.where(valid[..., None], solution, jnp.nan), valid

    def signal_from_separation(self, rors, c_value, alpha, separation):
        solution, valid_geometry = self.solution_from_separation(rors, separation)
        green = self._power2_green(c_value, alpha)
        c_valid = (
            jnp.isfinite(c_value)
            & (c_value >= self.c_bounds[0])
            & (c_value <= self.c_bounds[1])
        )
        alpha_valid = (
            jnp.isfinite(alpha)
            & (alpha >= self.alpha_bounds[0])
            & (alpha <= self.alpha_bounds[1])
        )
        valid = valid_geometry & c_valid & alpha_valid
        signal = solution @ green - 1.0
        return jnp.where(valid, signal, jnp.nan), valid

    def evaluate_with_status(self, theta, times):
        """Evaluate a fixed-duration single-planet signal at arbitrary cadences."""
        theta = jnp.asarray(theta, dtype=jnp.float64)
        rors, c_value, alpha = theta[0], theta[1], theta[2]
        times = jnp.asarray(times, dtype=jnp.float64)
        period = jnp.asarray(self.geometry.period, dtype=jnp.float64)
        half_period = 0.5 * period
        dt = jnp.mod(
            times - (jnp.asarray(self.geometry.t0) - half_period), period
        ) - half_period
        speed = 2.0 * jnp.sqrt(
            jnp.maximum(0.0, (1.0 + rors) ** 2 - self.geometry.b**2)
        ) / self.geometry.duration
        separation = jnp.sqrt((speed * dt) ** 2 + self.geometry.b**2)
        signal, valid = self.signal_from_separation(
            rors, c_value, alpha, separation
        )
        duration_mask = jnp.abs(dt) < 0.5 * self.geometry.duration
        signal = jnp.where(duration_mask, signal, 0.0)
        return signal, jnp.all(valid)


def build_power2_contact_basis_grid(
    geometry: FixedPower2TransitGeometry,
    *,
    n_rors: int = 513,
    n_full: int = 257,
    n_partial: int = 257,
    rors_bounds=RORS_PRIOR_BOUNDS,
    build_batch_size: int = 1024,
    max_grid_bytes: int = 2 * 1024**3,
) -> Power2ContactBasisGrid:
    """Build exact degree-12 solution-vector tables in bounded batches."""
    _require_x64()
    rors, full_q, partial_q = make_contact_basis_axes(
        n_rors=n_rors,
        n_full=n_full,
        n_partial=n_partial,
        rors_bounds=rors_bounds,
    )
    estimated = int(rors.size * (full_q.size + partial_q.size) * 13 * 8)
    if estimated > max_grid_bytes:
        raise MemoryError(
            f"Contact-basis grid requires {estimated / 1024**3:.3f} GiB, "
            f"above the {max_grid_bytes / 1024**3:.3f} GiB limit."
        )
    solution = solution_vector(12, order=10)

    def evaluate_pairs(r_values, q_values, region):
        rr, qq = np.meshgrid(r_values, q_values, indexing="ij")
        flat_r = rr.ravel()
        if region == "full":
            flat_z = (qq * (1.0 - rr)).ravel()
        else:
            flat_z = (1.0 - rr + 2.0 * rr * qq).ravel()
        pairs = np.column_stack([flat_z, flat_r])

        def one(pair):
            return solution(pair[0], pair[1])

        values = _evaluate_in_fixed_batches(one, pairs, build_batch_size)
        return values.reshape(rr.shape + (13,))

    full_grid = evaluate_pairs(rors, full_q, "full")
    partial_grid = evaluate_pairs(rors, partial_q, "partial")
    return Power2ContactBasisGrid(
        geometry=geometry,
        rors_knots=rors,
        full_q_knots=full_q,
        partial_q_knots=partial_q,
        full_solution_grid=full_grid,
        partial_solution_grid=partial_grid,
    )
