"""Radius-Hermite follow-up for the experimental power-2 transit grid.

Only radius ratio uses cubic Hermite interpolation.  The exact derivative
``d flux / d rors`` is stored at every existing grid knot; power-2 ``c`` and
``alpha`` remain bilinear.  This doubles artifact memory, but makes the radius
derivative continuous at radius knots and directly targets the dominant error
seen in the multilinear prototype.

This remains an evaluation-only prototype and is not serializable or connected
to the fitting model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .experimental_power2_grid import (
    FixedPower2TransitGeometry,
    Power2TransitGrid,
    _evaluate_in_fixed_batches,
    _hash_array,
    build_power2_transit_grid,
    make_exact_power2_evaluator,
)


@dataclass(frozen=True)
class Power2RadiusHermiteGrid:
    """Power-2 grid with exact radius derivatives at all tensor knots."""

    base_grid: Power2TransitGrid
    rors_derivative_grid: jax.Array
    fingerprint: str = field(init=False)

    def __post_init__(self):
        derivative = np.asarray(self.rors_derivative_grid, dtype=np.float64)
        if derivative.shape != tuple(self.base_grid.flux_grid.shape):
            raise ValueError(
                "rors_derivative_grid must match flux_grid shape; received "
                f"{derivative.shape} and {self.base_grid.flux_grid.shape}."
            )
        if not np.all(np.isfinite(derivative)):
            raise ValueError("rors_derivative_grid must contain only finite values.")
        object.__setattr__(
            self,
            "rors_derivative_grid",
            jnp.asarray(derivative, dtype=jnp.float64),
        )
        digest = hashlib.sha256()
        digest.update(b"power2-radius-hermite-grid-v1")
        digest.update(self.base_grid.fingerprint.encode("ascii"))
        _hash_array(digest, derivative)
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @property
    def times(self):
        return self.base_grid.times

    @property
    def geometry(self):
        return self.base_grid.geometry

    @property
    def rors_knots(self):
        return self.base_grid.rors_knots

    @property
    def c_knots(self):
        return self.base_grid.c_knots

    @property
    def alpha_knots(self):
        return self.base_grid.alpha_knots

    @property
    def flux_grid(self):
        return self.base_grid.flux_grid

    @property
    def exact_kernel(self):
        return self.base_grid.exact_kernel

    @property
    def exact_implementation_fingerprint(self):
        return self.base_grid.exact_implementation_fingerprint

    @property
    def estimated_bytes(self) -> int:
        return 2 * self.base_grid.estimated_bytes

    @staticmethod
    def _hermite(y0, y1, derivative0, derivative1, fraction, spacing):
        s = fraction
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        return (
            h00 * y0
            + h10 * spacing * derivative0
            + h01 * y1
            + h11 * spacing * derivative1
        )

    def _interpolate_raw(self, rors, c, alpha):
        interval = self.base_grid._axis_interval
        ir, fr, valid_r = interval(self.rors_knots, rors)
        ic, fc, valid_c = interval(self.c_knots, c)
        ia, fa, valid_a = interval(self.alpha_knots, alpha)
        spacing = self.rors_knots[ir + 1] - self.rors_knots[ir]
        flux = self.flux_grid
        derivative = self.rors_derivative_grid

        def radius_curve(c_index, alpha_index):
            return self._hermite(
                flux[ir, c_index, alpha_index],
                flux[ir + 1, c_index, alpha_index],
                derivative[ir, c_index, alpha_index],
                derivative[ir + 1, c_index, alpha_index],
                fr,
                spacing,
            )

        r00 = radius_curve(ic, ia)
        r01 = radius_curve(ic, ia + 1)
        r10 = radius_curve(ic + 1, ia)
        r11 = radius_curve(ic + 1, ia + 1)
        c0 = r00 + fa * (r01 - r00)
        c1 = r10 + fa * (r11 - r10)
        result = c0 + fc * (c1 - c0)
        return result, valid_r & valid_c & valid_a

    def evaluate_with_status(self, rors, c, alpha):
        flux, valid = self._interpolate_raw(rors, c, alpha)
        return jnp.where(valid, flux, jnp.nan), valid


@dataclass(frozen=True)
class Power2RadiusCubicGrid:
    """C1 radius interpolation using finite-difference knot slopes.

    Unlike :class:`Power2RadiusHermiteGrid`, this stores no derivative tensor.
    It gathers up to four radius planes and uses centered secants as the knot
    slopes (one-sided at domain boundaries).  This is the modest-memory smooth
    cubic candidate.
    """

    base_grid: Power2TransitGrid
    fingerprint: str = field(init=False)

    def __post_init__(self):
        digest = hashlib.sha256()
        digest.update(b"power2-radius-finite-difference-cubic-v1")
        digest.update(self.base_grid.fingerprint.encode("ascii"))
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @property
    def times(self):
        return self.base_grid.times

    @property
    def geometry(self):
        return self.base_grid.geometry

    @property
    def rors_knots(self):
        return self.base_grid.rors_knots

    @property
    def c_knots(self):
        return self.base_grid.c_knots

    @property
    def alpha_knots(self):
        return self.base_grid.alpha_knots

    @property
    def flux_grid(self):
        return self.base_grid.flux_grid

    @property
    def exact_kernel(self):
        return self.base_grid.exact_kernel

    @property
    def exact_implementation_fingerprint(self):
        return self.base_grid.exact_implementation_fingerprint

    @property
    def estimated_bytes(self) -> int:
        return self.base_grid.estimated_bytes

    def _knot_slope(self, knot_index, c_index, alpha_index):
        last = self.rors_knots.shape[0] - 1
        left = jnp.maximum(knot_index - 1, 0)
        right = jnp.minimum(knot_index + 1, last)
        numerator = (
            self.flux_grid[right, c_index, alpha_index]
            - self.flux_grid[left, c_index, alpha_index]
        )
        denominator = self.rors_knots[right] - self.rors_knots[left]
        return numerator / denominator

    def _interpolate_raw(self, rors, c, alpha):
        interval = self.base_grid._axis_interval
        ir, fr, valid_r = interval(self.rors_knots, rors)
        ic, fc, valid_c = interval(self.c_knots, c)
        ia, fa, valid_a = interval(self.alpha_knots, alpha)
        spacing = self.rors_knots[ir + 1] - self.rors_knots[ir]

        def radius_curve(c_index, alpha_index):
            return Power2RadiusHermiteGrid._hermite(
                self.flux_grid[ir, c_index, alpha_index],
                self.flux_grid[ir + 1, c_index, alpha_index],
                self._knot_slope(ir, c_index, alpha_index),
                self._knot_slope(ir + 1, c_index, alpha_index),
                fr,
                spacing,
            )

        r00 = radius_curve(ic, ia)
        r01 = radius_curve(ic, ia + 1)
        r10 = radius_curve(ic + 1, ia)
        r11 = radius_curve(ic + 1, ia + 1)
        c0 = r00 + fa * (r01 - r00)
        c1 = r10 + fa * (r11 - r10)
        result = c0 + fc * (c1 - c0)
        return result, valid_r & valid_c & valid_a

    def evaluate_with_status(self, rors, c, alpha):
        flux, valid = self._interpolate_raw(rors, c, alpha)
        return jnp.where(valid, flux, jnp.nan), valid


def augment_power2_grid_with_radius_derivatives(
    base_grid: Power2TransitGrid,
    *,
    derivative_batch_size: int = 64,
    max_grid_bytes: int = 2 * 1024**3,
) -> Power2RadiusHermiteGrid:
    """Evaluate exact radius JVPs at all knots and attach them to ``base_grid``."""
    if derivative_batch_size <= 0:
        raise ValueError("derivative_batch_size must be positive.")
    required_bytes = 2 * base_grid.estimated_bytes
    if required_bytes > max_grid_bytes:
        raise MemoryError(
            "Refusing radius-Hermite grid requiring approximately "
            f"{required_bytes / 1024**3:.3f} GiB; max_grid_bytes permits "
            f"{max_grid_bytes / 1024**3:.3f} GiB."
        )

    rr, cc, aa = np.meshgrid(
        np.asarray(base_grid.rors_knots),
        np.asarray(base_grid.c_knots),
        np.asarray(base_grid.alpha_knots),
        indexing="ij",
    )
    theta = np.stack([rr.ravel(), cc.ravel(), aa.ravel()], axis=-1)
    exact = make_exact_power2_evaluator(
        np.asarray(base_grid.times),
        base_grid.geometry,
        kernel=base_grid.exact_kernel,
    )
    radius_tangent = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)

    def exact_radius_derivative(one_theta):
        return jax.jvp(exact, (one_theta,), (radius_tangent,))[1]

    derivatives = _evaluate_in_fixed_batches(
        exact_radius_derivative, theta, derivative_batch_size
    )
    derivatives = derivatives.reshape(base_grid.flux_grid.shape)
    return Power2RadiusHermiteGrid(
        base_grid=base_grid,
        rors_derivative_grid=derivatives,
    )


def build_power2_radius_hermite_grid(
    times: Sequence[float],
    geometry: FixedPower2TransitGeometry,
    rors_knots: Sequence[float],
    c_knots: Sequence[float],
    alpha_knots: Sequence[float],
    *,
    exact_kernel: str = "stock",
    build_batch_size: int = 128,
    derivative_batch_size: int = 64,
    max_grid_bytes: int = 2 * 1024**3,
) -> Power2RadiusHermiteGrid:
    """Build exact flux and radius-derivative tensors in bounded batches."""
    if max_grid_bytes < 2:
        raise MemoryError("max_grid_bytes is too small for a Hermite grid.")
    base = build_power2_transit_grid(
        times,
        geometry,
        rors_knots,
        c_knots,
        alpha_knots,
        exact_kernel=exact_kernel,
        build_batch_size=build_batch_size,
        max_grid_bytes=max_grid_bytes // 2,
    )
    return augment_power2_grid_with_radius_derivatives(
        base,
        derivative_batch_size=derivative_batch_size,
        max_grid_bytes=max_grid_bytes,
    )


def build_power2_radius_cubic_grid(
    times: Sequence[float],
    geometry: FixedPower2TransitGeometry,
    rors_knots: Sequence[float],
    c_knots: Sequence[float],
    alpha_knots: Sequence[float],
    *,
    exact_kernel: str = "stock",
    build_batch_size: int = 128,
    max_grid_bytes: int = 2 * 1024**3,
) -> Power2RadiusCubicGrid:
    """Build the no-extra-storage finite-difference cubic candidate."""
    base = build_power2_transit_grid(
        times,
        geometry,
        rors_knots,
        c_knots,
        alpha_knots,
        exact_kernel=exact_kernel,
        build_batch_size=build_batch_size,
        max_grid_bytes=max_grid_bytes,
    )
    return Power2RadiusCubicGrid(base)
