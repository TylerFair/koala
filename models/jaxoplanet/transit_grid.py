"""Contact-aware interpolation for duration-parameterized transit models."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from ..common import get_I_power2
from ..detrend import _prepare_power2_poly


_POWER2_MUS, _POWER2_PROJECTION = _prepare_power2_poly()


def _chebyshev_lobatto_nodes(size: int):
    """Return increasing nodes in [0, 1], clustered at both endpoints."""
    if int(size) < 4:
        raise ValueError("Each transit-grid segment requires at least four nodes.")
    index = np.arange(int(size), dtype=np.float64)
    return jnp.asarray(0.5 * (1.0 - np.cos(np.pi * index / (size - 1))))


def _chebyshev_coordinate(value):
    """Map a physical segment fraction to its uniform Lobatto coordinate."""
    epsilon = 16.0 * jnp.finfo(jnp.float64).eps
    value = jnp.clip(
        jnp.asarray(value, dtype=jnp.float64), epsilon, 1.0 - epsilon
    )
    return jnp.arccos(jnp.clip(1.0 - 2.0 * value, -1.0, 1.0)) / jnp.pi


def _local_quintic_metadata(coordinates, segment_sizes, segment_offsets):
    """Return six-point stencil indices and Lagrange weights."""
    coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
    segment_sizes = jnp.asarray(segment_sizes, dtype=jnp.int32)
    segment_offsets = jnp.asarray(segment_offsets, dtype=jnp.int32)
    stencil_size = 6
    scaled = coordinates * (segment_sizes - 1)
    bracket = jnp.floor(scaled).astype(jnp.int32)
    start = jnp.clip(
        bracket - 2, 0, segment_sizes - stencil_size
    )
    stencil_offsets = jnp.arange(stencil_size, dtype=jnp.int32)
    indices = start[..., None] + segment_offsets[..., None] + stencil_offsets

    # Writing the Lagrange basis polynomials explicitly avoids the 0/0
    # form of barycentric interpolation when a cadence lands on a grid node.
    weights = []
    for column in range(stencil_size):
        weight = jnp.ones_like(coordinates, dtype=jnp.float64)
        for other in range(stencil_size):
            if other != column:
                weight = weight * (
                    (scaled - (start + other)) / (column - other)
                )
        weights.append(weight)
    return indices, jnp.stack(weights, axis=-1)


def _local_quintic_uniform(values, coordinates, segment_sizes, segment_offsets):
    """Six-point local interpolation on one or more uniform node segments."""
    values = jnp.asarray(values, dtype=jnp.float64)
    if int(values.shape[0]) < 6:
        raise ValueError("Quintic interpolation requires at least six nodes.")
    indices, weights = _local_quintic_metadata(
        coordinates, segment_sizes, segment_offsets
    )
    y = values[indices]
    result = jnp.zeros_like(coordinates, dtype=jnp.float64)
    for column in range(6):
        result = result + weights[..., column] * y[..., column]
    return result


def build_duration_transit_grid(*, impact, radius_ratio, num_nodes: int):
    """Return the contact-split separation grid used by the interpolant."""
    num_nodes = int(num_nodes)
    if num_nodes < 13:
        raise ValueError("transit_grid_nodes must be at least 13.")

    inner_size = num_nodes // 2
    outer_size = num_nodes - inner_size
    inner_nodes = _chebyshev_lobatto_nodes(inner_size)
    outer_nodes = _chebyshev_lobatto_nodes(outer_size)
    impact = jnp.abs(jnp.asarray(impact, dtype=jnp.float64))
    radius_ratio = jnp.abs(jnp.asarray(radius_ratio, dtype=jnp.float64))
    outer_contact = 1.0 + radius_ratio
    inner_contact = 1.0 - radius_ratio
    epsilon = 4.0 * jnp.finfo(jnp.float64).eps
    inner_edge = inner_contact - epsilon * jnp.maximum(1.0, inner_contact)
    inner_separation = impact + (inner_edge - impact) * inner_nodes
    outer_separation = inner_contact + (
        outer_contact - inner_contact
    ) * outer_nodes
    return jnp.concatenate((inner_separation, outer_separation), axis=0)


def _duration_interpolation_metadata(
    phase_offsets,
    phase_mask,
    duration,
    impact,
    radius_ratio,
    num_nodes,
):
    """Return stencil metadata and the physical in-transit mask."""
    inner_size = int(num_nodes) // 2
    outer_size = int(num_nodes) - inner_size
    outer_contact = 1.0 + radius_ratio
    inner_contact = 1.0 - radius_ratio
    normalized_phase = jnp.clip(
        2.0 * jnp.abs(phase_offsets) / duration, 0.0, 1.0
    )
    speed_scale = jnp.maximum(
        0.0, jnp.square(outer_contact) - jnp.square(impact)
    )
    separation = jnp.sqrt(
        jnp.square(impact) + jnp.square(normalized_phase) * speed_scale
    )
    inner_coordinate = jnp.clip(
        (separation - impact) / (inner_contact - impact), 0.0, 1.0
    )
    outer_coordinate = jnp.clip(
        (separation - inner_contact) / (2.0 * radius_ratio), 0.0, 1.0
    )
    use_inner = separation < inner_contact
    segment_coordinate = jnp.where(
        use_inner, inner_coordinate, outer_coordinate
    )
    segment_size = jnp.where(use_inner, inner_size, outer_size)
    segment_offset = jnp.where(use_inner, 0, inner_size)
    indices, weights = _local_quintic_metadata(
        _chebyshev_coordinate(segment_coordinate),
        segment_size,
        segment_offset,
    )
    physical_mask = phase_mask & (separation < outer_contact)
    return indices, weights, physical_mask


def interpolate_duration_transit_grid(
    flux_grid,
    phase_offsets,
    phase_mask,
    *,
    duration,
    impact,
    radius_ratio,
):
    """Interpolate already-evaluated grid fluxes onto original cadences."""
    flux_grid = jnp.asarray(flux_grid, dtype=jnp.float64)
    if int(flux_grid.shape[0]) < 13:
        raise ValueError("transit_grid_nodes must be at least 13.")
    num_nodes = int(flux_grid.shape[0])
    impact = jnp.abs(jnp.asarray(impact, dtype=jnp.float64))
    radius_ratio = jnp.abs(jnp.asarray(radius_ratio, dtype=jnp.float64))
    duration = jnp.asarray(duration, dtype=jnp.float64)
    phase_offsets = jnp.asarray(phase_offsets, dtype=jnp.float64)
    phase_mask = jnp.asarray(phase_mask, dtype=bool)
    indices, weights, physical_mask = _duration_interpolation_metadata(
        phase_offsets,
        phase_mask,
        duration,
        impact,
        radius_ratio,
        num_nodes,
    )
    selected = flux_grid[indices]
    flux = jnp.zeros_like(phase_offsets, dtype=jnp.float64)
    for column in range(6):
        flux = flux + weights[..., column] * selected[..., column]
    return jnp.where(physical_mask, flux, 0.0)


def _duration_transit_from_radius_u(
    light_curve_kernel,
    radius,
    u_reference,
    phase_offsets,
    phase_mask,
    *,
    duration,
    impact,
    num_nodes: int,
    order: int = 10,
):
    phase_offsets = jnp.asarray(phase_offsets, dtype=jnp.float64)
    phase_mask = jnp.asarray(phase_mask, dtype=bool)
    duration = jnp.asarray(duration, dtype=jnp.float64)
    impact = jnp.asarray(impact, dtype=jnp.float64)
    separation_grid = build_duration_transit_grid(
        impact=impact,
        radius_ratio=radius,
        num_nodes=num_nodes,
    )
    flux_grid = light_curve_kernel(
        u_reference, separation_grid, radius, order=order
    ).at[-1].set(0.0)
    return interpolate_duration_transit_grid(
        flux_grid,
        phase_offsets,
        phase_mask,
        duration=duration,
        impact=impact,
        radius_ratio=radius,
    )


def _power2_duration_transit_primal(
    light_curve_kernel,
    theta,
    u_reference,
    phase_offsets,
    phase_mask,
    *,
    duration,
    impact,
    num_nodes: int,
    order: int = 10,
):
    """Ordinary reverse-mode reference for the power-2 grid transit."""
    radius, coefficient, exponent = theta
    mus, projection = _POWER2_MUS, _POWER2_PROJECTION
    profiles = get_I_power2(coefficient, exponent, mus)
    computed_u = projection @ (1.0 - profiles)
    u = (
        jax.lax.stop_gradient(jnp.asarray(u_reference, dtype=jnp.float64))
        + computed_u
        - jax.lax.stop_gradient(computed_u)
    )
    return _duration_transit_from_radius_u(
        light_curve_kernel,
        radius,
        u,
        phase_offsets,
        phase_mask,
        duration=duration,
        impact=impact,
        num_nodes=num_nodes,
        order=order,
    )


def _power2_duration_transit_sensitivities(
    light_curve_kernel,
    theta,
    u_reference,
    phase_offsets,
    phase_mask,
    *,
    duration,
    impact,
    num_nodes: int,
    order: int = 10,
):
    """Return cadence derivatives with respect to radius, c, and alpha."""
    radius, coefficient, exponent = theta[:3]
    mus, projection = _POWER2_MUS, _POWER2_PROJECTION
    mu_power = mus**exponent
    log_mu = jnp.where(mus > 0.0, jnp.log(mus), 0.0)
    du_dc = projection @ (1.0 - mu_power)
    du_dalpha = projection @ (-coefficient * mu_power * log_mu)
    base = lambda trial_radius, trial_u: _duration_transit_from_radius_u(
        light_curve_kernel,
        trial_radius,
        trial_u,
        phase_offsets,
        phase_mask,
        duration=duration,
        impact=impact,
        num_nodes=num_nodes,
        order=order,
    )
    _, radius_tangent = jax.jvp(
        lambda value: base(value, u_reference),
        (radius,),
        (jnp.ones_like(radius),),
    )
    _, c_tangent = jax.jvp(
        lambda value: base(radius, value),
        (u_reference,),
        (du_dc,),
    )
    _, alpha_tangent = jax.jvp(
        lambda value: base(radius, value),
        (u_reference,),
        (du_dalpha,),
    )
    return jnp.stack((radius_tangent, c_tangent, alpha_tangent), axis=1)


def interpolate_power2_duration_transit(
    light_curve_kernel,
    c1,
    c2,
    u_reference,
    phase_offsets,
    phase_mask,
    *,
    duration,
    impact,
    radius_ratio,
    num_nodes: int,
    order: int = 10,
):
    """Evaluate a power-2 grid transit with forward-mode sensitivities.

    The custom reverse rule recomputes the cadence-flux Jacobian with respect
    to only ``(radius_ratio, c1, c2)`` using three forward JVPs. Its backward
    pass immediately contracts those columns in a dense cadence reduction and
    never transposes the colliding interpolation gather. The forward Jacobian
    includes both grid-flux and interpolation-weight dependence on radius.
    """

    def primal(theta, reference_u, phases, masks, transit_duration, b):
        return _power2_duration_transit_primal(
            light_curve_kernel,
            theta,
            reference_u,
            phases,
            masks,
            duration=transit_duration,
            impact=b,
            num_nodes=num_nodes,
            order=order,
        )

    @jax.custom_vjp
    def evaluate(theta, reference_u, phases, masks, transit_duration, b):
        return primal(theta, reference_u, phases, masks, transit_duration, b)

    def evaluate_fwd(
        theta, reference_u, phases, masks, transit_duration, b
    ):
        radius = theta[0]
        flux = _duration_transit_from_radius_u(
            light_curve_kernel,
            jax.lax.optimization_barrier(radius),
            jax.lax.optimization_barrier(reference_u),
            phases,
            masks,
            duration=transit_duration,
            impact=b,
            num_nodes=num_nodes,
            order=order,
        )
        flux = jax.lax.optimization_barrier(flux)
        return flux, (
            theta, reference_u, phases, masks, transit_duration, b
        )

    def evaluate_bwd(residual, cotangent):
        theta, reference_u, phases, masks, transit_duration, b = residual
        jacobian = _power2_duration_transit_sensitivities(
            light_curve_kernel,
            theta,
            reference_u,
            phases,
            masks,
            duration=transit_duration,
            impact=b,
            num_nodes=num_nodes,
            order=order,
        )
        return (
            jnp.einsum("ip,i->p", jacobian, cotangent),
            None,
            None,
            None,
            None,
            None,
        )

    evaluate.defvjp(evaluate_fwd, evaluate_bwd)
    theta = jnp.stack((radius_ratio, c1, c2))
    return evaluate(
        theta,
        jax.lax.stop_gradient(u_reference),
        phase_offsets,
        phase_mask,
        duration,
        impact,
    )


def power2_grid_reduced_log_likelihood(
    light_curve_kernel,
    theta,
    u_reference,
    phase_offsets,
    phase_mask,
    time_basis,
    exp_basis,
    observed,
    reported_error,
    active_mask,
    oot_reference_beta,
    oot_group_yerr,
    oot_group_count,
    oot_group_reference_sse,
    oot_group_x_reference_residual,
    oot_group_xx,
    *,
    duration,
    impact,
    num_nodes: int,
    order: int = 10,
):
    """Return active/OOT log likelihoods with a fused analytic reverse rule.

    ``theta`` is ``(radius, c, alpha, trend_c, trend_v, trend_A, jitter)``.
    The forward result is algebraically identical to the separate Normal and
    grouped-statistics likelihoods. Backward recomputes the three transit JVP
    columns and contracts them with the residual immediately, avoiding both a
    gather transpose and a materialized cadence-Jacobian residual.
    """
    u_reference = jax.lax.stop_gradient(
        jnp.asarray(u_reference, dtype=jnp.float64)
    )
    phase_offsets = jnp.asarray(phase_offsets, dtype=jnp.float64)
    phase_mask = jnp.asarray(phase_mask, dtype=bool)
    time_basis = jnp.asarray(time_basis, dtype=jnp.float64)
    exp_basis = jnp.asarray(exp_basis, dtype=jnp.float64)
    observed = jnp.asarray(observed, dtype=jnp.float64)
    reported_error = jnp.asarray(reported_error, dtype=jnp.float64)
    active_mask = jnp.asarray(active_mask, dtype=bool)
    duration = jnp.asarray(duration, dtype=jnp.float64)
    impact = jnp.asarray(impact, dtype=jnp.float64)

    def values(
        parameters,
        reference_u,
        phases,
        transit_mask,
        linear_time,
        exponential,
        y,
        yerr,
        likelihood_mask,
        reference_beta,
        group_yerr,
        group_count,
        reference_sse,
        group_xr,
        group_xx,
        transit_duration,
        b,
    ):
        radius = parameters[0]
        beta = parameters[3:6]
        jitter = parameters[6]
        transit = _duration_transit_from_radius_u(
            light_curve_kernel,
            radius,
            reference_u,
            phases,
            transit_mask,
            duration=transit_duration,
            impact=b,
            num_nodes=num_nodes,
            order=order,
        )
        model = (
            transit
            + beta[0]
            + beta[1] * linear_time
            + beta[2] * exponential
        )
        error = jnp.sqrt(yerr**2 + jitter**2)
        active_log_prob = dist.Normal(model, error).log_prob(y)
        active_log_prob = jnp.where(
            likelihood_mask, active_log_prob, 0.0
        )
        delta = beta - reference_beta
        group_sse = (
            reference_sse
            - 2.0 * jnp.einsum("p,gp->g", delta, group_xr)
            + jnp.einsum("p,gpq,q->g", delta, group_xx, delta)
        )
        group_variance = group_yerr**2 + jitter**2
        group_log_prob = -0.5 * (
            group_sse / group_variance
            + group_count * jnp.log(2.0 * jnp.pi * group_variance)
        )
        return jnp.stack((jnp.sum(active_log_prob), jnp.sum(group_log_prob)))

    @jax.custom_vjp
    def evaluate(*arguments):
        return values(*arguments)

    def evaluate_fwd(*arguments):
        result = jax.lax.optimization_barrier(values(*arguments))
        return result, arguments

    def evaluate_bwd(residual, cotangent):
        (
            parameters,
            reference_u,
            phases,
            transit_mask,
            linear_time,
            exponential,
            y,
            yerr,
            likelihood_mask,
            reference_beta,
            group_yerr,
            group_count,
            reference_sse,
            group_xr,
            group_xx,
            transit_duration,
            b,
        ) = residual
        radius = parameters[0]
        beta = parameters[3:6]
        jitter = parameters[6]
        transit = _duration_transit_from_radius_u(
            light_curve_kernel,
            radius,
            reference_u,
            phases,
            transit_mask,
            duration=transit_duration,
            impact=b,
            num_nodes=num_nodes,
            order=order,
        )
        transit_jacobian = _power2_duration_transit_sensitivities(
            light_curve_kernel,
            parameters,
            reference_u,
            phases,
            transit_mask,
            duration=transit_duration,
            impact=b,
            num_nodes=num_nodes,
            order=order,
        )
        model = (
            transit
            + beta[0]
            + beta[1] * linear_time
            + beta[2] * exponential
        )
        residual_flux = y - model
        variance = yerr**2 + jitter**2
        model_cotangent = jnp.where(
            likelihood_mask, residual_flux / variance, 0.0
        )
        active_gradient = jnp.zeros((7,), dtype=jnp.float64)
        active_gradient = active_gradient.at[:3].set(
            jnp.einsum("ip,i->p", transit_jacobian, model_cotangent)
        )
        active_gradient = active_gradient.at[3].set(
            jnp.sum(model_cotangent)
        )
        active_gradient = active_gradient.at[4].set(
            jnp.sum(model_cotangent * linear_time)
        )
        active_gradient = active_gradient.at[5].set(
            jnp.sum(model_cotangent * exponential)
        )
        active_jitter = jnp.where(
            likelihood_mask,
            jitter * (
                residual_flux**2 / variance**2 - 1.0 / variance
            ),
            0.0,
        )
        active_gradient = active_gradient.at[6].set(jnp.sum(active_jitter))

        delta = beta - reference_beta
        group_sse = (
            reference_sse
            - 2.0 * jnp.einsum("p,gp->g", delta, group_xr)
            + jnp.einsum("p,gpq,q->g", delta, group_xx, delta)
        )
        group_variance = group_yerr**2 + jitter**2
        group_sse_gradient = (
            -2.0 * group_xr
            + 2.0 * jnp.einsum("gpq,q->gp", group_xx, delta)
        )
        oot_gradient = jnp.zeros((7,), dtype=jnp.float64)
        oot_gradient = oot_gradient.at[3:6].set(
            jnp.sum(
                -0.5 * group_sse_gradient / group_variance[:, None],
                axis=0,
            )
        )
        oot_gradient = oot_gradient.at[6].set(
            jnp.sum(
                jitter
                * (
                    group_sse / group_variance**2
                    - group_count / group_variance
                )
            )
        )
        parameter_cotangent = (
            cotangent[0] * active_gradient
            + cotangent[1] * oot_gradient
        )
        return (parameter_cotangent,) + (None,) * 16

    evaluate.defvjp(evaluate_fwd, evaluate_bwd)
    return evaluate(
        theta,
        u_reference,
        phase_offsets,
        phase_mask,
        time_basis,
        exp_basis,
        observed,
        reported_error,
        active_mask,
        jnp.asarray(oot_reference_beta, dtype=jnp.float64),
        jnp.asarray(oot_group_yerr, dtype=jnp.float64),
        jnp.asarray(oot_group_count, dtype=jnp.float64),
        jnp.asarray(oot_group_reference_sse, dtype=jnp.float64),
        jnp.asarray(oot_group_x_reference_residual, dtype=jnp.float64),
        jnp.asarray(oot_group_xx, dtype=jnp.float64),
        duration,
        impact,
    )


def interpolate_duration_transit(
    light_curve_kernel,
    u,
    phase_offsets,
    phase_mask,
    *,
    duration,
    impact,
    radius_ratio,
    num_nodes: int,
    order: int = 10,
):
    """Interpolate a transit on smooth regions separated at contact points.

    The duration geometry makes first/fourth contact occur at normalized phase
    one for every radius ratio. The caller guarantees that the full radius
    prior is non-grazing. The grid is split at second/third contact and each
    segment uses Chebyshev--Lobatto nodes. A local quintic interpolant is used
    within each smooth segment, so no stencil crosses a contact point.
    """
    impact = jnp.abs(jnp.asarray(impact, dtype=jnp.float64))
    radius_ratio = jnp.abs(jnp.asarray(radius_ratio, dtype=jnp.float64))
    duration = jnp.asarray(duration, dtype=jnp.float64)
    phase_offsets = jnp.asarray(phase_offsets, dtype=jnp.float64)
    phase_mask = jnp.asarray(phase_mask, dtype=bool)

    separation_grid = build_duration_transit_grid(
        impact=impact,
        radius_ratio=radius_ratio,
        num_nodes=num_nodes,
    )
    flux_grid = light_curve_kernel(
        u, separation_grid, radius_ratio, order=order
    )
    # The physical transit signal is exactly zero at first/fourth contact.
    # Enforce that endpoint explicitly: the high-degree polynomial Green basis
    # can otherwise leave a cancellation residue for pathological prior-edge
    # coefficients, which would pollute the final interpolation stencil.
    flux_grid = flux_grid.at[-1].set(0.0)

    return interpolate_duration_transit_grid(
        flux_grid,
        phase_offsets,
        phase_mask,
        duration=duration,
        impact=impact,
        radius_ratio=radius_ratio,
    )


__all__ = [
    "build_duration_transit_grid",
    "interpolate_duration_transit",
    "interpolate_duration_transit_grid",
    "interpolate_power2_duration_transit",
    "power2_grid_reduced_log_likelihood",
]
