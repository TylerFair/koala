"""Validated-domain Chebyshev acceleration of direct power-2 disk integrals.

Only fixed duration geometry can use this table. Outside its radius/exponent
domain the direct kernel remains the target; the table never truncates priors.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .power2_exact import (
    light_curve, occulted_power2_moment, uniform_overlap, _all_lanes_in_domain,
)
from .transit_grid import build_duration_transit_grid, interpolate_duration_transit_grid



def _chebyshev_basis(x, degree):
    values = [jnp.ones_like(x)]
    if degree:
        values.append(x)
    for _ in range(2, degree + 1):
        values.append(2.0 * x * values[-1] - values[-2])
    return jnp.stack(values)


@lru_cache(maxsize=8)
def prepare_table(*, impact, radius_bounds, exponent_bounds=(0., 1.), num_nodes=769,
                  radius_degree=12, exponent_degree=10, order=128):
    """Build a host-serializable table; bounds must stay below grazing contact."""
    impact = abs(float(impact))
    rlo, rhi = map(float, radius_bounds)
    alo, ahi = map(float, exponent_bounds)
    if not (0 < rlo < rhi < 1.0-impact and 0 <= alo < ahi):
        raise ValueError("Power-2 tables require positive radii below grazing and nonnegative exponents.")
    nr, na = radius_degree+1, exponent_degree+1
    xr = np.cos(np.pi*(np.arange(nr)+.5)/nr)
    xa = np.cos(np.pi*(np.arange(na)+.5)/na)
    radii = (rhi+rlo)/2 + (rhi-rlo)/2*xr
    exponents = jnp.asarray((ahi+alo)/2 + (ahi-alo)/2*xa)
    @jax.jit
    def evaluate(r):
        z = build_duration_transit_grid(impact=impact, radius_ratio=r, num_nodes=num_nodes)
        return occulted_power2_moment(z[None, :], r, exponents[:, None], order=order)
    # Bound peak memory: evaluate one radius slab at a time.
    values = np.stack([np.asarray(evaluate(r)) for r in radii])
    tr = 2./nr * np.cos(np.outer(np.arange(nr), np.arccos(xr)))
    ta = 2./na * np.cos(np.outer(np.arange(na), np.arccos(xa)))
    tr[0] *= .5
    ta[0] *= .5
    coefficients = np.einsum('ir,ja,ran->ijn', tr, ta, values, optimize=True)
    @jax.jit
    def evaluate_area(r):
        z = build_duration_transit_grid(impact=impact, radius_ratio=r, num_nodes=num_nodes)
        return uniform_overlap(z, r)
    area_values = np.stack([np.asarray(evaluate_area(r)) for r in radii])
    result = dict(coefficients=coefficients, area_coefficients=tr @ area_values,
                  radius_bounds=np.array([rlo, rhi]), exponent_bounds=np.array([alo, ahi]),
                  impact=impact, order=int(order))
    # The node curves share a much smaller time basis than the tensor degree
    # suggests. Compress it with a uniform coefficient-error bound, then run
    # the same held-out flux/gradient validation on the compressed evaluator.
    matrix = coefficients.reshape(nr*na, num_nodes)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    for rank in range(4, matrix.shape[0]+4, 4):
        rank = min(rank, matrix.shape[0])
        weighted_left = left[:, :rank]*singular[:rank]
        remainder = matrix-weighted_left@right[:rank]
        if np.max(np.sum(np.abs(remainder),axis=0)) <= 5e-14:
            if rank*(nr*na+num_nodes) < matrix.size:
                result['moment_left'] = weighted_left.reshape(nr,na,rank)
                result['moment_right'] = right[:rank]
            break
    # Verify held-out values AND derivatives before a table enters inference.
    # Include both bounds; coefficient extrema cover the linear numerator.
    parameters = np.array([(r,c,a) for r in np.linspace(rlo,rhi,5)
                           for a in (alo, (alo+ahi)/2, ahi) for c in (0.,1.)])
    candidate = lambda p: table_node_flux(result,*p)
    def reference(p):
        r,c,a = p
        z = build_duration_transit_grid(impact=impact,radius_ratio=r,num_nodes=num_nodes)
        return light_curve(c,a,z,r,order=order).at[-1].set(0.)
    evaluate_candidate = jax.jit(jax.vmap(lambda p:(candidate(p),jax.jacfwd(candidate)(p))))
    evaluate_reference = jax.jit(jax.vmap(lambda p:(reference(p),jax.jacfwd(reference)(p))))
    # Slab evaluation also bounds validation memory on small GPUs.
    value_error,gradient_error = 0.,0.
    for block in np.array_split(parameters,5):
        actual, actual_gradient = evaluate_candidate(block)
        expected, expected_gradient = evaluate_reference(block)
        value_error = max(value_error,float(jnp.max(jnp.abs(actual-expected))))
        gradient_error = max(gradient_error,float(jnp.max(jnp.abs(actual_gradient-expected_gradient))))
    if not (value_error <= 2e-10 and gradient_error <= 1e-7):
        if radius_degree < 24 or exponent_degree < 16:
            return prepare_table(impact=impact,radius_bounds=(rlo,rhi),
                exponent_bounds=(alo,ahi),num_nodes=num_nodes,radius_degree=24,
                exponent_degree=16,order=order)
        raise ValueError(f'Power-2 table failed validation: flux={value_error:g}; gradient={gradient_error:g}.')
    return result


@lru_cache(maxsize=8)
def validated_grid_size(*, impact, radius_bounds, exponent_bounds=(0., 1.),
                        minimum_nodes=1025, order=128):
    """Select a fixed-geometry grid against flux AND physical derivatives.

    This validates cadence interpolation, separately from the table's node
    approximation. Grazing/out-of-domain evaluations keep the caller's larger
    direct grid. An accuracy failure increases resolution instead of changing
    the physical parameter support.
    """
    phases = jnp.linspace(-.5+1e-10, .5-1e-10, 8193)
    mask = jnp.ones(phases.shape, dtype=bool)
    alo, ahi = exponent_bounds
    exponents = sorted(set((alo, min(ahi, max(alo, .001)),
                            min(ahi, max(alo, .1)), (alo+ahi)/2, ahi)))
    parameters = np.array([(r, a) for r in np.linspace(*radius_bounds, 5)
                           for a in exponents])
    def reference(p):
        r, alpha = p
        squared = impact**2 + 4*((1+r)**2-impact**2)*phases**2
        z = jnp.where(squared>0,jnp.sqrt(jnp.where(squared>0,squared,1.)),0.)
        return light_curve(1., alpha, z, r, order=order)
    reference_eval = jax.jit(lambda p:(reference(p),jax.jacfwd(reference)(p)))
    # Hold only one reference slab in device memory.
    references = [(np.asarray(v),np.asarray(g)) for v,g in
                  (reference_eval(p) for p in parameters)]
    for nodes in sorted(set((int(minimum_nodes), max(2049,int(minimum_nodes)),
                             max(4097,int(minimum_nodes))))):
        def candidate(p):
            return duration_light_curve(p[0],1.,p[1],phases,mask,duration=1.,
                impact=impact,num_nodes=nodes,order=order)
        evaluate = jax.jit(lambda p:(candidate(p),jax.jacfwd(candidate)(p)))
        error, gradient_error = 0.,0.
        for p,(expected,expected_gradient) in zip(parameters,references):
            actual,gradient = evaluate(p)
            error=max(error,float(jnp.max(jnp.abs(actual-expected))))
            gradient_error=max(gradient_error,float(jnp.max(jnp.abs(gradient-expected_gradient))))
        print(f'[power2 grid validation] nodes={nodes}, flux_error_ppm={error*1e6:.3g}, '
              f'gradient_error={gradient_error:.3g}', flush=True)
        if error <= 2e-9 and gradient_error <= 2e-7:
            return nodes
    raise ValueError('Power-2 cadence interpolation failed flux/gradient validation.')


def table_node_flux(table, radius, coefficient, exponent, *, indices=None):
    """Evaluate within the table's domain; callers must guard its bounds."""
    coefficients = jnp.asarray(table['coefficients'])
    rlo, rhi = table['radius_bounds']
    alo, ahi = table['exponent_bounds']
    r_basis = _chebyshev_basis((2*radius-rhi-rlo)/(rhi-rlo), coefficients.shape[0]-1)
    a_basis = _chebyshev_basis((2*exponent-ahi-alo)/(ahi-alo), coefficients.shape[1]-1)
    if 'moment_left' in table:
        weights = jnp.einsum('i,j,ijk->k',r_basis,a_basis,jnp.asarray(table['moment_left']))
        right = jnp.asarray(table['moment_right'])
        if indices is not None:
            right = right[:, indices]
        moment = jnp.einsum('k,k...->...', weights, right)
    else:
        selected = coefficients if indices is None else coefficients[:, :, indices]
        moment = jnp.einsum('i,j,ij...->...', r_basis, a_basis, selected)
    z = build_duration_transit_grid(impact=table['impact'], radius_ratio=radius,
                                    num_nodes=coefficients.shape[-1])
    if 'area_coefficients' in table:
        selected = jnp.asarray(table['area_coefficients'])
        if indices is not None:
            selected = selected[:, indices]
        area = jnp.einsum('i,i...->...', r_basis, selected)
    else:
        if indices is not None:
            z = z[indices]
        area = uniform_overlap(z, radius)
    normalization = jnp.pi*(1.0-coefficient*exponent/(exponent+2.0))
    flux = -((1-coefficient)*area+coefficient*moment)/normalization
    if indices is not None:
        return jnp.where(indices == coefficients.shape[-1]-1, 0., flux)
    return flux.at[-1].set(0.)


def _table_cadence_flux(table, radius, coefficient, exponent, phases, mask, *, duration, impact):
    """Evaluate exactly the same six-node stencil without the full node curve."""
    from .transit_grid import _duration_interpolation_metadata
    nodes = table['coefficients'].shape[-1]
    if 6*phases.size >= nodes:
        return interpolate_duration_transit_grid(table_node_flux(table, radius, coefficient, exponent),
            phases, mask, duration=duration, impact=impact, radius_ratio=radius)
    indices, weights, physical_mask = _duration_interpolation_metadata(
        phases, mask, duration, impact, radius, nodes)
    selected = table_node_flux(table, radius, coefficient, exponent, indices=indices)
    flux = jnp.zeros_like(phases)
    for column in range(6):
        flux = flux+weights[..., column]*selected[..., column]
    return jnp.where(physical_mask, flux, 0.)



def _streamed_light_curve(coefficient, exponent, separation, radius, *, order=None):
    """Bound quadrature memory across cadence and channel vmaps.

    Rematerializing each small slab also bounds reverse-mode residuals. The
    quadrature rule and physical target are identical to the vector kernel.
    """
    block_size = 32
    size = separation.shape[0]
    padding = (-size) % block_size
    padded = jnp.pad(separation, (0, padding), constant_values=2.0+jnp.abs(radius))
    evaluate = jax.checkpoint(lambda z:light_curve(coefficient,exponent,z,radius,order=order))
    values = jax.lax.map(evaluate, padded.reshape(-1, block_size))
    return values.reshape(-1)[:size]

def duration_light_curve(radius, coefficient, exponent, phases, mask, *, duration,
                         impact, num_nodes=769, table=None, order=128, use_table=None,
                         strict_table_domain=False):
    """Direct or table-accelerated power-2 grid for fixed duration geometry."""
    impact = jnp.abs(impact)
    def primal(theta):
        r, c, alpha = theta
        squared = impact**2 + 4*jnp.maximum(0., (1+r)**2-impact**2)*phases**2/duration**2
        # Contact neighborhoods are poorly resolved in a Lobatto coordinate.
        # Correct a small fixed-size subset physically instead of abandoning
        # interpolation for every observed cadence in every batched lane.
        contact_tolerance = 1e-5
        inner_distance = jnp.where(r<1.,jnp.abs(squared-(1-r)**2),jnp.inf)
        outer_distance = jnp.abs(squared-(1+r)**2)
        distance = jnp.minimum(inner_distance,outer_distance)
        midpoint_distance = jnp.abs(squared-impact**2)
        near_contact = mask & ((distance<=contact_tolerance) |
            (midpoint_distance<=64*jnp.finfo(jnp.float64).eps))
        distance = jnp.minimum(distance,midpoint_distance)
        safe_mask = mask & ~near_contact
        has_contact = ~_all_lanes_in_domain(~jnp.any(near_contact))
        correction_size = min(16,phases.shape[0])
        sparse_contacts = _all_lanes_in_domain(jnp.sum(near_contact)<=correction_size)
        def correct_contacts(flux):
            def correct(_):
                _,indices = jax.lax.top_k(-jnp.where(near_contact,distance,jnp.inf),correction_size)
                selected = squared[indices]
                z = jnp.where(selected>0,jnp.sqrt(jnp.where(selected>0,selected,1.)),0.)
                physical = light_curve(c,alpha,z,r,order=max(order,128))
                replacement = jnp.where(near_contact[indices],physical,flux[indices])
                return flux.at[indices].set(replacement)
            return jax.lax.cond(has_contact,correct,lambda _:flux,None)
        def interpolate(nodes):
            flux = interpolate_duration_transit_grid(nodes,phases,safe_mask,duration=duration,
                                                     impact=impact,radius_ratio=r)
            return correct_contacts(flux)
        def direct(_):
            def cadence(_):
                squared = impact**2 + 4*jnp.maximum(0., (1+r)**2-impact**2)*phases**2/duration**2
                z = jnp.where(squared>0, jnp.sqrt(jnp.where(squared>0,squared,1.)), 0.)
                return jax.lax.cond(has_contact,
                    lambda _:jnp.where(mask,_streamed_light_curve(c,alpha,z,r,order=max(order,128)),0.),
                    lambda _:jnp.where(mask,_streamed_light_curve(c,alpha,z,r,order=order),0.),None)
            def grid(_):
                z = build_duration_transit_grid(impact=impact,radius_ratio=r,num_nodes=num_nodes)
                evaluate = _streamed_light_curve if table is not None else light_curve
                return interpolate(evaluate(c,alpha,z,r,order=order).at[-1].set(0.))
            if table is not None and strict_table_domain:
                return cadence(None)
            # The inner grid segment collapses at the grazing boundary.
            # Retain the physical derivative there through cadence integration.
            safe_grid = _all_lanes_in_domain(
                (jnp.abs(impact-(1-r)) > 1e-5) & (r >= .003) & (r <= .5)) & sparse_contacts
            return jax.lax.cond(safe_grid, grid, cadence, None)
        if table is None:
            return direct(None)
        rlo,rhi=table['radius_bounds'];alo,ahi=table['exponent_bounds']
        within=((r>=rlo)&(r<=rhi)&(alpha>=alo)&(alpha<=ahi)
                &(c>=0.)&(c<=1.)&(jnp.abs(impact)==table['impact']))
        if use_table is not None:
            within=within & use_table
        within=_all_lanes_in_domain(within) & sparse_contacts
        return jax.lax.cond(within,
            lambda _:correct_contacts(_table_cadence_flux(table,r,c,alpha,phases,safe_mask,
                                        duration=duration,impact=impact)),direct,None)
    theta=jnp.stack((radius,coefficient,exponent))
    if table is None:
        return primal(theta)
    # Form the small parameter Jacobian in forward mode. Reverse mode then
    # contracts it with the cadence cotangent, avoiding a batched scatter
    # transpose through the local interpolation stencils.
    @jax.custom_jvp
    def evaluate(theta):
        return primal(theta)
    @evaluate.defjvp
    def evaluate_jvp(primals,tangents):
        theta,=primals
        tangent,=tangents
        def value_aux(theta):
            flux=primal(theta)
            return flux,flux
        jacobian,flux=jax.jacfwd(value_aux,has_aux=True)(theta)
        return flux,jacobian@tangent
    return evaluate(theta)
