#!/usr/bin/env python3
"""Audit grazing and quadratic-LD PRISM transit interpolation.

The cadence grid is evaluated in normalized duration phase, exactly as in the
production model.  Every family includes prior edges and contact-focused
phases; quadratic sum/difference cases span the production wide-uniform box.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from jaxoplanet.core.limb_dark import light_curve as stock_light_curve

from models.common import get_I_power2
from models.detrend import _prepare_power2_poly
from models.jaxoplanet.limb_dark_streamed import light_curve as streamed_light_curve
from models.jaxoplanet.transit_grid import interpolate_duration_transit


jax.config.update("jax_enable_x64", True)
MUS, PROJECTION = _prepare_power2_poly(degree=12)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="513,769,1025,1281,1537,2049")
    parser.add_argument(
        "--families",
        default="power2_grazing,quadratic_coefficients,quadratic_uplus_uminus",
    )
    parser.add_argument("--phase-points", type=int, default=257)
    parser.add_argument("--case-batch", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _coefficients(parameters, family):
    if family == "power2_grazing":
        c, alpha = parameters[1], parameters[2]
        return PROJECTION @ (1.0 - get_I_power2(c, alpha, MUS))
    if family == "quadratic_coefficients":
        return parameters[1:3]
    if family == "quadratic_uplus_uminus":
        u_plus, u_minus = parameters[1], parameters[2]
        return jnp.array(
            [0.5 * (u_plus + u_minus), 0.5 * (u_plus - u_minus)],
            dtype=jnp.float64,
        )
    raise ValueError(family)


def _separation(q, impact, radius):
    return jnp.sqrt(
        impact**2
        + q**2 * jnp.maximum(0.0, (1.0 + radius) ** 2 - impact**2)
    )


def _direct(parameters, impact, q, family):
    radius = parameters[0]
    separation = _separation(q, impact, radius)
    kernel = (
        streamed_light_curve if family == "power2_grazing" else stock_light_curve
    )
    flux = kernel(
        _coefficients(parameters, family), separation, radius, order=10
    )
    return jnp.where(separation < 1.0 + radius, flux, 0.0)


def _grid(parameters, impact, q, family, nodes):
    radius = parameters[0]
    return interpolate_duration_transit(
        stock_light_curve,
        _coefficients(parameters, family),
        0.5 * q,
        jnp.ones_like(q, dtype=bool),
        duration=jnp.float64(1.0),
        impact=impact,
        radius_ratio=radius,
        num_nodes=nodes,
        order=10,
    )


def _axis(low, high, count, quick):
    return np.linspace(low, high, 3 if quick else count, dtype=np.float64)


def _radii(quick):
    if quick:
        return np.array([np.sqrt(1e-5), 0.1, np.sqrt(0.5)])
    return np.unique(np.concatenate((
        np.geomspace(np.sqrt(1e-5), 0.1, 5),
        np.linspace(0.1, np.sqrt(0.5), 5),
    )))


def _impacts(radius, *, grazing_only, quick):
    inner = max(0.0, 1.0 - radius)
    outer = 1.0 + radius
    epsilon = min(1.0e-7, 0.05 * radius)
    if grazing_only:
        values = [
            inner,
            min(outer, inner + epsilon),
            inner + 0.05 * (outer - inner),
            0.5 * (inner + outer),
            outer - 0.05 * (outer - inner),
            max(inner, outer - epsilon),
            outer,
        ]
    else:
        values = [
            0.0,
            0.12120438,
            0.464,
            0.637,
            0.8523,
            max(0.0, inner - epsilon),
            inner,
            min(outer, inner + epsilon),
            0.5 * (inner + outer),
            max(inner, outer - epsilon),
            outer,
        ]
    values = np.unique(np.clip(values, 0.0, outer))
    if quick and values.size > 5:
        values = values[np.linspace(0, values.size - 1, 5).round().astype(int)]
    return values


def _cases(family, quick):
    radii = _radii(quick)
    if family == "power2_grazing":
        first = _axis(0.0, 1.0, 7, quick)
        second = np.unique(np.concatenate((
            np.geomspace(0.001, 0.1, 4 if quick else 5),
            _axis(0.1, 1.0, 6, quick),
        )))
        grazing_only = True
    elif family == "quadratic_coefficients":
        first = _axis(0.0, 1.0, 7, quick)
        second = _axis(0.0, 1.0, 7, quick)
        grazing_only = False
    elif family == "quadratic_uplus_uminus":
        # This is the exact wide-uniform production rectangle.  Its transform
        # reaches individual physical coefficients from -1.5 through 2.0.
        first = _axis(-1.0, 2.0, 7, quick)
        second = _axis(-2.0, 2.0, 9, quick)
        grazing_only = False
    else:
        raise ValueError(family)
    parameters = []
    impacts = []
    for radius in radii:
        for impact in _impacts(
            radius, grazing_only=grazing_only, quick=quick
        ):
            for left in first:
                for right in second:
                    parameters.append((radius, left, right))
                    impacts.append(impact)
    return np.asarray(parameters), np.asarray(impacts)


def _phases(parameters, impacts, phase_points):
    uniform = np.linspace(0.0, 1.0, phase_points, dtype=np.float64)
    offsets = np.concatenate((np.array([0.0]), np.geomspace(1e-9, 1e-2, 36)))
    rows = []
    for parameters_row, impact in zip(parameters, impacts):
        radius = parameters_row[0]
        denominator = max(0.0, (1.0 + radius) ** 2 - impact**2)
        numerator = (1.0 - radius) ** 2 - impact**2
        focused = [
            np.clip(offsets, 0.0, 1.0),
            np.clip(1.0 - offsets, 0.0, 1.0),
        ]
        if numerator > 0.0 and denominator > 0.0:
            inner_phase = np.sqrt(numerator / denominator)
            focused.extend((
                np.clip(inner_phase - offsets, 0.0, 1.0),
                np.clip(inner_phase + offsets, 0.0, 1.0),
            ))
        else:
            focused.extend((np.zeros_like(offsets), np.zeros_like(offsets)))
        rows.append(np.concatenate((uniform, *focused)))
    return np.asarray(rows)


def _ready(tree):
    return jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready") else value,
        tree,
    )


def _evaluate(family, nodes, parameters, impacts, phases, batch_size):
    reference_one = lambda theta, b, q: _direct(theta, b, q, family)
    candidate_one = lambda theta, b, q: _grid(theta, b, q, family, nodes)
    reference = jax.jit(jax.vmap(lambda theta, b, q: (
        reference_one(theta, b, q),
        jax.jacfwd(reference_one)(theta, b, q),
    )))
    candidate = jax.jit(jax.vmap(lambda theta, b, q: (
        candidate_one(theta, b, q),
        jax.jacfwd(candidate_one)(theta, b, q),
    )))
    maxima = np.zeros(4, dtype=np.float64)
    worst = [None] * 4
    smooth_radius_max = 0.0
    started = time.perf_counter()
    for start in range(0, len(parameters), batch_size):
        stop = min(start + batch_size, len(parameters))
        real_count = stop - start
        theta = parameters[start:stop]
        impact = impacts[start:stop]
        q = phases[start:stop]
        if real_count < batch_size:
            padding = batch_size - real_count
            theta = np.concatenate((theta, np.repeat(theta[-1:], padding, axis=0)))
            impact = np.concatenate((impact, np.repeat(impact[-1:], padding)))
            q = np.concatenate((q, np.repeat(q[-1:], padding, axis=0)))
        direct_flux, direct_gradient = _ready(reference(theta, impact, q))
        grid_flux, grid_gradient = _ready(candidate(theta, impact, q))
        flux_error = np.abs(np.asarray(grid_flux - direct_flux)[:real_count])
        gradient_error = np.abs(
            np.asarray(grid_gradient - direct_gradient)[:real_count]
        )
        # Production makes this decision from the fixed handoff before JIT:
        # if even the smallest allowed planet can approach outer contact, the
        # complete stage retains the original direct model.
        eligible = (
            impacts[start:stop]
            < 1.0 + np.sqrt(1.0e-5) - 1.0e-5
        )
        flux_error = np.where(eligible[:, None], flux_error, 0.0)
        gradient_error = np.where(
            eligible[:, None, None], gradient_error, 0.0
        )
        components = (flux_error,) + tuple(
            gradient_error[..., index] for index in range(3)
        )
        for component_index, component in enumerate(components):
            local_index = np.unravel_index(int(np.argmax(component)), component.shape)
            local_value = float(component[local_index])
            if local_value > maxima[component_index]:
                case_index = start + local_index[0]
                maxima[component_index] = local_value
                worst[component_index] = {
                    "parameters": parameters[case_index].tolist(),
                    "impact": float(impacts[case_index]),
                    "phase": float(phases[case_index, local_index[1]]),
                }

        active_parameters = parameters[start:stop]
        active_impacts = impacts[start:stop]
        radii = active_parameters[:, 0, None]
        inner = 1.0 - radii
        outer = 1.0 + radii
        separation = np.sqrt(
            active_impacts[:, None] ** 2
            + phases[start:stop] ** 2
            * np.maximum(0.0, outer**2 - active_impacts[:, None] ** 2)
        )
        smooth = (
            (np.abs(separation - inner) > 1e-10)
            & (np.abs(separation - outer) > 1e-10)
            & (active_impacts[:, None] < outer - 1e-10)
        )
        if np.any(smooth):
            smooth_radius_max = max(
                smooth_radius_max,
                float(np.max(np.where(smooth, gradient_error[..., 0], 0.0))),
            )
    parameter_names = (
        ("radius_ratio", "c", "alpha")
        if family == "power2_grazing"
        else (
            ("radius_ratio", "u1", "u2")
            if family == "quadratic_coefficients"
            else ("radius_ratio", "u_plus", "u_minus")
        )
    )
    return {
        "nodes": int(nodes),
        "direct_fallback_case_count": int(np.sum(
            impacts >= 1.0 + np.sqrt(1.0e-5) - 1.0e-5
        )),
        "max_flux_error_ppm": 1e6 * float(maxima[0]),
        "max_gradient_error_ppm_per_unit": {
            name: 1e6 * float(value)
            for name, value in zip(parameter_names, maxima[1:])
        },
        "max_smooth_radius_gradient_error_ppm_per_unit": (
            1e6 * smooth_radius_max
        ),
        "worst": {
            name: payload for name, payload in zip(
                ("flux",) + parameter_names, worst
            )
        },
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    args = _parser().parse_args()
    node_counts = [int(value) for value in args.nodes.split(",")]
    families = tuple(value.strip() for value in args.families.split(","))
    reports = {}
    for family in families:
        parameters, impacts = _cases(family, args.quick)
        phases = _phases(parameters, impacts, args.phase_points)
        family_results = []
        for nodes in node_counts:
            result = _evaluate(
                family,
                nodes,
                parameters,
                impacts,
                phases,
                args.case_batch,
            )
            family_results.append(result)
            print(json.dumps({"family": family, **result}), flush=True)
        reports[family] = {
            "case_count": int(len(parameters)),
            "phase_points_per_case": int(phases.shape[1]),
            "results": family_results,
        }
    report = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "prior_boxes": {
            "radius_ratio": [float(np.sqrt(1e-5)), float(np.sqrt(0.5))],
            "impact": "contact-focused from 0 through 1 + radius_ratio",
            "power2": {"c": [0.0, 1.0], "alpha": [0.001, 1.0]},
            "quadratic_coefficients": {"u1": [0.0, 1.0], "u2": [0.0, 1.0]},
            "quadratic_uplus_uminus": {
                "u_plus": [-1.0, 2.0], "u_minus": [-2.0, 2.0]
            },
        },
        "families": reports,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
