#!/usr/bin/env python3
"""Audit contact-aware PRISM transit interpolation over the model prior box."""

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

from models.common import get_I_power2
from models.detrend import _prepare_power2_poly
from models.jaxoplanet.limb_dark_streamed import light_curve
from models.jaxoplanet.transit_grid import interpolate_duration_transit


jax.config.update("jax_enable_x64", True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="257,513,641,769,1025,2049")
    parser.add_argument("--impact", type=float, default=0.12120438)
    parser.add_argument("--phase-points", type=int, default=1025)
    parser.add_argument("--case-batch", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


MUS, PROJECTION = _prepare_power2_poly(degree=12)


def _coefficients(c, alpha):
    intensity = get_I_power2(c, alpha, MUS)
    return PROJECTION @ (1.0 - intensity)


def _separation(q, impact, radius):
    return jnp.sqrt(
        impact**2
        + q**2 * jnp.maximum(0.0, (1.0 + radius) ** 2 - impact**2)
    )


def _direct(theta, q, impact):
    radius, c, alpha = theta
    u = _coefficients(c, alpha)
    # Match the production duration-orbit mask. At exactly first/fourth
    # contact the transit is physically and operationally zero; evaluating
    # the Green-basis kernel without that mask can expose a large cancellation
    # residue for pathological prior-edge limb-darkening coefficients.
    flux = light_curve(u, _separation(q, impact, radius), radius, order=10)
    return jnp.where(q < 1.0, flux, 0.0)


def _grid(theta, q, impact, nodes):
    radius, c, alpha = theta
    u = _coefficients(c, alpha)
    return interpolate_duration_transit(
        light_curve,
        u,
        0.5 * q,
        jnp.ones_like(q, dtype=bool),
        duration=jnp.float64(1.0),
        impact=jnp.float64(impact),
        radius_ratio=radius,
        num_nodes=nodes,
        order=10,
    )


def _parameter_cases(quick):
    if quick:
        radii = np.array([np.sqrt(1.0e-5), 0.03, 0.1, 0.3, np.sqrt(0.5)])
        cs = np.array([0.0, 0.5, 1.0])
        alphas = np.array([0.001, 0.1, 0.5, 1.0])
    else:
        radii = np.unique(np.concatenate((
            np.geomspace(np.sqrt(1.0e-5), 0.1, 9),
            np.linspace(0.1, np.sqrt(0.5), 9),
        )))
        cs = np.linspace(0.0, 1.0, 9)
        alphas = np.unique(np.concatenate((
            np.geomspace(0.001, 0.1, 6),
            np.linspace(0.1, 1.0, 7),
        )))
    return np.asarray(
        [(r, c, a) for r in radii for c in cs for a in alphas],
        dtype=np.float64,
    )


def _phase_cases(cases, impact, phase_points):
    uniform = np.linspace(0.0, 1.0, phase_points, dtype=np.float64)
    relative_offsets = np.concatenate((
        np.array([0.0]),
        np.geomspace(1.0e-8, 1.0e-2, 48),
    ))
    rows = []
    for radius in cases[:, 0]:
        denominator = (1.0 + radius) ** 2 - impact**2
        numerator = (1.0 - radius) ** 2 - impact**2
        inner = np.sqrt(max(0.0, numerator) / denominator)
        focused = np.concatenate((
            np.clip(inner - relative_offsets, 0.0, 1.0),
            np.clip(inner + relative_offsets, 0.0, 1.0),
            np.clip(1.0 - relative_offsets, 0.0, 1.0),
        ))
        rows.append(np.concatenate((uniform, focused)))
    return np.asarray(rows, dtype=np.float64)


def _ready(tree):
    return jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready") else value,
        tree,
    )


def _evaluate(nodes, cases, phases, impact, case_batch):
    candidate_one = lambda theta, q: _grid(theta, q, impact, nodes)
    direct_one = lambda theta, q: _direct(theta, q, impact)
    candidate = jax.jit(jax.vmap(lambda theta, q: (
        candidate_one(theta, q), jax.jacfwd(candidate_one)(theta, q)
    )))
    reference = jax.jit(jax.vmap(lambda theta, q: (
        direct_one(theta, q), jax.jacfwd(direct_one)(theta, q)
    )))

    max_flux = 0.0
    max_gradient = np.zeros(3, dtype=np.float64)
    worst_flux = None
    worst_gradient = [None, None, None]
    started = time.perf_counter()
    for start in range(0, len(cases), case_batch):
        stop = min(start + case_batch, len(cases))
        real_count = stop - start
        theta = cases[start:stop]
        q = phases[start:stop]
        if real_count < case_batch:
            theta = np.concatenate((
                theta,
                np.repeat(theta[-1:], case_batch - real_count, axis=0),
            ))
            q = np.concatenate((
                q,
                np.repeat(q[-1:], case_batch - real_count, axis=0),
            ))
        direct_flux, direct_gradient = _ready(reference(theta, q))
        grid_flux, grid_gradient = _ready(candidate(theta, q))
        flux_error = np.abs(np.asarray(grid_flux - direct_flux)[:real_count])
        gradient_error = np.abs(
            np.asarray(grid_gradient - direct_gradient)[:real_count]
        )
        active_cases = cases[start:stop]
        active_phases = phases[start:stop]
        radii = active_cases[:, 0, None]
        denominator = (1.0 + radii) ** 2 - impact**2
        numerator = (1.0 - radii) ** 2 - impact**2
        inner_phase = np.sqrt(np.maximum(0.0, numerator) / denominator)
        differentiable = (
            (np.abs(active_phases - inner_phase) > 1.0e-12)
            & (active_phases < 1.0 - 1.0e-12)
        )
        gradient_error = np.where(
            differentiable[..., None], gradient_error, 0.0
        )
        local_flux_index = np.unravel_index(
            int(np.argmax(flux_error)), flux_error.shape
        )
        local_flux = float(flux_error[local_flux_index])
        if local_flux > max_flux:
            case_index = start + local_flux_index[0]
            max_flux = local_flux
            worst_flux = {
                "theta_r_c_alpha": cases[case_index].tolist(),
                "phase": float(phases[case_index, local_flux_index[1]]),
            }
        for parameter in range(3):
            component = gradient_error[..., parameter]
            local_index = np.unravel_index(
                int(np.argmax(component)), component.shape
            )
            local_value = float(component[local_index])
            if local_value > max_gradient[parameter]:
                case_index = start + local_index[0]
                max_gradient[parameter] = local_value
                worst_gradient[parameter] = {
                    "theta_r_c_alpha": cases[case_index].tolist(),
                    "phase": float(phases[case_index, local_index[1]]),
                }
    return {
        "nodes": int(nodes),
        "max_flux_error_ppm": 1.0e6 * max_flux,
        "max_gradient_error_ppm_per_unit": {
            name: 1.0e6 * float(value)
            for name, value in zip(("radius_ratio", "c", "alpha"), max_gradient)
        },
        "worst_flux": worst_flux,
        "worst_gradient": {
            name: value for name, value in
            zip(("radius_ratio", "c", "alpha"), worst_gradient)
        },
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    args = _parser().parse_args()
    nodes = [int(item) for item in args.nodes.split(",")]
    if any(item < 13 for item in nodes):
        raise ValueError("Every node count must be at least 13.")
    cases = _parameter_cases(args.quick)
    phases = _phase_cases(cases, args.impact, args.phase_points)
    results = []
    for count in nodes:
        result = _evaluate(
            count, cases, phases, args.impact, args.case_batch
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "impact": args.impact,
        "prior": {
            "radius_ratio": [float(np.sqrt(1.0e-5)), float(np.sqrt(0.5))],
            "c": [0.0, 1.0],
            "alpha": [0.001, 1.0],
        },
        "case_count": int(len(cases)),
        "phase_points_per_case": int(phases.shape[1]),
        "total_flux_points_per_node_count": int(phases.size),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
