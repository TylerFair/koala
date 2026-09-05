#!/usr/bin/env python3
"""Repeatable injection recovery through the production surface model and Laplace-NUTS.

Geometry and limb darkening are known in this bounded validation; emission,
spot contrast, and normalization are inferred. This is not a substitute for
checking convergence and identifiability in a full observation.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS

from models.independent_nuts import prepare_laplace_metric
from models.jaxoplanet.builder import (
    create_vectorized_model,
    phase_flux_to_conditional_quantile,
)
from models.jaxoplanet.config import parse_surface_config
from models.jaxoplanet.surface import compute_surface_model
from models.jaxoplanet.surface_basis import (
    prepare_emission_light_curve_basis,
    prepare_spot_light_curve_basis,
)
from surface_outputs import save_surface_results


def validate(scenario, output, draws=500, phase_nightside_ppm=None):
    planet = {"eclipse_depth_ppm": 1200., "eclipse_depth_prior_width_ppm": 500.,
              "dayside_flux_ppm": 1200., "dayside_flux_prior_width_ppm": 300.,
              "nightside_flux_ppm": 400., "nightside_flux_prior_width_ppm": 150.,
              "hotspot_offset_deg": 20., "hotspot_offset_prior_width_deg": 15.}
    if phase_nightside_ppm is not None:
        planet["nightside_flux_ppm"] = float(phase_nightside_ppm)
    stellar = {}
    mode = scenario
    if scenario == "stellar_spots":
        mode = "transit"
        stellar = {"rotation_period": 2., "spots": [{"latitude_deg": 15.,
                   "longitude_deg": 0., "radius_deg": 20., "contrast": .4,
                   "contrast_prior_width": .15}]}
    # The exact spot basis is available only when geometry is fixed. Other
    # scenarios condition their production geometry sites explicitly below.
    config = parse_surface_config(
        {
            "light_curve_model": mode,
            "fit_geometry": False,
        },
        planet,
        stellar,
        1,
    )
    t = jnp.linspace(.35, .65, 81) if mode == "eclipse" else jnp.linspace(-.1, 1.1, 121)
    physical = {"period": 1., "t0": 0., "a_rs": 6., "b": .25,
                "rors": .1, "u": jnp.array([.3, .2])}
    sites = ["eclipse_depth"] if mode == "eclipse" else ["dayside_flux", "nightside_flux", "hotspot_offset"]
    if scenario == "stellar_spots":
        physical.update(stellar_rotation_period=2., stellar_spot_contrast=jnp.array([.4]))
        sites = ["stellar_spot_contrast"]
    else:
        physical.update({site: config[site][0] for site in sites})
    signal = compute_surface_model(physical, t, model=mode, spots=config["spots"])
    surface_basis = None
    basis_preparation_seconds = 0.0
    if scenario == "stellar_spots":
        start = time.perf_counter()
        surface_basis = prepare_spot_light_curve_basis(
            physical,
            t,
            spots=config["spots"],
        )
        basis_preparation_seconds = time.perf_counter() - start
    else:
        start = time.perf_counter()
        surface_basis = prepare_emission_light_curve_basis(
            physical,
            t,
            model=mode,
        )
        basis_preparation_seconds = time.perf_counter() - start
    baseline = 1. / float(jnp.median(1. + signal))
    truth_flux = baseline * (1. + np.asarray(signal))
    rng = np.random.default_rng(20260904)
    error = jnp.full((1, len(t)), 4e-5)
    observed = jnp.array(truth_flux + rng.normal(0., 4e-5, len(t)))[None, :]
    model = create_vectorized_model(detrend_type="linear", ld_mode="fixed",
                                    param_method="a_rs", surface_config=config,
                                    surface_basis=surface_basis)
    model = numpyro.handlers.condition(model, data={"rors": jnp.array([[.1]]),
                                       "log_jitter": jnp.array([np.log(1e-6)]), "v": jnp.array([0.])})
    kwargs = {"y": observed, "mu_t0": jnp.array([0.]), "mu_b": jnp.array([.25]),
              "mu_a_rs": jnp.array([6.]), "PERIOD": jnp.array([1.]),
              "ld_fixed": jnp.array([[.3, .2]])}
    kwargs["mu_depths"] = jnp.array([.01])
    initial = {"c": jnp.array([baseline])}
    for site in sites:
        if site == "nightside_flux" and config["nightside_flux_prior_width"][0] > 0:
            initial["_nightside_flux_quantile_0"] = jnp.array([
                phase_flux_to_conditional_quantile(
                    physical["nightside_flux"],
                    physical["dayside_flux"],
                    config["nightside_flux"][0],
                    config["nightside_flux_prior_width"][0],
                )
            ])
        else:
            initial[f"_{site}_0"] = jnp.array(
                [float(np.ravel(physical[site])[0])]
            )
    start = time.perf_counter()
    metric = prepare_laplace_metric(model, jax.random.PRNGKey(2), initial, t, error,
                                    model_kwargs=kwargs, max_iterations=30,
                                    sequential_evaluations=True)
    jax.block_until_ready(metric.inverse_mass_matrix)
    preparation_seconds = time.perf_counter() - start
    sampler = MCMC(NUTS(model, dense_mass=True, inverse_mass_matrix=metric.inverse_mass_matrix,
                        adapt_mass_matrix=False, target_accept_prob=.9, max_tree_depth=7),
                   num_warmup=100, num_samples=draws, progress_bar=False)
    start = time.perf_counter()
    sampler.run(jax.random.PRNGKey(3), t, error, **kwargs, init_params=metric.unconstrained_map)
    samples = jax.device_get(sampler.get_samples())
    elapsed = time.perf_counter() - start
    report = {"scenario": scenario, "draws": draws, "backend": jax.default_backend(),
              "basis_preparation_seconds": basis_preparation_seconds,
              "laplace_preparation_seconds": preparation_seconds,
              "nuts_seconds_including_compile": elapsed,
              "divergences": int(np.sum(sampler.get_extra_fields()["diverging"])), "parameters": {}}
    for site in sites:
        chain = np.asarray(samples[site]).reshape(draws, -1)[:, 0]
        truth = float(np.ravel(physical[site])[0])
        low, median, high = np.percentile(chain, [.135, 50, 99.865])
        ess = float(numpyro.diagnostics.effective_sample_size(jnp.asarray(chain)[None, :]))
        report["parameters"][site] = {"truth": truth, "median": float(median),
                                      "interval_9973": [float(low), float(high)], "ess": ess,
                                      "recovered": bool(low <= truth <= high)}
    report["passed"] = report["divergences"] == 0 and all(
        p["recovered"] and np.isfinite(p["ess"]) and p["ess"] >= 50
        for p in report["parameters"].values())
    output.mkdir(parents=True, exist_ok=True)
    save_surface_results([3.5], [.1], samples, output / f"{scenario}.csv")
    (output / f"{scenario}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["eclipse", "phase_curve", "stellar_spots", "all"], nargs="?", default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/surface_validation"))
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument(
        "--phase-nightside-ppm",
        type=float,
        help="Override the injected nightside flux for boundary validation.",
    )
    args = parser.parse_args()
    scenarios = ["eclipse", "phase_curve", "stellar_spots"] if args.scenario == "all" else [args.scenario]
    results = []
    for name in scenarios:
        results.append(
            validate(
                name,
                args.output_dir,
                args.draws,
                phase_nightside_ppm=args.phase_nightside_ppm,
            )
        )
        # The three different map degrees compile separate programs. Release
        # those programs between examples to bound the CPU validation memory.
        jax.clear_caches()
    sys.exit(0 if all(results) else 1)
