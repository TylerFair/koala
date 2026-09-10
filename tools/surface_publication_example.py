#!/usr/bin/env python3
"""Executed synthetic surface tutorial using the production likelihood and priors.

Run from the repository root in the fitter environment. No downloaded data or
stellar-atmosphere grid is needed. Output archives retain chain and draw axes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import yaml
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from models.independent_nuts import prepare_laplace_metric
from models.jaxoplanet.builder import create_vectorized_model, phase_flux_to_conditional_quantile
from models.jaxoplanet.config import parse_surface_config
from models.jaxoplanet.surface_basis import (
    compute_emission_basis_model, compute_spot_basis_model,
    prepare_emission_light_curve_basis, prepare_spot_light_curve_basis,
)
from plotting_style import apply_publication_style
from surface_outputs import save_surface_results
from koala.config import parse_planet_parameter_specs
from tools.example_phase_curves import _scenario, _write_nirspec, GEOMETRY, NOISE_PPM

ROOT = Path(__file__).resolve().parents[1]
SITES = {"eclipse": ["eclipse_depth"],
         "phase_curve": ["dayside_flux", "nightside_flux", "hotspot_offset"],
         "stellar_spots": ["stellar_spot_contrast"]}
LABELS = {"eclipse_depth": "Eclipse depth [ppm]", "dayside_flux": "Day maximum [ppm]",
          "nightside_flux": "Night minimum [ppm]", "hotspot_offset": "Offset [deg]",
          "stellar_spot_contrast": "Spot contrast"}


def scale(site):
    return 180 / np.pi if site == "hotspot_offset" else (1 if site == "stellar_spot_contrast" else 1e6)


def thermal_map(day, night, offset, longitude, latitude):
    """Dipole intensity / disk-mean stellar intensity, times projected area ratio.

    Inputs/outputs use ppm and radians. Longitude increases in the direction
    of a later phase maximum; zero faces the observer at secondary eclipse.
    Disk integration gives mean + (day-night)/2*cos(viewing longitude-offset).
    This is a model-imposed latitude dependence, not an independent latitude fit.
    """
    return (day + night) / 2 + .75 * (day - night) * np.cos(latitude) * np.cos(longitude - offset)


def save(fig, directory, name):
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def posterior_lightcurves(name, samples, basis, t):
    """Reconstruct the production multiplicative surface normalization."""
    evaluate = compute_spot_basis_model if name == "stellar_spots" else compute_emission_basis_model
    params = {key: jnp.asarray(samples[key][..., 0, None]) for key in SITES[name]}
    if name == "stellar_spots":
        params["stellar_spot_contrast"] = jnp.asarray(samples["stellar_spot_contrast"])
    signal = np.asarray(evaluate(params, basis))
    return samples["c"][..., None] * (1 + signal) + samples["v"][..., None] * np.asarray(t - t.min())


def fit(name, output, draws=1000, warmup=500, seed=20260904):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "examples" / f"{name}.yaml").read_text())
    config = parse_surface_config(
        cfg["flags"], cfg["planet"], cfg["stellar"], 1,
        parameter_specs=parse_planet_parameter_specs(cfg["planet"]),
        param_method="a_rs",
    )
    # This tool fits with the injected geometry held fixed; the production
    # pipeline (fit_jwst.py) honours the priors in the example file.
    geometry = GEOMETRY[name]
    PERIOD, T0, A_RS, B, RPRS = (
        geometry["period"], geometry["t0"], geometry["a_rs"],
        geometry["b"], geometry["rprs"],
    )
    wave = np.linspace(2.9, 5., 14)
    time, signal, truth = _scenario(name, wave)
    rng = np.random.default_rng(seed)
    error = np.full_like(signal, NOISE_PPM * 1e-6)
    observed = signal + rng.normal(size=signal.shape) * error
    _write_nirspec(directory / "synthetic.fits", time, wave, observed, error)
    truth["seed"] = seed
    (directory / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg))
    # Work in days from transit to avoid carrying large absolute epochs into NUTS.
    t = jnp.asarray(time - T0)
    physical = dict(period=PERIOD, t0=0., a_rs=A_RS, b=B, rors=RPRS, u=jnp.array([.3, .2]))
    if name == "stellar_spots":
        physical["stellar_rotation_period"] = cfg["stellar"]["rotation_period"]
        basis = prepare_spot_light_curve_basis(physical, t, spots=config["spots"])
    else:
        basis = prepare_emission_light_curve_basis(physical, t, model=name)
    model = create_vectorized_model(detrend_type="linear", ld_mode="fixed", param_method="a_rs",
                                    surface_config=config, surface_basis=basis)
    # The known injected noise is used; normalization and slope remain free.
    model = numpyro.handlers.condition(model, data={"log_jitter": jnp.full(len(wave), np.log(1e-6))})
    kwargs = dict(y=jnp.asarray(observed.T), mu_t0=jnp.array([0.]), mu_b=jnp.array([B]),
                  mu_a_rs=jnp.array([A_RS]), PERIOD=jnp.array([PERIOD]),
                  ld_fixed=jnp.tile(jnp.array([.3, .2]), (len(wave), 1)), mu_depths=jnp.array([RPRS**2]))
    initial = {"c": jnp.ones(len(wave)), "v": jnp.zeros(len(wave))}
    for site in SITES[name]:
        if site == "nightside_flux":
            value = phase_flux_to_conditional_quantile(config[site][0], config["dayside_flux"][0],
                                                       config[site][0], config[site + "_prior_width"][0])
            key = "_nightside_flux_quantile_0"
        else:
            value = config[site][0] if site != "stellar_spot_contrast" else config["spots"][0]["contrast"]
            key = f"_{site}_0"
        initial[key] = jnp.full(len(wave), value)
    print(f"{name}: preparing {len(wave)} channels and two sequential chains", flush=True)
    metric = prepare_laplace_metric(model, jax.random.PRNGKey(seed), initial, t, jnp.asarray(error.T),
                                    model_kwargs=kwargs, max_iterations=50, sequential_evaluations=True)
    # Separate dispersed starts in unconstrained coordinates; adapt each chain.
    starts = jax.tree.map(lambda x: jnp.stack([x - .05, x + .05]), metric.unconstrained_map)
    sampler = MCMC(NUTS(model, dense_mass=True, inverse_mass_matrix=metric.inverse_mass_matrix,
                        adapt_mass_matrix=False, target_accept_prob=.9), num_warmup=warmup,
                   num_samples=draws, num_chains=2, chain_method="sequential", progress_bar=False)
    sampler.run(jax.random.PRNGKey(seed + 1), t, jnp.asarray(error.T), **kwargs, init_params=starts)
    chains = {key: np.asarray(value) for key, value in sampler.get_samples(group_by_chain=True).items()}
    samples = {key: value.reshape((-1,) + value.shape[2:]) for key, value in chains.items()}
    extra = {key: np.asarray(value) for key, value in sampler.get_extra_fields(group_by_chain=True).items()}
    np.savez_compressed(directory / "posterior.npz", **chains)
    np.savez_compressed(directory / "sampler.npz", **extra)
    diag = summary({key: chains[key] for key in SITES[name] + ["c", "v"]})
    records = []
    for key, values in diag.items():
        for channel, wavelength in enumerate(wave):
            records.append(dict(parameter=key, wavelength_um=wavelength,
                                ess=float(np.ravel(values["n_eff"])[channel]),
                                r_hat=float(np.ravel(values["r_hat"])[channel])))
    pd.DataFrame(records).to_csv(directory / "diagnostics.csv", index=False)
    report = dict(scenario=name, seed=seed, draws_per_chain=draws, warmup_per_chain=warmup, chains=2,
                  jax=jax.__version__, numpyro=numpyro.__version__, backend=jax.default_backend(),
                  divergences=int(extra["diverging"].sum()), min_ess=min(r["ess"] for r in records),
                  max_r_hat=max(r["r_hat"] for r in records))
    report["passed"] = bool(report["divergences"] == 0 and all(
        np.isfinite(r["ess"]) and r["ess"] >= 400 and np.isfinite(r["r_hat"]) and r["r_hat"] < 1.01
        for r in records))
    (directory / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    save_surface_results(wave, np.diff(wave)[0] / 2, samples, directory / "spectrum.csv")
    # Exact basis evaluation of the fitted likelihood, including nuisance trends.
    prediction = posterior_lightcurves(name, samples, basis, t)
    quantiles = np.percentile(prediction, [15.865, 50, 84.135], axis=0)
    np.savez_compressed(directory / "lightcurves.npz", time_bmjd_tdb=time, wavelength_um=wave,
                        observed=observed, uncertainty=error, truth=signal,
                        model_quantiles=quantiles)
    plot_products(name, directory, wave, time, observed, error, truth, samples, quantiles)
    print(json.dumps(report, indent=2), flush=True)
    return report


def plot_products(name, directory, wave, time, observed, error, truth, samples, quantiles):
    PERIOD, T0 = GEOMETRY[name]["period"], GEOMETRY[name]["t0"]
    import corner
    apply_publication_style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    sites = SITES[name]
    injected = {}
    for key in sites:
        truth_key = key + ("_deg" if key == "hotspot_offset" else "_ppm")
        value = truth["spots"][0]["contrast"] if name == "stellar_spots" else truth[truth_key]
        injected[key] = np.broadcast_to(value, wave.shape)
    fig, axes = plt.subplots(len(sites), 1, figsize=(7, 2.7 * len(sites)), squeeze=False, sharex=True, layout="constrained")
    for ax, key in zip(axes[:, 0], sites):
        lo, med, hi = np.percentile(samples[key][..., 0] * scale(key), [15.865, 50, 84.135], axis=0)
        ax.plot(wave, injected[key], "--", color="0.3", label="Injection")
        ax.errorbar(wave, med, yerr=[med-lo, hi-med], xerr=np.diff(wave)[0]/2, fmt="o", color="#8b368c", label="Posterior 68.27%")
        ax.set_ylabel(LABELS[key])
    axes[0, 0].legend(fontsize=9)
    axes[-1, 0].set_xlabel("Wavelength [µm]")
    save(fig, directory, "recovery")
    channel = len(wave) // 2
    phase = (time - T0) / PERIOD
    lo, med, hi = quantiles[:, channel]
    residual = (observed[:, channel] - med) * 1e6
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), layout="constrained")
    axes[0].errorbar(phase, (observed[:, channel]-1)*1e6, yerr=error[:, channel]*1e6, fmt=".", alpha=.45, color="0.4")
    axes[0].plot(phase, (med-1)*1e6, color="#8b368c")
    axes[0].fill_between(phase, (lo-1)*1e6, (hi-1)*1e6, color="#8b368c", alpha=.3)
    axes[0].set(ylabel="Relative flux − 1 [ppm]", title=f"Synthetic {name.replace('_', ' ')} · {wave[channel]:.2f} µm")
    axes[1].plot(phase, residual, ".", color="0.4")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set(xlabel="Orbital phase (transit = 0)", ylabel="Residual [ppm]")
    if name == "phase_curve":
        axes[0].set_ylim(-300, 1600)
        axes[0].text(.02, .05, "Transit depth extends below display", transform=axes[0].transAxes, fontsize=9)
    sizes = np.array([1, 2, 4, 8, 16])
    rms = [np.std(residual[:len(residual)//n*n].reshape(-1, n).mean(axis=1), ddof=1) for n in sizes]
    axes[2].loglog(sizes, rms, "o-", label="Binned residual RMS")
    axes[2].loglog(sizes, rms[0]/np.sqrt(sizes), "--", label="1/√N reference")
    axes[2].set(xlabel="Cadences per bin", ylabel="RMS [ppm]")
    axes[2].legend(fontsize=9)
    save(fig, directory, "lightcurve")
    columns = [samples[key][:, channel, 0] * scale(key) for key in sites]
    columns += [(samples["c"][:, channel]-1)*1e6, samples["v"][:, channel]*1e6]
    labels = [LABELS[key] for key in sites] + ["Baseline − 1 [ppm]", "Slope [ppm/day]"]
    with plt.rc_context({"figure.constrained_layout.use": False, "xtick.labelsize": 8,
                         "ytick.labelsize": 8}):
        fig = corner.corner(np.column_stack(columns), labels=labels,
                            truths=[injected[key][channel] for key in sites] + [0., 0.],
                            quantiles=[.15865, .5, .84135], show_titles=True, title_fmt=".2f",
                            title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 10})
    save(fig, directory, "corner")
    if name == "phase_curve":
        lon, lat = np.meshgrid(np.linspace(-np.pi, np.pi, 121), np.linspace(-np.pi/2, np.pi/2, 61))
        maps = thermal_map(samples["dayside_flux"][:, channel, 0, None, None]*1e6,
                           samples["nightside_flux"][:, channel, 0, None, None]*1e6,
                           samples["hotspot_offset"][:, channel, 0, None, None], lon, lat)
        quantiles = np.percentile(maps, [15.865, 50, 84.135], axis=0)
        np.savez_compressed(directory / "thermal_map.npz", longitude_deg=np.rad2deg(lon), latitude_deg=np.rad2deg(lat), quantiles_ppm=quantiles)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
        for ax, values, title in zip(axes, [quantiles[1], (quantiles[2]-quantiles[0])/2], ["Posterior median dipole map", "Half-width of pointwise 68.27% interval"]):
            im = ax.pcolormesh(np.rad2deg(lon), np.rad2deg(lat), values, shading="auto", cmap="inferno")
            fig.colorbar(im, ax=ax, label="Area-scaled intensity [ppm]")
            ax.set(title=title, xlabel="Longitude [deg; positive = later maximum]", ylabel="Latitude [deg]")
        save(fig, directory, "map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=[*SITES, "all"], nargs="?", default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/surface_publication"))
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--plot-only", action="store_true", help="Rebuild figures from an existing run without sampling.")
    args = parser.parse_args()
    if args.draws < 4 or args.warmup < 1:
        parser.error("use at least four draws and one warmup step")
    reports = []
    for name in SITES if args.scenario == "all" else [args.scenario]:
        if args.plot_only:
            directory = args.output_dir / name
            with np.load(directory / "posterior.npz") as archive:
                samples = {key: value.reshape((-1,) + value.shape[2:]) for key, value in archive.items()}
            with np.load(directory / "lightcurves.npz") as data:
                plot_products(name, directory, data["wavelength_um"], data["time_bmjd_tdb"],
                              data["observed"], data["uncertainty"],
                              json.loads((directory / "truth.json").read_text()), samples, data["model_quantiles"])
            reports.append(json.loads((directory / "run.json").read_text()))
        else:
            reports.append(fit(name, args.output_dir, args.draws, args.warmup, args.seed))
        jax.clear_caches()
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
