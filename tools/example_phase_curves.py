#!/usr/bin/env python3
"""Generate the small synthetic datasets used by the surface-model examples.

The files use the six-HDU NIRSpec layout consumed by ``createdatacube.py``.
They are deliberately small enough for a laptop smoke test and deterministic
unless a different seed is requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits


PERIOD = 2.0
T0 = 60_000.0
A_RS = 6.0
B = 0.25
RPRS = 0.1
NOISE_PPM = 120.0


def _surface_signals(
    time, radius_ratios, model, channel_params=None, **surface_params
):
    """Evaluate the same JAXoplanet/Starry path used by the fitter."""
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    import jax
    # Absolute observation epochs near BMJD 60000 lose cadence precision in
    # float32. Match the production fitter without requiring an environment flag.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from models.jaxoplanet.surface import compute_surface_model

    spots = surface_params.pop("spots", ())
    channel_params = channel_params or {}
    names = tuple(channel_params)

    def evaluate(radius_ratio, *values):
        params = {
            "period": PERIOD,
            "t0": T0,
            "a_rs": A_RS,
            "b": B,
            "rors": radius_ratio,
            "ecc": 0.0,
            "omega": 0.0,
            "u": jnp.asarray([0.3, 0.2]),
            **surface_params,
            **dict(zip(names, values)),
        }
        return compute_surface_model(
            params, jnp.asarray(time), model=model, spots=spots
        )

    inputs = [jnp.asarray(radius_ratios)] + [
        jnp.asarray(channel_params[name]) for name in names
    ]
    if len(radius_ratios) == 1:
        scalar_inputs = [value[0] for value in inputs]
        return np.asarray(evaluate(*scalar_inputs))[:, None]
    return np.asarray(jax.jit(jax.vmap(evaluate))(*inputs)).T


def _scenario(name, wavelengths):
    if name == "eclipse":
        time = np.linspace(T0 + 0.35 * PERIOD, T0 + 0.65 * PERIOD, 161)
    elif name == "phase_curve":
        time = np.linspace(T0 - 0.05 * PERIOD, T0 + 1.05 * PERIOD, 321)
    elif name == "stellar_spots":
        time = np.linspace(T0 - 0.1 * PERIOD, T0 + 1.1 * PERIOD, 321)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(name)

    ntime = time.size
    nwave = wavelengths.size
    noiseless = np.empty((ntime, nwave))
    truth = {
        "scenario": name,
        "period_days": PERIOD,
        "t0_bmjd_tdb": T0,
        "a_rs": A_RS,
        "impact_parameter": B,
        "radius_ratio": RPRS,
        "noise_ppm_per_channel": NOISE_PPM,
        "wavelength_um": wavelengths.tolist(),
    }

    # The example fits fix geometry, so every channel must use that exact
    # configured radius; a transmission slope would bias surface recovery.
    radius_ratio = np.full(nwave, RPRS)
    if name == "eclipse":
        eclipse_depth_ppm = 700.0 + 180.0 * (wavelengths - wavelengths.min())
        noiseless[:] = 1.0 + _surface_signals(
            time,
            radius_ratio,
            "eclipse",
            channel_params={"eclipse_depth": eclipse_depth_ppm * 1e-6},
        )
        truth["eclipse_depth_ppm"] = eclipse_depth_ppm.tolist()
    elif name == "phase_curve":
        dayside_ppm = 850.0 + 160.0 * (wavelengths - wavelengths.min())
        nightside_ppm = np.full(nwave, 300.0)
        offset_deg = 18.0
        noiseless[:] = 1.0 + _surface_signals(
            time,
            radius_ratio,
            "phase_curve",
            channel_params={
                "dayside_flux": dayside_ppm * 1e-6,
                "nightside_flux": nightside_ppm * 1e-6,
            },
            hotspot_offset=np.deg2rad(offset_deg),
        )
        truth.update(
            dayside_flux_ppm=dayside_ppm.tolist(),
            nightside_flux_ppm=nightside_ppm.tolist(),
            hotspot_offset_deg=offset_deg,
        )
    else:
        spots_user = (
            dict(
                latitude_deg=15.0, longitude_deg=0.0,
                radius_deg=20.0, contrast=0.40,
            ),
        )
        spots_model = tuple(
            {
                "latitude": np.deg2rad(spot["latitude_deg"]),
                "longitude": np.deg2rad(spot["longitude_deg"]),
                "radius": np.deg2rad(spot["radius_deg"]),
                "contrast": spot["contrast"],
            }
            for spot in spots_user
        )
        spotted_curve = _surface_signals(
            time,
            radius_ratio[:1],
            "transit",
            spots=spots_model,
            stellar_rotation_period=2.0,
            stellar_spot_contrast=np.asarray([0.40]),
        )
        noiseless[:] = 1.0 + np.repeat(spotted_curve, nwave, axis=1)
        truth.update(rotation_period_days=2.0, spots=list(spots_user))
    return time, noiseless, truth


def _write_nirspec(path, time, wavelengths, flux, uncertainty):
    # The loader removes five detector-edge columns on either side. Keeping
    # explicit padding here makes this a faithful, minimal ExoTEDRF-style file.
    padded_wave = np.pad(wavelengths, 5, constant_values=np.nan)
    spacing = float(np.median(np.diff(wavelengths)))
    padded_wave_err = np.pad(
        np.full_like(wavelengths, 0.5 * spacing), 5, constant_values=np.nan
    )
    padded_flux = np.pad(flux, ((0, 0), (5, 5)), constant_values=np.nan)
    padded_uncertainty = np.pad(
        uncertainty, ((0, 0), (5, 5)), constant_values=np.nan
    )
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(padded_wave, name="WAVELENGTH"),
            fits.ImageHDU(padded_wave_err, name="WAVELENGTH_ERROR"),
            fits.ImageHDU(padded_flux, name="FLUX"),
            fits.ImageHDU(padded_uncertainty, name="FLUX_ERROR"),
            fits.ImageHDU(np.asarray(time), name="TIME"),
        ]
    )
    hdus.writeto(path, overwrite=True)


def generate(name, output_dir, rng):
    wavelengths = np.linspace(2.9, 5.0, 14)
    time, noiseless, truth = _scenario(name, wavelengths)
    uncertainty = np.full_like(noiseless, NOISE_PPM * 1e-6)
    flux = noiseless + rng.normal(scale=uncertainty)
    fits_path = output_dir / f"synthetic_{name}.fits"
    truth_path = output_dir / f"synthetic_{name}_truth.json"
    _write_nirspec(fits_path, time, wavelengths, flux, uncertainty)
    truth["fits_file"] = fits_path.name
    truth_path.write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {fits_path}")
    print(f"wrote {truth_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=("all", "eclipse", "phase_curve", "stellar_spots"),
        default="all",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/data/generated")
    )
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        ("eclipse", "phase_curve", "stellar_spots")
        if args.scenario == "all"
        else (args.scenario,)
    )
    rng = np.random.default_rng(args.seed)
    for name in names:
        generate(name, args.output_dir, rng)


if __name__ == "__main__":
    main()
