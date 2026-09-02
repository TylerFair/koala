#!/usr/bin/env python3
"""Compare existing nearest-based LD caches to trilinear recomputations on exact Rreference grids.

Run this on the VM where the full stellar model data and run directories exist.

Example
-------
python tyler_scripts/audit_ld_interpolation_rreference.py \
  --run-glob "/scratch/midway3/tfairnington/*STELLARINFORMEDLD*" \
  --config-dir tyler_scripts/configs_fiducial_stellarinformed \
  --fit-jwst-path ../JWSTJaxFit-main/fit_jwst.py \
  --output-dir tyler_scripts/LD_INTERPOLATION_AUDIT_VM \
  --include-whitelight
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CacheAuditRow:
    planet: str
    run_dir: str
    cache_file: str
    kind: str
    instrument: str
    order: int | None
    n_bins: int
    teff_sigma: float
    logg_sigma: float
    feh_sigma: float
    nearest_c1_min: float
    nearest_c2_min: float
    nearest_c1_median: float
    nearest_c2_median: float
    trilinear_c1_min: float
    trilinear_c2_min: float
    trilinear_c1_median: float
    trilinear_c2_median: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-glob", required=True, help="Quoted glob for run directories.")
    parser.add_argument(
        "--config-dir",
        default="tyler_scripts/configs_fiducial_stellarinformed",
        help="Directory containing YAML configs keyed by output_dir.",
    )
    parser.add_argument(
        "--fit-jwst-path",
        default="../JWSTJaxFit-main/fit_jwst.py",
        help="Path to fit_jwst.py so this audit reuses the exact JWSTJaxFit LD logic.",
    )
    parser.add_argument(
        "--output-dir",
        default="tyler_scripts/LD_INTERPOLATION_AUDIT_VM",
        help="Directory for CSV and plots.",
    )
    parser.add_argument(
        "--include-whitelight",
        action="store_true",
        help="Also include whitelight cache files in addition to Rreference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Reference sigma threshold to annotate in summary plots.",
    )
    return parser.parse_args()


def load_fit_module(path: Path):
    fit_parent = str(path.resolve().parent)
    if fit_parent not in sys.path:
        sys.path.insert(0, fit_parent)
    spec = importlib.util.spec_from_file_location("fit_jwst_for_ld_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load fit_jwst module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_config_index(config_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        outdir = data.get("output_dir")
        if outdir:
            index[str(outdir)] = data
    return index


def locate_spectro_pickle(run_path: Path) -> Path:
    matches = sorted(run_path.glob("*_spectroscopy_data_*referenceHR.pkl"))
    if not matches:
        raise FileNotFoundError(f"No spectroscopy_data_*referenceHR.pkl found in {run_path}")
    return matches[0]


def get_wavelength_inputs(fitmod, run_path: Path, cache_df: pd.DataFrame, kind: str):
    if kind == "whitelight":
        spectro_path = locate_spectro_pickle(run_path)
        data = fitmod.SpectroData.load(spectro_path)
        return np.asarray(data.wavelengths_unbinned, dtype=float), 0.0
    return cache_df["wavelength"].to_numpy(float), cache_df["wavelength_err"].to_numpy(float)


def compute_ld_prior(
    fitmod,
    config: dict[str, Any],
    run_path: Path,
    cache_df: pd.DataFrame,
    kind: str,
    interpolate_type: str,
):
    stellar_cfg = config["stellar"]
    instrument = config["instrument"]
    order = config.get("order")
    if isinstance(order, str):
        order = None
    order = int(order) if order is not None else None

    wavelengths, wavelength_err = get_wavelength_inputs(fitmod, run_path, cache_df, kind)

    ld_prior_model = stellar_cfg.get("ld_prior_model", stellar_cfg.get("ld_model", "stagger"))
    ld_data_path = stellar_cfg.get("ld_data_path", "../exotic_ld_data")
    teff = float(stellar_cfg["teff"])
    logg = float(stellar_cfg["logg"])
    feh = float(stellar_cfg["feh"])
    teff_sigma = float(stellar_cfg.get("teff_sigma", 0.0))
    logg_sigma = float(stellar_cfg.get("logg_sigma", 0.0))
    feh_sigma = float(stellar_cfg.get("feh_sigma", 0.0))
    n_grid = int(stellar_cfg.get("ld_prior_n_grid", 5))
    nsigma = float(stellar_cfg.get("ld_prior_nsigma", 3.0))

    teff_axis = fitmod._build_axis(teff, teff_sigma, n_grid, nsigma)
    logg_axis = fitmod._build_axis(logg, logg_sigma, n_grid, nsigma)
    feh_axis = fitmod._build_axis(feh, feh_sigma, n_grid, nsigma)

    combos = []
    for mh_i in feh_axis:
        for teff_i in teff_axis:
            for logg_i in logg_axis:
                weight = (
                    fitmod._gaussian_pdf(mh_i, feh, feh_sigma)
                    * fitmod._gaussian_pdf(teff_i, teff, teff_sigma)
                    * fitmod._gaussian_pdf(logg_i, logg, logg_sigma)
                )
                combos.append((mh_i, teff_i, logg_i, weight))

    successful_mu = []
    weights = []
    for mh_i, teff_i, logg_i, weight in combos:
        sld_grid = fitmod.StellarLimbDarkening(
            M_H=mh_i,
            Teff=teff_i,
            logg=logg_i,
            ld_model=ld_prior_model,
            ld_data_path=ld_data_path,
            interpolate_type=interpolate_type,
            verbose=0,
        )
        mu_i = fitmod.get_limb_darkening(
            sld_grid,
            wavelengths,
            wavelength_err,
            instrument,
            order=order,
            ld_profile="power2",
            return_sigmas=False,
        )
        mu_i = np.asarray(mu_i, dtype=float)
        if mu_i.ndim == 1:
            mu_i = mu_i[None, :]
        successful_mu.append(mu_i)
        weights.append(weight)

    mu_stack = np.stack(successful_mu, axis=0)
    weights = np.asarray(weights, dtype=float)
    weights /= np.sum(weights)

    c1_mean, _, _, c1_star = fitmod._combine_weighted_means_and_sigmas(mu_stack[:, :, 0], weights)
    c2_mean, _, _, c2_star = fitmod._combine_weighted_means_and_sigmas(mu_stack[:, :, 1], weights)
    return {
        "c1_mean": np.asarray(c1_mean, dtype=float),
        "c2_mean": np.asarray(c2_mean, dtype=float),
        "c1_sigma_star": np.asarray(c1_star, dtype=float),
        "c2_sigma_star": np.asarray(c2_star, dtype=float),
    }


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 220,
            "font.family": "serif",
            "axes.linewidth": 1.35,
            "axes.labelsize": 14,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.frameon": False,
        }
    )


def make_histogram(df: pd.DataFrame, outdir: Path, threshold: float) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    max_val = max(
        float(df["nearest_c1_min"].max()),
        float(df["trilinear_c1_min"].max()),
        float(df["nearest_c2_min"].max()),
        float(df["trilinear_c2_min"].max()),
        threshold,
    )
    bins = np.linspace(0.0, max(0.02, max_val * 1.05), 28)
    for ax, coeff in zip(axes, ["c1", "c2"], strict=True):
        ax.hist(
            df[f"nearest_{coeff}_min"],
            bins=bins,
            histtype="stepfilled",
            alpha=0.35,
            color="#7b1fa2",
            label="Existing nearest cache",
        )
        ax.hist(
            df[f"trilinear_{coeff}_min"],
            bins=bins,
            histtype="step",
            lw=2.0,
            color="#00897b",
            label="Trilinear recompute",
        )
        ax.axvline(threshold, color="0.35", ls="--", lw=1.2)
        ax.set_title(rf"Min $\sigma_{{{coeff},\star}}$")
        ax.set_xlabel(rf"$\min(\sigma_{{{coeff},\star}})$")
        ax.set_ylabel("Cache files")
        ax.legend(loc="upper right", fontsize=9.5)
    fig.suptitle("Existing Nearest LD Caches vs Trilinear Recompute", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = outdir / "ld_interpolation_rreference_histograms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def classify_cache(cache_name: str) -> str | None:
    if "Rreference" in cache_name:
        return "Rreference"
    if "whitelight" in cache_name:
        return "whitelight"
    return None


def summarize(df: pd.DataFrame, threshold: float) -> None:
    print("cache files audited:", len(df))
    for coeff in ["c1", "c2"]:
        near = df[f"nearest_{coeff}_min"]
        tri = df[f"trilinear_{coeff}_min"]
        print()
        print(f"{coeff} min sigma_star")
        print(f"  nearest median:   {np.median(near):.5f}")
        print(f"  trilinear median: {np.median(tri):.5f}")
        print(f"  nearest < {threshold:g}:   {(near < threshold).sum()} / {len(near)}")
        print(f"  trilinear < {threshold:g}: {(tri < threshold).sum()} / {len(tri)}")


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fitmod = load_fit_module(Path(args.fit_jwst_path).resolve())
    config_index = build_config_index(Path(args.config_dir))

    rows: list[CacheAuditRow] = []
    for run_path_str in sorted(glob.glob(args.run_glob)):
        run_path = Path(run_path_str)
        config = config_index.get(run_path.name)
        if config is None:
            print(f"[skip] no config matched output_dir={run_path.name}")
            continue
        cache_dir = run_path / "ld_prior_cache"
        if not cache_dir.is_dir():
            print(f"[skip] no ld_prior_cache in {run_path}")
            continue
        for cache_path in sorted(cache_dir.glob("*.csv")):
            kind = classify_cache(cache_path.name)
            if kind is None:
                continue
            if kind == "whitelight" and not args.include_whitelight:
                continue
            cache_df = pd.read_csv(cache_path)
            required = {"wavelength", "wavelength_err", "c1_sigma_star", "c2_sigma_star", "c1_mean", "c2_mean"}
            if not required.issubset(cache_df.columns):
                print(f"[skip] missing required columns in {cache_path}")
                continue
            tri = compute_ld_prior(fitmod, config, run_path, cache_df, kind, "trilinear")
            rows.append(
                CacheAuditRow(
                    planet=config["planet"]["name"],
                    run_dir=run_path.name,
                    cache_file=cache_path.name,
                    kind=kind,
                    instrument=config["instrument"],
                    order=int(config.get("order")) if config.get("order") not in (None, "#") else None,
                    n_bins=len(cache_df),
                    teff_sigma=float(config["stellar"].get("teff_sigma", 0.0)),
                    logg_sigma=float(config["stellar"].get("logg_sigma", 0.0)),
                    feh_sigma=float(config["stellar"].get("feh_sigma", 0.0)),
                    nearest_c1_min=float(cache_df["c1_sigma_star"].min()),
                    nearest_c2_min=float(cache_df["c2_sigma_star"].min()),
                    nearest_c1_median=float(cache_df["c1_sigma_star"].median()),
                    nearest_c2_median=float(cache_df["c2_sigma_star"].median()),
                    trilinear_c1_min=float(np.min(tri["c1_sigma_star"])),
                    trilinear_c2_min=float(np.min(tri["c2_sigma_star"])),
                    trilinear_c1_median=float(np.median(tri["c1_sigma_star"])),
                    trilinear_c2_median=float(np.median(tri["c2_sigma_star"])),
                )
            )

    df = pd.DataFrame([r.__dict__ for r in rows]).sort_values(["planet", "run_dir", "cache_file"])
    csv_path = outdir / "ld_interpolation_rreference_summary.csv"
    df.to_csv(csv_path, index=False)
    hist_path = make_histogram(df, outdir, args.threshold)
    summarize(df, args.threshold)
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()

