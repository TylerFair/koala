#!/usr/bin/env python3
"""Build manuscript-ready light-curve tables and overview figures.

The final planet sample is intentionally hard-coded.  Input metadata come from
configs_fiducial_stellarinformed and fit products are read from STELLARINFORMED.
No files under scratch are modified.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


FINAL_PLANETS = (
    "HAT-P-11", "HAT-P-12", "HAT-P-18", "HAT-P-26", "HAT-P-30",
    "HAT-P-65", "KELT-7", "Kepler-12", "NGTS-10", "NGTS-2",
    "WASP-107", "WASP-121", "WASP-127", "WASP-166", "WASP-17",
    "WASP-39", "WASP-52", "WASP-63", "WASP-69", "WASP-94", "WASP-96",
)

PLANET_RANK = {name: i for i, name in enumerate(FINAL_PLANETS)}


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"),
                     ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def fmt(value: object, digits: int = 6) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(x):
        return "--"
    return f"{x:.{digits}g}"


def config_planet(cfg: dict) -> str:
    return str(cfg.get("planet", {}).get("name", ""))


def product_suffix(config_path: Path) -> str:
    match = re.search(r"_v([0-9]+)_config$", config_path.stem, re.I)
    return f"_V{match.group(1)}" if match else ""


def choose_product(directory: Path, pattern: str, suffix: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    if suffix:
        matching = [p for p in files if p.stem.endswith(suffix)]
        if matching:
            return matching[0]
    plain = [p for p in files if not re.search(r"_V[0-9]+$", p.stem)]
    return plain[0] if plain else (files[0] if files else None)


def dataset_label(cfg: dict, config_path: Path) -> str:
    instrument = str(cfg.get("instrument", "unknown"))
    detector = cfg.get("nrs")
    order = cfg.get("order")
    pieces = [instrument]
    if instrument.upper().startswith("NIRSPEC") and detector not in (None, ""):
        pieces.append(f"NRS{detector}")
    if "SOSS" in instrument.upper() and order not in (None, ""):
        pieces.append(f"order {order}")
    suffix = product_suffix(config_path)
    if suffix:
        pieces.append(f"visit {suffix[2:]}")
    return ", ".join(pieces)


def discover(config_dir: Path, scratch_root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    found_planets: set[str] = set()
    for config_path in sorted(config_dir.glob("*_config.yaml")):
        with config_path.open() as handle:
            cfg = yaml.safe_load(handle) or {}
        planet = config_planet(cfg)
        if planet not in PLANET_RANK:
            continue
        found_planets.add(planet)
        output_name = str(cfg.get("output_dir", ""))
        directory = scratch_root / output_name
        suffix = product_suffix(config_path)
        params = choose_product(directory, "*whitelight_bestfit_params*.csv", suffix) if directory.is_dir() else None
        timeseries = choose_product(directory, "*whitelight_timeseries*.csv", suffix) if directory.is_dir() else None
        lowres = choose_product(directory, "*_R20.csv", suffix) if directory.is_dir() else None
        status = "complete" if (directory / (suffix + ".done" if suffix else ".done")).exists() else "incomplete"
        if not directory.is_dir():
            status = "missing output directory"
        elif params is None or timeseries is None:
            status = "missing white-light products"
        rows.append({
            "planet": planet,
            "dataset": dataset_label(cfg, config_path),
            "instrument": cfg.get("instrument", ""),
            "detector": f"NRS{cfg['nrs']}" if str(cfg.get("instrument", "")).upper().startswith("NIRSPEC") and cfg.get("nrs") not in (None, "") else "",
            "order": (cfg.get("order", "") or "") if "SOSS" in str(cfg.get("instrument", "")).upper() else "",
            "visit": suffix.removeprefix("_V"),
            "detrending": cfg.get("flags", {}).get("detrending_type", ""),
            "white_clip_sigma": cfg.get("outlier_clip", {}).get("whitelight_sigma", ""),
            "spectroscopic_clip_sigma": cfg.get("outlier_clip", {}).get("spectroscopic_sigma", ""),
            "config": str(config_path),
            "output_directory": str(directory),
            "status": status,
            "params_file": str(params) if params else "",
            "timeseries_file": str(timeseries) if timeseries else "",
            "lowres_spectrum_file": str(lowres) if lowres else "",
        })
    rows.sort(key=lambda r: (PLANET_RANK[r["planet"]], r["dataset"]))
    missing = [p for p in FINAL_PLANETS if p not in found_planets]
    return rows, missing


def read_first_row(path: str) -> dict:
    if not path:
        return {}
    with open(path, newline="") as handle:
        return next(csv.DictReader(handle), {})


def build_parameter_rows(datasets: list[dict]) -> list[dict]:
    output = []
    columns = (
        "period", "duration", "duration_err_low", "duration_err_high", "t0",
        "t0_err_low", "t0_err_high", "b", "b_err_low", "b_err_high", "rors",
        "rors_err_low", "rors_err_high", "a_rs", "a_rs_err_low", "a_rs_err_high",
        "ecc", "omega",
    )
    for dataset in datasets:
        row = read_first_row(dataset["params_file"])
        if not row:
            continue
        result = {"planet": dataset["planet"], "dataset": dataset["dataset"]}
        result.update({key: row.get(key, "") for key in columns})
        output.append(result)
    return output


def build_quality_rows(datasets: list[dict]) -> list[dict]:
    output = []
    for dataset in datasets:
        path = dataset["timeseries_file"]
        if not path:
            continue
        data = pd.read_csv(path)
        residual_ppm = pd.to_numeric(data.get("residual_ppm"), errors="coerce")
        mask = pd.to_numeric(data.get("is_outlier", 0), errors="coerce").fillna(0).astype(bool)
        good = residual_ppm[~mask & np.isfinite(residual_ppm)]
        cadence = np.nanmedian(np.diff(pd.to_numeric(data["time_bjd"], errors="coerce"))) * 86400
        output.append({
            "planet": dataset["planet"], "dataset": dataset["dataset"],
            "n_cadences": len(data), "n_white_outliers": int(mask.sum()),
            "white_outlier_fraction": float(mask.mean()), "cadence_seconds": cadence,
            "residual_rms_ppm": float(np.sqrt(np.mean(good**2))),
            "residual_mad_ppm": float(1.4826 * np.median(np.abs(good - np.median(good)))),
        })
    return output


def write_dataset_tex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{longtable}{llll}",
        r"\caption{Light-curve datasets and baseline models.}\label{tab:lightcurve_baselines}\\",
        r"Planet & Instrument & Detector/order/visit & Baseline \\",
        r"\hline",
        r"\endfirsthead",
        r"Planet & Instrument & Detector/order/visit & Baseline \\",
        r"\hline",
        r"\endhead",
    ]
    for row in rows:
        detail = ", ".join(x for x in (row["detector"], f"order {row['order']}" if row["order"] else "", f"visit {row['visit']}" if row["visit"] else "") if x) or "--"
        lines.append(" & ".join(latex_escape(x) for x in (row["planet"], row["instrument"], detail, row["detrending"])) + r" \\")
    lines.extend((r"\hline", r"\end{longtable}"))
    path.write_text("\n".join(lines) + "\n")


def asym(value: dict, key: str, digits: int = 6) -> str:
    centre = fmt(value.get(key), digits)
    lo = fmt(value.get(key + "_err_low"), 2)
    hi = fmt(value.get(key + "_err_high"), 2)
    return rf"${centre}_{{-{lo}}}^{{+{hi}}}$"


def write_parameters_tex(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{longtable}{llcccccc}",
        r"\caption{White-light transit parameters. Times are in the time system of the input fit products and durations are in days.}\label{tab:system-parameters}\\",
        r"Planet & Dataset & $P$ (d) & $t_0$ & $T_{14}$ (d) & $b$ & $R_{\rm p}/R_\star$ & $a/R_\star$ \\",
        r"\hline", r"\endfirsthead",
        r"Planet & Dataset & $P$ (d) & $t_0$ & $T_{14}$ (d) & $b$ & $R_{\rm p}/R_\star$ & $a/R_\star$ \\",
        r"\hline", r"\endhead",
    ]
    for row in rows:
        fields = (latex_escape(row["planet"]), latex_escape(row["dataset"]), fmt(row["period"], 9),
                  asym(row, "t0", 11), asym(row, "duration", 7), asym(row, "b", 5),
                  asym(row, "rors", 7), asym(row, "a_rs", 7))
        lines.append(" & ".join(fields) + r" \\")
    lines.extend((r"\hline", r"\end{longtable}"))
    path.write_text("\n".join(lines) + "\n")


def write_priors_tex(path: Path) -> None:
    """Write the priors used by models/jaxoplanet/builder.py."""
    lines = [
        r"\begin{table*}[p]", r"\centering", r"\scriptsize",
        r"\caption{Light-curve parameter priors. Here $\mathcal{U}(a,b)$ and $\mathcal{N}(\mu,\sigma)$ denote uniform and normal distributions, and $\mathcal{TN}(\mu,\sigma;a,b)$ is a normal distribution truncated to $[a,b]$. The period, eccentricity, and argument of periastron were fixed to the adopted values. Spectroscopic $t_0$, $b$, and $T_{14}$ (or $a/R_\star$) were fixed to the white-light solution. Baseline parameters were included only for datasets using the corresponding model in Supplementary Table~\ref{tab:lightcurve_baselines}.}",
        r"\label{tab:lightcurve_priors}",
        r"\begin{tabular}{llll}", r"\hline",
        r"Parameter & White-light prior & Spectroscopic prior & Notes \\", r"\hline",
        r"$P$ & Fixed & Fixed & Adopted literature period \\",
        r"$t_0$ & $\mathcal{U}(t_{\min},t_{\max})$ & Fixed & Per visit \\",
        r"$R_{\rm p}/R_\star$ & $\mathcal{U}(10^{-3},\sqrt{0.5})$ & $\mathcal{U}(\sqrt{10^{-5}},\sqrt{0.5})$ & Independently fitted per channel \\",
        r"Auxiliary $\tilde b$ & $\mathcal{U}(-2,2)$ & Fixed & $b=|\tilde b|$ \\",
        r"$\log T_{14}$ & $\mathcal{U}[\log(7\times10^{-4}),\log(1)]$ & Fixed & Duration parameterization; days \\",
        r"$\log(a/R_\star)$ & $\mathcal{U}[\log a_{\min},\log a_{\max}]$ & Fixed & Used only for $a/R_\star$ parameterization \\",
        r"$c_1$ & $\mathcal{TN}(c_{1,\star},\sigma_{c_1};0,1)$ & Same, wavelength dependent & Stellar-informed power-2 LD \\",
        r"$c_2$ & $\mathcal{TN}(c_{2,\star},\sigma_{c_2};0.001,1)$ & Same, wavelength dependent & Stellar-informed power-2 LD \\",
        r"$\log\sigma_{\rm jit}$ & $\mathcal{U}[\log(10^{-5}),\log(10^{-2})]$ & $\mathcal{U}[\log(10^{-6}),\log(1)]$ & Relative-flux units \\",
        r"$c$ & $\mathcal{U}(0.9,1.1)$ & $\mathcal{U}(0.9,1.1)$ & Baseline intercept \\",
        r"$v,v_2,v_3,v_4$ & $\mathcal{U}(-0.1,0.1)$ & $\mathcal{U}(-0.1,0.1)$ & Terms included as required \\",
        r"$A$ & $\mathcal{U}(-0.1,0.1)$ & $\mathcal{U}(-0.1,0.1)$ & Exponential-ramp amplitude \\",
        r"$\log\tau$ & $\mathcal{U}[\log(10^{-3}),\log(10^{-1})]$ & Same & Exponential-ramp timescale; days \\",
        r"$t_{\rm jump}$ & $\mathcal{N}(t_{\rm jump,0},0.01)$ & Fixed template & Discontinuity model \\",
        r"$\Delta_{\rm jump}$ & $\mathcal{N}(\Delta_0,0.01)$ & Fixed template & Discontinuity model \\",
        r"$A_{\rm jump}$ & -- & $\mathcal{U}(0.5,2)$ & Spectroscopic template scaling \\",
        r"$A_{\rm spot}$ & $\mathcal{U}(0,0.1)$ & $\mathcal{U}(0.5,2)$ & White-light amplitude / spectral scaling \\",
        r"$t_{\rm spot}$ & $\mathcal{N}(t_{\rm spot,0},0.01)$ & Fixed template & Spot-crossing centre; days \\",
        r"$\sigma_{\rm spot}$ & $\mathcal{U}(10^{-4},0.1)$ & Fixed template & Spot-crossing width; days \\",
        r"$A_{\rm spot,2},t_{\rm spot,2},\sigma_{\rm spot,2}$ & As for first spot & Template scaling $\mathcal{U}(0.5,2)$ & Two-spot datasets only \\",
        r"\hline", r"\end{tabular}", r"\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_quality(rows: list[dict], outdir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    height = max(8, 0.27 * len(df))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, height), sharey=True,
                                  gridspec_kw={"width_ratios": [1.45, 1]})
    ax1.scatter(df["residual_rms_ppm"], y, s=18, color="#176d9c")
    ax2.scatter(100 * df["white_outlier_fraction"], y, s=18, color="#c44e52")
    ax1.set_yticks(y, [f"{p} — {d}" for p, d in zip(df.planet, df.dataset)], fontsize=6.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("White-light residual RMS (ppm)")
    ax2.set_xlabel("Masked white-light cadences (%)")
    for ax in (ax1, ax2):
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "lightcurve_quality_overview.pdf", bbox_inches="tight")
    fig.savefig(outdir / "lightcurve_quality_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_atlas(datasets: list[dict], outdir: Path) -> None:
    representatives = {}
    for row in datasets:
        if row["timeseries_file"] and row["planet"] not in representatives:
            representatives[row["planet"]] = row
    planets = [p for p in FINAL_PLANETS if p in representatives]
    if not planets:
        return
    fig, axes = plt.subplots(7, 3, figsize=(10, 14), squeeze=False)
    for ax, planet in zip(axes.flat, planets):
        row = representatives[planet]
        data = pd.read_csv(row["timeseries_file"])
        x = pd.to_numeric(data["time_from_t0_hr"], errors="coerce")
        y = pd.to_numeric(data["detrended_flux"], errors="coerce")
        model = pd.to_numeric(data["transit_model"], errors="coerce")
        mask = pd.to_numeric(data.get("is_outlier", 0), errors="coerce").fillna(0).astype(bool)
        ax.scatter(x[~mask], y[~mask], s=1.2, alpha=0.45, color="#4c72b0", rasterized=True)
        ax.plot(x, model, lw=1.1, color="#c44e52")
        if mask.any():
            ax.scatter(x[mask], y[mask], s=7, marker="x", color="black", linewidths=0.5)
        ax.set_title(planet, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in axes.flat[len(planets):]:
        ax.axis("off")
    fig.supxlabel("Time from fitted mid-transit (h)", fontsize=11)
    fig.supylabel("Detrended normalized flux", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "white_light_atlas.pdf", bbox_inches="tight")
    fig.savefig(outdir / "white_light_atlas.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs_fiducial_stellarinformed"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/scratch/midway3/tfairnington/STELLARINFORMED"))
    parser.add_argument("--output-dir", type=Path, default=Path("supplementary_lightcurve_products"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets, missing = discover(args.config_dir, args.scratch_root)
    parameters = build_parameter_rows(datasets)
    quality = build_quality_rows(datasets)
    pd.DataFrame(datasets).to_csv(args.output_dir / "lightcurve_dataset_inventory.csv", index=False)
    pd.DataFrame(parameters).to_csv(args.output_dir / "white_light_parameters.csv", index=False)
    pd.DataFrame(quality).to_csv(args.output_dir / "lightcurve_quality_metrics.csv", index=False)
    write_dataset_tex(datasets, args.output_dir / "lightcurve_baselines.tex")
    write_parameters_tex(parameters, args.output_dir / "system_parameters.tex")
    write_priors_tex(args.output_dir / "lightcurve_priors.tex")
    plot_quality(quality, args.output_dir)
    plot_atlas(datasets, args.output_dir)

    complete = sum(row["status"] == "complete" for row in datasets)
    report = [
        "# Light-curve supplementary-product build report", "",
        f"- Final planet list: {len(FINAL_PLANETS)} planets",
        f"- Planets represented by light-curve configs: {len(set(r['planet'] for r in datasets))}",
        f"- Configured datasets: {len(datasets)} ({complete} marked complete)",
        f"- White-light parameter rows: {len(parameters)}",
        f"- Quality-metric rows: {len(quality)}",
        f"- Missing planets: {', '.join(missing) if missing else 'none'}", "",
        "The inventory is the authoritative audit trail for input paths and completion status.",
        "The atlas uses the first available configured dataset per planet only; all datasets are retained in the tables.",
    ]
    (args.output_dir / "BUILD_REPORT.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
