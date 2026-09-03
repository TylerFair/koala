#!/usr/bin/env python3
"""Compare matched stage-replay samples for an LD coordinate change."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def scalar_channels(value):
    value = np.asarray(value)
    if value.ndim > 2:
        value = value.reshape(value.shape[0], value.shape[1], -1)[..., 0]
    return value


def ess_by_channel(value):
    value = scalar_channels(value)
    dataset = az.ess(az.from_dict(posterior={"value": value[None]}))
    return np.asarray(dataset["value"], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("dump")
    parser.add_argument("--gate-json")
    parser.add_argument("--figure")
    parser.add_argument("--title", default="LD coordinate comparison")
    parser.add_argument(
        "--candidate-label",
        default="Laplace NUTS (Jacobian-corrected coordinates)",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = load_npz(args.reference)
    candidate = load_npz(args.candidate)
    with Path(args.dump).open("rb") as stream:
        wavelengths = np.asarray(pickle.load(stream)["meta"]["wavelength"])

    components = [(site, site, None) for site in ("depths", "c1", "c2")
                  if site in reference and site in candidate]
    if "u" in reference and "u" in candidate:
        components.extend((("u1", "u", 0), ("u2", "u", 1)))
    rows = []
    summaries = {}
    for site, source, component in components:
        ref = (scalar_channels(reference[source]) if component is None else
               np.asarray(reference[source])[..., component])
        cand = (scalar_channels(candidate[source]) if component is None else
                np.asarray(candidate[source])[..., component])
        ref_median = np.median(ref, axis=0)
        cand_median = np.median(cand, axis=0)
        ref_sigma = np.std(ref, axis=0, ddof=1)
        cand_sigma = np.std(cand, axis=0, ddof=1)
        pooled = np.sqrt((ref_sigma ** 2 + cand_sigma ** 2) / 2.0)
        shifts = np.abs(cand_median - ref_median) / pooled
        ratios = cand_sigma / ref_sigma
        ess = ess_by_channel(cand)
        summaries[site] = {
            "median_shift_p50_p95_max_pooled_sigma": [
                float(np.percentile(shifts, q)) for q in (50, 95, 100)
            ],
            "sigma_ratio_p05_p50_p95": [
                float(np.percentile(ratios, q)) for q in (5, 50, 95)
            ],
            "ess_min_median": [float(np.min(ess)), float(np.median(ess))],
        }
        rows.extend({
            "site": site, "channel": int(index),
            "abs_median_shift_pooled_sigma": float(shifts[index]),
            "sigma_ratio": float(ratios[index]), "ess": float(ess[index]),
        } for index in range(shifts.size))

    ref_depth = scalar_channels(reference["depths"])
    cand_depth = scalar_channels(candidate["depths"])
    ref_median = np.median(ref_depth, axis=0)
    cand_median = np.median(cand_depth, axis=0)
    ref_sigma = np.std(ref_depth, axis=0, ddof=1)
    cand_sigma = np.std(cand_depth, axis=0, ddof=1)
    delta_ppm = (cand_median - ref_median) * 1e6
    weight = 1.0 / np.maximum(ref_sigma * 1e6, 1e-30) ** 2
    centered_wave = wavelengths - np.average(wavelengths, weights=weight)
    design = np.column_stack((np.ones(wavelengths.size), centered_wave))
    covariance = np.linalg.inv(design.T @ (weight[:, None] * design))
    beta = covariance @ design.T @ (weight * delta_ppm)
    residual = delta_ppm - beta[0]
    ref_ess = ess_by_channel(ref_depth)
    cand_ess = ess_by_channel(cand_depth)
    median_mc_sigma_ppm = np.sqrt(np.pi / 2.0 * (
        ref_sigma ** 2 / ref_ess + cand_sigma ** 2 / cand_ess
    )) * 1e6
    spectrum = {
        "weighted_offset_ppm": float(beta[0]),
        "weighted_slope_ppm_per_um": float(beta[1]),
        "slope_uncertainty_ppm_per_um": float(np.sqrt(covariance[1, 1])),
        "rms_after_offset_ppm": float(np.sqrt(np.mean(residual ** 2))),
        "median_mc_rms_expectation_ppm": float(np.sqrt(np.mean(median_mc_sigma_ppm ** 2))),
        "error_ratio_p05_p50_p95": [float(np.percentile(cand_sigma / ref_sigma, q))
                                      for q in (5, 50, 95)],
    }

    if args.figure:
        figure_path = Path(args.figure)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        ref_ppm = ref_median * 1e6
        cand_ppm = cand_median * 1e6
        ref_error_ppm = ref_sigma * 1e6
        cand_error_ppm = cand_sigma * 1e6
        difference_error_ppm = np.sqrt(ref_sigma ** 2 + cand_sigma ** 2) * 1e6
        fig, (upper, lower) = plt.subplots(
            2, 1, figsize=(9.0, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
        )
        upper.errorbar(wavelengths, ref_ppm, yerr=ref_error_ppm, fmt="o",
                       ms=2.8, lw=0.7, capsize=0, alpha=0.75,
                       label="Legacy adaptive joint NUTS (coefficients)")
        upper.errorbar(wavelengths, cand_ppm, yerr=cand_error_ppm, fmt="o",
                       ms=2.8, lw=0.7, capsize=0, alpha=0.75,
                       label=args.candidate_label)
        upper.set_ylabel("Transit depth (ppm)")
        upper.set_title(args.title)
        upper.legend(fontsize=8)
        lower.axhline(0.0, color="0.25", lw=0.8)
        lower.errorbar(wavelengths, delta_ppm, yerr=difference_error_ppm,
                       fmt="o", ms=2.8, lw=0.7, capsize=0)
        lower.set_xlabel("Wavelength (µm)")
        lower.set_ylabel("New − legacy\n(ppm)")
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    gates = None
    if args.gate_json:
        gate_payload = json.loads(Path(args.gate_json).read_text())
        gates = gate_payload.get("calibrated_gates", [])
    output = {"summaries": summaries, "spectrum": spectrum,
              "calibrated_gates": gates, "rows": rows}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"summaries": summaries, "spectrum": spectrum}, indent=2))


if __name__ == "__main__":
    main()
