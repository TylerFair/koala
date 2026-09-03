#!/usr/bin/env python3
"""Compare the matched legacy and wide-u+/u- quadratic-uniform runs."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.spectro_stage_inputs import load_stage_inputs

ROOT = Path("/scratch/midway3/tfairnington/accel_gpu_results")
LEGACY = ROOT / "432_ld_uniform_legacy_coefficients_v2/legacy_coefficients"
WIDE = ROOT / "433_ld_uniform_uplus_uminus_v2/uplus_uminus"
DUMP = ROOT / ("348_uniform_quadratic_dump_retry2/stage_inputs/"
               "WASP-39_NIRSPEC_G395H_nrs1_Rreference_high_resolution_inputs.pkl")
FIGURE = Path("acceleration_reports/ld_uplus_uminus_vs_legacy_WASP-39_G395H.png")
SUMMARY = Path("acceleration_reports/ld_uplus_uminus_vs_legacy_WASP-39_G395H.json")


def _depth(path):
    with path.open("rb") as stream:
        samples = pickle.load(stream)
    return np.asarray(samples["rors"], dtype=float)[..., 0] ** 2


def _depth_ess(path):
    rows = json.loads(path.read_text())["rows"]
    values = np.full(68, np.nan)
    for row in rows:
        if row["site"] == "rors[0]":
            values[int(row["channel"])] = float(row["ess_bulk"])
    if not np.all(np.isfinite(values)):
        raise ValueError("missing per-channel depth ESS")
    return values


def main():
    if FIGURE.exists() or SUMMARY.exists():
        raise FileExistsError("comparison outputs already exist")
    legacy = _depth(Path(f"{LEGACY}.pkl"))
    wide = _depth(Path(f"{WIDE}.pkl"))
    wave = np.asarray(load_stage_inputs(DUMP).meta["wavelength"], dtype=float)
    l16, lmed, l84 = np.percentile(legacy, [16, 50, 84], axis=0)
    w16, wmed, w84 = np.percentile(wide, [16, 50, 84], axis=0)
    lsig, wsig = (l84-l16)/2, (w84-w16)/2
    difference = wmed-lmed
    variance = lsig**2 + wsig**2
    weight = 1/variance
    x = wave - np.average(wave, weights=weight)
    offset = np.sum(weight*difference)/np.sum(weight)
    slope = np.sum(weight*x*(difference-offset))/np.sum(weight*x*x)
    residual = difference-offset-slope*x
    rms = np.sqrt(np.mean(residual**2))
    less, wess = (_depth_ess(Path(f"{LEGACY}.arviz.json")),
                  _depth_ess(Path(f"{WIDE}.arviz.json")))
    # Normal-posterior approximation for the Monte Carlo error of a median.
    mcse = np.sqrt((1.253314*lsig/np.sqrt(less))**2
                   + (1.253314*wsig/np.sqrt(wess))**2)
    mc_floor = np.sqrt(np.mean(mcse**2))
    legacy_diag = json.loads(Path(f"{LEGACY}.diagnostics.json").read_text())
    wide_prod = json.loads(Path(f"{WIDE}.production.json").read_text())
    metrics = {
        "channels": int(wave.size),
        "box": {"u_plus": [-1.0, 2.0], "u_minus": [-2.0, 2.0]},
        "legacy_wall_seconds": json.loads(Path(f"{LEGACY}.timing.json").read_text())["total_wall_seconds"],
        "wide_wall_seconds": wide_prod["wall_seconds"],
        "legacy_divergences": int(legacy_diag["num_divergences_total"]),
        "wide_attempt_divergences": wide_prod["attempt_divergences"],
        "wide_sampler_counts": wide_prod["sampler_counts"],
        "lanes_swapped": int(sum(v for k,v in wide_prod["sampler_counts"].items() if k != "independent_nuts")),
        "legacy_min_depth_ess": float(np.min(less)),
        "wide_min_depth_ess": float(np.min(wess)),
        "weighted_offset_ppm": float(offset*1e6),
        "weighted_slope_ppm_per_um": float(slope*1e6),
        "offset_slope_removed_rms_ppm": float(rms*1e6),
        "median_mc_floor_ppm": float(mc_floor*1e6),
        "rms_over_mc_floor": float(rms/mc_floor),
        "median_error_bar_ratio_wide_over_legacy": float(np.median(wsig/lsig)),
    }
    SUMMARY.write_text(json.dumps(metrics, indent=2)+"\n")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].errorbar(wave, lmed*1e6, yerr=[(lmed-l16)*1e6,(l84-lmed)*1e6],
                     fmt=".", alpha=.65, label="legacy U(0,1)^2; adaptive joint NUTS")
    axes[0].errorbar(wave, wmed*1e6, yerr=[(wmed-w16)*1e6,(w84-wmed)*1e6],
                     fmt=".", alpha=.65, label="wide u+/u-; production gate/swap")
    axes[0].set_ylabel("Transit depth (ppm)"); axes[0].legend()
    axes[1].plot(wave, difference*1e6, "o", ms=3, label="wide - legacy")
    axes[1].plot(wave, (offset+slope*x)*1e6, "k--", label="weighted offset + slope")
    axes[1].axhline(offset*1e6, color="0.5", lw=1)
    axes[1].set_ylabel("Difference (ppm)"); axes[1].set_xlabel("Wavelength (micron)")
    axes[1].legend(); fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
