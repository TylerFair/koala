#!/usr/bin/env python3
"""Combine detector-level eclipse products into one emission spectrum."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_style import (
    DATA_MARKER_STYLE,
    ERRORBAR_COLOR,
    SPECTRUM_COLOR,
    ZERO_LINE_COLOR,
    apply_publication_style,
    style_axis,
)


REQUIRED = {
    "wavelength",
    "wavelength_err",
    "eclipse_depth_ppm",
    "eclipse_depth_ppm_err_low",
    "eclipse_depth_ppm_err_high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine NRS1 and NRS2 eclipse-depth CSV products."
    )
    parser.add_argument("--nrs1", required=True, type=Path)
    parser.add_argument("--nrs2", required=True, type=Path)
    parser.add_argument(
        "--output-stem",
        required=True,
        type=Path,
        help="Output path without .csv/.png suffixes.",
    )
    return parser.parse_args()


def load_detector(path: Path, detector: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {detector} emission product: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame.insert(0, "detector", detector)
    frame["source_file"] = str(path.resolve())
    return frame


def main() -> None:
    args = parse_args()
    frames = [load_detector(args.nrs1, "NRS1"), load_detector(args.nrs2, "NRS2")]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_columns = ["wavelength"]
    if "planet_index" in combined.columns:
        sort_columns.insert(0, "planet_index")
    combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    output_stem = args.output_stem
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    png_path = output_stem.with_suffix(".png")
    combined.to_csv(csv_path, index=False)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.errorbar(
        combined["wavelength"],
        combined["eclipse_depth_ppm"],
        xerr=combined["wavelength_err"],
        yerr=np.vstack(
            [
                combined["eclipse_depth_ppm_err_low"],
                combined["eclipse_depth_ppm_err_high"],
            ]
        ),
        color=SPECTRUM_COLOR,
        mec=SPECTRUM_COLOR,
        ecolor=ERRORBAR_COLOR,
        zorder=3,
        **DATA_MARKER_STYLE,
    )
    ax.axhline(0, color=ZERO_LINE_COLOR, lw=1.7, ls="--", zorder=1)
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_ylabel("Planet/star flux [ppm]")
    ax.set_title(r"WASP-39 b NIRSpec/G395H emission spectrum ($R = 300$)")
    style_axis(ax)
    fig.savefig(png_path, dpi=250)
    plt.close(fig)
    print(f"Wrote {len(combined)} channels to {csv_path}")
    print(f"Wrote combined emission plot to {png_path}")


if __name__ == "__main__":
    main()
