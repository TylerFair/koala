"""Emission and stellar-surface posterior products in user-facing units."""

from pathlib import Path

import numpy as np
import pandas as pd


_PRODUCTS = {
    "eclipse_depth_ppm": ("eclipse_depth", 1e6, "Planet/star flux [ppm]"),
    "dayside_flux_ppm": ("dayside_flux", 1e6, "Dayside flux [ppm]"),
    "nightside_flux_ppm": ("nightside_flux", 1e6, "Nightside flux [ppm]"),
    "hotspot_offset_deg": ("hotspot_offset", 180 / np.pi, "Hotspot offset [deg]"),
}


def save_surface_results(wavelengths, wavelength_err, samples, csv_filename):
    """Write an emission CSV and figure alongside the usual radius spectrum.

    Samples have the pipeline's flattened draw, wavelength, planet layout.
    The CSV reports median and equal-tailed 68.27% intervals; planet indices
    are explicit. Pure transit fits produce no additional files.
    """
    products = {}
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    widths = np.broadcast_to(np.asarray(wavelength_err, dtype=float), wavelengths.shape)
    if "stellar_spot_contrast" in samples:
        _save_spot_results(wavelengths, widths, samples["stellar_spot_contrast"], csv_filename)
    for name, (site, scale, label) in _PRODUCTS.items():
        if name in samples:
            values = np.asarray(samples[name], dtype=float)
        elif site in samples:
            values = np.asarray(samples[site], dtype=float) * scale
        else:
            continue
        if values.ndim == 2:
            values = values[..., None]
        if values.ndim != 3 or values.shape[1] != wavelengths.size:
            raise ValueError(f"{name} must have draw, wavelength[, planet] axes.")
        products[name] = (np.nanpercentile(values, [15.865, 50, 84.135], axis=0), label)
    if not products:
        return None
    planet_counts = {value[0].shape[-1] for value in products.values()}
    if len(planet_counts) != 1:
        raise ValueError("Emission posterior products have inconsistent planet axes.")
    rows = []
    provenance = getattr(samples, "sampler_used", None)
    if provenance is not None and len(provenance) != wavelengths.size:
        raise ValueError("sampler_used provenance does not match wavelength axis")
    for planet in range(next(iter(planet_counts))):
        for channel, wavelength in enumerate(wavelengths):
            row = {"wavelength": wavelength, "wavelength_err": widths[channel], "planet_index": planet}
            for name, (quantiles, _) in products.items():
                low, med, high = quantiles[:, channel, planet]
                row.update({name: med, name + "_err_low": med - low, name + "_err_high": high - med})
            if provenance is not None:
                row["sampler_used"] = provenance[channel]
            rows.append(row)
    stem = Path(csv_filename).with_suffix("")
    output = stem.with_name(stem.name + "_emission.csv")
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)

    import matplotlib.pyplot as plt
    from plotting_style import apply_publication_style, save_figure, style_axis

    apply_publication_style()
    fig, axes = plt.subplots(len(products), 1, squeeze=False, sharex=True,
                             figsize=(8.8, max(4.5, 3.7 * len(products))),
                             constrained_layout=True)
    for ax, (name, (quantiles, label)) in zip(axes[:, 0], products.items()):
        for planet in range(quantiles.shape[-1]):
            low, med, high = quantiles[:, :, planet]
            ax.errorbar(wavelengths, med, xerr=widths,
                        yerr=np.stack([med - low, high - med]),
                        fmt="o", label=f"Planet {planet + 1}")
        ax.set_ylabel(label)
        style_axis(ax)
        if quantiles.shape[-1] > 1:
            ax.legend()
    axes[-1, 0].set_xlabel("Wavelength [$\\mu$m]")
    save_figure(fig, stem.with_name(stem.name + "_emission.png"))
    return frame


def _save_spot_results(wavelengths, widths, values, csv_filename):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3 or values.shape[1] != wavelengths.size:
        raise ValueError("stellar_spot_contrast must have draw, wavelength, spot axes.")
    quantiles = np.nanpercentile(values, [15.865, 50, 84.135], axis=0)
    rows = []
    import matplotlib.pyplot as plt
    from plotting_style import apply_publication_style, save_figure, style_axis

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    for spot in range(values.shape[-1]):
        low, med, high = quantiles[:, :, spot]
        ax.errorbar(wavelengths, med, xerr=widths,
                    yerr=np.stack([med - low, high - med]), fmt="o",
                    label=f"Spot {spot + 1}")
        for channel, wavelength in enumerate(wavelengths):
            rows.append({"wavelength": wavelength, "wavelength_err": widths[channel],
                         "spot_index": spot, "contrast": med[channel],
                         "contrast_err_low": med[channel] - low[channel],
                         "contrast_err_high": high[channel] - med[channel]})
    ax.set_xlabel("Wavelength [$\\mu$m]")
    ax.set_ylabel("Spot contrast")
    ax.legend()
    style_axis(ax)
    stem = Path(csv_filename).with_suffix("")
    pd.DataFrame(rows).to_csv(stem.with_name(stem.name + "_stellar_spots.csv"), index=False)
    save_figure(fig, stem.with_name(stem.name + "_stellar_spots.png"))
