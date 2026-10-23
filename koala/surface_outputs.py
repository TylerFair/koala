"""Emission and stellar-surface posterior products in user-facing units."""

from pathlib import Path
from .output_paths import OutputPaths

import numpy as np
import pandas as pd


_PRODUCTS = {
    "eclipse_depth_ppm": ("eclipse_depth", 1e6, "Planet/star flux [ppm]"),
    "phase_c1_ppm": ("phase_c1", 1e6, "Phase C1 [ppm]"),
    "phase_d1_ppm": ("phase_d1", 1e6, "Phase D1 [ppm]"),
    # Derived sinusoidal phase-curve quantities (see
    # models.jaxoplanet.surface.phase_curve_derived_quantities).
    "nightside_flux_ppm": ("nightside_flux", 1e6, "Nightside flux [ppm]"),
    "phase_amplitude_ppm": ("phase_amplitude", 1e6, "Phase amplitude [ppm]"),
    "phase_offset_deg": ("phase_offset_deg", 1.0, "Peak offset before eclipse [deg]"),
}


def _with_derived_phase_quantities(samples):
    """Add nightside flux, amplitude, and offset when C1 and D1 are present."""
    if not {"eclipse_depth", "phase_c1", "phase_d1"} <= set(samples):
        return samples
    from models.jaxoplanet.surface import phase_curve_derived_quantities

    derived = phase_curve_derived_quantities(
        np.asarray(samples["eclipse_depth"], dtype=float),
        np.asarray(samples["phase_c1"], dtype=float),
        np.asarray(samples["phase_d1"], dtype=float),
    )
    extended = dict(samples)
    for name, values in derived.items():
        extended.setdefault(name, np.asarray(values))
    return extended


def save_surface_results(wavelengths, wavelength_err, samples, csv_filename, plots="full"):
    """Write an emission CSV and figure alongside the usual radius spectrum.

    Samples have the pipeline's flattened draw, wavelength, planet layout.
    The CSV reports median and equal-tailed 68.27% intervals; planet indices
    are explicit. Pure transit fits produce no additional files.
    """
    products = {}
    provenance = getattr(samples, "sampler_used", None)
    samples = _with_derived_phase_quantities(samples)
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    widths = np.broadcast_to(np.asarray(wavelength_err, dtype=float), wavelengths.shape)
    if "stellar_spot_contrast" in samples:
        _save_spot_results(wavelengths, widths, samples["stellar_spot_contrast"], csv_filename, plots=plots)
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
    paths, stage = OutputPaths.from_spectrum(csv_filename)
    output = paths.table(stage, "emission")
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)

    import matplotlib.pyplot as plt
    from .plotting_style import (DATA_MARKER_STYLE, apply_publication_style, save_figure, style_axis, figure_title,
                                 accent_color_for_label, figsize)

    apply_publication_style()
    fig, axes = plt.subplots(len(products), 1, squeeze=False, sharex=True,
                             figsize=figsize(max(3.6, 3.0 * len(products))),
                             constrained_layout=True)
    for ax, (name, (quantiles, label)) in zip(axes[:, 0], products.items()):
        for planet in range(quantiles.shape[-1]):
            low, med, high = quantiles[:, :, planet]
            ax.errorbar(wavelengths, med, xerr=widths,
                        yerr=np.stack([med - low, high - med]),
                        label=f"Planet {planet + 1}", **{**DATA_MARKER_STYLE, 'color': accent_color_for_label(paths.prefix)})
        ax.set_ylabel(label)
        style_axis(ax)
        if quantiles.shape[-1] > 1:
            ax.legend()
    axes[-1, 0].set_xlabel("Wavelength [$\\mu$m]")
    from .diagnostic_figures import grid_title
    figure_title(axes[0, 0], paths.prefix, grid_title(stage) + " | Emission")
    save_figure(fig, paths.figure(stage or paths.prefix, "emission"))
    return frame


def _save_spot_results(wavelengths, widths, values, csv_filename, plots="full"):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3 or values.shape[1] != wavelengths.size:
        raise ValueError("stellar_spot_contrast must have draw, wavelength, spot axes.")
    quantiles = np.nanpercentile(values, [15.865, 50, 84.135], axis=0)
    rows = []
    import matplotlib.pyplot as plt
    from .plotting_style import (DATA_MARKER_STYLE, apply_publication_style, save_figure, style_axis, figure_title,
                                 accent_color_for_label, figsize)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize(3.6))
    for spot in range(values.shape[-1]):
        low, med, high = quantiles[:, :, spot]
        ax.errorbar(wavelengths, med, xerr=widths,
                    yerr=np.stack([med - low, high - med]), label=f"Spot {spot + 1}",
                    **{**DATA_MARKER_STYLE, 'color': accent_color_for_label(str(csv_filename))})
        for channel, wavelength in enumerate(wavelengths):
            rows.append({"wavelength": wavelength, "wavelength_err": widths[channel],
                         "spot_index": spot, "contrast": med[channel],
                         "contrast_err_low": med[channel] - low[channel],
                         "contrast_err_high": high[channel] - med[channel]})
    ax.set_xlabel("Wavelength [$\\mu$m]")
    ax.set_ylabel("Spot contrast")
    ax.legend()
    style_axis(ax)
    paths, stage = OutputPaths.from_spectrum(csv_filename)
    pd.DataFrame(rows).to_csv(paths.table(stage, "stellar_spots"), index=False)
    from .diagnostic_figures import grid_title
    figure_title(ax, paths.prefix, grid_title(stage) + " | Stellar spots")
    if plots == "full":
        save_figure(fig, paths.figure(stage or paths.prefix, "stellar_spots"))
    else:
        plt.close(fig)
