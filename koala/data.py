"""Spectroscopic data preparation helpers."""

import jax.numpy as jnp

def _pad_spectro_cadences_exact(t, y, yerr, multiple=256):
    """Pad cadence tails with an explicit zero-weight likelihood mask."""
    t = jnp.asarray(t, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    yerr = jnp.asarray(yerr, dtype=jnp.float64)
    original = int(t.shape[0])
    padded = ((original + int(multiple) - 1) // int(multiple)) * int(multiple)
    count = padded - original
    mask = jnp.arange(padded) < original
    if count == 0:
        return t, y, yerr, mask
    cadence = t[-1] - t[-2] if original > 1 else jnp.asarray(1.0)
    tail_t = t[-1] + cadence * jnp.arange(1, count + 1, dtype=jnp.float64)
    padded_t = jnp.concatenate((t, tail_t))
    padded_y = jnp.pad(y, ((0, 0), (0, count)), constant_values=1.0)
    padded_yerr = jnp.pad(yerr, ((0, 0), (0, count)), constant_values=1.0)
    return padded_t, padded_y, padded_yerr, mask


def jax_bin_lightcurve(time, flux, duration, points_per_transit=20):
    dt = duration / points_per_transit
    t_min = jnp.min(time)
    t_max = jnp.max(time)
    total_range = t_max - t_min
    num_bins = jnp.ceil(total_range / dt).astype(int) + 1
    bin_indices = jnp.clip(((time - t_min) / dt).astype(int), 0, num_bins - 1)
    flux_sums = jnp.zeros(num_bins)
    time_sums = jnp.zeros(num_bins)
    counts = jnp.zeros(num_bins)
    flux_sums = flux_sums.at[bin_indices].add(flux)
    time_sums = time_sums.at[bin_indices].add(time)
    counts = counts.at[bin_indices].add(1.0)
    binned_flux = jnp.where(counts > 0, flux_sums / counts, jnp.nan)
    binned_time = jnp.where(counts > 0, time_sums / counts, jnp.nan)
    return binned_time, binned_flux
