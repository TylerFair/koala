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
