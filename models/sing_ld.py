"""Utilities for the opt-in Sing et al. (2026) quadratic LD prior."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SING_TABULATED_OFFSET = {"l": 0.020, "delta": -0.003}
SING_TABULATED_SCATTER = {"l": 0.031, "delta": 0.016}


def quadratic_to_sing(c1, c2):
    """Convert quadratic coefficients to ``(l, delta)``."""
    c1, c2 = np.asarray(c1), np.asarray(c2)
    u_plus = c1 + c2
    u_minus = c1 - c2
    return 1.0 - u_plus, (u_plus - u_minus) / 8.0


def sing_to_quadratic(l, delta):
    """Convert ``(l, delta)`` to quadratic ``(c1, c2)``."""
    l, delta = np.asarray(l), np.asarray(delta)
    u_plus = 1.0 - l
    u_minus = u_plus - 8.0 * delta
    return (u_plus + u_minus) / 2.0, (u_plus - u_minus) / 2.0


def estimate_gray_offset(fitted_c, model_c, fitted_sigma):
    """Inverse-variance weighted mean of fitted-minus-model in (l, delta)."""
    fitted_c = np.asarray(fitted_c, dtype=float)
    model_c = np.asarray(model_c, dtype=float)
    fitted_sigma = np.asarray(fitted_sigma, dtype=float)
    if fitted_c.shape != model_c.shape or fitted_c.shape != fitted_sigma.shape:
        raise ValueError("fitted_c, model_c, and fitted_sigma must have identical [channel,2] shapes")
    fl, fd = quadratic_to_sing(fitted_c[:, 0], fitted_c[:, 1])
    ml, md = quadratic_to_sing(model_c[:, 0], model_c[:, 1])
    # Linear error propagation, deliberately ignoring an unavailable c1/c2 covariance.
    sl = np.hypot(fitted_sigma[:, 0], fitted_sigma[:, 1])
    sd = fitted_sigma[:, 1] / 4.0
    result = {}
    for name, residual, sigma in (("l", fl - ml, sl), ("delta", fd - md, sd)):
        good = np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0)
        if not np.any(good):
            raise ValueError(f"no finite positive-uncertainty channels for {name}")
        weights = 1.0 / sigma[good] ** 2
        result[name] = float(np.sum(weights * residual[good]) / np.sum(weights))
        result[f"{name}_sigma"] = float(np.sqrt(1.0 / np.sum(weights)))
        result[f"{name}_n"] = int(np.sum(good))
    return result


def write_offset_artifact(path, offsets, fingerprint_inputs):
    """Create a new fingerprinted JSON artifact; never replaces an existing file."""
    payload = json.dumps(fingerprint_inputs, sort_keys=True, default=str).encode()
    document = dict(offsets)
    document["fingerprint_sha256"] = hashlib.sha256(payload).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(target, flags, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return document
