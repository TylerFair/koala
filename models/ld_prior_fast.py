"""Fast, exact construction of the power-2 limb-darkening prior grid.

The stellar-informed prior averages power-2 coefficients over a grid of
(Teff, logg, [M/H]) combinations.  The legacy builder calls exotic_ld once per
combination per wavelength bin, which is slow because exotic_ld reloads the
stellar model, rebuilds the 500-point Gauss-Legendre rule, re-reads the
throughput file and runs a scalar Levenberg-Marquardt fit for every call.

This module reproduces exotic_ld's numbers with three exact reductions:

1. exotic_ld's trilinear interpolation makes each combination's intensity
   table a fixed linear combination of at most eight stellar-grid nodes.
2. The passband integral I(mu) is linear in the intensity table, so each
   unique node is integrated once per bin and the combinations are formed
   by blending the node integrals with the same trilinear weights.
3. The nonlinear power-2 fits for every (combination, bin) pair are run as
   one batched Levenberg-Marquardt in JAX (float64), with the same initial
   guess and mu cut as exotic_ld's ``curve_fit(..., method="lm")``.

The integration replicates exotic_ld's ``_integrate_I_mu`` (strict-interior
masks, zero fill outside the masked grid, the "fewer than two points" constant
fallback) so the blended I(mu) agrees to rounding.  Fits agree with SciPy's
MINPACK result to its convergence tolerance (~1e-8).
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_grids import StellarGrids
from scipy.interpolate import interp1d
from scipy.special import roots_legendre

jax.config.update("jax_enable_x64", True)

POWER2_MU_MIN = 0.10  # exotic_ld's default for compute_power2_ld_coeffs.


@functools.lru_cache(maxsize=None)
def _legendre_rule(n=500):
    roots, weights = roots_legendre(n)
    return roots, weights


@functools.lru_cache(maxsize=None)
def _sensitivity(ld_data_path, mode, ld_data_version):
    path = os.path.join(
        ld_data_path, "Sensitivity_files",
        "{}_throughput{}.csv".format(mode, ld_data_version),
    )
    if not os.path.exists(path):
        # Let exotic_ld download it through its own machinery.
        probe = StellarLimbDarkening(
            M_H=0.0, Teff=5500.0, logg=4.5, ld_model="stagger",
            ld_data_path=ld_data_path, verbose=0,
        )
        return probe._read_sensitivity_data(mode)
    data = np.loadtxt(path, skiprows=1, delimiter=",")
    return data[:, 0], data[:, 1]


def _node_key(M_H, Teff, logg):
    """Key matching exotic_ld's on-disk node naming."""
    M_H = 0.0 if M_H == -0.0 else M_H
    return (str(round(M_H, 2)), int(round(Teff)), str(round(logg, 1)))


def stellar_node_weights(M_H, Teff, logg, ld_model, ld_data_path,
                         interpolate_type="trilinear", ld_data_version="",
                         remote_ld_data_path="https://www.star.bris.ac.uk/exotic-ld-data"):
    """Return ``{node_key: (M_H, Teff, logg, weight)}`` replicating exotic_ld.

    For ``trilinear`` this is the eight-vertex cuboid blend (falling back to
    the nearest node exactly as exotic_ld does when no cuboid exists); for
    ``nearest`` it is the single nearest node with weight one.
    """
    sg = StellarGrids(M_H, Teff, logg, ld_model, ld_data_path,
                      remote_ld_data_path, ld_data_version, interpolate_type, 0)
    if interpolate_type == "trilinear":
        vertices = sg._get_surrounding_grid_cuboid()
        if vertices is not None:
            x0, x1, y0, y1, z0, z1 = vertices
            x0 *= sg._r_M_H
            x1 *= sg._r_M_H
            y0 *= sg._r_Teff
            y1 *= sg._r_Teff
            z0 *= sg._r_logg
            z1 *= sg._r_logg
            xd = (M_H - x0) / (x1 - x0) if x0 != x1 else 0.0
            yd = (Teff - y0) / (y1 - y0) if y0 != y1 else 0.0
            zd = (logg - z0) / (z1 - z0) if z0 != z1 else 0.0
            out = {}
            for (mh, wx) in ((x0, 1 - xd), (x1, xd)):
                for (te, wy) in ((y0, 1 - yd), (y1, yd)):
                    for (lg, wz) in ((z0, 1 - zd), (z1, zd)):
                        w = wx * wy * wz
                        key = _node_key(mh, te, lg)
                        if key in out:
                            out[key] = (mh, te, lg, out[key][3] + w)
                        else:
                            out[key] = (mh, te, lg, w)
            return {k: v for k, v in out.items() if v[3] != 0.0}
        interpolate_type = "nearest"
    if interpolate_type != "nearest":
        raise ValueError("interpolate_type not recognised.")
    _, nearest_idx = sg._stellar_kd_tree.query(sg.x, k=1)
    mh, te, lg = sg._stellar_kd_tree.data[nearest_idx]
    mh *= sg._r_M_H
    te *= sg._r_Teff
    lg *= sg._r_logg
    return {_node_key(mh, te, lg): (mh, te, lg, 1.0)}


@functools.lru_cache(maxsize=64)
def _read_node_cached(ld_model, ld_data_path, key, M_H, Teff, logg, ld_data_version, remote):
    sg = StellarGrids(M_H, Teff, logg, ld_model, ld_data_path, remote,
                      ld_data_version, "nearest", 0)
    # exotic_ld's own reader (np.loadtxt), so values are bit-identical; the
    # lru_cache above is what removes the repeated reads.
    return sg._read_in_stellar_model(M_H, Teff, logg)


def _read_node(ld_model, ld_data_path, M_H, Teff, logg, ld_data_version, remote):
    key = _node_key(M_H, Teff, logg)
    wvs, mus, intens = _read_node_cached(
        ld_model, os.path.realpath(ld_data_path), key,
        float(M_H), float(Teff), float(logg), ld_data_version, remote)
    return wvs, mus, intens


def integrate_passband(wvs, intensities, s_wavelengths, s_throughputs, ranges):
    """Unnormalised I(mu) per bin, replicating exotic_ld ``_integrate_I_mu``.

    ``intensities`` is ``[n_wavelength, n_mu]``; ``ranges`` is ``[n_bins, 2]``
    in angstroms.  Returns ``[n_bins, n_mu]``.
    """
    roots, weights = _legendre_rule(500)
    n_mu = intensities.shape[1]
    out = np.zeros((len(ranges), n_mu))
    for bi, (a, b) in enumerate(ranges):
        a, b = float(min(a, b)), float(max(a, b))
        if b < s_wavelengths[0] or s_wavelengths[-1] < a:
            raise ValueError(
                f"Wavelength range {a}-{b} A has no overlap with the instrument "
                f"mode's range {s_wavelengths[0]}-{s_wavelengths[-1]} A.")
        if b < wvs[0] or wvs[-1] < a:
            raise ValueError(
                f"Wavelength range {a}-{b} A has no overlap with the stellar "
                f"spectra's range {wvs[0]}-{wvs[-1]} A.")
        s_mask = np.logical_and(a < s_wavelengths, s_wavelengths < b)
        i_mask = np.logical_and(a < wvs, wvs < b)
        s_wvs = s_wavelengths[s_mask]
        s_thp = s_throughputs[s_mask]
        i_wvs = wvs[i_mask]
        i_int = intensities[i_mask]
        t = (b - a) / 2 * roots + (a + b) / 2
        if s_wvs.shape[0] >= 2:
            s_t = interp1d(s_wvs, s_thp, kind="linear", bounds_error=False,
                           fill_value=0.)(t)
        else:
            idx = np.argmin(np.abs(s_wavelengths - np.mean([a, b])))
            s_t = np.ones(t.shape) * s_throughputs[idx]
        if i_wvs.shape[0] >= 2:
            i_t = interp1d(i_wvs, i_int, kind="linear", axis=0,
                           bounds_error=False, fill_value=0.)(t)
        else:
            idx = np.argmin(np.abs(wvs - np.mean([a, b])))
            i_t = np.ones((t.shape[0], n_mu)) * intensities[idx][None, :]
        out[bi] = (b - a) / 2. * ((s_t[:, None] * i_t).T @ weights)
    return out


def _power2_model(p, mu):
    c, alpha = p
    return 1. - c * (1. - mu ** alpha)


def _lm_fit_one(I_mu, mu, n_iter=60):
    """Levenberg-Marquardt for the power-2 law from exotic_ld's guess (1, 1)."""
    def resid(p):
        return I_mu - _power2_model(p, mu)

    jac = jax.jacfwd(resid)

    def cost(p):
        r = resid(p)
        return jnp.dot(r, r)

    def body(_, state):
        p, lam = state
        r = resid(p)
        J = jac(p)
        JTJ = J.T @ J
        g = J.T @ r
        A = JTJ + lam * jnp.diag(jnp.diag(JTJ))
        step = -jnp.linalg.solve(A, g)
        p_new = p + step
        better = cost(p_new) < cost(p)
        p = jnp.where(better, p_new, p)
        lam = jnp.where(better, lam * 0.3, lam * 5.0)
        return p, lam

    p0 = jnp.array([1.0, 1.0])
    p, _ = jax.lax.fori_loop(0, n_iter, body, (p0, jnp.asarray(1e-3)))
    # Polish with a few pure Gauss-Newton steps once inside the basin.
    def gn(_, p):
        r = resid(p)
        J = jac(p)
        step = -jnp.linalg.solve(J.T @ J, J.T @ r)
        p_new = p + step
        return jnp.where(cost(p_new) <= cost(p), p_new, p)
    p = jax.lax.fori_loop(0, 8, gn, p)
    r = resid(p)
    grad_norm = jnp.linalg.norm(jac(p).T @ r)
    return p, grad_norm


@functools.partial(jax.jit, static_argnums=())
def _lm_fit_batch(I_mu_batch, mu):
    return jax.vmap(_lm_fit_one, in_axes=(0, None))(I_mu_batch, mu)


def fit_power2_batch(I_mu, mus, mu_min=POWER2_MU_MIN, chunk=8192):
    """Fit the power-2 law to ``I_mu[..., n_mu]`` (already normalised).

    Returns ``coeffs[..., 2]`` and ``grad_norm[...]``.
    """
    mus = np.asarray(mus, dtype=float)
    mask = mus >= mu_min
    if mask.sum() < 2:
        raise ValueError("mu_min set too high, must be >= 2 mu values remaining.")
    lead = I_mu.shape[:-1]
    flat = np.asarray(I_mu, dtype=float).reshape(-1, I_mu.shape[-1])[:, mask]
    mu = jnp.asarray(mus[mask])
    coeffs = np.empty((flat.shape[0], 2))
    gnorm = np.empty(flat.shape[0])
    for start in range(0, flat.shape[0], chunk):
        block = jnp.asarray(flat[start:start + chunk])
        p, g = _lm_fit_batch(block, mu)
        coeffs[start:start + chunk] = np.asarray(p)
        gnorm[start:start + chunk] = np.asarray(g)
    return coeffs.reshape(*lead, 2), gnorm.reshape(*lead)


def build_power2_grid(combos, ranges, mode, ld_model, ld_data_path,
                      interpolate_type="trilinear", mu_min=POWER2_MU_MIN,
                      ld_data_version="", log=print):
    """Power-2 coefficients for every (stellar combination, bin).

    ``combos`` is an iterable of ``(M_H, Teff, logg)``; ``ranges`` is
    ``[n_bins, 2]`` in angstroms (already clipped to the instrument mode).
    Returns ``coeffs[n_combos, n_bins, 2]`` and ``grad_norm[n_combos, n_bins]``.
    """
    combos = [tuple(map(float, c)) for c in combos]
    ranges = np.asarray(ranges, dtype=float).reshape(-1, 2)
    remote = "https://www.star.bris.ac.uk/exotic-ld-data"
    if ld_data_version == "3.2.0":
        ld_data_version = ""  # exotic_ld's backwards-compatible alias.
    ld_model = {"1D": "kurucz", "3D": "stagger"}.get(ld_model, ld_model)

    weights = [stellar_node_weights(mh, te, lg, ld_model, ld_data_path,
                                    interpolate_type, ld_data_version, remote)
               for (mh, te, lg) in combos]
    nodes = {}
    for w in weights:
        for key, (mh, te, lg, _) in w.items():
            nodes.setdefault(key, (mh, te, lg))
    log(f"[LD prior fast] {len(combos)} stellar combinations span "
        f"{len(nodes)} unique {ld_model} grid nodes; {len(ranges)} bins")

    s_wvs, s_thp = _sensitivity(ld_data_path, mode, ld_data_version)
    node_I = {}
    mus_ref = None
    for key, (mh, te, lg) in nodes.items():
        wvs, mus, intens = _read_node(ld_model, ld_data_path, mh, te, lg,
                                      ld_data_version, remote)
        if mus_ref is None:
            mus_ref = mus
        elif not np.array_equal(mus, mus_ref):
            raise ValueError("Stellar grid nodes disagree on the mu grid.")
        node_I[key] = integrate_passband(wvs, intens, s_wvs, s_thp, ranges)

    n_mu = mus_ref.shape[0]
    I_all = np.zeros((len(combos), len(ranges), n_mu))
    for ci, w in enumerate(weights):
        for key, (_, _, _, wk) in w.items():
            I_all[ci] += wk * node_I[key]
    if np.any(I_all[:, :, 0] == 0.):
        raise ValueError("Zero intensity in this passband, check your "
                         "wavelength range is correct and in angstroms.")
    I_all /= I_all[:, :, :1]

    coeffs, gnorm = fit_power2_batch(I_all, mus_ref, mu_min=mu_min)
    return coeffs, gnorm
