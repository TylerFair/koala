"""Pure JAX implementation of N_c=1 power-2 transit for GPU execution.

Fully JIT-compilable via XLA.  Supports ``jax.grad`` for exact autodiff
gradients (no finite differences).  Implements the *same* mathematics as
the CUDA kernel ``power2_nc1_kernel.cu``: grid-bisection root finder,
T+/T- segment classification, Gauss-Legendre quadrature on planet-limb
segments, and closed-form star-limb integrals.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

# Regularization for complex magnitude computations to prevent NaN gradients
# at z=0 where |z| = sqrt(x² + y²) has undefined gradient.
_COMPLEX_ABS_EPS = 1e-14


@jax.custom_jvp
def _safe_sqrt(x):
    """Return ``sqrt(max(x, 0))`` without perturbing the forward model."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


@_safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    value = _safe_sqrt(x)
    denom = jnp.where(x > 0.0, value, 1.0)
    derivative = jnp.where(x > 0.0, 0.5 * dx / denom, 0.0)
    return value, derivative


# ============================================================
# Gradient-safe complex operations
# ============================================================
# JAX's jnp.abs and jnp.angle have singular gradients at z=0.
# When used inside jnp.where, gradient leakage from unselected branches
# can produce NaN even when the branch isn't taken. These custom
# implementations provide stable gradients everywhere.

@jax.custom_jvp
def _safe_complex_abs(z):
    """Compute |z| with gradient that is stable at z=0."""
    return jnp.abs(z)


@_safe_complex_abs.defjvp
def _safe_complex_abs_jvp(primals, tangents):
    (z,) = primals
    (dz,) = tangents
    abs_z = jnp.abs(z)
    # At z ≈ 0, the gradient d|z|/dz = z*/|z| is undefined.
    # Return zero gradient for |z| < eps (measure-zero region).
    safe_abs = jnp.maximum(abs_z, _COMPLEX_ABS_EPS)
    # Gradient: Re(z* · dz) / |z| = Re(conj(z) · dz) / |z|
    grad = jnp.real(jnp.conj(z) * dz) / safe_abs
    # Zero out gradient when |z| is tiny to prevent NaN leakage
    grad = jnp.where(abs_z < _COMPLEX_ABS_EPS, 0.0, grad)
    return abs_z, grad


@jax.custom_jvp
def _safe_complex_angle(z):
    """Compute angle(z) with gradient that is stable at z=0."""
    return jnp.angle(z)


@_safe_complex_angle.defjvp
def _safe_complex_angle_jvp(primals, tangents):
    (z,) = primals
    (dz,) = tangents
    angle_z = jnp.angle(z)
    abs_z_sq = jnp.real(z) ** 2 + jnp.imag(z) ** 2
    # At z ≈ 0, the gradient d(angle(z))/dz is undefined.
    # Return zero gradient for |z|² < eps.
    safe_abs_sq = jnp.maximum(abs_z_sq, _COMPLEX_ABS_EPS ** 2)
    # Gradient of atan2(y, x): d(angle)/dx = -y/(x²+y²), d(angle)/dy = x/(x²+y²)
    # For complex dz = dx + i·dy: grad = (-y·dx + x·dy) / (x² + y²)
    #                                  = Im(conj(z) · dz) / |z|²
    grad = jnp.imag(jnp.conj(z) * dz) / safe_abs_sq
    # Zero out gradient when |z|² is tiny
    grad = jnp.where(abs_z_sq < _COMPLEX_ABS_EPS ** 2, 0.0, grad)
    return angle_z, grad


# ============================================================
# Constants
# ============================================================
# Plain Python floats — NOT jnp.float64 — so that they stay concrete even
# when this module is first imported during a JAX transformation (lazy import
# inside custom_primitives.harmonica_transit_power2_ld).
_HC_PI = math.pi
_HC_TWOPI = 2.0 * math.pi
_HC_PI_D_2 = math.pi / 2.0

_INTERSECT_TOL = 1.0e-7
_GEOMETRY_TOL = 1.0e-12

_ROOT_GRID_SIZE = 128
_ROOT_BISECT_STEPS = 50
_ROOT_GRID = np.linspace(
    -_HC_PI, _HC_PI, _ROOT_GRID_SIZE + 1, dtype=np.float64
)
_ROOT_GRID_COS = np.cos(_ROOT_GRID)
_ROOT_GRID_SIN = np.sin(_ROOT_GRID)

# 50-point Gauss-Legendre quadrature roots and weights (from CUDA kernel).
# Use np.array (not jnp) so these are concrete even if module is imported
# during a JAX transformation scope.
_GL_ROOTS = np.array([
    -0.998866404420071, -0.9940319694320907, -0.9853540840480057,
    -0.972864385106692, -0.9566109552428079, -0.936656618944878,
    -0.9130785566557917, -0.8859679795236131, -0.8554297694299462,
    -0.821582070859336, -0.7845558329003992, -0.7444943022260685,
    -0.7015524687068223, -0.6558964656854394, -0.6077029271849502,
    -0.5571583045146501, -0.5044581449074641, -0.44980633497403877,
    -0.39341431189756515, -0.3355002454194374, -0.27628819377953207,
    -0.21600723687604179, -0.15489058999814587, -0.09317470156008616,
    -0.031098338327188835, 0.031098338327188835, 0.09317470156008616,
    0.15489058999814587, 0.21600723687604179, 0.27628819377953207,
    0.3355002454194374, 0.39341431189756515, 0.44980633497403877,
    0.5044581449074641, 0.5571583045146501, 0.6077029271849502,
    0.6558964656854394, 0.7015524687068223, 0.7444943022260685,
    0.7845558329003992, 0.821582070859336, 0.8554297694299462,
    0.8859679795236131, 0.9130785566557917, 0.936656618944878,
    0.9566109552428079, 0.972864385106692, 0.9853540840480057,
    0.9940319694320907, 0.998866404420071,
], dtype=np.float64)

_GL_WEIGHTS = np.array([
    0.002908622553150225, 0.006759799195744691, 0.010590548383652589,
    0.014380822761487319, 0.01811556071348955, 0.02178024317012366,
    0.02536067357001322, 0.028842993580535235, 0.03221372822357793,
    0.035459835615146054, 0.03856875661258746, 0.04152846309014782,
    0.044327504338803454, 0.04695505130394862, 0.049400938449466386,
    0.05165570306958117, 0.05371062188899652, 0.05555774480621284,
    0.05718992564772871, 0.05860084981322267, 0.05978505870426556,
    0.06073797084177046, 0.06145589959031681, 0.06193606742068351,
    0.062176616655347454, 0.062176616655347454, 0.06193606742068351,
    0.06145589959031681, 0.06073797084177046, 0.05978505870426556,
    0.05860084981322267, 0.05718992564772871, 0.05555774480621284,
    0.05371062188899652, 0.05165570306958117, 0.049400938449466386,
    0.04695505130394862, 0.044327504338803454, 0.04152846309014782,
    0.03856875661258746, 0.035459835615146054, 0.03221372822357793,
    0.028842993580535235, 0.02536067357001322, 0.02178024317012366,
    0.01811556071348955, 0.014380822761487319, 0.010590548383652589,
    0.006759799195744691, 0.002908622553150225,
], dtype=np.float64)


# ============================================================
# Kepler solver — iterative Newton-Raphson
# ============================================================
def _solve_kepler(M, ecc):
    """Solve Kepler's equation M = E - ecc*sin(E) for sin(f), cos(f).

    Uses 15 Newton-Raphson iterations with a high-eccentricity starter that
    avoids convergence to the wrong 2-pi branch.
    """
    # Reduce M to [0, 2pi)
    MA = jnp.mod(M, _HC_TWOPI)
    MA = jnp.where(MA < 0.0, MA + _HC_TWOPI, MA)
    # Reflect into [0, pi]
    MAsign = jnp.where(MA > _HC_PI, -1.0, 1.0)
    MA = jnp.where(MA > _HC_PI, _HC_TWOPI - MA, MA)

    # A plain E0=M Newton solve can jump to the wrong 2-pi branch for high
    # eccentricity. The 0.85*e offset is a standard robust starter, while the
    # exact endpoint roots are retained explicitly.
    high_e_guess = MA + 0.85 * ecc
    at_endpoint = (MA < 1e-12) | ((_HC_PI - MA) < 1e-12)
    E0 = jnp.where(
        (ecc > 0.8) & (~at_endpoint),
        high_e_guess,
        MA,
    )

    def newton_step(E, _):
        sE = jnp.sin(E)
        cE = jnp.cos(E)
        f_val = E - ecc * sE - MA
        fp_val = 1.0 - ecc * cE
        dE = f_val / fp_val
        return E - dE, None

    E_final, _ = lax.scan(newton_step, E0, None, length=15)

    sinE = MAsign * jnp.sin(E_final)
    cosE = jnp.cos(E_final)

    # True anomaly from eccentric anomaly via half-angle formula.
    denom_ta = 1.0 + cosE
    ome = 1.0 - ecc
    tanf2 = jnp.sqrt((1.0 + ecc) / jnp.maximum(ome, 1e-30)) * sinE / jnp.maximum(denom_ta, 1e-30)
    tanf2_sq = tanf2 * tanf2
    inv_denom = 1.0 / (1.0 + tanf2_sq)
    sinf = 2.0 * tanf2 * inv_denom
    cosf = (1.0 - tanf2_sq) * inv_denom

    # Handle denom_ta ~ 0 (E ~ pi, f ~ pi)
    sinf = jnp.where(denom_ta > 1e-10, sinf, 0.0)
    cosf = jnp.where(denom_ta > 1e-10, cosf, -1.0)

    return sinf, cosf


# ============================================================
# Orbit computation
# ============================================================
def _compute_orbit(time, t0, period, a_rs, inc, ecc, omega):
    """Compute projected separation d, z-coordinate, and nu angle.

    Returns (d, z, nu) where d is the sky-projected separation, z is the
    line-of-sight component (negative = behind star), and nu is the angle
    from the observer to the planet center in the sky plane.
    """
    n = _HC_TWOPI / period
    sin_inc = jnp.sin(inc)
    cos_inc = jnp.cos(inc)

    def circular_branch(_):
        tp = t0 - _HC_PI_D_2 / n
        mean_anomaly = (time - tp) * n
        sin_M = jnp.sin(mean_anomaly)
        cos_M = jnp.cos(mean_anomaly)
        x = a_rs * cos_M
        y = a_rs * cos_inc * sin_M
        z = a_rs * sin_inc * sin_M
        # C++ uses atan(-cos/sin), not atan2. atan(y/x) returns
        # (-pi/2, pi/2); the guard retains its IEEE endpoint behaviour.
        psi = cos_inc * jnp.arctan(
            -cos_M / jnp.where(sin_M == 0.0, 1e-300, sin_M)
        )
        d = _safe_sqrt(x * x + y * y)
        nu = jnp.arctan2(y, x) - psi
        return d, z, nu

    def eccentric_branch(_):
        sin_omega = jnp.sin(omega)
        cos_omega = jnp.cos(omega)
        some = jnp.sqrt(jnp.maximum(1.0 - ecc, 1e-30))
        sope = jnp.sqrt(1.0 + ecc)
        E0 = 2.0 * jnp.arctan2(
            some * cos_omega, sope * (1.0 + sin_omega)
        )
        M0 = E0 - ecc * jnp.sin(E0)
        tp = t0 - M0 / n
        mean_anomaly = (time - tp) * n
        sinf, cosf = _solve_kepler(mean_anomaly, ecc)
        r = (
            a_rs
            * (1.0 - ecc * ecc)
            / jnp.maximum(1.0 + ecc * cosf, 1e-30)
        )
        sin_fpw = cosf * sin_omega + sinf * cos_omega
        cos_fpw = cosf * cos_omega - sinf * sin_omega
        x = r * cos_fpw
        y = r * cos_inc * sin_fpw
        z = r * sin_inc * sin_fpw
        psi = cos_inc * jnp.arctan(
            -cos_fpw / jnp.where(sin_fpw == 0.0, 1e-300, sin_fpw)
        )
        d = _safe_sqrt(x * x + y * y)
        nu = jnp.arctan2(y, x) - psi
        return d, z, nu

    # The orbit type is shared by every sample. A scalar conditional lets XLA
    # skip all 15 Kepler iterations for the circular JWST fits.
    return lax.cond(ecc < 1e-15, circular_branch, eccentric_branch, operand=None)


# ============================================================
# Intersection function
# ============================================================
def _intersection_func(theta, d, nu, r0, a1, b1, dd):
    """f(theta) = rp^2 + d^2 - 2*d*rp*cos(theta - nu) - 1."""
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    rp = r0 + a1 * cos_t + b1 * sin_t
    cos_tmnu = cos_t * jnp.cos(nu) + sin_t * jnp.sin(nu)
    return rp * rp + dd - 2.0 * d * rp * cos_tmnu - 1.0


# ============================================================
# Robust intersection finder with implicit root derivatives
# ============================================================
@jax.custom_jvp
def _grid_bisect_roots(d, nu, r0, a1, b1):
    """Find up to four N_c=1 intersections on a fixed angular grid.

    The brackets and bisection decisions are discrete, but the returned root
    tangents are supplied by the implicit-function theorem.  This avoids the
    ill-conditioned monic quartic used by the previous implementation when
    ``a1`` and ``b1`` approach zero.
    """
    dd = d * d
    grid = jnp.asarray(_ROOT_GRID)
    grid_cos = jnp.asarray(_ROOT_GRID_COS)
    grid_sin = jnp.asarray(_ROOT_GRID_SIN)
    rp_grid = r0 + a1 * grid_cos + b1 * grid_sin
    cos_grid_mnu = (
        grid_cos * jnp.cos(nu) + grid_sin * jnp.sin(nu)
    )
    values = (
        rp_grid * rp_grid + dd - 2.0 * d * rp_grid * cos_grid_mnu - 1.0
    )
    left_values = values[:-1]
    right_values = values[1:]
    changes = (
        ((left_values < 0.0) & (right_values >= 0.0))
        | ((left_values > 0.0) & (right_values <= 0.0))
    )
    indices = jnp.nonzero(changes, size=4, fill_value=0)[0]
    n_roots = jnp.minimum(jnp.sum(changes.astype(jnp.int32)), 4)
    valid = jnp.arange(4) < n_roots
    lo0 = grid[indices]
    hi0 = grid[indices + 1]
    flo0 = left_values[indices]

    def bisect_step(_, state):
        lo, hi, flo = state
        mid = 0.5 * (lo + hi)
        fmid = jax.vmap(
            lambda theta: _intersection_func(theta, d, nu, r0, a1, b1, dd)
        )(mid)
        same_side = ((flo < 0.0) & (fmid < 0.0)) | ((flo > 0.0) & (fmid > 0.0))
        lo = jnp.where(same_side, mid, lo)
        flo = jnp.where(same_side, fmid, flo)
        hi = jnp.where(same_side, hi, mid)
        return lo, hi, flo

    lo, hi, _ = lax.fori_loop(
        0, _ROOT_BISECT_STEPS, bisect_step, (lo0, hi0, flo0), unroll=10
    )
    roots = jnp.where(valid, 0.5 * (lo + hi), 0.0)
    return roots, valid.astype(jnp.float64), n_roots.astype(jnp.float64)


@_grid_bisect_roots.defjvp
def _grid_bisect_roots_jvp(primals, tangents):
    d, nu, r0, a1, b1 = primals
    dd_t, nu_t, r0_t, a1_t, b1_t = tangents
    roots, valid_float, n_roots_float = _grid_bisect_roots(d, nu, r0, a1, b1)
    valid = valid_float > 0.5

    sin_theta = jnp.sin(roots)
    cos_theta = jnp.cos(roots)
    sin_nu = jnp.sin(nu)
    cos_nu = jnp.cos(nu)
    sin_tmnu = sin_theta * cos_nu - cos_theta * sin_nu
    cos_tmnu = cos_theta * cos_nu + sin_theta * sin_nu
    rp = r0 + a1 * cos_theta + b1 * sin_theta
    drp_dtheta = -a1 * sin_theta + b1 * cos_theta
    df_dtheta = (
        2.0 * rp * drp_dtheta
        - 2.0 * d * (drp_dtheta * cos_tmnu - rp * sin_tmnu)
    )

    drp_t = r0_t + a1_t * cos_theta + b1_t * sin_theta
    df_params_t = (
        2.0 * rp * drp_t
        + 2.0 * d * dd_t
        - 2.0
        * (
            dd_t * rp * cos_tmnu
            + d * drp_t * cos_tmnu
            + d * rp * sin_tmnu * nu_t
        )
    )
    # Invalid entries are padded with theta=0. Their df/dtheta can also be
    # zero; masking only after division leaks 0/0 through reverse-mode.
    nonsingular = valid & (jnp.abs(df_dtheta) > 1e-14)
    safe_df_dtheta = jnp.where(nonsingular, df_dtheta, 1.0)
    root_t = jnp.where(nonsingular, -df_params_t / safe_df_dtheta, 0.0)
    return (
        roots,
        valid_float,
        n_roots_float,
    ), (
        root_t,
        jnp.zeros_like(valid_float),
        jnp.zeros_like(n_roots_float),
    )


def _find_intersections(d, nu, r0, a1, b1, dd):
    del dd
    thetas, valid_float, n_roots_float = _grid_bisect_roots(
        d, nu, r0, a1, b1
    )
    return (
        thetas,
        valid_float > 0.5,
        jnp.rint(n_roots_float).astype(jnp.int32),
    )


# ============================================================
# T+/T- segment classification (matching CUDA kernel)
# ============================================================
def _compute_T_dT(theta, d, nu, r0, a1, b1, dd):
    """Compute intersection type T (1=T+, 0=T-) and gradient direction dT."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    cos_nu = jnp.cos(nu)
    sin_nu = jnp.sin(nu)
    cos_tmnu = cos_theta * cos_nu + sin_theta * sin_nu
    sin_tmnu = sin_theta * cos_nu - cos_theta * sin_nu
    dcos = d * cos_tmnu
    dsin = d * sin_tmnu
    rp = r0 + a1 * cos_theta + b1 * sin_theta
    drp = -a1 * sin_theta + b1 * cos_theta

    disc = jnp.maximum(dcos * dcos - dd + 1.0, 0.0)
    rs_plus = dcos + _safe_sqrt(disc)

    # T = 1 when d <= 1 (planet center inside stellar disc) OR rp ~ rs_plus.
    inside = d <= 1.0
    T = jnp.where(inside, 1.0, jnp.where(jnp.abs(rp - rs_plus) < _INTERSECT_TOL, 1.0, 0.0))

    # Gradient of (rp - rs) at intersection.
    grad = drp + dsin
    sqrt_disc = _safe_sqrt(disc)
    safe_sqrt_disc = jnp.where(disc > 0.0, sqrt_disc, 1.0)
    frac = jnp.where(disc > 0.0, dsin * dcos / safe_sqrt_disc, 0.0)
    grad = grad + jnp.where(T > 0.5, frac, -frac)
    dT = jnp.where(grad > 0.0, 1.0, 0.0)

    return T, dT


def _classify_segment(T_j, dT_j, T_jp1, dT_jp1):
    """Return 1.0 for planet segment, 0.0 for star segment.

    Implements the lookup table from the CUDA kernel's
    characterise_intersection_pairs().
    """
    is_planet = (
        ((T_j > 0.5) & (T_jp1 > 0.5) & (dT_j < 0.5) & (dT_jp1 > 0.5)) |
        ((T_j < 0.5) & (T_jp1 < 0.5) & (dT_j > 0.5) & (dT_jp1 < 0.5)) |
        ((T_j > 0.5) & (T_jp1 < 0.5) & (dT_j < 0.5) & (dT_jp1 < 0.5)) |
        ((T_j < 0.5) & (T_jp1 > 0.5) & (dT_j > 0.5) & (dT_jp1 > 0.5))
    )
    # Fallback to planet (matches CUDA default).
    is_star = (
        ((T_j > 0.5) & (T_jp1 > 0.5) & (dT_j > 0.5) & (dT_jp1 < 0.5)) |
        ((T_j < 0.5) & (T_jp1 < 0.5) & (dT_j < 0.5) & (dT_jp1 > 0.5)) |
        ((T_j > 0.5) & (T_jp1 < 0.5) & (dT_j > 0.5) & (dT_jp1 > 0.5)) |
        ((T_j < 0.5) & (T_jp1 > 0.5) & (dT_j < 0.5) & (dT_jp1 < 0.5))
    )
    return jnp.where(is_star, 0.0, 1.0)


# ============================================================
# zeta_power_term — power-2 limb darkening kernel
# ============================================================
def _zeta_power_term(zp, alpha):
    """Compute (1 - zp^(alpha+2)) / ((alpha+2) * (1 - zp^2))."""
    a = alpha + 2.0
    omzp_sq = 1.0 - zp * zp
    safe = jnp.abs(omzp_sq) >= 1e-10
    # Use safe denominator to avoid NaN in gradient.
    safe_denom = jnp.where(safe, omzp_sq, 1.0)
    result = (1.0 - jnp.power(jnp.maximum(zp, 0.0), a)) / (a * safe_denom)
    return jnp.where(safe, result, 0.5)


# ============================================================
# Planet-limb segment: Gauss-Legendre quadrature for s0 and salpha
# ============================================================
def _planet_segment_quad(theta_j, theta_jp1, d, nu, r0, a1, b1, dd, alpha):
    """Integrate s0 and salpha over a planet-limb segment [theta_j, theta_jp1]."""
    omdd = 1.0 - dd
    half_range = (theta_jp1 - theta_j) * 0.5
    t_k = half_range * (_GL_ROOTS + 1.0) + theta_j  # shape (50,)

    cos_tk = jnp.cos(t_k)
    sin_tk = jnp.sin(t_k)
    rp = r0 + a1 * cos_tk + b1 * sin_tk
    rp_sq = rp * rp
    drp = -a1 * sin_tk + b1 * cos_tk

    cos_nu = jnp.cos(nu)
    sin_nu = jnp.sin(nu)
    cos_tmnu = cos_tk * cos_nu + sin_tk * sin_nu
    sin_tmnu = sin_tk * cos_nu - cos_tk * sin_nu
    d_rp_cos = d * rp * cos_tmnu
    d_drp_sin = d * drp * sin_tmnu

    eta = rp_sq - d_rp_cos - d_drp_sin

    s0 = half_range * jnp.sum(0.5 * eta * _GL_WEIGHTS)

    zp_sq = jnp.maximum(omdd - rp_sq + 2.0 * d_rp_cos, 0.0)
    zp = _safe_sqrt(zp_sq)
    zeta = jax.vmap(lambda z: _zeta_power_term(z, alpha))(zp)
    salpha = half_range * jnp.sum(zeta * eta * _GL_WEIGHTS)

    return s0, salpha


# ============================================================
# Star-limb segment: closed-form integrals
# ============================================================
def _star_segment(theta_j, theta_jp1, d, nu, r0, a1, b1, alpha):
    """Compute s0 and salpha for a star-limb segment."""
    rp_j = r0 + a1 * jnp.cos(theta_j) + b1 * jnp.sin(theta_j)
    sin_jmnu = jnp.sin(theta_j - nu)
    cos_jmnu = jnp.cos(theta_j - nu)
    phi_j = jnp.arctan2(-rp_j * sin_jmnu, -rp_j * cos_jmnu + d)

    rp_jp1 = r0 + a1 * jnp.cos(theta_jp1) + b1 * jnp.sin(theta_jp1)
    sin_jp1mnu = jnp.sin(theta_jp1 - nu)
    cos_jp1mnu = jnp.cos(theta_jp1 - nu)
    phi_jp1 = jnp.arctan2(-rp_jp1 * sin_jp1mnu, -rp_jp1 * cos_jp1mnu + d)

    phi_diff = phi_jp1 - phi_j
    s0 = 0.5 * phi_diff
    salpha = phi_diff / (alpha + 2.0)
    return s0, salpha


# ============================================================
# Entire planet segment (GL quadrature over full planet boundary)
# ============================================================
def _entire_planet_segment(d, nu, r0, a1, b1, dd, alpha):
    """Compute s0, salpha when the planet is entirely inside the star."""
    return _planet_segment_quad(nu - _HC_PI, nu + _HC_PI,
                                d, nu, r0, a1, b1, dd, alpha)


# ============================================================
# Entire star segment (closed-form over full stellar disc)
# ============================================================
def _entire_star_segment(alpha):
    """Compute s0, salpha when the planet is entirely outside the star.

    The integral covers the full stellar boundary, giving phi_diff = 2*pi.
    """
    s0 = 0.5 * _HC_TWOPI
    salpha = _HC_TWOPI / (alpha + 2.0)
    return s0, salpha


def _planet_segment_moments(
    theta_j, theta_jp1, d, nu, r0, a1, b1, dd, powers, active=True
):
    """Integrate radial intensity moments ``mu**powers`` on a planet segment."""
    omdd = 1.0 - dd
    # Padded segments have zero output, but evaluating them at a true
    # intersection can still create singular internal reverse derivatives.
    # Replace their geometry before quadrature so masking cannot leak a NaN.
    theta_j = jnp.where(active, theta_j, 0.0)
    theta_jp1 = jnp.where(active, theta_jp1, 0.0)
    half_range = (theta_jp1 - theta_j) * 0.5
    t_k = half_range * (_GL_ROOTS + 1.0) + theta_j
    cos_tk = jnp.cos(t_k)
    sin_tk = jnp.sin(t_k)
    rp = r0 + a1 * cos_tk + b1 * sin_tk
    rp_sq = rp * rp
    drp = -a1 * sin_tk + b1 * cos_tk
    cos_nu = jnp.cos(nu)
    sin_nu = jnp.sin(nu)
    cos_tmnu = cos_tk * cos_nu + sin_tk * sin_nu
    sin_tmnu = sin_tk * cos_nu - cos_tk * sin_nu
    d_rp_cos = d * rp * cos_tmnu
    eta = rp_sq - d_rp_cos - d * drp * sin_tmnu
    zp_sq = jnp.maximum(omdd - rp_sq + 2.0 * d_rp_cos, 0.0)
    eta = jnp.where(active, eta, 0.0)
    zp_sq = jnp.where(active, zp_sq, 0.25)
    zp = _safe_sqrt(zp_sq)

    weighted_eta = eta * _GL_WEIGHTS
    s0 = half_range * 0.5 * jnp.sum(weighted_eta)

    if powers.shape[0] == 2:
        # Power-2 requests [mu^0, mu^alpha]. The zero-order zeta is exactly
        # 1/2, so only the alpha moment needs the generic power evaluation.
        zeta_alpha = jax.vmap(
            lambda z: _zeta_power_term(z, powers[1])
        )(zp)
        s_alpha = half_range * jnp.sum(zeta_alpha * weighted_eta)
        return jnp.stack((s0, s_alpha))

    if powers.shape[0] == 3:
        # Quadratic requests [mu^0, mu^1, mu^2]. In addition to zeta_0=1/2,
        # zeta_2=(1+mu^2)/4 is exact, leaving only zeta_1 generic.
        zeta_1 = (zp + 1.0 / (1.0 + zp)) / 3.0
        s1 = half_range * jnp.sum(zeta_1 * weighted_eta)
        s2 = half_range * jnp.sum(0.25 * (1.0 + zp * zp) * weighted_eta)
        return jnp.stack((s0, s1, s2))

    def integrate_power(power):
        zeta = jax.vmap(lambda z: _zeta_power_term(z, power))(zp)
        return half_range * jnp.sum(zeta * weighted_eta)

    return jax.vmap(integrate_power)(powers)


def _star_segment_moments(
    theta_j, theta_jp1, d, nu, r0, a1, b1, powers, active=True
):
    """Closed-form radial intensity moments on a stellar-limb segment."""
    theta_j = jnp.where(active, theta_j, 0.0)
    theta_jp1 = jnp.where(active, theta_jp1, 0.0)
    cos_nu = jnp.cos(nu)
    sin_nu = jnp.sin(nu)
    cos_j = jnp.cos(theta_j)
    sin_j = jnp.sin(theta_j)
    cos_jmnu = cos_j * cos_nu + sin_j * sin_nu
    sin_jmnu = sin_j * cos_nu - cos_j * sin_nu
    rp_j = r0 + a1 * cos_j + b1 * sin_j
    phi_j_y = -rp_j * sin_jmnu
    phi_j_x = -rp_j * cos_jmnu + d
    phi_j = jnp.arctan2(
        jnp.where(active, phi_j_y, 0.0),
        jnp.where(active, phi_j_x, 1.0),
    )
    cos_jp1 = jnp.cos(theta_jp1)
    sin_jp1 = jnp.sin(theta_jp1)
    cos_jp1mnu = cos_jp1 * cos_nu + sin_jp1 * sin_nu
    sin_jp1mnu = sin_jp1 * cos_nu - cos_jp1 * sin_nu
    rp_jp1 = r0 + a1 * cos_jp1 + b1 * sin_jp1
    phi_jp1_y = -rp_jp1 * sin_jp1mnu
    phi_jp1_x = -rp_jp1 * cos_jp1mnu + d
    phi_jp1 = jnp.arctan2(
        jnp.where(active, phi_jp1_y, 0.0),
        jnp.where(active, phi_jp1_x, 1.0),
    )
    return (phi_jp1 - phi_j) / (powers + 2.0)


def _compute_occulted_moments(d, z, nu, r0, a1, b1, powers):
    """Return the occulted radial moments for one projected configuration."""
    dd = d * d
    amplitude = _safe_sqrt(a1 * a1 + b1 * b1)
    min_rp = r0 - amplitude
    max_rp = r0 + amplitude
    inside = d <= 1.0
    triv_entire_planet = inside & (max_rp <= 1.0 - d + _GEOMETRY_TOL)
    triv_entire_star = jnp.where(
        inside,
        min_rp >= 1.0 + d - _GEOMETRY_TOL,
        min_rp >= d + 1.0 - _GEOMETRY_TOL,
    )
    triv_beyond = (~inside) & (max_rp <= d - 1.0 + _GEOMETRY_TOL)

    thetas, _valid, n_roots = _find_intersections(
        d, nu, r0, a1, b1, dd
    )
    rp_nu = r0 + a1 * jnp.cos(nu) + b1 * jnp.sin(nu)
    # A radial planet domain contains its own centre. Therefore, when d>1 it
    # cannot be wholly inside the stellar disc. If a narrow external root pair
    # is missed near tangency, "beyond" is the bounded limiting fallback;
    # treating it as an entire-planet segment creates a full-depth spike.
    nr0_entire_planet = inside & (rp_nu < 1.0 + d)
    nr0_beyond = ~inside
    nr0_entire_star = ~nr0_entire_planet & ~nr0_beyond
    # Closed curves have an even number of transverse intersections. An odd
    # count is an exact tangency (or grid-endpoint degeneracy), for which the
    # no-root topology is the stable limiting classification.
    has_roots = (n_roots >= 2) & ((n_roots % 2) == 0)
    safe_nr = jnp.maximum(n_roots, 1)
    padded = jnp.where(jnp.arange(4) < n_roots, thetas, thetas[0])
    starts_with_roots = padded
    ends_raw = jnp.roll(padded, -1)
    is_last = jnp.arange(4) == (safe_nr - 1)
    ends_with_roots = jnp.where(
        jnp.arange(4) < n_roots,
        jnp.where(is_last, padded[0] + _HC_TWOPI, ends_raw),
        starts_with_roots,
    )
    # Reuse the first padded quadrature slot for the full-planet fallback.
    # This removes a fifth, separately evaluated 50-point segment while
    # retaining exactly the same integration rule and limits.
    fallback_slot = jnp.arange(4) == 0
    starts = jnp.where(
        has_roots,
        starts_with_roots,
        jnp.where(fallback_slot, nu - _HC_PI, 0.0),
    )
    ends = jnp.where(
        has_roots,
        ends_with_roots,
        jnp.where(fallback_slot, nu + _HC_PI, 0.0),
    )
    active = jnp.where(has_roots, jnp.arange(4) < n_roots, fallback_slot)

    T_starts, dT_starts = jax.vmap(
        lambda theta: _compute_T_dT(theta, d, nu, r0, a1, b1, dd)
    )(starts)
    T_ends, dT_ends = jax.vmap(
        lambda theta: _compute_T_dT(theta, d, nu, r0, a1, b1, dd)
    )(ends)
    is_planet = jax.vmap(_classify_segment)(
        T_starts, dT_starts, T_ends, dT_ends
    )

    def segment_moments(start, end, is_active):
        planet = _planet_segment_moments(
            start, end, d, nu, r0, a1, b1, dd, powers, active=is_active
        )
        star = _star_segment_moments(
            start, end, d, nu, r0, a1, b1, powers, active=is_active
        )
        return planet, star

    planet_all, star_all = jax.vmap(segment_moments)(starts, ends, active)
    segment_all = jnp.where(is_planet[:, None] > 0.5, planet_all, star_all)
    intersect = jnp.sum(jnp.where(active[:, None], segment_all, 0.0), axis=0)
    entire_planet = planet_all[0]
    entire_star = _HC_TWOPI / (powers + 2.0)
    no_root = jnp.where(
        nr0_entire_planet,
        entire_planet,
        jnp.where(nr0_entire_star, entire_star, jnp.zeros_like(powers)),
    )
    moments = jnp.where(
        triv_entire_planet,
        entire_planet,
        jnp.where(
            triv_entire_star,
            entire_star,
            jnp.where(
                triv_beyond,
                jnp.zeros_like(powers),
                jnp.where(has_roots, intersect, no_root),
            ),
        ),
    )
    is_beyond = triv_beyond | ((~has_roots) & nr0_beyond)
    visible = (z >= 0.0) & (~is_beyond)
    return jnp.where(visible, moments, jnp.zeros_like(moments))


# ============================================================
# Core single-point flux computation
# ============================================================
def _compute_flux_nc1_power2(d, z, nu, c_ld, alpha, r0, a1, b1):
    """Compute transit flux for one time sample given orbit solution."""
    I_0_bt = 1.0 - c_ld + 2.0 * c_ld / (alpha + 2.0)
    I_0 = 1.0 / (_HC_PI * I_0_bt)
    s0, salpha = _compute_occulted_moments(
        d, z, nu, r0, a1, b1, jnp.array([0.0, alpha])
    )
    return 1.0 - I_0 * ((1.0 - c_ld) * s0 + c_ld * salpha)


def _compute_flux_nc1_quadratic(d, z, nu, u1, u2, r0, a1, b1):
    """Compute one N_c=1 transit sample with standard quadratic LD."""
    s0, s1, s2 = _compute_occulted_moments(
        d, z, nu, r0, a1, b1, jnp.array([0.0, 1.0, 2.0])
    )
    p0 = 1.0 - u1 - u2
    p1 = u1 + 2.0 * u2
    p2 = -u2
    norm = _HC_PI * (1.0 - u1 / 3.0 - u2 / 6.0)
    return 1.0 - (p0 * s0 + p1 * s1 + p2 * s2) / norm


# ============================================================
# Full transit computation: orbit + flux for a single sample
# ============================================================
def _transit_flux_single_raw(time, t0, period, a_rs, inc, ecc, omega,
                             c_ld, alpha_ld, r0, a1, b1):
    """Compute transit flux for a single time point (raw, may have NaN grads)."""
    d, z, nu = _compute_orbit(time, t0, period, a_rs, inc, ecc, omega)
    return _compute_flux_nc1_power2(d, z, nu, c_ld, alpha_ld, r0, a1, b1)


def _transit_flux_single_quadratic_raw(
    time, t0, period, a_rs, inc, ecc, omega, u1, u2, r0, a1, b1
):
    d, z, nu = _compute_orbit(time, t0, period, a_rs, inc, ecc, omega)
    return _compute_flux_nc1_quadratic(d, z, nu, u1, u2, r0, a1, b1)


# ============================================================
# Public API
# ============================================================
def _harmonica_transit_power2_nc1_single_curve(times_1d, t0, period, a_rs, inc,
                                                ecc, omega, c, alpha, r0, a1, b1):
    """Compute transit for a single light curve (1D times, scalar params)."""
    d, z, nu = _compute_orbit(
        times_1d, t0, period, a_rs, inc, ecc, omega
    )
    return jax.vmap(
        lambda d_i, z_i, nu_i: _compute_flux_nc1_power2(
            d_i, z_i, nu_i, c, alpha, r0, a1, b1
        )
    )(d, z, nu)


def _channel_count(values):
    sizes = [
        jnp.asarray(value).shape[0]
        for value in values
        if jnp.asarray(value).ndim > 0
    ]
    return max(sizes, default=None)


def _broadcast_channel_values(values, n_channels):
    return tuple(
        jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), (n_channels,))
        for value in values
    )


def _harmonica_transit_power2_nc1_shared_times(
    times_1d, t0, period, a_rs, inc, ecc, omega, c, alpha, r0, a1, b1
):
    """Evaluate channels while computing their shared orbit only once."""
    d, z, nu = _compute_orbit(
        times_1d, t0, period, a_rs, inc, ecc, omega
    )

    def compute_curve(c_i, alpha_i, r0_i, a1_i, b1_i):
        return jax.vmap(
            lambda d_i, z_i, nu_i: _compute_flux_nc1_power2(
                d_i, z_i, nu_i, c_i, alpha_i, r0_i, a1_i, b1_i
            )
        )(d, z, nu)

    return jax.vmap(compute_curve)(c, alpha, r0, a1, b1)


def harmonica_transit_power2_nc1_jax(times, t0, period, a_rs, inc, ecc, omega,
                                     c, alpha, r0, a1, b1):
    """Pure JAX transit model for N_c=1, power-2 limb darkening.

    All inputs are JAX arrays (scalars or broadcastable).  The function is
    fully JIT-compilable and differentiable via ``jax.grad``.

    Supports batched evaluation: if ``times`` has shape ``(n_lcs, N)`` and
    per-channel parameters (``c``, ``alpha``, ``r0``, ``a1``, ``b1``) have
    shape ``(n_lcs,)``, the function automatically vmaps over the channel
    axis and returns shape ``(n_lcs, N)``.

    Parameters
    ----------
    times : array, shape (N,) or (n_lcs, N)
        Observation times.  2D triggers batched evaluation.
    t0 : scalar
        Mid-transit time.
    period : scalar
        Orbital period.
    a_rs : scalar
        Semi-major axis in units of stellar radii.
    inc : scalar
        Orbital inclination (radians).
    ecc : scalar
        Orbital eccentricity.
    omega : scalar
        Argument of periastron (radians).
    c : scalar or array (n_lcs,)
        Power-2 limb darkening coefficient c.
    alpha : scalar or array (n_lcs,)
        Power-2 limb darkening exponent alpha.
    r0 : scalar or array (n_lcs,)
        Mean planet radius in stellar radii.
    a1 : scalar or array (n_lcs,)
        Cosine coefficient of the first-order planet shape.
    b1 : scalar or array (n_lcs,)
        Sine coefficient of the first-order planet shape.

    Returns
    -------
    flux : array, shape (N,) or (n_lcs, N)
        Normalised transit flux at each time.
    """
    times = jnp.asarray(times, dtype=jnp.float64)

    if times.ndim <= 1:
        n_channels = _channel_count((c, alpha, r0, a1, b1))
        if n_channels is not None:
            c, alpha, r0, a1, b1 = _broadcast_channel_values(
                (c, alpha, r0, a1, b1), n_channels
            )
            return _harmonica_transit_power2_nc1_shared_times(
                jnp.atleast_1d(times), t0, period, a_rs, inc, ecc, omega,
                c, alpha, r0, a1, b1,
            )
        return _harmonica_transit_power2_nc1_single_curve(
            jnp.atleast_1d(times), t0, period, a_rs, inc, ecc, omega,
            c, alpha, r0, a1, b1,
        )

    # Batched: times is (n_lcs, n_times).  Vmap over channel axis for all
    # per-channel params; orbital params shared across channels.
    c, alpha, r0, a1, b1 = _broadcast_channel_values(
        (c, alpha, r0, a1, b1), times.shape[0]
    )
    return jax.vmap(
        _harmonica_transit_power2_nc1_single_curve,
        in_axes=(0, None, None, None, None, None, None, 0, 0, 0, 0, 0),
    )(times, t0, period, a_rs, inc, ecc, omega, c, alpha, r0, a1, b1)


def _harmonica_transit_quadratic_nc1_single_curve(
    times_1d, t0, period, a_rs, inc, ecc, omega, u1, u2, r0, a1, b1
):
    d, z, nu = _compute_orbit(
        times_1d, t0, period, a_rs, inc, ecc, omega
    )
    return jax.vmap(
        lambda d_i, z_i, nu_i: _compute_flux_nc1_quadratic(
            d_i, z_i, nu_i, u1, u2, r0, a1, b1
        )
    )(d, z, nu)


def _harmonica_transit_quadratic_nc1_shared_times(
    times_1d, t0, period, a_rs, inc, ecc, omega, u1, u2, r0, a1, b1
):
    """Evaluate quadratic-LD channels with one shared orbit calculation."""
    d, z, nu = _compute_orbit(
        times_1d, t0, period, a_rs, inc, ecc, omega
    )

    def compute_curve(u1_i, u2_i, r0_i, a1_i, b1_i):
        return jax.vmap(
            lambda d_i, z_i, nu_i: _compute_flux_nc1_quadratic(
                d_i, z_i, nu_i, u1_i, u2_i, r0_i, a1_i, b1_i
            )
        )(d, z, nu)

    return jax.vmap(compute_curve)(u1, u2, r0, a1, b1)


def harmonica_transit_quadratic_nc1_jax(
    times, t0, period, a_rs, inc, ecc, omega, u1, u2, r0, a1, b1
):
    """Pure-JAX N_c=1 Harmonica transit with direct quadratic ``u1,u2``."""
    times = jnp.asarray(times, dtype=jnp.float64)
    if times.ndim <= 1:
        n_channels = _channel_count((u1, u2, r0, a1, b1))
        if n_channels is not None:
            u1, u2, r0, a1, b1 = _broadcast_channel_values(
                (u1, u2, r0, a1, b1), n_channels
            )
            return _harmonica_transit_quadratic_nc1_shared_times(
                jnp.atleast_1d(times), t0, period, a_rs, inc, ecc, omega,
                u1, u2, r0, a1, b1,
            )
        return _harmonica_transit_quadratic_nc1_single_curve(
            jnp.atleast_1d(times),
            t0,
            period,
            a_rs,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r0,
            a1,
            b1,
        )
    u1, u2, r0, a1, b1 = _broadcast_channel_values(
        (u1, u2, r0, a1, b1), times.shape[0]
    )
    return jax.vmap(
        _harmonica_transit_quadratic_nc1_single_curve,
        in_axes=(0, None, None, None, None, None, None, 0, 0, 0, 0, 0),
    )(times, t0, period, a_rs, inc, ecc, omega, u1, u2, r0, a1, b1)
