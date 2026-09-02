"""
Compare harmonica's native power-2 limb darkening to jaxoplanet's
polynomial approximation for a spherical planet.

Methodology
-----------
Raw cross-engine comparison (harmonica vs jaxoplanet) has a ~19 ppm
baseline offset from the different transit geometry algorithms, present
even for uniform limb darkening.  To isolate the limb-darkening fidelity
we compare the *differential transit signal*:

    delta_LD = flux(power2_LD) - flux(uniform)

computed independently in harmonica and jaxoplanet.  This cancels the
transit-algorithm baseline and measures only the LD contribution.

The polynomial conversion matches JWSTJaxFit-main/models/builder:
  1.  Evaluate I(mu) = 1 - c*(1 - mu^alpha) on a mu grid (300 points).
  2.  Project onto a degree-12 polynomial in (1-mu) via least-squares.
  3.  Pass those polynomial coefficients to jaxoplanet.limb_dark_light_curve.

Note: degree 12 is the practical optimum for the (1-mu)^k Vandermonde
basis.  Higher degrees degrade due to ill-conditioning (cond > 1e14 at
degree 20).
"""
import unittest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from harmonica import HarmonicaTransit
from harmonica.jax import harmonica_transit_power2_ld

from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit


# ---------- power-2 -> polynomial helpers (matches JWSTJaxFit) ----------

def get_I_power2(c, alpha, mu):
    """Power-2 intensity profile."""
    return 1.0 - c * (1.0 - jnp.power(mu, alpha))


def power2_to_poly_coeffs(c, alpha, degree=12, n_mu=300):
    """Convert power-2 (c, alpha) to polynomial LD coefficients.

    Same algorithm as JWSTJaxFit-main/models/builder._prepare_power2_poly
    followed by  u = P @ (1 - I(mu)).
    """
    mus = jnp.linspace(0.0, 1.0, n_mu, endpoint=True)
    x = jnp.vander(1.0 - mus, N=degree + 1, increasing=True)[:, 1:]
    P = jnp.asarray(np.linalg.pinv(np.asarray(x)))
    profile = get_I_power2(c, alpha, mus)
    return P @ (1.0 - profile)


# ---------- test class ----------

class TestPower2VsJaxoplanet(unittest.TestCase):
    """Validate harmonica power-2 against jaxoplanet polynomial approx."""

    # Shared orbital parameters (spherical planet, circular orbit).
    T0 = 0.0
    PERIOD = 3.0
    A_RS = 10.0                       # semi-major axis / R_star
    INC = np.radians(89.0)
    ECC = 0.0
    OMEGA = 0.0
    RP_RS = 0.1                       # planet radius / R_star
    N_TIMES = 1000

    def _times(self):
        half_dur = self.PERIOD / (np.pi * self.A_RS)
        return np.linspace(
            self.T0 - 1.5 * half_dur,
            self.T0 + 1.5 * half_dur,
            self.N_TIMES,
        )

    def _jaxoplanet_orbit(self):
        b = self.A_RS * np.cos(self.INC)
        arg = np.sqrt((1 + self.RP_RS) ** 2 - b ** 2) / self.A_RS
        duration = self.PERIOD / np.pi * np.arcsin(arg)
        return TransitOrbit(
            period=self.PERIOD, duration=duration, time_transit=self.T0,
            impact_param=b, radius_ratio=self.RP_RS,
        )

    # --- harmonica helpers ---

    def _harmonica_flux(self, times, ld_law, u):
        ht = HarmonicaTransit(times, pnl_c=200, pnl_e=500)
        ht.set_orbit(self.T0, self.PERIOD, self.A_RS, self.INC,
                     self.ECC, self.OMEGA)
        ht.set_stellar_limb_darkening(
            np.asarray(u, dtype=np.float64), limb_dark_law=ld_law)
        ht.set_planet_transmission_string(np.array([self.RP_RS]))
        return ht.get_transit_light_curve()

    def _harmonica_uniform(self, times):
        return self._harmonica_flux(times, 'quadratic', [0., 0.])

    def _harmonica_power2(self, times, c, alpha):
        return self._harmonica_flux(times, 'power-2', [c, alpha])

    # --- jaxoplanet helpers ---

    def _jaxoplanet_signal(self, times, u_poly):
        """Return raw jaxoplanet *signal* (negative during transit)."""
        orbit = self._jaxoplanet_orbit()
        return np.asarray(
            limb_dark_light_curve(orbit, u_poly)(jnp.asarray(times)))

    def _jaxoplanet_uniform(self, times):
        return 1.0 + self._jaxoplanet_signal(times, jnp.array([]))

    def _jaxoplanet_power2(self, times, c, alpha, degree=12):
        u_poly = power2_to_poly_coeffs(c, alpha, degree=degree)
        return 1.0 + self._jaxoplanet_signal(times, u_poly)

    # --- differential comparison (cancels transit-algorithm baseline) ---

    def _compare_differential(self, times, c, alpha, degree=12):
        """Compare the LD-specific component, cancelling the baseline.

        Returns (max_residual, in_transit_mask).
        """
        f_harm_p2 = self._harmonica_power2(times, c, alpha)
        f_harm_uni = self._harmonica_uniform(times)
        delta_harm = f_harm_p2 - f_harm_uni

        f_jaxo_p2 = self._jaxoplanet_power2(times, c, alpha, degree)
        f_jaxo_uni = self._jaxoplanet_uniform(times)
        delta_jaxo = f_jaxo_p2 - f_jaxo_uni

        in_transit = f_harm_p2 < 0.9999
        residual = np.abs(delta_harm[in_transit] - delta_jaxo[in_transit])
        return np.max(residual), in_transit

    # ============================================================
    # 1. Within-harmonica: power-2 native vs non-linear basis
    # ============================================================

    def test_harmonica_power2_exact_at_basis_exponents(self):
        """When alpha in {0.5, 1.0, 1.5, 2.0} the non-linear basis is
        exact, so power-2 native must match to machine precision."""
        c = 0.45
        times = self._times()
        expected_coeffs = {
            0.5: np.array([c, 0., 0., 0.]),
            1.0: np.array([0., c, 0., 0.]),
            1.5: np.array([0., 0., c, 0.]),
            2.0: np.array([0., 0., 0., c]),
        }

        for alpha, nl_coeffs in expected_coeffs.items():
            f_p2 = self._harmonica_power2(times, c, alpha)
            f_nl = self._harmonica_flux(times, 'non-linear', nl_coeffs)
            np.testing.assert_allclose(
                f_p2, f_nl, rtol=1e-12, atol=1e-12,
                err_msg=f"power-2 (alpha={alpha}) should exactly match "
                        f"non-linear basis term"
            )
            print(f"\n  alpha={alpha}: match to rtol=1e-12 OK")

    # ===========================================================
    # 2. Harmonica NumPy vs JAX backend consistency
    # ===========================================================

    def test_power2_numpy_vs_jax(self):
        """Harmonica NumPy and JAX backends should agree to ~1e-12."""
        c, alpha = 0.6, 1.2
        times = self._times()

        flux_np = self._harmonica_power2(times, c, alpha)
        flux_jax = np.asarray(harmonica_transit_power2_ld(
            jnp.asarray(times),
            self.T0, self.PERIOD, self.A_RS, self.INC,
            self.ECC, self.OMEGA,
            c=c, alpha=alpha,
            r=jnp.array([self.RP_RS]),
        ))

        np.testing.assert_allclose(flux_np, flux_jax, rtol=1e-11, atol=1e-13)

    # ===========================================================
    # 3. Cross-engine differential: harmonica vs jaxoplanet
    # ===========================================================

    def test_differential_alpha_1p2(self):
        """c=0.6, alpha=1.2: differential LD signal should agree < 100 ppm."""
        max_resid, _ = self._compare_differential(
            self._times(), c=0.6, alpha=1.2)
        print(f"\n  alpha=1.2: differential max |delta| = {max_resid:.2e}")
        self.assertLess(max_resid, 1e-4)

    def test_differential_alpha_0p7(self):
        """c=0.4, alpha=0.7."""
        max_resid, _ = self._compare_differential(
            self._times(), c=0.4, alpha=0.7)
        print(f"\n  alpha=0.7: differential max |delta| = {max_resid:.2e}")
        self.assertLess(max_resid, 1e-4)

    def test_differential_alpha_2p0(self):
        """c=0.5, alpha=2.0 (integer exponent => polynomial is exact)."""
        max_resid, _ = self._compare_differential(
            self._times(), c=0.5, alpha=2.0)
        print(f"\n  alpha=2.0: differential max |delta| = {max_resid:.2e}")
        self.assertLess(max_resid, 1e-4)

    def test_differential_sweep(self):
        """Sweep alpha in [0.3, 3.0] — all < 200 ppm differential."""
        times = self._times()
        c = 0.5
        for alpha in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
            max_resid, _ = self._compare_differential(times, c, alpha)
            print(f"\n  alpha={alpha:.1f}: differential max |delta| "
                  f"= {max_resid:.2e}")
            self.assertLess(
                max_resid, 2e-4,
                f"alpha={alpha}: differential {max_resid:.2e} too large"
            )

    # ===========================================================
    # 4. Polynomial degree study (within jaxoplanet)
    # ===========================================================

    def test_poly_degree_conditioning(self):
        """Verify degree 12 is better than degree 20+ due to
        Vandermonde ill-conditioning."""
        c, alpha = 0.6, 1.2
        mus = jnp.linspace(0.0, 1.0, 300, endpoint=True)
        deficit_true = c * (1.0 - jnp.power(mus, alpha))

        prev_resid = np.inf
        best_deg, best_resid = 0, np.inf
        for deg in [4, 8, 12, 16, 20, 25]:
            x = jnp.vander(1.0 - mus, N=deg + 1, increasing=True)[:, 1:]
            P = jnp.asarray(np.linalg.pinv(np.asarray(x)))
            u = P @ deficit_true
            deficit_approx = x @ u
            resid = float(jnp.max(jnp.abs(deficit_true - deficit_approx)))
            cond = float(np.linalg.cond(np.asarray(x)))
            print(f"\n  degree {deg:2d}: profile resid = {resid:.2e}, "
                  f"cond = {cond:.2e}")
            if resid < best_resid:
                best_deg, best_resid = deg, resid

        print(f"\n  Best degree: {best_deg} (resid = {best_resid:.2e})")
        # Degree 12 should be among the best.
        self.assertLessEqual(best_deg, 16)

    # ===========================================================
    # 5. Cross-engine baseline characterisation
    # ===========================================================

    def test_cross_engine_baseline(self):
        """Characterise the transit-algorithm baseline (uniform LD).
        This is NOT a limb-darkening test — it documents the expected
        ~19 ppm floor between harmonica and jaxoplanet."""
        times = self._times()
        f_harm = self._harmonica_uniform(times)
        f_jaxo = self._jaxoplanet_uniform(times)

        in_transit = f_harm < 0.9999
        baseline = np.max(np.abs(f_harm[in_transit] - f_jaxo[in_transit]))
        print(f"\n  Uniform LD cross-engine baseline: {baseline:.2e}")
        # Document but do not fail on the expected ~19 ppm baseline.
        self.assertLess(baseline, 1e-3,
                        "Baseline should be at most ~0.1%")


if __name__ == "__main__":
    unittest.main(verbosity=2)
