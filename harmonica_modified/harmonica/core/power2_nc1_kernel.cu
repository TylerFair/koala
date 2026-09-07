// ============================================================
// power2_nc1_kernel.cu
// CUDA kernels for power-2 limb darkening with N_c=1
// (first-order odd-cosine asymmetric planet transit model)
//
// Each thread computes the transit flux for one time sample.
// Planet-star intersection angles are found via grid search +
// bisection on f(theta) = rp^2 + d^2 - 2*d*rp*cos(theta-nu) - 1,
// which is robust for all parameter regimes (replaces Laguerre on
// quartic polynomial which failed for small a1 near-circular cases).
// Both s0 and s_alpha use Gauss-Legendre quadrature on planet
// limb segments; star limb segments use closed forms.
// Derivatives use central finite differences of the flux kernel.
// ============================================================

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// ============================================================
// Constants
// ============================================================
#define HC_PI     3.14159265358979323846264338327950288
#define HC_TWOPI  (2.0 * HC_PI)
#define HC_PI_D_2 (HC_PI / 2.0)

#define HC_INTERSECT_TOL  1.0e-7

#define HC_N_LEGENDRE 50

// 50-point Gauss-Legendre quadrature roots and weights in constant memory.
__constant__ double GL_ROOTS[HC_N_LEGENDRE] = {
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
  0.9940319694320907, 0.998866404420071
};

__constant__ double GL_WEIGHTS[HC_N_LEGENDRE] = {
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
  0.006759799195744691, 0.002908622553150225
};



// ============================================================
// Kepler solver (ported from kepler.cpp)
// ============================================================
__device__ void solve_kepler_device(double M, double ecc,
                                    double& out_sinf, double& out_cosf) {
  // Constants for polynomial boundaries.
  const double g2s_e = 0.2588190451025207623489 * ecc;
  const double g3s_e = 0.5 * ecc;
  const double g4s_e = 0.7071067811865475244008 * ecc;
  const double g5s_e = 0.8660254037844386467637 * ecc;
  const double g6s_e = 0.9659258262890682867497 * ecc;

  double one_over_ecc = 1e17;
  if (ecc > 1e-17) one_over_ecc = 1.0 / ecc;

  int MAsign = 1;
  double MA = fmod(M, HC_TWOPI);
  if (MA < 0.0) MA += HC_TWOPI;
  if (MA > HC_PI) { MAsign = -1; MA = HC_TWOPI - MA; }

  double EA;

  if (2.0 * MA + 1.0 - ecc < 0.2) {
    // Series expansion for singular corner.
    double ome = 1.0 - ecc;
    double sqrt_ome = sqrt(ome);
    double chi = MA / (sqrt_ome * ome);
    double Lam = sqrt(8.0 + 9.0 * chi * chi);
    double S = cbrt(Lam + 3.0 * chi);
    double sigma = 6.0 * chi / (2.0 + S * S + 4.0 / (S * S));
    double s2 = sigma * sigma;
    double s4 = s2 * s2;
    double denom_ea = 1.0 / (s2 + 2.0);
    EA = sigma * (1.0 + s2 * ome * denom_ea * ((s2 + 20.0) / 60.0
         + s2 * ome * denom_ea * denom_ea * (s2 * s4 + 25.0 * s4
         + 340.0 * s2 + 840.0) / 1400.0));
    EA *= sqrt_ome;
  } else {
    // Polynomial interpolation.
    double bounds[13];
    bounds[0] = 0.0;
    bounds[1]  = HC_PI / 12.0 - g2s_e;
    bounds[2]  = HC_PI / 6.0  - g3s_e;
    bounds[3]  = HC_PI / 4.0  - g4s_e;
    bounds[4]  = HC_PI / 3.0  - g5s_e;
    bounds[5]  = 5.0 * HC_PI / 12.0 - g6s_e;
    bounds[6]  = HC_PI / 2.0  - ecc;
    bounds[7]  = 7.0 * HC_PI / 12.0 - g6s_e;
    bounds[8]  = 2.0 * HC_PI / 3.0  - g5s_e;
    bounds[9]  = 3.0 * HC_PI / 4.0  - g4s_e;
    bounds[10] = 5.0 * HC_PI / 6.0  - g3s_e;
    bounds[11] = 11.0 * HC_PI / 12.0 - g2s_e;
    bounds[12] = HC_PI;

    int k;
    for (k = 11; k > 0; k--) {
      if (MA > bounds[k]) break;
    }

    double EA_tab0 = k * HC_PI / 12.0;
    double EA_tab6 = (k + 1) * HC_PI / 12.0;

    int sign_k = (k >= 6) ? 1 : -1;
    int abs_6mk = (6 - k >= 0) ? (6 - k) : (k - 6);
    int abs_5mk = (5 - k >= 0) ? (5 - k) : (k - 5);

    double x1 = 1.0 / (1.0 - ((6 - k) * HC_PI / 12.0 + sign_k * bounds[abs_6mk]));
    double y1 = -0.5 * (k * HC_PI / 12.0 - bounds[k]);
    double EA_tab1 = x1;
    double EA_tab2 = y1 * x1 * x1 * x1;

    double x2 = 1.0 / (1.0 - ((5 - k) * HC_PI / 12.0 + sign_k * bounds[abs_5mk]));
    double y2 = -0.5 * ((k + 1) * HC_PI / 12.0 - bounds[k + 1]);
    double EA_tab7 = x2;
    double EA_tab8 = y2 * x2 * x2 * x2;

    double idx = 1.0 / (bounds[k + 1] - bounds[k]);
    double B0 = idx * (-EA_tab2 - idx * (EA_tab1 - idx * HC_PI / 12.0));
    double B1 = idx * (-2.0 * EA_tab2 - idx * (EA_tab1 - EA_tab7));
    double B2 = idx * (EA_tab8 - EA_tab2);

    double EA_tab3 = B2 - 4.0 * B1 + 10.0 * B0;
    double EA_tab4 = (-2.0 * B2 + 7.0 * B1 - 15.0 * B0) * idx;
    double EA_tab5 = (B2 - 3.0 * B1 + 6.0 * B0) * idx * idx;

    double dx = MA - bounds[k];
    EA = EA_tab0 + dx * (EA_tab1 + dx * (EA_tab2 + dx * (EA_tab3
         + dx * (EA_tab4 + dx * EA_tab5))));
  }

  // Sine and cosine of EA.
  double sE, cE;
  if (EA < HC_PI / 4.0) {
    sE = sin(EA); cE = sqrt(1.0 - sE * sE);
  } else if (EA > 3.0 * HC_PI / 4.0) {
    sE = sin(HC_PI - EA); cE = -sqrt(1.0 - sE * sE);
  } else {
    cE = sin(HC_PI_D_2 - EA); sE = sqrt(1.0 - cE * cE);
  }

  // Halley's method refinement.
  double num = (MA - EA) * one_over_ecc + sE;
  double den = one_over_ecc - cE;
  double dEA = num * den / (den * den + 0.5 * sE * num);

  double sinE, cosE;
  if (ecc < 0.78 || MA > 0.4) {
    sinE = MAsign * (sE * (1.0 - 0.5 * dEA * dEA) + dEA * cE);
    cosE = cE * (1.0 - 0.5 * dEA * dEA) - dEA * sE;
  } else {
    dEA = num / (den + dEA * (0.5 * sE + (1.0 / 6.0) * cE * dEA));
    sinE = MAsign * (sE * (1.0 - 0.5 * dEA * dEA) + dEA * cE
           * (1.0 - dEA * dEA / 6.0));
    cosE = cE * (1.0 - 0.5 * dEA * dEA) - dEA * sE
           * (1.0 - dEA * dEA / 6.0);
  }

  // True anomaly from eccentric anomaly.
  double denom_ta = 1.0 + cosE;
  if (denom_ta > 1.0e-10) {
    double ome = 1.0 - ecc;
    double tanf2 = sqrt((1.0 + ecc) / ome) * sinE / denom_ta;
    double tanf2_sq = tanf2 * tanf2;
    double inv_denom = 1.0 / (1.0 + tanf2_sq);
    out_sinf = 2.0 * tanf2 * inv_denom;
    out_cosf = (1.0 - tanf2_sq) * inv_denom;
  } else {
    out_sinf = 0.0;
    out_cosf = -1.0;
  }
}


// ============================================================
// Orbit computation (ported from trajectories.cpp)
// ============================================================
__device__ void compute_orbit_device(
    double time, double t0, double period, double a_rs,
    double inc, double ecc, double omega,
    double& out_d, double& out_z, double& out_nu) {

  double n = HC_TWOPI / period;
  double sin_inc, cos_inc;
  sincos(inc, &sin_inc, &cos_inc);

  if (ecc == 0.0) {
    // Circular orbit.
    double tp = t0 - HC_PI_D_2 / n;
    double M = (time - tp) * n;
    double sin_M, cos_M;
    sincos(M, &sin_M, &cos_M);

    double x = a_rs * cos_M;
    double y = a_rs * cos_inc * sin_M;
    out_z = a_rs * sin_inc * sin_M;

    double psi = cos_inc * atan(-cos_M / sin_M);
    double d_sq = x * x + y * y;
    out_d = sqrt(d_sq);
    out_nu = atan2(y, x) - psi;
  } else {
    // Eccentric orbit.
    double sin_omega, cos_omega;
    sincos(omega, &sin_omega, &cos_omega);

    double some = sqrt(1.0 - ecc);
    double sope = sqrt(1.0 + ecc);
    double E0 = 2.0 * atan2(some * cos_omega, sope * (1.0 + sin_omega));
    double M0 = E0 - ecc * sin(E0);
    double tp = t0 - M0 / n;
    double M = (time - tp) * n;

    double sinf, cosf;
    solve_kepler_device(M, ecc, sinf, cosf);

    double omes = 1.0 - ecc * ecc;
    double ope_cosf = 1.0 + ecc * cosf;
    double r = a_rs * omes / ope_cosf;

    double sin_fpw = cosf * sin_omega + sinf * cos_omega;
    double cos_fpw = cosf * cos_omega - sinf * sin_omega;

    double x = r * cos_fpw;
    double y = r * cos_inc * sin_fpw;
    out_z = r * sin_inc * sin_fpw;

    double psi = cos_inc * atan(-cos_fpw / sin_fpw);
    double d_sq = x * x + y * y;
    out_d = sqrt(d_sq);
    out_nu = atan2(y, x) - psi;
  }
}


// ============================================================
// Grid search + bisection intersection finder for N_c=1
// ============================================================
// Intersection function: f(theta) = rp(theta)^2 + d^2 - 2*d*rp(theta)*cos(theta - nu) - 1
// where rp(theta) = r0 + a1*cos(theta) + b1*sin(theta).
// Zeros of f correspond to planet-star boundary intersections.

#define HC_GRID_N       128
#define HC_BISECT_ITER  50

__device__ __forceinline__ double intersection_func(
    double theta, double d, double nu, double r0,
    double a1_real, double b1_imag, double dd) {
  double sin_t, cos_t;
  sincos(theta, &sin_t, &cos_t);
  double rp = r0 + a1_real * cos_t + b1_imag * sin_t;
  double rp_sq = rp * rp;
  double cos_tmnu = cos(theta - nu);
  return rp_sq + dd - 2.0 * d * rp * cos_tmnu - 1.0;
}

// Find intersection angles by grid search + bisection.
// Returns number of intersections found (0, 2, or 4 for N_c=1).
__device__ int find_intersections_grid_bisection(
    double d, double nu, double r0, double a1_real, double b1_imag,
    double dd,  // d*d precomputed
    double theta_out[4]) {

  int n_roots = 0;

  // Step 1: Evaluate f(theta) at N_grid equally spaced points on [-pi, pi).
  const double grid_step = HC_TWOPI / HC_GRID_N;
  double f_prev = intersection_func(-HC_PI, d, nu, r0, a1_real, b1_imag, dd);
  double t_prev = -HC_PI;

  for (int i = 1; i <= HC_GRID_N; i++) {
    double t_curr = -HC_PI + i * grid_step;
    // Last point wraps to pi; use same value as -pi for continuity.
    double f_curr;
    if (i == HC_GRID_N) {
      t_curr = HC_PI;
      f_curr = intersection_func(-HC_PI, d, nu, r0, a1_real, b1_imag, dd);
    } else {
      f_curr = intersection_func(t_curr, d, nu, r0, a1_real, b1_imag, dd);
    }

    // Step 2: Detect sign changes.
    if ((f_prev > 0.0 && f_curr < 0.0) || (f_prev < 0.0 && f_curr > 0.0)) {
      // Bracket found: [t_prev, t_curr].
      // Step 3: Bisection refinement.
      double a = t_prev;
      double b = t_curr;
      double fa = f_prev;

      for (int iter = 0; iter < HC_BISECT_ITER; iter++) {
        double mid = 0.5 * (a + b);
        double fmid = intersection_func(mid, d, nu, r0, a1_real, b1_imag, dd);
        if ((fa > 0.0 && fmid > 0.0) || (fa < 0.0 && fmid < 0.0)) {
          a = mid;
          fa = fmid;
        } else {
          b = mid;
        }
      }

      double root = 0.5 * (a + b);
      if (n_roots < 4) {
        theta_out[n_roots++] = root;
      }
    }

    // Also check for exact zeros (f_curr == 0.0).
    if (f_curr == 0.0 && i < HC_GRID_N && n_roots < 4) {
      theta_out[n_roots++] = t_curr;
    }

    t_prev = t_curr;
    f_prev = f_curr;
  }

  return n_roots;
}


// ============================================================
// Bubble sort for small arrays (up to 4 elements + 1).
// ============================================================
__device__ void sort_ascending(double arr[], int n) {
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        double tmp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = tmp;
      }
    }
  }
}


// ============================================================
// zeta_power_term for power-2 limb darkening
// ============================================================
__device__ double zeta_power_term(double zp, double alpha) {
  double omzp_sq = 1.0 - zp * zp;
  if (fabs(omzp_sq) < 1.0e-10) {
    return 0.5;
  }
  double a = alpha + 2.0;
  return (1.0 - pow(zp, a)) / (a * omzp_sq);
}


// ============================================================
// Core flux computation for N_c=1, power-2 limb darkening
// ============================================================
__device__ double compute_flux_nc1_power2(
    double d, double z, double nu,
    double c_ld, double alpha,
    double r0, double a1_real, double b1_imag) {

  // Planet behind star.
  if (z < 0.0) return 1.0;

  // Normalization: I_0 = 1 / (pi * (1 - c + 2c/(alpha+2)))
  double I_0_bt = 1.0 - c_ld + 2.0 * c_ld / (alpha + 2.0);
  double I_0 = 1.0 / (HC_PI * I_0_bt);

  // Min/max planet radius.
  // r_p(theta) = r0 + a1*cos(theta) + b1*sin(theta)
  // Amplitude of oscillation = sqrt(a1^2 + b1^2)
  double A_osc = sqrt(a1_real * a1_real + b1_imag * b1_imag);
  double min_rp = r0 - A_osc;
  double max_rp = r0 + A_osc;

  // Pre-compute position quantities.
  double dd = d * d;
  double omdd = 1.0 - dd;

  // Solution integrals.
  double s0 = 0.0;
  double salpha = 0.0;

  // ---- Check trivial cases (no obvious intersections) ----
  // These handle cases where the planet is fully inside/outside the star.
  int trivial_case = -1;  // -1 = need to find intersections
  // 0 = entire planet, 1 = entire star, 2 = beyond (no overlap)

  const double geometry_tol = 1.0e-12;
  if (d <= 1.0) {
    if (max_rp <= 1.0 - d + geometry_tol) {
      trivial_case = 0;  // entire planet
    } else if (min_rp >= 1.0 + d - geometry_tol) {
      trivial_case = 1;  // entire star
    }
  } else {
    if (max_rp <= d - 1.0 + geometry_tol) {
      trivial_case = 2;  // beyond
    } else if (min_rp >= d + 1.0 - geometry_tol) {
      trivial_case = 1;  // entire star
    }
  }

  // Theta array and types.
  double thetas[6];  // max 4 intersections + 1 wrap
  int theta_types[5]; // segment types
  int n_thetas = 0;
  int n_segments = 0;

  // Intersection type constants.
  const int SEG_PLANET = 0;
  const int SEG_ENTIRE_PLANET = 1;
  const int SEG_STAR = 2;
  const int SEG_ENTIRE_STAR = 3;
  const int SEG_BEYOND = 4;

  if (trivial_case == 0) {
    // Entire planet.
    thetas[0] = nu - HC_PI;
    thetas[1] = nu + HC_PI;
    theta_types[0] = SEG_ENTIRE_PLANET;
    n_thetas = 2;
    n_segments = 1;
  } else if (trivial_case == 1) {
    // Entire star.
    thetas[0] = -HC_PI;
    thetas[1] = HC_PI;
    theta_types[0] = SEG_ENTIRE_STAR;
    n_thetas = 2;
    n_segments = 1;
  } else if (trivial_case == 2) {
    // Beyond - no overlap.
    return 1.0;
  } else {
    // ---- Find intersections via grid search + bisection ----
    // Handles all cases uniformly: degenerate, 0, 2, or 4 roots.
    double theta_roots[4];
    int n_roots = find_intersections_grid_bisection(
        d, nu, r0, a1_real, b1_imag, dd, theta_roots);

    if (n_roots == 0 || (n_roots % 2) != 0) {
      // No intersections found. Determine trivial configuration.
      double rp_nu = r0 + a1_real * cos(nu) + b1_imag * sin(nu);
      if (d <= 1.0) {
        if (rp_nu < 1.0 + d) {
          thetas[0] = nu - HC_PI;
          thetas[1] = nu + HC_PI;
          theta_types[0] = SEG_ENTIRE_PLANET;
          n_thetas = 2;
          n_segments = 1;
        } else {
          thetas[0] = -HC_PI;
          thetas[1] = HC_PI;
          theta_types[0] = SEG_ENTIRE_STAR;
          n_thetas = 2;
          n_segments = 1;
        }
      } else {
        // A radial planet domain contains its centre, so for d>1 it cannot be
        // wholly inside the star. Missing a very close external root pair must
        // approach the no-overlap limit, not a full-depth planet segment.
        return 1.0;
      }
    } else {
      // Sort roots ascending.
      sort_ascending(theta_roots, n_roots);

      // Copy to thetas array and close the loop.
      for (int i = 0; i < n_roots; i++) {
        thetas[i] = theta_roots[i];
      }
      thetas[n_roots] = theta_roots[0] + HC_TWOPI;
      n_thetas = n_roots + 1;

      // ---- Characterize intersection pairs ----
      n_segments = n_roots;

      // First intersection: compute T and dT for j=0.
      double dcos_0 = d * cos(thetas[0] - nu);
      double dsin_0 = d * sin(thetas[0] - nu);
      double rp_0 = r0 + a1_real * cos(thetas[0]) + b1_imag * sin(thetas[0]);
      int T_j;
      if (d <= 1.0) {
        T_j = 1;  // Always T+ when planet center inside stellar disc.
      } else {
        double disc_0 = dcos_0 * dcos_0 - dd + 1.0;
        double rs_plus_0 = dcos_0 + sqrt(fmax(disc_0, 0.0));
        T_j = (fabs(rp_0 - rs_plus_0) < HC_INTERSECT_TOL) ? 1 : 0;
      }

      double drp_0 = -a1_real * sin(thetas[0]) + b1_imag * cos(thetas[0]);
      double grad_0 = drp_0 + dsin_0;
      double disc_g0 = dcos_0 * dcos_0 - dd + 1.0;
      double frac_0 = (dsin_0 * dcos_0) / sqrt(fmax(disc_g0, 1.0e-30));
      grad_0 += (T_j == 1) ? frac_0 : -frac_0;
      int dT_j = (grad_0 > 0.0) ? 1 : 0;

      for (int j = 0; j < n_roots; j++) {
        // Compute T and dT for j+1.
        double dcos_jp1 = d * cos(thetas[j + 1] - nu);
        double dsin_jp1 = d * sin(thetas[j + 1] - nu);
        double rp_jp1 = r0 + a1_real * cos(thetas[j + 1])
                           + b1_imag * sin(thetas[j + 1]);
        int T_jp1;
        if (d <= 1.0) {
          T_jp1 = 1;
        } else {
          double disc_jp1 = dcos_jp1 * dcos_jp1 - dd + 1.0;
          double rs_plus_jp1 = dcos_jp1 + sqrt(fmax(disc_jp1, 0.0));
          T_jp1 = (fabs(rp_jp1 - rs_plus_jp1) < HC_INTERSECT_TOL) ? 1 : 0;
        }

        double drp_jp1 = -a1_real * sin(thetas[j + 1])
                        + b1_imag * cos(thetas[j + 1]);
        double grad_jp1 = drp_jp1 + dsin_jp1;
        double disc_g_jp1 = dcos_jp1 * dcos_jp1 - dd + 1.0;
        double frac_jp1 = (dsin_jp1 * dcos_jp1)
                          / sqrt(fmax(disc_g_jp1, 1.0e-30));
        grad_jp1 += (T_jp1 == 1) ? frac_jp1 : -frac_jp1;
        int dT_jp1 = (grad_jp1 > 0.0) ? 1 : 0;

        // Lookup table matching CPU characterise_intersection_pairs().
        if      (T_j==1 && T_jp1==1 && dT_j==0 && dT_jp1==1)
          theta_types[j] = SEG_PLANET;
        else if (T_j==1 && T_jp1==1 && dT_j==1 && dT_jp1==0)
          theta_types[j] = SEG_STAR;
        else if (T_j==0 && T_jp1==0 && dT_j==0 && dT_jp1==1)
          theta_types[j] = SEG_STAR;
        else if (T_j==0 && T_jp1==0 && dT_j==1 && dT_jp1==0)
          theta_types[j] = SEG_PLANET;
        else if (T_j==1 && T_jp1==0 && dT_j==0 && dT_jp1==0)
          theta_types[j] = SEG_PLANET;
        else if (T_j==1 && T_jp1==0 && dT_j==1 && dT_jp1==1)
          theta_types[j] = SEG_STAR;
        else if (T_j==0 && T_jp1==1 && dT_j==0 && dT_jp1==0)
          theta_types[j] = SEG_STAR;
        else if (T_j==0 && T_jp1==1 && dT_j==1 && dT_jp1==1)
          theta_types[j] = SEG_PLANET;
        else
          theta_types[j] = SEG_PLANET;  // Fallback.

        // Advance: cache j+1 values for next iteration.
        T_j = T_jp1;
        dT_j = dT_jp1;
      }
    }
  }

  // ---- Compute line integrals for each segment ----
  for (int seg = 0; seg < n_segments; seg++) {
    double theta_j = thetas[seg];
    double theta_jp1 = thetas[seg + 1];
    int seg_type = theta_types[seg];

    if (seg_type == SEG_PLANET || seg_type == SEG_ENTIRE_PLANET) {
      // ---- Planet limb segment: Gauss-Legendre quadrature ----
      double half_range = (theta_jp1 - theta_j) * 0.5;
      double s0_seg = 0.0;
      double salpha_seg = 0.0;

      for (int k = 0; k < HC_N_LEGENDRE; k++) {
        double t_k = half_range * (GL_ROOTS[k] + 1.0) + theta_j;

        // Planet radius and derivative at t_k.
        double cos_tk, sin_tk;
        sincos(t_k, &sin_tk, &cos_tk);
        double rp_tk = r0 + a1_real * cos_tk + b1_imag * sin_tk;
        double rp_sq = rp_tk * rp_tk;
        double drp_dtk = -a1_real * sin_tk + b1_imag * cos_tk;

        // Geometric quantities.
        double sin_tmnu, cos_tmnu;
        sincos(t_k - nu, &sin_tmnu, &cos_tmnu);
        double d_rp_cos = d * rp_tk * cos_tmnu;
        double d_drp_sin = d * drp_dtk * sin_tmnu;

        // Integrand factor eta.
        double eta = rp_sq - d_rp_cos - d_drp_sin;

        // s0 integrand: (1/2) * eta
        s0_seg += 0.5 * eta * GL_WEIGHTS[k];

        // s_alpha integrand: zeta_alpha(zp) * eta
        double zp_sq = omdd - rp_sq + 2.0 * d_rp_cos;
        if (zp_sq < 0.0) zp_sq = 0.0;  // Numerical safety.
        double zp = sqrt(zp_sq);
        double zeta = zeta_power_term(zp, alpha);
        salpha_seg += zeta * eta * GL_WEIGHTS[k];
      }

      s0     += half_range * s0_seg;
      salpha += half_range * salpha_seg;

    } else if (seg_type == SEG_STAR || seg_type == SEG_ENTIRE_STAR) {
      // ---- Star limb segment: closed-form ----
      double phi_j, phi_jp1;

      if (seg_type == SEG_ENTIRE_STAR) {
        phi_j = -HC_PI;
        phi_jp1 = HC_PI;
      } else {
        // Convert theta to phi (stellar frame).
        double rp_j = r0 + a1_real * cos(theta_j) + b1_imag * sin(theta_j);
        double sin_jmnu, cos_jmnu;
        sincos(theta_j - nu, &sin_jmnu, &cos_jmnu);
        phi_j = atan2(-rp_j * sin_jmnu, -rp_j * cos_jmnu + d);

        double rp_jp1 = r0 + a1_real * cos(theta_jp1)
                            + b1_imag * sin(theta_jp1);
        double sin_jp1mnu, cos_jp1mnu;
        sincos(theta_jp1 - nu, &sin_jp1mnu, &cos_jp1mnu);
        phi_jp1 = atan2(-rp_jp1 * sin_jp1mnu, -rp_jp1 * cos_jp1mnu + d);
      }

      double phi_diff = phi_jp1 - phi_j;
      s0     += 0.5 * phi_diff;
      salpha += phi_diff / (alpha + 2.0);

    }
    // SEG_BEYOND: no contribution.
  }

  // ---- Assemble flux ----
  double flux_alpha = I_0 * ((1.0 - c_ld) * s0 + c_ld * salpha);
  double flux = 1.0 - flux_alpha;

  return flux;
}


// ============================================================
// Full transit flux: orbit + flux for a single sample.
// ============================================================
__device__ double compute_full_flux(
    double time, double t0, double period, double a_rs,
    double inc, double ecc, double omega,
    double c_ld, double alpha_ld,
    double r0, double a1, double b1) {

  double d, z, nu;
  compute_orbit_device(time, t0, period, a_rs, inc, ecc, omega, d, z, nu);
  return compute_flux_nc1_power2(d, z, nu, c_ld, alpha_ld, r0, a1, b1);
}


// ============================================================
// KERNEL: Flux-only for N_c=1 power-2
// ============================================================
// Buffer layout (matching JAX FFI contract):
//   buffers[0]  = int32  n_times
//   buffers[1]  = int32  n_rs
//   buffers[2]  = float64 times[n_samples]
//   buffers[3]  = float64 t0s[n_samples]
//   buffers[4]  = float64 periods[n_samples]
//   buffers[5]  = float64 semi_major_axes[n_samples]
//   buffers[6]  = float64 incs[n_samples]
//   buffers[7]  = float64 eccs[n_samples]
//   buffers[8]  = float64 omegas[n_samples]
//   buffers[9]  = float64 cs[n_samples]
//   buffers[10] = float64 alphas[n_samples]
//   buffers[11] = float64 rs0[n_samples]  (r0)
//   buffers[12] = float64 rs1[n_samples]  (a1)
//   buffers[13] = float64 rs2[n_samples]  (b1)
//   buffers[14] = float64 flux_out[n_samples]
__global__ void power2_nc1_flux_kernel(
    int n_samples,
    const double* __restrict__ times,
    const double* __restrict__ t0s,
    const double* __restrict__ periods,
    const double* __restrict__ semi_major_axes,
    const double* __restrict__ incs,
    const double* __restrict__ eccs,
    const double* __restrict__ omegas,
    const double* __restrict__ cs,
    const double* __restrict__ alphas,
    const double* __restrict__ rs0,
    const double* __restrict__ rs1,
    const double* __restrict__ rs2,
    double* __restrict__ flux_out) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_samples) return;

  flux_out[idx] = compute_full_flux(
    times[idx], t0s[idx], periods[idx], semi_major_axes[idx],
    incs[idx], eccs[idx], omegas[idx],
    cs[idx], alphas[idx],
    rs0[idx], rs1[idx], rs2[idx]);
}


// ============================================================
// KERNEL: Flux + Jacobian via central finite differences
// ============================================================
// Output layout:
//   flux_out[i]                      = transit flux
//   jac_out[i * n_derivs + j]        = df/dparam_j
// where j: 0=t0, 1=period, 2=a, 3=inc, 4=ecc, 5=omega, 6=c, 7=alpha,
//          8=r0, 9=a1, 10=b1
__global__ void power2_nc1_jacobian_kernel(
    int n_samples,
    int n_derivs,
    const double* __restrict__ times,
    const double* __restrict__ t0s,
    const double* __restrict__ periods,
    const double* __restrict__ semi_major_axes,
    const double* __restrict__ incs,
    const double* __restrict__ eccs,
    const double* __restrict__ omegas,
    const double* __restrict__ cs,
    const double* __restrict__ alphas,
    const double* __restrict__ rs0,
    const double* __restrict__ rs1,
    const double* __restrict__ rs2,
    double* __restrict__ flux_out,
    double* __restrict__ jac_out) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_samples) return;

  // Load all parameters.
  double params[11];
  params[0] = t0s[idx];
  params[1] = periods[idx];
  params[2] = semi_major_axes[idx];
  params[3] = incs[idx];
  params[4] = eccs[idx];
  params[5] = omegas[idx];
  params[6] = cs[idx];
  params[7] = alphas[idx];
  params[8] = rs0[idx];
  params[9] = rs1[idx];
  params[10] = rs2[idx];

  double time = times[idx];

  // Compute base flux.
  double f0 = compute_full_flux(time, params[0], params[1], params[2],
                                params[3], params[4], params[5],
                                params[6], params[7],
                                params[8], params[9], params[10]);
  flux_out[idx] = f0;

  // Central finite differences for each parameter.
  const double eps_rel = 1.0e-7;
  const double eps_abs = 1.0e-12;

  for (int j = 0; j < n_derivs; j++) {
    double h = fmax(fabs(params[j]) * eps_rel, eps_abs);

    // Perturb parameter j positively.
    double p_plus[11];
    double p_minus[11];
    for (int k = 0; k < 11; k++) {
      p_plus[k] = params[k];
      p_minus[k] = params[k];
    }
    p_plus[j]  += h;
    p_minus[j] -= h;

    double f_plus = compute_full_flux(time, p_plus[0], p_plus[1], p_plus[2],
                                      p_plus[3], p_plus[4], p_plus[5],
                                      p_plus[6], p_plus[7],
                                      p_plus[8], p_plus[9], p_plus[10]);
    double f_minus = compute_full_flux(time, p_minus[0], p_minus[1], p_minus[2],
                                       p_minus[3], p_minus[4], p_minus[5],
                                       p_minus[6], p_minus[7],
                                       p_minus[8], p_minus[9], p_minus[10]);

    jac_out[idx * n_derivs + j] = (f_plus - f_minus) / (2.0 * h);
  }
}


// ============================================================
// Host launch functions (called from bindings.cpp)
// ============================================================
extern "C" {

void launch_power2_nc1_flux(cudaStream_t stream, void** buffers) {
  // Read metadata from device.
  int n_times = 0;
  int n_rs = 0;
  cudaMemcpyAsync(&n_times, buffers[0], sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&n_rs, buffers[1], sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (n_times <= 0) return;

  // This kernel is for N_c=1 only (r0, a1, b1 -> n_rs == 3).
  if (n_rs != 3) return;

  const int n_samples = n_times;
  const int input_offset = 2;
  const int output_offset = 11 + n_rs;  // After metadata + inputs

  const double* times   = reinterpret_cast<const double*>(buffers[input_offset + 0]);
  const double* t0s     = reinterpret_cast<const double*>(buffers[input_offset + 1]);
  const double* periods = reinterpret_cast<const double*>(buffers[input_offset + 2]);
  const double* a_rs    = reinterpret_cast<const double*>(buffers[input_offset + 3]);
  const double* incs    = reinterpret_cast<const double*>(buffers[input_offset + 4]);
  const double* eccs    = reinterpret_cast<const double*>(buffers[input_offset + 5]);
  const double* omegas  = reinterpret_cast<const double*>(buffers[input_offset + 6]);
  const double* cs      = reinterpret_cast<const double*>(buffers[input_offset + 7]);
  const double* alphas_ptr = reinterpret_cast<const double*>(buffers[input_offset + 8]);
  const double* rs0     = reinterpret_cast<const double*>(buffers[input_offset + 9]);
  const double* rs1     = reinterpret_cast<const double*>(buffers[input_offset + 10]);
  const double* rs2     = reinterpret_cast<const double*>(buffers[input_offset + 11]);

  double* flux_out = reinterpret_cast<double*>(buffers[output_offset]);

  int block_size = 256;
  int grid_size = (n_samples + block_size - 1) / block_size;

  power2_nc1_flux_kernel<<<grid_size, block_size, 0, stream>>>(
    n_samples, times, t0s, periods, a_rs, incs, eccs, omegas,
    cs, alphas_ptr, rs0, rs1, rs2, flux_out);
}


void launch_power2_nc1_deriv(cudaStream_t stream, void** buffers) {
  // Read metadata from device.
  int n_times = 0;
  int n_rs = 0;
  cudaMemcpyAsync(&n_times, buffers[0], sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&n_rs, buffers[1], sizeof(int),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (n_times <= 0) return;

  // This kernel is for N_c=1 only (r0, a1, b1 -> n_rs == 3).
  if (n_rs != 3) return;

  const int n_samples = n_times;
  const int n_derivs = 6 + 2 + n_rs;  // orbital + LD + ripple
  const int input_offset = 2;
  const int output_offset = 11 + n_rs;

  const double* times   = reinterpret_cast<const double*>(buffers[input_offset + 0]);
  const double* t0s     = reinterpret_cast<const double*>(buffers[input_offset + 1]);
  const double* periods = reinterpret_cast<const double*>(buffers[input_offset + 2]);
  const double* a_rs    = reinterpret_cast<const double*>(buffers[input_offset + 3]);
  const double* incs    = reinterpret_cast<const double*>(buffers[input_offset + 4]);
  const double* eccs    = reinterpret_cast<const double*>(buffers[input_offset + 5]);
  const double* omegas  = reinterpret_cast<const double*>(buffers[input_offset + 6]);
  const double* cs      = reinterpret_cast<const double*>(buffers[input_offset + 7]);
  const double* alphas_ptr = reinterpret_cast<const double*>(buffers[input_offset + 8]);
  const double* rs0     = reinterpret_cast<const double*>(buffers[input_offset + 9]);
  const double* rs1     = reinterpret_cast<const double*>(buffers[input_offset + 10]);
  const double* rs2     = reinterpret_cast<const double*>(buffers[input_offset + 11]);

  double* flux_out = reinterpret_cast<double*>(buffers[output_offset]);
  double* jac_out  = reinterpret_cast<double*>(buffers[output_offset + 1]);

  int block_size = 128;  // Fewer threads: more registers per thread for Jacobian.
  int grid_size = (n_samples + block_size - 1) / block_size;

  power2_nc1_jacobian_kernel<<<grid_size, block_size, 0, stream>>>(
    n_samples, n_derivs,
    times, t0s, periods, a_rs, incs, eccs, omegas,
    cs, alphas_ptr, rs0, rs1, rs2,
    flux_out, jac_out);
}

}  // extern "C"
