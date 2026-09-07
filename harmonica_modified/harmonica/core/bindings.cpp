#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/bindings.hpp"
#include "constants/constants.hpp"
#include "orbit/trajectories.hpp"
#include "orbit/gradients.hpp"
#include "light_curve/fluxes.hpp"
#include "light_curve/gradients.hpp"

#ifdef HARMONICA_ENABLE_CUDA
#include <cuda_runtime_api.h>

// Forward declarations for CUDA kernel launch functions (defined in power2_nc1_kernel.cu)
extern "C" void launch_power2_nc1_flux(cudaStream_t stream, void** buffers);
extern "C" void launch_power2_nc1_deriv(cudaStream_t stream, void** buffers);
#endif

namespace py = pybind11;

// The OpenMP runtime functions used below are declared directly so the
// build does not depend on an omp.h header being present.
#ifdef _OPENMP
extern "C" int omp_get_num_threads(void);
extern "C" int omp_get_thread_num(void);
#endif


namespace {

template <typename Fn>
inline void parallelize_sample_ranges(const int n_samples, const Fn& fn) {
#ifdef _OPENMP
  if (n_samples > 1) {
#pragma omp parallel
    {
      const int n_threads = omp_get_num_threads();
      const int thread_id = omp_get_thread_num();
      const int begin = (n_samples * thread_id) / n_threads;
      const int end = (n_samples * (thread_id + 1)) / n_threads;
      fn(begin, end);
    }
    return;
  }
#endif
  fn(0, n_samples);
}

inline void compute_orbit_trajectory_sample(
    OrbitTrajectories& orbital, const double ecc, const double time,
    double& out_d, double& out_z, double& out_nu) {
  if (ecc == 0.) {
    orbital.compute_circular_orbit(time, out_d, out_z, out_nu);
  } else {
    orbital.compute_eccentric_orbit(time, out_d, out_z, out_nu);
  }
}

}  // namespace


void compute_orbit_trajectories(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  py::array_t<double, py::array::c_style> times_py,
  py::array_t<double, py::array::c_style> out_ds_py,
  py::array_t<double, py::array::c_style> out_zs_py,
  py::array_t<double, py::array::c_style> out_nus_py) {

  const int n_times = static_cast<int>(times_py.size());
  const double* times = times_py.data();
  double* out_ds = out_ds_py.mutable_data();
  double* out_zs = out_zs_py.mutable_data();
  double* out_nus = out_nus_py.mutable_data();

  py::gil_scoped_release release;
  parallelize_sample_ranges(n_times, [&](int begin, int end) {
    OrbitTrajectories orbital(t0, period, a, inc, ecc, omega);
    for (int i = begin; i < end; ++i) {
      compute_orbit_trajectory_sample(
        orbital, ecc, times[i], out_ds[i], out_zs[i], out_nus[i]);
    }
  });
}


void compute_harmonica_light_curve(
  const double t0, const double period, const double a,
  const double inc, const double ecc, const double omega,
  int ld_law,
  py::array_t<double, py::array::c_style> us_py,
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> times_py,
  py::array_t<double, py::array::c_style> out_fs_py,
  int pnl_c, int pnl_e) {

  const int n_us = static_cast<int>(us_py.size());
  const int n_rs = static_cast<int>(rs_py.size());
  const int n_times = static_cast<int>(times_py.size());
  std::vector<double> us(n_us);
  std::vector<double> rs(n_rs);
  for (int i = 0; i < n_us; ++i) {
    us[i] = us_py.data()[i];
  }
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = rs_py.data()[i];
  }
  const double* times = times_py.data();
  double* out_fs = out_fs_py.mutable_data();

  py::gil_scoped_release release;
  parallelize_sample_ranges(n_times, [&](int begin, int end) {
    OrbitTrajectories orbital(t0, period, a, inc, ecc, omega);
    Fluxes flux(ld_law, us.data(), n_rs, rs.data(), pnl_c, pnl_e);
    for (int i = begin; i < end; ++i) {
      double d, z, nu;
      compute_orbit_trajectory_sample(
        orbital, ecc, times[i], d, z, nu);
      flux.transit_flux(d, z, nu, out_fs[i]);
    }
  });
}


void compute_transmission_string(
  py::array_t<double, py::array::c_style> rs_py,
  py::array_t<double, py::array::c_style> thetas_py,
  py::array_t<double, py::array::c_style> out_transmission_string_py) {

  // Unpack python arrays.
  auto rs_py_ = rs_py.unchecked<1>();
  int n_rs = rs_py_.shape(0);
  double us[2] = {0., 0.};
  double rs[n_rs];
  for (int i = 0; i < n_rs; ++i) {
    rs[i] = rs_py_(i);
  }
  auto thetas_py_ = thetas_py.mutable_unchecked<1>();
  auto out_transmission_string_py_ =
    out_transmission_string_py.mutable_unchecked<1>();

  // Compute transmission string.
  Fluxes flux(0, us, n_rs, rs, 0, 0);
  for (int i = 0; i < thetas_py_.shape(0); i++) {
    out_transmission_string_py_(i) = flux.rp_theta(thetas_py_(i));
  }
}


const void jax_light_curve_quad_ld(void* out_tuple, const void** in) {

  // Unpack input meta.
  int n_times = *((int*) in[0]);
  int n_rs = *((int*) in[1]);

  // Unpack input data structures. Each parameter may vary across the
  // flattened sample axis, which enables batched multi-light-curve
  // evaluation when the Python wrapper broadcasts inputs to a common grid.
  double* times = (double*) in[2];
  double* t0s = (double*) in[3];
  double* periods = (double*) in[4];
  double* semi_major_axes = (double*) in[5];
  double* incs = (double*) in[6];
  double* eccs = (double*) in[7];
  double* omegas = (double*) in[8];
  double* u1s = (double*) in[9];
  double* u2s = (double*) in[10];
  std::vector<double*> rs_inputs(n_rs);
  for (int i = 0; i < n_rs; ++i) {
    rs_inputs[i] = (double*) in[i + 11];
  }

  // Unpack output data structures.
  int n_x_derivatives = 6;
  int n_y_derivatives = 2 + 2 + n_rs;
  int n_z_derivatives = 6 + 2 + n_rs;
  void **out = reinterpret_cast<void **>(out_tuple);
  double* f = (double*) out[0];
  double* df_dz = (double*) out[1];

  const double* times_data = times;
  const double* t0s_data = t0s;
  const double* periods_data = periods;
  const double* semi_major_axes_data = semi_major_axes;
  const double* incs_data = incs;
  const double* eccs_data = eccs;
  const double* omegas_data = omegas;
  const double* u1s_data = u1s;
  const double* u2s_data = u2s;
  double* const* rs_inputs_data = rs_inputs.data();

  parallelize_sample_ranges(n_times, [&](int begin, int end) {
    std::unique_ptr<OrbitDerivatives> orbital;
    std::unique_ptr<FluxDerivatives> flux;
    bool orbit_ready = false;
    bool flux_ready = false;
    double current_t0 = 0., current_period = 0., current_a = 0.;
    double current_inc = 0., current_ecc = 0., current_omega = 0.;
    double current_us[2] = {0., 0.};
    std::vector<double> current_rs(n_rs, 0.);
    std::vector<double> rs(n_rs, 0.);

    for (int i = begin; i < end; i++) {
      const double t0 = t0s_data[i];
      const double period = periods_data[i];
      const double a = semi_major_axes_data[i];
      const double inc = incs_data[i];
      const double ecc = eccs_data[i];
      const double omega = omegas_data[i];
      double us[2] = {u1s_data[i], u2s_data[i]};

      const bool orbit_changed = (
        !orbit_ready
        || t0 != current_t0
        || period != current_period
        || a != current_a
        || inc != current_inc
        || ecc != current_ecc
        || omega != current_omega
      );
      if (orbit_changed) {
        orbital = std::make_unique<OrbitDerivatives>(
          t0, period, a, inc, ecc, omega);
        orbit_ready = true;
        current_t0 = t0;
        current_period = period;
        current_a = a;
        current_inc = inc;
        current_ecc = ecc;
        current_omega = omega;
      }

      bool rs_changed = !flux_ready;
      for (int j = 0; j < n_rs; ++j) {
        rs[j] = rs_inputs_data[j][i];
        rs_changed = rs_changed || rs[j] != current_rs[j];
      }

      const bool flux_changed = (
        !flux_ready
        || us[0] != current_us[0]
        || us[1] != current_us[1]
        || rs_changed
      );
      if (flux_changed) {
        flux = std::make_unique<FluxDerivatives>(
          limb_darkening::quadratic, us, n_rs, rs.data(), 20, 50);
        flux_ready = true;
        current_us[0] = us[0];
        current_us[1] = us[1];
        current_rs = rs;
      }

      // Compute orbit and derivatives wrt x={t0, p, a, i, e, w}.
      double d, z, nu;
      double dd_dx[n_x_derivatives], dnu_dx[n_x_derivatives];
      orbital->compute_orbit_and_derivatives(
        times_data[i], d, z, nu, dd_dx, dnu_dx);

      // Compute flux and derivatives wrt y={d, nu, {us}, {rs}}.
      double df_dy[n_y_derivatives];
      flux->transit_flux_and_derivatives(d, z, nu, f[i], df_dy);

      // Compute total derivatives wrt z={t0, p, a, i, e, w, {us}, {rs}}.
      int idx_ravel = i * n_z_derivatives;
      for (int j = 0; j < n_z_derivatives; j++) {
        if (j < 6) {
          df_dz[idx_ravel + j] = df_dy[0] * dd_dx[j] + df_dy[1] * dnu_dx[j];
        } else {
          df_dz[idx_ravel + j] = df_dy[j - 4];
        }
      }
    }
  });
}


const void jax_light_curve_power2_ld(void* out_tuple, const void** in) {

  int n_times = *((int*) in[0]);
  int n_rs = *((int*) in[1]);

  double* times = (double*) in[2];
  double* t0s = (double*) in[3];
  double* periods = (double*) in[4];
  double* semi_major_axes = (double*) in[5];
  double* incs = (double*) in[6];
  double* eccs = (double*) in[7];
  double* omegas = (double*) in[8];
  double* cs = (double*) in[9];
  double* alphas = (double*) in[10];
  std::vector<double*> rs_inputs(n_rs);
  for (int i = 0; i < n_rs; ++i) {
    rs_inputs[i] = (double*) in[i + 11];
  }

  int n_x_derivatives = 6;
  int n_y_derivatives = 2 + 2 + n_rs;
  int n_z_derivatives = 6 + 2 + n_rs;
  void **out = reinterpret_cast<void **>(out_tuple);
  double* f = (double*) out[0];
  double* df_dz = (double*) out[1];

  const double* times_data = times;
  const double* t0s_data = t0s;
  const double* periods_data = periods;
  const double* semi_major_axes_data = semi_major_axes;
  const double* incs_data = incs;
  const double* eccs_data = eccs;
  const double* omegas_data = omegas;
  const double* cs_data = cs;
  const double* alphas_data = alphas;
  double* const* rs_inputs_data = rs_inputs.data();

  parallelize_sample_ranges(n_times, [&](int begin, int end) {
    std::unique_ptr<OrbitDerivatives> orbital;
    std::unique_ptr<FluxDerivatives> flux;
    bool orbit_ready = false;
    bool flux_ready = false;
    double current_t0 = 0., current_period = 0., current_a = 0.;
    double current_inc = 0., current_ecc = 0., current_omega = 0.;
    double current_us[2] = {0., 0.};
    std::vector<double> current_rs(n_rs, 0.);
    std::vector<double> rs(n_rs, 0.);

    for (int i = begin; i < end; i++) {
      const double t0 = t0s_data[i];
      const double period = periods_data[i];
      const double a = semi_major_axes_data[i];
      const double inc = incs_data[i];
      const double ecc = eccs_data[i];
      const double omega = omegas_data[i];
      double us[2] = {cs_data[i], alphas_data[i]};

      const bool orbit_changed = (
        !orbit_ready
        || t0 != current_t0
        || period != current_period
        || a != current_a
        || inc != current_inc
        || ecc != current_ecc
        || omega != current_omega
      );
      if (orbit_changed) {
        orbital = std::make_unique<OrbitDerivatives>(
          t0, period, a, inc, ecc, omega);
        orbit_ready = true;
        current_t0 = t0;
        current_period = period;
        current_a = a;
        current_inc = inc;
        current_ecc = ecc;
        current_omega = omega;
      }

      bool rs_changed = !flux_ready;
      for (int j = 0; j < n_rs; ++j) {
        rs[j] = rs_inputs_data[j][i];
        rs_changed = rs_changed || rs[j] != current_rs[j];
      }

      const bool flux_changed = (
        !flux_ready
        || us[0] != current_us[0]
        || us[1] != current_us[1]
        || rs_changed
      );
      if (flux_changed) {
        flux = std::make_unique<FluxDerivatives>(
          limb_darkening::power_2, us, n_rs, rs.data(), 20, 50);
        flux_ready = true;
        current_us[0] = us[0];
        current_us[1] = us[1];
        current_rs = rs;
      }

      double d, z, nu;
      double dd_dx[n_x_derivatives], dnu_dx[n_x_derivatives];
      orbital->compute_orbit_and_derivatives(
        times_data[i], d, z, nu, dd_dx, dnu_dx);

      double df_dy[n_y_derivatives];
      flux->transit_flux_and_derivatives(d, z, nu, f[i], df_dy);

      int idx_ravel = i * n_z_derivatives;
      for (int j = 0; j < n_z_derivatives; j++) {
        if (j < 6) {
          df_dz[idx_ravel + j] = df_dy[0] * dd_dx[j] + df_dy[1] * dnu_dx[j];
        } else {
          df_dz[idx_ravel + j] = df_dy[j - 4];
        }
      }
    }
  });
}


const void jax_light_curve_power2_ld_flux(void* out, const void** in) {

  int n_times = *((int*) in[0]);
  int n_rs = *((int*) in[1]);

  double* times = (double*) in[2];
  double* t0s = (double*) in[3];
  double* periods = (double*) in[4];
  double* semi_major_axes = (double*) in[5];
  double* incs = (double*) in[6];
  double* eccs = (double*) in[7];
  double* omegas = (double*) in[8];
  double* cs = (double*) in[9];
  double* alphas = (double*) in[10];
  std::vector<double*> rs_inputs(n_rs);
  for (int i = 0; i < n_rs; ++i) {
    rs_inputs[i] = (double*) in[i + 11];
  }

  double* f = reinterpret_cast<double*>(out);

  const double* times_data = times;
  const double* t0s_data = t0s;
  const double* periods_data = periods;
  const double* semi_major_axes_data = semi_major_axes;
  const double* incs_data = incs;
  const double* eccs_data = eccs;
  const double* omegas_data = omegas;
  const double* cs_data = cs;
  const double* alphas_data = alphas;
  double* const* rs_inputs_data = rs_inputs.data();

  parallelize_sample_ranges(n_times, [&](int begin, int end) {
    std::unique_ptr<OrbitTrajectories> orbital;
    std::unique_ptr<Fluxes> flux;
    bool orbit_ready = false;
    bool flux_ready = false;
    double current_t0 = 0., current_period = 0., current_a = 0.;
    double current_inc = 0., current_ecc = 0., current_omega = 0.;
    double current_us[2] = {0., 0.};
    std::vector<double> current_rs(n_rs, 0.);
    std::vector<double> rs(n_rs, 0.);

    for (int i = begin; i < end; i++) {
      const double t0 = t0s_data[i];
      const double period = periods_data[i];
      const double a = semi_major_axes_data[i];
      const double inc = incs_data[i];
      const double ecc = eccs_data[i];
      const double omega = omegas_data[i];
      double us[2] = {cs_data[i], alphas_data[i]};

      const bool orbit_changed = (
        !orbit_ready
        || t0 != current_t0
        || period != current_period
        || a != current_a
        || inc != current_inc
        || ecc != current_ecc
        || omega != current_omega
      );
      if (orbit_changed) {
        orbital = std::make_unique<OrbitTrajectories>(
          t0, period, a, inc, ecc, omega);
        orbit_ready = true;
        current_t0 = t0;
        current_period = period;
        current_a = a;
        current_inc = inc;
        current_ecc = ecc;
        current_omega = omega;
      }

      bool rs_changed = !flux_ready;
      for (int j = 0; j < n_rs; ++j) {
        rs[j] = rs_inputs_data[j][i];
        rs_changed = rs_changed || rs[j] != current_rs[j];
      }

      const bool flux_changed = (
        !flux_ready
        || us[0] != current_us[0]
        || us[1] != current_us[1]
        || rs_changed
      );
      if (flux_changed) {
        flux = std::make_unique<Fluxes>(
          limb_darkening::power_2, us, n_rs, rs.data(), 20, 50);
        flux_ready = true;
        current_us[0] = us[0];
        current_us[1] = us[1];
        current_rs = rs;
      }

      double d, z, nu;
      compute_orbit_trajectory_sample(
        *orbital, ecc, times_data[i], d, z, nu);
      flux->transit_flux(d, z, nu, f[i]);
    }
  });
}


const void jax_light_curve_nonlinear_ld(void* out_tuple, const void** in) {

  // Unpack input meta.
  int n_times = *((int*) in[0]);
  int n_rs = *((int*) in[1]);

  // Unpack input data structures. Each parameter may vary across the
  // flattened sample axis, which enables batched multi-light-curve
  // evaluation when the Python wrapper broadcasts inputs to a common grid.
  double* times = (double*) in[2];
  double* t0s = (double*) in[3];
  double* periods = (double*) in[4];
  double* semi_major_axes = (double*) in[5];
  double* incs = (double*) in[6];
  double* eccs = (double*) in[7];
  double* omegas = (double*) in[8];
  double* u1s = (double*) in[9];
  double* u2s = (double*) in[10];
  double* u3s = (double*) in[11];
  double* u4s = (double*) in[12];
  std::vector<double*> rs_inputs(n_rs);
  for (int i = 0; i < n_rs; ++i) {
    rs_inputs[i] = (double*) in[i + 13];
  }

  // Unpack output data structures.
  int n_x_derivatives = 6;
  int n_y_derivatives = 2 + 4 + n_rs;
  int n_z_derivatives = 6 + 4 + n_rs;
  void **out = reinterpret_cast<void **>(out_tuple);
  double* f = (double*) out[0];
  double* df_dz = (double*) out[1];

  const double* times_data = times;
  const double* t0s_data = t0s;
  const double* periods_data = periods;
  const double* semi_major_axes_data = semi_major_axes;
  const double* incs_data = incs;
  const double* eccs_data = eccs;
  const double* omegas_data = omegas;
  const double* u1s_data = u1s;
  const double* u2s_data = u2s;
  const double* u3s_data = u3s;
  const double* u4s_data = u4s;
  double* const* rs_inputs_data = rs_inputs.data();

  parallelize_sample_ranges(n_times, [&](int begin, int end) {
    std::unique_ptr<OrbitDerivatives> orbital;
    std::unique_ptr<FluxDerivatives> flux;
    bool orbit_ready = false;
    bool flux_ready = false;
    double current_t0 = 0., current_period = 0., current_a = 0.;
    double current_inc = 0., current_ecc = 0., current_omega = 0.;
    double current_us[4] = {0., 0., 0., 0.};
    std::vector<double> current_rs(n_rs, 0.);
    std::vector<double> rs(n_rs, 0.);

    for (int i = begin; i < end; i++) {
      const double t0 = t0s_data[i];
      const double period = periods_data[i];
      const double a = semi_major_axes_data[i];
      const double inc = incs_data[i];
      const double ecc = eccs_data[i];
      const double omega = omegas_data[i];
      double us[4] = {u1s_data[i], u2s_data[i], u3s_data[i], u4s_data[i]};

      const bool orbit_changed = (
        !orbit_ready
        || t0 != current_t0
        || period != current_period
        || a != current_a
        || inc != current_inc
        || ecc != current_ecc
        || omega != current_omega
      );
      if (orbit_changed) {
        orbital = std::make_unique<OrbitDerivatives>(
          t0, period, a, inc, ecc, omega);
        orbit_ready = true;
        current_t0 = t0;
        current_period = period;
        current_a = a;
        current_inc = inc;
        current_ecc = ecc;
        current_omega = omega;
      }

      bool rs_changed = !flux_ready;
      for (int j = 0; j < n_rs; ++j) {
        rs[j] = rs_inputs_data[j][i];
        rs_changed = rs_changed || rs[j] != current_rs[j];
      }

      const bool flux_changed = (
        !flux_ready
        || us[0] != current_us[0]
        || us[1] != current_us[1]
        || us[2] != current_us[2]
        || us[3] != current_us[3]
        || rs_changed
      );
      if (flux_changed) {
        flux = std::make_unique<FluxDerivatives>(
          limb_darkening::non_linear, us, n_rs, rs.data(), 20, 50);
        flux_ready = true;
        current_us[0] = us[0];
        current_us[1] = us[1];
        current_us[2] = us[2];
        current_us[3] = us[3];
        current_rs = rs;
      }

      // Compute orbit and derivatives wrt x={t0, p, a, i, e, w}.
      double d, z, nu;
      double dd_dx[n_x_derivatives], dnu_dx[n_x_derivatives];
      orbital->compute_orbit_and_derivatives(
        times_data[i], d, z, nu, dd_dx, dnu_dx);

      // Compute flux and derivatives wrt y={d, nu, {us}, {rs}}.
      double df_dy[n_y_derivatives];
      flux->transit_flux_and_derivatives(d, z, nu, f[i], df_dy);

      // Compute total derivatives wrt z={t0, p, a, i, e, w, {us}, {rs}}.
      int idx_ravel = i * n_z_derivatives;
      for (int j = 0; j < n_z_derivatives; j++) {
        if (j < 6) {
          df_dz[idx_ravel + j] = df_dy[0] * dd_dx[j] + df_dy[1] * dnu_dx[j];
        } else {
          df_dz[idx_ravel + j] = df_dy[j - 4];
        }
      }
    }
  });
}


template <typename T>
py::capsule encapsulate(T* fn) {
  // JAX callables must be wrapped in this py::capsule.
  return py::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}


py::dict jax_registrations() {
  // Dictionary of JAX callables.
  py::dict dict;
  dict["jax_light_curve_quad_ld"] = encapsulate(
    jax_light_curve_quad_ld);
  dict["jax_light_curve_power2_ld"] = encapsulate(
    jax_light_curve_power2_ld);
  dict["jax_light_curve_power2_ld_flux"] = encapsulate(
    jax_light_curve_power2_ld_flux);
  dict["jax_light_curve_nonlinear_ld"] = encapsulate(
    jax_light_curve_nonlinear_ld);
  return dict;
}


#ifdef HARMONICA_ENABLE_CUDA
namespace {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(what) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
void copy_device_to_host(
    cudaStream_t stream, const T* device, T* host, std::size_t count,
    const char* what) {
  cuda_check(
    cudaMemcpyAsync(
      host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
    what);
}

template <typename T>
void copy_host_to_device(
    cudaStream_t stream, T* device, const T* host, std::size_t count,
    const char* what) {
  cuda_check(
    cudaMemcpyAsync(
      device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream),
    what);
}

void dispatch_power2_ld_gpu(
    cudaStream_t stream, void **buffers, bool with_derivatives) {

  int n_times = 0;
  int n_rs = 0;
  copy_device_to_host(
    stream, reinterpret_cast<const int*>(buffers[0]), &n_times, 1,
    "copy power2 n_times");
  copy_device_to_host(
    stream, reinterpret_cast<const int*>(buffers[1]), &n_rs, 1,
    "copy power2 n_rs");
  cuda_check(
    cudaStreamSynchronize(stream), "synchronize power2 metadata copies");

  if (n_times < 0) {
    throw std::runtime_error("power2 GPU call received a negative n_times");
  }
  if (n_rs < 1) {
    throw std::runtime_error("power2 GPU call requires at least one r coeff");
  }

  // For N_c=1 (n_rs==3), use native CUDA kernels.
  if (n_rs == 3) {
    if (with_derivatives) {
      launch_power2_nc1_deriv(stream, buffers);
    } else {
      launch_power2_nc1_flux(stream, buffers);
    }
    return;
  }

  // Fallback: CPU bridge for other cases.
  const std::size_t n_samples = static_cast<std::size_t>(n_times);
  const int input_offset = 2;
  const int output_offset = 11 + n_rs;

  std::vector<double> times(n_samples);
  std::vector<double> t0s(n_samples);
  std::vector<double> periods(n_samples);
  std::vector<double> semi_major_axes(n_samples);
  std::vector<double> incs(n_samples);
  std::vector<double> eccs(n_samples);
  std::vector<double> omegas(n_samples);
  std::vector<double> cs(n_samples);
  std::vector<double> alphas(n_samples);
  std::vector<std::vector<double>> rs(n_rs, std::vector<double>(n_samples));

  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 0]),
    times.data(), n_samples, "copy power2 times");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 1]),
    t0s.data(), n_samples, "copy power2 t0s");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 2]),
    periods.data(), n_samples, "copy power2 periods");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 3]),
    semi_major_axes.data(), n_samples, "copy power2 semi-major axes");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 4]),
    incs.data(), n_samples, "copy power2 inclinations");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 5]),
    eccs.data(), n_samples, "copy power2 eccentricities");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 6]),
    omegas.data(), n_samples, "copy power2 omegas");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 7]),
    cs.data(), n_samples, "copy power2 c coefficients");
  copy_device_to_host(
    stream, reinterpret_cast<const double*>(buffers[input_offset + 8]),
    alphas.data(), n_samples, "copy power2 alpha coefficients");
  for (int i = 0; i < n_rs; ++i) {
    copy_device_to_host(
      stream, reinterpret_cast<const double*>(buffers[input_offset + 9 + i]),
      rs[i].data(), n_samples,
      "copy power2 ripple coefficients");
  }
  cuda_check(
    cudaStreamSynchronize(stream), "synchronize power2 input copies");

  std::vector<const void*> host_inputs;
  host_inputs.reserve(static_cast<std::size_t>(output_offset));
  host_inputs.push_back(&n_times);
  host_inputs.push_back(&n_rs);
  host_inputs.push_back(times.data());
  host_inputs.push_back(t0s.data());
  host_inputs.push_back(periods.data());
  host_inputs.push_back(semi_major_axes.data());
  host_inputs.push_back(incs.data());
  host_inputs.push_back(eccs.data());
  host_inputs.push_back(omegas.data());
  host_inputs.push_back(cs.data());
  host_inputs.push_back(alphas.data());
  for (int i = 0; i < n_rs; ++i) {
    host_inputs.push_back(rs[i].data());
  }

  if (with_derivatives) {
    std::vector<double> host_flux(n_samples);
    std::vector<double> host_df_dz(
      n_samples * static_cast<std::size_t>(8 + n_rs));
    void* host_outputs[] = {
      host_flux.data(),
      host_df_dz.data(),
    };

    jax_light_curve_power2_ld(
      host_outputs,
      host_inputs.data());

    copy_host_to_device(
      stream,
      reinterpret_cast<double*>(buffers[output_offset + 0]),
      host_flux.data(), n_samples, "copy power2 flux output");
    copy_host_to_device(
      stream,
      reinterpret_cast<double*>(buffers[output_offset + 1]),
      host_df_dz.data(), host_df_dz.size(), "copy power2 derivative output");
  } else {
    std::vector<double> host_flux(n_samples);
    jax_light_curve_power2_ld_flux(
      host_flux.data(),
      host_inputs.data());

    copy_host_to_device(
      stream,
      reinterpret_cast<double*>(buffers[output_offset + 0]),
      host_flux.data(), n_samples, "copy power2 flux output");
  }

  cuda_check(
    cudaStreamSynchronize(stream), "synchronize power2 output copies");
}

const void jax_light_curve_power2_ld_gpu(
    cudaStream_t stream, void **buffers, const char *opaque,
    std::size_t opaque_len) {
  (void)opaque;
  (void)opaque_len;
  dispatch_power2_ld_gpu(stream, buffers, true);
}

const void jax_light_curve_power2_ld_flux_gpu(
    cudaStream_t stream, void **buffers, const char *opaque,
    std::size_t opaque_len) {
  (void)opaque;
  (void)opaque_len;
  dispatch_power2_ld_gpu(stream, buffers, false);
}

}  // namespace
#endif


py::dict jax_gpu_registrations() {
  py::dict dict;
#ifdef HARMONICA_ENABLE_CUDA
  dict["jax_light_curve_power2_ld"] = encapsulate(
    jax_light_curve_power2_ld_gpu);
  dict["jax_light_curve_power2_ld_flux"] = encapsulate(
    jax_light_curve_power2_ld_flux_gpu);
#endif
  return dict;
}


PYBIND11_MODULE(bindings, m) {

  m.def("orbit", &compute_orbit_trajectories,
    py::arg("t0") = py::none(),
    py::arg("period") = py::none(),
    py::arg("a") = py::none(),
    py::arg("inc") = py::none(),
    py::arg("ecc") = py::none(),
    py::arg("omega") = py::none(),
    py::arg("times") = py::none(),
    py::arg("ds") = py::none(),
    py::arg("zs") = py::none(),
    py::arg("nus") = py::none());

  m.def("light_curve", &compute_harmonica_light_curve,
    py::arg("t0") = py::none(),
    py::arg("period") = py::none(),
    py::arg("a") = py::none(),
    py::arg("inc") = py::none(),
    py::arg("ecc") = py::none(),
    py::arg("omega") = py::none(),
    py::arg("ld_law") = py::none(),
    py::arg("us") = py::none(),
    py::arg("rs") = py::none(),
    py::arg("times") = py::none(),
    py::arg("fs") = py::none(),
    py::arg("pnl_c") = 50,
    py::arg("pnl_e") = 500);

  m.def("transmission_string", &compute_transmission_string,
    py::arg("rs") = py::none(),
    py::arg("thetas") = py::none(),
    py::arg("transmission_string") = py::none());

  m.def("jax_registrations", &jax_registrations);
  m.def("jax_gpu_registrations", &jax_gpu_registrations);

}
