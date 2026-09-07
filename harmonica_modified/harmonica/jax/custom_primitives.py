import jax
import numpy as np
import jax.numpy as jnp
from jax.interpreters import ad
import os

from harmonica.core import bindings

# Enable double floating precision.
jax.config.update("jax_enable_x64", True)


_POWER2_VJP_TARGET_JACOBIAN_ELEMENTS = int(
    os.environ.get("HARMONICA_POWER2_VJP_TARGET_JACOBIAN_ELEMENTS", "65536")
)


def _register_ffi_targets(platform, registrations):
    """Register custom-call targets for a specific backend platform."""
    for name, capsule in registrations.items():
        try:
            jax.ffi.register_ffi_target(
                name,
                capsule,
                platform=platform,
                api_version=0,
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already" not in message and "duplicate" not in message:
                raise


def _gpu_backend_available():
    """Return whether JAX currently sees a GPU backend."""
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def _register_legacy_ffi_targets():
    """Register the existing CPU custom-call targets via jax.ffi."""
    _register_ffi_targets("cpu", bindings.jax_registrations())


def _register_optional_gpu_ffi_targets():
    """Register optional GPU custom-call targets when the backend is present."""
    registrations_fn = None
    for attr_name in (
        "jax_gpu_registrations",
        "gpu_jax_registrations",
        "gpu_registrations",
    ):
        registrations_fn = getattr(bindings, attr_name, None)
        if registrations_fn is not None:
            break

    if registrations_fn is None or not _gpu_backend_available():
        return False

    _register_ffi_targets("CUDA", registrations_fn())
    return True


_register_legacy_ffi_targets()
HAS_GPU_FFI_TARGETS = _register_optional_gpu_ffi_targets()


def _broadcast_param_to_shape(param, target_shape):
    """Broadcast a scalar or array-like parameter to the target shape."""
    param = jnp.asarray(param, dtype=jnp.float64)
    if param.ndim == 0:
        return jnp.broadcast_to(param, target_shape)

    try:
        return jnp.broadcast_to(param, target_shape)
    except ValueError:
        if param.ndim <= len(target_shape):
            expanded_shape = param.shape + (1,) * (len(target_shape) - param.ndim)
            try:
                return jnp.broadcast_to(
                    jnp.reshape(param, expanded_shape),
                    target_shape,
                )
            except ValueError:
                pass

    raise ValueError(
        f"Parameter with shape {param.shape} cannot be broadcast to "
        f"`times` shape {target_shape}."
    )


def _broadcast_r_to_shape(r, time_shape):
    """Broadcast ripple coefficients to `time_shape + (n_coeffs,)`."""
    r = jnp.asarray(r, dtype=jnp.float64)
    if r.ndim < 1:
        raise ValueError("`r` must include at least one coefficient axis.")

    if r.shape[-1] % 2 == 0:
        r = jnp.pad(
            r,
            [(0, 0)] * (r.ndim - 1) + [(0, 1)],
            constant_values=0.0,
        )

    target_shape = time_shape + (r.shape[-1],)
    try:
        return jnp.broadcast_to(r, target_shape)
    except ValueError:
        leading_shape = r.shape[:-1]
        if len(leading_shape) <= len(time_shape):
            expanded_shape = (
                leading_shape
                + (1,) * (len(time_shape) - len(leading_shape))
                + (r.shape[-1],)
            )
            try:
                return jnp.broadcast_to(
                    jnp.reshape(r, expanded_shape),
                    target_shape,
                )
            except ValueError:
                pass

    raise ValueError(
        f"`r` with shape {r.shape} cannot be broadcast to batched "
        f"shape {target_shape}."
    )


def _legacy_cpu_ffi_call(target_name, n_static_params, times, *params):
    """Call a legacy CPU custom-call target through the jax.ffi frontend."""
    n_params = len(params)
    n_rs = n_params - n_static_params
    if n_rs < 1:
        raise ValueError(
            f"{target_name} requires at least one ripple coefficient; got {n_rs}."
        )

    result_shapes = (
        jax.ShapeDtypeStruct(times.shape, times.dtype),
        jax.ShapeDtypeStruct(times.shape + (n_params,), times.dtype),
    )
    call = jax.ffi.ffi_call(
        target_name,
        result_shapes,
        # The legacy kernel takes scalar metadata operands (`n_times`, `n_rs`)
        # ahead of the array buffers, so sequential vmap preserves the old
        # semantics without trying to batch those scalars.
        vmap_method="sequential",
        custom_call_api_version=2,
    )

    n_times = jnp.asarray(np.prod(times.shape), dtype=jnp.int32)
    n_rs = jnp.asarray(np.int32(n_rs))
    return call(n_times, n_rs, times, *params)


def _legacy_cpu_ffi_call_one_result(target_name, n_static_params, times, *params):
    """Call a legacy CPU custom-call target that returns flux only."""
    n_params = len(params)
    n_rs = n_params - n_static_params
    if n_rs < 1:
        raise ValueError(
            f"{target_name} requires at least one ripple coefficient; got {n_rs}."
        )

    result_shape = jax.ShapeDtypeStruct(times.shape, times.dtype)
    call = jax.ffi.ffi_call(
        target_name,
        result_shape,
        vmap_method="sequential",
        custom_call_api_version=2,
    )

    n_times = jnp.asarray(np.prod(times.shape), dtype=jnp.int32)
    n_rs = jnp.asarray(np.int32(n_rs))
    return call(n_times, n_rs, times, *params)


def _jax_light_curve_quad_ld_ffi(times, *params):
    return _legacy_cpu_ffi_call("jax_light_curve_quad_ld", 8, times, *params)


def _jax_light_curve_power2_ld_flux_ffi(times, *params):
    return _legacy_cpu_ffi_call_one_result(
        "jax_light_curve_power2_ld_flux", 8, times, *params
    )


def _jax_light_curve_power2_ld_ffi(times, *params):
    return _legacy_cpu_ffi_call("jax_light_curve_power2_ld", 8, times, *params)


def _jax_light_curve_nonlinear_ld_ffi(times, *params):
    return _legacy_cpu_ffi_call("jax_light_curve_nonlinear_ld", 10, times, *params)


def _prepare_harmonica_inputs(times, params, r):
    """Broadcast and sanitize inputs shared by the JAX Harmonica wrappers."""
    min_abs = 1e-9
    sigmoid_width = 0.1 * min_abs
    safe_sign_eps = 1e-12

    def smooth_min_abs(x, floor=min_abs, softness=sigmoid_width):
        abs_x = jnp.abs(x)
        scale = (abs_x - floor) / softness
        blend = jax.nn.sigmoid(scale)
        safe_sign = x / jnp.sqrt(x**2 + safe_sign_eps)
        return blend * x + (1 - blend) * floor * safe_sign

    def ensure_last_two_nonzero(coeffs, min_tail=min_abs, softness=sigmoid_width):
        tail = coeffs[..., -2:]
        abs_tail = jnp.abs(tail)
        blend = jax.nn.sigmoid(
            (jnp.sum(abs_tail, axis=-1, keepdims=True) - min_tail) / softness
        )
        safe_tail = jnp.sign(tail + safe_sign_eps) * min_tail
        new_tail = blend * tail + (1 - blend) * safe_tail
        return coeffs.at[..., -2:].set(new_tail)

    times = jnp.atleast_1d(jnp.asarray(times, dtype=jnp.float64))
    if times.ndim > 2:
        raise ValueError("`times` must be a scalar, 1D array, or 2D array")
    time_shape = times.shape

    broadcasted_params = [
        _broadcast_param_to_shape(param, time_shape)
        for param in params
    ]

    r = _broadcast_r_to_shape(r, time_shape)
    flip = r[..., 0] < 0

    r0_fixed = smooth_min_abs(jnp.abs(r[..., 0]))
    r = r.at[..., 0].set(r0_fixed)
    r = ensure_last_two_nonzero(r)

    return times, time_shape, broadcasted_params, r, flip


def _sanitize_harmonica_flux(flux, flip):
    """Apply the signed-radius convention without hiding invalid evaluations."""
    return jnp.where(flip, 2.0 - flux, flux)


def _harmonica_transit_common(ffi_fn, times, params, r):
    """
    Internal helper to evaluate JAX-based Harmonica with broadcasted inputs.

    Returns ``(flux, *rest)``. Non-finite backend results are deliberately
    propagated so NUTS rejects the state and diagnostics can report the
    numerical failure instead of silently replacing its gradient with zero.
    """
    times, _time_shape, broadcasted_params, r, flip = _prepare_harmonica_inputs(
        times, params, r
    )

    r_list = [r[..., i] for i in range(r.shape[-1])]
    args = [
        jnp.asarray(arg, dtype=jnp.float64)
        for arg in (times, *broadcasted_params, *r_list)
    ]

    flux, *rest = ffi_fn(*args)

    flux = _sanitize_harmonica_flux(flux, flip)
    return (flux, *rest)


def _harmonica_transit_flux_only_common(ffi_fn, times, params, r):
    """Evaluate a flux-only Harmonica CPU custom call."""
    times, _time_shape, broadcasted_params, r, flip = _prepare_harmonica_inputs(
        times, params, r
    )
    r_list = [r[..., i] for i in range(r.shape[-1])]
    args = [
        jnp.asarray(arg, dtype=jnp.float64)
        for arg in (times, *broadcasted_params, *r_list)
    ]
    flux = ffi_fn(*args)
    return _sanitize_harmonica_flux(flux, flip)


def _expanded_param_shape(input_shape, target_shape):
    """Reconstruct the broadcasted param shape used by `_broadcast_param_to_shape`."""
    input_shape = tuple(input_shape)
    target_shape = tuple(target_shape)

    try:
        if np.broadcast_shapes(input_shape, target_shape) == target_shape:
            return (1,) * (len(target_shape) - len(input_shape)) + input_shape
    except ValueError:
        pass

    if len(input_shape) <= len(target_shape):
        expanded_shape = input_shape + (1,) * (len(target_shape) - len(input_shape))
        try:
            if np.broadcast_shapes(expanded_shape, target_shape) == target_shape:
                return expanded_shape
        except ValueError:
            pass

    raise ValueError(
        f"Parameter with shape {input_shape} cannot be broadcast to shape {target_shape}."
    )


def _expanded_r_shape(r_shape, time_shape):
    """Reconstruct the broadcasted ripple shape used by `_broadcast_r_to_shape`."""
    r_shape = tuple(r_shape)
    time_shape = tuple(time_shape)
    target_shape = time_shape + (r_shape[-1],)

    try:
        if np.broadcast_shapes(r_shape, target_shape) == target_shape:
            return (1,) * (len(target_shape) - len(r_shape)) + r_shape
    except ValueError:
        pass

    leading_shape = r_shape[:-1]
    expanded_shape = (
        leading_shape
        + (1,) * (len(time_shape) - len(leading_shape))
        + (r_shape[-1],)
    )
    try:
        if np.broadcast_shapes(expanded_shape, target_shape) == target_shape:
            return expanded_shape
    except ValueError:
        pass

    raise ValueError(
        f"`r` with shape {r_shape} cannot be broadcast to shape {target_shape}."
    )


def _reduce_param_cotangent(cotangent, param, time_shape):
    """Undo `_broadcast_param_to_shape` for reverse-mode cotangents."""
    param = jnp.asarray(param, dtype=jnp.float64)
    expanded_shape = _expanded_param_shape(param.shape, tuple(time_shape))
    axes = tuple(
        i
        for i, (src, dst) in enumerate(zip(expanded_shape, tuple(time_shape)))
        if src == 1 and dst != 1
    )
    if axes:
        cotangent = jnp.sum(cotangent, axis=axes, keepdims=True)
    return jnp.reshape(cotangent, expanded_shape).reshape(param.shape)


def _reduce_r_cotangent(cotangent, r, time_shape):
    """Undo `_broadcast_r_to_shape` for reverse-mode cotangents."""
    r = jnp.asarray(r, dtype=jnp.float64)
    if r.shape[-1] % 2 == 0 and cotangent.shape[-1] == r.shape[-1] + 1:
        cotangent = cotangent[..., : r.shape[-1]]
    expanded_shape = _expanded_r_shape(r.shape, tuple(time_shape))
    target_shape = tuple(time_shape) + (cotangent.shape[-1],)
    axes = tuple(
        i
        for i, (src, dst) in enumerate(zip(expanded_shape, target_shape))
        if src == 1 and dst != 1
    )
    if axes:
        cotangent = jnp.sum(cotangent, axis=axes, keepdims=True)
    return jnp.reshape(cotangent, expanded_shape).reshape(r.shape)


def _depends_on_curve_axis(param, time_shape, expanded_shape_fn):
    """Return whether an input contributes distinct values per light curve."""
    if len(time_shape) < 2:
        return False
    return expanded_shape_fn(jnp.asarray(param).shape, tuple(time_shape))[0] != 1


def _slice_param_curve_chunk(param, time_shape, start, stop):
    """Slice a parameter along the leading light-curve axis when needed."""
    param = jnp.asarray(param, dtype=jnp.float64)
    if not _depends_on_curve_axis(param, time_shape, _expanded_param_shape):
        return param
    if param.ndim == 0:
        return param
    return param[start:stop, ...]


def _slice_r_curve_chunk(r, time_shape, start, stop):
    """Slice ripple coefficients along the leading light-curve axis when needed."""
    r = jnp.asarray(r, dtype=jnp.float64)
    if not _depends_on_curve_axis(r, time_shape, _expanded_r_shape):
        return r
    if r.ndim == 1:
        return r
    return r[start:stop, ...]


def _accumulate_curve_chunk(total, chunk, depends_on_curve_axis, start, stop):
    """Accumulate a chunk cotangent into the full parameter cotangent."""
    if depends_on_curve_axis and getattr(total, "ndim", 0) > 0:
        return total.at[start:stop, ...].add(chunk)
    return total + chunk


def _power2_vjp_curve_chunk_size(time_shape, n_r_coeffs):
    """Choose a curve-axis chunk size that caps per-chunk Jacobian size."""
    if len(time_shape) < 2:
        return time_shape[0]
    n_curves, n_times = time_shape
    n_derivatives = 8 + int(n_r_coeffs)
    max_chunk = max(
        1,
        _POWER2_VJP_TARGET_JACOBIAN_ELEMENTS // max(1, n_times * n_derivatives),
    )
    return min(n_curves, max_chunk)


def _contract_jacobian_tangents(df_dz, time_shape, param_tangents, r_tangent):
    """Combine broadcasted tangents with the backend-provided Jacobian."""
    df = jnp.zeros(time_shape, dtype=df_dz.dtype)

    for idx, tangent in enumerate(param_tangents):
        if isinstance(tangent, ad.Zero):
            continue
        df += _broadcast_param_to_shape(tangent, time_shape) * df_dz[..., idx]

    if not isinstance(r_tangent, ad.Zero):
        r_tangent = _broadcast_r_to_shape(r_tangent, time_shape)
        df += jnp.sum(r_tangent * df_dz[..., len(param_tangents):], axis=-1)

    return df


def _quad_ld_flux_and_derivatives_impl(times, t0, period, a, inc, ecc, omega, u1, u2, r):
    return _harmonica_transit_common(
        _jax_light_curve_quad_ld_ffi,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2],
        r,
    )


@jax.custom_jvp
def _quad_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega, u1, u2, r):
    """Return both the flux and its derivatives for the quadratic LD model."""
    return _quad_ld_flux_and_derivatives_impl(
        times, t0, period, a, inc, ecc, omega, u1, u2, r
    )


@_quad_ld_flux_and_derivatives.defjvp
def _quad_ld_flux_and_derivatives_jvp(primals, tangents):
    times, t0, period, a, inc, ecc, omega, u1, u2, r = primals
    _, dt0, dperiod, da, dinc, decc, domega, du1, du2, dr = tangents

    f, df_dz = _quad_ld_flux_and_derivatives_impl(
        times, t0, period, a, inc, ecc, omega, u1, u2, r
    )
    df = _contract_jacobian_tangents(
        df_dz,
        f.shape,
        (dt0, dperiod, da, dinc, decc, domega, du1, du2),
        dr,
    )
    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


def _power2_ld_flux_and_derivatives_impl(
    times, t0, period, a, inc, ecc, omega, c, alpha, r
):
    return _harmonica_transit_common(
        _jax_light_curve_power2_ld_ffi,
        times,
        [t0, period, a, inc, ecc, omega, c, alpha],
        r,
    )


@jax.custom_jvp
def _power2_ld_flux_and_derivatives(times, t0, period, a, inc, ecc, omega, c, alpha, r):
    """Return both the flux and its derivatives for the power-2 LD model."""
    return _power2_ld_flux_and_derivatives_impl(
        times, t0, period, a, inc, ecc, omega, c, alpha, r
    )


@_power2_ld_flux_and_derivatives.defjvp
def _power2_ld_flux_and_derivatives_jvp(primals, tangents):
    times, t0, period, a, inc, ecc, omega, c, alpha, r = primals
    _, dt0, dperiod, da, dinc, decc, domega, dc, dalpha, dr = tangents

    f, df_dz = _power2_ld_flux_and_derivatives_impl(
        times, t0, period, a, inc, ecc, omega, c, alpha, r
    )
    df = _contract_jacobian_tangents(
        df_dz,
        f.shape,
        (dt0, dperiod, da, dinc, decc, domega, dc, dalpha),
        dr,
    )
    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


def _power2_ld_flux_impl(times, t0, period, a, inc, ecc, omega, c, alpha, r):
    return _harmonica_transit_flux_only_common(
        _jax_light_curve_power2_ld_flux_ffi,
        times,
        [t0, period, a, inc, ecc, omega, c, alpha],
        r,
    )


@jax.custom_vjp
def _power2_ld_flux(times, t0, period, a, inc, ecc, omega, c, alpha, r):
    """Return flux for the power-2 LD model without forcing Jacobian output."""
    return _power2_ld_flux_impl(times, t0, period, a, inc, ecc, omega, c, alpha, r)


def _power2_ld_flux_fwd(times, t0, period, a, inc, ecc, omega, c, alpha, r):
    flux = _power2_ld_flux_impl(times, t0, period, a, inc, ecc, omega, c, alpha, r)
    return flux, (times, t0, period, a, inc, ecc, omega, c, alpha, r)


def _power2_ld_flux_bwd(residuals, cotangent):
    times, t0, period, a, inc, ecc, omega, c, alpha, r = residuals
    cotangent = jnp.asarray(cotangent, dtype=jnp.float64)
    time_shape = cotangent.shape

    def _reduce(weighted_slice, param, shape):
        return _reduce_param_cotangent(weighted_slice, param, shape)

    def _reduce_r(weighted_slice, r_param, shape):
        return _reduce_r_cotangent(weighted_slice, r_param, shape)

    if len(time_shape) < 2:
        _flux, df_dz = _power2_ld_flux_and_derivatives_impl(
            times, t0, period, a, inc, ecc, omega, c, alpha, r
        )
        weighted = cotangent[..., None] * df_dz
        return (
            jnp.zeros_like(times),
            _reduce(weighted[..., 0], t0, time_shape),
            _reduce(weighted[..., 1], period, time_shape),
            _reduce(weighted[..., 2], a, time_shape),
            _reduce(weighted[..., 3], inc, time_shape),
            _reduce(weighted[..., 4], ecc, time_shape),
            _reduce(weighted[..., 5], omega, time_shape),
            _reduce(weighted[..., 6], c, time_shape),
            _reduce(weighted[..., 7], alpha, time_shape),
            _reduce_r(weighted[..., 8:], r, time_shape),
        )

    curve_deps = (
        _depends_on_curve_axis(t0, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(period, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(a, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(inc, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(ecc, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(omega, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(c, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(alpha, time_shape, _expanded_param_shape),
        _depends_on_curve_axis(r, time_shape, _expanded_r_shape),
    )
    chunk_size = _power2_vjp_curve_chunk_size(time_shape, jnp.asarray(r).shape[-1])
    if chunk_size >= time_shape[0]:
        _flux, df_dz = _power2_ld_flux_and_derivatives_impl(
            times, t0, period, a, inc, ecc, omega, c, alpha, r
        )
        weighted = cotangent[..., None] * df_dz
        return (
            jnp.zeros_like(times),
            _reduce(weighted[..., 0], t0, time_shape),
            _reduce(weighted[..., 1], period, time_shape),
            _reduce(weighted[..., 2], a, time_shape),
            _reduce(weighted[..., 3], inc, time_shape),
            _reduce(weighted[..., 4], ecc, time_shape),
            _reduce(weighted[..., 5], omega, time_shape),
            _reduce(weighted[..., 6], c, time_shape),
            _reduce(weighted[..., 7], alpha, time_shape),
            _reduce_r(weighted[..., 8:], r, time_shape),
        )

    grad_t0 = jnp.zeros_like(jnp.asarray(t0, dtype=jnp.float64))
    grad_period = jnp.zeros_like(jnp.asarray(period, dtype=jnp.float64))
    grad_a = jnp.zeros_like(jnp.asarray(a, dtype=jnp.float64))
    grad_inc = jnp.zeros_like(jnp.asarray(inc, dtype=jnp.float64))
    grad_ecc = jnp.zeros_like(jnp.asarray(ecc, dtype=jnp.float64))
    grad_omega = jnp.zeros_like(jnp.asarray(omega, dtype=jnp.float64))
    grad_c = jnp.zeros_like(jnp.asarray(c, dtype=jnp.float64))
    grad_alpha = jnp.zeros_like(jnp.asarray(alpha, dtype=jnp.float64))
    grad_r = jnp.zeros_like(jnp.asarray(r, dtype=jnp.float64))

    for start in range(0, time_shape[0], chunk_size):
        stop = min(start + chunk_size, time_shape[0])
        cotangent_chunk = cotangent[start:stop, ...]
        t0_chunk = _slice_param_curve_chunk(t0, time_shape, start, stop)
        period_chunk = _slice_param_curve_chunk(period, time_shape, start, stop)
        a_chunk = _slice_param_curve_chunk(a, time_shape, start, stop)
        inc_chunk = _slice_param_curve_chunk(inc, time_shape, start, stop)
        ecc_chunk = _slice_param_curve_chunk(ecc, time_shape, start, stop)
        omega_chunk = _slice_param_curve_chunk(omega, time_shape, start, stop)
        c_chunk = _slice_param_curve_chunk(c, time_shape, start, stop)
        alpha_chunk = _slice_param_curve_chunk(alpha, time_shape, start, stop)
        r_chunk = _slice_r_curve_chunk(r, time_shape, start, stop)

        _flux, df_dz = _power2_ld_flux_and_derivatives_impl(
            times[start:stop, ...],
            t0_chunk,
            period_chunk,
            a_chunk,
            inc_chunk,
            ecc_chunk,
            omega_chunk,
            c_chunk,
            alpha_chunk,
            r_chunk,
        )
        weighted = cotangent_chunk[..., None] * df_dz
        chunk_shape = cotangent_chunk.shape
        grad_t0 = _accumulate_curve_chunk(
            grad_t0,
            _reduce_param_cotangent(weighted[..., 0], t0_chunk, chunk_shape),
            curve_deps[0],
            start,
            stop,
        )
        grad_period = _accumulate_curve_chunk(
            grad_period,
            _reduce_param_cotangent(weighted[..., 1], period_chunk, chunk_shape),
            curve_deps[1],
            start,
            stop,
        )
        grad_a = _accumulate_curve_chunk(
            grad_a,
            _reduce_param_cotangent(weighted[..., 2], a_chunk, chunk_shape),
            curve_deps[2],
            start,
            stop,
        )
        grad_inc = _accumulate_curve_chunk(
            grad_inc,
            _reduce_param_cotangent(weighted[..., 3], inc_chunk, chunk_shape),
            curve_deps[3],
            start,
            stop,
        )
        grad_ecc = _accumulate_curve_chunk(
            grad_ecc,
            _reduce_param_cotangent(weighted[..., 4], ecc_chunk, chunk_shape),
            curve_deps[4],
            start,
            stop,
        )
        grad_omega = _accumulate_curve_chunk(
            grad_omega,
            _reduce_param_cotangent(weighted[..., 5], omega_chunk, chunk_shape),
            curve_deps[5],
            start,
            stop,
        )
        grad_c = _accumulate_curve_chunk(
            grad_c,
            _reduce_param_cotangent(weighted[..., 6], c_chunk, chunk_shape),
            curve_deps[6],
            start,
            stop,
        )
        grad_alpha = _accumulate_curve_chunk(
            grad_alpha,
            _reduce_param_cotangent(weighted[..., 7], alpha_chunk, chunk_shape),
            curve_deps[7],
            start,
            stop,
        )
        grad_r = _accumulate_curve_chunk(
            grad_r,
            _reduce_r_cotangent(weighted[..., 8:], r_chunk, chunk_shape),
            curve_deps[8],
            start,
            stop,
        )

    return (
        jnp.zeros_like(times),
        grad_t0,
        grad_period,
        grad_a,
        grad_inc,
        grad_ecc,
        grad_omega,
        grad_c,
        grad_alpha,
        grad_r,
    )


_power2_ld_flux.defvjp(_power2_ld_flux_fwd, _power2_ld_flux_bwd)


def _nonlinear_ld_flux_and_derivatives_impl(
    times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
):
    return _harmonica_transit_common(
        _jax_light_curve_nonlinear_ld_ffi,
        times,
        [t0, period, a, inc, ecc, omega, u1, u2, u3, u4],
        r,
    )


@jax.custom_jvp
def _nonlinear_ld_flux_and_derivatives(
    times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
):
    """Return both the flux and its derivatives for the non-linear LD model."""
    return _nonlinear_ld_flux_and_derivatives_impl(
        times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
    )


@_nonlinear_ld_flux_and_derivatives.defjvp
def _nonlinear_ld_flux_and_derivatives_jvp(primals, tangents):
    times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r = primals
    _, dt0, dperiod, da, dinc, decc, domega, du1, du2, du3, du4, dr = tangents

    f, df_dz = _nonlinear_ld_flux_and_derivatives_impl(
        times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
    )
    df = _contract_jacobian_tangents(
        df_dz,
        f.shape,
        (dt0, dperiod, da, dinc, decc, domega, du1, du2, du3, du4),
        dr,
    )
    dummy_tangent = jnp.zeros_like(df_dz)
    return (f, df_dz), (df, dummy_tangent)


def harmonica_transit_quad_ld(
    times,
    t0,
    period,
    a,
    inc,
    ecc=0.0,
    omega=0.0,
    u1=0.0,
    u2=0.0,
    r=jnp.array([0.1]),
):
    """Harmonica transits with jax -- quadratic limb darkening."""
    r_arr = jnp.atleast_1d(jnp.asarray(r, dtype=jnp.float64))
    n_rs = r_arr.shape[-1]
    if jax.default_backend() == "gpu" and n_rs > 3:
        raise NotImplementedError(
            "Pure-JAX GPU quadratic Harmonica currently supports only "
            "N_c<=1 (r=[r0,a1,b1]); use the CPU backend for higher orders."
        )
    if jax.default_backend() == "gpu" and n_rs <= 3:
        from harmonica.jax.power2_nc1_jax import (
            harmonica_transit_quadratic_nc1_jax,
        )

        r0_val = r_arr[..., 0]
        a1_val = (
            r_arr[..., 1] if n_rs >= 2 else jnp.zeros_like(r0_val)
        )
        b1_val = (
            r_arr[..., 2] if n_rs >= 3 else jnp.zeros_like(r0_val)
        )
        return harmonica_transit_quadratic_nc1_jax(
            times,
            t0,
            period,
            a,
            inc,
            ecc,
            omega,
            u1,
            u2,
            r0_val,
            a1_val,
            b1_val,
        )
    return _quad_ld_flux_and_derivatives(
        times, t0, period, a, inc, ecc, omega, u1, u2, r
    )[0]


def harmonica_transit_power2_ld(
    times,
    t0,
    period,
    a,
    inc,
    ecc=0.0,
    omega=0.0,
    c=0.0,
    alpha=1.0,
    r=jnp.array([0.1]),
):
    """Harmonica transits with jax -- exact power-2 limb darkening."""
    alpha = jnp.maximum(jnp.asarray(alpha, dtype=jnp.float64), 1e-10)

    # Determine the number of ripple coefficients (before any padding).
    r_arr = jnp.atleast_1d(jnp.asarray(r, dtype=jnp.float64))
    n_rs = r_arr.shape[-1]
    if jax.default_backend() == "gpu" and n_rs > 3:
        raise NotImplementedError(
            "Pure-JAX GPU power-2 Harmonica currently supports only "
            "N_c<=1 (r=[r0,a1,b1]); use the CPU backend for higher orders."
        )

    # On GPU with N_c=1 (n_rs <= 3), use the pure JAX implementation which
    # avoids the C++ FFI custom call that is only compiled for CPU.
    use_jax_gpu = (
        jax.default_backend() == "gpu"
        and n_rs <= 3  # N_c=1
    )

    if use_jax_gpu:
        from harmonica.jax.power2_nc1_jax import harmonica_transit_power2_nc1_jax
        # Unpack r coefficients: r0, a1, b1 (pad b1=0 if only 2 given).
        r0_val = r_arr[..., 0]
        a1_val = jnp.where(n_rs >= 2, r_arr[..., jnp.minimum(1, n_rs - 1)], 0.0)
        b1_val = jnp.where(n_rs >= 3, r_arr[..., jnp.minimum(2, n_rs - 1)], 0.0)
        return harmonica_transit_power2_nc1_jax(
            times, t0, period, a, inc, ecc, omega, c, alpha,
            r0_val, a1_val, b1_val,
        )

    return _power2_ld_flux(times, t0, period, a, inc, ecc, omega, c, alpha, r)


@jax.jit
def harmonica_transit_nonlinear_ld(
    times,
    t0,
    period,
    a,
    inc,
    ecc=0.0,
    omega=0.0,
    u1=0.0,
    u2=0.0,
    u3=0.0,
    u4=0.0,
    r=jnp.array([0.1]),
):
    """Harmonica transits with jax -- non-linear limb darkening."""
    return _nonlinear_ld_flux_and_derivatives(
        times, t0, period, a, inc, ecc, omega, u1, u2, u3, u4, r
    )[0]
