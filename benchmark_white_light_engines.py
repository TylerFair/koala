import argparse
import os
import sys
import time


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
HARMONICA_REPO = os.path.join(os.path.dirname(REPO_DIR), "harmonica_bell-main")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if os.path.isdir(HARMONICA_REPO) and HARMONICA_REPO not in sys.path:
    sys.path.insert(0, HARMONICA_REPO)

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import numpy as np
from jaxoplanet.experimental import calc_poly_coeffs
import numpyro.infer as infer
from numpyro.infer.util import log_density

from models.builder import create_whitelight_model
from models.core import (
    get_I_power2,
)
from models.trends import compute_lc_linear


POLY_DEGREE = 12
MUS = jnp.linspace(0.0, 1.0, 300, endpoint=True)


def power2_to_u(c_ld, alpha_ld):
    profile = get_I_power2(c_ld, alpha_ld, MUS)
    return calc_poly_coeffs(MUS, profile, poly_degree=POLY_DEGREE)


def build_times(n_times, t0, duration):
    span = 4.0 * duration
    return jnp.linspace(t0 - span, t0 + span, n_times)


def make_truth(args):
    t = build_times(args.n_times, args.t0, args.duration)
    yerr = jnp.full_like(t, args.yerr)
    u = power2_to_u(args.c_ld, args.alpha_ld)

    theta_jax = jnp.array([
        jnp.log(args.duration),
        args.b,
        args.rors**2,
        1.0,
        0.0,
    ], dtype=jnp.float64)
    theta_harm = jnp.array([
        jnp.log(args.a_rs),
        args.b,
        args.rors**2,
        1.0,
        0.0,
    ], dtype=jnp.float64)

    return t, yerr, u, theta_jax, theta_harm


def unpack_jaxoplanet(theta, args, u):
    log_duration, b_raw, depth, c0, v0 = theta
    return {
        "period": jnp.array([args.period], dtype=jnp.float64),
        "duration": jnp.array([jnp.exp(log_duration)], dtype=jnp.float64),
        "t0": jnp.array([args.t0], dtype=jnp.float64),
        "b": jnp.array([jnp.abs(b_raw)], dtype=jnp.float64),
        "rors": jnp.array([jnp.sqrt(jnp.maximum(depth, 1e-10))], dtype=jnp.float64),
        "u": u,
        "c": c0,
        "v": v0,
    }


def unpack_harmonica(theta, args):
    log_a_rs, b_raw, depth, c0, v0 = theta
    return {
        "period": jnp.array([args.period], dtype=jnp.float64),
        "t0": jnp.array([args.t0], dtype=jnp.float64),
        "b": jnp.array([jnp.abs(b_raw)], dtype=jnp.float64),
        "rors": jnp.array([jnp.sqrt(jnp.maximum(depth, 1e-10))], dtype=jnp.float64),
        "a_rs": jnp.array([jnp.exp(log_a_rs)], dtype=jnp.float64),
        "ecc": jnp.array([args.ecc], dtype=jnp.float64),
        "omega": jnp.array([args.omega], dtype=jnp.float64),
        "c": c0,
        "v": v0,
        "c_ld": jnp.asarray(args.c_ld, dtype=jnp.float64),
        "alpha_ld": jnp.asarray(args.alpha_ld, dtype=jnp.float64),
    }


def build_objectives(args, t, yerr, u, theta_jax, theta_harm):
    params_jax_truth = unpack_jaxoplanet(theta_jax, args, u)
    params_harm_truth = unpack_harmonica(theta_harm, args)

    flux_truth_jax = compute_lc_linear(params_jax_truth, t)
    flux_truth_harm = compute_lc_linear(params_harm_truth, t)

    # Use a tiny deterministic perturbation so gradients are non-zero.
    perturb = args.yerr * 0.1 * jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, t.size))
    y_jax = flux_truth_jax + perturb
    y_harm = flux_truth_harm + perturb

    def flux_jax(theta):
        return compute_lc_linear(unpack_jaxoplanet(theta, args, u), t)

    def flux_harm(theta):
        return compute_lc_linear(unpack_harmonica(theta, args), t)

    def nll_jax(theta):
        resid = (flux_jax(theta) - y_jax) / yerr
        return 0.5 * jnp.sum(resid**2)

    def nll_harm(theta):
        resid = (flux_harm(theta) - y_harm) / yerr
        return 0.5 * jnp.sum(resid**2)

    return flux_jax, flux_harm, nll_jax, nll_harm, y_jax, y_harm


def benchmark(label, fn, arg, repeats):
    compiled = jax.jit(fn)

    t0 = time.perf_counter()
    out = compiled(arg)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    samples = []
    for _ in range(repeats):
        t1 = time.perf_counter()
        out = compiled(arg)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t1)

    mean_ms = 1e3 * float(np.mean(samples))
    std_ms = 1e3 * float(np.std(samples))
    print(
        f"{label:20s} compile+first={compile_s:8.3f}s  "
        f"steady={mean_ms:8.3f} +/- {std_ms:6.3f} ms"
    )
    return mean_ms, std_ms


def make_prior_params(args):
    return {
        "period": jnp.array([args.period], dtype=jnp.float64),
        "u": jnp.array([args.c_ld, args.alpha_ld], dtype=jnp.float64),
        "ecc": jnp.array([args.ecc], dtype=jnp.float64),
        "omega": jnp.array([args.omega], dtype=jnp.float64),
        "a_rs_prior_min": jnp.array([args.a_rs_prior_min], dtype=jnp.float64),
        "a_rs_prior_max": jnp.array([args.a_rs_prior_max], dtype=jnp.float64),
    }


def constrained_params_jax(theta):
    log_duration, b_raw, depth, c0, v0 = theta
    return {
        "t0_0": jnp.array(0.0, dtype=jnp.float64),
        "depths_0": jnp.array(depth, dtype=jnp.float64),
        "_b_0": jnp.array(b_raw, dtype=jnp.float64),
        "logD_0": jnp.array(log_duration, dtype=jnp.float64),
        "log_jitter": jnp.array(np.log(1e-4), dtype=jnp.float64),
        "c": jnp.array(c0, dtype=jnp.float64),
        "v": jnp.array(v0, dtype=jnp.float64),
    }


def constrained_params_harm(theta, args):
    log_a_rs, b_raw, depth, c0, v0 = theta
    return {
        "t0_0": jnp.array(args.t0, dtype=jnp.float64),
        "depths_0": jnp.array(depth, dtype=jnp.float64),
        "log_a_rs_0": jnp.array(log_a_rs, dtype=jnp.float64),
        "_b_0": jnp.array(b_raw, dtype=jnp.float64),
        "a1": jnp.array(args.a1, dtype=jnp.float64),
        "log_jitter": jnp.array(np.log(1e-4), dtype=jnp.float64),
        "c": jnp.array(c0, dtype=jnp.float64),
        "v": jnp.array(v0, dtype=jnp.float64),
    }


def benchmark_log_density(label, model, params, t, yerr, y, prior_params, repeats):
    def target(p):
        lp, _ = log_density(model, (t, yerr), {"y": y, "prior_params": prior_params}, p)
        return lp

    lp_ms, _ = benchmark(f"{label} log_density", target, params, repeats)
    grad_ms, _ = benchmark(f"{label} log_density grad", jax.grad(target), params, repeats)
    return lp_ms, grad_ms


def benchmark_nuts(label, model, t, yerr, y, prior_params, num_warmup, num_samples, dense_mass):
    if num_warmup <= 0 or num_samples <= 0:
        return
    kernel = infer.NUTS(model, dense_mass=dense_mass)
    mcmc = infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
        jit_model_args=True,
    )
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(0), t, yerr, y=y, prior_params=prior_params)
    elapsed = time.perf_counter() - t0
    total_iters = num_warmup + num_samples
    print(
        f"{label:20s} elapsed={elapsed:8.3f}s  "
        f"sec_per_iter={elapsed / total_iters:8.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU white-light timings for jaxoplanet vs harmonica."
    )
    parser.add_argument("--n-times", type=int, default=1376)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--period", type=float, default=3.9502001)
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.18516666666666667)
    parser.add_argument("--b", type=float, default=0.15)
    parser.add_argument("--rors", type=float, default=0.107)
    parser.add_argument("--a-rs", dest="a_rs", type=float, default=7.3)
    parser.add_argument("--ecc", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--c-ld", dest="c_ld", type=float, default=0.5430773)
    parser.add_argument("--alpha-ld", dest="alpha_ld", type=float, default=0.38179205)
    parser.add_argument("--yerr", type=float, default=7e-4)
    parser.add_argument("--a-rs-prior-min", dest="a_rs_prior_min", type=float, default=3.65)
    parser.add_argument("--a-rs-prior-max", dest="a_rs_prior_max", type=float, default=14.6)
    parser.add_argument("--a1", type=float, default=0.0)
    parser.add_argument("--nuts-warmup", type=int, default=0)
    parser.add_argument("--nuts-samples", type=int, default=0)
    parser.add_argument("--dense-mass", action="store_true")
    args = parser.parse_args()

    print("Benchmark setup")
    print(
        {
            "n_times": args.n_times,
            "period": args.period,
            "duration": args.duration,
            "b": args.b,
            "rors": args.rors,
            "a_rs": args.a_rs,
            "c_ld": args.c_ld,
            "alpha_ld": args.alpha_ld,
            "yerr": args.yerr,
            "a_rs_prior_min": args.a_rs_prior_min,
            "a_rs_prior_max": args.a_rs_prior_max,
            "a1": args.a1,
            "repeats": args.repeats,
            "nuts_warmup": args.nuts_warmup,
            "nuts_samples": args.nuts_samples,
            "dense_mass": args.dense_mass,
        }
    )

    t, yerr, u, theta_jax, theta_harm = make_truth(args)
    flux_jax, flux_harm, nll_jax, nll_harm, y_jax, y_harm = build_objectives(
        args, t, yerr, u, theta_jax, theta_harm
    )
    prior_params = make_prior_params(args)

    grad_jax = jax.grad(nll_jax)
    grad_harm = jax.grad(nll_harm)

    jax_flux_ms, _ = benchmark("jaxoplanet flux", flux_jax, theta_jax, args.repeats)
    harm_flux_ms, _ = benchmark("harmonica flux", flux_harm, theta_harm, args.repeats)
    jax_grad_ms, _ = benchmark("jaxoplanet grad", grad_jax, theta_jax, args.repeats)
    harm_grad_ms, _ = benchmark("harmonica grad", grad_harm, theta_harm, args.repeats)

    print("Derived metrics")
    print(
        {
            "n_times": int(t.size),
            "jaxoplanet_flux_us_per_point": 1e3 * jax_flux_ms / float(t.size),
            "harmonica_flux_us_per_point": 1e3 * harm_flux_ms / float(t.size),
            "jaxoplanet_grad_us_per_point": 1e3 * jax_grad_ms / float(t.size),
            "harmonica_grad_us_per_point": 1e3 * harm_grad_ms / float(t.size),
            "harmonica_over_jax_flux": harm_flux_ms / jax_flux_ms,
            "harmonica_over_jax_grad": harm_grad_ms / jax_grad_ms,
        }
    )

    print("Exact white-light log-density")
    jax_model = create_whitelight_model(
        detrend_type="linear",
        n_planets=1,
        ld_profile="power2",
        transit_engine="jaxoplanet",
        ld_mode="fixed",
    )
    harm_model = create_whitelight_model(
        detrend_type="linear",
        n_planets=1,
        ld_profile="power2",
        transit_engine="harmonica",
        max_harmonic_order=1,
        ld_mode="fixed",
    )
    benchmark_log_density(
        "jaxoplanet wl",
        jax_model,
        constrained_params_jax(theta_jax),
        t,
        yerr,
        y_jax,
        prior_params,
        args.repeats,
    )
    benchmark_log_density(
        "harmonica wl",
        harm_model,
        constrained_params_harm(theta_harm, args),
        t,
        yerr,
        y_harm,
        prior_params,
        args.repeats,
    )
    benchmark_nuts(
        "harmonica wl NUTS",
        harm_model,
        t,
        yerr,
        y_harm,
        prior_params,
        args.nuts_warmup,
        args.nuts_samples,
        args.dense_mass,
    )


if __name__ == "__main__":
    main()

