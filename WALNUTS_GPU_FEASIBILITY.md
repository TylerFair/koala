# WALNUTS / `walnutpie` GPU feasibility

## Decision

Do not replace the current NumPyro sampler with WALNUTS for the GPU light-curve
pipeline yet. The present upstream implementation is a multithreaded C++20
sampler with a Python callback interface, not a native JAX/XLA/PJRT or CUDA
sampler. It can consume a callable that happens to use JAX, but that is not the
same as keeping an HMC trajectory on the accelerator.

This audit used upstream commit
`ab9895f11986b86f9dc3a0dfe024646c71e2f7c8` (2026-08-11). The project formerly
called WALNUTS is currently packaged as [`walnutpie`](https://github.com/flatironinstitute/walnuts).

## Why the current bridge is unsuitable

`python/src/walnutpie/pyfunc.py` exposes a C callback whose generic path:

1. receives a host pointer from the C++ sampler;
2. presents it to Python as a NumPy array;
3. invokes the Python log-density and gradient callable; and
4. copies the returned log density and gradient back into host buffers.

The source itself marks a faster JAX/PJRT path as a TODO. With a GPU-jitted JAX
likelihood, every leapfrog/micro-step would therefore cross C++ -> Python ->
JAX and synchronize/copy between host and device. Multiple C++ chain threads
do not turn those scalar callbacks into a compiled batched GPU trajectory and
can contend in the Python callback. This loses the key advantage of the
current NumPyro implementation: the complete trajectory control flow and
differentiated potential remain in compiled JAX.

The package's asynchronous stopping concerns multiple MCMC chains reaching
convergence at different times. It is not an asynchronous work queue for
independent wavelength posteriors, and it cannot refill a vacated GPU lane
inside a still-running vectorized NUTS tree.

## What would make it worth revisiting

A useful GPU WALNUTS experiment needs one of these substantially different
implementations:

- a native PJRT/custom-call bridge that passes device buffers without Python
  or host NumPy on every gradient evaluation; or
- a JAX/BlackJAX implementation of the WALNUTS transition and adaptation state
  machine, allowing `jit`, `vmap`, and accelerator-resident loops.

The second route is the better match for this repository, but it is a new
sampler project. Before production use it would need posterior-equivalence,
ESS/second, divergence, adaptation, reproducibility, and real HAT-P-12/HAT-P-65
wall-time comparisons against NumPyro NUTS. Until then, the exact light-curve
kernel and full-potential optimizations have higher return and lower
mathematical risk.
