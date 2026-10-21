"""Optional CUDA accelerator for the closed-form quadratic model.

Compilation is lazy, on first use with a CUDA device. A missing compiler,
unsupported architecture or failed build leaves the JAX implementation in
place. Nothing is compiled during package installation. Higher derivatives
are evaluated by JAX from the same closed-form expressions.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading

import jax
import jax.numpy as jnp

from . import quadratic_closed, quadratic_likelihood
from .core import orbit_separation


_REGISTRATION_LOCK = threading.RLock()


def _nvcc():
    executable = shutil.which('nvcc')
    if executable:
        return executable
    directory = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if directory:
        executable = Path(directory) / 'bin/nvcc'
        if executable.is_file() and os.access(executable, os.X_OK):
            return str(executable)
    return None


def _architectures():
    # ROCm devices have no NVIDIA compute capability. Merely importing this
    # module does not initialize CUDA or run a compiler.
    capabilities = {getattr(device, 'compute_capability', None) for device in jax.devices()}
    return tuple(sorted('sm_'+str(value).replace('.', '')
                        for value in capabilities if value))


def _build(nvcc, architectures):
    """Build once per source/toolchain/architecture identity in the user cache."""
    import fcntl
    import jaxlib

    supported = subprocess.check_output([nvcc, '--list-gpu-code'], text=True, timeout=20).split()
    if not set(architectures).issubset(supported):
        raise ValueError('nvcc does not support '+', '.join(architectures))
    version = subprocess.check_output([nvcc, '--version'], text=True, timeout=20)
    source = Path(__file__).with_suffix('.cu')
    header = source.with_suffix('.h')
    include = Path(jax.ffi.include_dir())
    compiler = shutil.which('g++', path=os.defpath) or shutil.which('g++')
    if compiler is None:
        raise ValueError('no C++ host compiler for nvcc')
    compiler_version = subprocess.check_output([compiler, '--version'], text=True, timeout=20)
    flags = ['-std=c++17', '-O3', '--fmad=false', '--shared', '-Xcompiler', '-fPIC']
    identity = dict(nvcc=version, compiler=compiler_version, architectures=architectures,
                    flags=flags, jax=jax.__version__, jaxlib=jaxlib.__version__,
                    machine=platform.machine(), sources={
                        p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [source, header, *sorted((include/'xla/ffi/api').glob('*.h'))]
                    })
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache')) / 'koala/quadratic_cuda' / key
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    library = cache / 'quadratic.so'
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            command = [nvcc, *flags, '-ccbin', compiler, '-I'+str(include)]
            for arch in architectures:
                command += ['-gencode', f'arch=compute_{arch[3:]},code={arch}']
            environment = dict(os.environ)
            # Avoid injecting a different host compiler's headers or linker.
            environment['PATH'] = str(Path(compiler).parent)+os.pathsep+environment.get('PATH', '')
            for name in ('GCC_EXEC_PREFIX', 'COMPILER_PATH', 'LIBRARY_PATH', 'CPATH',
                         'CPLUS_INCLUDE_PATH', 'LD_LIBRARY_PATH'):
                environment.pop(name, None)
            with tempfile.TemporaryDirectory(prefix='build-', dir=cache) as temporary:
                output = Path(temporary)/'quadratic.so'
                result = subprocess.run(command+[str(source), '-o', str(output)],
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        timeout=180, env=environment)
                (cache/'build.log').write_text(result.stdout)
                result.check_returncode()
                os.replace(output, library)
            (cache/'identity.json').write_text(json.dumps(identity, indent=2))
    return library, key


@lru_cache(maxsize=1)
def _load_accelerator():
    architectures = _architectures()
    reason = 'no CUDA device'
    try:
        if architectures:
            compiler = _nvcc()
            if compiler is None:
                reason = 'nvcc was not found'
            elif os.name != 'posix':
                reason = 'unsupported CUDA build host'
            else:
                path, key = _build(compiler, architectures)
                library = ctypes.CDLL(str(path))
                targets = {}
                for operation in ('partials', 'likelihood'):
                    symbol = 'koala_quadratic_'+operation
                    name = symbol+'_'+key
                    jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(getattr(library, symbol)),
                                                platform='CUDA')
                    targets[operation] = name
                print('Quadratic transit: CUDA accelerator; JAX for higher derivatives.', flush=True)
                return library, targets  # Keep the library alive while executables use it.
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as error:
        reason = 'CUDA build/load unavailable ('+str(error).splitlines()[0]+')'
    print('Quadratic transit: JAX closed form ('+reason+').', flush=True)
    return None


def _accelerator():
    # lru_cache alone can execute two simultaneous misses. Serialize loading
    # and FFI registration as well as the cross-process build file lock.
    with _REGISTRATION_LOCK:
        return _load_accelerator()


_accelerator.cache_clear = _load_accelerator.cache_clear


def _active_jvp(function, primals, tangents, *, start=0):
    indices = [i for i in range(start, len(primals))
               if not isinstance(tangents[i], jax.custom_derivatives.SymbolicZero)]
    if not indices:
        return jnp.zeros_like(function(*primals))
    def selected(*args):
        values = list(primals)
        for i, value in zip(indices, args):
            values[i] = value
        return function(*values)
    return jax.jvp(selected, tuple(primals[i] for i in indices),
                   tuple(tangents[i] for i in indices))[1]


def _jax_parts(u, radius, separation):
    return jnp.stack(quadratic_closed.partials(u, radius, separation), axis=-1)


@lru_cache(maxsize=1)
def _native_parts():
    target = _accelerator()[1]['partials']
    def native(u, radius, separation):
        packed = jnp.stack(jnp.broadcast_arrays(u[0], u[1], radius, separation), axis=-1)
        shape = jax.ShapeDtypeStruct(packed.shape[:-1]+(5,), jnp.float64)
        return jax.ffi.ffi_call(target, shape, vmap_method='broadcast_all')(packed)

    @jax.custom_jvp
    def pair(u, radius, separation):
        return jax.lax.platform_dependent(u, radius, separation, cuda=native, default=_jax_parts)

    def pair_rule(primals, tangents):
        return pair(*primals), _active_jvp(_jax_parts, primals, tangents)
    pair.defjvp(pair_rule, symbolic_zeros=True)
    return pair


def partials(u, radius, separation):
    """Flux and all four local partials, falling back exactly to JAX."""
    if _accelerator() is None or jnp.result_type(u, radius, separation) != jnp.float64:
        return quadratic_closed.partials(u, radius, separation)
    values = _native_parts()(u, radius, separation)
    return tuple(values[..., i] for i in range(5))


@lru_cache(maxsize=1)
def fixed_orbit_evaluator():
    if _accelerator() is None:
        return quadratic_closed.fixed_orbit_evaluator()

    @jax.custom_jvp
    def evaluate(u, radius, separation):
        return partials(u, radius, separation)[0]

    def rule(primals, tangents):
        value, dr, dz, du1, du2 = partials(*primals)
        u_dot, r_dot, z_dot = tangents
        zero = jax.custom_derivatives.SymbolicZero
        tangent = jnp.zeros_like(value)
        if not isinstance(u_dot, zero):
            tangent += du1*u_dot[0]+du2*u_dot[1]
        if not isinstance(r_dot, zero):
            tangent += dr*r_dot
        if not isinstance(z_dot, zero):
            tangent += dz*z_dot
        return value, tangent
    evaluate.defjvp(rule, symbolic_zeros=True)
    return evaluate


def _jax_score(*args):
    value, gradient = quadratic_likelihood.value_and_grad(*args)
    return jnp.concatenate((value[None], gradient))


@lru_cache(maxsize=1)
def _native_likelihood():
    target = _accelerator()[1]['likelihood']
    def native(theta, t, y, variance, mask, design, orbit):
        period, a_rs, impact, ecc, omega = orbit
        def separation(epoch):
            return orbit_separation(t, period, epoch, a_rs, impact, ecc, omega)
        (z, front), (z_dot, _) = jax.jvp(separation, (theta[3],), (jnp.ones_like(theta[3]),))
        outputs = (jax.ShapeDtypeStruct((8,), jnp.float64),
                   jax.ShapeDtypeStruct(((len(t)+127)//128, 8), jnp.float64))
        value, _ = jax.ffi.ffi_call(target, outputs, vmap_method='broadcast_all')(
            theta, z, z_dot, y, variance, mask, design, front)
        return value

    @jax.custom_jvp
    def pair(*args):
        return jax.lax.platform_dependent(*args, cuda=native, default=_jax_score)

    def pair_rule(primals, tangents):
        return pair(*primals), _active_jvp(_jax_score, primals, tangents)
    pair.defjvp(pair_rule, symbolic_zeros=True)

    @jax.custom_jvp
    def evaluate(*args):
        return pair(*args)[0]

    def rule(primals, tangents):
        values = pair(*primals)
        tangent = (jnp.zeros_like(values[0])
                   if isinstance(tangents[0], jax.custom_derivatives.SymbolicZero)
                   else jnp.sum(values[1:]*tangents[0]))
        if any(not isinstance(x, jax.custom_derivatives.SymbolicZero) for x in tangents[1:]):
            tangent += _active_jvp(quadratic_likelihood.log_likelihood, primals, tangents, start=1)
        return values[0], tangent
    evaluate.defjvp(rule, symbolic_zeros=True)
    return evaluate


def log_likelihood(theta, t, y, variance, mask, design, orbit):
    """Use the fused CUDA score for two-column trends, otherwise the JAX score."""
    if (design.shape[-1] != 2 or theta.dtype != jnp.float64 or _accelerator() is None):
        return quadratic_likelihood.log_likelihood(theta, t, y, variance, mask, design, orbit)
    return _native_likelihood()(theta, t, y, variance, mask, design, orbit)
