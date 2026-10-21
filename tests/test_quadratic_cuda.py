"""Automatic selection and exact CPU fallback for the optional CUDA kernel."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.jaxoplanet import quadratic_closed, quadratic_cuda, quadratic_likelihood


@pytest.fixture(autouse=True)
def reset_selection():
    functions = [quadratic_cuda._accelerator, quadratic_cuda._native_parts,
                 quadratic_cuda._native_likelihood, quadratic_cuda.fixed_orbit_evaluator]
    for function in functions:
        function.cache_clear()
    yield
    for function in functions:
        function.cache_clear()


def test_cpu_never_searches_for_compiler_and_logs_once(monkeypatch, capsys):
    monkeypatch.setattr(quadratic_cuda, '_architectures', lambda: ())
    def unexpected():
        pytest.fail('CPU evaluation must not look for nvcc')
    monkeypatch.setattr(quadratic_cuda, '_nvcc', unexpected)
    assert quadratic_cuda.fixed_orbit_evaluator() is quadratic_closed.fixed_orbit_evaluator()
    assert quadratic_cuda.fixed_orbit_evaluator() is quadratic_closed.fixed_orbit_evaluator()
    output = capsys.readouterr().out
    assert output.count('Quadratic transit: JAX closed form') == 1


@pytest.mark.parametrize('failure', ['missing', 'unsupported', 'compile', 'load'])
def test_unavailable_accelerator_keeps_jax(monkeypatch, capsys, failure):
    monkeypatch.setattr(quadratic_cuda, '_architectures', lambda: ('sm_70',))
    monkeypatch.setattr(quadratic_cuda, '_nvcc', lambda: None if failure == 'missing' else '/nvcc')
    def build(*args):
        if failure == 'unsupported':
            raise ValueError('unsupported architecture')
        if failure == 'compile':
            raise subprocess.CalledProcessError(1, ['nvcc'])
        return Path('/unavailable/library.so'), 'fake-key'
    monkeypatch.setattr(quadratic_cuda, '_build', build)
    assert quadratic_cuda.fixed_orbit_evaluator() is quadratic_closed.fixed_orbit_evaluator()
    assert quadratic_cuda._accelerator() is None
    assert capsys.readouterr().out.count('Quadratic transit: JAX closed form') == 1


def test_nvidia_architectures_only(monkeypatch):
    devices = [SimpleNamespace(compute_capability='7.0'), SimpleNamespace(),
               SimpleNamespace(compute_capability='8.0'), SimpleNamespace(compute_capability='7.0')]
    monkeypatch.setattr(jax, 'devices', lambda: devices)
    assert quadratic_cuda._architectures() == ('sm_70', 'sm_80')


def test_success_registers_cuda_and_retains_library(monkeypatch, capsys):
    monkeypatch.setattr(quadratic_cuda, '_architectures', lambda: ('sm_70',))
    monkeypatch.setattr(quadratic_cuda, '_nvcc', lambda: '/nvcc')
    builds = []
    def build(*args):
        builds.append(args)
        time.sleep(.02)  # Exercise simultaneous first uses, as a real compiler would.
        return Path('/cache/quad.so'), 'abc123'
    monkeypatch.setattr(quadratic_cuda, '_build', build)
    library = SimpleNamespace(koala_quadratic_partials=object(), koala_quadratic_likelihood=object())
    monkeypatch.setattr(quadratic_cuda.ctypes, 'CDLL', lambda path: library)
    monkeypatch.setattr(jax.ffi, 'pycapsule', lambda value: value)
    calls = []
    monkeypatch.setattr(jax.ffi, 'register_ffi_target', lambda *a, **kw: calls.append((a, kw)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: quadratic_cuda._accelerator(), range(4)))
    result = results[0]
    assert all(value is result for value in results) and len(builds) == 1
    assert result[0] is library
    assert len(calls) == 2 and all(kw == {'platform': 'CUDA'} for _, kw in calls)
    assert set(result[1]) == {'partials', 'likelihood'}
    assert quadratic_cuda._accelerator() is result
    assert capsys.readouterr().out.count('Quadratic transit: CUDA accelerator') == 1


def test_cached_build_identity_and_user_cache(monkeypatch, tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'quadratic_cuda.cu').write_text('first source')
    (source/'quadratic_cuda.h').write_text('header')
    monkeypatch.setattr(quadratic_cuda, '__file__', str(source/'quadratic_cuda.py'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    def output(command, **kwargs):
        return 'sm_70 sm_80' if '--list-gpu-code' in command else 'test compiler version'
    monkeypatch.setattr(subprocess, 'check_output', output)
    calls = []
    def compile(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b'fake test library')
        return SimpleNamespace(stdout='compiled', check_returncode=lambda: None)
    monkeypatch.setattr(subprocess, 'run', compile)
    first, first_key = quadratic_cuda._build('/nvcc', ('sm_70',))
    second, second_key = quadratic_cuda._build('/nvcc', ('sm_70',))
    assert first == second and first_key == second_key and len(calls) == 1
    assert first.is_relative_to(tmp_path/'cache/koala/quadratic_cuda')
    assert '--fmad=false' in calls[0]
    (source/'quadratic_cuda.cu').write_text('changed source')
    third, third_key = quadratic_cuda._build('/nvcc', ('sm_70',))
    assert third != first and third_key != first_key and len(calls) == 2
    with pytest.raises(ValueError, match='does not support'):
        quadratic_cuda._build('/nvcc', ('sm_90',))


def test_cpu_flux_gradient_and_hessian_are_bitwise_jax(monkeypatch):
    monkeypatch.setattr(quadratic_cuda, '_architectures', lambda: ())
    z = jnp.asarray([0., .12, .37, .68, .82, .95, 1.05, 1.2, 1.4])
    q = jnp.asarray([.14683, .3, .2, .001])
    def function(evaluator):
        return lambda q: jnp.sum(evaluator(q[1:3], q[0], z+q[3])**2)
    reference = function(quadratic_closed.fixed_orbit_evaluator())
    candidate = function(quadratic_cuda.fixed_orbit_evaluator())
    for transform in [lambda f: f, jax.grad, jax.hessian]:
        np.testing.assert_array_equal(jax.jit(transform(candidate))(q),
                                      jax.jit(transform(reference))(q))


@pytest.mark.parametrize('columns', [1, 2, 4])
def test_cpu_likelihood_fallback_is_bitwise_jax(monkeypatch, columns):
    monkeypatch.setattr(quadratic_cuda, '_architectures', lambda: ())
    t = jnp.linspace(-.1, .1, 51)
    design = jnp.stack([t**i for i in range(columns)], axis=1)
    theta = jnp.array([.15, .3, .2, .001, 1., *([.002]*(columns-1)), .0002])
    args = (theta, t, jnp.ones_like(t), jnp.full_like(t, 1e-6),
            jnp.ones_like(t, dtype=bool), design, jnp.array([4., 9., .35, 0., 0.]))
    for transform in [jax.value_and_grad, jax.hessian]:
        actual = jax.jit(transform(quadratic_cuda.log_likelihood))(*args)
        expected = jax.jit(transform(quadratic_likelihood.log_likelihood))(*args)
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(left, right)


def test_non_linear_trend_uses_jax_without_native_build(monkeypatch):
    def unexpected():
        pytest.fail('Unsupported trend layout should go directly to JAX')
    monkeypatch.setattr(quadratic_cuda, '_accelerator', unexpected)
    marker = object()
    monkeypatch.setattr(quadratic_likelihood, 'log_likelihood', lambda *args: marker)
    args = (jnp.zeros(9), None, None, None, None, jnp.zeros((3, 4)), None)
    assert quadratic_cuda.log_likelihood(*args) is marker
