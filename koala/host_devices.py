"""CPU host devices for independent spectroscopic sampling.

JAX exposes one CPU device by default, and a vmap over channels on that device
keeps only a few cores busy. ``--xla_force_host_platform_device_count`` gives
JAX one device per core, and the independent samplers then split their lanes
across those devices. The flag is read only when the JAX backend first starts,
so :func:`configure_cpu_host_devices` must run before any module that imports
JAX. This module therefore imports nothing numerical.
"""
import argparse
import glob
import os

import yaml

# Number of host devices this process asked for; checked once JAX is running.
REQUESTED_ENV = 'KOALA_CPU_HOST_DEVICES'
_DEVICE_FLAG = '--xla_force_host_platform_device_count'
_EIGEN_FLAG = '--xla_cpu_multi_thread_eigen'


def available_cpu_cores():
    """Cores this process may run on, bounded by a Slurm allocation."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    slurm = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm and slurm.isdigit():
        cores = min(cores, int(slurm))
    return max(1, int(cores or 1))


def _runs_on_cpu(cfg):
    """Whether ``host_device`` resolves to CPU, decided without starting JAX."""
    requested = str(cfg.get('host_device', 'auto')).lower()
    if requested == 'cpu' or os.environ.get('JAX_PLATFORMS', '').lower() == 'cpu':
        return True
    if requested not in {'gpu', 'auto', 'cuda', 'rocm'}:
        return False
    # 'gpu' and 'auto' fall back to CPU only when no accelerator exists; any
    # visible device node leaves the process exactly as it was.
    return not (glob.glob('/dev/nvidia[0-9]*') or os.path.exists('/dev/kfd'))


def configure_cpu_host_devices(argv=None):
    """Give JAX one CPU device per allocated core for a CPU ``koala`` run.

    ``flags.cpu_host_devices`` overrides the count; 1 restores the single
    device. Multi-device runs also turn off XLA's multi-threaded Eigen
    unless ``XLA_FLAGS`` already sets it. GPU runs, an explicit device count already in ``XLA_FLAGS``, and
    a process that has already imported JAX are left untouched. Returns the
    requested device count, or ``None`` when nothing was changed.
    """
    import sys
    if 'jax' in sys.modules:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config')
    known, _ = parser.parse_known_args(argv)
    if not known.config or not os.path.exists(known.config):
        return None
    with open(known.config, 'r') as stream:
        cfg = yaml.safe_load(stream) or {}
    if not _runs_on_cpu(cfg):
        return None
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if _DEVICE_FLAG in xla_flags:
        return None
    requested = (cfg.get('flags') or {}).get('cpu_host_devices')
    if requested is None:
        count = available_cpu_cores()
    elif isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError('flags.cpu_host_devices must be an integer >= 1.')
    else:
        count = int(requested)
    if count < 2:
        return None
    # Placed first: XLA stops reading XLA_FLAGS at a malformed entry.
    flags = [f'{_DEVICE_FLAG}={count}']
    if _EIGEN_FLAG not in xla_flags:
        # Each device already owns a core; Eigen's shared intra-op pool only
        # oversubscribes them (native HATS-6 PRISM, 24 devices: 1921 s
        # against 2144 s and 2152 s with it).
        flags.append(f'{_EIGEN_FLAG}=false')
    os.environ['XLA_FLAGS'] = ' '.join(flags + [xla_flags]).strip()
    os.environ[REQUESTED_ENV] = str(count)
    return count


def verify_cpu_host_devices(jax, platform):
    """Fail loudly if the requested host devices did not reach JAX."""
    requested = os.environ.get(REQUESTED_ENV)
    if not requested or platform != 'cpu':
        return
    available = len(jax.devices('cpu'))
    if available != int(requested):
        raise RuntimeError(
            f'{REQUESTED_ENV}={requested} CPU host devices were requested, but JAX '
            f'started with {available}. JAX was initialised before '
            'koala.host_devices.configure_cpu_host_devices ran; call it before '
            'importing koala or JAX, or set flags.cpu_host_devices: 1.'
        )
