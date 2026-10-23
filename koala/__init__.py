"""Koala: joint white-light and spectroscopic JWST inference."""


from .console import install_warning_routing

install_warning_routing()


def fit(config_path, verbose=False):
    """Fit a YAML configuration, with progress shown in the calling session.

    Call before importing JAX in a fresh notebook kernel so CPU host devices
    can be configured. Uses the same pipeline and defaults as ``koala -c``.
    """
    import os

    from .host_devices import configure_cpu_host_devices

    from .console import configure
    configure(verbose)
    config_path = os.fspath(config_path)
    configure_cpu_host_devices(['-c', config_path])
    from .config import load_config
    from .pipeline import run

    return run(load_config(config_path), config_path=config_path, verbose=verbose)
