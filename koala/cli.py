"""Installed command-line entry point."""


def main():
    import sys

    from .host_devices import configure_cpu_host_devices

    configure_cpu_host_devices(sys.argv[1:])
    from .pipeline import main as run

    return run()
