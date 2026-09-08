"""Installed command-line entry point."""


def main():
    from .pipeline import SamplerInputsDumpExit, main as run

    try:
        return run()
    except SamplerInputsDumpExit as error:
        print(
            "Requested sampler-input dump is complete; exiting before "
            f"high-resolution sampling ({error}).",
            flush=True,
        )
