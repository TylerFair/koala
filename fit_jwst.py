"""Compatibility facade and command-line entry point for Koala.

The implementation lives in :mod:`koala.pipeline`. Attribute assignment is
forwarded as well as lookup so existing monkeypatch seams retain the behavior
of the former single-module implementation.
"""

import os
import sys
import types


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from koala import pipeline as _pipeline


for _name, _value in vars(_pipeline).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__all__ = tuple(name for name in vars(_pipeline) if not name.startswith("__"))


class _CompatibilityFacade(types.ModuleType):
    def __getattr__(self, name):
        return getattr(_pipeline, name)

    def __setattr__(self, name, value):
        if not name.startswith("__"):
            for module_name, module in tuple(sys.modules.items()):
                if (
                    (module_name == "koala.pipeline" or module_name.startswith("koala."))
                    and module is not None
                    and hasattr(module, name)
                ):
                    setattr(module, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CompatibilityFacade


if __name__ == "__main__":
    try:
        main()
    except SamplerInputsDumpExit as error:
        print(
            "Requested sampler-input dump is complete; exiting before "
            f"high-resolution sampling ({error}).",
            flush=True,
        )
