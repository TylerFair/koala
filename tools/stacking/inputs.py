"""Read the fitted likelihood inputs used for model stacking."""

from dataclasses import dataclass
from importlib import import_module
import pickle

import numpy as np


@dataclass(frozen=True)
class StackingInputs:
    """A fitted model, its observations, and its wavelength metadata."""

    model: object
    t: object
    yerr: object
    y: object
    model_kwargs: dict
    meta: dict

    @property
    def num_channels(self):
        return np.shape(self.y)[0]

    @property
    def num_cadences(self):
        return np.shape(self.y)[1]


def load_stacking_inputs(path):
    """Load a likelihood export, including existing version-one stage files."""
    with open(path, "rb") as stream:
        payload = pickle.load(stream)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported stacking input schema.")
    spec = payload["model_builder"]
    module, name = spec["identity"].rsplit(".", 1)
    builder = getattr(import_module(module), name)
    kwargs = dict(spec.get("kwargs", {}))
    # Old quadratic exports predate the rotated uniform prior coordinates.
    if kwargs.get("ld_mode") == "uniform" and kwargs.get("ld_profile") == "quadratic":
        kwargs.setdefault("ld_uniform_basis", "coefficients")
    # Historical power-2 fits used the degree-12 projection. Replaying their
    # likelihood must preserve that original target, even after the factory
    # default changes. New exports explicitly record the evaluation method.
    if (spec['identity'] == 'models.jaxoplanet.builder.create_vectorized_model'
            and kwargs.get('ld_profile') == 'power2'):
        kwargs.setdefault('power2_evaluation', 'polynomial')
    run_kwargs = payload.get('model_kwargs', {})
    if (spec['identity'] == 'models.jaxoplanet.builder.create_vectorized_model'
            and kwargs.get('param_method', 'duration') == 'duration'
            and 'mu_a_rs' not in run_kwargs and 'rors_reference' not in run_kwargs):
        raise ValueError('Historical fixed-T14 export has no reference orbit. Refit/export '
                         'with the wavelength-independent geometry model before replay.')
    model = builder if spec.get("callable_is_model") else builder(**kwargs)
    return StackingInputs(
        model, payload["t"], payload["yerr"], payload["indiv_y"],
        payload.get("model_kwargs", {}), payload["meta"],
    )
