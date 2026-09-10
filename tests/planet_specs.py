"""Shared helper: default planet-parameter specifications for builder tests.

Reproduces the historical white-light priors in the mapping form that
``koala.config`` now requires: period fixed, t0 uniform +/- 0.05 d around
its centre, b uniform on [0, 1.5], rprs uniform on [0.01, 0.5], and
duration log-uniform on [0.01, 1] (or a_rs log-uniform on [2, 100]).
"""

from koala.config import parse_planet_parameter_specs


def default_planet_specs(period=3.0, t0=1.0, b=0.2, rprs=0.1, duration=0.12,
                         a_rs=10.0, param_method="duration", **overrides):
    planet = {
        "period": {"value": float(period), "prior": "fixed"},
        "t0": {"value": float(t0), "prior": "uniform",
               "low": float(t0) - 0.05, "high": float(t0) + 0.05},
        "b": {"value": float(b), "prior": "uniform", "low": 0.0, "high": 1.5},
        "rprs": {"value": float(rprs), "prior": "uniform",
                 "low": 0.01, "high": 0.5},
    }
    if param_method == "duration":
        planet["duration"] = {"value": float(duration), "prior": "log_uniform",
                              "low": 0.01, "high": 1.0}
    else:
        planet["a_rs"] = {"value": float(a_rs), "prior": "log_uniform",
                          "low": 2.0, "high": 100.0}
    planet.update(overrides)
    return parse_planet_parameter_specs(planet)
