"""Keep the repeatable injections consistent with their fixed fit geometry."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from tools import example_phase_curves as examples


def test_generator_preserves_cadences_at_absolute_observation_epochs(monkeypatch):
    import jax
    from models.jaxoplanet import surface

    def elapsed_time(params, time, **kwargs):
        return time - time[0]

    monkeypatch.setattr(surface, "compute_surface_model", elapsed_time)
    original_precision = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        time = np.array([60000.0, 60000.0 + 1e-4])
        signal = examples._surface_signals(time, np.array([.1]), "eclipse")
        np.testing.assert_allclose(signal[:, 0], time - time[0], atol=1e-12, rtol=0)
    finally:
        jax.config.update("jax_enable_x64", original_precision)


@pytest.mark.parametrize("scenario", ["eclipse", "phase_curve", "stellar_spots"])
def test_example_injection_matches_fixed_config_geometry(monkeypatch, scenario):
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "examples" / f"{scenario}.yaml").read_text()
    )
    planet = config["planet"]
    assert "fit_geometry" not in config.get("flags", {})
    # The injected geometry is fixed in the example fits: every geometry
    # parameter carries ``prior: fixed``.
    # ``t0`` (or ``eclipse_time`` for an eclipse) and the surface fluxes may
    # be free; the radius ratio and period the injection relies on are fixed.
    assert planet["period"]["prior"] == "fixed"
    assert planet["rprs"]["prior"] == "fixed"
    assert ("t0" in planet) != ("eclipse_time" in planet)
    seen_radii = []

    def record_geometry(time, radius_ratios, *args, **kwargs):
        seen_radii.extend(radius_ratios)
        return np.zeros((len(time), len(radius_ratios)))

    monkeypatch.setattr(examples, "_surface_signals", record_geometry)
    _, _, truth = examples._scenario(scenario, np.linspace(2.9, 5.0, 14))
    np.testing.assert_allclose(seen_radii, planet["rprs"]["value"], rtol=0, atol=0)
    for truth_key, config_key in (
        ("period_days", "period"), ("a_rs", "a_rs"),
        ("impact_parameter", "b"), ("radius_ratio", "rprs"),
    ):
        assert truth[truth_key] == planet[config_key]["value"]
    if "t0" in planet:
        assert truth["t0_bmjd_tdb"] == planet["t0"]["value"]
    else:
        # Circular orbit: the configured eclipse time is half a period after t0.
        assert truth["t0_bmjd_tdb"] + 0.5 * truth["period_days"] == pytest.approx(
            planet["eclipse_time"]["value"], abs=1e-9
        )
    if scenario == "stellar_spots":
        assert truth["rotation_period_days"] == config["stellar"]["rotation_period"]
        assert len(truth["spots"]) == len(config["stellar"]["spots"])
        for injected, configured in zip(truth["spots"], config["stellar"]["spots"]):
            for key, value in injected.items():
                assert configured[key] == value
