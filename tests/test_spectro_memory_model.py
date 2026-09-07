import pytest

from koala import config
from models.channel_batching import (
    SpectroMemoryModel,
    resolve_spectro_auto_width,
)


def test_auto_width_applies_headroom_and_speed_cap():
    model = SpectroMemoryModel(100, 0, 10, 0)
    assert resolve_spectro_auto_width(
        model, bytes_limit=1100, cadences=10, draws=100,
        speed_cap=20, max_channels=50, headroom_fraction=0.25,
    ) == 7
    assert resolve_spectro_auto_width(
        model, bytes_limit=10_000, cadences=10, draws=100,
        speed_cap=8, max_channels=50,
    ) == 8


def test_auto_width_rejects_impossible_budget():
    with pytest.raises(ValueError, match="Even one"):
        resolve_spectro_auto_width(
            SpectroMemoryModel(1000, 0, 0), bytes_limit=1000,
            cadences=10, draws=1, speed_cap=4, max_channels=4,
        )


class _MemoryDevice:
    def memory_stats(self):
        return {"bytes_limit": 12_701_761_536}


def _prism_width(monkeypatch, flags):
    monkeypatch.setattr(config.jax, "devices", lambda: [_MemoryDevice()])
    return config._resolve_stage_vmap_width(
        flags,
        "highres",
        "auto",
        num_cadences=40_715,
        mcmc_kwargs={"num_samples": 1000},
        sampler_backend="independent_nuts",
        trend_inference="sampled_uniform",
        transit_engine="jaxoplanet",
        ld_profile="power2",
        ld_mode="stellarprior",
        detrend_type="explinear_spectroscopic",
        param_method="duration",
        n_planets=1,
        transit_window="auto",
        transit_grid_non_grazing=True,
    )


def test_long_prism_auto_width_uses_measured_accelerated_cap(monkeypatch):
    assert _prism_width(monkeypatch, {}) == 40


def test_long_prism_auto_width_retains_conservative_cap_when_disabled(
    monkeypatch,
):
    assert _prism_width(
        monkeypatch, {"spectro_transit_grid": "off"}
    ) == 4
