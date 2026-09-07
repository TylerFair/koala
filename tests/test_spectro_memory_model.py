import pytest

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
