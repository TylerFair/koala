"""Spectroscopic resident-width memory estimates."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpectroMemoryModel:
    """Affine peak-memory model for a fixed sampler/model family."""

    intercept_bytes: float
    bytes_per_lane: float
    bytes_per_lane_cadence: float
    bytes_per_draw_lane: float = 0.0

    def predict(self, width: int, cadences: int, draws: int = 0) -> int:
        width, cadences, draws = int(width), int(cadences), int(draws)
        if width < 1 or cadences < 1 or draws < 0:
            raise ValueError("width/cadences must be positive and draws non-negative.")
        value = (
            self.intercept_bytes
            + self.bytes_per_lane * width
            + self.bytes_per_lane_cadence * width * cadences
            + self.bytes_per_draw_lane * width * draws
        )
        if not math.isfinite(value) or value < 0:
            raise ValueError("Memory model predicted an invalid byte count.")
        return int(math.ceil(value))


def resolve_spectro_auto_width(
    model,
    *,
    bytes_limit,
    cadences,
    draws,
    speed_cap,
    max_channels,
    headroom_fraction=0.25,
):
    """Return the largest predicted-safe width up to the speed cap."""
    if not isinstance(model, SpectroMemoryModel):
        model = SpectroMemoryModel(**dict(model))
    bytes_limit = int(bytes_limit)
    headroom_fraction = float(headroom_fraction)
    if bytes_limit <= 0 or not 0.0 <= headroom_fraction < 1.0:
        raise ValueError("Invalid device limit or headroom fraction.")
    cap = min(int(speed_cap), int(max_channels))
    if cap < 1:
        raise ValueError("speed_cap and max_channels must be positive.")
    budget = int(math.floor(bytes_limit * (1.0 - headroom_fraction)))
    if model.predict(1, cadences, draws) > budget:
        raise ValueError("Even one spectroscopic lane exceeds the memory budget.")
    low, high = 1, cap
    while low < high:
        middle = (low + high + 1) // 2
        if model.predict(middle, cadences, draws) <= budget:
            low = middle
        else:
            high = middle - 1
    return low
