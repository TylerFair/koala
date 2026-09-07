"""Generate the dependency-light schematic figures used by the tutorials."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_static"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "koala-tutorial-figures"


def finish(fig, name):
    """Save a compact, accessible SVG with editable text."""
    fig.tight_layout()
    fig.savefig(
        OUT / name, format="svg", bbox_inches="tight", metadata={"Date": None}
    )
    plt.close(fig)


def trend_types():
    """Illustrate the baseline components implemented in models/trends.py."""
    t = np.linspace(0.0, 1.0, 500)
    curves = {
        "Linear": 1.0 + 0.018 * (t - 0.5),
        "Quadratic": 1.0 + 0.025 * (t - 0.5) ** 2 - 0.006,
        "Exponential + linear": 1.0 + 0.018 * np.exp(-t / 0.18) - 0.008 * t,
        "Spot template": 1.0 + 0.018 * np.exp(-0.5 * ((t - 0.55) / 0.07) ** 2),
        "Smooth step": 1.0 + 0.016 / (1.0 + np.exp(-(t - 0.55) / 0.018)),
    }
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for (label, y), color in zip(curves.items(), colors):
        ax.plot(t, y, lw=2.2, label=label, color=color)
    ax.axhline(1.0, color="0.75", lw=1, zorder=0)
    ax.set(xlabel="Time through the visit", ylabel="Relative baseline")
    ax.set_xticks([0, 0.5, 1], ["start", "midpoint", "end"])
    ax.legend(
        frameon=False, ncol=3, loc="lower left",
        bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01, 0.02, "Illustrative amplitudes; the transit is omitted",
        transform=ax.transAxes, fontsize=9, color="0.35",
    )
    finish(fig, "tutorial_trend_types.svg")


def limb_darkening():
    """Illustrate the quadratic and power-2 laws used by the fitter."""
    mu = np.linspace(0.0, 1.0, 500)
    quadratic = 1.0 - 0.34 * (1.0 - mu) - 0.20 * (1.0 - mu) ** 2
    power2 = 1.0 - 0.62 * (1.0 - mu**0.72)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(mu, quadratic, lw=2.6, color="#0072B2", label="Quadratic")
    ax.plot(mu, power2, lw=2.6, color="#D55E00", ls="--", label="Power-2")
    ax.scatter([0, 1], [quadratic[0], 1], s=26, color="#0072B2", zorder=3)
    ax.set(
        xlabel=r"$\mu = \cos(\theta)$  (limb $\longrightarrow$ disc centre)",
        ylabel=r"Specific intensity  $I(\mu) / I(1)$",
        xlim=(0, 1), ylim=(0.34, 1.03),
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.98, 0.03, "Illustrative coefficients",
        transform=ax.transAxes, fontsize=9, color="0.35", ha="right",
    )
    finish(fig, "tutorial_limb_darkening.svg")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    trend_types()
    limb_darkening()
