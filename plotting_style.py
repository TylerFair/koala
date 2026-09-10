"""Shared publication plotting style for pipeline and notebook figures.

The optional :mod:`scienceplots` and :mod:`cmcrameri` packages improve the
appearance and palette selection, but every helper has a Matplotlib-only
fallback so plotting remains available in the base scientific environment.
"""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np

try:  # Import registers the ``science`` and ``nature`` style sheets.
    import scienceplots  # noqa: F401
except Exception:  # pragma: no cover - availability is environment-specific.
    scienceplots = None

try:
    from cmcrameri import cm as cmc
except Exception:  # pragma: no cover - availability is environment-specific.
    cmc = None


PLOT_DPI = 250
MODEL_COLOR = "k"
ZERO_LINE_COLOR = "0.40"
ERRORBAR_COLOR = "0.72"
DEFAULT_PALETTE = "orchid"
CURRENT_PALETTE = DEFAULT_PALETTE
LIGHTCURVE_COLOR = "mediumorchid"
SPECTRUM_COLOR = LIGHTCURVE_COLOR
AVAILABLE_PALETTES = (
    "orchid",
    "batlow",
    "bamako",
    "lajolla",
    "tokyo",
    "oslo",
    "hawaii",
    "lipari",
)

DATA_MARKER_STYLE = {
    "fmt": "o",
    "ms": 3.6,
    "mfc": "white",
    "mew": 1.7,
    "elinewidth": 1.7,
    "capsize": 0,
    "linestyle": "none",
}
OUTLIER_MARKER_STYLE = {
    "linestyle": "none",
    "marker": "o",
    "ms": 2.6,
    "mfc": "0.82",
    "mec": "0.82",
    "mew": 0.0,
    "alpha": 0.8,
}


def _cmc_sample(name: str, value: float):
    if cmc is None or not hasattr(cmc, name):
        return "mediumorchid"
    return getattr(cmc, name)(value)


PALETTE_TO_COLOR = {
    "orchid": "mediumorchid",
    "batlow": _cmc_sample("batlow", 0.72),
    "bamako": _cmc_sample("bamako", 0.68),
    "lajolla": _cmc_sample("lajolla", 0.58),
    "tokyo": _cmc_sample("tokyo", 0.66),
    "oslo": _cmc_sample("oslo", 0.78),
    "hawaii": _cmc_sample("hawaii", 0.63),
    "lipari": _cmc_sample("lipari", 0.70),
}


def set_plot_palette(name: str = DEFAULT_PALETTE) -> None:
    """Select one house palette without requiring cmcrameri."""
    global CURRENT_PALETTE, LIGHTCURVE_COLOR, SPECTRUM_COLOR
    if name not in PALETTE_TO_COLOR:
        raise ValueError(
            f"Unknown palette {name!r}; choose one of {AVAILABLE_PALETTES}."
        )
    CURRENT_PALETTE = name
    LIGHTCURVE_COLOR = PALETTE_TO_COLOR[name]
    SPECTRUM_COLOR = LIGHTCURVE_COLOR


def accent_color_for_label(label: str | None = None):
    """Return a related accent shade for a JWST observing mode."""
    if CURRENT_PALETTE == "orchid" or label is None:
        return LIGHTCURVE_COLOR

    upper = str(label).upper().replace(" ", "_")
    if "NIRISS" in upper or "SOSS" in upper:
        sample = 0.26 if ("ORDER2" in upper or "O2" in upper) else 0.18
    elif "PRISM" in upper:
        sample = 0.38
    elif "G140H" in upper:
        sample = 0.50
    elif "G235H" in upper:
        sample = 0.58
    elif "G395M" in upper:
        sample = 0.68
    elif "G395H" in upper:
        sample = 0.80
    elif "LRS" in upper or "MIRI" in upper:
        sample = 0.88
    else:
        sample = 0.70
    return _cmc_sample(CURRENT_PALETTE, sample)


def related_accent_colors(label: str | None, count: int) -> list:
    """Make a quiet same-hue sequence for multi-curve diagnostic figures."""
    if count <= 0:
        return []
    accent = np.asarray(mpl_colors.to_rgb(accent_color_for_label(label)))
    blends = np.linspace(0.45, 0.0, count)
    return [tuple((1.0 - amount) * accent + amount) for amount in blends]


def model_palette_colors(count: int) -> list:
    """Return distinct, colorblind-safe colors for model identities.

    The discrete scientific ``batlow`` palette is deliberately independent of
    the observing-mode accent: model colors identify assumptions, while the
    accent continues to identify the combined science result.  Matplotlib's
    qualitative palette keeps the same distinction when cmcrameri is absent.
    """
    if count <= 0:
        return []
    if cmc is not None and hasattr(cmc, "batlowS"):
        colors = cmc.batlowS.colors
        # Start with mid/dark entries because model curves are intentionally
        # faded; the lightest categorical entries would disappear on white.
        preferred = (0, 4, 2, 3, 7, 8, 5, 6, 9, 1)
        ordered = [colors[index] for index in preferred]
        ordered.extend(colors[index] for index in range(len(preferred), len(colors)))
        return [tuple(ordered[index % len(ordered)]) for index in range(count)]
    fallback = plt.get_cmap("tab10")
    return [fallback(index % fallback.N) for index in range(count)]


def _latex_toolchain_available() -> bool:
    """Return True when Matplotlib's ``text.usetex`` can actually render."""
    return all(shutil.which(tool) for tool in ("latex", "dvipng")) or bool(
        shutil.which("latex") and shutil.which("gs")
    )


def _apply_science_style() -> bool:
    """Apply the SciencePlots style if it is installed and usable."""
    if scienceplots is None:
        return False
    try:
        plt.style.use(["science", "nature"])
    except Exception as exc:  # pragma: no cover - environment-specific.
        warnings.warn(
            f"SciencePlots is installed but its style could not be applied "
            f"({exc}); using the Matplotlib serif fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
        plt.rcdefaults()
        return False
    if plt.rcParams.get("text.usetex", False) and not _latex_toolchain_available():
        # The ``science`` style renders text with LaTeX, which crashes the
        # first savefig on machines without a TeX install.
        plt.rcParams["text.usetex"] = False
    return True


def apply_publication_style() -> None:
    """Apply the reference ``science``/``nature`` style or serif fallback."""
    if not _apply_science_style():
        plt.rcParams.update(
            {
                "font.family": "serif",
                "axes.spines.top": True,
                "axes.spines.right": True,
            }
        )
    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.linewidth": 1.4,
            "axes.labelpad": 12.0,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "xtick.major.pad": 8.0,
            "ytick.major.pad": 8.0,
            "figure.constrained_layout.use": True,
            "legend.frameon": False,
            "savefig.bbox": None,
        }
    )


def _infer_half_widths(wavelengths: np.ndarray) -> np.ndarray:
    """Infer wavelength-bin half widths from adjacent bin centres."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.size <= 1:
        return np.full_like(wavelengths, 0.05, dtype=float)
    mids = 0.5 * (wavelengths[1:] + wavelengths[:-1])
    edges = np.empty(wavelengths.size + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = wavelengths[0] - (mids[0] - wavelengths[0])
    edges[-1] = wavelengths[-1] + (wavelengths[-1] - mids[-1])
    return 0.5 * (edges[1:] - edges[:-1])


def choose_symmetric_residual_ticks(
    residual_ppm: np.ndarray,
    residual_err_ppm: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Choose a symmetric five-tick residual scale containing the data."""
    residual = np.asarray(residual_ppm, dtype=float)
    finite = np.isfinite(residual)
    if residual_err_ppm is not None:
        error = np.asarray(residual_err_ppm, dtype=float)
        finite &= np.isfinite(error)
        extent = (
            np.nanmax(np.abs(residual[finite]) + np.abs(error[finite]))
            if np.any(finite)
            else np.nan
        )
    else:
        extent = np.nanmax(np.abs(residual[finite])) if np.any(finite) else np.nan

    if not np.isfinite(extent) or extent <= 0:
        middle_tick = 500.0
    else:
        target = extent / 2.0
        candidates = [
            mantissa * 10.0**power
            for power in range(-1, 7)
            for mantissa in (1.0, 2.5, 5.0)
        ]
        middle_tick = next(
            (candidate for candidate in candidates if candidate >= target),
            candidates[-1],
        )
    ticks = middle_tick * np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    return 2.0 * middle_tick, ticks


def enforce_min_major_ticks(ax, *, x: bool = True, y: bool = True) -> None:
    """Keep sparse diagnostic panels from collapsing to only two ticks."""
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))


def style_axis(ax, *, x_locator: bool = True, y_locator: bool = True) -> None:
    """Apply the reference inward major ticks and disable minor ticks."""
    enforce_min_major_ticks(ax, x=x_locator, y=y_locator)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        length=6,
        width=1.4,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        bottom=False,
        left=False,
        top=False,
        right=False,
    )


def _label_tokens(label: str) -> list[str]:
    name = Path(str(label)).stem
    name = re.sub(r"^[0-9]+_", "", name)
    return [token for token in re.split(r"[_\s/]+", name) if token]


def format_instrument_stamp(label: str | None) -> str:
    """Format a compact observing-mode stamp from a product label or path."""
    if not label:
        return "JWST"
    tokens = _label_tokens(str(label))
    upper = [token.upper() for token in tokens]
    if "NIRISS" in upper:
        idx = upper.index("NIRISS")
        order = next(
            (token.upper() for token in tokens[idx + 1 :] if token.upper().startswith("ORDER")),
            "",
        )
        return " ".join(part for part in ("JWST NIRISS/SOSS", order) if part)
    if "NIRSPEC" in upper:
        idx = upper.index("NIRSPEC")
        grating = next(
            (token.upper() for token in tokens[idx + 1 :] if re.fullmatch(r"G[0-9]{3}[HM](-F0[7-9]0|-F100)?|PRISM", token.upper())),
            "",
        )
        detector = next(
            (token.upper() for token in tokens[idx + 1 :] if re.fullmatch(r"NRS[12]", token.upper())),
            "",
        )
        mode = "/".join(part for part in (grating, detector) if part)
        return " ".join(part for part in ("JWST NIRSpec", mode) if part)
    if "NIRCAM" in upper:
        idx = upper.index("NIRCAM")
        filt = next(
            (token.upper() for token in tokens[idx + 1 :] if re.fullmatch(r"F[0-9]{3}W2?", token.upper())),
            "",
        )
        return " ".join(part for part in ("JWST NIRCam", filt) if part)
    if "MIRI" in upper:
        return "JWST MIRI/LRS" if "LRS" in upper else "JWST MIRI"
    return Path(str(label)).stem.replace("_", " ")


def format_instrument_stamp_display(label: str | None) -> str:
    stamp = format_instrument_stamp(label)
    if plt.rcParams.get("text.usetex", False):
        escaped = stamp.replace("_", r"\_")
        return rf"\textbf{{{escaped}}}"
    return stamp


def add_instrument_stamp(
    ax,
    label: str | None,
    *,
    fontsize: float = 16.0,
    y_data: float | None = None,
):
    """Add the single allowed free-text stamp to an ordinary pipeline plot."""
    if y_data is None:
        y = 0.96
        transform = ax.transAxes
    else:
        y = float(y_data)
        transform = blended_transform_factory(ax.transAxes, ax.transData)
    return ax.text(
        0.02,
        y,
        format_instrument_stamp_display(label),
        transform=transform,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        zorder=20,
    )


def save_figure(fig, filename, *, dpi: int = PLOT_DPI, close: bool = True) -> None:
    """Save consistently while retaining Matplotlib's extension semantics."""
    fig.savefig(filename, dpi=dpi)
    if close:
        plt.close(fig)
