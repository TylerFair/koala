#!/usr/bin/env python3
"""Score completed stacking variants and render saved stacking products."""
from __future__ import annotations
import argparse, csv, io, json, pickle, re, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml
from plotting_style import (
    DATA_MARKER_STYLE,
    ZERO_LINE_COLOR,
    _infer_half_widths,
    accent_color_for_label,
    apply_publication_style,
    model_palette_colors,
    save_figure,
    style_axis,
)


_ARRAYS_SUFFIX = "_stacking_arrays.npz"
_MODEL_LABELS = {
    "fixed_power2_linear": "Fixed power-2",
    "stellarprior_power2_linear": "Stellar-prior power-2",
    "uniform_quadratic_linear": "Uniform quadratic",
    "uniform_quadratic_linear_uplus_v2": "Wide-uniform quadratic",
    "sing_quadratic_linear": "Sing quadratic",
    "sing_quadratic_linear_fitted_v2": "Fitted-Sing quadratic",
    "stellarprior_linear": "Stellar-prior linear",
    "uniform_linear": "Uniform linear",
    "fixed_linear": "Fixed linear",
    "stellarprior_linear_step": "Stellar-prior linear + step",
}

def _normalized_stage_kind(value):
    return (
        str(value)
        .lower()
        .replace("-", "_")
        .replace("highres", "high_resolution")
        .replace("lowres", "low_resolution")
    )


def _stage_checkpoint_prefix(stage):
    label = str(stage.meta.get("stage_label", ""))
    kind = _normalized_stage_kind(stage.meta.get("stage_kind", ""))
    suffix = f"_{kind}" if kind else ""
    if suffix and label.endswith(suffix):
        return label[:-len(suffix)]
    raise ValueError(
        "Stage inputs must include matching stage_label and stage_kind metadata."
    )


def _load_chunks(output_dir, stage):
    """Load the one complete checkpoint family matching the selected stage."""
    families = {}
    for path in (output_dir / "chunks").glob("*chunk_*_*.pkl"):
        match = re.search(r"_chunk_(\d+)_(\d+)\.pkl$", path.name)
        if match:
            prefix = re.sub(r"_chunk_\d+_\d+\.pkl$", "", path.name)
            families.setdefault(prefix, []).append((int(match[1]), int(match[2]), path))
    stage_prefix = _stage_checkpoint_prefix(stage)
    stage_kind = _normalized_stage_kind(stage.meta["stage_kind"])
    expected_channels = int(
        stage.meta.get("num_channels", len(np.asarray(stage.meta["wavelength"])))
    )
    matching = []
    incomplete = []
    for prefix, family in families.items():
        if prefix != stage_prefix and not prefix.startswith(stage_prefix + "_"):
            continue
        manifest_path = output_dir / "chunks" / f"{prefix}.manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            signature_kind = _normalized_stage_kind(
                manifest.get("checkpoint_signature", {}).get("stage", "")
            )
            if signature_kind and signature_kind != stage_kind:
                continue
        chunks = sorted(family)
        contiguous = (
            chunks[0][0] == 0
            and all(chunks[i][1] == chunks[i + 1][0]
                    for i in range(len(chunks) - 1))
            and chunks[-1][1] == expected_channels
        )
        if contiguous:
            matching.append((prefix, chunks))
        else:
            incomplete.append(prefix)
    if incomplete:
        raise ValueError(
            f"Incomplete checkpoint families for {stage.meta['stage_label']!r} "
            f"under {output_dir}: {sorted(incomplete)}"
        )
    if not matching:
        raise FileNotFoundError(
            f"Expected one complete checkpoint family for "
            f"{stage.meta['stage_label']!r} with {expected_channels} channels "
            f"under {output_dir}; found none."
        )
    if len(matching) > 1:
        raise ValueError(
            f"Ambiguous checkpoint families for {stage.meta['stage_label']!r} "
            f"under {output_dir}: {[prefix for prefix, _ in matching]}"
        )
    _, chunks = matching[0]
    payloads = []
    for _, _, path in chunks:
        with path.open("rb") as stream: payload = pickle.load(stream)
        # ESS-selective fallback checkpoints include routing metadata beside
        # the posterior mapping.
        payloads.append(payload.get("samples", payload))
    return {key: np.concatenate([np.asarray(x[key]) for x in payloads], axis=1)
            for key in payloads[0]}

def _stage(stage_inputs_dir, stage_kind):
    from tools.spectro_stage_inputs import load_stage_inputs

    paths = sorted(Path(stage_inputs_dir).glob(f"*{stage_kind}_inputs.pkl"))
    if len(paths) != 1: raise FileNotFoundError(f"Expected one stage dump; found {len(paths)}")
    return load_stage_inputs(paths[0])


def _resolve_manifest_path(manifest_path, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    return path.resolve()


def _variant_paths(manifest_path, spec, variant, index, queue_start=300,
                   legacy_midway=False):
    """Resolve one fit and stage-input directory from a portable manifest."""
    fit_value = variant.get("fit_dir")
    stage_value = variant.get("stage_inputs")
    if fit_value is not None or stage_value is not None:
        if fit_value is None or stage_value is None:
            raise ValueError(
                f"Variant {variant['name']!r} must define both fit_dir and "
                "stage_inputs."
            )
        return (
            _resolve_manifest_path(manifest_path, fit_value),
            _resolve_manifest_path(manifest_path, stage_value),
        )
    if not (legacy_midway or bool(spec.get("legacy_midway", False))):
        raise ValueError(
            f"Variant {variant['name']!r} has no fit_dir or stage_inputs. "
            "Create a portable manifest with run_matrix.py, or explicitly "
            "select the legacy Midway layout."
        )
    dataset = str(spec["dataset"])
    name = str(variant["name"])
    queue_number = int(variant.get("queue_number", queue_start + index))
    fit_dir = Path("/scratch/midway3/tfairnington/accel_stacking") / dataset / name
    result_dir = (
        Path("/scratch/midway3/tfairnington/accel_gpu_results")
        / f"{queue_number}_stacking_{name}"
    )
    return fit_dir, result_dir / "stage_inputs"

def _write_loglik_batched(path, samples, stage, batch_size=25):
    """Evaluate draws in bounded-memory batches and stream them into one NPZ."""
    from models.stacking import pointwise_loglik

    n_draw = next(iter(samples.values())).shape[0]
    # Stage dumps may carry zero-weight padding cadences (a likelihood mask);
    # drop them so every model contributes the same observed points to LOO.
    mask = stage.model_kwargs.get("likelihood_mask")
    valid = None if mask is None else np.asarray(mask, dtype=bool)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4, allowZip64=True) as archive:
        for start in range(0, n_draw, batch_size):
            stop = min(start + batch_size, n_draw)
            batch = {key: value[start:stop] for key, value in samples.items()}
            values = pointwise_loglik(batch, stage).astype(np.float32)
            if valid is not None:
                values = values[..., valid]
            payload = io.BytesIO()
            np.lib.format.write_array(payload, values, allow_pickle=False)
            archive.writestr(f"loglik_{start:06d}_{stop:06d}.npy", payload.getvalue())

def _score_loglik_cache(path):
    """Run PSIS a channel at a time without materializing the full cache."""
    from models.stacking import psis_loo

    with np.load(path) as cached:
        keys = sorted(cached.files)
        if keys == ["loglik"]:
            values = np.asarray(cached["loglik"], dtype=np.float32)
        else:
            shape = (sum(cached[key].shape[0] for key in keys),) + cached[keys[0]].shape[1:]
            values = np.empty(shape, dtype=np.float32)
            start = 0
            for key in keys:
                batch = cached[key]
                values[start:start + batch.shape[0]] = batch
                start += batch.shape[0]
    elpd = np.empty(values.shape[1:], dtype=np.float64)
    khat = np.empty(values.shape[1:], dtype=np.float64)
    for channel in range(values.shape[1]):
            elpd_channel, _, khat_channel = psis_loo(
                np.asarray(values[:, channel:channel + 1, :], dtype=np.float64))
            elpd[channel] = elpd_channel[0]
            khat[channel] = khat_channel[0]
    return elpd, khat

def _model_label(name):
    return _MODEL_LABELS.get(name, name.replace("_", " "))


def _wavelength_edges(wavelength):
    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("wavelength must be a non-empty one-dimensional array")
    if wavelength.size == 1:
        half_width = _infer_half_widths(wavelength)[0]
        return np.array([wavelength[0] - half_width, wavelength[0] + half_width])
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("wavelength must be strictly increasing")
    midpoints = 0.5 * (wavelength[:-1] + wavelength[1:])
    return np.concatenate((
        [wavelength[0] - (midpoints[0] - wavelength[0])],
        midpoints,
        [wavelength[-1] + (wavelength[-1] - midpoints[-1])],
    ))


def _validate_plot_arrays(wavelength, names, model_medians, weights):
    wavelength = np.asarray(wavelength, dtype=float)
    model_medians = np.asarray(model_medians, dtype=float)
    weights = np.asarray(weights, dtype=float)
    expected = (len(names), wavelength.size)
    if model_medians.shape != expected:
        raise ValueError(f"model medians have shape {model_medians.shape}; expected {expected}")
    if weights.shape != expected:
        raise ValueError(f"stacking weights have shape {weights.shape}; expected {expected}")
    if not np.all(np.isfinite(model_medians)) or not np.all(np.isfinite(weights)):
        raise ValueError("plot inputs must be finite")
    if np.any(weights < -1e-10):
        raise ValueError("stacking weights must be non-negative")
    if not np.allclose(np.sum(weights, axis=0), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("stacking weights must sum to one in every channel")


def _add_panel_tag(ax, tag):
    ax.text(
        0.02,
        0.95,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=15,
        zorder=20,
    )


def plot_stacking_publication(path, wavelength, names, model_medians,
                              aligned_summary, weights, model_errors=None):
    """Write the publication stacking figure.

    Panels: (a) offset-aligned stacked spectrum with candidate medians,
    (b) per-channel depth precision of each candidate and of the stack
    (only when ``model_errors`` is given), (c) stacking weights.
    """
    apply_publication_style()
    wavelength = np.asarray(wavelength, dtype=float)
    model_medians = np.asarray(model_medians, dtype=float)
    weights = np.asarray(weights, dtype=float)
    _validate_plot_arrays(wavelength, names, model_medians, weights)
    if model_errors is not None:
        model_errors = np.asarray(model_errors, dtype=float)
        if model_errors.shape != model_medians.shape:
            raise ValueError(
                f"model errors have shape {model_errors.shape}; "
                f"expected {model_medians.shape}"
            )
    colors = model_palette_colors(len(names))
    accent = accent_color_for_label(str(path))
    wavelength_err = _infer_half_widths(wavelength)
    wavelength_edges = _wavelength_edges(wavelength)

    n_panels = 3 if model_errors is not None else 2
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(8.8, 9.0 if n_panels == 3 else 6.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1.7, 1.4] if n_panels == 3 else [3, 1]},
    )
    for name, median, color in zip(names, model_medians, colors):
        axes[0].plot(
            wavelength,
            median * 1e6,
            color=color,
            lw=1.1,
            alpha=0.32,
            label=_model_label(name),
            zorder=1,
        )
    stacked_median = np.asarray(aligned_summary["median"], dtype=float) * 1e6
    stacked_error = np.vstack((
        np.asarray(aligned_summary["depth_err_lo"], dtype=float) * 1e6,
        np.asarray(aligned_summary["depth_err_hi"], dtype=float) * 1e6,
    ))
    axes[0].plot(
        wavelength,
        stacked_median,
        color=accent,
        lw=1.8,
        zorder=3,
    )
    stacked_artist = axes[0].errorbar(
        wavelength,
        stacked_median,
        xerr=wavelength_err,
        yerr=stacked_error,
        color=accent,
        ecolor=accent,
        label="Offset-aligned stack",
        zorder=4,
        **DATA_MARKER_STYLE,
    )
    axes[0].set_ylabel("Transit Depth [ppm]")
    model_legend_artists = [
        Line2D([], [], color=color, lw=1.8, label=_model_label(name))
        for name, color in zip(names, colors)
    ]
    axes[0].legend(
        handles=[*model_legend_artists, stacked_artist],
        loc="upper right",
        ncol=2 if len(names) >= 3 else 1,
        frameon=False,
        fontsize=10,
    )

    if model_errors is not None:
        precision_ax = axes[1]
        stacked_precision = 0.5 * np.sum(stacked_error, axis=0)
        # The stack is drawn as a wide translucent halo behind the candidate
        # points so both remain legible where they overlap.
        precision_ax.plot(
            wavelength,
            stacked_precision,
            color=accent,
            lw=7.0,
            alpha=0.28,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=0,
            label="Offset-aligned stack",
        )
        for name, error, color in zip(names, model_errors, colors):
            precision_ax.plot(
                wavelength,
                error * 1e6,
                color=color,
                lw=0.7,
                alpha=0.55,
                zorder=2,
            )
            precision_ax.plot(
                wavelength,
                error * 1e6,
                color=color,
                ls="none",
                marker="o",
                ms=3.0,
                mec="none",
                alpha=0.95,
                zorder=3,
            )
        precision_ax.set_ylabel("Precision [ppm]")
        precision_ax.set_ylim(bottom=0.0)

    weight_ax = axes[-1]
    stepped_weights = np.column_stack((weights, weights[:, -1]))
    weight_ax.stackplot(
        wavelength_edges,
        stepped_weights,
        colors=colors,
        alpha=0.88,
        edgecolor="none",
        step="post",
    )
    weight_ax.set_ylabel("Model weight")
    weight_ax.set_xlabel("Wavelength [$\\mu$m]")
    for ax, tag in zip(axes, ("a", "b", "c")):
        _add_panel_tag(ax, tag)
        style_axis(ax)
    weight_ax.set_xlim(wavelength_edges[0], wavelength_edges[-1])
    weight_ax.set_ylim(0.0, 1.0)
    weight_ax.set_yticks([0.0, 0.5, 1.0])
    fig.align_ylabels(axes)
    save_figure(fig, path)
    return fig


def _smooth_weights(weights, window):
    """Running mean of each model's weights over ``window`` channels, renormalised."""
    weights = np.asarray(weights, dtype=float)
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    half = window // 2
    padded = np.pad(weights, ((0, 0), (half, half)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.vstack([np.convolve(row, kernel, mode="valid") for row in padded])
    return smoothed / np.sum(smoothed, axis=0, keepdims=True)


def _khat_max_by_channel(khat):
    khat = np.asarray(khat, dtype=float)
    if khat.ndim < 2:
        raise ValueError("khat must have model and channel axes")
    reduction_axes = tuple(axis for axis in range(khat.ndim) if axis != 1)
    return np.nanmax(khat, axis=reduction_axes)


def plot_stacking_diagnostics(path, wavelength, aligned_summary,
                              absolute_summary, khat):
    """Write the report-only stack comparison and reliability diagnostics."""
    apply_publication_style()
    wavelength = np.asarray(wavelength, dtype=float)
    accent = accent_color_for_label(str(path))
    khat_max = _khat_max_by_channel(khat)
    if khat_max.shape != wavelength.shape:
        raise ValueError("khat channel axis does not match wavelength")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.8, 8.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )
    for summary, color, label, zorder in (
        (absolute_summary, ZERO_LINE_COLOR, "Absolute stack", 1),
        (aligned_summary, accent, "Offset-aligned stack", 2),
    ):
        median = np.asarray(summary["median"], dtype=float) * 1e6
        low = median - np.asarray(summary["depth_err_lo"], dtype=float) * 1e6
        high = median + np.asarray(summary["depth_err_hi"], dtype=float) * 1e6
        axes[0].fill_between(wavelength, low, high, color=color, alpha=0.16,
                             linewidth=0, zorder=zorder)
        axes[0].plot(wavelength, median, color=color, lw=1.8, label=label,
                     zorder=zorder + 2)
    axes[0].set_ylabel("Transit Depth [ppm]")
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(
        wavelength,
        khat_max,
        color=accent,
        lw=1.6,
        marker="o",
        ms=3.2,
        mfc="white",
        mec=accent,
        mew=1.2,
        label="Maximum",
    )
    axes[1].axhline(
        0.7,
        color=ZERO_LINE_COLOR,
        ls="--",
        lw=1.5,
        label="0.7 threshold",
    )
    axes[1].set_ylabel(r"Pareto $\hat{k}$")
    axes[1].legend(loc="best", frameon=False)

    axes[2].plot(
        wavelength,
        np.asarray(aligned_summary["disagreement"], dtype=float),
        color=accent,
        lw=1.8,
        label="Offset-aligned",
    )
    axes[2].plot(
        wavelength,
        np.asarray(absolute_summary["disagreement"], dtype=float),
        color=ZERO_LINE_COLOR,
        lw=1.6,
        label="Absolute",
    )
    axes[2].axhline(1.0, color=ZERO_LINE_COLOR, ls="--", lw=1.2, alpha=0.7)
    axes[2].set_ylabel("Disagreement ratio")
    axes[2].set_xlabel("Wavelength [$\\mu$m]")
    axes[2].legend(loc="best", frameon=False)
    for ax in axes:
        style_axis(ax)
    edges = _wavelength_edges(wavelength)
    axes[-1].set_xlim(edges[0], edges[-1])
    fig.align_ylabels(axes)
    save_figure(fig, path)
    return fig


def _summary_from_mixture(mixture, disagreement):
    q16, median, q84 = np.percentile(
        np.asarray(mixture, dtype=float), [16, 50, 84], axis=0
    )
    return {
        "median": median,
        "depth_err_lo": median - q16,
        "depth_err_hi": q84 - median,
        "disagreement": np.asarray(disagreement, dtype=float),
    }


def _read_stacked_csv(path, names):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    def column(name):
        try:
            return np.asarray([float(row[name]) for row in rows], dtype=float)
        except KeyError as error:
            raise ValueError(f"Missing {name!r} in {path}") from error

    aligned = {
        "median": column("depth"),
        "depth_err_lo": column("depth_err_lo"),
        "depth_err_hi": column("depth_err_hi"),
        "disagreement": column("disagreement"),
    }
    absolute = {
        "median": column("absolute_depth"),
        "depth_err_lo": column("absolute_depth_err_lo"),
        "depth_err_hi": column("absolute_depth_err_hi"),
        "disagreement": column("absolute_disagreement"),
    }
    model_medians = np.vstack([column(f"{name}_depth") for name in names])
    model_errors = np.vstack([
        0.5 * (column(f"{name}_depth_err_lo") + column(f"{name}_depth_err_hi"))
        for name in names
    ])
    return column("wavelength"), model_medians, aligned, absolute, model_errors


def _saved_prefix(arrays_path):
    arrays_path = Path(arrays_path)
    if not arrays_path.name.endswith(_ARRAYS_SUFFIX):
        raise ValueError(f"Saved arrays filename must end with {_ARRAYS_SUFFIX!r}")
    return arrays_path.with_name(arrays_path.name[:-len(_ARRAYS_SUFFIX)])


def render_stacking_figures_from_arrays(arrays_path):
    """Render both figures without evaluating a model or changing data products."""
    arrays_path = Path(arrays_path)
    prefix = _saved_prefix(arrays_path)
    diagnostic_json = prefix.with_name(prefix.name + "_diagnostics.json")
    spectrum_csv = prefix.with_name(prefix.name + "_stacked.csv")

    with np.load(arrays_path, allow_pickle=False) as saved:
        wavelength = np.asarray(saved["wavelength"], dtype=float)
        weights = np.asarray(saved["stacking_weights"], dtype=float)
        khat = np.asarray(saved["khat"], dtype=float)
        if diagnostic_json.exists():
            names = list(json.loads(diagnostic_json.read_text())["models"])
        elif "model_names" in saved:
            names = [str(name) for name in saved["model_names"]]
        else:
            names = [f"Model {index + 1}" for index in range(weights.shape[0])]

        if spectrum_csv.exists():
            (csv_wavelength, model_medians, aligned, absolute,
             model_errors) = _read_stacked_csv(spectrum_csv, names)
            if not np.allclose(wavelength, csv_wavelength, rtol=0.0, atol=1e-12):
                raise ValueError("Saved arrays and stacked CSV wavelength grids differ")
        else:
            required = {
                "model_depth_median",
                "aligned_mixture",
                "absolute_mixture",
                "aligned_disagreement",
                "absolute_disagreement",
            }
            missing = sorted(required.difference(saved.files))
            if missing:
                raise FileNotFoundError(
                    f"{spectrum_csv} is absent and {arrays_path} lacks {missing}"
                )
            model_medians = np.asarray(saved["model_depth_median"], dtype=float)
            model_errors = None
            aligned = _summary_from_mixture(
                saved["aligned_mixture"], saved["aligned_disagreement"]
            )
            absolute = _summary_from_mixture(
                saved["absolute_mixture"], saved["absolute_disagreement"]
            )

    publication_path = prefix.with_name(prefix.name + "_stacking.png")
    diagnostics_path = prefix.with_name(prefix.name + "_stacking_diagnostics.png")
    plot_stacking_publication(
        publication_path, wavelength, names, model_medians, aligned, weights,
        model_errors=model_errors,
    )
    plot_stacking_diagnostics(
        diagnostics_path, wavelength, aligned, absolute, khat
    )
    return publication_path, diagnostics_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_spec", nargs="?")
    parser.add_argument(
        "--render-saved",
        action="append",
        type=Path,
        metavar="ARRAYS_NPZ",
        help="render figures from an existing *_stacking_arrays.npz product",
    )
    parser.add_argument("--output", default="stacking_results")
    parser.add_argument("--queue-start", type=int, default=300)
    parser.add_argument(
        "--legacy-midway",
        action="store_true",
        help="Use the former site-specific scratch layout for old manifests.",
    )
    parser.add_argument("--n-out", type=int, default=20000)
    parser.add_argument(
        "--stage",
        choices=("low_resolution", "high_resolution"),
        default="high_resolution",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Output filename prefix (defaults to the manifest dataset).",
    )
    parser.add_argument(
        "--weights",
        choices=("stacking", "pseudo_bma_plus"),
        default="stacking",
        help="Which per-channel model weights to combine the posteriors with.",
    )
    parser.add_argument(
        "--smooth-weights",
        type=int,
        default=0,
        metavar="N",
        help="Running-mean the chosen weights over N neighbouring channels "
             "(odd N; 0 disables) and renormalise before stacking.",
    )
    args = parser.parse_args()
    if args.render_saved:
        if args.matrix_spec is not None:
            parser.error("matrix_spec cannot be combined with --render-saved")
        for arrays_path in args.render_saved:
            publication_path, diagnostics_path = render_stacking_figures_from_arrays(
                arrays_path
            )
            print(f"Wrote {publication_path}")
            print(f"Wrote {diagnostics_path}")
        return
    if args.matrix_spec is None:
        parser.error("matrix_spec is required unless --render-saved is used")

    from models.stacking import (
        aligned_stack_posteriors,
        pseudo_bma_plus_weights,
        stacking_weights,
        write_stacked_spectrum,
    )

    manifest_path = Path(args.matrix_spec).resolve()
    with manifest_path.open() as stream:
        spec = yaml.safe_load(stream)
    variants = list(spec["variants"])
    names = [variant["name"] for variant in variants]
    label = str(args.label or spec["dataset"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    depths = []
    elpds = []
    khats = []
    wave = None
    for index, (name, variant) in enumerate(zip(names, variants)):
        fit, stage_inputs = _variant_paths(
            manifest_path,
            spec,
            variant,
            index,
            queue_start=args.queue_start,
            legacy_midway=args.legacy_midway,
        )
        stage = _stage(stage_inputs, args.stage)
        samples = _load_chunks(fit, stage)
        llpath = output / f"{label}_{name}_pointwise_loglik.npz"
        if not llpath.exists():
            _write_loglik_batched(llpath, samples, stage)
        elpd_i, khat = _score_loglik_cache(llpath)
        elpds.append(elpd_i)
        khats.append(khat)
        depths.append(np.asarray(samples["rors"])[..., 0] ** 2)
        current = np.asarray(stage.meta["wavelength"])
        if wave is None:
            wave = current
        elif not np.allclose(wave, current):
            raise ValueError("Variant wavelength grids differ")
    elpds = np.asarray(elpds)
    khats = np.asarray(khats)
    weights = np.column_stack([
        stacking_weights(elpds[:, channel, :])
        for channel in range(elpds.shape[1])
    ])
    pbma = np.column_stack([
        pseudo_bma_plus_weights(elpds[:, channel, :], rng=800 + channel)
        for channel in range(elpds.shape[1])
    ])
    combine = pbma if args.weights == "pseudo_bma_plus" else weights
    if args.smooth_weights and args.smooth_weights > 1:
        combine = _smooth_weights(combine, int(args.smooth_weights))
    stacked = aligned_stack_posteriors(depths, combine, args.n_out, rng=20260902)
    write_stacked_spectrum(
        output / f"{label}_stacked.csv",
        wave,
        stacked["aligned_summary"],
        names,
        depths,
        combine,
        absolute_summary=stacked["absolute_summary"],
        offsets=stacked["offsets"],
        offset_uncertainties=stacked["offset_uncertainties"],
    )
    arrays_path = output / f"{label}{_ARRAYS_SUFFIX}"
    np.savez_compressed(
        arrays_path,
        wavelength=wave,
        stacking_weights=combine,
        raw_stacking_weights=weights,
        pseudo_bma_plus_weights=pbma,
        khat=khats,
        aligned_mixture=stacked["aligned_mixture"].astype(np.float32),
        absolute_mixture=stacked["absolute_mixture"].astype(np.float32),
        offsets=stacked["offsets"],
        offset_uncertainties=stacked["offset_uncertainties"],
        average_offset=stacked["average_offset"],
    )
    diagnostics = {
        "models": names,
        "stacking_weights": combine.tolist(),
        "raw_stacking_weights": weights.tolist(),
        "weight_method": args.weights,
        "weight_smoothing_channels": int(args.smooth_weights),
        "pseudo_bma_plus_weights": pbma.tolist(),
        "khat_max_by_model_channel": np.max(khats, axis=2).tolist(),
        "achromatic_offsets": stacked["offsets"].tolist(),
        "achromatic_offset_uncertainties": stacked[
            "offset_uncertainties"
        ].tolist(),
        "weighted_average_offset": stacked["average_offset"],
        "laplace_bma": (
            "unavailable unless each run emits white-light MAP/Hessian/log-joint"
        ),
    }
    (output / f"{label}_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )
    render_stacking_figures_from_arrays(arrays_path)


if __name__ == "__main__":
    main()
