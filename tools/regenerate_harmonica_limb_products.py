#!/usr/bin/env python3
"""Regenerate schema-v2 Harmonica limb products from MCMC checkpoints.

This utility is deliberately non-destructive: it reads trusted local pickle
checkpoints, writes products with a ``schema_v2`` tag, and refuses to replace
any existing file.  It does not run optimization or MCMC.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Regeneration only needs host-side posterior summaries and plotting.  Avoid
# claiming GPU memory or initializing CUDA when this tool is used on a login
# node.
# This is a standalone post-processing tool, not a model benchmark.  Force the
# CPU even on clusters that export a CUDA-first JAX_PLATFORMS globally.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")


_CHUNK_RE = re.compile(
    r"^(?P<prefix>.+)_chunk_(?P<start>[0-9]+)_(?P<end>[0-9]+)\.pkl$"
)
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ChunkSpec:
    path: Path
    prefix: str
    start: int
    end: int


def _parse_chunk(path: Path) -> ChunkSpec | None:
    match = _CHUNK_RE.fullmatch(path.name)
    if match is None:
        return None
    return ChunkSpec(
        path=path,
        prefix=match.group("prefix"),
        start=int(match.group("start")),
        end=int(match.group("end")),
    )


def discover_chunk_specs(result_dir: Path, checkpoint_prefix: str | None = None):
    """Return a validated, contiguous set of checkpoint chunks."""
    chunk_dir = result_dir / "chunks"
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {chunk_dir}")

    parsed = [
        spec
        for path in chunk_dir.glob("*_chunk_*_*.pkl")
        if (spec := _parse_chunk(path)) is not None
    ]
    prefixes = sorted({spec.prefix for spec in parsed})
    if checkpoint_prefix is None:
        if not prefixes:
            raise FileNotFoundError(f"No chunk checkpoint files found in {chunk_dir}")
        if len(prefixes) != 1:
            raise ValueError(
                "Multiple checkpoint families were found; select one with "
                f"--checkpoint-prefix. Available prefixes: {prefixes}"
            )
        checkpoint_prefix = prefixes[0]
    elif checkpoint_prefix not in prefixes:
        raise FileNotFoundError(
            f"No chunks with prefix {checkpoint_prefix!r} were found in {chunk_dir}. "
            f"Available prefixes: {prefixes}"
        )

    specs = sorted(
        (spec for spec in parsed if spec.prefix == checkpoint_prefix),
        key=lambda spec: (spec.start, spec.end),
    )
    expected_start = 0
    seen_ranges = set()
    for spec in specs:
        if spec.end <= spec.start:
            raise ValueError(f"Invalid checkpoint range {spec.start}:{spec.end}: {spec.path}")
        range_key = (spec.start, spec.end)
        if range_key in seen_ranges:
            raise ValueError(f"Duplicate checkpoint range {spec.start}:{spec.end}")
        seen_ranges.add(range_key)
        if spec.start != expected_start:
            relation = "overlap" if spec.start < expected_start else "gap"
            raise ValueError(
                f"Checkpoint {relation}: expected a chunk starting at "
                f"{expected_start}, found {spec.start}:{spec.end} ({spec.path.name})"
            )
        expected_start = spec.end

    return specs


def _load_trusted_pickle(path: Path):
    # Pickle can execute code while loading.  This tool is intended only for
    # checkpoints and data products created by this pipeline.
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_limb_samples(chunk_specs: Sequence[ChunkSpec]):
    """Load and concatenate only the posterior fields used by limb products."""
    if not chunk_specs:
        raise ValueError("At least one checkpoint chunk is required.")

    from fit_jwst import HARMONICA_ODD_HARMONICS, _concatenate_chunk_samples

    relevant_chunks = []
    expected_keys = None
    expected_draws = None
    expected_trailing_shapes = {}
    for spec in chunk_specs:
        payload = _load_trusted_pickle(spec.path)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Checkpoint must contain a mapping: {spec.path}")
        if "rors" not in payload:
            raise KeyError(f"Checkpoint does not contain 'rors': {spec.path}")

        harmonic_keys = tuple(name for name in HARMONICA_ODD_HARMONICS if name in payload)
        if not harmonic_keys:
            raise KeyError(
                f"Checkpoint contains no supported odd Harmonica coefficient "
                f"({', '.join(HARMONICA_ODD_HARMONICS)}): {spec.path}"
            )
        # ``half_area`` checkpoints retain the total-area radius in ``rors``
        # and the actual Harmonica mean polar radius in deterministic ``a0``.
        # Preserve the optional field so regenerated limb products use the
        # same geometry as the fitted forward model.  Legacy checkpoints have
        # no ``a0`` and remain byte-for-byte compatible at this selection step.
        base_keys = ("rors", "a0") if "a0" in payload else ("rors",)
        selected = {
            name: np.asarray(payload[name])
            for name in (*base_keys, *harmonic_keys)
        }
        selected_keys = tuple(selected)
        if expected_keys is None:
            expected_keys = selected_keys
        elif selected_keys != expected_keys:
            raise ValueError(
                f"Posterior fields differ across chunks: expected {expected_keys}, "
                f"found {selected_keys} in {spec.path}"
            )

        chunk_width = spec.end - spec.start
        for name, values in selected.items():
            if values.ndim < 2:
                raise ValueError(
                    f"{name!r} must have (draw, channel, ...) axes; "
                    f"found shape {values.shape} in {spec.path}"
                )
            if values.shape[1] != chunk_width:
                raise ValueError(
                    f"{name!r} channel width {values.shape[1]} does not match "
                    f"filename range {spec.start}:{spec.end} in {spec.path}"
                )
            if expected_draws is None:
                expected_draws = values.shape[0]
            elif values.shape[0] != expected_draws:
                raise ValueError(
                    f"{name!r} has {values.shape[0]} draws in {spec.path}; "
                    f"expected {expected_draws}"
                )
            trailing_shape = values.shape[2:]
            if name not in expected_trailing_shapes:
                expected_trailing_shapes[name] = trailing_shape
            elif trailing_shape != expected_trailing_shapes[name]:
                raise ValueError(
                    f"{name!r} trailing shape changed from "
                    f"{expected_trailing_shapes[name]} to {trailing_shape} in {spec.path}"
                )
        relevant_chunks.append(selected)

    samples = _concatenate_chunk_samples(relevant_chunks)
    if not np.all(np.isfinite(np.asarray(samples["rors"]))):
        raise ValueError("The concatenated 'rors' posterior contains non-finite values.")
    for name in expected_keys[1:]:
        if not np.all(np.isfinite(np.asarray(samples[name]))):
            raise ValueError(f"The concatenated {name!r} posterior contains non-finite values.")
    return samples


def discover_data_pickle(result_dir: Path, data_pickle: Path | None = None):
    if data_pickle is not None:
        resolved = data_pickle.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Spectroscopy data pickle does not exist: {resolved}")
        return resolved

    candidates = sorted(result_dir.glob("*spectroscopy_data*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No spectroscopy data pickle found directly inside {result_dir}; "
            "provide --data-pickle."
        )
    if len(candidates) != 1:
        raise ValueError(
            "Multiple spectroscopy data pickles were found; provide --data-pickle: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0].resolve()


def _data_field(data, name):
    if isinstance(data, Mapping):
        if name not in data:
            raise AttributeError(name)
        return data[name]
    if not hasattr(data, name):
        raise AttributeError(name)
    return getattr(data, name)


def resolve_wavelength_axis(data, num_channels: int, wavelength_axis: str = "auto"):
    """Select the HR or LR wavelength axis matching the posterior channels."""
    axes = {}
    for short_name in ("hr", "lr"):
        try:
            wavelength = np.asarray(_data_field(data, f"wavelengths_{short_name}"))
            wavelength_err = np.asarray(_data_field(data, f"wavelengths_err_{short_name}"))
        except AttributeError:
            continue
        axes[short_name] = (wavelength, wavelength_err)

    if wavelength_axis != "auto":
        if wavelength_axis not in axes:
            raise ValueError(f"Data pickle does not contain a {wavelength_axis.upper()} axis.")
        candidates = {wavelength_axis: axes[wavelength_axis]}
    else:
        candidates = {
            name: values for name, values in axes.items()
            if np.atleast_1d(values[0]).size == num_channels
        }
        if len(candidates) != 1:
            lengths = {name: np.atleast_1d(values[0]).size for name, values in axes.items()}
            raise ValueError(
                f"Could not uniquely infer a wavelength axis for {num_channels} channels. "
                f"Available lengths: {lengths}; specify --wavelength-axis."
            )

    axis_name, (wavelength, wavelength_err) = next(iter(candidates.items()))
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
    wavelength_err = np.atleast_1d(np.asarray(wavelength_err, dtype=float))
    if wavelength.size != num_channels:
        raise ValueError(
            f"The {axis_name.upper()} wavelength axis has {wavelength.size} entries, "
            f"but the checkpoints contain {num_channels} channels."
        )
    if wavelength_err.size not in (1, num_channels):
        raise ValueError(
            f"The {axis_name.upper()} wavelength-error axis has {wavelength_err.size} "
            f"entries; expected 1 or {num_channels}."
        )
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(wavelength_err)):
        raise ValueError(f"The {axis_name.upper()} wavelength axis contains non-finite values.")
    return axis_name, wavelength, wavelength_err


def schema_v2_output_paths(result_dir: Path, output_stem: str, tag: str = "schema_v2"):
    for label, value in (("output stem", output_stem), ("tag", tag)):
        if not value or _SAFE_NAME_RE.fullmatch(value) is None:
            raise ValueError(
                f"Invalid {label} {value!r}; use only letters, numbers, '.', '_' and '-'."
            )
    return {
        "csv": result_dir / f"{output_stem}_limb_spectra_{tag}.csv",
        "limb_spectrum": result_dir / f"35_{output_stem}_limb_spectra_{tag}.png",
        "transmission_strings": result_dir / f"36_{output_stem}_transmission_strings_{tag}.png",
        "posterior_strings": result_dir / f"37_{output_stem}_transmission_string_posterior_{tag}.png",
        "posterior_samples": result_dir / f"{output_stem}_limb_posterior_samples_{tag}.npz",
    }


def _refuse_existing(paths):
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing schema-v2 product(s): "
            + ", ".join(str(path) for path in existing)
        )


def regenerate_limb_products(
    result_dir: Path,
    checkpoint_prefix: str | None = None,
    data_pickle: Path | None = None,
    wavelength_axis: str = "auto",
    output_stem: str | None = None,
    title_prefix: str | None = None,
    tag: str = "schema_v2",
    planet_index: int = 0,
    random_seed: int = 0,
    dry_run: bool = False,
):
    result_dir = result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

    chunk_specs = discover_chunk_specs(result_dir, checkpoint_prefix)
    checkpoint_prefix = chunk_specs[0].prefix
    samples = load_limb_samples(chunk_specs)
    num_channels = chunk_specs[-1].end
    if np.asarray(samples["rors"]).shape[1] != num_channels:
        raise ValueError("Concatenated posterior width does not match checkpoint coverage.")

    data_pickle = discover_data_pickle(result_dir, data_pickle)
    data = _load_trusted_pickle(data_pickle)
    axis_name, wavelengths, wavelength_err = resolve_wavelength_axis(
        data, num_channels, wavelength_axis
    )

    output_stem = checkpoint_prefix if output_stem is None else output_stem
    output_paths = schema_v2_output_paths(result_dir, output_stem, tag=tag)
    _refuse_existing(output_paths)

    summary = {
        "result_dir": result_dir,
        "checkpoint_prefix": checkpoint_prefix,
        "chunks": len(chunk_specs),
        "draws": int(np.asarray(samples["rors"]).shape[0]),
        "channels": num_channels,
        "harmonics": tuple(name for name in ("a1", "a3", "a5") if name in samples),
        "limb_radius_field": "a0" if "a0" in samples else "rors",
        "data_pickle": data_pickle,
        "wavelength_axis": axis_name,
        "output_paths": output_paths,
    }
    if dry_run:
        return summary

    from fit_jwst import HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION, save_harmonica_limb_products

    if HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION != 2:
        raise RuntimeError(
            "This utility is pinned to limb-product schema v2, but fit_jwst reports "
            f"schema v{HARMONICA_LIMB_PRODUCT_SCHEMA_VERSION}."
        )
    harmonic_samples = {
        name: samples[name] for name in ("a1", "a3", "a5") if name in samples
    }
    title_prefix = checkpoint_prefix if title_prefix is None else title_prefix

    # Stage every file on the destination filesystem.  Hard-linking each final
    # product is atomic and fails if a file appeared after the preflight check.
    with tempfile.TemporaryDirectory(prefix=".harmonica_schema_v2_", dir=result_dir) as tmp:
        tmp_dir = Path(tmp)
        staged = {name: tmp_dir / path.name for name, path in output_paths.items()}
        random_state = np.random.get_state()
        np.random.seed(random_seed)
        try:
            save_harmonica_limb_products(
                wavelengths=wavelengths,
                wavelength_err=wavelength_err,
                rors_samples=samples.get("a0", samples["rors"]),
                harmonic_samples=harmonic_samples,
                csv_path=staged["csv"],
                limb_spectrum_path=staged["limb_spectrum"],
                transmission_strings_path=staged["transmission_strings"],
                posterior_strings_path=staged["posterior_strings"],
                posterior_samples_path=staged["posterior_samples"],
                title_prefix=title_prefix,
                planet_index=planet_index,
            )
        finally:
            np.random.set_state(random_state)

        missing = [path for path in staged.values() if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(
                "Product generation did not create every staged file: "
                + ", ".join(str(path) for path in missing)
            )
        _refuse_existing(output_paths)
        for name, target in output_paths.items():
            os.link(staged[name], target)
            staged[name].unlink()

    return summary


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate Catwoman-comparable Harmonica limb-product schema v2 "
            "from existing chunk checkpoints, without refitting or overwriting files."
        )
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--checkpoint-prefix")
    parser.add_argument("--data-pickle", type=Path)
    parser.add_argument("--wavelength-axis", choices=("auto", "hr", "lr"), default="auto")
    parser.add_argument("--output-stem")
    parser.add_argument("--title-prefix")
    parser.add_argument("--tag", default="schema_v2")
    parser.add_argument("--planet-index", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print destinations without writing products.",
    )
    return parser


def main(argv: Sequence[str] | None = None):
    args = _build_parser().parse_args(argv)
    summary = regenerate_limb_products(
        result_dir=args.result_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        data_pickle=args.data_pickle,
        wavelength_axis=args.wavelength_axis,
        output_stem=args.output_stem,
        title_prefix=args.title_prefix,
        tag=args.tag,
        planet_index=args.planet_index,
        random_seed=args.random_seed,
        dry_run=args.dry_run,
    )

    action = "Validated" if args.dry_run else "Created"
    print(
        f"{action} schema-v2 limb products from {summary['chunks']} chunks, "
        f"{summary['draws']} draws, and {summary['channels']} "
        f"{summary['wavelength_axis'].upper()} channels."
    )
    print(f"Harmonics: {', '.join(summary['harmonics'])}")
    print(f"Data: {summary['data_pickle']}")
    print("Destinations:")
    for path in summary["output_paths"].values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
