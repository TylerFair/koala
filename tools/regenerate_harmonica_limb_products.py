#!/usr/bin/env python3
"""Regenerate canonical Harmonica tables and PNGs from a finished run's state."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import pickle
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from koala.output_paths import OutputPaths


@dataclass(frozen=True)
class ChunkSpec:
    path: Path
    prefix: str
    start: int
    end: int


def discover_chunk_specs(directory, grid=None):
    families = {}
    for path in OutputPaths(directory).chunks_dir.glob('*_chunk_*_*.*'):
        match = re.fullmatch(r'(.+)_chunk_(\d+)_(\d+)\.(?:pkl|npz)', path.name)
        if match and (grid is None or re.search(r'_'+re.escape(grid)+r'(?:_|$)', match[1])):
            families.setdefault(match[1], []).append(ChunkSpec(path, match[1], int(match[2]), int(match[3])))
    if len(families) != 1:
        raise ValueError(f'Expected one checkpoint family; found {len(families)}. Select a single stage with --grid.')
    specs = sorted(next(iter(families.values())), key=lambda spec: spec.start)
    if specs[0].start != 0 or any(a.end != b.start for a, b in zip(specs, specs[1:])):
        raise ValueError('Checkpoint coverage has a gap or overlap.')
    return specs


def load_limb_samples(specs):
    chunks = []
    for spec in specs:
        if spec.path.suffix == '.npz':
            with np.load(spec.path, allow_pickle=False) as source:
                payload = dict(source)
        else:
            with spec.path.open('rb') as source:
                payload = pickle.load(source)
            payload = payload.get('samples', payload)
        names = [name for name in ('rors', 'a0') if name in payload]
        names += sorted((name for name in payload if re.fullmatch(r'a[1-9][0-9]*', name)),
                        key=lambda name: int(name[1:]))
        chunks.append({name: np.asarray(payload[name]) for name in names})
    if any(set(chunk) != set(chunks[0]) for chunk in chunks):
        raise ValueError('Checkpoint limb parameters do not match.')
    return {name: np.concatenate([chunk[name] for chunk in chunks], axis=1) for name in chunks[0]}


def resolve_wavelength_axis(data, channels, axis='auto'):
    data = data if isinstance(data, dict) else vars(data)
    matches = [name for name in ('lr', 'hr') if (axis == 'auto' or axis == name)
               and np.asarray(data.get('wavelengths_'+name, [])).size == channels]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely infer the wavelength axis from channel count.')
    name = matches[0]
    return name, np.asarray(data['wavelengths_'+name]), np.asarray(data['wavelengths_err_'+name])


def schema_v3_output_paths(directory, prefix):
    """Current schema, with the same filenames as the fitting pipeline."""
    # Native checkpoint names may have a sampler/config suffix after the grid.
    match = re.match(r'^(.*_(?:R[^_]+|pix[^_]+))(?:_.*)?$', prefix)
    prefix = match[1] if match else prefix
    paths, grid = OutputPaths.from_stem(Path(directory)/prefix)
    return dict(csv=Path(paths.table(grid, 'limbs')),
                posterior_samples=Path(paths.table(grid, 'limb_samples', 'npz')),
                limbs=Path(paths.figure(grid, 'limbs')),
                profiles=Path(paths.figure(grid, 'limb_profiles')),
                outline=Path(paths.figure(grid, 'limb_outline')))


def regenerate_limb_products(directory, *, dry_run=False, random_seed=0, force=False, grid=None, axis="auto"):
    directory = Path(directory)
    specs = discover_chunk_specs(directory, grid=grid)
    paths = schema_v3_output_paths(directory, specs[0].prefix)
    if not force and any(path.exists() for path in paths.values()):
        raise FileExistsError('Refusing to overwrite an existing limb product; use --force.')
    samples = load_limb_samples(specs)
    prepared = list((directory/'state').glob('*spectroscopy_data*.pkl'))
    if len(prepared) != 1:
        raise ValueError('Expected one prepared-spectra file in state/.')
    with prepared[0].open('rb') as source:
        data = pickle.load(source)
    axis, wavelength, width = resolve_wavelength_axis(data, specs[-1].end, axis=axis)
    result = dict(output_paths=paths, wavelength_axis=axis, channels=len(wavelength))
    if dry_run:
        return result
    from koala.harmonica_products import save_harmonica_limb_products
    from koala.console import grid_description
    display, grid = OutputPaths.from_stem(paths['csv'].with_name(paths['csv'].stem.removesuffix('_limbs')))
    title = display.prefix + ' | ' + grid_description(grid)
    # This seed controls only the representative outline draws, never a fit.
    previous = np.random.get_state()
    try:
        np.random.seed(random_seed)
        save_harmonica_limb_products(wavelength, width, samples.get('a0', samples['rors']),
            {name: values for name, values in samples.items() if name.startswith('a') and name != 'a0'},
            paths['csv'], paths['limbs'], paths['profiles'], title,
            posterior_strings_path=paths['outline'], posterior_samples_path=paths['posterior_samples'])
    finally:
        np.random.set_state(previous)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--grid', help='Select a wavelength grid, e.g. R100 or pix8')
    parser.add_argument('--axis', choices=['auto', 'lr', 'hr'], default='auto')
    args = parser.parse_args()
    result = regenerate_limb_products(args.directory, dry_run=args.dry_run, force=args.force,
                                      grid=args.grid, axis=args.axis)
    for name, path in result['output_paths'].items():
        print(f'{name}: {path}')


if __name__ == '__main__':
    main()
