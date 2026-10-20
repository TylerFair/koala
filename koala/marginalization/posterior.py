"""Verified posterior chunks, without retaining an entire model grid in RAM."""
from pathlib import Path
import json
import shutil
import numpy as np

from .scoring import _file_digest, _save_npz


class PosteriorStore:
    def __init__(self, directory, fingerprint, *, lanes, chains, draws):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'progress.json'
        self.fingerprint = fingerprint
        self.lanes, self.num_chains, self.draws = lanes, chains, draws
        if any(int(v) != v or v < 1 for v in (lanes, chains, draws)):
            raise ValueError('Posterior store dimensions must be positive integers.')
        self.sampled_sites = None
        self.entries = {}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved.get('fingerprint') == fingerprint and saved.get('shape') == [draws*chains, lanes]:
                self.entries = {entry['begin']:entry for entry in saved['entries']}
                self.sampled_sites = tuple(saved['sampled_sites'])

    def accepted(self, begin, end):
        entry = self.entries.get(begin)
        if entry is None or entry['end'] != end or not entry['quality']['passed']:
            return None
        path = self.directory/entry['file']
        if not path.exists() or _file_digest(path) != entry['digest']:
            return None
        return entry

    def save(self, begin, end, samples, quality):
        if not 0 <= begin < end <= self.lanes:
            raise ValueError('Posterior chunk range is outside the declared lanes.')
        if not quality['passed']:
            raise ValueError('Unconverged posteriors cannot enter the accepted chunk store.')
        arrays = {name:np.asarray(value) for name,value in samples.items()}
        if any(value.shape[:2] != (self.draws*self.num_chains, end-begin) for value in arrays.values()):
            raise ValueError('Posterior chunk dimensions do not match its declared lanes and chains.')
        if not arrays or any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError('Accepted posterior chunks must have finite samples.')
        sites = tuple(getattr(samples, 'sampled_sites', None) or samples)
        if self.sampled_sites is not None and sites != self.sampled_sites:
            raise ValueError('Sampled posterior sites changed across chunks.')
        if self.entries and set(arrays) != set(next(iter(self.entries.values()))['shapes']):
            raise ValueError('Posterior sites changed across chunks.')
        if any(key != begin and max(begin, key) < min(end, entry['end'])
               for key, entry in self.entries.items()):
            raise ValueError('Posterior chunk ranges overlap.')
        self.sampled_sites = sites
        path = self.directory/f'posterior_{begin}_{end}.npz'
        _save_npz(path, **arrays)
        self.entries[begin] = dict(begin=begin, end=end, file=path.name,
            digest=_file_digest(path), quality=quality, shapes={k:list(v.shape) for k,v in arrays.items()})
        metadata = dict(schema_version=1, fingerprint=self.fingerprint,
            shape=[self.draws*self.num_chains, self.lanes], sampled_sites=list(self.sampled_sites),
            entries=[self.entries[key] for key in sorted(self.entries)])
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(self.path)

    def complete(self, *, verify=True):
        entries = [self.entries[key] for key in sorted(self.entries)]
        return bool(entries and entries[0]['begin'] == 0 and entries[-1]['end'] == self.lanes
            and all(0 <= e['begin'] < e['end'] <= self.lanes for e in entries)
            and all(a['end'] == b['begin'] for a,b in zip(entries, entries[1:]))
            and (not verify or all(self.accepted(e['begin'], e['end']) is not None for e in entries)))

    def export(self, begin, end, directory, prefix):
        """Write only intersecting chunks; copy complete archives without loading."""
        if not 0 <= begin < end <= self.lanes:
            raise ValueError('Posterior export range is outside the declared lanes.')
        if not self.complete(verify=False):
            raise ValueError('Cannot export an incomplete or changed posterior store.')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for entry in (self.entries[key] for key in sorted(self.entries)):
            lo, hi = max(begin, entry['begin']), min(end, entry['end'])
            if lo >= hi:
                continue
            if self.accepted(entry['begin'], entry['end']) is None:
                raise ValueError('Posterior chunk changed before export.')
            source = self.directory/entry['file']
            target = directory/f'{prefix}_chunk_{lo-begin}_{hi-begin}.npz'
            if lo == entry['begin'] and hi == entry['end']:
                temporary = target.with_suffix('.npz.tmp')
                shutil.copyfile(source, temporary)
                temporary.replace(target)
            else:
                with np.load(source, allow_pickle=False) as saved:
                    _save_npz(target, **{k:v[:, lo-entry['begin']:hi-entry['begin']]
                                        for k,v in saved.items()})
            paths.append(target)
        return paths
