"""Names and directories shared by fit writers and finished-run readers."""
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    prefix: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'root', Path(self.root))

    @staticmethod
    def _parent(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def stem(self, stage):
        return str(self.root / '_'.join(part for part in (self.prefix, stage) if part))

    def table(self, stage, product, extension='csv'):
        name = '_'.join(str(part) for part in (self.prefix, stage, product) if part)
        return self._parent(self.root / f'{name}.{extension}')

    def figure(self, stage, product=None):
        name = '_'.join(str(part) for part in (stage, product) if part)
        return self._parent(self.root / 'figures' / f'{name}.png')

    def state(self, name):
        return self._parent(self.root / 'state' / name)

    def joint_table(self, product):
        return self._parent(self.root / f'joint_{product}.csv')

    def dataset_table(self, member):
        return self.state(f'{member}_spectrum.csv')

    @property
    def state_dir(self):
        path = self.root / 'state'
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def chunks_dir(self):
        path = Path(self.state_dir) / 'chunks'
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def from_stem(cls, stem):
        """Recover a stage from the canonical prefix accepted by product helpers."""
        path = Path(stem)
        match = re.match(r'^(.*)_(whitelight|R[^_]+|pix[^_]+)$', path.name)
        return (cls(path.parent, match[1]), match[2]) if match else (cls(path.parent, path.name), '')

    @classmethod
    def from_spectrum(cls, filename):
        path = Path(filename)
        stem = path.stem.removesuffix('_spectrum')
        return cls.from_stem(path.with_name(stem))


def state_path(output_dir, name):
    return OutputPaths(output_dir).state(name)
