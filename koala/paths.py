"""User cache locations shared by the pipeline and stellar-prior builder."""
import os
from pathlib import Path

DEFAULT_LD_DATA_PATH = str(
    Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'koala' / 'exotic_ld_data'
)
