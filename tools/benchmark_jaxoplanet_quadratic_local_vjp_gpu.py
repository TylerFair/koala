#!/usr/bin/env python3
"""Run the standard quadratic benchmark against the local-VJP prototype."""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.jaxoplanet.experimental_quadratic_local_vjp import light_curve
from tools import benchmark_jaxoplanet_quadratic_gpu as benchmark


benchmark.specialized_light_curve = lambda: light_curve


if __name__ == "__main__":
    raise SystemExit(benchmark.main())
