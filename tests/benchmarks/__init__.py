"""Tests of the benchmark tooling in benchmarks/ (fork-local).

The benchmark modules import one another as top-level modules (they run as
scripts from benchmarks/), so that directory is put on the import path here.
"""

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
