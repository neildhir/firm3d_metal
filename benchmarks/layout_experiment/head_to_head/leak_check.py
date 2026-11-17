"""Does repeated calling of their tracing binding grow the process footprint?

Calls metal_boozer_vacuum_tracing with tmax = 0 (no stepping: only the host-side
table conversion, buffer creation and an empty dispatch) and prints the process's
physical footprint after each call. Run like run_theirs.py.
"""

import ctypes
import os
import sys
from pathlib import Path

import firm3dpp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from h2h_config import load_config  # noqa: E402

N_CALLS = 6  # calls to make, dimensionless
N_MARKERS = 8  # markers per call, dimensionless
RUSAGE_INFO_V2 = 2
FOOTPRINT_OFFSET = (
    72  # byte offset of ri_phys_footprint in struct rusage_info_v2 (macOS <libproc.h>)
)


def phys_footprint_mb() -> float:
    """This process's physical footprint [MB], as Activity Monitor reports it."""
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    buf = ctypes.create_string_buffer(512)
    if libc.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V2, buf) != 0:
        raise OSError("proc_pid_rusage failed")
    return (
        int.from_bytes(buf.raw[FOOTPRINT_OFFSET : FOOTPRINT_OFFSET + 8], "little")
        / 2**20
    )


def main() -> None:
    """Print the footprint before and after each zero-length call."""
    cfg = load_config(Path(sys.argv[1]))
    inp = dict(np.load(cfg.data_dir / "inputs.npz"))
    table = np.ascontiguousarray(inp["table_feb"], dtype=np.float64)
    print(
        f"table as float32: {table.size * 4 / 2**20:.0f} MB; "
        f"footprint before: {phys_footprint_mb():.0f} MB"
    )
    for i in range(N_CALLS):
        firm3dpp.metal_boozer_vacuum_tracing(
            quad_pts=table,
            x1_range=inp["srange"],
            x2_range=inp["trange"],
            x3_range=inp["zrange"],
            loc=np.ascontiguousarray(inp["points"][:N_MARKERS]),
            vpar=np.ascontiguousarray(inp["vpar"][:N_MARKERS]),
            v_total=float(inp["velocity"]),
            m=float(inp["mass"]),
            q=float(inp["charge"]),
            psi0=float(inp["psi0"]),
            tmax=0.0,
            tol=1e-6,
            n_points=N_MARKERS,
        )
        print(f"after call {i + 1}: footprint {phys_footprint_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
