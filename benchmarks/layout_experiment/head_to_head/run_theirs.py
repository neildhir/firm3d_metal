"""Trace the shared inputs with sumaddury/firm3d metal-port (commit e2e72538).

Run from the metal-port worktree (its kernel source is found relative to the cwd),
with that build's firm3dpp first on PYTHONPATH:
    python run_theirs.py config.yaml
Reads <data_dir>/inputs.npz written by run_ours.py; writes <data_dir>/theirs.npz.
"""

import sys
import time
from pathlib import Path

import firm3dpp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from h2h_config import H2HConfig, load_config  # noqa: E402


def trace_final(
    inp: dict[str, np.ndarray], table: np.ndarray, n: int, tmax: float, tol: float
) -> np.ndarray:
    """One call of their tracing binding on the first n markers.

    Returns (n, 5) rows [t, s, theta, zeta, vpar].
    """
    out = firm3dpp.metal_boozer_vacuum_tracing(
        quad_pts=table,
        x1_range=inp["srange"],
        x2_range=inp["trange"],
        x3_range=inp["zrange"],
        loc=np.ascontiguousarray(inp["points"][:n]),
        vpar=np.ascontiguousarray(inp["vpar"][:n]),
        v_total=float(inp["velocity"]),
        m=float(inp["mass"]),
        q=float(inp["charge"]),
        psi0=float(inp["psi0"]),
        tmax=tmax,
        tol=tol,
        n_points=n,
    )
    return np.reshape(out, (n, 5))


def main(cfg: H2HConfig) -> None:
    """Time their kernel, and its fixed per-call cost (a zero-length trace)."""
    inp = dict(np.load(cfg.data_dir / "inputs.npz"))
    table = np.ascontiguousarray(
        inp["table_feb"], dtype=np.float64
    )  # their binding takes float64
    n = cfg.nparticles
    trace_final(inp, table, cfg.warmup_nparticles, cfg.warmup_tmax_s, cfg.tols[0])

    out: dict[str, np.ndarray] = {}
    walls0 = []
    for _ in range(cfg.repeats):
        t0 = time.perf_counter()
        trace_final(inp, table, n, 0.0, cfg.tols[0])
        walls0.append(time.perf_counter() - t0)
    out["wall_tmax0"] = np.array(walls0)
    print(
        f"theirs tmax=0 (per-call overhead): walls {np.round(walls0, 3)} s", flush=True
    )

    for tol in cfg.tols:
        walls = []
        for _ in range(cfg.repeats):
            t0 = time.perf_counter()
            final = trace_final(inp, table, n, cfg.tmax_s, tol)
            walls.append(time.perf_counter() - t0)
        out[f"wall_{tol:g}"] = np.array(walls)
        out[f"final_{tol:g}"] = final
        print(
            f"theirs tol={tol:g}: walls {np.round(walls, 3)} s, "
            f"lost(s>=1) {int(np.sum(final[:, 1] >= 1.0))}",
            flush=True,
        )
    np.savez(cfg.data_dir / "theirs.npz", **out)


if __name__ == "__main__":
    main(load_config(Path(sys.argv[1])))
