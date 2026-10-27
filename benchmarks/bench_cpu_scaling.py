"""CPU (MPI) scaling benchmark for FIRM3D tracing.

The CPU counterpart of bench_metal_gpu.py: traces the same fusion-birth
ensembles with the double-precision MPI tracer and writes timings, final
states, and the loss-fraction curve to JSON (rank 0 only).

Usage
-----
mpiexec -n 10 python -m mpi4py bench_cpu_scaling.py config.yaml out.json

(-m mpi4py turns an uncaught exception on any rank into MPI_Abort; without it
the surviving ranks deadlock in the next collective.)
"""

import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI  # noqa: F401  (must precede firm3d modules)

import bench_common as bc
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.trajectory_helpers import compute_loss_fraction
from firm3d.util.mpi import comm_world, verbose


def trace(
    field, points: np.ndarray, vpar: np.ndarray, tmax: float, tol: float
) -> list[np.ndarray]:
    """One CPU trace across the MPI ranks; returns per-particle rows."""
    res_tys, _ = trace_particles_boozer(
        field,
        points,
        vpar,
        tmax=tmax,
        mass=bc.MASS,
        charge=bc.CHARGE,
        Ekin=bc.EKIN,
        comm=comm_world,
        abstol=tol,
        reltol=tol,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
        forget_exact_path=True,
    )
    return res_tys


def main(config_path: Path, out_path: Path) -> None:
    """Run the CPU benchmark and write results from rank 0.

    Parameters
    ----------
    config_path : Path
        Benchmark YAML config.
    out_path : Path
        JSON results file to write.
    """
    cfg = bc.load_config(config_path)
    tol = float(cfg["tracing"]["tol"])
    tmax = float(cfg["tracing"]["tmax_s"])
    seed = int(cfg["seed"])

    results: dict = {
        "backend": "cpu",
        "ranks": comm_world.size if comm_world is not None else 1,
        "resolution": int(cfg["interpolation"]["resolution"]),
        "tol": tol,
    }

    t0 = time.time()
    bri, field = bc.build_field(cfg, comm=comm_world)
    results["field_build_s"] = time.time() - t0
    if verbose:
        print(f"equilibrium build: {results['field_build_s']:.1f} s", flush=True)

    scaling = []
    for n in cfg["scaling_nparticles"]:
        points, vpar = bc.sample_particles(field, int(n), seed, comm=comm_world)
        t0 = time.time()
        res_tys = trace(field, points, vpar, tmax, tol)
        wall = time.time() - t0
        bc.snap_survivor_times_inplace(res_tys, tmax)
        final_times = np.array([ty[-1, 0] for ty in res_tys])
        n_lost = int(np.sum(final_times < tmax * (1.0 - bc.TMAX_REL_TOL)))
        scaling.append(
            {"nparticles": int(n), "tmax_s": tmax, "wall_s": wall, "n_lost": n_lost}
        )
        if verbose:
            print(f"n={n}: {wall:.2f} s wall, {n_lost} lost", flush=True)
    results["scaling"] = scaling

    val = cfg["validation"]
    n_val = int(val["nparticles"])
    tmax_val = float(val["tmax_s"])
    points, vpar = bc.sample_particles(field, n_val, seed, comm=comm_world)
    t0 = time.time()
    res_tys = trace(field, points, vpar, tmax_val, tol)
    wall = time.time() - t0
    bc.snap_survivor_times_inplace(res_tys, tmax_val)
    times, loss_frac = compute_loss_fraction(
        res_tys, tmin=float(val["tmin_s"]), tmax=tmax_val
    )
    final = np.array([ty[-1, :] for ty in res_tys])
    results["validation"] = {
        "nparticles": n_val,
        "tmax_s": tmax_val,
        "wall_s": wall,
        "loss_times_s": times.tolist(),
        "loss_fraction": loss_frac.tolist(),
        "final_states": final.tolist(),
    }
    if verbose:
        print(
            f"validation n={n_val}, tmax={tmax_val:g}: {wall:.2f} s wall, "
            f"final loss fraction {loss_frac[-1]:.4f}",
            flush=True,
        )
        bc.write_results(out_path, results)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
