"""CPU (MPI) conservation benchmark for FIRM3D tracing.

Traces the benchmark's markers with the double-precision MPI tracer in the
exactly quasi-symmetric field, at every (resolution, tolerance) of the
config, and records each cell's tracing time and the per-marker drift of
p_eta (bench_conservation_common). Rank 0 writes the summary to JSON and the
drifts to a sibling .npz. Normally launched by run_benchmark.py; by hand,
one rank per physical core and one maths thread per rank:

Usage
-----
mpiexec -n 10 -x VECLIB_MAXIMUM_THREADS=1 -x OMP_NUM_THREADS=1 \
    python -m mpi4py bench_cpu_conservation.py conservation-qh.yaml cpu.json
"""

import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI  # noqa: F401  (must precede firm3d modules)

import bench_common as bc
import bench_conservation_common as cc
from firm3d.field.boozermagneticfield import InterpolatedBoozerField
from firm3d.field.tracing import (
    MaxToroidalFluxStoppingCriterion,
    trace_particles_boozer,
)
from firm3d.util.mpi import comm_world, verbose


def trace(
    field: InterpolatedBoozerField,
    points: np.ndarray,
    vpar: np.ndarray,
    tmax: float,
    dt_save: float,
    tol: float,
) -> list[np.ndarray]:
    """One CPU trace across the MPI ranks, saving every dt_save."""
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
        forget_exact_path=False,
        dt_save=dt_save,
    )
    return res_tys


def main(config_path: Path, out_path: Path) -> None:
    """Run the CPU conservation benchmark and write results from rank 0.

    Parameters
    ----------
    config_path : Path
        Benchmark YAML config.
    out_path : Path
        JSON results file to write.
    """
    cfg = bc.load_config(config_path)
    resolutions = [int(r) for r in cfg["resolutions"]]
    tols = [float(t) for t in cfg["tolerances"]]
    tmax = float(cfg["tracing"]["tmax_s"])
    dt_save = float(cfg["tracing"]["dt_save_s"])
    n = int(cfg["nparticles"])

    results: dict = {
        "backend": "cpu",
        "ranks": comm_world.size if comm_world is not None else 1,
        "nparticles": n,
        "tmax_s": tmax,
        "dt_save_s": dt_save,
    }
    t0 = time.time()
    spline = cc.build_spline_field(cfg, comm=comm_world)
    results["spline_build_s"] = time.time() - t0
    points, vpar = cc.sample_particles(spline, n, int(cfg["seed"]), comm=comm_world)

    cells = []
    drift = np.zeros((len(resolutions), len(tols), n))
    for i, res in enumerate(resolutions):
        comm_world.Barrier()
        t0 = time.time()
        field = InterpolatedBoozerField(
            spline,
            int(cfg["interpolation"]["degree"]),
            ns_interp=res,
            ntheta_interp=res,
            nzeta_interp=res,
            nfp=spline.nfp,
            stellsym=spline.stellsym,
            comm=comm_world,
        )
        build_s = time.time() - t0
        for j, tol in enumerate(tols):
            comm_world.Barrier()  # rank 0's drift evaluation must not be timed
            t0 = time.time()
            res_tys = trace(field, points, vpar, tmax, dt_save, tol)
            wall = time.time() - t0
            if not verbose:
                continue
            drift[i, j] = cc.peta_drift(spline, res_tys, cfg)
            cells.append(
                {
                    "resolution": res,
                    "tol": tol,
                    "field_build_s": build_s,
                    "wall_s": wall,
                    "n_lost": bc.n_lost(res_tys, tmax),
                    "drift": cc.summarise(drift[i, j]),
                }
            )
            print(
                f"res {res}, tol {tol:g}: {wall:.2f} s wall, "
                f"median drift {cells[-1]['drift']['p50']:.2e}",
                flush=True,
            )
        del field  # never hold two resolutions' interpolants at once
    results["cells"] = cells
    if verbose:
        cc.write_results(
            out_path, results, drift, cc.initial_peta(spline, points, vpar, cfg)
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
