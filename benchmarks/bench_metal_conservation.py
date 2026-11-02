"""Metal GPU conservation benchmark for FIRM3D tracing.

The GPU counterpart of bench_cpu_conservation.py: traces the same markers in
the same exactly quasi-symmetric field with the Metal CATAPULT backend in
single precision, at every (resolution, tolerance) of the config, and
records each cell's tracing time and the per-marker drift of p_eta. The GPU
tables are built from the exact field, as in examples/gpu_boozer_tracing.
Normally launched by run_benchmark.py; by hand:

Usage
-----
python bench_metal_conservation.py conservation-qh.yaml metal.json
"""

import sys
import time
from pathlib import Path

import numpy as np

import bench_common as bc
import bench_conservation_common as cc
from firm3d.catapult.field import CatapultBoozerField
from firm3d.catapult.tracing import trace_particles_boozer_gpu


def trace(
    cfield: CatapultBoozerField,
    points: np.ndarray,
    vpar: np.ndarray,
    tmax: float,
    dt_save: float,
    tol: float,
) -> list[np.ndarray]:
    """One Metal trace, saving every dt_save."""
    res_tys, _ = trace_particles_boozer_gpu(
        cfield,
        points,
        vpar,
        tmax=tmax,
        mass=bc.MASS,
        charge=bc.CHARGE,
        Ekin=bc.EKIN,
        tol=tol,
        dt_save=dt_save,
        forget_exact_path=False,
    )
    return res_tys


def main(config_path: Path, out_path: Path) -> None:
    """Run the GPU conservation benchmark and write results.

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
        "backend": "metal",
        "nparticles": n,
        "tmax_s": tmax,
        "dt_save_s": dt_save,
    }
    t0 = time.time()
    spline = cc.build_spline_field(cfg)
    results["spline_build_s"] = time.time() - t0
    points, vpar = cc.sample_particles(spline, n, int(cfg["seed"]))

    cells = []
    drift = np.zeros((len(resolutions), len(tols), n))
    for i, res in enumerate(resolutions):
        t0 = time.time()
        cfield = CatapultBoozerField(spline, res, res, res, precision="single")
        build_s = time.time() - t0
        # warm up: first use compiles the Metal pipelines for this table
        trace(cfield, points[:8], vpar[:8], 1e-7, 1e-7, tols[0])
        for j, tol in enumerate(tols):
            t0 = time.time()
            res_tys = trace(cfield, points, vpar, tmax, dt_save, tol)
            wall = time.time() - t0
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
        del cfield
    results["cells"] = cells
    cc.write_results(
        out_path, results, drift, cc.initial_peta(spline, points, vpar, cfg)
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
