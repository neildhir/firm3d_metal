"""Metal GPU scaling benchmark for FIRM3D tracing.

Traces fusion-birth ensembles of increasing size with the Metal CATAPULT
backend in single precision, then the validation ensemble, and writes
timings, final states, and the loss-fraction curve to JSON.

Usage
-----
python bench_metal_gpu.py bench_scaling_config.yaml metal_results.json
"""

import sys
import time
from pathlib import Path

import numpy as np

import bench_common as bc
from firm3d.catapult.field import CatapultBoozerField
from firm3d.catapult.tracing import trace_particles_boozer_gpu
from firm3d.trajectory_helpers import compute_loss_fraction


def trace(
    cfield: CatapultBoozerField,
    points: np.ndarray,
    vpar: np.ndarray,
    tmax: float,
    tol: float,
) -> list[np.ndarray]:
    """One Metal trace; returns the per-particle trajectory rows."""
    res_tys, _ = trace_particles_boozer_gpu(
        cfield,
        points,
        vpar,
        tmax=tmax,
        mass=bc.MASS,
        charge=bc.CHARGE,
        Ekin=bc.EKIN,
        tol=tol,
        forget_exact_path=True,
    )
    return res_tys


def main(config_path: Path, out_path: Path) -> None:
    """Run the GPU benchmark and write results.

    Parameters
    ----------
    config_path : Path
        Benchmark YAML config.
    out_path : Path
        JSON results file to write.
    """
    cfg = bc.load_config(config_path)
    res = int(cfg["interpolation"]["resolution"])
    tol = float(cfg["tracing"]["tol"])
    tmax = float(cfg["tracing"]["tmax_s"])
    seed = int(cfg["seed"])

    results: dict = {"backend": "metal", "resolution": res, "tol": tol}

    t0 = time.time()
    bri, field = bc.build_field(cfg)
    results["field_build_s"] = time.time() - t0
    print(f"equilibrium build: {results['field_build_s']:.1f} s", flush=True)

    t0 = time.time()
    cfield = CatapultBoozerField(field, res, res, res, precision="single")
    results["table_build_s"] = time.time() - t0
    print(f"CATAPULT table build: {results['table_build_s']:.1f} s", flush=True)

    # warm up: first use compiles the Metal pipelines
    pts, vp = bc.sample_particles(field, 8, seed)
    trace(cfield, pts, vp, 1e-7, tol)

    scaling = []
    for n in cfg["scaling_nparticles"]:
        points, vpar = bc.sample_particles(field, int(n), seed)
        t0 = time.time()
        res_tys = trace(cfield, points, vpar, tmax, tol)
        wall = time.time() - t0
        bc.snap_survivor_times_inplace(res_tys, tmax)
        final_times = np.array([ty[-1, 0] for ty in res_tys])
        n_lost = int(np.sum(final_times < tmax * (1.0 - bc.TMAX_REL_TOL)))
        scaling.append(
            {"nparticles": int(n), "tmax_s": tmax, "wall_s": wall, "n_lost": n_lost}
        )
        print(f"n={n}: {wall:.2f} s wall, {n_lost} lost", flush=True)
    results["scaling"] = scaling

    val = cfg["validation"]
    n_val = int(val["nparticles"])
    tmax_val = float(val["tmax_s"])
    points, vpar = bc.sample_particles(field, n_val, seed)
    t0 = time.time()
    res_tys = trace(cfield, points, vpar, tmax_val, tol)
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
    print(
        f"validation n={n_val}, tmax={tmax_val:g}: {wall:.2f} s wall, "
        f"final loss fraction {loss_frac[-1]:.4f}",
        flush=True,
    )

    bc.write_results(out_path, results)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
