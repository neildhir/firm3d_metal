"""Build the shared inputs and trace them with the neildhir/firm3d_metal Metal kernel.

Run from firm3d_metal/benchmarks with that repo's .venv:
    python run_ours.py config.yaml
Writes <data_dir>/inputs.npz (table in both layouts, grid ranges, markers, constants)
and <data_dir>/ours.npz (timings and final states per tolerance).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # benchmarks/
sys.path.insert(0, str(Path(__file__).parent))
import bench_common as bc  # noqa: E402
from firm3d.catapult.field import CatapultBoozerField  # noqa: E402
from firm3d.catapult.tracing import trace_particles_boozer_gpu  # noqa: E402
from h2h_config import H2HConfig, load_config  # noqa: E402


def trace_final(
    cfield: CatapultBoozerField,
    points: np.ndarray,
    vpar: np.ndarray,
    tmax: float,
    tol: float,
) -> np.ndarray:
    """Trace with the public GPU API; returns (n, 5) rows [t, s, theta, zeta, vpar]."""
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
    return np.array([ty[-1, :5] for ty in res_tys])


def main(cfg: H2HConfig) -> None:
    """Build the inputs once, save them, and time this kernel on them."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    bench = bc.load_config(cfg.benchmark_config)
    res = int(bench["interpolation"]["resolution"])
    seed = int(bench["seed"])
    _, field = bc.build_field(bench)
    cfield = CatapultBoozerField(field, res, res, res, precision="single")
    points, vpar = bc.sample_particles(field, cfg.nparticles, seed)

    # current upstream layout is (cells, fields, 64); the Feb 2026 layout their
    # kernel reads is (cells*64, fields)
    table = np.asarray(cfield.quad_info)
    table_feb = np.ascontiguousarray(np.swapaxes(table, 1, 2)).reshape(
        -1, table.shape[1]
    )
    np.savez(
        cfg.data_dir / "inputs.npz",
        table_feb=table_feb,
        srange=np.asarray(cfield.srange, dtype=float),
        trange=np.asarray(cfield.trange, dtype=float),
        zrange=np.asarray(cfield.zrange, dtype=float),
        psi0=cfield.psi0,
        points=points,
        vpar=vpar,
        mass=bc.MASS,
        charge=bc.CHARGE,
        velocity=bc.VELOCITY,
    )

    wpts, wvp = bc.sample_particles(field, cfg.warmup_nparticles, seed)
    trace_final(cfield, wpts, wvp, cfg.warmup_tmax_s, cfg.tols[0])

    out: dict[str, np.ndarray] = {}
    for tol in cfg.tols:
        walls = []
        for _ in range(cfg.repeats):
            t0 = time.perf_counter()
            final = trace_final(cfield, points, vpar, cfg.tmax_s, tol)
            walls.append(time.perf_counter() - t0)
        out[f"wall_{tol:g}"] = np.array(walls)
        out[f"final_{tol:g}"] = final
        print(
            f"ours tol={tol:g}: walls {np.round(walls, 3)} s, "
            f"lost(s>=1) {int(np.sum(final[:, 1] >= 1.0))}",
            flush=True,
        )
    np.savez(cfg.data_dir / "ours.npz", **out)


if __name__ == "__main__":
    main(load_config(Path(sys.argv[1])))
