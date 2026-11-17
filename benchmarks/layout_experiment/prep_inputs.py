"""Write the layout experiment's inputs as raw files the Objective-C++ harness reads.

Run from benchmarks/ (the benchmark config names its field file relative to it):
    ../.venv/bin/python layout_experiment/prep_inputs.py layout_experiment/config.yaml
Writes <data_dir>/table.f32 (the CATAPULT single-precision table, current layout),
<data_dir>/particles.f32 (nparticles_max x 4: x1, x2, zeta, vpar)
and <data_dir>/meta.txt (grid ranges, psi0, mass, charge, vtotal).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
sys.path.insert(0, str(Path(__file__).parent))
import bench_common as bc  # noqa: E402
from firm3d.catapult.field import CatapultBoozerField  # noqa: E402
from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)  # noqa: E402
from layout_config import LayoutConfig, load_config  # noqa: E402


def pseudo_cartesian_state(points: np.ndarray, vpar: np.ndarray) -> np.ndarray:
    """(n, 4) float32 rows (s cos theta, s sin theta, zeta, vpar), as catapult does.

    Parameters
    ----------
    points : np.ndarray
        (n, 3) float64 Boozer positions (s, theta, zeta).
    vpar : np.ndarray
        (n,) float64 parallel speeds [m/s].
    """
    x = np.array(points, dtype=np.float32, order="C")
    s = x[:, 0].copy()
    theta = x[:, 1].copy()
    x[:, 0] = s * np.cos(theta)
    x[:, 1] = s * np.sin(theta)
    return np.column_stack((x, vpar.astype(np.float32))).astype(np.float32)


def main(cfg: LayoutConfig) -> None:
    """Build the field table and markers once and write them out."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    bench = bc.load_config(cfg.benchmark_config)
    res = int(bench["interpolation"]["resolution"])
    if cfg.field == "vacuum":
        _, field = bc.build_field(bench)
    else:
        # the same equilibrium without the vacuum assumption: the general (K != 0)
        # equations, which tabulate 12 quantities instead of 6
        interp = bench["interpolation"]
        bri = BoozerRadialInterpolant(bench["boozmn_file"], int(interp["degree"]))
        field = InterpolatedBoozerField(
            bri,
            int(interp["degree"]),
            ns_interp=res,
            ntheta_interp=res,
            nzeta_interp=res,
        )
    cfield = CatapultBoozerField(field, res, res, res, precision="single")
    points, vpar = bc.sample_particles(field, cfg.nparticles_max, int(bench["seed"]))

    np.ascontiguousarray(cfield.quad_info, dtype=np.float32).tofile(
        cfg.data_dir / "table.f32"
    )
    pseudo_cartesian_state(points, vpar).tofile(cfg.data_dir / "particles.f32")
    lines = [
        "srange " + " ".join(repr(float(v)) for v in cfield.srange),
        "trange " + " ".join(repr(float(v)) for v in cfield.trange),
        "zrange " + " ".join(repr(float(v)) for v in cfield.zrange),
        f"psi0 {float(cfield.psi0)!r}",
        f"mass {float(bc.MASS)!r}",
        f"charge {float(bc.CHARGE)!r}",
        f"vtotal {float(bc.VELOCITY)!r}",
        f"rhs_id {0 if cfield.vacuum else 1}",
    ]
    (cfg.data_dir / "meta.txt").write_text("\n".join(lines) + "\n")
    print(
        f"field type {cfield.field_type!r}; wrote table "
        f"({cfield.quad_info.nbytes / 2**20:.0f} MB) and {cfg.nparticles_max} "
        f"markers to {cfg.data_dir}"
    )


if __name__ == "__main__":
    main(load_config(Path(sys.argv[1])))
