"""Shared pieces of the CPU and Metal scaling benchmarks.

Field setup and fusion-birth sampling, identical between the two scripts so
that both tracers integrate the same markers in the same field.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import sigmav

MASS = ALPHA_PARTICLE_MASS
CHARGE = ALPHA_PARTICLE_CHARGE
EKIN = FUSION_ALPHA_PARTICLE_ENERGY
VELOCITY = float(np.sqrt(2.0 * EKIN / MASS))

# Survivor classification margin, relative to tmax. The GPU integrates time in
# float32 and stops at the first step boundary at or after tmax, so a survivor
# can finish exactly on float32(tmax), which sits ~2.2e-9 below the float64
# tmax at tmax = 1e-2 (one float32 ulp there is ~9.3e-10). 1e-6 is comfortably
# above that and far below any physical loss-time spacing.
TMAX_REL_TOL = 1e-6


def snap_survivor_times_inplace(res_tys: list[np.ndarray], tmax: float) -> None:
    """Snap final times within TMAX_REL_TOL of tmax onto tmax, mutating res_tys.

    Without this, strict float64 comparisons downstream (for example the
    ``t < times - 1e-15`` test in ``compute_loss_fraction``) count a float32
    survivor that landed exactly on float32(tmax) as a lost particle.

    Parameters
    ----------
    res_tys : list of np.ndarray
        Per-particle trajectory rows, shape (nsteps, >=1); column 0 is time [s].
    tmax : float
        Requested tracing time [s].
    """
    for ty in res_tys:
        if ty[-1, 0] >= tmax * (1.0 - TMAX_REL_TOL):
            ty[-1, 0] = tmax


def load_config(path: Path) -> dict[str, Any]:
    """Load the benchmark configuration from YAML."""
    yaml = YAML(typ="safe")
    with open(path) as f:
        return yaml.load(f)


def build_field(
    cfg: dict[str, Any], comm: Any = None
) -> tuple[BoozerRadialInterpolant, InterpolatedBoozerField]:
    """Build the interpolated vacuum equilibrium both tracers share.

    The vacuum field is enforced, as in examples/gpu_boozer_tracing, so that
    the CPU and GPU tracers integrate the same (gc_vac) equations.

    Parameters
    ----------
    cfg : dict[str, Any]
        Parsed benchmark config.
    comm : MPI communicator or None
        Passed through to the field build.

    Returns
    -------
    tuple[BoozerRadialInterpolant, InterpolatedBoozerField]
        The radial interpolant (which must stay alive) and the tabulated
        equilibrium at the configured resolution.
    """
    interp = cfg["interpolation"]
    res = int(interp["resolution"])
    bri = BoozerRadialInterpolant(
        cfg["boozmn_file"], int(interp["degree"]), enforce_vacuum=True, comm=comm
    )
    # comm goes to the radial interpolant only, as in the upstream examples:
    # it parallelises over Fourier modes, so every rank's internal Allreduce
    # has the same buffer size. Passing comm here as well splits the grid
    # points across ranks, and the nested mode-sum Allreduces then disagree
    # in size between ranks (MPI_ERR_TRUNCATE / deadlock).
    field = InterpolatedBoozerField(
        bri,
        int(interp["degree"]),
        ns_interp=res,
        ntheta_interp=res,
        nzeta_interp=res,
    )
    return bri, field


def sample_particles(
    field: InterpolatedBoozerField, nparticles: int, seed: int, comm: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sample fusion-birth positions and uniform parallel velocities.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (nparticles, 3) Boozer positions and (nparticles,) parallel speeds,
        both float64. Deterministic in (field, nparticles, seed).
    """

    def reactivity(s: np.ndarray) -> np.ndarray:
        n_d = 1.0 - s**5
        temperature_kev = 11.5 * (1.0 - s)
        return n_d * n_d * sigmav(temperature_kev)

    points = initialize_position_profile(
        field, nparticles, reactivity, comm=comm, seed=seed
    )
    vpar = initialize_velocity_uniform(VELOCITY, nparticles, comm=comm, seed=seed)
    return points, vpar


def write_results(path: Path, results: dict[str, Any]) -> None:
    """Write benchmark results as JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
