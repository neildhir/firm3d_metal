"""Shared pieces of the CPU and Metal conservation benchmarks.

In a field whose strength is exactly quasi-symmetric with helicity (M, N),
the guiding-centre equations conserve the canonical momentum p_eta, so any
change of p_eta along a trace is numerical error: of the field's
representation on a grid (resolution), of the integrator (tolerance), and of
the arithmetic (float64 or float32). Both legs build the same exactly
quasi-symmetric field, trace the same markers on a grid of resolutions and
tolerances, and measure p_eta on that exact field, so the two backends are
held to one yardstick.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench_common import CHARGE, MASS, VELOCITY
from firm3d.field.boozermagneticfield import BoozerSplineField
from firm3d.field.tracing_helpers import (
    initialize_position_uniform_vol,
    initialize_velocity_uniform,
)
from firm3d.trajectory_helpers import compute_peta

PERCENTILES = (50, 90, 99)  # of the per-marker drift, summarised per cell


def build_spline_field(cfg: dict[str, Any], comm: Any = None) -> BoozerSplineField:
    """The exactly quasi-symmetric equilibrium both legs start from.

    Non-symmetric harmonics of B are filtered out, so the field depends on
    (s, M theta - N zeta) only, and K is dropped (field type "nok"), as in
    upstream's examples/resolution_scan. Dropping K matters: with a filtered
    K kept, p_eta drifts by ~1e-5 whatever the resolution, so the drift
    would measure the model rather than the numerics.

    Parameters
    ----------
    cfg : dict[str, Any]
        Parsed benchmark config.
    comm : MPI communicator or None
        Parallelises the construction across ranks.

    Returns
    -------
    BoozerSplineField
        The field, of type "nok".
    """
    return BoozerSplineField(
        cfg["boozmn_file"],
        helicity_M=cfg["helicity"]["m"],
        helicity_N=cfg["helicity"]["n"],
        no_K=True,
        comm=comm,
    )


def sample_particles(
    field: BoozerSplineField, nparticles: int, seed: int, comm: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Markers uniform in volume, parallel speeds uniform in pitch.

    Sampled on the exact field, so both legs trace identical markers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (nparticles, 3) Boozer positions and (nparticles,) parallel speeds,
        both float64. Deterministic in (field, nparticles, seed).
    """
    points = initialize_position_uniform_vol(field, nparticles, comm=comm, seed=seed)
    vpar = initialize_velocity_uniform(VELOCITY, nparticles, comm=comm, seed=seed)
    # both broadcast from rank 0, so every rank holds the markers
    if points is None or vpar is None:
        raise RuntimeError(f"sampling {nparticles} markers returned nothing")
    return points, vpar


def peta_scale(field: BoozerSplineField) -> float:
    """q |psi_edge|, the natural scale of p_eta, that drifts are measured in.

    Raises
    ------
    ValueError
        If the field carries no edge flux.
    """
    if field.psi0 is None:
        raise ValueError("the field has no psi0 (edge toroidal flux / 2 pi)")
    return CHARGE * abs(field.psi0)


def initial_peta(
    field: BoozerSplineField, points: np.ndarray, vpar: np.ndarray, cfg: dict[str, Any]
) -> np.ndarray:
    """p_eta of the markers' initial states on the exact field.

    Stored with the drifts so that upstream's per-marker relative error,
    drift * peta_scale / |p_eta(0)|, can be recovered.

    Returns
    -------
    np.ndarray
        (nparticles,) float64 p_eta [kg m^2 / s, i.e. J s].
    """
    m, n = cfg["helicity"]["m"], cfg["helicity"]["n"]
    return compute_peta(field, points, vpar, MASS, CHARGE, m, n)


def peta_drift(
    field: BoozerSplineField, res_tys: list[np.ndarray], cfg: dict[str, Any]
) -> np.ndarray:
    """Largest change of p_eta along each trace, on the exact field.

    Normalised by q |psi_edge|, the natural scale of p_eta itself, rather
    than by each marker's initial p_eta: some markers start with p_eta near
    zero, which would turn an ordinary drift into a huge relative error.

    Parameters
    ----------
    field : BoozerSplineField
        The exact field the traces are measured on.
    res_tys : list[np.ndarray]
        Per marker, the saved states, rows (t, s, theta, zeta, vpar).
    cfg : dict[str, Any]
        Parsed benchmark config (for the helicity).

    Returns
    -------
    np.ndarray
        (nparticles,) float64 drift max_t |p_eta(t) - p_eta(0)| / (q |psi_edge|).
    """
    m, n = cfg["helicity"]["m"], cfg["helicity"]["n"]
    scale = peta_scale(field)
    drift = np.empty(len(res_tys))
    for i, ty in enumerate(res_tys):
        peta = compute_peta(
            field,
            np.ascontiguousarray(ty[:, 1:4]),
            np.ascontiguousarray(ty[:, 4]),
            MASS,
            CHARGE,
            m,
            n,
        )
        drift[i] = np.max(np.abs(peta - peta[0])) / scale
    return drift


def summarise(drift: np.ndarray) -> dict[str, float]:
    """Percentiles and maximum of the per-marker drift."""
    stats = {f"p{q}": float(np.percentile(drift, q)) for q in PERCENTILES}
    stats["max"] = float(drift.max())
    return stats


def drift_path(results_path: Path) -> Path:
    """Where the per-marker drifts belonging to a results JSON are stored."""
    return results_path.with_name(f"{results_path.stem}_drift.npz")


def write_results(
    path: Path, results: dict[str, Any], drift: np.ndarray, peta0: np.ndarray
) -> None:
    """Write the summary as JSON and the per-marker data beside it.

    The drifts are stored in float32, ample for an error measure, which
    keeps a run's file under the repository's 500 KB pre-commit limit.

    Parameters
    ----------
    path : Path
        JSON file to write.
    results : dict[str, Any]
        Timings and drift summaries per (resolution, tolerance) cell.
    drift : np.ndarray
        (nresolutions, ntolerances, nparticles) per-marker drift, in the
        config's resolution and tolerance order.
    peta0 : np.ndarray
        (nparticles,) float64 initial p_eta of the markers (initial_peta).
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")  # the repo's end-of-file-fixer hook would rewrite it
    np.savez_compressed(drift_path(path), drift=drift.astype(np.float32), peta0=peta0)
