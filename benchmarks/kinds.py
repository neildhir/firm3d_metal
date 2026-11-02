"""What differs between kinds of benchmark, for run_benchmark.py.

A benchmark's config names its kind (``kind:``); configs written before
kinds existed are loss benchmarks. Each kind has its own leg scripts, its
own run-time sanity check (a tripwire for a broken build or install, not a
physics comparison), and its own printed summary.

- loss: fusion-birth alphas, wall time vs ensemble size and the
  loss-fraction history (bench_cpu_scaling.py, bench_metal_gpu.py)
- conservation: p_eta drift in an exactly quasi-symmetric field over a grid
  of resolutions and tolerances (bench_cpu_conservation.py,
  bench_metal_conservation.py)
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Results = dict[str, Any]

# Loss: CPU and GPU loss counts must agree within this many standard
# deviations of the difference of two independent Poisson counts. Both
# backends trace the same markers, so genuine disagreement is far smaller:
# the bound is loose on purpose, and only a broken build or install exceeds it.
SANITY_SIGMAS = 3.0
# Conservation: in cells whose tolerance float32 resolves (its machine epsilon
# is 1.2e-7), CPU and GPU median drifts must agree within this factor, either
# way. Both are integration-error dominated there and agree to well within 2x.
SANITY_MIN_TOL = 1e-6
SANITY_FACTOR = 10.0


@dataclass(frozen=True)
class Kind:
    """The leg scripts, sanity check, and summary of one kind of benchmark."""

    cpu_script: str
    metal_script: str
    sanity_check: Callable[[Results, Results], Results]
    summary: Callable[[Results, Results], list[str]]
    # environment for the MPI ranks, passed as mpiexec -x NAME=VALUE
    mpi_env: tuple[str, ...] = ()


def kind_of(cfg: dict[str, Any]) -> str:
    """The config's kind; configs from before kinds existed are loss ones."""
    return cfg.get("kind", "loss")


def loss_sanity(cpu: Results, metal: Results) -> Results:
    """Check CPU and GPU loss counts agree, workload by workload.

    Returns
    -------
    Results
        ``passed`` (bool) and the per-workload ``checks``.
    """
    pairs = [
        (f"{c['nparticles']} x {c['tmax_s']:g} s", c["n_lost"], g["n_lost"])
        for c, g in zip(cpu["scaling"], metal["scaling"], strict=True)
    ]
    v_cpu, v_gpu = cpu["validation"], metal["validation"]
    pairs.append(
        (
            f"{v_cpu['nparticles']} x {v_cpu['tmax_s']:g} s (validation)",
            v_cpu["n_lost"],
            v_gpu["n_lost"],
        )
    )
    checks = [
        {
            "workload": name,
            "n_lost_cpu": n_cpu,
            "n_lost_metal": n_gpu,
            "passed": abs(n_cpu - n_gpu) <= SANITY_SIGMAS * math.sqrt(n_cpu + n_gpu),
        }
        for name, n_cpu, n_gpu in pairs
    ]
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def loss_summary(cpu: Results, metal: Results) -> list[str]:
    """Wall times, speedups, and loss counts, one line per workload."""
    rows = [
        *zip(cpu["scaling"], metal["scaling"], strict=True),
        (cpu["validation"], metal["validation"]),
    ]
    lines = [f"{'workload':>24}  {'CPU [s]':>8}  {'GPU [s]':>8}  speedup  lost CPU/GPU"]
    for c, g in rows:
        workload = f"{c['nparticles']} x {c['tmax_s']:g} s"
        lines.append(
            f"{workload:>24}  {c['wall_s']:8.2f}  {g['wall_s']:8.2f}  "
            f"{c['wall_s'] / g['wall_s']:6.1f}x  {c['n_lost']:>5} / {g['n_lost']}"
        )
    return lines


def conservation_sanity(cpu: Results, metal: Results) -> Results:
    """Check CPU and GPU median drifts agree where float32 can resolve tol.

    Returns
    -------
    Results
        ``passed`` (bool) and the per-cell ``checks``.
    """
    checks = []
    for c, g in zip(cpu["cells"], metal["cells"], strict=True):
        if c["tol"] < SANITY_MIN_TOL:
            continue
        d_cpu, d_gpu = c["drift"]["p50"], g["drift"]["p50"]
        ratio = d_gpu / d_cpu if d_cpu > 0 else math.inf
        checks.append(
            {
                "workload": f"resolution {c['resolution']}, tol {c['tol']:g}",
                "median_drift_cpu": d_cpu,
                "median_drift_metal": d_gpu,
                "passed": 1 / SANITY_FACTOR <= ratio <= SANITY_FACTOR,
            }
        )
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def conservation_summary(cpu: Results, metal: Results) -> list[str]:
    """Wall times and median drifts, one line per (resolution, tolerance)."""
    lines = [
        f"{'resolution, tol':>18}  {'CPU [s]':>8}  {'GPU [s]':>8}  speedup  "
        "median drift CPU / GPU"
    ]
    for c, g in zip(cpu["cells"], metal["cells"], strict=True):
        cell = f"{c['resolution']}, {c['tol']:g}"
        lines.append(
            f"{cell:>18}  {c['wall_s']:8.2f}  {g['wall_s']:8.2f}  "
            f"{c['wall_s'] / g['wall_s']:6.1f}x  "
            f"{c['drift']['p50']:.2e} / {g['drift']['p50']:.2e}"
        )
    return lines


KINDS = {
    "loss": Kind(
        "bench_cpu_scaling.py", "bench_metal_gpu.py", loss_sanity, loss_summary
    ),
    "conservation": Kind(
        "bench_cpu_conservation.py",
        "bench_metal_conservation.py",
        conservation_sanity,
        conservation_summary,
        # one maths thread per rank: the field builds otherwise oversubscribe
        # the cores, doubling build time and per-rank memory
        mpi_env=("VECLIB_MAXIMUM_THREADS=1", "OMP_NUM_THREADS=1"),
    ),
}
