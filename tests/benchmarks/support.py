"""Synthetic benchmark runs for the tooling tests: machines, traces, folders.

Builds run folders in the layout run_benchmark.py files (run.json, cpu.json,
metal.json, per-marker .npz), small enough for unit tests. Seeds and sizes
come from config.yaml; every trace of one benchmark follows the same
markers, as real runs do.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

CONFIG = YAML(typ="safe").load((Path(__file__).parent / "config.yaml").read_text())
SEED = int(CONFIG["seed"])
N = int(CONFIG["nparticles"])
TMAX = float(CONFIG["tmax_s"])
N_CONSERVATION = int(CONFIG["conservation_nparticles"])

RESOLUTIONS = (16, 32)  # synthetic conservation grid
TOLERANCES = (1e-4, 1e-6, 1e-8)
LOSS_CONFIG = {
    "description": "synthetic alphas",
    "interpolation": {"resolution": 16},
    "tracing": {"tmax_s": 1e-3, "tol": 1e-6},
    "validation": {"nparticles": N, "tmax_s": TMAX},
}
CONSERVATION_CONFIG = {
    "kind": "conservation",
    "description": "synthetic QH markers",
    "resolutions": list(RESOLUTIONS),
    "tolerances": list(TOLERANCES),
    "tracing": {"tmax_s": 1e-4, "dt_save_s": 1e-5},
    "nparticles": N_CONSERVATION,
}


def rng(offset: int = 0) -> np.random.Generator:
    """A generator seeded from config.yaml, offset for independent streams."""
    return np.random.default_rng(SEED + offset)


def machine_record(**overrides: Any) -> dict[str, Any]:
    """A machine record as machine.machine() returns it (base M4 defaults)."""
    record = {
        "model_name": "MacBook Pro",
        "model_identifier": "Mac16,1",
        "chip": "Apple M4",
        "cpu_cores": 10,
        "cpu_performance_cores": 4,
        "cpu_efficiency_cores": 6,
        "gpu_cores": 10,
        "memory_gb": 16,
        "macos": "26.6.2",
        "macos_build": "25G83",
    }
    record.update(overrides)
    return record


def final_states(
    n: int, tmax: float, lost_fraction: float, offset: int = 0
) -> np.ndarray:
    """(n, 5) float64 final states (t, s, theta, zeta, vpar).

    The first lost_fraction of the markers end before tmax; the rest survive
    and end on float32(tmax), as single-precision survivors do.
    """
    generator = rng(offset)
    states = generator.uniform(0.0, 1.0, (n, 5))
    states[:, 0] = np.float64(np.float32(tmax))
    n_lost = int(lost_fraction * n)
    states[:n_lost, 0] = generator.uniform(1e-5, 0.9 * tmax, n_lost)
    return states


def loss_results(states: np.ndarray, wall_scale: float) -> dict[str, Any]:
    """cpu.json / metal.json of a loss run with these validation final states."""
    lost = states[:, 0] < TMAX * (1 - 1e-6)
    times = np.logspace(-5, np.log10(TMAX), 50)
    fraction = [float(np.sum(states[lost, 0] < t)) / len(states) for t in times]
    return {
        "scaling": [
            {
                "nparticles": n,
                "tmax_s": 1e-3,
                "wall_s": wall_scale * n / 1000,
                "n_lost": n // 100,
            }
            for n in (100, 1000)
        ],
        "validation": {
            "nparticles": len(states),
            "tmax_s": TMAX,
            "wall_s": 20.0 * wall_scale,
            "n_lost": int(lost.sum()),
            "loss_times_s": times.tolist(),
            "loss_fraction": fraction,
        },
    }


def write_run(
    bench_dir: Path,
    name: str,
    *,
    created: str,
    tag: str,
    machine: dict[str, Any],
    config: dict[str, Any],
    cpu: dict[str, Any],
    metal: dict[str, Any],
    arrays: dict[str, dict[str, Any]],
    sanity: bool = True,
    fingerprints: dict[str, str] | None = None,
    mpi_ranks: int | None = None,
) -> Path:
    """Write one run folder; arrays maps .npz file names to their arrays.

    The CPU leg uses one rank per core unless mpi_ranks caps it.
    """
    folder = bench_dir / name
    folder.mkdir(parents=True)
    (folder / "cpu.json").write_text(json.dumps(cpu))
    (folder / "metal.json").write_text(json.dumps(metal))
    for filename, data in arrays.items():
        np.savez_compressed(folder / filename, **data)
    record = {
        "created": created,
        "tag": tag,
        "machine": machine,
        "mpi_ranks": machine["cpu_cores"] if mpi_ranks is None else mpi_ranks,
        "config": config,
        "fingerprints": fingerprints
        or {"config": "c0", "equilibrium": "e0", "metal_sources": "m0"},
        "sanity": {"passed": sanity, "checks": []},
    }
    (folder / "run.json").write_text(json.dumps(record))
    return folder


def write_loss_run(
    bench_dir: Path,
    name: str,
    *,
    created: str,
    tag: str = "neil",
    machine: dict[str, Any] | None = None,
    gpu_states: np.ndarray | None = None,
    sanity: bool = True,
) -> Path:
    """A loss run: GPU final states default to the CPU's (identical traces)."""
    cpu_states = final_states(N, TMAX, float(CONFIG["lost_fraction"]))
    gpu = cpu_states if gpu_states is None else gpu_states
    return write_run(
        bench_dir,
        name,
        created=created,
        tag=tag,
        machine=machine or machine_record(),
        config=LOSS_CONFIG,
        cpu=loss_results(cpu_states, wall_scale=7.0),
        metal=loss_results(gpu, wall_scale=1.0),
        arrays={
            "cpu_final_states.npz": {"final_states": cpu_states},
            "metal_final_states.npz": {"final_states": gpu},
        },
        sanity=sanity,
    )


def conservation_drift(floor: float, offset: int) -> np.ndarray:
    """(nresolutions, ntolerances, nparticles) drifts: tol-limited, then floored."""
    generator = rng(offset)
    medians = np.array(
        [[max(tol * 10.0, floor) for tol in TOLERANCES] for _ in RESOLUTIONS]
    )
    spread = generator.lognormal(
        0.0, 0.5, (len(RESOLUTIONS), len(TOLERANCES), N_CONSERVATION)
    )
    return (medians[:, :, None] * spread).astype(np.float32)


def conservation_results(
    drift: np.ndarray, wall_scale: float, build_s: float
) -> dict[str, Any]:
    """cpu.json / metal.json of a conservation run with these drifts."""
    cells = []
    for i, res in enumerate(RESOLUTIONS):
        for j, tol in enumerate(TOLERANCES):
            d = drift[i, j].astype(np.float64)
            cells.append(
                {
                    "resolution": res,
                    "tol": tol,
                    "field_build_s": build_s * res / 16,
                    "wall_s": wall_scale * (1.0 + j),
                    "n_lost": 3,
                    "drift": {
                        "p50": float(np.median(d)),
                        "p90": float(np.percentile(d, 90)),
                        "p99": float(np.percentile(d, 99)),
                        "max": float(d.max()),
                    },
                }
            )
    return {"cells": cells}


def write_conservation_run(
    bench_dir: Path,
    name: str,
    *,
    created: str,
    tag: str = "neil",
    machine: dict[str, Any] | None = None,
    gpu_offset: int = 1,
    mpi_ranks: int | None = None,
) -> Path:
    """A conservation run: the GPU floors at 2e-6, the CPU at 1e-8.

    gpu_offset seeds the GPU's drifts; two runs with the same offset hold
    bitwise identical GPU drifts, as two runs on one chip do.
    """
    cpu_drift = conservation_drift(floor=1e-8, offset=0)
    gpu_drift = conservation_drift(floor=2e-6, offset=gpu_offset)
    peta0 = rng(2).uniform(1e-19, 1e-18, N_CONSERVATION)
    return write_run(
        bench_dir,
        name,
        created=created,
        tag=tag,
        machine=machine or machine_record(),
        config=CONSERVATION_CONFIG,
        cpu=conservation_results(cpu_drift, wall_scale=4.0, build_s=1.0),
        metal=conservation_results(gpu_drift, wall_scale=1.0, build_s=2.0),
        arrays={
            "cpu_drift.npz": {"drift": cpu_drift, "peta0": peta0},
            "metal_drift.npz": {"drift": gpu_drift, "peta0": peta0},
        },
        mpi_ranks=mpi_ranks,
    )
