"""Benchmark this Mac: CPU (MPI) vs Metal GPU tracing, filed for comparison.

The one command contributors run. It checks the machine is fit to benchmark,
traces the benchmark's workloads with the MPI CPU tracer (one rank per
physical core), pauses so the machine cools, traces them again with the Metal
GPU tracer, and files everything in a new folder results/<benchmark>/<run-id>/:

- run.json: machine, run conditions, code and config fingerprints, and a
  CPU-vs-GPU sanity check of the loss counts
- cpu.json, metal.json: timings and loss-fraction curves per backend
- cpu_final_states.npz, metal_final_states.npz: per-marker final states

The benchmark is named by its config file: alpha-loss-wistell-a.yaml files
its runs under results/alpha-loss-wistell-a/. make_plots.py reads them all.
A run in progress (or one that crashed) is kept in <run-id>.incomplete/.

Usage
-----
cd benchmarks
../.venv/bin/python run_benchmark.py alpha-loss-wistell-a.yaml --tag <github-handle>
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import firm3dpp

import bench_common as bc
import machine as mach

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
INSTALL_SCRIPT = "install/macOS/install_firm3d_metal.sh"
METAL_KERNEL = "src/firm3dpp/metal_kernel.metal"
METAL_SOURCES = (METAL_KERNEL, "src/firm3dpp/metal_wrapper.mm")

# Pause between the CPU and GPU legs [s], so that the GPU leg does not start
# on a machine heat-soaked by the CPU leg (sustained load costs the M4 ~8%).
COOLDOWN_S = 60
# CPU and GPU loss counts must agree within this many standard deviations of
# the difference of two independent Poisson counts. Both backends trace the
# same markers, so genuine disagreement is far smaller: the bound is loose on
# purpose, and only a broken build or install exceeds it.
SANITY_SIGMAS = 3.0


def main() -> None:
    """Check the machine, run both legs, sanity-check, and file the run."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = bc.load_config(config_path)
    preflight(args.allow_dirty)
    # the code and inputs the run starts from; edits made while it runs must
    # not change what it records
    prints, sw = fingerprints(cfg), software()

    hw = mach.machine()
    created = datetime.now().astimezone()
    benchmark = config_path.stem
    run_id = f"{created:%Y-%m-%d-%H%M}_{mach.slug(hw)}_{args.tag}"
    out = RESULTS_DIR / benchmark / f"{run_id}.incomplete"
    out.mkdir(parents=True)
    print(f"benchmarking {mach.label(hw)}", flush=True)

    ranks = hw["cpu_cores"]
    legs = {"cpu": run_leg(cpu_command(ranks, config_path, out / "cpu.json"))}
    print(f"cooling down for {COOLDOWN_S} s", flush=True)
    time.sleep(COOLDOWN_S)
    legs["metal"] = run_leg(
        [
            sys.executable,
            "bench_metal_gpu.py",
            str(config_path),
            str(out / "metal.json"),
        ]
    )

    cpu = json.loads((out / "cpu.json").read_text())
    metal = json.loads((out / "metal.json").read_text())
    record = {
        "benchmark": benchmark,
        "run_id": run_id,
        "created": created.isoformat(timespec="seconds"),
        "tag": args.tag,
        "machine": hw,
        "mpi_ranks": ranks,
        "legs": legs,
        "fingerprints": prints,
        "software": sw,
        "config": cfg,
        "sanity": sanity_check(cpu, metal),
    }
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    final = out.rename(out.with_name(run_id))
    report(record, cpu, metal, final)


def parse_args() -> argparse.Namespace:
    """Command line: the benchmark config, the contributor tag, and overrides."""
    parser = argparse.ArgumentParser(
        description="Benchmark this Mac: CPU (MPI) vs Metal GPU tracing."
    )
    parser.add_argument(
        "config", help="benchmark config, e.g. alpha-loss-wistell-a.yaml"
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="your GitHub handle (or initials): names the run folder",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="benchmark with uncommitted changes (recorded in run.json)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", args.tag):
        parser.error(f"--tag {args.tag!r}: use letters, digits, and hyphens only")
    return args


def preflight(allow_dirty: bool) -> None:
    """Refuse to benchmark when the results would be wrong or not comparable.

    Parameters
    ----------
    allow_dirty : bool
        Accept uncommitted changes to tracked files.

    Raises
    ------
    SystemExit
        With the reason and the fix, at the first failed check.
    """
    backend = getattr(firm3dpp, "gpu_backend", None)
    if backend != "metal":
        sys.exit(
            f"firm3dpp was built without Metal (gpu_backend = {backend!r}); "
            f"run {INSTALL_SCRIPT}"
        )
    # The kernel source is embedded verbatim in the extension, so a build that
    # predates the last `git pull` shows up as a mismatch here.
    kernel = (REPO_ROOT / METAL_KERNEL).read_bytes()
    built = firm3dpp.__file__
    if built is None or kernel not in Path(built).read_bytes():
        sys.exit(
            f"the built Metal kernel is stale (it differs from {METAL_KERNEL}); "
            f"re-run {INSTALL_SCRIPT}"
        )
    knobs = sorted(k for k in os.environ if k.startswith("FIRM3D_METAL_"))
    if knobs:
        sys.exit(f"unset the Metal debug variables first: {', '.join(knobs)}")
    if shutil.which("mpiexec") is None:
        sys.exit(f"mpiexec not found; run {INSTALL_SCRIPT}")

    now = mach.conditions()
    if now["power_source"] != "AC Power":
        sys.exit("plug in the power adapter: on battery macOS caps performance")
    if now["low_power_mode"]:
        sys.exit("turn off Low Power Mode (System Settings > Battery)")
    if now["thermal_state"] != "nominal":
        print(
            "warning: the machine is already warm "
            f"(thermal state {now['thermal_state']})",
            flush=True,
        )
    if git("status", "--porcelain", "--untracked-files=no") and not allow_dirty:
        sys.exit(
            "tracked files have uncommitted changes: "
            "commit or stash them, or pass --allow-dirty"
        )

    print("running tests/field/test_metal.py", flush=True)
    tests = [sys.executable, "-m", "pytest", "tests/field/test_metal.py", "-q"]
    if subprocess.run(tests, cwd=REPO_ROOT).returncode != 0:
        sys.exit("the Metal tests failed; fix the install before benchmarking")


def cpu_command(ranks: int, config_path: Path, out_path: Path) -> list[str]:
    """mpiexec command line of the CPU leg."""
    # -m mpi4py turns an uncaught exception on any rank into MPI_Abort instead
    # of leaving the other ranks deadlocked in their next collective
    return [
        "mpiexec",
        "-n",
        str(ranks),
        sys.executable,
        "-m",
        "mpi4py",
        "bench_cpu_scaling.py",
        str(config_path),
        str(out_path),
    ]


def run_leg(cmd: list[str]) -> dict[str, Any]:
    """Run one benchmark leg; return its run conditions and duration.

    Raises
    ------
    subprocess.CalledProcessError
        If the leg fails; the run folder is then left as <run-id>.incomplete.
    """
    before = mach.conditions()
    t0 = time.time()
    subprocess.run(cmd, cwd=BENCH_DIR, check=True)
    return {
        "conditions_before": before,
        "conditions_after": mach.conditions(),
        "duration_s": time.time() - t0,
    }


def fingerprints(cfg: dict[str, Any]) -> dict[str, str]:
    """What must match for two runs to be comparable, and what may differ.

    ``config`` and ``equilibrium`` define the workload: runs that differ in
    either are never plotted together. ``metal_sources`` is the GPU code;
    runs that differ in it are plotted together, with a warning.
    """
    canonical = json.dumps(cfg, sort_keys=True).encode()
    return {
        "config": hashlib.sha256(canonical).hexdigest(),
        "equilibrium": sha256_file(BENCH_DIR / cfg["boozmn_file"]),
        "metal_sources": hashlib.sha256(
            b"".join((REPO_ROOT / p).read_bytes() for p in METAL_SOURCES)
        ).hexdigest(),
    }


def software() -> dict[str, Any]:
    """Code version and toolchain of this run."""
    mpi = subprocess.run(["mpiexec", "--version"], capture_output=True, text=True)
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version.split()[0],
        "mpi": mpi.stdout.splitlines()[0] if mpi.stdout else "unknown",
    }


def sanity_check(cpu: dict[str, Any], metal: dict[str, Any]) -> dict[str, Any]:
    """Check CPU and GPU loss counts agree, workload by workload.

    Parameters
    ----------
    cpu, metal : dict[str, Any]
        The two legs' results, as written by the bench scripts.

    Returns
    -------
    dict[str, Any]
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


def report(
    record: dict[str, Any], cpu: dict[str, Any], metal: dict[str, Any], folder: Path
) -> None:
    """Print the run's headline numbers and what to do with the results.

    Raises
    ------
    SystemExit
        Non-zero, if the sanity check failed.
    """
    rows = [
        *zip(cpu["scaling"], metal["scaling"], strict=True),
        (cpu["validation"], metal["validation"]),
    ]
    print(f"\n{'workload':>24}  {'CPU [s]':>8}  {'GPU [s]':>8}  speedup  lost CPU/GPU")
    for c, g in rows:
        workload = f"{c['nparticles']} x {c['tmax_s']:g} s"
        print(
            f"{workload:>24}  {c['wall_s']:8.2f}  {g['wall_s']:8.2f}  "
            f"{c['wall_s'] / g['wall_s']:6.1f}x  {c['n_lost']:>5} / {g['n_lost']}"
        )
    print(f"\nresults: {folder.relative_to(BENCH_DIR)}")
    if not record["sanity"]["passed"]:
        sys.exit(
            "SANITY CHECK FAILED: CPU and GPU loss counts disagree beyond "
            "Poisson noise (see run.json). Do not submit this run; report it."
        )
    print(
        "Sanity check passed. To submit, see 'Submitting a run' in "
        "benchmarks/README.md."
    )


def git(*args: str) -> str:
    """Output of a git command in the repository."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
