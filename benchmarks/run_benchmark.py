"""Benchmark this Mac: CPU (MPI) vs Metal GPU tracing, filed for comparison.

The one command contributors run. It checks the machine is fit to benchmark,
traces the benchmark's workloads with the MPI CPU tracer (one rank per
physical core, capped by memory when the config says what a rank needs),
pauses so the machine cools, traces them again with the Metal GPU tracer, and
files everything in a new folder results/<benchmark>/<run-id>/:

- run.json: machine, run conditions, code and config fingerprints, and a
  CPU-vs-GPU sanity check
- cpu.json, metal.json: per-backend results, by kind of benchmark (kinds.py)
- cpu_*.npz, metal_*.npz: per-marker data

The benchmark is named by its config file: alpha-loss-wistell-a.yaml files
its runs under results/alpha-loss-wistell-a/. make_plots.py reads them all.
A run in progress (or one that crashed) is kept in <run-id>.incomplete/.

Usage
-----
cd benchmarks
../.venv/bin/python run_benchmark.py <benchmark>.yaml --tag <github-handle>
"""

import argparse
import hashlib
import json
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
import kinds
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
# Memory left to macOS and other apps when the CPU leg's ranks are capped by
# the config's per-rank need [GB]
MEMORY_RESERVE_GB = 4.0
# 1-minute load average per CPU core above which the preflight warns that
# other processes compete with the CPU leg. macOS counts load generously: a
# base M4 with a browser open idles near 4 on 10 cores, so lower bounds would
# warn on every run.
BUSY_LOAD_PER_CORE = 0.5


def main() -> None:
    """Check the machine, run both legs, sanity-check, and file the run."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = bc.load_config(config_path)
    kind = kinds.KINDS[kinds.kind_of(cfg)]
    at_launch = preflight(args.allow_dirty)
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

    ranks = cpu_ranks(hw, cfg)
    if ranks < hw["cpu_cores"]:
        print(f"CPU leg capped at {ranks} MPI ranks by memory", flush=True)
    legs = {"cpu": run_leg(cpu_command(kind, ranks, config_path, out / "cpu.json"))}
    print(f"cooling down for {COOLDOWN_S} s", flush=True)
    time.sleep(COOLDOWN_S)
    legs["metal"] = run_leg(
        [
            sys.executable,
            kind.metal_script,
            str(config_path),
            str(out / "metal.json"),
        ]
    )

    cpu = json.loads((out / "cpu.json").read_text())
    metal = json.loads((out / "metal.json").read_text())
    record = {
        "benchmark": benchmark,
        "kind": kinds.kind_of(cfg),
        "run_id": run_id,
        "created": created.isoformat(timespec="seconds"),
        "tag": args.tag,
        "machine": hw,
        "mpi_ranks": ranks,
        "conditions_at_launch": at_launch,
        "legs": legs,
        "fingerprints": prints,
        "software": sw,
        "config": cfg,
        "sanity": kind.sanity_check(cpu, metal),
    }
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    final = out.rename(out.with_name(run_id))
    report(record, kind.summary(cpu, metal), final)


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


def preflight(allow_dirty: bool) -> dict[str, Any]:
    """Refuse to benchmark when the results would be wrong or not comparable.

    Parameters
    ----------
    allow_dirty : bool
        Accept uncommitted changes to tracked files.

    Returns
    -------
    dict[str, Any]
        Run conditions at launch, before the test suite loads the machine.

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
    if now["load_average"] > BUSY_LOAD_PER_CORE * (os.cpu_count() or 1):
        print(
            f"warning: other processes are busy (load average "
            f"{now['load_average']:.1f}); quit them for clean CPU timings",
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
    return now


def cpu_ranks(hw: dict[str, Any], cfg: dict[str, Any]) -> int:
    """One MPI rank per physical core, capped by memory if the config says.

    A config's ``execution.cpu_rank_memory_gb`` is the peak memory one rank
    needs; the ranks then fit in the machine's memory less
    MEMORY_RESERVE_GB, so an 8 GB Mac runs fewer ranks rather than swapping.
    """
    per_rank = cfg.get("execution", {}).get("cpu_rank_memory_gb")
    if per_rank is None:
        return hw["cpu_cores"]
    fit = int((hw["memory_gb"] - MEMORY_RESERVE_GB) // float(per_rank))
    return max(1, min(hw["cpu_cores"], fit))


def cpu_command(
    kind: kinds.Kind, ranks: int, config_path: Path, out_path: Path
) -> list[str]:
    """mpiexec command line of the CPU leg."""
    env = [arg for var in kind.mpi_env for arg in ("-x", var)]
    # -m mpi4py turns an uncaught exception on any rank into MPI_Abort instead
    # of leaving the other ranks deadlocked in their next collective
    return [
        "mpiexec",
        "-n",
        str(ranks),
        *env,
        sys.executable,
        "-m",
        "mpi4py",
        kind.cpu_script,
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
    runs that differ in it are plotted together, with a warning. The
    config's ``execution`` section says how the legs run, not what they
    compute, so it is left out: tuning it keeps runs comparable.
    """
    workload = {k: v for k, v in cfg.items() if k != "execution"}
    canonical = json.dumps(workload, sort_keys=True).encode()
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


def report(record: dict[str, Any], summary: list[str], folder: Path) -> None:
    """Print the run's headline numbers and what to do with the results.

    Raises
    ------
    SystemExit
        Non-zero, if the sanity check failed.
    """
    print("\n" + "\n".join(summary))
    print(f"\nresults: {folder.relative_to(BENCH_DIR)}")
    if not record["sanity"]["passed"]:
        sys.exit(
            "SANITY CHECK FAILED: CPU and GPU disagree beyond what numerics "
            "explain (see run.json). Do not submit this run; report it."
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
